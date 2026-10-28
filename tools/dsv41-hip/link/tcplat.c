/* tcplat: TCP request/response and symmetric-exchange latency for the TP2 link gate (G-b).
 *
 * One binary, two roles. The server (-l) listens on one or more local addresses and forks a
 * child per session. The client (-c) sends the whole test configuration, so every knob lives
 * on the client command line. A session can stripe each message across k connections (one per
 * address) to test two cables as one rail.
 *
 * Modes
 *   pp  ping-pong: client sends N, server receives all N then sends N back; client times
 *       send-start .. last byte received (one full request/response).
 *   ex  symmetric exchange: both sides send N and receive N at once (what a 2-rank all-gather
 *       does); client times send-start .. last byte received.
 * A test is a pattern of message sizes repeated `reps` times (after `warm` unrecorded reps).
 * A one-entry pattern is a size sweep point; a 94-entry pattern is a simulated decode round.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif
#ifndef TCP_RTO_MIN_US
#define TCP_RTO_MIN_US 45
#endif
#ifndef SO_INCOMING_NAPI_ID
#define SO_INCOMING_NAPI_ID 56
#endif

#define MAGIC 0x544c4154u
#define VERSION 4u
#define MAXK 4
#define MAXPAT 512
#define MAXLISTEN 4

struct cfg {
    uint32_t magic, version;
    uint64_t group;
    uint32_t k, idx;
    uint32_t mode;          /* 0 ping-pong, 1 symmetric exchange */
    uint32_t spin;          /* 1 nonblocking spin, 0 nonblocking + poll() when idle */
    int32_t busy_poll;      /* SO_BUSY_POLL us; -1 leaves the socket default (sysctl busy_read) */
    int32_t srv_cpu;        /* server thread CPU, -1 unpinned */
    uint32_t quickack;
    int32_t sockbuf;        /* SO_SNDBUF/SO_RCVBUF bytes, 0 autotune */
    uint32_t gap_us, gap_sleep;
    uint32_t verify;        /* 1 full payload check every step, 0 head/tail stamps */
    uint32_t npat;
    uint64_t reps, warm;
    int32_t rto_min_us;     /* TCP_RTO_MIN_US per socket, 0 = kernel default */
    uint32_t pat[MAXPAT];
};

struct result {
    uint32_t magic, errors;
    uint64_t utime_ns, stime_ns, wall_ns;
    uint64_t sys_busy, sys_total;
    uint32_t napi_id[MAXK];
    int32_t busy_poll_eff;
    uint32_t retrans[MAXK], rto_us[MAXK];
};

static void tcpinfo(int fd, uint32_t *retrans, uint32_t *rto) {
    struct tcp_info ti;
    socklen_t l = sizeof ti;
    memset(&ti, 0, sizeof ti);
    getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &l);
    *retrans = ti.tcpi_total_retrans;
    *rto = ti.tcpi_rto;
}

static int g_quickack;

static inline uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void die(const char *m) { perror(m); exit(2); }

static void sys_ticks(uint64_t *busy, uint64_t *total) {
    unsigned long long v[10] = {0};
    FILE *f = fopen("/proc/stat", "r");
    *busy = *total = 0;
    if (!f) return;
    if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
               &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]) >= 8) {
        *busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
        *total = *busy + v[3] + v[4];
    }
    fclose(f);
}

static uint64_t tv_ns(struct timeval t) { return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_usec * 1000ull; }

static void pin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof s, &s)) die("sched_setaffinity");
}

static int sockopts(int fd, const struct cfg *c) {
    int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one)) die("TCP_NODELAY");
    if (c->busy_poll >= 0) {
        int v = c->busy_poll;
        if (setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &v, sizeof v))
            fprintf(stderr, "warn: SO_BUSY_POLL=%d: %s\n", v, strerror(errno));
    }
    if (c->rto_min_us > 0) {
        int v = c->rto_min_us;
        if (setsockopt(fd, IPPROTO_TCP, TCP_RTO_MIN_US, &v, sizeof v))
            fprintf(stderr, "warn: TCP_RTO_MIN_US=%d: %s\n", v, strerror(errno));
    }
    if (c->sockbuf > 0) {
        int v = c->sockbuf;
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &v, sizeof v);
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &v, sizeof v);
    }
    int eff = -1;
    socklen_t l = sizeof eff;
    getsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &eff, &l);
    return eff;
}

static void split(size_t n, int k, size_t *off, size_t *len) {
    size_t ch = (n + (size_t)k - 1) / (size_t)k;
    for (int i = 0; i < k; i++) {
        size_t o = (size_t)i * ch;
        off[i] = o < n ? o : n;
        len[i] = n - off[i] < ch ? n - off[i] : ch;
    }
}

/* Progress k sends and k receives together until all bytes moved. Returns 0 or -1. */
static int xfer(const int *fd, int k, const char *sb, size_t ns, char *rb, size_t nr, int spin) {
    size_t soff[MAXK], slen[MAXK], sdone[MAXK] = {0}, roff[MAXK], rlen[MAXK], rdone[MAXK] = {0};
    split(ns, k, soff, slen);
    split(nr, k, roff, rlen);
    size_t left = ns + nr;
    uint64_t idle = 0, idle_since = 0;
    while (left) {
        int progress = 0;
        for (int i = 0; i < k; i++) {
            if (sdone[i] < slen[i]) {
                ssize_t r = send(fd[i], sb + soff[i] + sdone[i], slen[i] - sdone[i], MSG_DONTWAIT | MSG_NOSIGNAL);
                if (r > 0) { sdone[i] += (size_t)r; left -= (size_t)r; progress = 1; }
                else if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return -1;
            }
            if (rdone[i] < rlen[i]) {
                ssize_t r = recv(fd[i], rb + roff[i] + rdone[i], rlen[i] - rdone[i], MSG_DONTWAIT);
                if (r > 0) {
                    rdone[i] += (size_t)r; left -= (size_t)r; progress = 1;
                    if (g_quickack) { int one = 1; setsockopt(fd[i], IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one); }
                } else if (r == 0) { errno = ECONNRESET; return -1; }
                else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return -1;
            }
        }
        if (progress) { idle = 0; continue; }
        if (++idle == 1) idle_since = now_ns();
        if (spin) {
            if ((idle & 0xffff) == 0 && now_ns() - idle_since > 10000000000ull) { errno = ETIMEDOUT; return -1; }
            continue;
        }
        struct pollfd p[MAXK];
        int np = 0;
        for (int i = 0; i < k; i++) {
            short ev = (short)((sdone[i] < slen[i] ? POLLOUT : 0) | (rdone[i] < rlen[i] ? POLLIN : 0));
            if (ev) { p[np].fd = fd[i]; p[np].events = ev; p[np].revents = 0; np++; }
        }
        int rc = poll(p, (nfds_t)np, 1000);
        if (rc < 0 && errno != EINTR) return -1;
        if (now_ns() - idle_since > 10000000000ull) { errno = ETIMEDOUT; return -1; }
    }
    return 0;
}

/* Payload: 8-byte words seed*phi + j. Stamp mode writes only the first and last word. */
static inline uint64_t word(uint64_t seed, uint64_t j) { return seed * 0x9E3779B97F4A7C15ull + j; }

static void fill(char *b, size_t n, uint64_t seed, int full) {
    if (n < 8) { memset(b, (int)(seed & 0xff), n); return; }
    if (full) {
        for (size_t j = 0; j + 8 <= n; j += 8) { uint64_t w = word(seed, j / 8); memcpy(b + j, &w, 8); }
    } else {
        uint64_t w0 = word(seed, 0), w1 = word(seed, (n - 8) / 8);
        memcpy(b, &w0, 8);
        memcpy(b + n - 8, &w1, 8);
    }
}

static int check(const char *b, size_t n, uint64_t seed, int full) {
    if (n < 8) { for (size_t j = 0; j < n; j++) if ((unsigned char)b[j] != (seed & 0xff)) return 1; return 0; }
    uint64_t w;
    if (full) {
        for (size_t j = 0; j + 8 <= n; j += 8) { memcpy(&w, b + j, 8); if (w != word(seed, j / 8)) return 1; }
        return 0;
    }
    memcpy(&w, b, 8);
    if (w != word(seed, 0)) return 1;
    memcpy(&w, b + n - 8, 8);
    return w != word(seed, (n - 8) / 8);
}

static void gap(const struct cfg *c) {
    if (!c->gap_us) return;
    if (c->gap_sleep) {
        struct timespec d = {0, (long)c->gap_us * 1000};
        nanosleep(&d, NULL);
    } else {
        uint64_t end = now_ns() + (uint64_t)c->gap_us * 1000ull;
        while (now_ns() < end) __builtin_ia32_pause();
    }
}

static size_t maxsize(const struct cfg *c) {
    size_t m = 8;
    for (uint32_t i = 0; i < c->npat; i++) if (c->pat[i] > m) m = c->pat[i];
    return m;
}

static int read_full(int fd, void *b, size_t n) {
    char *p = b;
    while (n) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int write_full(int fd, const void *b, size_t n) {
    const char *p = b;
    while (n) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        p += r; n -= (size_t)r;
    }
    return 0;
}

/* ---------------- server ---------------- */

static void serve_session(int *fd, struct cfg *c) {
    pin(c->srv_cpu);
    g_quickack = (int)c->quickack;
    int k = (int)c->k, spin = (int)c->spin, full = (int)c->verify;
    struct result res;
    memset(&res, 0, sizeof res);
    res.magic = MAGIC;
    for (int i = 0; i < k; i++) res.busy_poll_eff = sockopts(fd[i], c);
    size_t mx = maxsize(c);
    char *sb = aligned_alloc(4096, (mx + 4095) & ~(size_t)4095);
    char *rb = aligned_alloc(4096, (mx + 4095) & ~(size_t)4095);
    if (!sb || !rb) die("alloc");
    memset(sb, 0, mx);
    memset(rb, 0, mx);
    char ack = 1;
    if (write_full(fd[0], &ack, 1)) die("ack");
    struct rusage r0, r1;
    uint64_t w0 = 0, b0 = 0, t0 = 0;
    uint64_t total = (c->warm + c->reps) * c->npat;
    for (uint64_t s = 0; s < total; s++) {
        if (s == c->warm * c->npat) { getrusage(RUSAGE_SELF, &r0); w0 = now_ns(); sys_ticks(&b0, &t0); }
        size_t n = c->pat[s % c->npat];
        gap(c);
        fill(sb, n, (s << 1) | 1u, full);
        int rc;
        if (c->mode == 0) {
            rc = xfer(fd, k, NULL, 0, rb, n, spin);
            if (!rc) rc = xfer(fd, k, sb, n, NULL, 0, spin);
        } else {
            rc = xfer(fd, k, sb, n, rb, n, spin);
        }
        if (rc) { perror("server xfer"); exit(3); }
        if (check(rb, n, s << 1, full)) res.errors++;
    }
    getrusage(RUSAGE_SELF, &r1);
    res.wall_ns = now_ns() - w0;
    res.utime_ns = tv_ns(r1.ru_utime) - tv_ns(r0.ru_utime);
    res.stime_ns = tv_ns(r1.ru_stime) - tv_ns(r0.ru_stime);
    uint64_t b1, t1;
    sys_ticks(&b1, &t1);
    res.sys_busy = b1 - b0;
    res.sys_total = t1 - t0;
    for (int i = 0; i < k; i++) {
        uint32_t id = 0;
        socklen_t l = sizeof id;
        getsockopt(fd[i], SOL_SOCKET, SO_INCOMING_NAPI_ID, &id, &l);
        res.napi_id[i] = id;
        tcpinfo(fd[i], &res.retrans[i], &res.rto_us[i]);
    }
    if (write_full(fd[0], &res, sizeof res)) die("result");
    char z;
    (void)recv(fd[0], &z, 1, 0); /* wait for client close */
    for (int i = 0; i < k; i++) close(fd[i]);
}

struct pending { uint64_t group; uint32_t k, have; int fd[MAXK]; struct cfg c; };

static int listen_on(const char *ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) die("socket");
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) { fprintf(stderr, "bad ip %s\n", ip); exit(2); }
    if (bind(s, (struct sockaddr *)&a, sizeof a)) die("bind");
    if (listen(s, 16)) die("listen");
    return s;
}

static int run_server(char **ips, int nip, int port) {
    signal(SIGCHLD, SIG_IGN);
    struct pollfd lp[MAXLISTEN];
    for (int i = 0; i < nip; i++) { lp[i].fd = listen_on(ips[i], port); lp[i].events = POLLIN; }
    struct pending pend[8];
    memset(pend, 0, sizeof pend);
    fprintf(stderr, "tcplat server listening on %d address(es) port %d\n", nip, port);
    for (;;) {
        if (poll(lp, (nfds_t)nip, -1) < 0) { if (errno == EINTR) continue; die("poll"); }
        for (int i = 0; i < nip; i++) {
            if (!(lp[i].revents & POLLIN)) continue;
            int fd = accept(lp[i].fd, NULL, NULL);
            if (fd < 0) continue;
            struct timeval tv = {5, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
            struct cfg c;
            if (read_full(fd, &c, sizeof c) || c.magic != MAGIC || c.version != VERSION ||
                c.k < 1 || c.k > MAXK || c.idx >= c.k || c.npat < 1 || c.npat > MAXPAT) {
                fprintf(stderr, "bad session header\n");
                close(fd);
                continue;
            }
            struct timeval tz = {0, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tz, sizeof tz);
            int slot = -1;
            for (int j = 0; j < 8; j++) if (pend[j].k && pend[j].group == c.group) slot = j;
            if (slot < 0) for (int j = 0; j < 8; j++) if (!pend[j].k) { slot = j; break; }
            if (slot < 0) { close(fd); continue; }
            struct pending *p = &pend[slot];
            if (!p->k) { p->group = c.group; p->k = c.k; p->have = 0; p->c = c; }
            p->fd[c.idx] = fd;
            p->have++;
            if (p->have < p->k) continue;
            pid_t pid = fork();
            if (pid == 0) {
                for (int j = 0; j < nip; j++) close(lp[j].fd);
                serve_session(p->fd, &p->c);
                _exit(0);
            }
            for (uint32_t j = 0; j < p->k; j++) close(p->fd[j]);
            memset(p, 0, sizeof *p);
        }
    }
}

/* ---------------- client ---------------- */

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static double pct32(const uint32_t *s, size_t n, double p) { size_t i = (size_t)(p * (double)(n - 1) + 0.5); return s[i]; }
static double pct64(const uint64_t *s, size_t n, double p) { size_t i = (size_t)(p * (double)(n - 1) + 0.5); return (double)s[i]; }

static int parse_pattern(const char *spec, struct cfg *c) {
    /* "80x122880,2x307200,1x1548288" or "122880" */
    char *dup = strdup(spec), *save = NULL;
    c->npat = 0;
    for (char *t = strtok_r(dup, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
        unsigned long cnt = 1, sz;
        char *x = strchr(t, 'x');
        if (x) { cnt = strtoul(t, NULL, 10); sz = strtoul(x + 1, NULL, 10); }
        else sz = strtoul(t, NULL, 10);
        for (unsigned long i = 0; i < cnt; i++) {
            if (c->npat >= MAXPAT) { free(dup); return -1; }
            c->pat[c->npat++] = (uint32_t)sz;
        }
    }
    free(dup);
    return c->npat ? 0 : -1;
}

static int connect_to(const char *ip, int port) {
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) { fprintf(stderr, "bad ip %s\n", ip); exit(2); }
    for (int tries = 0; tries < 50; tries++) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) die("socket");
        if (!connect(s, (struct sockaddr *)&a, sizeof a)) return s;
        close(s);
        usleep(100000);
    }
    fprintf(stderr, "connect %s:%d failed\n", ip, port);
    exit(2);
}

static void usage(void) {
    fprintf(stderr,
        "server: tcplat -l -B ip[,ip2] [-p port]\n"
        "client: tcplat -c -B ip[,ip2] [-p port] [-m pp|ex] [-P pattern] [-n reps] [-w warm]\n"
        "        [-C cpu] [-S srvcpu] [-b busy_poll_us|-1] [-y spin|block] [-g gap_us] [-G]\n"
        "        [-q] [-k sockbuf] [-V] [-t label] [-o rawdir]\n");
    exit(2);
}

int main(int argc, char **argv) {
    int server = 0, client = 0, port = 47110, cpu = -1, opt;
    char *ips[MAXK];
    int nip = 0;
    const char *pattern = "122880", *label = "run", *rawdir = NULL;
    struct cfg c;
    memset(&c, 0, sizeof c);
    c.magic = MAGIC; c.version = VERSION; c.mode = 0; c.spin = 1; c.busy_poll = -1; c.srv_cpu = -1;
    c.reps = 10000; c.warm = 1000;
    while ((opt = getopt(argc, argv, "lcB:p:m:P:n:w:C:S:b:y:g:Gqk:Vt:o:R:")) != -1) {
        switch (opt) {
        case 'l': server = 1; break;
        case 'c': client = 1; break;
        case 'B': {
            char *dup = strdup(optarg), *save = NULL;
            for (char *t = strtok_r(dup, ",", &save); t && nip < MAXK; t = strtok_r(NULL, ",", &save)) ips[nip++] = t;
            break;
        }
        case 'p': port = atoi(optarg); break;
        case 'm': c.mode = !strcmp(optarg, "ex"); break;
        case 'P': pattern = optarg; break;
        case 'n': c.reps = strtoull(optarg, NULL, 10); break;
        case 'w': c.warm = strtoull(optarg, NULL, 10); break;
        case 'C': cpu = atoi(optarg); break;
        case 'S': c.srv_cpu = atoi(optarg); break;
        case 'b': c.busy_poll = atoi(optarg); break;
        case 'y': c.spin = !strcmp(optarg, "spin"); break;
        case 'g': c.gap_us = (uint32_t)atoi(optarg); break;
        case 'G': c.gap_sleep = 1; break;
        case 'q': c.quickack = 1; break;
        case 'k': c.sockbuf = atoi(optarg); break;
        case 'V': c.verify = 1; break;
        case 't': label = optarg; break;
        case 'o': rawdir = optarg; break;
        case 'R': c.rto_min_us = atoi(optarg); break;
        default: usage();
        }
    }
    if (server == client || nip < 1) usage();
    if (server) return run_server(ips, nip, port);

    if (parse_pattern(pattern, &c)) { fprintf(stderr, "bad pattern\n"); return 2; }
    c.k = (uint32_t)nip;
    c.group = now_ns() ^ ((uint64_t)getpid() << 32);
    g_quickack = (int)c.quickack;
    int fd[MAXK];
    int bp_eff = -1;
    for (int i = 0; i < nip; i++) {
        fd[i] = connect_to(ips[i], port);
        c.idx = (uint32_t)i;
        if (write_full(fd[i], &c, sizeof c)) die("send cfg");
    }
    for (int i = 0; i < nip; i++) bp_eff = sockopts(fd[i], &c);
    pin(cpu);
    char ack;
    if (read_full(fd[0], &ack, 1)) die("ack");

    size_t mx = maxsize(&c);
    char *sb = aligned_alloc(4096, (mx + 4095) & ~(size_t)4095);
    char *rb = aligned_alloc(4096, (mx + 4095) & ~(size_t)4095);
    if (!sb || !rb) die("alloc");
    memset(sb, 0, mx);
    memset(rb, 0, mx);
    uint64_t nsteps = c.reps * c.npat;
    uint32_t *lat = calloc(nsteps, sizeof *lat);
    uint64_t *round = calloc(c.reps, sizeof *round);
    uint64_t *rsum = calloc(c.reps, sizeof *rsum);
    if (!lat || !round || !rsum) die("alloc");
    uint32_t errors = 0;
    struct rusage r0, r1;
    uint64_t w0 = 0, b0 = 0, t0s = 0, rstart = 0;
    uint64_t total = (c.warm + c.reps) * c.npat;
    int k = nip, spin = (int)c.spin, full = (int)c.verify;
    for (uint64_t s = 0; s < total; s++) {
        uint64_t j = s % c.npat;
        if (s == c.warm * c.npat) { getrusage(RUSAGE_SELF, &r0); w0 = now_ns(); sys_ticks(&b0, &t0s); }
        size_t n = c.pat[j];
        gap(&c);
        fill(sb, n, s << 1, full);
        uint64_t t0 = now_ns();
        if (j == 0) rstart = t0;
        if (xfer(fd, k, sb, n, rb, n, spin)) { perror("client xfer"); return 3; }
        uint64_t t1 = now_ns();
        if (s >= c.warm * c.npat) {
            uint64_t m = s - c.warm * c.npat;
            lat[m] = (uint32_t)(t1 - t0);
            rsum[m / c.npat] += t1 - t0;
            if (j == c.npat - 1) round[m / c.npat] = t1 - rstart;
        }
        if (check(rb, n, (s << 1) | 1u, full)) errors++;
    }
    getrusage(RUSAGE_SELF, &r1);
    uint64_t wall = now_ns() - w0;
    uint64_t b1, t1s;
    sys_ticks(&b1, &t1s);
    struct result res;
    if (read_full(fd[0], &res, sizeof res) || res.magic != MAGIC) die("result");
    uint32_t napi[MAXK] = {0}, retrans[MAXK] = {0}, rto[MAXK] = {0};
    for (int i = 0; i < k; i++) {
        socklen_t l = sizeof napi[i];
        getsockopt(fd[i], SOL_SOCKET, SO_INCOMING_NAPI_ID, &napi[i], &l);
        tcpinfo(fd[i], &retrans[i], &rto[i]);
    }
    for (int i = 0; i < k; i++) close(fd[i]);

    /* Per-size statistics. */
    printf("# label=%s mode=%s k=%d ips=", label, c.mode ? "ex" : "pp", k);
    for (int i = 0; i < k; i++) printf("%s%s", ips[i], i + 1 < k ? "," : "");
    printf(" spin=%u busy_poll=%d(eff %d/srv %d) cpu=%d srvcpu=%d gap=%uus%s quickack=%u sockbuf=%d verify=%u reps=%" PRIu64 " warm=%" PRIu64 " npat=%u\n",
           c.spin, c.busy_poll, bp_eff, res.busy_poll_eff, cpu, c.srv_cpu, c.gap_us, c.gap_sleep ? "(sleep)" : "",
           c.quickack, c.sockbuf, c.verify, c.reps, c.warm, c.npat);
    uint32_t sizes[MAXPAT];
    int nsz = 0;
    for (uint32_t j = 0; j < c.npat; j++) {
        int seen = 0;
        for (int q = 0; q < nsz; q++) if (sizes[q] == c.pat[j]) seen = 1;
        if (!seen) sizes[nsz++] = c.pat[j];
    }
    uint32_t *tmp = malloc(nsteps * sizeof *tmp);
    for (int q = 0; q < nsz; q++) {
        size_t m = 0;
        double sum = 0;
        for (uint64_t s = 0; s < nsteps; s++)
            if (c.pat[s % c.npat] == sizes[q]) { tmp[m++] = lat[s]; sum += lat[s]; }
        qsort(tmp, m, sizeof *tmp, cmp_u32);
        printf("size=%u n=%zu min=%.1f p50=%.1f p90=%.1f p95=%.1f p99=%.1f p99.9=%.1f max=%.1f mean=%.1f us\n",
               sizes[q], m, tmp[0] / 1e3, pct32(tmp, m, .5) / 1e3, pct32(tmp, m, .9) / 1e3, pct32(tmp, m, .95) / 1e3,
               pct32(tmp, m, .99) / 1e3, pct32(tmp, m, .999) / 1e3, tmp[m - 1] / 1e3, sum / (double)m / 1e3);
    }
    if (c.npat > 1) {
        uint64_t *rs = malloc(c.reps * sizeof *rs);
        double sum = 0;
        memcpy(rs, round, c.reps * sizeof *rs);
        for (uint64_t i = 0; i < c.reps; i++) sum += (double)rs[i];
        qsort(rs, c.reps, sizeof *rs, cmp_u64);
        uint64_t bytes = 0;
        for (uint32_t j = 0; j < c.npat; j++) bytes += c.pat[j];
        printf("round steps=%u bytes_each_way=%" PRIu64 " n=%" PRIu64 " min=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f mean=%.3f ms\n",
               c.npat, bytes, c.reps, rs[0] / 1e6, pct64(rs, c.reps, .5) / 1e6, pct64(rs, c.reps, .95) / 1e6,
               pct64(rs, c.reps, .99) / 1e6, rs[c.reps - 1] / 1e6, sum / (double)c.reps / 1e6);
        memcpy(rs, rsum, c.reps * sizeof *rs);
        sum = 0;
        for (uint64_t i = 0; i < c.reps; i++) sum += (double)rs[i];
        qsort(rs, c.reps, sizeof *rs, cmp_u64);
        printf("roundsum (sum of step latencies, excludes gaps) min=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f mean=%.3f ms\n",
               rs[0] / 1e6, pct64(rs, c.reps, .5) / 1e6, pct64(rs, c.reps, .95) / 1e6,
               pct64(rs, c.reps, .99) / 1e6, rs[c.reps - 1] / 1e6, sum / (double)c.reps / 1e6);
        free(rs);
    }
    double hz = (double)sysconf(_SC_CLK_TCK);
    double cw = (double)(tv_ns(r1.ru_utime) - tv_ns(r0.ru_utime)) / (double)wall;
    double cs = (double)(tv_ns(r1.ru_stime) - tv_ns(r0.ru_stime)) / (double)wall;
    printf("cpu client_proc=%.2f(usr %.2f sys %.2f) server_proc=%.2f(usr %.2f sys %.2f) client_box=%.2f server_box=%.2f cores; wall=%.2fs\n",
           cw + cs, cw, cs, (double)(res.utime_ns + res.stime_ns) / (double)res.wall_ns,
           (double)res.utime_ns / (double)res.wall_ns, (double)res.stime_ns / (double)res.wall_ns,
           (double)(b1 - b0) / hz / ((double)wall / 1e9), (double)res.sys_busy / hz / ((double)res.wall_ns / 1e9), wall / 1e9);
    printf("napi client=");
    for (int i = 0; i < k; i++) printf("%u%s", napi[i], i + 1 < k ? "," : "");
    printf(" server=");
    for (int i = 0; i < k; i++) printf("%u%s", res.napi_id[i], i + 1 < k ? "," : "");
    printf(" retrans client=");
    for (int i = 0; i < k; i++) printf("%u%s", retrans[i], i + 1 < k ? "," : "");
    printf(" server=");
    for (int i = 0; i < k; i++) printf("%u%s", res.retrans[i], i + 1 < k ? "," : "");
    printf(" rto_us=%u rto_min_us=%d errors client=%u server=%u\n", rto[0], c.rto_min_us, errors, res.errors);
    if (rawdir) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s.lat", rawdir, label);
        FILE *f = fopen(path, "w");
        if (f) {
            for (uint64_t s = 0; s < nsteps; s++) fprintf(f, "%u %u\n", c.pat[s % c.npat], lat[s]);
            fclose(f);
        }
        if (c.npat > 1) {
            snprintf(path, sizeof path, "%s/%s.rounds", rawdir, label);
            f = fopen(path, "w");
            if (f) { for (uint64_t i = 0; i < c.reps; i++) fprintf(f, "%" PRIu64 "\n", round[i]); fclose(f); }
        }
    }
    fflush(stdout);
    return errors || res.errors ? 4 : 0;
}
