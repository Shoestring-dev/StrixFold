// umcbw: DRAM read/write bandwidth of the whole box from the data-fabric PMU (amd_df).
// The amd_umc PMUs report 0 counters on Strix Halo (dmesg: "amd_uncore: 0 amd_umc_N counters detected"), so this
// counts the DF's DRAM data beats per channel instead, Zen 4 encoding: event 0x1f + 0x40*channel, umask 0x7fe (read
// beats) and 0x7ff (write beats), one beat = 64 bytes, channels 0-15 (all 16 count on node0/node1). 32 events share the
// DF's 8 counters, so the kernel multiplexes them and each count is scaled by time_enabled / time_running.
// Prints one line per interval and a summary on exit (SIGINT/SIGTERM or after -d seconds). Needs CAP_PERFMON or
// perf_event_paranoid <= 0 (run with sudo); only reads counters.
// usage: umcbw [-i interval_ms] [-d duration_s] [-t label] [-c channels]
#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAXCH 16
static volatile sig_atomic_t stop;
static void on_sig(int s) { (void)s; stop = 1; }

static uint64_t df_config(unsigned ev, unsigned um) {
    return (uint64_t)(ev & 0xff) | ((uint64_t)((ev >> 8) & 0x3f) << 32) | ((uint64_t)(um & 0xff) << 8) |
           ((uint64_t)((um >> 8) & 0xf) << 24);
}

static int open_ev(int type, uint64_t config) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = type;
    a.config = config;
    a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int)syscall(SYS_perf_event_open, &a, -1, 0, -1, 0);
}

/* scaled count of one multiplexed event since it was opened */
static double scaled(int fd) {
    uint64_t v[3] = {0};
    if (read(fd, v, sizeof v) != sizeof v) { perror("read"); exit(1); }
    return v[2] ? (double)v[0] * (double)v[1] / (double)v[2] : 0.0;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int interval_ms = 1000, duration_s = 0, nch = MAXCH, opt;
    const char *label = "umcbw";
    while ((opt = getopt(argc, argv, "i:d:t:c:")) != -1) {
        if (opt == 'i') interval_ms = atoi(optarg);
        else if (opt == 'd') duration_s = atoi(optarg);
        else if (opt == 't') label = optarg;
        else if (opt == 'c') nch = atoi(optarg);
        else { fprintf(stderr, "usage: umcbw [-i ms] [-d s] [-t label] [-c channels]\n"); return 2; }
    }
    if (nch < 1 || nch > MAXCH) nch = MAXCH;
    FILE *f = fopen("/sys/bus/event_source/devices/amd_df/type", "r");
    int type = -1;
    if (!f || fscanf(f, "%d", &type) != 1) { fprintf(stderr, "no amd_df PMU\n"); return 1; }
    fclose(f);
    int fd_rd[MAXCH], fd_wr[MAXCH];
    for (int ch = 0; ch < nch; ch++) {
        fd_rd[ch] = open_ev(type, df_config(0x1f + 0x40 * ch, 0x7fe));
        fd_wr[ch] = open_ev(type, df_config(0x1f + 0x40 * ch, 0x7ff));
        if (fd_rd[ch] < 0 || fd_wr[ch] < 0) { perror("perf_event_open (amd_df)"); return 1; }
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    double base_rd = 0, base_wr = 0, prev_rd = 0, prev_wr = 0, t0 = 0, tp = 0, peak = 0;
    int samples = 0;
    for (int k = 0;; k++) {
        double srd = 0, swr = 0;
        for (int ch = 0; ch < nch; ch++) {
            srd += scaled(fd_rd[ch]);
            swr += scaled(fd_wr[ch]);
        }
        double t = now();
        if (k == 0) {
            base_rd = srd;
            base_wr = swr;
            t0 = t;
        } else {
            double dt = t - tp, rd = (srd - prev_rd) * 64.0 / dt / 1e9, wr = (swr - prev_wr) * 64.0 / dt / 1e9;
            printf("%s t=%.1f rd=%.2f wr=%.2f total=%.2f GB/s\n", label, t - t0, rd, wr, rd + wr);
            fflush(stdout);
            if (rd + wr > peak) peak = rd + wr;
            samples++;
        }
        prev_rd = srd;
        prev_wr = swr;
        tp = t;
        if (stop || (duration_s > 0 && t - t0 >= duration_s)) break;
        struct timespec ts = { interval_ms / 1000, (interval_ms % 1000) * 1000000L };
        while (nanosleep(&ts, &ts) && !stop) {}  // a signal ends the sleep; the partial interval is still read
    }
    double span = tp - t0;
    if (samples > 0 && span > 0)
        printf("%s summary: %d channels, %.1f s, mean rd=%.2f wr=%.2f total=%.2f GB/s, peak %.2f GB/s (interval %d ms)\n",
               label, nch, span, (prev_rd - base_rd) * 64.0 / span / 1e9, (prev_wr - base_wr) * 64.0 / span / 1e9,
               (prev_rd - base_rd + prev_wr - base_wr) * 64.0 / span / 1e9, peak, interval_ms);
    return 0;
}
