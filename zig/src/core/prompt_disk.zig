//! sf-disk: kept prompt states on disk as chain-keyed blocks and state files, written by a background thread.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const c = std.c;
const Sha256 = std.crypto.hash.sha2.Sha256;
const format = @import("prompt_disk_format.zig");
const fs = @import("prompt_disk_fs.zig");
const index = @import("prompt_disk_index.zig");
const namespace = @import("prompt_disk_namespace.zig");
const writer = @import("prompt_disk_writer.zig");
const now = fs.now;
const sleepNs = fs.sleepNs;

pub const Key = [32]u8;
pub const version = format.version;
pub const head_bytes = format.head_bytes;
pub const Kind = format.Kind;
pub const Head = format.Head;
pub const Fifo = format.Fifo;
pub const bitmap = format.bitmap;
pub const bit = format.bit;
pub const identityDiff = namespace.identityDiff;
pub const fsFree = fs.fsFree;
/// The most blocks a state lists (2^22 tokens at 2048 a block: past any context this engine serves).
pub const max_blocks = 4096;

// -- keys ---------------------------------------------------------------------------------------------------------

/// H_0 = SHA-256("sf-disk/1" || identity): the chain's seed for a namespace's identity text.
pub fn seed(identity: []const u8) Key {
    var h = Sha256.init(.{});
    h.update("sf-disk/1");
    h.update(identity);
    return h.finalResult();
}

/// The namespace (directory name) of an identity: the first 16 bytes of its SHA-256.
pub fn namespaceOf(identity: []const u8) [16]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(identity, &d, .{});
    return d[0..16].*;
}

/// out[j-1] = H_j = SHA-256(H_{j-1} || block j's bytes), for the whole blocks of `block` bytes `tokens` holds.
pub fn chain(h0: Key, tokens: []const u8, block: usize, out: []Key) void {
    std.debug.assert(out.len * block <= tokens.len);
    var prev = h0;
    for (out, 0..) |*o, i| {
        var h = Sha256.init(.{});
        h.update(&prev);
        h.update(tokens[i * block ..][0..block]);
        o.* = h.finalResult();
        prev = o.*;
    }
}

/// A key's first 16 hex digits (log lines).
pub fn short(k: Key) [16]u8 {
    return std.fmt.bytesToHex(k[0..8].*, .lower);
}

// -- layout -------------------------------------------------------------------------------------------------------

/// Plane p of a run of n blocks starts here; block i's part is i * planes[p] further (as in the device plane).
pub fn rangeOffset(planes: []const u64, n: usize, p: usize) u64 {
    var off: u64 = 0;
    for (planes[0..p]) |x| off += x * n;
    return off;
}

pub fn sum(planes: []const u64) u64 {
    var n: u64 = 0;
    for (planes) |x| n += x;
    return n;
}

// -- the index ----------------------------------------------------------------------------------------------------

pub const Status = enum(u8) { pending, complete };

pub const StateRec = struct { k: u32, bytes: u64, used: i128, status: Status, keys: []Key };
pub const BlockRec = struct { bytes: u64, refs: u32 = 0, status: Status, pinned: bool = false, guard: bool = false };

/// A resume's answer on one rank, and the pair's outcome from both.
pub const Load = enum(u8) {
    ok = 0,
    bad = 1, // missing or corrupt: both ranks forget the state
    busy = 2, // still being written here: a miss, kept

    pub fn name(l: Load) []const u8 {
        return @tagName(l);
    }
};
pub const Outcome = enum(u8) {
    hit = 0,
    miss = 1,
    forget = 2,

    pub fn of(a: Load, b: Load) Outcome {
        if (a == .ok and b == .ok) return .hit;
        if (a == .bad or b == .bad) return .forget;
        return .miss;
    }
};

pub const Reason = enum { exists, queue, floor, budget, io, hash, off };

pub const Config = struct {
    root: []const u8,
    budget: u64,
    floor: u64 = 16 << 30,
    queue: u64 = 1 << 30,
    rate: u64 = 1 << 30, // the writer's bytes a second at most (0: no cap)
    follower: bool = false, // rank 1: refuses what passes its budget instead of evicting
    free_override: ?u64 = null, // tests: the free bytes the filesystem reports
    drain_ns: u64 = 20 * std.time.ns_per_s, // close: how long the queued writes may still run
    block_tokens: u64 = 2048, // a block's positions (log lines' tokens=)
};

/// A run of blocks a save or a stream writes: indices into the keys it was planned for.
pub const Run = struct { first: u32, n: u32 };

/// What `plan` decided: the evicted states (for the follower), the runs absent here, and the queue bytes reserved.
pub const Plan = struct {
    result: union(enum) { ok, skip: Reason },
    victims: []Key = &.{},
    runs: []Run = &.{},
    charge: u64 = 0,

    pub fn deinit(p: *Plan, gpa: Allocator) void {
        gpa.free(p.victims);
        gpa.free(p.runs);
        p.* = undefined;
    }
};

pub const Job = union(enum) {
    run: struct { keys: []Key, first_j: u32, data: []u8, charge: u64, t0: i128 },
    state: struct { key: Key, k: u32, keys: []Key, slot: []u8, charge: u64, t0: i128, new_blocks: u32, new_bytes: u64 },
    delete: struct { states: []Key, blocks: []Key },
    sweep: struct { name: [32]u8 },
};

pub const Done = struct { job: Job, ok: bool, reason: Reason = .io };

pub const Other = struct { name: [32]u8, bytes: u64, used: i128 };

pub const Counts = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    forgets: u64 = 0,
    busy: u64 = 0,
    saved: u64 = 0,
    skipped: u64 = 0,
    failed: u64 = 0,
    evicted: u64 = 0,
};

/// One rank's store; in a TP pair rank 0 decides saves, evictions and resumes and rank 1 mirrors them.
pub const Tier = struct {
    gpa: Allocator,
    cfg: Config,
    rank: u32,
    ns: [16]u8,
    ns_hex: [32]u8,
    dir: [:0]u8,
    slot_bytes: u64,
    planes: []u64,
    block_bytes: u64,
    states: std.AutoHashMapUnmanaged(Key, StateRec) = .empty,
    blocks: std.AutoHashMapUnmanaged(Key, BlockRec) = .empty,
    bytes: u64 = 0, // this namespace's files, pending and complete
    others: std.ArrayList(Other) = .empty,
    others_bytes: u64 = 0,
    counts: Counts = .{},
    // the writer (`mu` guards jobs, done, queued, stop)
    mu: c.pthread_mutex_t = .{},
    cv: c.pthread_cond_t = .{},
    jobs: Fifo(Job) = .{},
    done: std.ArrayList(Done) = .empty,
    queued: u64 = 0,
    stop: bool = false,
    running: bool = false, // the writer holds a job
    deadline: i128 = 0,
    thread: ?std.Thread = null,
    written: std.atomic.Value(u64) = .init(0), // the writer's totals (its own lines)
    clock: i128 = 0, // the last `used` handed out (wall ns, strictly increasing)

    pub const readState = index.readState;
    pub const readRun = index.readRun;
    pub const readRunOr = index.readRunOr;
    pub const blockPath = format.blockPath;
    pub const statePath = format.statePath;
    pub const push = writer.push;
    pub const pushDelete = writer.pushDelete;

    /// The store of `identity` under cfg.root (made when missing), its index rebuilt, its writer started.
    pub fn open(gpa: Allocator, cfg: Config, identity: []const u8, rank: u32, slot_bytes: u64, planes: []const u64) !*Tier {
        const t = try gpa.create(Tier);
        errdefer gpa.destroy(t);
        const ns = namespaceOf(identity);
        t.* = .{ .gpa = gpa, .cfg = cfg, .rank = rank, .ns = ns, .ns_hex = std.fmt.bytesToHex(ns, .lower), .dir = undefined, .slot_bytes = slot_bytes, .planes = undefined, .block_bytes = sum(planes) };
        t.cfg.root = try gpa.dupe(u8, cfg.root);
        errdefer gpa.free(t.cfg.root);
        t.planes = try gpa.dupe(u64, planes);
        errdefer gpa.free(t.planes);
        t.dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ cfg.root, &t.ns_hex }, 0);
        errdefer gpa.free(t.dir);
        try fs.makePath(gpa, t.dir);
        try t.sub("s");
        try t.sub("b");
        try namespace.claim(t, identity);
        errdefer t.freeIndex();
        try index.scan(t);
        namespace.scanOthers(t, identity);
        t.thread = try std.Thread.spawn(.{}, writer.writer, .{t});
        return t;
    }

    /// Stops the writer once the queued writes are done (at most cfg.drain_ns; the rest are dropped) and frees all.
    pub fn close(t: *Tier) void {
        _ = c.pthread_mutex_lock(&t.mu);
        t.stop = true;
        t.deadline = now() + t.cfg.drain_ns;
        _ = c.pthread_cond_broadcast(&t.cv);
        _ = c.pthread_mutex_unlock(&t.mu);
        if (t.thread) |th| th.join();
        t.drain();
        for (t.jobs.rest()) |*j| writer.freeJob(t.gpa, j);
        t.jobs.deinit(t.gpa);
        t.done.deinit(t.gpa);
        t.freeIndex();
        t.others.deinit(t.gpa);
        t.gpa.free(t.dir);
        t.gpa.free(t.planes);
        t.gpa.free(t.cfg.root);
        t.gpa.destroy(t);
    }

    fn freeIndex(t: *Tier) void {
        var it = t.states.valueIterator();
        while (it.next()) |s| t.gpa.free(s.keys);
        t.states.deinit(t.gpa);
        t.blocks.deinit(t.gpa);
        t.states = .empty;
        t.blocks = .empty;
    }

    fn sub(t: *Tier, name: []const u8) !void {
        var buf: [1100]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ t.dir, name }, 0);
        const rc = c.mkdir(p, 0o700);
        if (rc != 0 and c.errno(rc) != .EXIST) return error.DiskDir;
    }

    // -- the leader's and the follower's decisions (their engine thread) ----------------------------------------

    /// The writer's results applied: written states complete, failed ones gone with their blocks.
    pub fn drain(t: *Tier) void {
        _ = c.pthread_mutex_lock(&t.mu);
        const list = t.done;
        t.done = .empty;
        _ = c.pthread_mutex_unlock(&t.mu);
        var l = list;
        defer l.deinit(t.gpa);
        for (l.items) |*d| {
            switch (d.job) {
                .run => |r| for (r.keys) |k| {
                    const b = t.blocks.getPtr(k) orelse continue;
                    if (d.ok) {
                        b.status = .complete;
                    } else if (b.refs == 0 and !b.pinned) {
                        t.forgetBlock(k, null);
                    } // (else it stays pending: the state that needs it fails at its own write and lets it go)
                },
                .state => |s| if (t.states.getPtr(s.key)) |rec| {
                    if (d.ok) {
                        rec.status = .complete;
                        t.counts.saved += 1;
                    } else {
                        t.counts.failed += 1;
                        t.dropState(s.key);
                    }
                },
                else => {},
            }
            writer.freeJob(t.gpa, &d.job);
        }
    }

    /// Waits up to `max_ns` for the writer to go idle, then applies its results; true when it emptied in time.
    pub fn flush(t: *Tier, max_ns: u64) bool {
        const until = now() + max_ns;
        while (true) {
            _ = c.pthread_mutex_lock(&t.mu);
            const idle = t.jobs.len() == 0 and !t.running;
            _ = c.pthread_mutex_unlock(&t.mu);
            if (idle) break;
            if (now() > until) return false;
            sleepNs(std.time.ns_per_ms);
        }
        t.drain();
        return true;
    }

    /// A state's place in the store: none, being written, or on disk.
    pub fn stateOf(t: *Tier, key: Key) ?Status {
        t.drain();
        return if (t.states.get(key)) |s| s.status else null;
    }

    /// The largest k in (k_min, k_max] whose state is on disk under hashes[k-1] (the prompt's chain).
    pub fn longest(t: *Tier, hashes: []const Key, k_min: u32, k_max: u32) ?u32 {
        t.drain();
        var k = @min(k_max, @as(u32, @intCast(hashes.len)));
        while (k > k_min) : (k -= 1) if (t.states.get(hashes[k - 1])) |s| if (s.status == .complete and s.k == k) return k;
        return null;
    }

    /// The bytes on the root's filesystem a writer may still use (statvfs, less nothing).
    pub fn freeBytes(t: *const Tier) u64 {
        if (t.cfg.free_override) |v| return v;
        return fsFree(t.dir);
    }

    pub fn queueRoom(t: *Tier) u64 {
        _ = c.pthread_mutex_lock(&t.mu);
        defer _ = c.pthread_mutex_unlock(&t.mu);
        return t.cfg.queue -| t.queued;
    }

    /// The files a run of n blocks makes, and the host bytes it holds while queued.
    fn runFiles(t: *const Tier, n: u64) u64 {
        return n * (head_bytes + t.block_bytes);
    }

    pub fn stateFile(t: *const Tier, k: u64) u64 {
        return head_bytes + 32 * k + t.slot_bytes;
    }

    /// Plans a prompt's blocks `keys` and, with `state`, the state at the last (keys: H_1..H_k); a skip changes none.
    pub fn plan(t: *Tier, keys: []const Key, state: bool, pin: bool) !Plan {
        t.drain();
        const gpa = t.gpa;
        if (keys.len == 0) return .{ .result = .ok };
        if (state and t.states.contains(keys[keys.len - 1])) return .{ .result = .{ .skip = .exists } };
        var runs: std.ArrayList(Run) = .empty;
        defer runs.deinit(gpa);
        var victims: std.ArrayList(Key) = .empty;
        defer victims.deinit(gpa);
        var missing: u64 = 0;
        // the blocks absent here become runs (pending, pinned to the pass when `pin`)
        for (keys, 0..) |k, i| {
            if (t.blocks.contains(k)) continue;
            const items = runs.items;
            if (items.len > 0 and items[items.len - 1].first + items[items.len - 1].n == i) {
                items[items.len - 1].n += 1;
            } else try runs.append(gpa, .{ .first = @intCast(i), .n = 1 });
            missing += 1;
        }
        if (!state and missing == 0) return .{ .result = .ok };
        const kk: u64 = keys.len;
        const charge_ = missing * (t.block_bytes + 32) + (if (state) kk * 32 + t.slot_bytes else 0);
        const files = t.runFiles(missing) + (if (state) t.stateFile(kk) else 0);
        if (charge_ > t.queueRoom()) return t.skipped(.queue);
        if (t.freeBytes() < t.cfg.floor + files) return t.skipped(.floor);
        if (t.bytes + t.others_bytes + files > t.cfg.budget) {
            namespace.sweepOthers(t, files);
            // the follower refuses what passes its budget; the leader evicts and tells it (plan.victims)
            if (t.cfg.follower) {
                if (t.bytes + t.others_bytes + files > t.cfg.budget + t.cfg.budget / 16) return t.skipped(.budget);
            } else {
                // the leader evicts the least recently used; this plan's blocks stay unless the plan is refused
                for (keys) |k| if (t.blocks.getPtr(k)) |b| {
                    b.guard = true;
                };
                var stands = false;
                defer t.unguard(keys, !stands);
                while (t.bytes + t.others_bytes + files > t.cfg.budget) {
                    const v = t.victim() orelse break;
                    try victims.append(gpa, v);
                    t.dropState(v);
                    t.counts.evicted += 1;
                }
                if (t.bytes + t.others_bytes + files > t.cfg.budget) {
                    var p = t.skipped(.budget);
                    p.victims = try victims.toOwnedSlice(gpa);
                    return p;
                }
                stands = true;
            }
        }
        // the plan stands: blocks pending, the state pending with its references, the queue reserved
        try t.blocks.ensureUnusedCapacity(gpa, @intCast(missing));
        if (state) try t.states.ensureUnusedCapacity(gpa, 1);
        const owned: []Key = if (state) try gpa.dupe(Key, keys) else &.{};
        const out_victims = victims.toOwnedSlice(gpa) catch |e| {
            gpa.free(owned);
            return e;
        };
        const out_runs = runs.toOwnedSlice(gpa) catch |e| {
            gpa.free(owned);
            gpa.free(out_victims);
            return e;
        };
        for (out_runs) |r| for (keys[r.first..][0..r.n]) |k| {
            t.blocks.putAssumeCapacity(k, .{ .bytes = head_bytes + t.block_bytes, .status = .pending });
            t.bytes += head_bytes + t.block_bytes;
        };
        if (pin) for (keys) |k| if (t.blocks.getPtr(k)) |b| {
            b.pinned = true;
        };
        if (state) {
            t.states.putAssumeCapacity(keys[keys.len - 1], .{ .k = @intCast(kk), .bytes = t.stateFile(kk), .used = t.tick(), .status = .pending, .keys = owned });
            t.bytes += t.stateFile(kk);
            for (keys) |k| t.blocks.getPtr(k).?.refs += 1;
        }
        _ = c.pthread_mutex_lock(&t.mu);
        t.queued += charge_;
        _ = c.pthread_mutex_unlock(&t.mu);
        return .{ .result = .ok, .victims = out_victims, .runs = out_runs, .charge = charge_ };
    }

    /// A plan's guards released; with `reclaim`, its blocks no state or pass holds go (an eviction may have left them).
    fn unguard(t: *Tier, keys: []const Key, reclaim: bool) void {
        var del: std.ArrayList(Key) = .empty;
        for (keys) |k| if (t.blocks.getPtr(k)) |b| {
            b.guard = false;
            if (reclaim and b.refs == 0 and !b.pinned) t.forgetBlock(k, &del);
        };
        t.pushDelete(&.{}, del.toOwnedSlice(t.gpa) catch &.{});
    }

    fn skipped(t: *Tier, why: Reason) Plan {
        t.counts.skipped += 1;
        return .{ .result = .{ .skip = why } };
    }

    /// A run's bytes (plane-major over its n blocks, gpa) handed to the writer, which frees them and their reservation.
    pub fn submitRun(t: *Tier, keys: []const Key, first_j: u32, data: []u8) !void {
        const owned = t.gpa.dupe(Key, keys) catch |e| {
            t.gpa.free(data);
            return e;
        };
        t.push(.{ .run = .{ .keys = owned, .first_j = first_j, .data = data, .charge = (t.block_bytes + 32) * keys.len, .t0 = now() } });
    }

    /// The state's own bytes (gpa, owned now) after its runs; written once they are.
    pub fn submitState(t: *Tier, keys: []const Key, slot: []u8, new_blocks: u32) !void {
        const owned = t.gpa.dupe(Key, keys) catch |e| {
            t.gpa.free(slot);
            return e;
        };
        const kk = keys.len;
        t.push(.{ .state = .{ .key = keys[kk - 1], .k = @intCast(kk), .keys = owned, .slot = slot, .charge = kk * 32 + t.slot_bytes, .t0 = now(), .new_blocks = new_blocks, .new_bytes = new_blocks * (head_bytes + t.block_bytes) + t.stateFile(kk) } });
    }

    /// A plan whose bytes never came (a device copy failed): its reservation released, its state and blocks gone.
    pub fn abandon(t: *Tier, p: *const Plan, keys: []const Key, state: bool, submitted_runs: usize) void {
        var left = p.charge;
        for (p.runs[0..submitted_runs]) |r| left -|= (t.block_bytes + 32) * r.n;
        _ = c.pthread_mutex_lock(&t.mu);
        t.queued -|= left;
        _ = c.pthread_mutex_unlock(&t.mu);
        if (state) t.dropState(keys[keys.len - 1]);
        for (p.runs[submitted_runs..]) |r| for (keys[r.first..][0..r.n]) |k| if (t.blocks.get(k)) |b| if (b.refs == 0) t.forgetBlock(k, null);
    }

    /// The prompt pass ended: its streamed blocks no state took go.
    pub fn endPass(t: *Tier) void {
        t.drain();
        var gone: std.ArrayList(Key) = .empty;
        defer gone.deinit(t.gpa);
        var it = t.blocks.iterator();
        while (it.next()) |e| if (e.value_ptr.pinned) {
            e.value_ptr.pinned = false;
            if (e.value_ptr.refs == 0) gone.append(t.gpa, e.key_ptr.*) catch {};
        };
        if (gone.items.len == 0) return;
        var del: std.ArrayList(Key) = .empty;
        for (gone.items) |k| t.forgetBlock(k, &del);
        t.pushDelete(&.{}, del.toOwnedSlice(t.gpa) catch return);
    }

    /// A block found missing or corrupt: every state on disk that lists it goes (each would fail the same way).
    pub fn dropBlock(t: *Tier, key: Key) usize {
        t.drain();
        var gone: std.ArrayList(Key) = .empty;
        defer gone.deinit(t.gpa);
        var it = t.states.iterator();
        while (it.next()) |e| if (e.value_ptr.status == .complete) for (e.value_ptr.keys) |bk| if (std.mem.eql(u8, &bk, &key)) {
            gone.append(t.gpa, e.key_ptr.*) catch {};
            break;
        };
        for (gone.items) |k| t.dropState(k);
        t.line("event=drop-block states={d} key={s}", .{ gone.items.len, &short(key) });
        return gone.items.len;
    }

    /// States forgotten (evicted, found bad, or not on rank 0): their files and the blocks only they held go.
    pub fn drop(t: *Tier, keys: []const Key) void {
        t.drain();
        for (keys) |k| t.dropState(k);
    }

    fn dropState(t: *Tier, key: Key) void {
        const kv = t.states.fetchRemove(key) orelse return;
        t.bytes -|= kv.value.bytes;
        var del: std.ArrayList(Key) = .empty;
        for (kv.value.keys) |bk| if (t.blocks.getPtr(bk)) |b| {
            b.refs -|= 1;
            if (b.refs == 0 and !b.pinned and !b.guard) t.forgetBlock(bk, &del);
        };
        t.gpa.free(kv.value.keys);
        const states = t.gpa.dupe(Key, &.{key}) catch {
            del.deinit(t.gpa);
            return;
        };
        t.pushDelete(states, del.toOwnedSlice(t.gpa) catch &.{});
    }

    fn forgetBlock(t: *Tier, key: Key, del: ?*std.ArrayList(Key)) void {
        const b = t.blocks.fetchRemove(key) orelse return;
        t.bytes -|= b.value.bytes;
        if (del) |d| d.append(t.gpa, key) catch {} else {
            const one = t.gpa.dupe(Key, &.{key}) catch return;
            t.pushDelete(&.{}, one);
        }
    }

    /// The least recently used state on disk (pending ones are never evicted).
    fn victim(t: *Tier) ?Key {
        var best: ?Key = null;
        var used: i128 = 0;
        var it = t.states.iterator();
        while (it.next()) |e| if (e.value_ptr.status == .complete and (best == null or e.value_ptr.used < used)) {
            best = e.key_ptr.*;
            used = e.value_ptr.used;
        };
        return best;
    }

    /// A state was resumed: the most recently used, here and in its file's mtime (the order after a restart).
    pub fn touch(t: *Tier, key: Key) void {
        const s = t.states.getPtr(key) orelse return;
        s.used = t.tick();
        var pb: [1400]u8 = undefined;
        fs.touchPath(t.statePath(&pb, key, false) catch return);
    }

    fn tick(t: *Tier) i128 {
        t.clock = @max(t.clock + 1, now());
        return t.clock;
    }

    /// The states on disk (gpa), for the startup reconcile.
    pub fn completeKeys(t: *Tier) ![]Key {
        t.drain();
        var out: std.ArrayList(Key) = .empty;
        var it = t.states.iterator();
        while (it.next()) |e| if (e.value_ptr.status == .complete) try out.append(t.gpa, e.key_ptr.*);
        return out.toOwnedSlice(t.gpa);
    }

    pub fn hasComplete(t: *const Tier, key: Key) bool {
        return if (t.states.get(key)) |s| s.status == .complete else false;
    }


    /// One log line, "sf-disk: rank=R " and the event's fields (tests stay quiet).
    pub fn line(t: *const Tier, comptime fmt: []const u8, args: anytype) void {
        if (builtin.is_test) return;
        std.log.info("sf-disk: rank={d} " ++ fmt, .{t.rank} ++ args);
    }
};

test {
    _ = @import("prompt_disk_test.zig");
}
