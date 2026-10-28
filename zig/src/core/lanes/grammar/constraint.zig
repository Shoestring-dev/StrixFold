//! One reply's grammar as a backend's draws read it. `reset` puts it at the reply so far (the committed tokens: it
//! keeps its place between rounds and reads only the new ones), `feed` moves it over a verify window's drafted rows,
//! and at any point `allows` answers for one token and `top` gives a row's best allowed tokens: the best of the row's
//! top candidates the grammar allows, else (few allowed) the full mask from one walk of the vocabulary trie. Before
//! the `after` token (thinking on) nothing is constrained. A grammar that would allow no token at all (EOS ignored
//! after the value ended) stops constraining rather than strand the reply.
const std = @import("std");
const prog = @import("program.zig");
const machine = @import("machine.zig");
const vocab_mod = @import("vocab.zig");
const Allocator = std.mem.Allocator;
const State = machine.State;
const Vocab = vocab_mod.Vocab;

pub const Phase = enum {
    waiting, // the think block is open: nothing constrained yet
    active,
    ended, // EOS taken at the grammar's end
    off, // left: a token outside it was committed, or it allowed nothing
};

/// The row's top candidates read before falling back to the full mask.
pub const scan_least = 64;

const Walk = struct { node: u32, i: u32 };

pub const Constraint = struct {
    gpa: Allocator,
    program: *prog.Program,
    vocab: *const Vocab,
    after: ?u32,
    eos: []u32,
    seen: usize = 0,
    base: State,
    base_phase: Phase,
    cur: State,
    cur_phase: Phase,
    tmp: State = undefined,
    states: []State,
    stack: []Walk,
    mask: []u64,
    cand_ids: []u32,
    cand_vals: []f32,
    /// What the draws cost: full masks walked and single-token checks.
    masks: u64 = 0,
    checks: u64 = 0,

    /// The grammar of `s` for a reply that ends on any of `eos`; error.Unsupported with `diag` set for a request the
    /// server should have refused.
    pub fn init(gpa: Allocator, vocab: *const Vocab, s: prog.Structure, eos: []const u32, diag: *prog.Diag) !*Constraint {
        const p = try prog.compile(gpa, s, diag);
        errdefer {
            p.deinit();
            gpa.destroy(p);
        }
        const c = try gpa.create(Constraint);
        errdefer gpa.destroy(c);
        const start = State.init(p);
        const phase: Phase = if (s.after != null) .waiting else .active;
        c.* = .{
            .gpa = gpa,
            .program = p,
            .vocab = vocab,
            .after = s.after,
            .eos = try gpa.dupe(u32, eos),
            .base = start,
            .base_phase = phase,
            .cur = start,
            .cur_phase = phase,
            .states = &.{},
            .stack = &.{},
            .mask = &.{},
            .cand_ids = &.{},
            .cand_vals = &.{},
        };
        errdefer gpa.free(c.eos);
        c.states = try gpa.alloc(State, vocab.longest + 1);
        errdefer gpa.free(c.states);
        c.stack = try gpa.alloc(Walk, vocab.longest + 1);
        errdefer gpa.free(c.stack);
        c.mask = try gpa.alloc(u64, vocab.words());
        errdefer gpa.free(c.mask);
        c.cand_ids = try gpa.alloc(u32, 1024);
        errdefer gpa.free(c.cand_ids);
        c.cand_vals = try gpa.alloc(f32, 1024);
        return c;
    }

    pub fn deinit(c: *Constraint) void {
        const gpa = c.gpa;
        c.program.deinit();
        gpa.destroy(c.program);
        gpa.free(c.eos);
        gpa.free(c.states);
        gpa.free(c.stack);
        gpa.free(c.mask);
        gpa.free(c.cand_ids);
        gpa.free(c.cand_vals);
        gpa.destroy(c);
    }

    fn isEos(c: *const Constraint, t: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos, t) != null;
    }

    /// `t` taken by a machine in `phase` at `s`: the phase after it (off: outside the grammar).
    fn take(c: *Constraint, s: *State, phase: Phase, t: u32) Phase {
        switch (phase) {
            .waiting => {
                if (c.after != null and t == c.after.?) {
                    s.* = State.init(c.program);
                    return .active;
                }
                return .waiting;
            },
            .active => {
                if (t >= c.vocab.size or c.vocab.special[t]) return if (c.isEos(t) and machine.canEnd(c.program, s)) .ended else .off;
                return if (machine.feed(c.program, s, c.vocab.text(t))) .active else .off;
            },
            .ended, .off => return .off,
        }
    }

    /// The grammar at the end of the reply so far (`emitted`: every committed reply token); the window starts here.
    pub fn reset(c: *Constraint, emitted: []const u32) void {
        if (emitted.len < c.seen) {
            // a reply that restarted (not done by the round loop): read it from the start
            c.base = State.init(c.program);
            c.base_phase = if (c.after != null) .waiting else .active;
            c.seen = 0;
        }
        for (emitted[c.seen..]) |t| {
            const was = c.base_phase;
            c.base_phase = c.take(&c.base, c.base_phase, t);
            if (was == .active and c.base_phase == .off) std.log.warn("constrained reply took token {d} outside its grammar; the rest is unconstrained", .{t});
        }
        c.seen = emitted.len;
        c.cur.copy(&c.base);
        c.cur_phase = c.base_phase;
    }

    /// Moves past a window row's token (a draft): false when the grammar refuses it, and nothing after it is drawn
    /// constrained (those rows are never kept).
    pub fn feed(c: *Constraint, t: u32) bool {
        const next = c.take(&c.cur, c.cur_phase, t);
        const ok = !(c.cur_phase == .active and next == .off);
        c.cur_phase = next;
        return ok;
    }

    /// Whether the draw here is free (thinking, or the grammar left).
    pub fn free(c: *const Constraint) bool {
        return c.cur_phase != .active;
    }

    pub fn allows(c: *Constraint, t: u32) bool {
        if (c.cur_phase != .active) return true;
        c.checks += 1;
        if (t >= c.vocab.size) return false;
        if (c.vocab.special[t]) return c.isEos(t) and machine.canEnd(c.program, &c.cur);
        c.tmp.copy(&c.cur);
        return machine.feed(c.program, &c.tmp, c.vocab.text(t));
    }

    fn set(mask: []u64, t: u32) void {
        mask[t / 64] |= @as(u64, 1) << @intCast(t % 64);
    }

    pub fn isSet(mask: []const u64, t: u32) bool {
        return t / 64 < mask.len and mask[t / 64] & (@as(u64, 1) << @intCast(t % 64)) != 0;
    }

    /// Every token allowed here as a bitset over the vocabulary; null when nothing is constrained here (or the grammar
    /// allows nothing, which turns it off).
    pub fn fill(c: *Constraint) ?[]const u64 {
        if (c.cur_phase != .active) return null;
        c.masks += 1;
        const v = c.vocab;
        @memset(c.mask, 0);
        const f = c.cur.frames[c.cur.depth - 1];
        var tree = &v.all;
        if (f.st == .s_body and f.k == 0 and c.program.nodes[f.node].string.dfa == null) {
            // plain string text keeps the string open with `n` more characters: every such token that fits
            const str = c.program.nodes[f.node].string;
            const room = str.max - f.n;
            if (room >= v.json_plain_most) {
                @memcpy(c.mask, v.json_plain_bits);
            } else {
                @memcpy(c.mask, v.json_plain_upto[@min(room, vocab_mod.short_chars - 1)]);
                if (room >= vocab_mod.short_chars) for (v.json_long, v.json_long_chars) |t, n| if (n <= room) set(c.mask, t);
            }
            tree = &v.json_rest;
        } else if (f.st == .w_body and f.k == 0 and f.x & 0xff == 0) {
            @memcpy(c.mask, v.raw_plain_bits);
            tree = &v.raw_rest;
        }
        c.walk(tree);
        if (machine.canEnd(c.program, &c.cur)) for (c.eos) |t| if (t < v.size) set(c.mask, t);
        for (c.mask) |w| if (w != 0) return c.mask;
        std.log.warn("constrained reply: its grammar allows no token here; the rest is unconstrained", .{});
        c.cur_phase = .off;
        return null;
    }

    /// Marks every token of `tree` the machine takes from `cur`: depth first, states[d] after a node's d bytes.
    fn walk(c: *Constraint, tree: *const vocab_mod.Trie) void {
        const p = c.program;
        c.states[0].copy(&c.cur);
        c.stack[0] = .{ .node = 0, .i = 0 };
        var d: usize = 0;
        while (true) {
            const at = &c.stack[d];
            const node = tree.nodes[at.node];
            if (at.i == node.nedge) {
                if (d == 0) break;
                d -= 1;
                continue;
            }
            const e = tree.edges[node.edge0 + at.i];
            at.i += 1;
            const s = &c.states[d + 1];
            s.copy(&c.states[d]);
            if (!machine.step(p, s, e.byte)) continue;
            const child = tree.nodes[e.next];
            for (tree.toks[child.tok0..][0..child.ntok]) |t| set(c.mask, t);
            if (child.nedge > 0) {
                d += 1;
                c.stack[d] = .{ .node = e.next, .i = 0 };
            }
        }
    }

    /// The row's best `want` allowed tokens (value, then lower id first), into ids/vals; how many (fewer when fewer
    /// are allowed). A free draw gives the row's best as they are.
    pub fn top(c: *Constraint, row: []const f32, want: usize, ids: []u32, vals: []f32) usize {
        std.debug.assert(want > 0 and ids.len >= want and vals.len >= want);
        if (c.cur_phase != .active) return best(row, null, ids[0..want], vals[0..want]);
        const scan = @min(row.len, @min(c.cand_ids.len, @max(scan_least, 4 * want)));
        const got = best(row, null, c.cand_ids[0..scan], c.cand_vals[0..scan]);
        var n: usize = 0;
        for (c.cand_ids[0..got], c.cand_vals[0..got]) |t, x| {
            if (!c.allows(t)) continue;
            ids[n] = t;
            vals[n] = x;
            n += 1;
            if (n == want) return n;
        }
        if (got == row.len) return n; // every token was read
        const mask = c.fill() orelse return best(row, null, ids[0..want], vals[0..want]);
        return best(row, mask, ids[0..want], vals[0..want]);
    }
};

/// a before b: the larger value, then the lower id.
fn before(va: f32, ia: u32, vb: f32, ib: u32) bool {
    return va > vb or (va == vb and ia < ib);
}

/// The row's first ids.len tokens in (value, id) order, only those `mask` sets when it is given; how many.
pub fn best(row: []const f32, mask: ?[]const u64, ids: []u32, vals: []f32) usize {
    var top: Top = .{ .ids = ids, .vals = vals };
    const m = mask orelse {
        for (row, 0..) |v, i| top.offer(v, @intCast(i));
        return top.n;
    };
    var set: usize = 0;
    for (m) |w| set += @popCount(w);
    if (set * 8 < row.len) {
        // a sparse mask: its tokens only
        for (m, 0..) |w0, wi| {
            var w = w0;
            while (w != 0) : (w &= w - 1) {
                const id: u32 = @intCast(wi * 64 + @ctz(w));
                if (id < row.len) top.offer(row[id], id);
            }
        }
        return top.n;
    }
    for (row, 0..) |v, i| {
        const id: u32 = @intCast(i);
        if (Constraint.isSet(m, id)) top.offer(v, id);
    }
    return top.n;
}

/// The best ids.len (value, id) pairs offered so far, in order.
const Top = struct {
    ids: []u32,
    vals: []f32,
    n: usize = 0,

    fn offer(t: *Top, v: f32, id: u32) void {
        const k = t.ids.len;
        if (t.n == k and !before(v, id, t.vals[k - 1], t.ids[k - 1])) return;
        var j: usize = if (t.n < k) t.n else k - 1;
        while (j > 0 and before(v, id, t.vals[j - 1], t.ids[j - 1])) : (j -= 1) {
            t.vals[j] = t.vals[j - 1];
            t.ids[j] = t.ids[j - 1];
        }
        t.vals[j] = v;
        t.ids[j] = id;
        if (t.n < k) t.n += 1;
    }
};
