//! The byte machine a program (program.zig) runs: a stack of frames, one per open value, each with its own small
//! state; `step` takes one byte or refuses it. A refused byte leaves the state undefined, so callers step a copy.
//! JSON whitespace goes between tokens only, never before the root value: a run of spaces and tabs with at most one
//! newline (no carriage return), ws_cap bytes at most, so a reply cannot pad itself out in blank lines (served
//! replies did, before this rule, when the grammar refused the prose the model wanted); numbers are at most num_cap
//! bytes; strings are kept well-formed UTF-8.
const std = @import("std");
const prog = @import("program.zig");
const trie = @import("trie.zig");
const regex = @import("regex.zig");
const Program = prog.Program;

pub const max_depth = 40;
pub const ws_cap = 32;
pub const num_cap = 32;
/// A DSML parameter name of a tool without a schema, at most this many characters.
pub const free_key_cap = 64;

pub const St = enum(u8) {
    r_value, // the root: its value's first byte
    r_done,
    o_open,
    o_key, // x: key trie node
    o_colon,
    o_value,
    o_after,
    o_comma,
    a_open,
    a_after,
    a_comma,
    s_body, // k: UTF-8 decoder, n: characters, x: pattern DFA state
    s_esc,
    s_u1, // x: the \u escape's value so far, bit 16 set while it must be a low surrogate
    s_u2,
    s_u3,
    s_u4,
    s_hi, // a high surrogate's escape: its low half's backslash next
    s_hi_u,
    n_minus,
    n_zero,
    n_int,
    n_dot,
    n_frac,
    n_e,
    n_esign,
    n_exp,
    l_trie, // x: literal trie node
    x_dfa, // x: DFA state
    c_head, // x: trie node
    c_next,
    p_keys, // k: parameters behind, x: trie node
    p_flag, // k: the parameter's index + 1
    p_jvalue,
    p_close,
    p_freekey, // n: characters of the name, k: UTF-8 decoder
    p_freetail,
    w_body, // x: the closing tag's match (raw_*), k: UTF-8 decoder
    w_gt,
};

pub const Frame = struct { node: u32, x: u32 = 0, n: u32 = 0, st: St, k: u8 = 0 };

pub const State = struct {
    depth: u8 = 1,
    ws: u8 = 0, // whitespace bytes in the current run
    nl: bool = false, // the run has its newline
    nlen: u8 = 0,
    num: [num_cap]u8 = undefined,
    frames: [max_depth]Frame = undefined,

    pub fn init(p: *const Program) State {
        var s: State = .{};
        s.frames[0] = .{ .node = p.root, .st = .r_value };
        return s;
    }

    /// `src` into `dst`: the live frames only.
    pub fn copy(dst: *State, src: *const State) void {
        dst.depth = src.depth;
        dst.ws = src.ws;
        dst.nl = src.nl;
        dst.nlen = src.nlen;
        dst.num = src.num;
        @memcpy(dst.frames[0..src.depth], src.frames[0..src.depth]);
    }
};

fn isWs(b: u8) bool {
    return b == ' ' or b == '\n' or b == '\t';
}

fn ws(s: *State, b: u8) bool {
    if (s.ws >= ws_cap) return false;
    if (b == '\n') {
        if (s.nl) return false;
        s.nl = true;
    }
    s.ws += 1;
    return true;
}

fn push(s: *State, f: Frame) bool {
    if (s.depth == max_depth) return false;
    s.frames[s.depth] = f;
    s.depth += 1;
    return true;
}

fn pop(s: *State) bool {
    s.depth -= 1;
    return true;
}

/// A UTF-8 decoder in `k`: the continuation bytes still due << 3 | the range the next one must fall in. False on a
/// byte well-formed UTF-8 cannot have next.
pub fn utf8(k: *u8, b: u8) bool {
    if (k.* == 0) {
        if (b < 0x80) return true;
        k.* = switch (b) {
            0xC2...0xDF => 1 << 3,
            0xE0 => 2 << 3 | 1,
            0xE1...0xEC, 0xEE, 0xEF => 2 << 3,
            0xED => 2 << 3 | 2,
            0xF0 => 3 << 3 | 3,
            0xF1...0xF3 => 3 << 3,
            0xF4 => 3 << 3 | 4,
            else => return false,
        };
        return true;
    }
    const lo: u8, const hi: u8 = switch (k.* & 7) {
        0 => .{ 0x80, 0xBF },
        1 => .{ 0xA0, 0xBF },
        2 => .{ 0x80, 0x9F },
        3 => .{ 0x90, 0xBF },
        else => .{ 0x80, 0x8F },
    };
    if (b < lo or b > hi) return false;
    k.* = ((k.* >> 3) - 1) << 3;
    return true;
}

fn isDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}

/// The text a DSML raw value may not hold but as its closing tag's tail ("<" before it, ">" after).
pub const raw_tail = "/\u{ff5c}DSML\u{ff5c} parameter";
const raw_lt_start: u32 = 1 << 8; // the byte before the match was '<'
const raw_lt_last: u32 = 1 << 9; // the last byte was '<'

/// Takes `b`, or refuses it (the state is then undefined).
pub fn step(p: *const Program, s: *State, b: u8) bool {
    if (!isWs(b)) {
        s.ws = 0;
        s.nl = false;
    }
    const t = &p.tries;
    while (true) {
        const f = &s.frames[s.depth - 1];
        switch (f.st) {
            .r_value => {
                f.st = .r_done;
                return begin(p, s, p.root, b);
            },
            .r_done => return false,
            .o_open, .o_comma => {
                if (isWs(b)) return ws(s, b);
                const o = p.nodes[f.node].object;
                if (b == '}' and f.st == .o_open) return closable(o, f) and pop(s);
                if (b != '"' or !addable(o, f)) return false;
                if (o.extra != null) {
                    f.st = .o_colon;
                    return begin(p, s, prog.any_string, b);
                }
                f.x = t.child(o.keys[f.k], '"') orelse return false;
                f.st = .o_key;
                return true;
            },
            .o_key => {
                const next = t.child(f.x, b) orelse return false;
                f.x = next;
                if (t.value(next)) |v| {
                    f.k = @intCast(v + 1);
                    f.st = .o_colon;
                }
                return true;
            },
            .o_colon => {
                if (isWs(b)) return ws(s, b);
                if (b != ':') return false;
                f.st = .o_value;
                return true;
            },
            .o_value => {
                if (isWs(b)) return ws(s, b);
                const o = p.nodes[f.node].object;
                const v = o.extra orelse o.props[f.k - 1];
                f.st = .o_after;
                f.n += 1;
                return begin(p, s, v, b);
            },
            .o_after => {
                if (isWs(b)) return ws(s, b);
                const o = p.nodes[f.node].object;
                if (b == '}') return closable(o, f) and pop(s);
                if (b != ',' or !addable(o, f)) return false;
                f.st = .o_comma;
                return true;
            },
            .a_open, .a_comma => {
                if (isWs(b)) return ws(s, b);
                const a = p.nodes[f.node].array;
                if (b == ']' and f.st == .a_open) return f.n >= a.min and pop(s);
                const item = itemOf(a, f.n) orelse return false;
                f.st = .a_after;
                f.n += 1;
                return begin(p, s, item, b);
            },
            .a_after => {
                if (isWs(b)) return ws(s, b);
                const a = p.nodes[f.node].array;
                if (b == ']') return f.n >= a.min and pop(s);
                if (b != ',' or itemOf(a, f.n) == null) return false;
                f.st = .a_comma;
                return true;
            },
            .s_body => {
                const str = p.nodes[f.node].string;
                const dfa: ?*const regex.Dfa = if (str.dfa) |d| &p.dfas[d] else null;
                if (b == '"') {
                    if (f.k != 0 or f.n < str.min) return false;
                    if (dfa) |d| if (!d.accept[f.x]) return false;
                    return pop(s);
                }
                if (b < 0x20) return false;
                if (b == '\\') {
                    if (dfa != null or f.k != 0 or f.n >= str.max) return false;
                    f.n += 1;
                    f.st = .s_esc;
                    return true;
                }
                if (f.k == 0) {
                    if (f.n >= str.max) return false;
                    f.n += 1;
                }
                if (!utf8(&f.k, b)) return false;
                if (dfa) |d| {
                    f.x = d.step(f.x, b);
                    if (f.x == regex.dead) return false;
                }
                return true;
            },
            .s_esc => {
                if (b == 'u') {
                    f.st = .s_u1;
                    f.x = 0;
                    return true;
                }
                if (std.mem.indexOfScalar(u8, "\"\\/bfnrt", b) == null) return false;
                f.st = .s_body;
                return true;
            },
            .s_u1, .s_u2, .s_u3, .s_u4 => {
                const h = std.fmt.charToDigit(b, 16) catch return false;
                f.x = (f.x & 0x10000) | ((f.x & 0xfff) << 4) | h;
                if (f.st != .s_u4) {
                    f.st = @fromBackingInt(@intCast(@backingInt(f.st) + 1));
                    return true;
                }
                // surrogates only in pairs (a high half, then \u and a low half), so any JSON reader takes the string
                const v = f.x & 0xffff;
                const low = v >= 0xdc00 and v <= 0xdfff;
                if (f.x & 0x10000 != 0) {
                    if (!low) return false;
                } else if (v >= 0xd800 and v <= 0xdbff) {
                    f.st = .s_hi;
                    f.x = 0x10000;
                    return true;
                } else if (low) return false;
                f.st = .s_body;
                f.x = 0;
                return true;
            },
            .s_hi => {
                if (b != '\\') return false;
                f.st = .s_hi_u;
                return true;
            },
            .s_hi_u => {
                if (b != 'u') return false;
                f.st = .s_u1;
                return true;
            },
            .n_minus, .n_zero, .n_int, .n_dot, .n_frac, .n_e, .n_esign, .n_exp => {
                const nd = p.nodes[f.node].number;
                const bounded = nd.lo != null or nd.hi != null;
                const next: ?St = switch (f.st) {
                    // an integer is never -0 (some readers take it for the float -0.0)
                    .n_minus => if (b == '0') (if (nd.integer) null else .n_zero) else if (isDigit(b)) .n_int else null,
                    .n_zero => if (b == '.' and !nd.integer) .n_dot else if ((b == 'e' or b == 'E') and !nd.integer and !bounded) .n_e else null,
                    .n_int => if (isDigit(b)) .n_int else if (b == '.' and !nd.integer) .n_dot else if ((b == 'e' or b == 'E') and !nd.integer and !bounded) .n_e else null,
                    .n_dot => if (isDigit(b)) .n_frac else null,
                    .n_frac => if (isDigit(b)) .n_frac else if ((b == 'e' or b == 'E') and !bounded) .n_e else null,
                    .n_e => if (b == '+' or b == '-') .n_esign else if (isDigit(b)) .n_exp else null,
                    .n_esign => if (isDigit(b)) .n_exp else null,
                    .n_exp => if (isDigit(b)) .n_exp else null,
                    else => unreachable,
                };
                if (next) |st| {
                    if (!record(s, b, st)) return false;
                    f.st = st;
                    return !bounded or feasible(nd, s.num[0..s.nlen], st);
                }
                // anything else ends the number, and the byte is its parent's
                if (!numberEnds(nd, s, f.st)) return false;
                _ = pop(s);
                continue;
            },
            .l_trie => {
                if (t.child(f.x, b)) |next| {
                    f.x = next;
                    if (t.leaf(next)) _ = pop(s);
                    return true;
                }
                if (t.value(f.x) == null) return false;
                _ = pop(s); // a literal that is a prefix of another ends here
                continue;
            },
            .x_dfa => {
                const d = &p.dfas[p.nodes[f.node].regex];
                const next = d.step(f.x, b);
                if (next == regex.dead) return false;
                f.x = next;
                return true;
            },
            .c_head, .c_next => {
                const c = p.nodes[f.node].calls;
                const next = t.child(f.x, b) orelse return false;
                f.x = next;
                const v = t.value(next) orelse return true;
                if (v == prog.end_mark) return pop(s);
                f.st = .c_next;
                f.x = c.next;
                const params = c.params[@intCast(v)];
                return push(s, .{ .node = params, .st = .p_keys, .x = p.nodes[params].params.keys[0] });
            },
            .p_keys => {
                const pr = p.nodes[f.node].params;
                const next = t.child(f.x, b) orelse return false;
                f.x = next;
                const v = t.value(next) orelse return true;
                if (v == prog.end_mark) return pop(s);
                if (v == prog.free_mark) {
                    f.st = .p_freekey;
                    f.n = 0;
                    return true;
                }
                f.k = @intCast(v + 1);
                f.st = .p_flag;
                f.x = pr.flags[@intCast(v)];
                return true;
            },
            .p_flag, .p_freetail => {
                const pr = p.nodes[f.node].params;
                const next = t.child(f.x, b) orelse return false;
                f.x = next;
                const v = t.value(next) orelse return true;
                if (v == 1) {
                    f.st = .p_jvalue;
                    return true;
                }
                // a raw value: the frame waits for the next parameter while the value's own frame reads it
                const raw_enum = if (pr.free) trie.none else pr.raw_enums[f.k - 1];
                f.st = .p_keys;
                f.x = pr.keys[if (pr.free) 1 else f.k];
                if (raw_enum != trie.none) return push(s, .{ .node = f.node, .st = .l_trie, .x = raw_enum });
                return push(s, .{ .node = prog.raw_value, .st = .w_body });
            },
            .p_jvalue => {
                const pr = p.nodes[f.node].params;
                f.st = .p_close;
                f.x = pr.close;
                return begin(p, s, if (pr.free) prog.any_value else pr.values[f.k - 1], b);
            },
            .p_close => {
                const pr = p.nodes[f.node].params;
                const next = t.child(f.x, b) orelse return false;
                f.x = next;
                if (t.leaf(next)) {
                    f.st = .p_keys;
                    f.x = pr.keys[if (pr.free) 1 else f.k];
                }
                return true;
            },
            .p_freekey => {
                if (b == '"') {
                    if (f.n == 0 or f.k != 0) return false;
                    f.st = .p_freetail;
                    f.x = p.nodes[f.node].params.free_tail;
                    return true;
                }
                if (b < 0x20 or b == '<' or b == '>') return false;
                if (f.k == 0) {
                    // characters are counted where they start, so a name at the cap still finishes its last one
                    if (f.n >= free_key_cap) return false;
                    f.n += 1;
                }
                return utf8(&f.k, b);
            },
            .w_body => {
                if (!utf8(&f.k, b)) return false;
                var m = f.x & 0xff;
                var lt = f.x & raw_lt_start;
                if (m > 0 and raw_tail[m] == b) {
                    m += 1;
                } else if (b == raw_tail[0]) {
                    m = 1;
                    lt = if (f.x & raw_lt_last != 0) raw_lt_start else 0;
                } else m = 0;
                if (m == raw_tail.len) {
                    if (lt == 0) return false; // the parser would end the value here, without its '<'
                    f.st = .w_gt;
                    return true;
                }
                f.x = m | lt | (if (b == '<') raw_lt_last else 0);
                return true;
            },
            .w_gt => return b == '>' and pop(s),
        }
    }
}

fn closable(o: prog.Object, f: *const Frame) bool {
    if (f.n < o.min) return false;
    return o.props.len == 0 or o.closable[f.k];
}

fn addable(o: prog.Object, f: *const Frame) bool {
    if (f.n >= o.max) return false;
    if (o.extra != null) return true;
    return f.k < o.props.len;
}

fn itemOf(a: prog.Array, i: u32) ?prog.NodeId {
    if (i >= a.max) return null;
    if (i < a.prefix.len) return a.prefix[i];
    return a.items;
}

/// Pushes the frame of a value of node `id` that starts with `b`.
fn begin(p: *const Program, s: *State, id: prog.NodeId, b: u8) bool {
    const t = &p.tries;
    switch (p.nodes[id]) {
        .any => return begin(p, s, switch (b) {
            '{' => prog.any_object,
            '[' => prog.any_array,
            '"' => prog.any_string,
            '-', '0'...'9' => prog.any_number,
            't', 'f' => prog.boolean,
            'n' => prog.null_value,
            else => return false,
        }, b),
        .object => return b == '{' and push(s, .{ .node = id, .st = .o_open }),
        .array => return b == '[' and push(s, .{ .node = id, .st = .a_open }),
        .string => |str| return b == '"' and push(s, .{ .node = id, .st = .s_body, .x = if (str.dfa) |d| p.dfas[d].start else 0 }),
        .number => |nd| {
            const st: St = if (b == '-') .n_minus else if (b == '0') .n_zero else if (isDigit(b)) .n_int else return false;
            s.nlen = 0;
            if (!record(s, b, st) or !push(s, .{ .node = id, .st = st })) return false;
            return (nd.lo == null and nd.hi == null) or feasible(nd, s.num[0..s.nlen], st);
        },
        .lits => |root| {
            const next = t.child(root, b) orelse return false;
            if (!push(s, .{ .node = id, .st = .l_trie, .x = next })) return false;
            if (t.leaf(next)) _ = pop(s);
            return true;
        },
        .one_of => |u| {
            const un = &p.unions[u];
            const alt = un.table[b];
            return alt != 0xff and begin(p, s, un.alts[alt], b);
        },
        .regex => |d| {
            const dfa = &p.dfas[d];
            const next = dfa.step(dfa.start, b);
            return next != regex.dead and push(s, .{ .node = id, .st = .x_dfa, .x = next });
        },
        .calls => |c| {
            const next = t.child(c.head, b) orelse return false;
            return push(s, .{ .node = id, .st = .c_head, .x = next });
        },
        .params, .raw, .pending => return false,
    }
}

/// Appends a number's byte, keeping room for one more after a state that cannot end.
fn record(s: *State, b: u8, st: St) bool {
    const can_end = st == .n_zero or st == .n_int or st == .n_frac or st == .n_exp;
    if (s.nlen + @as(u8, if (can_end) 1 else 2) > num_cap) return false;
    s.num[s.nlen] = b;
    s.nlen += 1;
    return true;
}

fn numberEnds(nd: prog.Num, s: *const State, st: St) bool {
    if (!(st == .n_zero or st == .n_int or st == .n_frac or st == .n_exp)) return false;
    if (nd.lo == null and nd.hi == null) return true;
    const v = std.fmt.parseFloat(f64, s.num[0..s.nlen]) catch return false;
    if (nd.lo) |l| if (v < l.v or (v == l.v and l.open)) return false;
    if (nd.hi) |h| if (v > h.v or (v == h.v and h.open)) return false;
    return true;
}

/// Whether magnitudes [lo, hi] (hi left out when `open`), negated with `neg`, meet the node's bounds.
fn meets(nd: prog.Num, neg: bool, lo: f64, hi: f64, open: bool) bool {
    if (neg) return within(nd, -hi, open, -lo, false);
    return within(nd, lo, false, hi, open);
}

/// Whether [a, z] (each end left out when open) meets the node's bounds.
fn within(nd: prog.Num, a0: f64, a_open0: bool, z0: f64, z_open0: bool) bool {
    var a = a0;
    var a_open = a_open0;
    var z = z0;
    var z_open = z_open0;
    if (nd.lo) |l| if (l.v > a or (l.v == a and l.open)) {
        a = l.v;
        a_open = l.open;
    };
    if (nd.hi) |h| if (h.v < z or (h.v == z and h.open)) {
        z = h.v;
        z_open = h.open;
    };
    return a < z or (a == z and !a_open and !z_open);
}

/// Whether a bounded number that begins `text` (now in state `st`) can still end within its bounds.
fn feasible(nd: prog.Num, text: []const u8, st: St) bool {
    const neg = text[0] == '-';
    const body = if (neg) text[1..] else text;
    switch (st) {
        .n_minus => return within(nd, -std.math.inf(f64), false, 0, true), // a minus sign promises a value below 0
        .n_zero => return meets(nd, neg, 0, if (nd.integer) 0 else 1, !nd.integer),
        .n_int => {
            const whole = std.fmt.parseFloat(f64, body) catch return false;
            var scale: f64 = 1;
            var more: usize = 0;
            // more digits multiply the magnitude by ten each; 17 of them pass any bound this machine takes
            while (body.len + more < num_cap and more <= 17) : (more += 1) {
                const lo = whole * scale;
                const hi = (whole + 1) * scale;
                if (meets(nd, neg, lo, if (nd.integer) hi - 1 else hi, !nd.integer)) return true;
                scale *= 10;
            }
            return false;
        },
        .n_dot, .n_frac => {
            const dot = std.mem.indexOfScalar(u8, body, '.').?;
            const lo = std.fmt.parseFloat(f64, if (st == .n_dot) body[0..dot] else body) catch return false;
            const width = std.math.pow(f64, 10, -@as(f64, @floatFromInt(body.len - dot - 1)));
            return meets(nd, neg, lo, lo + width, true);
        },
        else => return true,
    }
}

/// Whether the reply may end here: the root's value is complete (a number, a literal or a regex that could go on
/// may also stop).
pub fn canEnd(p: *const Program, s: *const State) bool {
    const root = s.frames[0];
    if (s.depth == 1) {
        if (root.st == .r_done) return true;
        if (root.st == .r_value) if (p.nodes[p.root] == .regex) {
            const d = &p.dfas[p.nodes[p.root].regex];
            return d.accept[d.start];
        };
        return false;
    }
    if (s.depth != 2 or root.st != .r_done) return false;
    const f = s.frames[1];
    return switch (f.st) {
        .n_zero, .n_int, .n_frac, .n_exp => numberEnds(p.nodes[f.node].number, s, f.st),
        .l_trie => p.tries.value(f.x) != null,
        .x_dfa => p.dfas[p.nodes[f.node].regex].accept[f.x],
        else => false,
    };
}

/// Steps every byte of `bytes`; false at the first refused one.
pub fn feed(p: *const Program, s: *State, bytes: []const u8) bool {
    for (bytes) |b| if (!step(p, s, b)) return false;
    return true;
}
