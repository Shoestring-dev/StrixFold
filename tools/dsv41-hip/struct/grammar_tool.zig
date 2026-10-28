//! The grammar module (zig/src/core/lanes/grammar) on a real tokenizer, host only:
//!   grammar_tool bench MODEL_DIR                      what the vocabulary costs to load and what a constrained draw
//!                                                     costs in the states a reply passes through (JSON on stdout)
//!   grammar_tool samples MODEL_DIR KIND TEXT N SEED   N replies drawn from random logits under the grammar (KIND:
//!                                                     json, json_schema, choice, regex, calls; TEXT its text), one JSON
//!                                                     line each: {"text": ..., "ended": bool, "tokens": n}
//! Built with the module as a dependency (tools/dsv41-hip/struct/run_tool.sh).
const std = @import("std");
const grammar = @import("grammar");
const machine = grammar.machine;
const Constraint = grammar.Constraint;

fn ns(io: std.Io, since: std.Io.Timestamp) u64 {
    return @intCast(since.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
}

fn median(xs: []u64) u64 {
    std.mem.sort(u64, xs, {}, std.sort.asc(u64));
    return xs[xs.len / 2];
}

const Case = struct { name: []const u8, kind: @FieldType(grammar.Structure, "kind"), text: []const u8 = "", prefix: []const u8 };

const tools =
    \\[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}, "required": ["location"]}}}]
;
const person =
    \\{"type": "object", "properties": {"name": {"type": "string", "maxLength": 12}, "age": {"type": "integer", "minimum": 0, "maximum": 130}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age"]}
;
const D = "\u{ff5c}DSML\u{ff5c}";

const cases = [_]Case{
    .{ .name = "json_object start", .kind = .json, .prefix = "" },
    .{ .name = "json_object key", .kind = .json, .prefix = "{\"" },
    .{ .name = "json_object after key", .kind = .json, .prefix = "{\"name\"" },
    .{ .name = "json_object string value", .kind = .json, .prefix = "{\"name\": \"Al" },
    .{ .name = "json_object number", .kind = .json, .prefix = "{\"n\": 12" },
    .{ .name = "json_object after value", .kind = .json, .prefix = "{\"n\": 12, \"m\": true" },
    .{ .name = "schema key", .kind = .json_schema, .text = person, .prefix = "{\"name\": \"Al\", " },
    .{ .name = "schema short string (room 2)", .kind = .json_schema, .text = person, .prefix = "{\"name\": \"0123456789" },
    .{ .name = "schema bounded integer", .kind = .json_schema, .text = person, .prefix = "{\"name\": \"Al\", \"age\": 1" },
    .{ .name = "schema array of strings", .kind = .json_schema, .text = person, .prefix = "{\"name\": \"Al\", \"age\": 1, \"tags\": [\"x\", " },
    .{ .name = "calls start", .kind = .calls, .text = tools, .prefix = "" },
    .{ .name = "calls raw value", .kind = .calls, .text = tools, .prefix = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"true\">Par" },
    .{ .name = "calls json value", .kind = .calls, .text = tools, .prefix = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"true\">Paris</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"days\" string=\"false\">" },
};

fn bench(a: std.mem.Allocator, io: std.Io, dir: []const u8, out: *std.ArrayList(u8)) !void {
    const t0 = std.Io.Clock.awake.now(io);
    const v = try grammar.vocab.load(a, io, dir);
    defer v.deinit(a);
    const load_ns = ns(io, t0);
    try out.print(a, "{{\"vocab\": {{\"size\": {d}, \"load_ms\": {d:.1}, \"trie_nodes\": {d}, \"json_rest_nodes\": {d}, \"raw_rest_nodes\": {d}, \"json_long\": {d}, \"json_plain_most\": {d}, \"longest\": {d}}}, \"cases\": [", .{ v.size, @as(f64, @floatFromInt(load_ns)) / 1e6, v.all.nodes.len, v.json_rest.nodes.len, v.raw_rest.nodes.len, v.json_long.len, v.json_plain_most, v.longest });
    var prng: std.Random.DefaultPrng = .init(1);
    const rng = prng.random();
    const row = try a.alloc(f32, v.size);
    defer a.free(row);
    for (cases, 0..) |case, ci| {
        var diag: grammar.Diag = .{};
        const c = Constraint.init(a, v, .{ .kind = case.kind, .text = case.text }, &.{1}, &diag) catch |e| {
            std.debug.print("{s}: {s}\n", .{ case.name, diag.text() });
            return e;
        };
        defer c.deinit();
        if (!machine.feed(c.program, &c.cur, case.prefix)) return error.BadPrefix;
        const start = c.cur;
        var fills: [41]u64 = undefined;
        var allowed: usize = 0;
        for (&fills) |*f| {
            c.cur = start;
            const s = std.Io.Clock.awake.now(io);
            const mask = c.fill() orelse return error.NoMask;
            f.* = ns(io, s);
            allowed = 0;
            for (mask) |w| allowed += @popCount(w);
        }
        // one check per token, over the whole vocabulary
        c.cur = start;
        const s1 = std.Io.Clock.awake.now(io);
        var yes: usize = 0;
        for (0..v.size) |t| yes += @intFromBool(c.allows(@intCast(t)));
        const check_ns = ns(io, s1);
        if (yes != allowed) {
            std.debug.print("{s}: mask {d} tokens, single checks {d}\n", .{ case.name, allowed, yes });
            return error.MaskMismatch;
        }
        // top-1 and top-20 from random logits (the fallback path when the row's best are refused)
        var tops: [21]u64 = undefined;
        var top20: [21]u64 = undefined;
        var ids: [20]u32 = undefined;
        var vals: [20]f32 = undefined;
        for (&tops, &top20) |*x, *y| {
            for (row) |*r| r.* = rng.floatNorm(f32) * 3;
            c.cur = start;
            const s2 = std.Io.Clock.awake.now(io);
            _ = c.top(row, 1, &ids, &vals);
            x.* = ns(io, s2);
            c.cur = start;
            const s3 = std.Io.Clock.awake.now(io);
            _ = c.top(row, 20, &ids, &vals);
            y.* = ns(io, s3);
        }
        const us = struct {
            fn of(x: u64) f64 {
                return @as(f64, @floatFromInt(x)) / 1e3;
            }
        }.of;
        const check_per = @as(f64, @floatFromInt(check_ns)) / @as(f64, @floatFromInt(v.size));
        try out.print(a, "{s}\n {{\"case\": \"{s}\", \"allowed\": {d}, ", .{ if (ci == 0) "" else ",", case.name, allowed });
        try out.print(a, "\"fill_us_median\": {d:.1}, \"fill_us_max\": {d:.1}, \"check_ns_per_token\": {d:.1}, ", .{ us(median(&fills)), us(std.mem.max(u64, &fills)), check_per });
        try out.print(a, "\"top1_us_median\": {d:.1}, \"top20_us_median\": {d:.1}}}", .{ us(median(&tops)), us(median(&top20)) });
    }
    try out.appendSlice(a, "\n]}\n");
}

fn samples(a: std.mem.Allocator, io: std.Io, dir: []const u8, kind_name: []const u8, text: []const u8, n: usize, seed: u64, out: *std.ArrayList(u8)) !void {
    const v = try grammar.vocab.load(a, io, dir);
    defer v.deinit(a);
    const kind = std.meta.stringToEnum(@FieldType(grammar.Structure, "kind"), kind_name) orelse return error.BadKind;
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    const row = try a.alloc(f32, v.size);
    defer a.free(row);
    const eos: u32 = 1;
    // pieces of the markup (DSML tags, JSON punctuation) and short tokens favoured, so that values close and replies end
    const markup = "</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"\" string=\"true\">false\">\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>\"}],:";
    const bias = try a.alloc(f32, v.size);
    defer a.free(bias);
    for (bias, 0..) |*b, t| {
        const x = v.text(@intCast(t));
        b.* = (if (x.len <= 2) @as(f32, 2) else 0) + (if (x.len > 0 and std.mem.indexOf(u8, markup, x) != null) @as(f32, 6) else 0);
    }
    bias[eos] += 6;
    for (0..n) |_| {
        var diag: grammar.Diag = .{};
        const c = Constraint.init(a, v, .{ .kind = kind, .text = text }, &.{eos}, &diag) catch |e| {
            std.debug.print("{s}\n", .{diag.text()});
            return e;
        };
        defer c.deinit();
        var emitted: std.ArrayList(u32) = .empty;
        defer emitted.deinit(a);
        var reply: std.ArrayList(u8) = .empty;
        defer reply.deinit(a);
        var ended = false;
        // random logits, the end and short tokens favoured so that replies finish
        while (emitted.items.len < 800) {
            c.reset(emitted.items);
            for (row, bias) |*r, b| r.* = rng.float(f32) * 6 + b;
            var ids: [8]u32 = undefined;
            var vals: [8]f32 = undefined;
            const k = c.top(row, 8, &ids, &vals);
            const t = ids[rng.uintLessThan(usize, k)];
            if (t == eos) {
                ended = true;
                break;
            }
            try emitted.append(a, t);
            try reply.appendSlice(a, v.text(t));
        }
        const quoted = try std.json.Stringify.valueAlloc(a, reply.items, .{});
        defer a.free(quoted);
        try out.print(a, "{{\"text\": {s}", .{quoted});
        try out.print(a, ", \"ended\": {}, \"tokens\": {d}}}\n", .{ ended, emitted.items.len });
    }
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    if (args.len == 3 and std.mem.eql(u8, args[1], "bench")) {
        try bench(a, io, args[2], &out);
    } else if (args.len == 7 and std.mem.eql(u8, args[1], "samples")) {
        try samples(a, io, args[2], args[3], args[4], try std.fmt.parseInt(usize, args[5], 10), try std.fmt.parseInt(u64, args[6], 10), &out);
    } else {
        std.debug.print("usage: grammar_tool bench MODEL_DIR | samples MODEL_DIR KIND TEXT N SEED\n", .{});
        return error.InvalidArguments;
    }
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.writeAll(out.items);
    try w.interface.flush();
}
