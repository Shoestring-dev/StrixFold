//! The grammar module's host tests: the machine on whole texts (accepted, refused, where it may end), the vocabulary
//! trie's masks against one check per token, and replies drawn from random logits that must parse as JSON and match
//! their schema (or read as a DSML calls block).
const std = @import("std");
const program = @import("program.zig");
const machine = @import("machine.zig");
const vocab_mod = @import("vocab.zig");
const constraint = @import("constraint.zig");
const Structure = program.Structure;
const Program = program.Program;
const Constraint = constraint.Constraint;
const Value = std.json.Value;

const gpa = std.testing.allocator;

fn compile(s: Structure) !*Program {
    var diag: program.Diag = .{};
    return program.compile(gpa, s, &diag) catch |e| {
        std.debug.print("compile failed: {s}\n", .{diag.text()});
        return e;
    };
}

fn free(p: *Program) void {
    p.deinit();
    gpa.destroy(p);
}

/// Whether the machine takes all of `text` and may end there.
fn accepts(p: *const Program, text: []const u8) bool {
    var s = machine.State.init(p);
    return machine.feed(p, &s, text) and machine.canEnd(p, &s);
}

/// Whether the machine takes all of `text` (it may need more).
fn takes(p: *const Program, text: []const u8) bool {
    var s = machine.State.init(p);
    return machine.feed(p, &s, text);
}

test "json_object: any object, nothing before but whitespace, nothing after" {
    const p = try compile(.{ .kind = .json });
    defer free(p);
    try std.testing.expect(accepts(p, "{\"a\": [1, 2.5, -3e2, true, null, \"x\\u00e9\\n\", {}], \"b\": {\"c\": []}}"));
    try std.testing.expect(accepts(p, "{}") and !takes(p, "\n {}") and !takes(p, " {}"));
    // whitespace between tokens: one newline a run at most, no carriage return
    try std.testing.expect(accepts(p, "{\n  \"a\": [1,\n    2],\n\t\"b\": {}\n}"));
    try std.testing.expect(!takes(p, "{\n\n") and !takes(p, "{ \n \n") and !takes(p, "{\r\n"));
    try std.testing.expect(!takes(p, "[1]"));
    try std.testing.expect(!takes(p, "{\"a\" 1}"));
    try std.testing.expect(!takes(p, "{} "));
    try std.testing.expect(!takes(p, "{\"a\": 01}"));
    try std.testing.expect(!takes(p, "{\"a\": \"\x01\"}"));
    try std.testing.expect(!takes(p, "{\"a\": \"\xc3\x28\"}"));
    try std.testing.expect(!takes(p, "{\"a\": tru }"));
    try std.testing.expect(takes(p, "{\"a\": \"\\ud83d\\ude00\"}"));
    try std.testing.expect(!takes(p, "{\"a\": \"\\ud83d x"));
    try std.testing.expect(!takes(p, "{\"a\": \"\\ude00"));
    try std.testing.expect(!accepts(p, "{\"a\": 1"));
    // whitespace runs are capped
    const spaces: [machine.ws_cap + 1]u8 = @splat(' ');
    try std.testing.expect(takes(p, "{" ++ spaces[0..machine.ws_cap]));
    try std.testing.expect(!takes(p, "{" ++ spaces));
    // numbers are capped too
    const zeros: [machine.num_cap]u8 = @splat('0');
    try std.testing.expect(!takes(p, "{\"a\": 1" ++ zeros));
}

const person =
    \\{"type": "object", "properties": {
    \\  "name": {"type": "string", "minLength": 1, "maxLength": 8},
    \\  "age": {"type": "integer", "minimum": 0, "maximum": 130},
    \\  "role": {"enum": ["admin", "user"]},
    \\  "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    \\  "email": {"type": ["string", "null"]}
    \\}, "required": ["name", "age"], "additionalProperties": false}
;

test "json_schema: declared keys in order, required ones present, value constraints" {
    const p = try compile(.{ .kind = .json_schema, .text = person });
    defer free(p);
    try std.testing.expect(accepts(p, "{\"name\": \"Ana\", \"age\": 30}"));
    try std.testing.expect(accepts(p, "{\"name\":\"Bo\",\"age\":0,\"role\":\"user\",\"tags\":[\"a\",\"b\"],\"email\":null}"));
    try std.testing.expect(accepts(p, "{\"name\": \"Bo\", \"age\": 7, \"email\": \"b@x\"}"));
    try std.testing.expect(!takes(p, "{\"age\": 3")); // name is required first
    try std.testing.expect(!accepts(p, "{\"name\": \"Bo\"}")); // age missing
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\"}"));
    try std.testing.expect(!takes(p, "{\"name\": \"\"")); // minLength
    try std.testing.expect(!takes(p, "{\"name\": \"123456789")); // maxLength
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": 131"));
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": -"));
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": 1.5"));
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": 1, \"role\": \"root\""));
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": 1, \"tags\": [\"a\", \"b\", \"c\""));
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": 1, \"x\""));
    try std.testing.expect(!takes(p, "{\"name\": \"Bo\", \"age\": 1, \"email\": null, \"role\"")); // order
}

test "number bounds: a prefix is taken only when some number it begins fits" {
    const int = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"integer\", \"minimum\": 10, \"maximum\": 20}" });
    defer free(int);
    try std.testing.expect(takes(int, "1") and !accepts(int, "1"));
    try std.testing.expect(accepts(int, "15") and accepts(int, "20") and accepts(int, "10"));
    try std.testing.expect(!takes(int, "3") and !takes(int, "21") and !takes(int, "-") and !takes(int, "0"));
    const ex = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"number\", \"exclusiveMinimum\": -1, \"exclusiveMaximum\": 1}" });
    defer free(ex);
    try std.testing.expect(accepts(ex, "0.5") and accepts(ex, "-0.99") and accepts(ex, "0"));
    try std.testing.expect(!takes(ex, "1") and !takes(ex, "-1") and !takes(ex, "0.5e1"));
    try std.testing.expect(takes(ex, "-0.") and !accepts(ex, "-0."));
    const neg = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"integer\", \"maximum\": -5}" });
    defer free(neg);
    try std.testing.expect(accepts(neg, "-5") and accepts(neg, "-12345") and !takes(neg, "4") and !accepts(neg, "-4"));
    try std.testing.expect(takes(neg, "-4")); // -40 fits
}

test "unions by first byte, nullable, recursive refs, tuples, formats and patterns" {
    const u = try compile(.{ .kind = .json_schema, .text = "{\"anyOf\": [{\"type\": \"string\"}, {\"type\": \"integer\"}, {\"type\": \"null\"}]}" });
    defer free(u);
    try std.testing.expect(accepts(u, "\"a\"") and accepts(u, "42") and accepts(u, "null") and !takes(u, "4.2") and !takes(u, "true"));
    const n = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"boolean\", \"nullable\": true}" });
    defer free(n);
    try std.testing.expect(accepts(n, "true") and accepts(n, "null") and !takes(n, "1"));
    const list = try compile(.{ .kind = .json_schema, .text =
        \\{"$defs": {"node": {"type": "object", "properties": {"v": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}}, "required": ["v", "next"]}}, "$ref": "#/$defs/node"}
    });
    defer free(list);
    try std.testing.expect(accepts(list, "{\"v\": 1, \"next\": {\"v\": 2, \"next\": null}}"));
    try std.testing.expect(!takes(list, "{\"v\": 1, \"next\": {\"v\": 2}}"));
    const tuple = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"array\", \"prefixItems\": [{\"type\": \"string\"}, {\"type\": \"number\"}], \"items\": false}" });
    defer free(tuple);
    try std.testing.expect(accepts(tuple, "[\"a\", 1]") and accepts(tuple, "[\"a\"]") and !takes(tuple, "[1") and !takes(tuple, "[\"a\", 1, "));
    const date = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"string\", \"format\": \"date\"}" });
    defer free(date);
    try std.testing.expect(accepts(date, "\"2026-10-19\"") and !takes(date, "\"2026-13") and !accepts(date, "\"2026-10\""));
    const pat = try compile(.{ .kind = .json_schema, .text = "{\"type\": \"string\", \"pattern\": \"^[A-Z]{2}[0-9]+$\"}" });
    defer free(pat);
    try std.testing.expect(accepts(pat, "\"AB12\"") and !takes(pat, "\"A1") and !accepts(pat, "\"AB\""));
    const enums = try compile(.{ .kind = .json_schema, .text = "{\"enum\": [1, 12, \"a\", null, {\"k\": [true]}]}" });
    defer free(enums);
    try std.testing.expect(accepts(enums, "1") and accepts(enums, "12") and accepts(enums, "{\"k\":[true]}") and !takes(enums, "13") and !accepts(enums, "\"b\""));
}

test "choice and regex at the root: no whitespace, an end only where a match is complete" {
    const ch = try compile(.{ .kind = .choice, .text = "[\"yes\", \"yesterday\", \"no\"]" });
    defer free(ch);
    try std.testing.expect(accepts(ch, "yes") and accepts(ch, "yesterday") and accepts(ch, "no"));
    try std.testing.expect(takes(ch, "yest") and !accepts(ch, "yest") and !takes(ch, " yes") and !takes(ch, "yess"));
    const re = try compile(.{ .kind = .regex, .text = "[0-9]{3}-[0-9]{4}" });
    defer free(re);
    try std.testing.expect(accepts(re, "555-1234") and !accepts(re, "555-123") and !takes(re, "5555"));
}

const D = "\u{ff5c}DSML\u{ff5c}";
const weather_tools =
    \\[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    \\  "location": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}, "days": {"type": "integer", "minimum": 1, "maximum": 7}},
    \\  "required": ["location"]}}},
    \\ {"type": "function", "function": {"name": "ping"}},
    \\ {"type": "function", "namespace": "ns1", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}}]
;

test "calls: the encoder's DSML block to an offered tool, its parameters by their schema" {
    const p = try compile(.{ .kind = .calls, .text = weather_tools });
    defer free(p);
    const call = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"true\">Paris, \"FR\" </x></" ++ D ++ " parameter>\n<" ++
        D ++ " parameter name=\"unit\" string=\"true\">c</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"days\" string=\"false\">3</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    try std.testing.expect(accepts(p, call));
    // two calls: one with no parameters, one namespaced with a free parameter
    const two = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"ping\">\n<" ++ D ++ " parameter name=\"any\" string=\"false\">[1, {\"a\": null}]</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++
        D ++ " invoke name=\"ns1::lookup\">\n<" ++ D ++ " parameter name=\"q\" string=\"false\">null</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    try std.testing.expect(accepts(p, two));
    const empty = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"ns1::lookup\">\n\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    try std.testing.expect(accepts(p, empty));
    // refused: prose first, an unknown tool, a missing required parameter, a bad enum, a string where JSON is due,
    // the closing tag's tail without its '<', text after the block
    try std.testing.expect(!takes(p, "Sure"));
    try std.testing.expect(!takes(p, "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"rain\""));
    try std.testing.expect(!takes(p, "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n</"));
    try std.testing.expect(!takes(p, "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"true\">x</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"unit\" string=\"true\">k"));
    try std.testing.expect(!takes(p, "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"false\""));
    try std.testing.expect(!takes(p, "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"true\">a/" ++ D ++ " parameter"));
    try std.testing.expect(!takes(p, call ++ " "));
    // parallel_tool_calls false: one invoke, then the block ends
    const one = try std.fmt.allocPrint(gpa, "{{\"tools\": {s}, \"parallel_tool_calls\": false}}", .{weather_tools});
    defer gpa.free(one);
    const single = try compile(.{ .kind = .calls, .text = one });
    defer free(single);
    try std.testing.expect(accepts(single, call) and accepts(single, empty) and !takes(single, two));
}

// -- draws over a small vocabulary -----------------------------------------------------------------------------------

/// Bytes 0-255 as their own tokens, these multi-byte ones, then EOS (special).
const extra = [_][]const u8{ "{\"", "\": ", "\", \"", "true", "false", "null", "  ", "\n  ", "abc", "\u{e9}", "\xc3", "\xa9x", ", ", "}]", "[{", "12", "0.5", "\\n", "\\u00", "<" ++ D, " parameter", "</" ++ D ++ " parameter>\n", "\n\n<" ++ D ++ " calls>\n", "\"}", "name", "\":\"", "-1" };

fn testVocab() !*vocab_mod.Vocab {
    return vocab_mod.forTest(gpa, &extra, 1);
}

/// Draws a reply from random logits with structural bytes favoured; the text, and whether it ended with EOS.
fn draw(c: *Constraint, v: *const vocab_mod.Vocab, rng: std.Random, eos: u32, limit: usize, check_masks: bool, text: *std.ArrayList(u8)) !bool {
    var emitted: std.ArrayList(u32) = .empty;
    defer emitted.deinit(gpa);
    const row = try gpa.alloc(f32, v.size);
    defer gpa.free(row);
    for (0..limit) |_| {
        c.reset(emitted.items);
        for (row, 0..) |*x, i| {
            x.* = rng.floatNorm(f32);
            if (i < 256 and std.mem.indexOfScalar(u8, "\"}],:", @intCast(i)) != null) x.* += 2.5;
        }
        row[eos] += 3;
        if (check_masks) {
            // the walked mask is exactly the tokens a single check allows
            if (c.fill()) |mask| {
                var any = false;
                for (0..v.size) |i| {
                    const t: u32 = @intCast(i);
                    const want = c.allows(t);
                    if (want != Constraint.isSet(mask, t)) {
                        const f = c.cur.frames[c.cur.depth - 1];
                        std.debug.print("token {d} '{f}' allows {} mask {}; top {s} n {d} k {d} x {d} depth {d}; text so far '{f}'\n", .{ t, std.zig.fmtString(v.text(t)), want, !want, @tagName(f.st), f.n, f.k, f.x, c.cur.depth, std.zig.fmtString(text.items) });
                    }
                    try std.testing.expectEqual(want, Constraint.isSet(mask, t));
                    any = any or want;
                }
                try std.testing.expect(any);
            }
        }
        var ids: [4]u32 = undefined;
        var vals: [4]f32 = undefined;
        const n = c.top(row, 4, &ids, &vals);
        try std.testing.expect(n >= 1);
        const t = ids[rng.uintLessThan(usize, n)];
        if (t == eos) return true;
        try emitted.append(gpa, t);
        try text.appendSlice(gpa, v.text(t));
    }
    return false;
}

/// A JSON schema check of `x` for the subset the tests use.
fn valid(schema: Value, x: Value) bool {
    if (schema != .object) return true;
    const o = schema.object;
    if (o.get("$ref") != null) return true; // the recursive case is checked by the machine tests
    if (o.get("anyOf")) |alts| {
        for (alts.array.items) |alt| if (valid(alt, x)) return true;
        return false;
    }
    if (o.get("enum")) |e| {
        for (e.array.items) |k| if (std.meta.activeTag(k) == std.meta.activeTag(x)) {
            if (k == .string and std.mem.eql(u8, k.string, x.string)) return true;
            if (k == .integer and k.integer == x.integer) return true;
        };
        return false;
    }
    if (o.get("type")) |t| {
        const names: []const Value = if (t == .array) t.array.items else &.{t};
        const ok = for (names) |n| {
            const name = n.string;
            const hit = if (std.mem.eql(u8, name, "object")) x == .object else if (std.mem.eql(u8, name, "array")) x == .array else if (std.mem.eql(u8, name, "string")) x == .string else if (std.mem.eql(u8, name, "integer")) x == .integer else if (std.mem.eql(u8, name, "number")) x == .integer or x == .float or x == .number_string else if (std.mem.eql(u8, name, "boolean")) x == .bool else x == .null;
            if (hit) break true;
        } else false;
        if (!ok) return false;
    }
    switch (x) {
        .object => |obj| {
            const props = if (o.get("properties")) |p| p.object else null;
            if (o.get("required")) |r| for (r.array.items) |k| if (obj.get(k.string) == null) return false;
            for (obj.keys(), obj.values()) |k, val| {
                const sub = if (props) |p| p.get(k) else null;
                if (sub == null and props != null) return false;
                if (sub) |s| if (!valid(s, val)) return false;
            }
        },
        .array => |arr| {
            if (o.get("maxItems")) |m| if (arr.items.len > m.integer) return false;
            if (o.get("minItems")) |m| if (arr.items.len < m.integer) return false;
            if (o.get("items")) |it| for (arr.items) |val| if (!valid(it, val)) return false;
        },
        .string => |s| {
            const n = std.unicode.utf8CountCodepoints(s) catch return false;
            if (o.get("maxLength")) |m| if (n > m.integer) return false;
            if (o.get("minLength")) |m| if (n < m.integer) return false;
        },
        .integer => |i| {
            if (o.get("minimum")) |m| if (i < m.integer) return false;
            if (o.get("maximum")) |m| if (i > m.integer) return false;
        },
        else => {},
    }
    return true;
}

test "random draws: masks equal single checks, ended replies parse and match their schema" {
    const v = try testVocab();
    defer v.deinit(gpa);
    const eos: u32 = v.size - 1;
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const rng = prng.random();
    const schemas = [_]?[]const u8{ null, person, "{\"type\": \"array\", \"items\": {\"type\": \"integer\", \"minimum\": -3, \"maximum\": 250}, \"minItems\": 1, \"maxItems\": 4}", "{\"type\": \"object\", \"properties\": {\"a\": {\"anyOf\": [{\"type\": \"string\", \"maxLength\": 3}, {\"type\": \"number\"}, {\"type\": \"null\"}]}, \"b\": {\"type\": \"array\", \"items\": {\"type\": \"boolean\"}}}, \"required\": [\"b\"]}" };
    for (schemas) |schema| {
        var ended: usize = 0;
        for (0..40) |round| {
            var diag: program.Diag = .{};
            const s: Structure = if (schema) |t| .{ .kind = .json_schema, .text = t } else .{ .kind = .json };
            const c = try Constraint.init(gpa, v, s, &.{eos}, &diag);
            defer c.deinit();
            var text: std.ArrayList(u8) = .empty;
            defer text.deinit(gpa);
            if (!try draw(c, v, rng, eos, 400, round < 8, &text)) continue;
            ended += 1;
            // any-key objects may repeat a key (JSON allows it; Python keeps the last)
            const parsed = std.json.parseFromSlice(Value, gpa, text.items, .{ .duplicate_field_behavior = .use_last }) catch |e| {
                std.debug.print("not JSON ({s}): {s}\n", .{ @errorName(e), text.items });
                return e;
            };
            defer parsed.deinit();
            if (schema) |t| {
                const sv = try std.json.parseFromSlice(Value, gpa, t, .{});
                defer sv.deinit();
                if (!valid(sv.value, parsed.value)) {
                    std.debug.print("does not match its schema: {s}\n", .{text.items});
                    return error.TestUnexpectedResult;
                }
            } else try std.testing.expect(parsed.value == .object);
        }
        try std.testing.expect(ended >= 10);
    }
}

test "random draws of a required call read as one DSML block" {
    const v = try testVocab();
    defer v.deinit(gpa);
    const eos: u32 = v.size - 1;
    var prng: std.Random.DefaultPrng = .init(7);
    const rng = prng.random();
    var ended: usize = 0;
    for (0..60) |round| {
        var diag: program.Diag = .{};
        const c = try Constraint.init(gpa, v, .{ .kind = .calls, .text = weather_tools }, &.{eos}, &diag);
        defer c.deinit();
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        if (!try draw(c, v, rng, eos, 600, round < 6, &text)) continue;
        ended += 1;
        const t = text.items;
        try std.testing.expect(std.mem.startsWith(u8, t, "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\""));
        try std.testing.expect(std.mem.endsWith(u8, t, "\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>"));
        // every parameter value ends at its own closing tag: no closing-tag tail inside a value
        var it = std.mem.splitSequence(u8, t, "/" ++ D ++ " parameter");
        _ = it.next();
        while (it.next()) |rest| try std.testing.expect(std.mem.startsWith(u8, rest, ">\n"));
    }
    try std.testing.expect(ended >= 10);
}

test "thinking first: free until the after token; drafts outside the grammar are refused" {
    const v = try testVocab();
    defer v.deinit(gpa);
    const eos: u32 = v.size - 1;
    var diag: program.Diag = .{};
    const after: u32 = 'X'; // stands in for </think>
    const c = try Constraint.init(gpa, v, .{ .kind = .json, .after = after }, &.{eos}, &diag);
    defer c.deinit();
    c.reset(&.{ 'h', 'm' });
    try std.testing.expect(c.free() and c.allows('?') and c.fill() == null);
    try std.testing.expect(c.feed(after));
    try std.testing.expect(!c.free() and !c.allows('?') and c.allows('{') and !c.allows(' '));
    try std.testing.expect(!c.allows(eos));
    try std.testing.expect(c.feed('{') and c.feed('}'));
    try std.testing.expect(c.allows(eos) and !c.allows(' '));
    // a window's draft outside the grammar: refused, and the rows after it are free
    c.reset(&.{ 'h', 'm', after });
    try std.testing.expect(!c.feed('a'));
    try std.testing.expect(c.free());
    // the committed reply moves the grammar on; reset reads only the new tokens
    c.reset(&.{ 'h', 'm', after, '{' });
    try std.testing.expect(c.allows('}') and !c.allows('{'));
}
