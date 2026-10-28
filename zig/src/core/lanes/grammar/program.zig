//! A structured-output request compiled for the byte machine (machine.zig): JSON schema nodes over literal tries, regex
//! DFAs and DeepSeek-V4.1's DSML calls block. A program owns its arena; nothing in it changes while it serves.
//!
//! The JSON schema subset: type (one or a list), object (properties in their declared order, required, additionalProperties
//! false or absent: declared keys only; with no properties, additionalProperties true or a schema: any keys),
//! min/maxProperties, array (items, prefixItems or an items list, min/maxItems), string (min/maxLength, pattern, the
//! formats date, time, date-time and uuid), integer and number (minimum, maximum and their exclusive forms; a bounded
//! number is written without an exponent), boolean, null, enum, const, anyOf/oneOf whose alternatives start apart (an
//! object and null, a string and a number, ...), allOf of one schema, nullable, and local $ref (#, #/$defs/..,
//! #/definitions/..). Keywords that would constrain the value in ways this machine does not (not, if/then/else,
//! patternProperties, uniqueItems, multipleOf, ...) are refused, never ignored; annotations (title, description,
//! default, examples, unknown formats) are ignored.
const std = @import("std");
const trie = @import("trie.zig");
const regex = @import("regex.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const NodeId = u32;
pub const Diag = regex.Diag;
pub const Error = error{ Unsupported, OutOfMemory };

/// A literal value meaning "the block, the call or the parameter list ends" (DSML tries).
pub const end_mark: i32 = -2;
/// A DSML parameter name of a tool without a parameter schema: the name follows as free text.
pub const free_mark: i32 = -3;

/// response_format and the guided_* fields as the engine API names them (engine_api.Structure), plus `calls`: a
/// DeepSeek-V4.1 DSML calls block to one or more of the offered tools (tool_choice "required" or a named function;
/// `text` is the offered tools as a JSON array, or {"tools": [...], "parallel_tool_calls": false} for one call only).
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar, calls },
    /// The schema (JSON text), regex, choices (a JSON array), EBNF or tools; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
};

pub const Bound = struct { v: f64, open: bool };

pub const Object = struct {
    props: []const NodeId, // each declared property's value, in key order
    keys: []const u32, // keys[pos]: the trie of keys allowed with `pos` properties behind (values: property index)
    closable: []const bool, // closable[pos]: no required property at or after pos
    extra: ?NodeId, // any key, this value (an object without declared properties)
    min: u32 = 0,
    max: u32 = std.math.maxInt(u32),
};

pub const Array = struct { prefix: []const NodeId = &.{}, items: ?NodeId, min: u32 = 0, max: u32 = std.math.maxInt(u32) };
pub const Str = struct { min: u32 = 0, max: u32 = std.math.maxInt(u32), dfa: ?u32 = null };
pub const Num = struct { integer: bool, lo: ?Bound = null, hi: ?Bound = null };

/// A union the first byte decides: table[b] is the alternative a value starting with b takes (0xff: none).
pub const Union = struct { alts: []const NodeId, table: [256]u8 = @splat(0xff) };

/// The DSML block: `head` reads "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"NAME\">\n" (value: the tool), `next`
/// another invoke's opening or the block's end (end_mark); params[i] is tool i's parameter list.
pub const Calls = struct { head: u32, next: u32, params: []const NodeId };

/// One invoke's parameters: keys[0] the first parameter or the end, keys[pos] (pos > 0) "\n" and the next one or the
/// end; per parameter its string="..." flags (0: true, a raw value; 1: false, a JSON value), its JSON value, and the
/// trie of its raw values when they are an enum (each followed by the closing tag). `free`: a tool without a schema,
/// any parameter names.
pub const Params = struct { keys: []const u32, flags: []const u32, values: []const NodeId, raw_enums: []const u32, close: u32, free: bool = false, free_tail: u32 = trie.none };

pub const Node = union(enum) {
    any,
    object: Object,
    array: Array,
    string: Str,
    number: Num,
    lits: u32, // a literal set (JSON text, or raw choices): the trie's root; a value ends where a literal does
    one_of: u32, // index into Program.unions
    regex: u32, // index into Program.dfas
    calls: Calls,
    params: Params,
    raw, // a DSML raw parameter value, through its closing tag
    pending, // a $ref being compiled
};

/// The nodes every program starts with.
pub const any_value: NodeId = 0;
pub const any_object: NodeId = 1;
pub const any_array: NodeId = 2;
pub const any_string: NodeId = 3;
pub const any_number: NodeId = 4;
pub const boolean: NodeId = 5;
pub const null_value: NodeId = 6;
pub const raw_value: NodeId = 7;

pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    nodes: []const Node,
    unions: []const Union,
    tries: trie.Tries,
    dfas: []const regex.Dfa,
    root: NodeId,

    pub fn deinit(p: *Program) void {
        var arena = p.arena;
        arena.deinit();
    }
};

const dsml = "\u{ff5c}DSML\u{ff5c}";
pub const param_close = "</" ++ dsml ++ " parameter>";

const Compiler = struct {
    a: Allocator,
    nodes: std.ArrayList(Node) = .empty,
    unions: std.ArrayList(Union) = .empty,
    flat: trie.Flat = .{},
    dfas: std.ArrayList(regex.Dfa) = .empty,
    doc: Value = .null,
    refs: std.StringHashMapUnmanaged(NodeId) = .empty,
    diag: *Diag,
    depth: u32 = 0,

    fn fail(c: *Compiler, comptime fmt: []const u8, args: anytype) Error {
        c.diag.set(fmt, args);
        return error.Unsupported;
    }

    fn add(c: *Compiler, n: Node) Error!NodeId {
        try c.nodes.append(c.a, n);
        return @intCast(c.nodes.items.len - 1);
    }

    fn lits(c: *Compiler, items: []const []const u8) Error!u32 {
        var b = try trie.Builder.init(c.a);
        for (items, 0..) |it, i| try b.add(c.a, it, @intCast(i));
        return b.finish(c.a, &c.flat);
    }

    fn builtins(c: *Compiler) Error!void {
        _ = try c.add(.any);
        _ = try c.add(.{ .object = .{ .props = &.{}, .keys = &.{}, .closable = &.{}, .extra = any_value } });
        _ = try c.add(.{ .array = .{ .items = any_value } });
        _ = try c.add(.{ .string = .{} });
        _ = try c.add(.{ .number = .{ .integer = false } });
        _ = try c.add(.{ .lits = try c.lits(&.{ "true", "false" }) });
        _ = try c.add(.{ .lits = try c.lits(&.{"null"}) });
        _ = try c.add(.raw);
    }

    fn json(c: *Compiler, v: Value) Error![]const u8 {
        return std.json.Stringify.valueAlloc(c.a, v, .{});
    }

    // -- JSON schema ----------------------------------------------------------------------------------------------

    const refused = [_][]const u8{ "not", "if", "then", "else", "patternProperties", "propertyNames", "dependentRequired", "dependentSchemas", "dependencies", "contains", "minContains", "maxContains", "multipleOf", "unevaluatedProperties", "unevaluatedItems", "$dynamicRef", "$recursiveRef" };

    fn schema(c: *Compiler, v: Value) Error!NodeId {
        c.depth += 1;
        defer c.depth -= 1;
        if (c.depth > 64) return c.fail("the JSON schema nests deeper than 64", .{});
        switch (v) {
            .bool => |b| return if (b) any_value else c.fail("a false schema admits no value", .{}),
            .object => {},
            else => return c.fail("a JSON schema must be an object or a boolean", .{}),
        }
        const o = v.object;
        for (refused) |k| if (o.get(k) != null) return c.fail("JSON schema keyword '{s}' is not supported", .{k});
        if (o.get("uniqueItems")) |u| if (u == .bool and u.bool) return c.fail("JSON schema keyword 'uniqueItems' is not supported", .{});
        if (o.get("$ref")) |r| return c.ref(r);
        if (o.get("allOf")) |all| {
            if (all != .array or all.array.items.len != 1) return c.fail("allOf is supported with one schema only", .{});
            return c.schema(all.array.items[0]);
        }
        var node = if (o.get("anyOf") orelse o.get("oneOf")) |alts| blk: {
            if (alts != .array or alts.array.items.len == 0) return c.fail("anyOf/oneOf must be a non-empty list of schemas", .{});
            var ids: std.ArrayList(NodeId) = .empty;
            for (alts.array.items) |alt| try ids.append(c.a, try c.schema(alt));
            break :blk try c.unionOf(ids.items);
        } else if (o.get("const")) |k| try c.add(.{ .lits = try c.lits(&.{try c.json(k)}) }) else if (o.get("enum")) |e| blk: {
            if (e != .array or e.array.items.len == 0) return c.fail("enum must be a non-empty list", .{});
            var items: std.ArrayList([]const u8) = .empty;
            for (e.array.items) |x| try items.append(c.a, try c.json(x));
            break :blk try c.add(.{ .lits = try c.lits(items.items) });
        } else try c.typed(o);
        if (o.get("nullable")) |n| if (n == .bool and n.bool) {
            node = try c.unionOf(&.{ node, null_value });
        };
        return node;
    }

    fn typeNames(c: *Compiler, o: std.json.ObjectMap) Error![]const []const u8 {
        const t = o.get("type") orelse {
            // no type: what its keywords describe, else any value
            if (o.get("properties") != null or o.get("required") != null or o.get("additionalProperties") != null) return &.{"object"};
            if (o.get("items") != null or o.get("prefixItems") != null) return &.{"array"};
            if (o.get("pattern") != null or o.get("minLength") != null or o.get("maxLength") != null) return &.{"string"};
            return &.{};
        };
        if (t == .string) return c.a.dupe([]const u8, &.{t.string});
        if (t != .array or t.array.items.len == 0) return c.fail("type must be a type name or a non-empty list of them", .{});
        var out: std.ArrayList([]const u8) = .empty;
        for (t.array.items) |x| {
            if (x != .string) return c.fail("type must be a type name or a non-empty list of them", .{});
            try out.append(c.a, x.string);
        }
        return out.items;
    }

    fn typed(c: *Compiler, o: std.json.ObjectMap) Error!NodeId {
        const names = try c.typeNames(o);
        if (names.len == 0) return any_value;
        const has_number = for (names) |n| {
            if (std.mem.eql(u8, n, "number")) break true;
        } else false;
        var ids: std.ArrayList(NodeId) = .empty;
        for (names) |n| {
            if (has_number and std.mem.eql(u8, n, "integer")) continue; // number covers integer
            try ids.append(c.a, try c.ofType(o, n));
        }
        return if (ids.items.len == 1) ids.items[0] else c.unionOf(ids.items);
    }

    fn count(c: *Compiler, o: std.json.ObjectMap, key: []const u8) Error!?u32 {
        const v = o.get(key) orelse return null;
        const n: i64 = switch (v) {
            .integer => |i| i,
            .float => |f| if (f == @floor(f) and f >= 0 and f < 4e9) @intFromFloat(f) else -1,
            else => -1,
        };
        if (n < 0) return c.fail("{s} must be a whole number", .{key});
        return @intCast(@min(n, std.math.maxInt(u32)));
    }

    fn number(c: *Compiler, v: Value, key: []const u8) Error!f64 {
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            .number_string => |s| std.fmt.parseFloat(f64, s) catch return c.fail("{s} must be a number", .{key}),
            else => c.fail("{s} must be a number", .{key}),
        };
    }

    fn ofType(c: *Compiler, o: std.json.ObjectMap, name: []const u8) Error!NodeId {
        if (std.mem.eql(u8, name, "object")) return c.object(o);
        if (std.mem.eql(u8, name, "array")) return c.array(o);
        if (std.mem.eql(u8, name, "string")) return c.string(o);
        if (std.mem.eql(u8, name, "boolean")) return boolean;
        if (std.mem.eql(u8, name, "null")) return null_value;
        const integer = std.mem.eql(u8, name, "integer");
        if (!integer and !std.mem.eql(u8, name, "number")) return c.fail("unknown JSON schema type '{s}'", .{name});
        var lo: ?Bound = null;
        var hi: ?Bound = null;
        if (o.get("minimum")) |v| lo = .{ .v = try c.number(v, "minimum"), .open = false };
        if (o.get("maximum")) |v| hi = .{ .v = try c.number(v, "maximum"), .open = false };
        if (o.get("exclusiveMinimum")) |v| {
            if (v == .bool) {
                if (v.bool) if (lo) |*b| {
                    b.open = true;
                };
            } else {
                const x = try c.number(v, "exclusiveMinimum");
                if (lo == null or x >= lo.?.v) lo = .{ .v = x, .open = true };
            }
        }
        if (o.get("exclusiveMaximum")) |v| {
            if (v == .bool) {
                if (v.bool) if (hi) |*b| {
                    b.open = true;
                };
            } else {
                const x = try c.number(v, "exclusiveMaximum");
                if (hi == null or x <= hi.?.v) hi = .{ .v = x, .open = true };
            }
        }
        if (integer) {
            // whole-number bounds, closed
            if (lo) |b| lo = .{ .v = if (b.open and b.v == @floor(b.v)) b.v + 1 else @ceil(b.v), .open = false };
            if (hi) |b| hi = .{ .v = if (b.open and b.v == @floor(b.v)) b.v - 1 else @floor(b.v), .open = false };
        }
        if (lo != null and hi != null and (lo.?.v > hi.?.v or (lo.?.v == hi.?.v and (lo.?.open or hi.?.open))))
            return c.fail("the {s} bounds admit no value", .{name});
        if (lo) |b| if (@abs(b.v) > 1e15) return c.fail("number bounds beyond 1e15 are not supported", .{});
        if (hi) |b| if (@abs(b.v) > 1e15) return c.fail("number bounds beyond 1e15 are not supported", .{});
        if (!integer and lo == null and hi == null) return any_number;
        return c.add(.{ .number = .{ .integer = integer, .lo = lo, .hi = hi } });
    }

    fn string(c: *Compiler, o: std.json.ObjectMap) Error!NodeId {
        const min = try c.count(o, "minLength") orelse 0;
        const max = try c.count(o, "maxLength") orelse std.math.maxInt(u32);
        if (min > max) return c.fail("minLength is above maxLength", .{});
        var dfa: ?u32 = null;
        var pattern: ?[]const u8 = null;
        var mode: regex.Mode = .search;
        if (o.get("pattern")) |p| {
            if (p != .string) return c.fail("pattern must be a string", .{});
            pattern = p.string;
        } else if (o.get("format")) |f| if (f == .string) {
            // formats with a fixed shape; any other format is an annotation
            const shapes = [_][2][]const u8{
                .{ "date", "\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])" },
                .{ "time", "([01]\\d|2[0-3]):[0-5]\\d:[0-5]\\d(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)?" },
                .{ "date-time", "\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])T([01]\\d|2[0-3]):[0-5]\\d:[0-5]\\d(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)" },
                .{ "uuid", "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}" },
            };
            for (shapes) |s| if (std.mem.eql(u8, f.string, s[0])) {
                pattern = s[1];
                mode = .full;
            };
        };
        if (pattern) |p| {
            const d = regex.compile(c.a, p, mode, c.diag) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Unsupported => {
                    var inner: [240]u8 = undefined;
                    const why = inner[0..c.diag.len];
                    @memcpy(why, c.diag.text());
                    return c.fail("string pattern: {s}", .{why});
                },
            };
            try c.dfas.append(c.a, d);
            dfa = @intCast(c.dfas.items.len - 1);
        }
        if (min == 0 and max == std.math.maxInt(u32) and dfa == null) return any_string;
        return c.add(.{ .string = .{ .min = min, .max = max, .dfa = dfa } });
    }

    fn array(c: *Compiler, o: std.json.ObjectMap) Error!NodeId {
        var prefix: std.ArrayList(NodeId) = .empty;
        var items: ?NodeId = any_value;
        if (o.get("prefixItems")) |p| {
            if (p != .array) return c.fail("prefixItems must be a list of schemas", .{});
            for (p.array.items) |x| try prefix.append(c.a, try c.schema(x));
        }
        if (o.get("items")) |it| {
            if (it == .array) {
                // draft 4's tuple form: the rest is additionalItems
                for (it.array.items) |x| try prefix.append(c.a, try c.schema(x));
                if (o.get("additionalItems")) |extra| items = if (extra == .bool and !extra.bool) null else try c.schema(extra);
            } else items = if (it == .bool and !it.bool) null else try c.schema(it);
        }
        const min = try c.count(o, "minItems") orelse 0;
        var max = try c.count(o, "maxItems") orelse std.math.maxInt(u32);
        if (items == null) max = @min(max, @as(u32, @intCast(prefix.items.len)));
        if (min > max) return c.fail("minItems is above the items the array may hold", .{});
        if (prefix.items.len == 0 and items != null and items.? == any_value and min == 0 and max == std.math.maxInt(u32)) return any_array;
        return c.add(.{ .array = .{ .prefix = prefix.items, .items = items, .min = min, .max = max } });
    }

    fn object(c: *Compiler, o: std.json.ObjectMap) Error!NodeId {
        var names: std.ArrayList([]const u8) = .empty;
        var values: std.ArrayList(NodeId) = .empty;
        if (o.get("properties")) |p| {
            if (p != .object) return c.fail("properties must be an object of schemas", .{});
            for (p.object.keys(), p.object.values()) |k, v| {
                try names.append(c.a, k);
                try values.append(c.a, try c.schema(v));
            }
        }
        var required: std.ArrayList(bool) = .empty;
        try required.appendNTimes(c.a, false, names.items.len);
        if (o.get("required")) |r| {
            if (r != .array) return c.fail("required must be a list of property names", .{});
            for (r.array.items) |x| {
                if (x != .string) return c.fail("required must be a list of property names", .{});
                const at = for (names.items, 0..) |n, i| {
                    if (std.mem.eql(u8, n, x.string)) break i;
                } else blk: {
                    // required without a schema: any value, after the declared ones
                    try names.append(c.a, x.string);
                    try values.append(c.a, any_value);
                    try required.append(c.a, false);
                    break :blk names.items.len - 1;
                };
                required.items[at] = true;
            }
        }
        const min = try c.count(o, "minProperties") orelse 0;
        const max = try c.count(o, "maxProperties") orelse std.math.maxInt(u32);
        if (names.items.len == 0) {
            // no declared properties: any keys unless additionalProperties is false
            var extra: ?NodeId = any_value;
            if (o.get("additionalProperties")) |ap| extra = if (ap == .bool) (if (ap.bool) any_value else null) else try c.schema(ap);
            if (extra == null and min > 0) return c.fail("an object with no properties cannot hold minProperties", .{});
            if (extra != null and extra.? == any_value and min == 0 and max == std.math.maxInt(u32)) return any_object;
            return c.add(.{ .object = .{ .props = &.{}, .keys = &.{}, .closable = &.{}, .extra = extra, .min = min, .max = max } });
        }
        if (names.items.len > 255) return c.fail("objects with more than 255 properties are not supported", .{});
        var nreq: u32 = 0;
        for (required.items) |r| nreq += @intFromBool(r);
        if (min > names.items.len or max < nreq) return c.fail("min/maxProperties cannot be met with these properties", .{});
        const n = names.items.len;
        const keys = try c.a.alloc(u32, n + 1);
        const closable = try c.a.alloc(bool, n + 1);
        for (0..n + 1) |pos| {
            closable[pos] = for (required.items[pos..]) |r| {
                if (r) break false;
            } else true;
            if (pos == n) {
                keys[pos] = trie.none;
                continue;
            }
            var b = try trie.Builder.init(c.a);
            for (pos..n) |i| {
                try b.add(c.a, try c.json(.{ .string = names.items[i] }), @intCast(i));
                if (required.items[i]) break;
            }
            keys[pos] = try b.finish(c.a, &c.flat);
        }
        return c.add(.{ .object = .{ .props = values.items, .keys = keys, .closable = closable, .extra = null, .min = min, .max = max } });
    }

    fn unionOf(c: *Compiler, alts: []const NodeId) Error!NodeId {
        for (alts) |x| if (x == any_value) return any_value;
        if (alts.len == 1) return alts[0];
        try c.unions.append(c.a, .{ .alts = try c.a.dupe(NodeId, alts) });
        return c.add(.{ .one_of = @intCast(c.unions.items.len - 1) });
    }

    /// The JSON pointer `r` in the root document, compiled once (a recursive reference reads its own node).
    fn ref(c: *Compiler, r: Value) Error!NodeId {
        if (r != .string) return c.fail("$ref must be a string", .{});
        const path = r.string;
        if (c.refs.get(path)) |id| return id;
        if (path.len == 0 or path[0] != '#') return c.fail("$ref '{s}' is not local: only #, #/$defs/.. and #/definitions/.. are supported", .{path});
        var at = c.doc;
        var it = std.mem.splitScalar(u8, path[1..], '/');
        if (path.len > 1 and path[1] != '/') return c.fail("$ref '{s}' is not a JSON pointer", .{path});
        if (path.len > 1) _ = it.next();
        while (it.next()) |raw| {
            const part = try std.mem.replaceOwned(u8, c.a, try std.mem.replaceOwned(u8, c.a, raw, "~1", "/"), "~0", "~");
            at = switch (at) {
                .object => |o| o.get(part) orelse return c.fail("$ref '{s}' names nothing in the schema", .{path}),
                .array => |arr| blk: {
                    const i = std.fmt.parseInt(usize, part, 10) catch return c.fail("$ref '{s}' names nothing in the schema", .{path});
                    if (i >= arr.items.len) return c.fail("$ref '{s}' names nothing in the schema", .{path});
                    break :blk arr.items[i];
                },
                else => return c.fail("$ref '{s}' names nothing in the schema", .{path}),
            };
        }
        const id = try c.add(.pending);
        try c.refs.put(c.a, path, id);
        const body = try c.schema(at);
        if (body == id) return c.fail("$ref '{s}' refers only to itself", .{path});
        c.nodes.items[id] = c.nodes.items[body];
        return id;
    }

    // -- unions: each alternative's first bytes ---------------------------------------------------------------------

    fn firstBytes(c: *Compiler, id: NodeId, out: *[256]bool, hops: u32) Error!void {
        if (hops > 64) return c.fail("anyOf/oneOf nests in itself without a value between", .{});
        switch (c.nodes.items[id]) {
            .any => for ("{[\"-0123456789tfn") |b| {
                out[b] = true;
            },
            .object => out['{'] = true,
            .array => out['['] = true,
            .string => out['"'] = true,
            .number => for ("-0123456789") |b| {
                out[b] = true;
            },
            .lits => |root| {
                const t = c.flat.tries();
                const n = t.nodes[root];
                for (t.edges[n.first..][0..n.len]) |e| out[e.byte] = true;
            },
            .one_of => |u| for (c.unions.items[u].alts) |alt| try c.firstBytes(alt, out, hops + 1),
            .pending => return c.fail("anyOf/oneOf nests in itself without a value between", .{}),
            else => return c.fail("this alternative cannot appear in anyOf/oneOf", .{}),
        }
    }

    fn tables(c: *Compiler) Error!void {
        for (c.unions.items) |*u| {
            for (u.alts, 0..) |alt, i| {
                var bytes: [256]bool = @splat(false);
                try c.firstBytes(alt, &bytes, 0);
                for (bytes, 0..) |on, b| if (on) {
                    if (u.table[b] != 0xff and u.table[b] != i) return c.fail("anyOf/oneOf alternatives that start alike (two objects, two arrays, two strings ...) are not supported", .{});
                    u.table[b] = @intCast(i);
                };
            }
        }
    }

    // -- DeepSeek-V4.1 DSML calls (encoding.py's render of an assistant message with tool calls) -------------------

    fn calls(c: *Compiler, offered: Value) Error!NodeId {
        var tools = offered;
        var parallel = true;
        if (offered == .object) {
            tools = offered.object.get("tools") orelse .null;
            if (offered.object.get("parallel_tool_calls")) |p| parallel = !(p == .bool and !p.bool);
        }
        if (tools != .array or tools.array.items.len == 0) return c.fail("a required tool call needs a non-empty list of tools", .{});
        var names: std.ArrayList([]const u8) = .empty;
        var params: std.ArrayList(NodeId) = .empty;
        for (tools.array.items) |tool| {
            if (tool != .object) return c.fail("each tool must be an object", .{});
            const f = if (tool.object.get("function")) |f| f else tool;
            if (f != .object) return c.fail("each tool's function must be an object", .{});
            const name = f.object.get("name") orelse return c.fail("each tool needs a function name", .{});
            if (name != .string or name.string.len == 0) return c.fail("each tool needs a function name", .{});
            // _tool_name_for_encoding: namespace::name when the tool has a namespace
            var ns: ?[]const u8 = null;
            if (tool.object.get("namespace") orelse f.object.get("namespace")) |n| switch (n) {
                .string => |s| ns = s,
                .object => |o| if (o.get("name")) |x| if (x == .string) {
                    ns = x.string;
                },
                else => {},
            };
            const full = if (ns != null and std.mem.indexOf(u8, name.string, "::") == null) try std.fmt.allocPrint(c.a, "{s}::{s}", .{ ns.?, name.string }) else name.string;
            try names.append(c.a, full);
            const saved = c.doc;
            defer c.doc = saved;
            const schema_v = f.object.get("parameters") orelse Value{ .null = {} };
            c.doc = schema_v;
            try params.append(c.a, try c.parameters(schema_v));
        }
        const open = "\n\n<" ++ dsml ++ " calls>\n";
        const invoke = "<" ++ dsml ++ " invoke name=\"";
        var head = try trie.Builder.init(c.a);
        var next = try trie.Builder.init(c.a);
        for (names.items, 0..) |n, i| {
            try head.add(c.a, try std.mem.concat(c.a, u8, &.{ open, invoke, n, "\">\n" }), @intCast(i));
            if (parallel) try next.add(c.a, try std.mem.concat(c.a, u8, &.{ "\n", invoke, n, "\">\n" }), @intCast(i));
        }
        try next.add(c.a, "\n</" ++ dsml ++ " calls>", end_mark);
        return c.add(.{ .calls = .{ .head = try head.finish(c.a, &c.flat), .next = try next.finish(c.a, &c.flat), .params = params.items } });
    }

    /// How a parameter's value is written: string="true" (raw text), string="false" (JSON), or either.
    const Modes = struct { raw: bool, json: bool, raw_enum: ?[]const []const u8 = null };

    fn modes(c: *Compiler, v: Value) Error!Modes {
        if (v != .object) return .{ .raw = true, .json = true };
        const o = v.object;
        if (o.get("$ref")) |r| {
            if (r != .string or r.string.len == 0 or r.string[0] != '#') return .{ .raw = true, .json = true };
            // follow it for the mode only; the JSON node resolves it again
            var at = c.doc;
            var it = std.mem.splitScalar(u8, r.string[1..], '/');
            if (r.string.len > 1) _ = it.next();
            while (it.next()) |part| at = if (at == .object) (at.object.get(part) orelse return .{ .raw = true, .json = true }) else return .{ .raw = true, .json = true };
            return c.modes(at);
        }
        if (o.get("enum") orelse (if (o.get("const")) |k| Value{ .array = blk: {
            var arr: std.json.Array = .init(c.a);
            try arr.append(k);
            break :blk arr;
        } } else null)) |e| {
            if (e != .array) return .{ .raw = false, .json = true };
            var strs: std.ArrayList([]const u8) = .empty;
            for (e.array.items) |x| {
                if (x != .string) return .{ .raw = false, .json = true };
                try strs.append(c.a, x.string);
            }
            return .{ .raw = true, .json = false, .raw_enum = strs.items };
        }
        if (o.get("anyOf") orelse o.get("oneOf")) |alts| {
            var m: Modes = .{ .raw = false, .json = false };
            if (alts == .array) for (alts.array.items) |alt| {
                const x = try c.modes(alt);
                m.raw = m.raw or x.raw;
                m.json = m.json or x.json or x.raw_enum != null;
            };
            return m;
        }
        const names = try c.typeNames(o);
        if (names.len == 0) return .{ .raw = true, .json = true };
        var m: Modes = .{ .raw = false, .json = false };
        for (names) |n| {
            if (std.mem.eql(u8, n, "string")) m.raw = true else m.json = true;
        }
        if (o.get("nullable")) |n| if (n == .bool and n.bool) {
            m.json = true;
        };
        return m;
    }

    fn parameters(c: *Compiler, schema_v: Value) Error!NodeId {
        const p_open = "<" ++ dsml ++ " parameter name=\"";
        const p_mid = "\" string=\"";
        const end = "\n</" ++ dsml ++ " invoke>";
        const props = if (schema_v == .object) schema_v.object.get("properties") else null;
        const close = try c.lits(&.{param_close});
        if (props == null or props.? != .object or props.?.object.count() == 0) {
            // no schema: any parameter names, either kind of value
            var first = try trie.Builder.init(c.a);
            try first.add(c.a, p_open, free_mark);
            try first.add(c.a, end, end_mark);
            var later = try trie.Builder.init(c.a);
            try later.add(c.a, "\n" ++ p_open, free_mark);
            try later.add(c.a, end, end_mark);
            const keys = try c.a.dupe(u32, &.{ try first.finish(c.a, &c.flat), try later.finish(c.a, &c.flat) });
            var tail = try trie.Builder.init(c.a);
            try tail.add(c.a, " string=\"true\">", 0);
            try tail.add(c.a, " string=\"false\">", 1);
            return c.add(.{ .params = .{ .keys = keys, .flags = &.{}, .values = &.{}, .raw_enums = &.{}, .close = close, .free = true, .free_tail = try tail.finish(c.a, &c.flat) } });
        }
        const po = props.?.object;
        const n = po.count();
        if (n > 255) return c.fail("tools with more than 255 parameters are not supported", .{});
        var required = try c.a.alloc(bool, n);
        @memset(required, false);
        if (schema_v.object.get("required")) |r| if (r == .array) for (r.array.items) |x| if (x == .string) {
            if (po.getIndex(x.string)) |i| required[i] = true;
        };
        const flags = try c.a.alloc(u32, n);
        const values = try c.a.alloc(NodeId, n);
        const raw_enums = try c.a.alloc(u32, n);
        for (po.keys(), po.values(), 0..) |_, v, i| {
            const m = try c.modes(v);
            var fb = try trie.Builder.init(c.a);
            if (m.raw or m.raw_enum != null) try fb.add(c.a, "true\">", 0);
            if (m.json) try fb.add(c.a, "false\">", 1);
            if (fb.count == 0) try fb.add(c.a, "false\">", 1);
            flags[i] = try fb.finish(c.a, &c.flat);
            values[i] = if (m.json or fb.count == 0) try c.schema(v) else any_value;
            raw_enums[i] = trie.none;
            if (m.raw_enum) |items| {
                var eb = try trie.Builder.init(c.a);
                for (items) |it| try eb.add(c.a, try std.mem.concat(c.a, u8, &.{ it, param_close }), 0);
                raw_enums[i] = try eb.finish(c.a, &c.flat);
            }
        }
        const keys = try c.a.alloc(u32, n + 1);
        for (0..n + 1) |pos| {
            var b = try trie.Builder.init(c.a);
            const lead: []const u8 = if (pos == 0) "" else "\n";
            for (pos..n) |i| {
                try b.add(c.a, try std.mem.concat(c.a, u8, &.{ lead, p_open, po.keys()[i], p_mid }), @intCast(i));
                if (required[i]) break;
            }
            const can_end = for (required[pos..]) |r| {
                if (r) break false;
            } else true;
            if (can_end) try b.add(c.a, end, end_mark);
            keys[pos] = try b.finish(c.a, &c.flat);
        }
        return c.add(.{ .params = .{ .keys = keys, .flags = flags, .values = values, .raw_enums = raw_enums, .close = close } });
    }
};

/// The program for `s`, or error.Unsupported with `diag` saying why (a malformed or unsupported request).
pub fn compile(gpa: Allocator, s: Structure, diag: *Diag) Error!*Program {
    const p = try gpa.create(Program);
    errdefer gpa.destroy(p);
    p.arena = .init(gpa);
    errdefer p.arena.deinit();
    var c: Compiler = .{ .a = p.arena.allocator(), .diag = diag };
    try c.builtins();
    var root: NodeId = undefined;
    switch (s.kind) {
        .json => root = any_object,
        .json_schema => {
            c.doc = std.json.parseFromSliceLeaky(Value, c.a, s.text, .{}) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return c.fail("the JSON schema is not valid JSON", .{}),
            };
            root = try c.schema(c.doc);
        },
        .choice => {
            const v = std.json.parseFromSliceLeaky(Value, c.a, s.text, .{}) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return c.fail("guided_choice must be a list of strings", .{}),
            };
            if (v != .array or v.array.items.len == 0) return c.fail("guided_choice must be a non-empty list of strings", .{});
            var items: std.ArrayList([]const u8) = .empty;
            for (v.array.items) |x| {
                if (x != .string or x.string.len == 0) return c.fail("guided_choice must be a list of non-empty strings", .{});
                if (!std.unicode.utf8ValidateSlice(x.string)) return c.fail("guided_choice strings must be UTF-8", .{});
                try items.append(c.a, x.string);
            }
            root = try c.add(.{ .lits = try c.lits(items.items) });
        },
        .regex => {
            try c.dfas.append(c.a, try regex.compile(c.a, s.text, .full, diag));
            root = try c.add(.{ .regex = @intCast(c.dfas.items.len - 1) });
        },
        .grammar => return c.fail("guided_grammar (EBNF) is not supported by this engine: use response_format, guided_json, guided_regex or guided_choice", .{}),
        .calls => {
            const v = std.json.parseFromSliceLeaky(Value, c.a, s.text, .{}) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return c.fail("the offered tools are not valid JSON", .{}),
            };
            root = try c.calls(v);
        },
    }
    try c.tables();
    for (c.nodes.items) |n| if (n == .pending) return c.fail("a $ref cycle never reaches a value", .{});
    p.nodes = c.nodes.items;
    p.unions = c.unions.items;
    p.tries = c.flat.tries();
    p.dfas = c.dfas.items;
    p.root = root;
    return p;
}

/// Why `s` cannot be served (a message in `diag`), or null when it compiles.
pub fn check(gpa: Allocator, s: Structure, diag: *Diag) error{OutOfMemory}!?[]const u8 {
    const p = compile(gpa, s, diag) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Unsupported => return diag.text(),
    };
    p.deinit();
    gpa.destroy(p);
    return null;
}

test "schemas outside the subset are refused with their keyword" {
    var diag: Diag = .{};
    const gpa = std.testing.allocator;
    try std.testing.expect((try check(gpa, .{ .kind = .json_schema, .text = "{\"type\": \"object\", \"properties\": {\"a\": {\"not\": {}}}}" }, &diag)) != null);
    try std.testing.expect(std.mem.indexOf(u8, diag.text(), "'not'") != null);
    try std.testing.expect((try check(gpa, .{ .kind = .json_schema, .text = "{\"anyOf\": [{\"type\": \"object\"}, {\"type\": \"object\", \"properties\": {\"x\": {}}}]}" }, &diag)) != null);
    try std.testing.expect((try check(gpa, .{ .kind = .json_schema, .text = "{\"$ref\": \"http://example.com/s.json\"}" }, &diag)) != null);
    try std.testing.expect((try check(gpa, .{ .kind = .grammar, .text = "root ::= \"a\"" }, &diag)) != null);
    try std.testing.expect((try check(gpa, .{ .kind = .json_schema, .text = "{\"type\": \"integer\", \"minimum\": 5, \"maximum\": 4}" }, &diag)) != null);
    try std.testing.expectEqual(@as(?[]const u8, null), try check(gpa, .{ .kind = .json_schema, .text = "{\"$defs\": {\"n\": {\"type\": \"object\", \"properties\": {\"next\": {\"anyOf\": [{\"$ref\": \"#/$defs/n\"}, {\"type\": \"null\"}]}}}}, \"$ref\": \"#/$defs/n\"}" }, &diag));
    try std.testing.expectEqual(@as(?[]const u8, null), try check(gpa, .{ .kind = .json_schema, .text = "{\"type\": \"object\", \"properties\": {\"d\": {\"type\": \"string\", \"format\": \"date\"}, \"e\": {\"type\": \"string\", \"format\": \"email\"}}}" }, &diag));
}
