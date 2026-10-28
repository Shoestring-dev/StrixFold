//! A DeepSeek-V4.1 reply read as the official encoder reads it (deepseek-ai/DeepSeek-V4.1-Flash-1017 @1bae0e10
//! encoding/encoding.py parse_message_from_completion_text and parse_tool_calls), for the DeepSeek-V4.1 dialect
//! (messages.dialectOf): the integration owner's decision U40, the reply-side twin of U38/U39. The reasoning runs to
//! </think> (thinking mode), the content to the calls block's "\n\n<｜DSML｜ calls" (only that separator left out, nothing
//! trimmed), and every invoke of the block is a call whatever the request offered, its name as written (namespace::name
//! kept) and its arguments the encoder's JSON text (decode_dsml_to_arguments: {"key": value, ...}, a string="true"
//! value json.dumps'd with ensure_ascii=False, a string="false" one as written). Where the encoder raises (a block it
//! cannot read, text after it) the reply keeps that text as content and makes no call; a reply still thinking at its
//! end is all reasoning. Whole and streamed replies come from the same reading: `shown` is what a streamed reply may
//! show so far, always a prefix of what `parse` gives once it ends. Other dialects keep the served lane's reading
//! (reply_text, tool_parse, tool_stream; reply_parity.zig's record).
const std = @import("std");
const json = @import("json.zig");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const think_end = "</think>";
/// encoding.py's tool_calls_start_token: "\n\n<" + dsml_token + tool_calls_block_name
pub const calls_start = "\n\n<\u{ff5c}DSML\u{ff5c} calls";
const calls_end = "</\u{ff5c}DSML\u{ff5c} calls>";
const invoke_start = "<\u{ff5c}DSML\u{ff5c} invoke";
const invoke_end = "</\u{ff5c}DSML\u{ff5c} invoke";
const param_start = "<\u{ff5c}DSML\u{ff5c} parameter";
const param_end = "/\u{ff5c}DSML\u{ff5c} parameter";

pub const Call = struct { name: []const u8, arguments: []const u8 };
pub const Parsed = struct { reasoning: []const u8, content: []const u8, calls: []const Call };

/// _read_until_stop: the text from `at` to the earliest of `stops` (its index), and where reading goes on.
const Read = struct { content: []const u8, next: usize, stop: ?usize };

fn readUntil(text: []const u8, at: usize, stops: []const []const u8) Read {
    var best: ?usize = null;
    var which: ?usize = null;
    for (stops, 0..) |s, i| if (std.mem.indexOfPos(u8, text, at, s)) |p| if (best == null or p < best.?) {
        best = p;
        which = i;
    };
    const p = best orelse return .{ .content = text[at..], .next = text.len, .stop = null };
    return .{ .content = text[at..p], .next = p + stops[which.?].len, .stop = which };
}

/// The reply's reasoning, content and calls (`thinking`: the prompt opened the think block).
pub fn parse(a: Allocator, text: []const u8, thinking: bool) Allocator.Error!Parsed {
    var at: usize = 0;
    var reasoning: []const u8 = "";
    if (thinking) {
        const r = readUntil(text, 0, &.{ think_end, calls_start });
        reasoning = r.content;
        if (r.stop == null) return .{ .reasoning = reasoning, .content = "", .calls = &.{} }; // still thinking
        if (r.stop.? == 1) return block(a, text, reasoning, "", r.next, r.next - calls_start.len);
        at = r.next;
    }
    const c = readUntil(text, at, &.{calls_start});
    if (c.stop == null) return .{ .reasoning = reasoning, .content = c.content, .calls = &.{} };
    return block(a, text, reasoning, c.content, c.next, at);
}

/// The calls block from `at` (past calls_start); a block the encoder cannot read leaves text[raw..] as the content.
fn block(a: Allocator, text: []const u8, reasoning: []const u8, content: []const u8, at: usize, raw: usize) Allocator.Error!Parsed {
    const calls = parseCalls(a, text, at) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Format => return .{ .reasoning = reasoning, .content = text[raw..], .calls = &.{} },
    };
    return .{ .reasoning = reasoning, .content = content, .calls = calls };
}

/// parse_tool_calls from `at`, then nothing but the end (the caller's tool_ends_text check).
fn parseCalls(a: Allocator, text: []const u8, start: usize) (Allocator.Error || error{Format})![]const Call {
    var calls: std.ArrayList(Call) = .empty;
    var at = start;
    while (at < text.len) {
        const r = readUntil(text, at, &.{ invoke_start, calls_end });
        if (!std.mem.eql(u8, r.content, ">\n")) return error.Format;
        at = r.next;
        if (r.stop == null) return error.Format;
        if (r.stop.? == 1) break;
        const n = readUntil(text, at, &.{ param_start, invoke_end });
        at = n.next;
        const name = toolName(n.content) orelse return error.Format;
        var keys: std.ArrayList([]const u8) = .empty;
        var args: std.ArrayList(u8) = .empty;
        try args.append(a, '{');
        var stop = n.stop;
        while (stop != null and stop.? == 0) {
            const pc = readUntil(text, at, &.{param_end});
            at = pc.next;
            const p = param(pc.content) orelse return error.Format;
            for (keys.items) |k| if (std.mem.eql(u8, k, p.name)) return error.Format; // a duplicate parameter
            try keys.append(a, p.name);
            if (keys.items.len > 1) try args.appendSlice(a, ", ");
            try args.appendSlice(a, try json.stringify(a, .{ .string = p.name }, .{ .ascii = false }));
            try args.appendSlice(a, ": ");
            try args.appendSlice(a, if (p.string) try json.stringify(a, .{ .string = p.value }, .{ .ascii = false }) else p.value);
            const sep = readUntil(text, at, &.{ param_start, invoke_end });
            if (!std.mem.eql(u8, sep.content, ">\n")) return error.Format;
            at = sep.next;
            stop = sep.stop;
        }
        if (stop == null) return error.Format;
        try args.append(a, '}');
        try calls.append(a, .{ .name = name, .arguments = args.items });
    }
    if (at != text.len) return error.Format; // text after the block
    return calls.items;
}

/// Python's \s for the bytes a name line may start with.
fn pySpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r') or (c >= 0x1c and c <= 0x1f);
}

/// ^\s*name="(.*?)">\n$ (DOTALL; $ also before a final newline).
fn toolName(s: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < s.len and pySpace(s[i])) i += 1;
    const r = s[i..];
    if (!std.mem.startsWith(u8, r, "name=\"")) return null;
    const body = r[6..];
    if (std.mem.endsWith(u8, body, "\">\n\n")) return body[0 .. body.len - 4];
    if (std.mem.endsWith(u8, body, "\">\n")) return body[0 .. body.len - 3];
    return null;
}

/// ^ name="(.*?)" string="(true|false)">(.*?)<$ (DOTALL; $ also before a final newline).
fn param(s: []const u8) ?struct { name: []const u8, string: bool, value: []const u8 } {
    if (!std.mem.startsWith(u8, s, " name=\"")) return null;
    const r = s[7..];
    const t = std.mem.indexOf(u8, r, "\" string=\"true\">");
    const f = std.mem.indexOf(u8, r, "\" string=\"false\">");
    const is_true = t != null and (f == null or t.? < f.?);
    const at = (if (is_true) t else f) orelse return null;
    const rest = r[at + (if (is_true) "\" string=\"true\">".len else "\" string=\"false\">".len) ..];
    const value = if (std.mem.endsWith(u8, rest, "<")) rest[0 .. rest.len - 1] else if (std.mem.endsWith(u8, rest, "<\n")) rest[0 .. rest.len - 2] else return null;
    return .{ .name = r[0..at], .string = is_true, .value = value };
}

/// What a streamed reply may show of `text` so far: the reasoning and content no later text changes (an end that
/// could still begin </think> or the calls block held back).
pub fn shown(text: []const u8, thinking: bool) struct { reasoning: []const u8, content: []const u8 } {
    var at: usize = 0;
    var reasoning: []const u8 = "";
    if (thinking) {
        const r = readUntil(text, 0, &.{ think_end, calls_start });
        if (r.stop == null) {
            const held = @max(reply_text.partialTag(text, think_end), reply_text.partialTag(text, calls_start));
            return .{ .reasoning = text[0 .. text.len - held], .content = "" };
        }
        reasoning = r.content;
        if (r.stop.? == 1) return .{ .reasoning = reasoning, .content = "" };
        at = r.next;
    }
    const rest = text[at..];
    const c = readUntil(rest, 0, &.{calls_start});
    if (c.stop != null) return .{ .reasoning = reasoning, .content = c.content };
    return .{ .reasoning = reasoning, .content = rest[0 .. rest.len - reply_text.partialTag(rest, calls_start)] };
}

/// The calls as OpenAI tool calls, each with a fresh id.
pub fn openai(a: Allocator, calls: []const Call) Allocator.Error!?[]Value {
    if (calls.len == 0) return null;
    const out = try a.alloc(Value, calls.len);
    for (calls, out) |c, *slot| {
        const function = try json.newObject(a);
        try function.put(a, "name", .{ .string = c.name });
        try function.put(a, "arguments", .{ .string = c.arguments });
        const call = try json.newObject(a);
        try call.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
        try call.put(a, "type", .{ .string = "function" });
        try call.put(a, "function", .{ .object = function });
        slot.* = .{ .object = call };
    }
    return out;
}

// -- the encoder's own renderings (encoding.py render_message of an assistant message after a user one) ------------

const D = "\u{ff5c}DSML\u{ff5c}";
const weather = "Checking.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"location\" string=\"true\">Paris</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";

test "a reply with reasoning, content and a call reads as the encoder reads it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try parse(a, "Let me check the weather.</think>" ++ weather, true);
    try std.testing.expectEqualStrings("Let me check the weather.", p.reasoning);
    try std.testing.expectEqualStrings("Checking.", p.content);
    try std.testing.expectEqual(@as(usize, 1), p.calls.len);
    try std.testing.expectEqualStrings("get_weather", p.calls[0].name);
    try std.testing.expectEqualStrings("{\"location\": \"Paris\"}", p.calls[0].arguments);
    // chat mode: no reasoning, the same content and call
    const c = try parse(a, weather, false);
    try std.testing.expectEqualStrings("", c.reasoning);
    try std.testing.expectEqualStrings("Checking.", c.content);
    try std.testing.expectEqualStrings("{\"location\": \"Paris\"}", c.calls[0].arguments);
}

test "nothing trimmed, every value kind, empty arguments, namespaced names, DSML-like values" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try parse(a, "\nlead </think>\n  answer\n\n", true);
    try std.testing.expectEqualStrings("\nlead ", p.reasoning);
    try std.testing.expectEqualStrings("\n  answer\n\n", p.content);
    // two calls: every JSON kind written raw (string="false"), a string with quotes and newlines escaped as
    // json.dumps(ensure_ascii=False) escapes it; then an empty invoke (the encoder's "\n\n" body) of a namespaced tool
    const two = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"search\">\n<" ++ D ++ " parameter name=\"query\" string=\"true\">say \"hi\"\n\u{e9} </" ++ D ++ " invoke></" ++ D ++ " parameter>\n<" ++
        D ++ " parameter name=\"n\" string=\"false\">1e22</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"tags\" string=\"false\">[\"a\", null]</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++
        D ++ " invoke name=\"ns1::lookup\">\n\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const q = try parse(a, two, false);
    try std.testing.expectEqualStrings("", q.content);
    try std.testing.expectEqual(@as(usize, 2), q.calls.len);
    try std.testing.expectEqualStrings("{\"query\": \"say \\\"hi\\\"\\n\u{e9} </" ++ D ++ " invoke>\", \"n\": 1e22, \"tags\": [\"a\", null]}", q.calls[0].arguments);
    try std.testing.expectEqualStrings("ns1::lookup", q.calls[1].name);
    try std.testing.expectEqualStrings("{}", q.calls[1].arguments);
}

test "what the encoder rejects stays text; a reply still thinking is all reasoning" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // a block cut off by the length: the content from the separator on, no call
    const cut = weather[0 .. weather.len - 20];
    const p = try parse(a, cut, false);
    try std.testing.expectEqualStrings(cut, p.content);
    try std.testing.expectEqual(@as(usize, 0), p.calls.len);
    // text after the block
    const after = try parse(a, weather ++ " more", false);
    try std.testing.expectEqualStrings(weather ++ " more", after.content);
    // a duplicate parameter
    const dup = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"f\">\n<" ++ D ++ " parameter name=\"x\" string=\"true\">1</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"x\" string=\"true\">2</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    try std.testing.expectEqual(@as(usize, 0), (try parse(a, dup, false)).calls.len);
    const thinking = try parse(a, "still going", true);
    try std.testing.expectEqualStrings("still going", thinking.reasoning);
    try std.testing.expectEqualStrings("", thinking.content);
}

test "a streamed reply shows a prefix of the final reading at every cut" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const full = "think\n\n</think>Checking.\n\n" ++ weather[11..];
    const end = try parse(a, full, true);
    var cut: usize = 0;
    while (cut <= full.len) : (cut += 1) {
        const s = shown(full[0..cut], true);
        try std.testing.expect(std.mem.startsWith(u8, end.reasoning, s.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, end.content, s.content));
    }
    // the separator is held back while it could begin the calls block, and shown when it does not
    try std.testing.expectEqualStrings("answer", shown("answer\n\n<", false).content);
    try std.testing.expectEqualStrings("answer\n\nmore", shown("answer\n\nmore", false).content);
    try std.testing.expectEqualStrings("almost", shown("almost</th", true).reasoning);
}
