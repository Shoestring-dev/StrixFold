//! One cost profile for lane admission and DSpark draft-depth selection.
//! Spark values are a reference default, never an AMD measurement.
//! From the strix component kit (port/strix-halo def87879, its commit 9c7ab8b), unchanged but for this line.
const std = @import("std");

pub const spark_round_ms = [_]f64{ 30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9, 75.5, 78.2, 81.3, 84.4, 87.5, 90.6, 93.7, 96.8, 99.9, 103.0, 106.1, 109.2 };
pub const spark_draft_ms: f64 = 3.5;
pub const max_rows = 48;
pub const Error = error{ BadRoundMs, BadDraftMs, BadDraftSelection };

pub const Profile = struct {
    round_ms: []const f64,
    draft_ms: f64,

    pub fn validate(p: Profile) Error!void {
        if (p.round_ms.len == 0 or p.round_ms.len > max_rows) return error.BadRoundMs;
        for (p.round_ms) |x| if (!std.math.isFinite(x) or x <= 0) return error.BadRoundMs;
        if (!std.math.isFinite(p.draft_ms) or p.draft_ms < 0) return error.BadDraftMs;
    }

    /// Rows past the measured range use the last entry, as the reference does.
    /// Qualification must cover the configured range before tuning is enabled.
    pub fn forwardMs(p: Profile, rows: usize) f64 {
        std.debug.assert(rows > 0 and p.round_ms.len > 0);
        return p.round_ms[@min(rows, p.round_ms.len) - 1];
    }

    /// First maximum wins. Float64 prefix survivals match the reference formula.
    pub fn choose(p: Profile, conf: []const f32, kmax: usize, live: usize, other_rows: usize) Error!usize {
        try p.validate();
        if (live == 0 or kmax > conf.len or other_rows > std.math.maxInt(usize) - kmax - 1) return error.BadDraftSelection;
        for (conf[0..kmax]) |x| if (!std.math.isFinite(x)) return error.BadDraftSelection;
        var best: usize = 0;
        var best_v: f64 = -1;
        var survival: f64 = 1;
        var expected: f64 = 1;
        for (0..kmax + 1) |k| {
            if (k > 0) {
                survival *= 1.0 / (1.0 + exp(-@as(f64, conf[k - 1])));
                expected += survival;
            }
            const value = expected * @as(f64, @floatFromInt(live)) / (p.forwardMs(other_rows + k + 1) + p.draft_ms);
            if (value > best_v) {
                best = k;
                best_v = value;
            }
        }
        return best;
    }
};

extern "c" fn exp(x: f64) f64;
pub const spark: Profile = .{ .round_ms = &spark_round_ms, .draft_ms = spark_draft_ms };

pub fn parseRounds(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    var out: std.ArrayList(f64) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |item| {
        if (out.items.len == max_rows) return error.BadRoundMs;
        const x = std.fmt.parseFloat(f64, std.mem.trim(u8, item, " \t\r\n")) catch return error.BadRoundMs;
        if (!std.math.isFinite(x) or x <= 0) return error.BadRoundMs;
        try out.append(a, x);
    }
    if (out.items.len == 0) return error.BadRoundMs;
    return out.toOwnedSlice(a);
}

pub fn parseDraft(text: []const u8) Error!f64 {
    const x = std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\r\n")) catch return error.BadDraftMs;
    if (!std.math.isFinite(x) or x < 0) return error.BadDraftMs;
    return x;
}

test "reject nonphysical costs and malformed profiles without leaking" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "", ",", "1,", "1,,2", "nan", "inf", "0", "-1", "2,-3", "1e999" }) |bad|
        try std.testing.expectError(error.BadRoundMs, parseRounds(a, bad));
    for ([_][]const u8{ "", "nan", "inf", "-1", "1e999" }) |bad|
        try std.testing.expectError(error.BadDraftMs, parseDraft(bad));
    const values = try parseRounds(a, " 2.5, 3.0\t,4 ");
    defer a.free(values);
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 3.0, 4.0 }, values);
    try std.testing.expectEqual(@as(f64, 0), try parseDraft("0"));
    try (Profile{ .round_ms = values, .draft_ms = 0 }).validate();
    try std.testing.expectError(error.BadRoundMs, (Profile{ .round_ms = &.{}, .draft_ms = 1 }).validate());
}

test "the profile changes the choice and ties favor the shorter draft" {
    const fast: Profile = .{ .round_ms = &.{ 1, 1, 1 }, .draft_ms = 0 };
    const slow: Profile = .{ .round_ms = &.{ 1, 20, 40 }, .draft_ms = 0 };
    try std.testing.expectEqual(@as(usize, 2), try fast.choose(&.{ 2, 2 }, 2, 1, 0));
    try std.testing.expectEqual(@as(usize, 0), try slow.choose(&.{ 2, 2 }, 2, 1, 0));
    // confidence 0 => expected tokens are 1 and 1.5; equal throughputs.
    const tie: Profile = .{ .round_ms = &.{ 1, 1.5 }, .draft_ms = 0 };
    try std.testing.expectEqual(@as(usize, 0), try tie.choose(&.{0}, 1, 1, 0));
    const expensive_draft: Profile = .{ .round_ms = &.{ 1, 1.5 }, .draft_ms = 100 };
    try std.testing.expectEqual(@as(usize, 1), try expensive_draft.choose(&.{0}, 1, 1, 0));
}

test "invalid selection input fails before slicing or indexing" {
    try std.testing.expectError(error.BadDraftSelection, spark.choose(&.{}, 1, 1, 0));
    try std.testing.expectError(error.BadDraftSelection, spark.choose(&.{0}, 1, 0, 0));
    try std.testing.expectError(error.BadDraftSelection, spark.choose(&.{std.math.nan(f32)}, 1, 1, 0));
    try std.testing.expectError(error.BadDraftSelection, spark.choose(&.{0}, 1, 1, std.math.maxInt(usize)));
    try std.testing.expectEqual(@as(usize, 0), try spark.choose(&.{}, 0, 1, 0));
}
