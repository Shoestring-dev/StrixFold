//! Image identity for prefix matching. The shadow tokens never reach the model.
const std = @import("std");
const Image = @import("lanes").stream.Image;
const Sha256 = std.crypto.hash.sha2.Sha256;
const key_words = Sha256.digest_length / @sizeOf(u32);
const marker: u32 = 1 << 31;

/// Use eight vocabulary-external values at the start of each image span.
pub fn make(a: std.mem.Allocator, prompt: []const u32, images: []const Image) !?[]u32 {
    var end: usize = 0;
    for (images) |image| {
        if (image.tokens < key_words or image.at < end or image.at > prompt.len or
            image.tokens > prompt.len - image.at) return null;
        end = @as(usize, image.at) + image.tokens;
    }
    const shadow = try a.dupe(u32, prompt);
    for (images) |image| {
        var hash = Sha256.init(.{});
        hash.update(&.{@intFromBool(image.rows)});
        hash.update(std.mem.asBytes(&image.tokens));
        hash.update(image.bytes);
        const digest = hash.finalResult();
        for (shadow[image.at..][0..key_words], 0..) |*word, i| {
            word.* = marker | std.mem.readInt(u32, digest[i * 4 ..][0..4], .little);
        }
    }
    return shadow;
}

/// Save and restore only between images so a prefix always contains the entire image key.
pub fn validCut(images: []const Image, at: usize) bool {
    for (images) |image| {
        if (image.at >= at) break;
        if (at < @as(usize, image.at) + image.tokens) return false;
    }
    return true;
}

test "same image and position keep the key, different bytes or positions change it" {
    const a = std.testing.allocator;
    const prompt: [20]u32 = @splat(7);
    const first = [_]Image{.{ .at = 1, .tokens = 8, .bytes = "pixels A" }};
    const other = [_]Image{.{ .at = 1, .tokens = 8, .bytes = "pixels B" }};
    const moved = [_]Image{.{ .at = 2, .tokens = 8, .bytes = "pixels A" }};
    const ka = (try make(a, &prompt, &first)).?;
    defer a.free(ka);
    const kb = (try make(a, &prompt, &other)).?;
    defer a.free(kb);
    const km = (try make(a, &prompt, &moved)).?;
    defer a.free(km);
    try std.testing.expect(!std.mem.eql(u32, ka, kb));
    try std.testing.expect(!std.mem.eql(u32, ka, km));
    try std.testing.expectEqualSlices(u32, prompt[9..], ka[9..]);
    try std.testing.expect(validCut(&first, 1));
    try std.testing.expect(!validCut(&first, 8));
    try std.testing.expect(validCut(&first, 9));
}
