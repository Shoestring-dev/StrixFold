//! The dsv41 family's CPU side (tools/dsv41-hip/ADAPTER.md 9.3): what tf-dsv41-check, tf-dsv41-rank-cache and
//! tf-dsv41-engram-check use, as the module "dsv41" of a build with no GPU module at all (zig build dsv41-cpu). These
//! files import only std, builtin and core.
pub const Config = @import("config.zig").Config;
pub const checkpoint = @import("checkpoint.zig");
pub const rank_cache = @import("rank_cache.zig");
pub const plan = @import("plan.zig");
pub const engram = @import("engram.zig");
pub const engram_io = @import("engram_io.zig");
pub const engram_aio = @import("engram_aio.zig");

test {
    const refAll = @import("std").testing.refAllDecls;
    refAll(@import("config.zig"));
    refAll(checkpoint);
    refAll(rank_cache);
    refAll(plan);
    refAll(engram);
    refAll(engram_io);
    refAll(engram_aio);
}
