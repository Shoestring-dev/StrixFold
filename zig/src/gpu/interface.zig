//! The backend-neutral GPU surface the dsv41 family uses (tools/dsv41-hip/ADAPTER.md section 3.2), checked at compile
//! time against each backend facade: every declaration, field and method the family may call, with its parameter
//! count. A facade that drifts fails its own test build, not the family's.

const std = @import("std");

const Decl = struct { name: []const u8, kind: enum { type, func, value } };

/// Top-level declarations of a facade.
const top = [_]Decl{
    .{ .name = "backend", .kind = .value },             .{ .name = "features", .kind = .value },
    .{ .name = "abi", .kind = .type },                  .{ .name = "Driver", .kind = .type },
    .{ .name = "Error", .kind = .type },                .{ .name = "Context", .kind = .type },
    .{ .name = "DeviceBuffer", .kind = .type },         .{ .name = "HostBuffer", .kind = .type },
    .{ .name = "Stream", .kind = .type },               .{ .name = "Event", .kind = .type },
    .{ .name = "Module", .kind = .type },               .{ .name = "Function", .kind = .type },
    .{ .name = "Args", .kind = .type },                 .{ .name = "Config", .kind = .type },
    .{ .name = "Dim3", .kind = .type },                 .{ .name = "launch", .kind = .type },
    .{ .name = "graph", .kind = .type },                .{ .name = "Usage", .kind = .type },
    .{ .name = "Staging", .kind = .type },              .{ .name = "usage", .kind = .func },
    .{ .name = "copyH2D", .kind = .func },              .{ .name = "copyD2H", .kind = .func },
    .{ .name = "copyD2DAsync", .kind = .func },         .{ .name = "fill8Async", .kind = .func },
    .{ .name = "fill32Async", .kind = .func },          .{ .name = "uploadPinned", .kind = .func },
    .{ .name = "streamCapturing", .kind = .func },      .{ .name = "synchronizeFor", .kind = .func },
    .{ .name = "health", .kind = .func },
};

/// Methods by type, with the parameter count the family's calls use (self included).
const Method = struct { owner: []const u8, name: []const u8, params: usize };
const methods = [_]Method{
    .{ .owner = "Driver", .name = "open", .params = 0 },             .{ .owner = "Driver", .name = "close", .params = 1 },
    .{ .owner = "Driver", .name = "check", .params = 3 },            .{ .owner = "Driver", .name = "deviceCount", .params = 1 },
    .{ .owner = "Context", .name = "init", .params = 2 },            .{ .owner = "Context", .name = "deinit", .params = 1 },
    .{ .owner = "Context", .name = "makeCurrent", .params = 1 },     .{ .owner = "Context", .name = "synchronize", .params = 1 },
    .{ .owner = "Context", .name = "attribute", .params = 2 },       .{ .owner = "Context", .name = "name", .params = 2 },
    .{ .owner = "Context", .name = "memInfo", .params = 1 },         .{ .owner = "DeviceBuffer", .name = "alloc", .params = 2 },
    .{ .owner = "DeviceBuffer", .name = "fromHost", .params = 2 },   .{ .owner = "DeviceBuffer", .name = "free", .params = 1 },
    .{ .owner = "DeviceBuffer", .name = "at", .params = 2 },         .{ .owner = "DeviceBuffer", .name = "upload", .params = 3 },
    .{ .owner = "DeviceBuffer", .name = "download", .params = 3 },   .{ .owner = "DeviceBuffer", .name = "uploadAsync", .params = 4 },
    .{ .owner = "DeviceBuffer", .name = "downloadAsync", .params = 4 }, .{ .owner = "DeviceBuffer", .name = "copyFrom", .params = 5 },
    .{ .owner = "DeviceBuffer", .name = "fill8", .params = 3 },      .{ .owner = "DeviceBuffer", .name = "fill32", .params = 3 },
    .{ .owner = "HostBuffer", .name = "alloc", .params = 2 },        .{ .owner = "HostBuffer", .name = "allocMapped", .params = 2 },
    .{ .owner = "HostBuffer", .name = "device", .params = 1 },       .{ .owner = "HostBuffer", .name = "free", .params = 1 },
    .{ .owner = "HostBuffer", .name = "slice", .params = 2 },        .{ .owner = "Stream", .name = "init", .params = 2 },
    .{ .owner = "Stream", .name = "deinit", .params = 1 },           .{ .owner = "Stream", .name = "synchronize", .params = 1 },
    .{ .owner = "Stream", .name = "done", .params = 1 },             .{ .owner = "Stream", .name = "wait", .params = 2 },
    .{ .owner = "Event", .name = "init", .params = 2 },              .{ .owner = "Event", .name = "deinit", .params = 1 },
    .{ .owner = "Event", .name = "record", .params = 2 },            .{ .owner = "Event", .name = "synchronize", .params = 1 },
    .{ .owner = "Event", .name = "done", .params = 1 },              .{ .owner = "Event", .name = "elapsedMs", .params = 2 },
    .{ .owner = "Module", .name = "load", .params = 2 },             .{ .owner = "Module", .name = "unload", .params = 1 },
    .{ .owner = "Module", .name = "function", .params = 2 },         .{ .owner = "Module", .name = "global", .params = 2 },
    .{ .owner = "Function", .name = "attribute", .params = 2 },      .{ .owner = "Function", .name = "occupancy", .params = 3 },
    .{ .owner = "Function", .name = "allowDynamicShared", .params = 2 }, .{ .owner = "Args", .name = "add", .params = 2 },
    .{ .owner = "Args", .name = "integer", .params = 2 },            .{ .owner = "Args", .name = "pointers", .params = 1 },
    .{ .owner = "Config", .name = "validate", .params = 1 },         .{ .owner = "launch", .name = "launch", .params = 4 },
    .{ .owner = "graph", .name = "beginCapture", .params = 2 },      .{ .owner = "graph", .name = "endCapture", .params = 1 },
    .{ .owner = "graph", .name = "captureStatus", .params = 1 },     .{ .owner = "Staging", .name = "init", .params = 2 },
    .{ .owner = "Staging", .name = "deinit", .params = 1 },          .{ .owner = "Staging", .name = "upload", .params = 4 },
};

/// Fields the family reads directly.
const fields = [_][2][]const u8{
    .{ "DeviceBuffer", "d" }, .{ "DeviceBuffer", "ptr" }, .{ "DeviceBuffer", "len" }, .{ "HostBuffer", "d" },
    .{ "HostBuffer", "bytes" }, .{ "Stream", "d" },        .{ "Stream", "handle" },   .{ "Event", "handle" },
    .{ "Context", "d" },       .{ "Function", "handle" },  .{ "Function", "d" },      .{ "Config", "grid" },
    .{ "Config", "block" },    .{ "Config", "shared" },    .{ "Config", "pdl" },
};

/// Fails compilation naming the first missing or mismatched item of facade `G`.
pub fn check(comptime G: type) void {
    @setEvalBranchQuota(20000);
    inline for (top) |d| {
        if (!@hasDecl(G, d.name)) @compileError("gpu facade lacks " ++ d.name);
        const v = @field(G, d.name);
        switch (d.kind) {
            .type => if (@TypeOf(v) != type) @compileError("gpu facade: " ++ d.name ++ " is not a type"),
            .func => if (@typeInfo(@TypeOf(v)) != .@"fn") @compileError("gpu facade: " ++ d.name ++ " is not a function"),
            .value => {},
        }
    }
    inline for (methods) |m| {
        const T = @field(G, m.owner);
        if (!@hasDecl(T, m.name)) @compileError("gpu facade: " ++ m.owner ++ " lacks " ++ m.name);
        const n = @typeInfo(@TypeOf(@field(T, m.name))).@"fn".param_types.len;
        if (n != m.params) @compileError(std.fmt.comptimePrint("gpu facade: {s}.{s} takes {d} parameters, the family calls it with {d}", .{ m.owner, m.name, n, m.params }));
    }
    inline for (fields) |f| {
        if (!@hasField(@field(G, f[0]), f[1])) @compileError("gpu facade: " ++ f[0] ++ " lacks field " ++ f[1]);
    }
    if (G.abi.DevicePtr != u64) @compileError("gpu facade: device addresses must be u64");
    if (@TypeOf(@as(G.DeviceBuffer, undefined).ptr) != u64) @compileError("gpu facade: DeviceBuffer.ptr must be u64");
}
