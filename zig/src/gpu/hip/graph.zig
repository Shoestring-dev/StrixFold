//! HIP graphs: captured from a stream or built node by node, instantiated once, replayed and updated in place (the CUDA
//! runtime's graph surface over hipStream*Capture / hipGraph*, after ashhart/TensorFold d46d17a zig/src/hip/graph.zig).
//! An Exec holds the device addresses it was captured with: every buffer it reads or writes outlives it.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");

pub const Node = abi.GraphNode;

/// Starts capturing `stream`; every launch on it is recorded until `endCapture`, nothing runs.
pub fn beginCapture(stream: Stream, mode: abi.CaptureMode) Error!void {
    try stream.d.check(stream.d.api.hipStreamBeginCapture(stream.handle, mode), "hipStreamBeginCapture");
}

/// Ends the capture; an invalidated capture returns an error and no graph.
pub fn endCapture(stream: Stream) Error!Graph {
    var g: abi.Graph = null;
    try stream.d.check(stream.d.api.hipStreamEndCapture(stream.handle, &g), "hipStreamEndCapture");
    if (g == null) return error.Invalid;
    return .{ .d = stream.d, .handle = g };
}

pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.api.hipStreamIsCapturing(stream.handle, &s), "hipStreamIsCapturing");
    return s;
}

fn nodeParams(f: Function, cfg: launch.Config, args: *launch.Args) Error!abi.KernelNodeParams {
    try cfg.validate();
    if (!cfg.plain()) return error.Unsupported;
    return .{
        .block = .{ .x = cfg.block.x, .y = cfg.block.y, .z = cfg.block.z },
        .extra = null,
        .func = f.handle,
        .grid = .{ .x = cfg.grid.x, .y = cfg.grid.y, .z = cfg.grid.z },
        .params = args.pointers(),
        .shared_bytes = cfg.shared,
    };
}

pub const Graph = struct {
    d: *const Driver,
    handle: abi.Graph,

    pub fn init(d: *const Driver) Error!Graph {
        var g: abi.Graph = null;
        try d.check(d.api.hipGraphCreate(&g, 0), "hipGraphCreate");
        if (g == null) return error.Invalid;
        return .{ .d = d, .handle = g };
    }

    pub fn deinit(self: *Graph) void {
        _ = self.d.api.hipGraphDestroy(self.handle);
        self.* = undefined;
    }

    /// A kernel node after `deps`; HIP copies the argument values, so `args` may change afterwards.
    pub fn addKernel(self: Graph, deps: []const Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!Node {
        const p = try nodeParams(f, cfg, args);
        var n: Node = null;
        try self.d.check(self.d.api.hipGraphAddKernelNode(&n, self.handle, if (deps.len > 0) deps.ptr else null, deps.len, &p), "hipGraphAddKernelNode");
        return n;
    }

    /// New arguments or geometry for a node of this (template) graph, before it is instantiated again or updated from.
    pub fn setKernel(self: Graph, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.api.hipGraphKernelNodeSetParams(node, &p), "hipGraphKernelNodeSetParams");
    }

    pub fn depend(self: Graph, from: Node, to: Node) Error!void {
        const a = [1]Node{from};
        const b = [1]Node{to};
        try self.d.check(self.d.api.hipGraphAddDependencies(self.handle, &a, &b, 1), "hipGraphAddDependencies");
    }

    pub fn nodeCount(self: Graph) Error!usize {
        var n: usize = 0;
        try self.d.check(self.d.api.hipGraphGetNodes(self.handle, null, &n), "hipGraphGetNodes");
        return n;
    }

    /// Fills `out` with the graph's nodes (capture order for captured graphs) and returns them.
    pub fn nodes(self: Graph, out: []Node) Error![]Node {
        var n: usize = out.len;
        try self.d.check(self.d.api.hipGraphGetNodes(self.handle, out.ptr, &n), "hipGraphGetNodes");
        return out[0..@min(n, out.len)];
    }

    pub fn instantiate(self: Graph) Error!Exec {
        var e: abi.GraphExec = null;
        try self.d.check(self.d.api.hipGraphInstantiateWithFlags(&e, self.handle, 0), "hipGraphInstantiateWithFlags");
        if (e == null) return error.Invalid;
        return .{ .d = self.d, .handle = e };
    }
};

pub const Exec = struct {
    d: *const Driver,
    handle: abi.GraphExec,

    pub fn deinit(self: *Exec) void {
        _ = self.d.api.hipGraphExecDestroy(self.handle);
        self.* = undefined;
    }

    /// Moves the graph's work to the device ahead of the first launch, so that launch pays no setup.
    pub fn upload(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.hipGraphUpload(self.handle, stream.handle), "hipGraphUpload");
    }

    pub fn launchOn(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.hipGraphLaunch(self.handle, stream.handle), "hipGraphLaunch");
    }

    /// Changes one kernel node's arguments or geometry in the executable graph without rebuilding it.
    pub fn setKernel(self: Exec, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.api.hipGraphExecKernelNodeSetParams(self.handle, node, &p), "hipGraphExecKernelNodeSetParams");
    }

    /// Takes every node's parameters from `g`, which must have the same topology; returns HIP's verdict (a refused
    /// update leaves the exec unchanged and is not an error).
    pub fn update(self: Exec, g: Graph) Error!abi.ExecUpdateResult {
        var node: Node = null;
        var result: abi.ExecUpdateResult = .success;
        const res = self.d.api.hipGraphExecUpdate(self.handle, g.handle, &node, &result);
        if (result != .success) return result;
        try self.d.check(res, "hipGraphExecUpdate");
        return .success;
    }
};

test "kernel nodes refuse CUDA-only attributes and bad geometry before HIP" {
    var d: Driver = undefined;
    var args: launch.Args = .{};
    const f: Function = .{ .d = &d, .handle = @ptrFromInt(64) };
    const p = try nodeParams(f, .{ .grid = .{ .x = 2 }, .block = .{ .x = 64, .y = 2 }, .shared = 128 }, &args);
    try std.testing.expectEqual(@as(c_uint, 2), p.grid.x);
    try std.testing.expectEqual(@as(c_uint, 2), p.block.y);
    try std.testing.expectEqual(@as(c_uint, 128), p.shared_bytes);
    try std.testing.expectError(error.Unsupported, nodeParams(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 }, .pdl = true }, &args));
    try std.testing.expectError(error.Invalid, nodeParams(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 2048 } }, &args));
}

test "a refused exec update reports HIP's verdict, not an error" {
    const Mock = struct {
        fn refuse(_: abi.GraphExec, _: abi.Graph, node: *abi.GraphNode, result: *abi.ExecUpdateResult) callconv(.c) abi.Result {
            node.* = null;
            result.* = .topology_changed;
            return 1;
        }
    };
    var d: Driver = undefined;
    d.api.hipGraphExecUpdate = Mock.refuse;
    const e: Exec = .{ .d = &d, .handle = @ptrFromInt(8) };
    const g: Graph = .{ .d = &d, .handle = @ptrFromInt(16) };
    try std.testing.expectEqual(abi.ExecUpdateResult.topology_changed, try e.update(g));
}
