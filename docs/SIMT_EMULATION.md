# Onyx CPU SIMT Emulation Specification

To guarantee that Onyx can be built, verified, and benchmarked on any developer machine without requiring an NVIDIA GPU or proprietary toolchain, Onyx includes a high-performance, hardware-accurate CPU SIMT emulation layer.

## Architectural Model

The SIMT engine models the execution semantics of an NVIDIA Streaming Multiprocessor (SM):

```text
+-------------------------------------------------------+
|                 SimtEngine (CPU Host)                 |
|                                                       |
|  Thread Block Grid (gridDim.x, gridDim.y, gridDim.z)  |
|  +-------------------------------------------------+  |
|  | Thread Block (blockDim.x, blockDim.y, blockDim.z)|  |
|  |                                                 |  |
|  |  Warp 0: Lanes  0..31  [Lockstep SIMT]          |  |
|  |  Warp 1: Lanes 32..63  [Lockstep SIMT]          |  |
|  |  Shared Memory Scratchpad [Aligned Memory]      |  |
|  +-------------------------------------------------+  |
+-------------------------------------------------------+
```

## Warp Intrinsics Emulation

### 1. Register Context (`SimtWarpContext`)
Each host thread maintains a thread-local warp context with an array of 32 64-bit virtual registers. When a thread writes or exchanges data, it updates its corresponding lane entry.

### 2. Warp Ballot (`warp_ballot`)
Each active lane records its boolean predicate in the warp register. The return value is computed as:
```cpp
uint32_t ballot = 0;
for (uint32_t i = 0; i < 32; ++i) {
    if ((mask & (1U << i)) && g_simt_warp_ctx.registers[i]) {
        ballot |= (1U << i);
    }
}
```

### 3. Warp Shuffle (`warp_shfl_down`, `warp_shfl_up`, `warp_shfl_xor`)
Lanes copy values into the warp register bank, and recipient lanes read directly from the target lane index calculated by the delta offset or bitwise mask.

## Memory Spaces and Emulated Runtime API

- **Device Memory:** Allocated via aligned system memory (`posix_memalign`, 128-byte cache line alignment).
- **Asynchronous Copies:** `cudaMemcpyAsync` executes zero-overhead memory copies with identical synchronization semantics.
- **Streams and Events:** Events record high-resolution timestamps, allowing exact microsecond latency profiling.
