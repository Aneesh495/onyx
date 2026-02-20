# SIMT Execution vs CPU SIMD Vectorization in Onyx

## 1. Architectural Divergence: SIMT vs SIMD

Data-parallel acceleration in modern high-performance database kernels generally targets one of two paradigms:

1. **CPU SIMD (Single Instruction, Multiple Data)**:
   Explicit vector registers (e.g. 512-bit AVX-512 or 128-bit ARM Neon). The compiler or engineer explicitly packs `N` scalar elements into a single vector register and issues vector instructions (`_mm512_add_ps`, `vaddq_f32`). Branching across lane elements requires manual bitmask blending and permutes.

2. **GPU SIMT (Single Instruction, Multiple Threads)**:
   A programming model where instructions are executed by warps of 32 threads in lockstep hardware execution. Each thread owns independent scalar registers and program counters logically, while physical instruction dispatchers issue one instruction across all active lanes in the warp.

```
+-------------------------------------------------------------------+
|                        SIMT Warp (32 Lanes)                       |
|  Lane 0   Lane 1   Lane 2   ...  Lane 29  Lane 30  Lane 31        |
|  [R0..Rn] [R0..Rn] [R0..Rn]      [R0..Rn] [R0..Rn] [R0..Rn]       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Warp Instruction Dispatch & Execution               |
|      __ballot_sync() -> 32-bit active bitmask in register         |
|      __shfl_sync()   -> direct inter-register register exchange   |
+-------------------------------------------------------------------+
```

## 2. Warp Divergence and Branch Masking

In standard scalar execution, an `if-else` condition forks instruction flow. In SIMT execution, when threads within the same warp evaluate different boolean branches, the hardware executes both paths serially with inactive threads masked off via hardware execution masks.

In Onyx, query predicate evaluation avoids divergent serialization by leveraging warp voting intrinsics:

```cpp
// All 32 threads in the warp evaluate predicate simultaneously
bool is_match = evaluate_predicate(col_data[row_idx], literal, op);

// Single-cycle ballot creates a 32-bit bitmask of matching lanes
uint32_t ballot = cuda::warp_ballot(is_match ? 1 : 0);
uint32_t warp_matches = bit_utils::popcount(ballot);

// Local compaction rank using lane mask
uint32_t lane_mask = bit_utils::lane_mask_lt(lid);
uint32_t warp_rank = bit_utils::popcount(ballot & lane_mask);
```

By computing each thread's rank purely through bitwise operations on the ballot register, no thread diverges, and contiguous output indices are generated with zero shared memory bank conflicts.

## 3. Register Pressure and Occupancy Tuning

GPU streaming multiprocessors (SMs) contain 64K 32-bit registers shared dynamically across active warps. If a database kernel consumes too many registers per thread, the maximum number of concurrent warps (occupancy) decreases:

$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}$$

Onyx kernels strictly bound register allocation:
- Filter kernel: 24 registers/thread (100% theoretical occupancy on SM 80/90).
- Radix sort scatter: 32 registers/thread.
- Fused pipeline: 40 registers/thread.

## 4. Hardware-Accurate CPU SIMT Emulation

On platforms without physical NVIDIA hardware (such as development on Apple Silicon or testing on continuous integration nodes), Onyx provides an exact, lockstep SIMT runtime (`SimtEngine`). The engine models warp registers, ballot voting, and cross-thread shuffles using pure C++23 bit manipulation, guaranteeing that kernel logic behaves identically across GPU and CPU environments.
