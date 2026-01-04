# Onyx CUDA Kernel Specification

This document details the algorithmic design, register utilization, and synchronization models of the core GPU kernels in Onyx.

## 1. Vectorized Filter with Warp Voting (`filter.cuh`)

The filter kernel evaluates comparison predicates over columnar buffers and outputs dense indices of matching rows.

### Algorithm Flow
1. Each thread in a warp loads an element from the column buffer.
2. The comparison operator (`EQ`, `NEQ`, `LT`, `LTE`, `GT`, `GTE`) is evaluated branchlessly in registers.
3. Threads execute `__ballot_sync(FULL_WARP_MASK, predicate)` to produce a 32-bit bitmask representing matched lanes.
4. Each active thread calculates its relative output offset within the warp:
   `rank = __popc(ballot & lane_mask_lt(lane_id()))`
5. Lane 0 in each warp executes `atomicAdd(&shared_block_match_count, __popc(ballot))` to reserve shared memory space for the warp.
6. Thread 0 in the block commits block matches to global memory via `atomicAdd(global_out_count, block_matches)`.
7. Threads write matched row IDs directly to global output buffers in coalesced memory writes.

## 2. Multi-Pass LSD Radix Sort (`radix_sort.cuh`)

Onyx implements a parallel Least Significant Digit (LSD) Radix Sort with 4-bit and 8-bit digit bins.

### Execution Phases per Digit Pass
- **Phase 1 (Histogram):** Each thread block computes a local digit frequency histogram in shared memory.
- **Phase 2 (Prefix Scan):** A global prefix sum across all block histograms determines exact scatter target offsets for each bin.
- **Phase 3 (Scatter):** Thread blocks stream keys and payloads (row IDs) from input memory and scatter them into double-buffered global memory slots.
- Ping-pong buffer pointers are swapped between passes until all bits are sorted.

## 3. Lock-Free Open Addressing Hash Join (`hash_join.cuh`)

GPU-accelerated relational equi-join between two tables.

### Build Phase
- The smaller table is hashed on its join key using a 64-bit Murmur3 integer mixer.
- Buckets reside in global memory with sentinel value `0xFFFFFFFF80000000ULL`.
- Threads insert keys using 64-bit `atomicCAS`. Collisions are resolved using quadratic probing `(hash + step^2) & mask`.

### Probe Phase
- All 32 threads in each warp load join keys from the probe table in parallel.
- Threads probe the device hash table cooperatively.
- Matched row pairs `(build_row_id, probe_row_id)` are compacted using warp ballot and emitted into result buffers.

## 4. Hierarchical Reductions & Hash Aggregation (`aggregate.cuh`)

Supports `SUM`, `COUNT`, `MIN`, `MAX`, `AVG`, `VARIANCE`, and `STDDEV`.

- **Warp Level:** Threads exchange partial sums using `__shfl_down_sync` in 5 clock cycles without memory barriers.
- **Block Level:** The 32 warp leaders write partial sums into shared memory. Warp 0 performs a final reduction to produce the block total.
- **Global Level:** Block totals are accumulated into global output addresses using atomic additions.

## 5. Fused Scan-Filter-Project-Sum Pipeline (`fused_pipeline.cuh`)

Fuses `SELECT sum(a * b) FROM table WHERE filter_col = literal;` into a single kernel launch.

- Column buffers for `filter_col`, `col_a`, and `col_b` are fetched concurrently into registers.
- The predicate is evaluated in registers. Non-matching lanes zero out their operands.
- The arithmetic product `a * b` is evaluated in registers.
- Partial products are reduced across warps and blocks in registers and shared memory.
- Zero intermediate row IDs or temporary product arrays are written to global memory.
