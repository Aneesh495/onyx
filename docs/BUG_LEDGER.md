# Onyx Bug Ledger and Regression Postmortem

This document logs critical edge-case bugs identified during the architecture and implementation of Onyx, along with reproduction tests and resolution mechanics.

---

### Bug 01: Warp Voting Offset Race Condition on Block Counter

- **Subsystem:** `kernels/filter.cuh`
- **Symptom:** When multiple warps within the same thread block had matching rows simultaneously, row IDs in the output index buffer were overwritten or omitted.
- **Root Cause:** Multiple warps were executing atomic increments against the global output counter independently, resulting in interleaved block offsets and non-contiguous coalescing.
- **Reproduction Test:** `tests/unit/test_filter_kernels.cpp` (`FilterKernelGreaterThan` with >256 elements).
- **Resolution:** Implemented a two-tier aggregation hierarchy:
  1. Each warp leader performs an atomic add to a thread-block shared counter `shared_block_match_count`.
  2. Block thread 0 performs a single atomic add to the global output pointer, storing the base offset in `shared_block_offset`.
  3. Individual threads compute output slots as `shared_block_offset + shared_warp_offset + warp_rank`.

---

### Bug 02: Radix Sort Ping-Pong Buffer Pointer Mismatch on Odd Bit Passes

- **Subsystem:** `kernels/radix_sort.cuh`
- **Symptom:** For 32-bit keys sorted with 4-bit digits (8 passes), results were sorted correctly. However, for odd bit passes (e.g. 12-bit custom fields), the output array remained unpopulated.
- **Root Cause:** Pointer swapping between `in_keys` and `out_keys` ended on the temporary heap buffer rather than the caller's target array when the total number of digit passes was odd.
- **Reproduction Test:** `tests/unit/test_radix_sort_kernels.cpp` (`RadixSortAscending`).
- **Resolution:** Added an explicit post-pass buffer reconciliation check:
  ```cpp
  if (in_k != keys) {
      std::memcpy(keys, in_k, num_elements * sizeof(KeyType));
      if (values) {
          std::memcpy(values, in_v, num_elements * sizeof(ValueType));
      }
  }
  ```

---

### Bug 03: Hash Join Bucket Probe Infinite Loop on High Load Factors

- **Subsystem:** `kernels/hash_join.cuh`
- **Symptom:** When build table size approached the hash table capacity, probe threads would cycle indefinitely through full buckets.
- **Root Cause:** Quadratic probing lacked an upper bound on probing steps, and table capacity was allocated with inadequate slack for open addressing.
- **Reproduction Test:** `tests/unit/test_hash_join_kernels.cpp` (`HashJoinBasicMatching`).
- **Resolution:**
  1. Enforced `table_capacity` to be a power of two with a maximum 50% load factor (`next_power_of_two(num_build_rows * 2)`).
  2. Added a hard cap `MAX_PROBE_STEPS = 64` to prevent infinite loops on unresolvable collisions.

---

### Bug 04: Memory Pool Class Index Underflow for Sub-64-Byte Allocations

- **Subsystem:** `cuda/gpu_memory_pool.cpp`
- **Symptom:** Small allocations (<64 bytes) resulted in out-of-bounds array access in `free_lists_`.
- **Root Cause:** `bit_utils::countr_zero64(rounded) - bit_utils::countr_zero64(MIN_BLOCK_SIZE)` produced negative values for inputs less than `MIN_BLOCK_SIZE = 64`.
- **Reproduction Test:** `tests/unit/test_gpu_memory_pool.cpp` (`GpuMemoryPoolAllocDealloc`).
- **Resolution:** Added an upfront guard:
  ```cpp
  if (bytes <= MIN_BLOCK_SIZE) return 0;
  ```
