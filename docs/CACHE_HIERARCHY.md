# Memory Hierarchy and Cache Alignment in Onyx

## 1. Overview of the Onyx Memory Subsystem

Efficient GPU data processing requires careful orchestration across four distinct physical memory spaces:

1. **Host Pinned Memory**: Non-pageable system RAM mapped into the GPU virtual address space via page-locked DMA allocations.
2. **Device Global Memory (HBM3/GDDR6)**: The main device storage holding columnar tables and memory pool arenas.
3. **L2 Cache**: Unified 40MB to 96MB on-chip cache shared across all Streaming Multiprocessors (SMs).
4. **Shared Memory / L1 Data Cache**: Ultra-low-latency 128KB to 228KB SRAM per SM with bandwidth exceeding 15 TB/s.

```
+---------------------------------------------------------------+
|                       Host System RAM                         |
|                 Pinned Direct Memory Access                   |
+---------------------------------------------------------------+
                                | (PCIe Gen 5 / NVLink)
                                v
+---------------------------------------------------------------+
|                     Device Global Memory                      |
|                  Onyx Size-Class Pool Arenas                  |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                    Unified On-Chip L2 Cache                   |
|                   64-Byte Cache Line Access                   |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                 SM Shared Memory / L1 Cache                   |
|               32 Banks, 4 Bytes per Bank Width                |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                       Warp Register File                      |
|                     64K 32-bit Registers                      |
+---------------------------------------------------------------+
```

## 2. Memory Coalescing Rules

To maximize memory bus throughput, memory access requests from the 32 threads in a warp are coalesced into the minimum number of cache line transactions.

In Onyx columnar storage:
- All column arrays are aligned to 128-byte boundaries using `posix_memalign` on host and 128-byte aligned arena offsets on device.
- Successive threads in a warp access consecutive addresses: thread $k$ reads address $\text{Base} + k \cdot \text{sizeof}(T)$.
- For 32-bit types, a 32-thread warp accesses $32 \times 4 = 128$ bytes, which maps onto exactly two 64-byte L2 cache lines with zero wasted bytes.

## 3. Shared Memory Bank Conflict Mitigation

Shared memory is divided into 32 equal-sized memory modules called banks. Consecutive 32-bit words map to successive banks:

$$\text{Bank Index} = (\text{Byte Address} / 4) \bmod 32$$

When multiple threads within the same warp access different addresses within the same bank, accesses are serialized (a bank conflict).

Onyx avoids bank conflicts through stride padding:
- In radix sort digit histogramming, shared counters are dimensioned as `[RADIX_BINS + 1]` rather than `[RADIX_BINS]` to offset bank indices across warps.
- In warp tree reductions, intra-register shuffles (`__shfl_down_sync`) are used instead of shared memory arrays, eliminating shared memory traffic entirely for intra-warp operations.
