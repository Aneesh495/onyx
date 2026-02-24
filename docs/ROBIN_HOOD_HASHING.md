# GPU-Accelerated Open-Addressing Hash Joins in Onyx

## 1. The Challenge of Join Execution on Massively Parallel GPUs

Relational equi-joins are traditionally computed via hash join:
1. **Build Phase**: Ingest the smaller relation into an in-memory hash table.
2. **Probe Phase**: Scan the larger relation, hash each join key, and retrieve matching tuples.

On massively parallel GPU architectures, traditional chained hash tables (linked lists of bucket nodes) suffer from severe performance penalties due to pointer chasing, non-coalesced memory reads, and dynamic memory allocation inside GPU kernels.

Onyx employs a flat, cache-aligned open-addressing hash table optimized specifically for SIMT memory subsystems.

## 2. Table Layout and Cache-Line Alignment

Each bucket in the Onyx hash table is packaged into a compact 16-byte struct:

```cpp
struct HashBucket {
    uint64_t key;       // 64-bit key (or 64-bit integer hash)
    uint32_t row_id;    // 32-bit row pointer into build batch
    uint32_t padding;   // 32-bit padding for strict 16-byte alignment
};
```

Because each bucket is exactly 16 bytes, four consecutive buckets fit into a single 64-byte GPU L2 cache line. When an entire warp of 32 threads executes a probe step, memory transactions coalesce across contiguous memory addresses, achieving peak bandwidth saturation.

```
+-----------------------------------------------------------------------+
|                       64-Byte Cache Line Segment                      |
|  Bucket 0 (16B)  |  Bucket 1 (16B)  |  Bucket 2 (16B)  | Bucket 3 (16B)|
|  [key | row_id]  |  [key | row_id]  |  [key | row_id]  | [key | row_id]|
+-----------------------------------------------------------------------+
```

## 3. Atomic Build Phase with Linear Probing

During the build phase, GPU threads insert keys concurrently using 64-bit atomic compare-and-swap (`atomicCAS`):

```cpp
uint64_t bucket_idx = bit_utils::murmur_hash64(key) & mask;

while (true) {
    uint64_t prev = atomicCAS(
        reinterpret_cast<unsigned long long*>(&table[bucket_idx].key),
        static_cast<unsigned long long>(HASH_EMPTY_KEY),
        static_cast<unsigned long long>(key));

    if (prev == HASH_EMPTY_KEY || prev == key) {
        table[bucket_idx].row_id = static_cast<uint32_t>(row_idx);
        break;
    }

    // Linear probe step with wrap-around
    bucket_idx = (bucket_idx + 1) & mask;
}
```

By sizing the table capacity to the next power of two such that the load factor remains below $\alpha = 0.50$, the expected probe sequence length is bounded:

$$\mathbb{E}[\text{Probe Length}] \le \frac{1}{2} \left( 1 + \frac{1}{1 - \alpha} \right) = 1.5 \text{ cache probes}$$

## 4. Multi-Match Join Resolution and Result Compaction

When join keys contain non-unique matches, the probe phase continues along the linear probe chain until an empty slot is encountered. Each matching pair is recorded into an intermediate match list via block-wide cooperative aggregation, preventing global atomic serialization.
