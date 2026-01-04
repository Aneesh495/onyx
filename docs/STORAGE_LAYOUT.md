# Onyx Storage Layout Specification

Onyx employs an in-memory columnar Structure of Arrays (SoA) layout designed specifically for hardware coalescing, zero-copy host-device streaming, and memory footprint minimization.

## Column Memory Layout

A column consists of two primary contiguous memory allocations:

```text
+---------------------------------------------------------------+
| Aligned Data Buffer (N * width bytes, 128-byte aligned)       |
+---------------------------------------------------------------+
| Null Validity Bitmap (ceil(N / 8) bytes, 128-byte aligned)   |
+---------------------------------------------------------------+
```

### 1. Data Buffer
Fixed-width types (`INT8`, `INT16`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`, `TIMESTAMP`) are laid out as flat arrays.
- Memory addresses are guaranteed to be 128-byte aligned.
- When 32 threads in a warp access consecutive indices, the access pattern maps to contiguous cache lines without split-cache-line penalties.

### 2. Null Validity Bitmap
Nullability is tracked using a bit-packed validity mask (1 bit per row):
- `1` indicates a valid (non-null) value.
- `0` indicates a null value.
- Byte index: `row_id >> 3`
- Bit index within byte: `row_id & 7`

### 3. Variable-Length String Storage
Strings are stored in an append-only byte heap paired with two auxiliary 32-bit integer arrays:
- `string_offsets`: byte offset of each string within the heap.
- `string_lengths`: length of each string in bytes.
Fixed-width columns in batches store only the 32-bit offset into the heap.

## Host-Pinned Circular Ring Buffer (`PinnedRingBuffer`)

Streaming micro-batches from network sockets or log ingestion pipelines stream into a lock-free circular ring buffer allocated in host-pinned memory.
- `head_` and `tail_` atomic pointers reside on independent 128-byte cache lines to eliminate false sharing.
- Consumer threads peek contiguous slices and transfer them via PCIe DMA to device memory without stopping analytical query execution.
