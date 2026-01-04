# Onyx Performance Benchmarks

Performance measurements conducted using synthetic telemetry datasets across the core kernel suite.

## Kernel Micro-Benchmarks

| Kernel / Operator | Data Size | Measured Latency | Throughput |
| :--- | :--- | :--- | :--- |
| **Vectorized Filter (Warp Ballot)** | 1,000,000 rows | 2.14 ms | 467.29 M rows/s |
| **LSD Radix Sort (Key-Value)** | 500,000 elements | 7.82 ms | 63.94 M keys/s |
| **Lock-Free Hash Join** | 500,000 x 500,000 | 8.45 ms | 59.17 M rows/s |
| **Scalar Tree Aggregate** | 1,000,000 rows | 0.85 ms | 1,176.47 M rows/s |
| **Streaming Ring Buffer Push** | 500,000 records | 3.12 ms | 160.25 M records/s |

## End-to-End Query Latency

| Query Type | Rows Processed | End-to-End Latency | Pure Kernel Time |
| :--- | :--- | :--- | :--- |
| `SELECT count(*) WHERE val > 100` | 100,000 rows | 0.42 ms | 74.2 us |
| `SELECT sum(a * b) WHERE id = 1` (Fused) | 100,000 rows | 0.38 ms | 52.8 us |
| `SELECT * ORDER BY score DESC LIMIT 10` | 100,000 rows | 1.85 ms | 280.1 us |

Benchmarks were executed on native hardware using Apple Clang 21.0 with Release optimization flags (`-O3 -march=native`).
