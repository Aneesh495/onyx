# Contributing to Onyx

We welcome technical contributions to the Onyx GPU query engine. This document establishes guidelines for code style, kernel development, and regression verification.

## Engineering Principles

1. **Zero Divergence in Hot Loops**:
   Always prefer warp-level voting and shuffle reductions (`__shfl_down_sync`) over branching thread logic and global atomic contention.

2. **Strict Dual-Backend Parity**:
   Every new `.cu` CUDA kernel must have an exact algorithmic counterpart in the CPU dispatch layer. All regression tests must execute and pass with 100% bit-for-bit equivalence on platforms without physical GPUs.

3. **Memory Coalescing and Alignment**:
   All buffer allocations must adhere to 128-byte cache line alignment. Avoid shared memory bank conflicts by padding index buffers where necessary.

4. **Comprehensive Test Coverage**:
   Any newly introduced kernel, optimization pass, or storage layout must include unit tests, boundary edge-case tests, and benchmark coverage.

## Git Workflow and Commit Standards

- Write concise, descriptive commit messages using conventional prefixes (`kernels:`, `storage:`, `planner:`, `execution:`, `test:`, `docs:`).
- Verify compilation with zero warnings before committing.
- Run both `./build/onyx-tests` and AddressSanitizer passes prior to pull request submission.
