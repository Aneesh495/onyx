# Building and Developing Onyx

This guide details how to build, test, and benchmark the Onyx query engine across diverse hardware environments.

## Prerequisites

### Native CUDA Hardware (Linux / x86_64 or aarch64)
- **NVIDIA GPU**: Compute Capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell).
- **CUDA Toolkit**: 11.8, 12.0, or newer (`nvcc` must be on `PATH`).
- **C++ Compiler**: GCC 11+, Clang 14+ with C++23 support.
- **CMake**: 3.22 or newer.

### Development / CPU SIMT Emulation (macOS / Linux without GPU)
- **C++ Compiler**: Clang 16+ or GCC 12+ with full C++23 standard library support.
- **CMake**: 3.22 or newer.
- No physical NVIDIA GPU is required; the built-in SIMT engine executes cooperative warp algorithms with bit-for-bit equivalence.

## Build Instructions

### Standard Release Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### Build Targets
- `libonyx_core.a`: Core database engine, SIMT runtime, and kernel library.
- `onyx-cli`: Interactive SQL command-line interface.
- `onyx-server`: REST and streaming HTTP server on port 9090.
- `onyx-tests`: Unified test and bit-for-bit verification binary.
- `onyx-bench`: Multi-operator micro-benchmark and TPC-H harness.

## Running the Automated Test Suite

Execute the unified verification test binary:
```bash
./build/onyx-tests
```

Run under AddressSanitizer to guarantee zero memory corruption or leaks:
```bash
cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=address -g"
cmake --build build
./build/onyx-tests
```

## Running the High-Throughput Benchmarks

Execute the micro-benchmarking harness:
```bash
./build/onyx-bench
```

## Python SDK Installation and Testing

```bash
cd python
pip install -e .
python3 -m unittest discover -s tests
```
