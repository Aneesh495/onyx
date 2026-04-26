.PHONY: all build test test-python test-all bench cli server clean format lint demo check-cuda

BUILD_DIR ?= build
BUILD_TYPE ?= Release
NUM_JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

all: build

$(BUILD_DIR)/CMakeCache.txt: CMakeLists.txt
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: $(BUILD_DIR)/CMakeCache.txt
	@cmake --build $(BUILD_DIR) -j $(NUM_JOBS)

test: build
	@echo "Running Onyx Test Suite..."
	@./$(BUILD_DIR)/onyx-tests

test-python:
	@echo "Running Onyx Python Test Suite..."
	@PYTHONPATH=python python3 -m unittest discover -s python/tests

test-all: test test-python

bench: build
	@echo "Running Onyx Performance Benchmarks..."
	@./$(BUILD_DIR)/onyx-bench

cli: build
	@./$(BUILD_DIR)/onyx-cli

server: build
	@./$(BUILD_DIR)/onyx-server

demo: build
	@echo "Starting Onyx Demonstration Pipeline..."
	@./$(BUILD_DIR)/onyx-cli --exec "SELECT count(*) FROM telemetry WHERE latency_us > 1500;"

clean:
	@rm -rf $(BUILD_DIR) bin lib

format:
	@find include src tests benchmarks -name "*.hpp" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.cu" | xargs clang-format -i

check-cuda:
	@which nvcc 2>/dev/null && echo "CUDA compiler detected: $$(nvcc --version | grep release)" || echo "No NVCC detected; building with CPU SIMT fallback"
