cmake_minimum_required(VERSION 3.22)
project(onyx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler optimization and warnings
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -g)
endif()

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)

option(ONYX_FORCE_SIMT_EMULATION "Force SIMT software emulation even if CUDA is available" OFF)

if(CMAKE_CUDA_COMPILER AND NOT ONYX_FORCE_SIMT_EMULATION)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(ONYX_HAS_CUDA ON)
    add_compile_definitions(ONYX_HAS_CUDA=1)
    message(STATUS "Native CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    
    # Set default CUDA architectures if not specified
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
else()
    set(ONYX_HAS_CUDA OFF)
    add_compile_definitions(ONYX_ENABLE_SIMT_EMULATION=1)
    add_compile_definitions(ONYX_HAS_CUDA=0)
    message(STATUS "Configuring Onyx with high-performance CPU SIMT emulation layer")
endif()

# Threading
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Core sources
set(ONYX_COMMON_SOURCES
    src/common/types.cpp
    src/common/error.cpp
    src/common/logger.cpp
)

set(ONYX_CUDA_SOURCES
    src/cuda/simt_engine.cpp
    src/cuda/gpu_memory_pool.cpp
    src/cuda/pinned_ring_buffer.cpp
    src/cuda/stream_manager.cpp
    src/cuda/cuda_graph.cpp
)

set(ONYX_STORAGE_SOURCES
    src/storage/column.cpp
    src/storage/column_batch.cpp
    src/storage/table.cpp
    src/storage/catalog.cpp
    src/storage/stream_ingestor.cpp
    src/storage/arrow_adapter.cpp
)

set(ONYX_SQL_SOURCES
    src/sql/token.cpp
    src/sql/lexer.cpp
    src/sql/ast.cpp
    src/sql/parser.cpp
)

set(ONYX_PLANNER_SOURCES
    src/planner/logical_plan.cpp
    src/planner/physical_plan.cpp
    src/planner/optimizer.cpp
    src/planner/cost_model.cpp
    src/planner/plan_visualizer.cpp
)

set(ONYX_EXECUTION_SOURCES
    src/execution/execution_context.cpp
    src/execution/operators.cpp
    src/execution/query_engine.cpp
    src/execution/pipeline_executor.cpp
)

set(ONYX_SERVER_SOURCES
    src/server/http_server.cpp
    src/server/telemetry.cpp
    src/server/json_serializer.cpp
)

# Kernels
if(ONYX_HAS_CUDA)
    set(ONYX_KERNEL_SOURCES
        src/kernels/filter_kernels.cu
        src/kernels/radix_sort_kernels.cu
        src/kernels/hash_join_kernels.cu
        src/kernels/aggregate_kernels.cu
        src/kernels/window_kernels.cu
        src/kernels/compression_kernels.cu
        src/kernels/string_kernels.cu
        src/kernels/fused_pipeline_kernels.cu
        src/kernels/dictionary_kernels.cu
        src/kernels/multi_radix_sort_kernels.cu
        src/kernels/rolling_window_kernels.cu
    )
else()
    set(ONYX_KERNEL_SOURCES
        src/kernels/filter_kernels.cpp
        src/kernels/radix_sort_kernels.cpp
        src/kernels/hash_join_kernels.cpp
        src/kernels/aggregate_kernels.cpp
        src/kernels/window_kernels.cpp
        src/kernels/compression_kernels.cpp
        src/kernels/string_kernels.cpp
        src/kernels/fused_pipeline_kernels.cpp
        src/kernels/dictionary_kernels.cpp
        src/kernels/multi_radix_sort_kernels.cpp
        src/kernels/rolling_window_kernels.cpp
    )
endif()

# Core static library
add_library(onyx_core STATIC
    ${ONYX_COMMON_SOURCES}
    ${ONYX_CUDA_SOURCES}
    ${ONYX_KERNEL_SOURCES}
    ${ONYX_STORAGE_SOURCES}
    ${ONYX_SQL_SOURCES}
    ${ONYX_PLANNER_SOURCES}
    ${ONYX_EXECUTION_SOURCES}
    ${ONYX_SERVER_SOURCES}
)

target_link_libraries(onyx_core PUBLIC Threads::Threads)
target_include_directories(onyx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# CLI Binary
add_executable(onyx-cli
    src/cli/repl.cpp
    src/cli/main.cpp
)
target_link_libraries(onyx-cli PRIVATE onyx_core)

# Unified Test Suite Binary
file(GLOB_RECURSE ONYX_TEST_SOURCES tests/*.cpp)
add_executable(onyx-tests
    ${ONYX_TEST_SOURCES}
)
target_include_directories(onyx-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(onyx-tests PRIVATE onyx_core)

# Benchmark Suite Binary
file(GLOB_RECURSE ONYX_BENCH_SOURCES benchmarks/*.cpp)
add_executable(onyx-bench
    ${ONYX_BENCH_SOURCES}
)
target_link_libraries(onyx-bench PRIVATE onyx_core)

# Server Binary
add_executable(onyx-server
    src/server/server_main.cpp
)
target_link_libraries(onyx-server PRIVATE onyx_core)
