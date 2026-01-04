# Onyx Query Planner and Optimizer Specification

The Onyx query optimization subsystem converts raw SQL text into an optimized physical execution DAG tailored for parallel GPU dispatch.

## Pipeline Architecture

```text
SQL Query Text
      |
      v
[ Lexer & Parser ] -----------------> Abstract Syntax Tree (AST)
                                              |
                                              v
[ Logical Plan Builder ] -----------> Logical Operator Tree
                                              |
                                              v
[ Rule-Based Optimizer ] -----------> Optimized Physical Plan DAG
                                              |
                                              v
[ Physical Executor Factory ] ------> GPU Kernel Execution Pipeline
```

## Optimization Passes

### 1. Predicate Pushdown
Pushes `LogicalFilter` operators directly adjacent to `LogicalScan` operators. This allows the GPU scan kernel to avoid copying unneeded rows into memory, reducing PCIe transfer bandwidth and device memory consumption.

### 2. Column Pruning
Scans are analyzed to identify the exact set of columns referenced by downstream expressions, filters, and joins. Unreferenced columns are omitted from device memory allocations.

### 3. Operator Fusion
Identifies sequences of operators matching:
`Scan -> Filter -> Aggregate(Sum(a * b))`
and collapses them into a single `PhysicalFusedPipeline` node. This executes the entire pipeline in registers without intermediate HBM allocations.

## Cost Model (`CostModel`)

For each operator, the cost model estimates:
- **Estimated Latency:** Calculated from expected cardinality and kernel throughput benchmarks.
- **Estimated GPU Memory:** Tracks device slab buffer requirements.
- **Arithmetic Intensity:** Ratio of floating-point/integer operations to memory bytes transferred (FLOPs/byte), ensuring kernels avoid memory bus saturation.
