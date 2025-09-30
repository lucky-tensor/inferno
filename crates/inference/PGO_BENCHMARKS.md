# Profile-Guided Optimization (PGO) Benchmarks

## Overview

The PGO benchmarks in this crate are designed to measure and validate the performance improvements achieved through Profile-Guided Optimization (PGO) in Rust ML inference workloads. PGO is a compiler optimization technique that uses runtime profiling data to guide code generation, potentially improving performance by 10-30% for CPU-intensive applications.

## What PGO Benchmarks Test

### Core Hypothesis
**PGO should provide meaningful performance improvements in CPU-intensive inference scenarios while having minimal impact in I/O or GPU-bound workloads.**

### Benchmark Categories

#### 1. **Cold Start vs Warm Performance**
- **`single_request_pgo_comparison`**: Measures full cold start performance including model loading and GPU initialization
- **`single_request_pgo_warm`**: Isolates CPU inference performance with pre-warmed GPU context

**What we expect**: Cold start tests show little PGO benefit (GPU/I/O bound), warm tests show significant PGO improvements (CPU bound).

#### 2. **Concurrency Scaling Analysis**
- **`medium_concurrency_pgo_comparison`**: Tests 10-50 concurrent requests
- **`high_concurrency_pgo_comparison`**: Tests 100-200 concurrent requests

**What we expect**: Higher concurrency levels should show more PGO benefit as CPU utilization increases relative to I/O overhead.

#### 3. **CPU-Intensive Workload Validation**
- **`cpu_intensive_pgo_comparison`**: Tests scenarios designed to maximize CPU inference time
- Uses warm-up cycles and complex prompts to stress CPU paths

**What we expect**: These should show the largest PGO improvements (15-30%) since they isolate the CPU optimization benefits.

## Detailed Metrics Collection

The benchmarks collect comprehensive performance data beyond simple wall-clock timing:

### Cold Start Breakdown
- **Backend Initialization**: Time to initialize ML backend (Candle/CUDA)
- **Engine Creation**: Time to create inference engine
- **Model Loading + GPU**: Time for model loading and GPU context setup
- **Total Cold Start**: Complete initialization time

### Concurrent Performance Analysis
- **Lock Wait Time**: Time spent waiting for shared inference engine access
- **Pure Inference Time**: CPU time spent in actual model inference (PGO target)
- **Parallelism Efficiency**: How well the system utilizes multiple CPU cores
- **Throughput Metrics**: Requests/second and tokens/second

### PGO Optimization Insights
- **Inference Percentage**: What % of total time is spent in CPU inference vs I/O
- **PGO Suitability Score**: Automated analysis of whether workload benefits from PGO
- **Performance Improvement**: Quantified benefit of PGO-optimized vs original binary

## Expected Results Pattern

### Scenarios Where PGO Should Excel
1. **Warm Inference**: 15-30% improvement in pure inference time
2. **High CPU Utilization**: When inference time > 50% of total request time
3. **CPU-Intensive Prompts**: Complex reasoning tasks that stress model computation
4. **High Concurrency**: Many concurrent requests maximizing CPU usage

### Scenarios Where PGO Shows Little Benefit
1. **Cold Start**: GPU initialization dominates timing
2. **I/O Bound**: Model loading and memory transfers are bottlenecks
3. **Low Concurrency**: Single requests with significant GPU overhead
4. **Simple Prompts**: Minimal CPU computation required

## Binary Verification Safety

The benchmarks include critical safety checks to ensure meaningful comparisons:

- **SHA256 Binary Verification**: Confirms original and PGO binaries are actually different
- **Automatic Failure**: Benchmarks abort if identical binaries are detected
- **Build Process Validation**: Ensures PGO instrumentation and optimization steps succeeded

## Usage

### Running PGO Benchmarks

1. **Generate PGO-Optimized Binaries**:
   ```bash
   ./crates/inference/build-pgo-concurrent.sh
   ```

2. **Run Comprehensive Benchmarks**:
   ```bash
   cargo bench --bench pgo_benches -p inferno-inference
   ```

3. **Set Model Path** (if needed):
   ```bash
   BENCH_MODEL_PATH=/path/to/model cargo bench --bench pgo_benches -p inferno-inference
   ```

### Interpreting Results

Look for these key indicators of successful PGO optimization:

- **Significant improvement in warm benchmarks** (10%+ faster)
- **High "inference percentage" values** in output logs (>50%)
- **Consistent improvements across CPU-intensive scenarios**
- **Minimal change in cold start benchmarks** (expected - I/O bound)

### Warning Signs

- **No performance difference**: May indicate PGO build failed
- **Performance regression**: Could suggest profile data mismatch
- **Identical benchmark results**: Binary verification should catch this

## Technical Implementation

### Architecture
- **Criterion.rs**: Statistical benchmarking with outlier detection
- **Regex Log Parsing**: Extracts detailed metrics from concurrent_inference output
- **Binary SHA Verification**: Prevents false benchmark results
- **Warm-up Cycles**: Isolates CPU optimization from GPU cold start penalties

### Dependencies
- `criterion`: Statistical benchmarking framework
- `regex`: Log parsing for detailed metrics extraction
- `concurrent_inference` example: Target binary for PGO optimization

## Contributing

When modifying PGO benchmarks:

1. **Maintain Safety Checks**: Always verify binaries are different before benchmarking
2. **Document Expected Results**: Update this file if adding new benchmark scenarios
3. **Test Both Scenarios**: Ensure benchmarks work with and without PGO benefits
4. **Statistical Rigor**: Use appropriate sample sizes (≥10) for reliable results

## Troubleshooting

### Common Issues
- **"assertion failed: n >= 10"**: Increase `sample_size()` in benchmark groups
- **Binary verification failure**: Run PGO build script to generate different binaries
- **No performance difference**: Check that PGO profile data was collected properly
- **Build timeouts**: Increase measurement time for resource-intensive benchmarks