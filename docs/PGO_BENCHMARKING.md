# PGO Benchmarking Guide

This document explains how to benchmark Profile-Guided Optimization (PGO) performance improvements in Inferno inference, with a focus on **true concurrency testing** within single processes.

## Overview

Profile-Guided Optimization (PGO) can dramatically improve performance, especially under concurrent workloads where:
- **Better branch prediction** reduces CPU pipeline stalls in hot async paths
- **Improved cache locality** reduces memory access latency under contention
- **Optimized lock contention** improves performance with shared model access
- **Enhanced memory access patterns** benefit high-concurrency scenarios

## 🚀 Quick Start

### 1. Set Model Path
```bash
# Required: Point to your model file
export BENCH_MODEL_PATH=/path/to/your/model.safetensors

# Or use default location (download model there first)
export BENCH_MODEL_PATH=~/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors
```

### 2. Run Complete PGO Benchmark Suite
```bash
# Navigate to project root
cd ~/inferno

# Single command - fully automated PGO benchmarking
cargo bench --package pgo-benchmarks
```

### 3. Development Mode (Skip Heavy Builds)
```bash
# For faster iteration during development
export SKIP_PGO_BUILD=1
cargo bench --package pgo-benchmarks

# Or see verbose build progress
cargo bench --package pgo-benchmarks -vv
```

## ✨ What Happens Automatically

The **build.rs** script handles everything:

1. **📦 Baseline Build**: Builds `inferno` + `concurrent_inference` examples
2. **🔄 PGO Profiling**: Runs real inference workloads to generate profile data
3. **🔥 Optimized Build**: Rebuilds binaries with profile-guided optimization
4. **📊 Comprehensive Testing**: Benchmarks baseline vs PGO across concurrency levels
5. **📈 Statistical Analysis**: Generates detailed performance comparison reports

**First run**: 5-15 minutes (includes PGO profiling)
**Subsequent runs**: ~30 seconds (cached binaries)

## 📊 Benchmark Architecture

### Modern Approach: True Concurrency Testing
Unlike traditional benchmarks that spawn separate processes, our PGO benchmarks test **real concurrency** within single processes:

- **✅ Single Process**: All concurrent requests share the same model instance
- **✅ Async Contention**: Real lock contention and memory access patterns
- **✅ Shared Resources**: Tests actual bottlenecks PGO optimizes
- **✅ Realistic Load**: Mirrors production inference server behavior

### Benchmark Categories

#### 🏃 Single Request Performance
- **Tests**: Various prompt lengths and complexity
- **Focus**: Cold start, model loading, individual inference latency
- **Expected PGO gains**: 10-20% faster

#### 🔥 Medium Concurrency (10-50 requests) ⭐ **PGO Sweet Spot**
- **Tests**: 10, 25, 50 concurrent requests in same process
- **Focus**: Lock contention, cache locality, memory access patterns
- **Expected PGO gains**: **20-40% faster**
- **Why it shines**: Optimal balance of contention without overwhelming resources

#### ⚡ High Concurrency (100-200+ requests) ⭐ **Maximum Benefits**
- **Tests**: 100, 200+ concurrent requests (stress test)
- **Focus**: Memory bandwidth, cache efficiency, extreme contention
- **Expected PGO gains**: **30-60% faster**
- **Why it's dramatic**: PGO optimizes hot paths under maximum load

## 🎯 Key Metrics & Expected Results

| Concurrency Level | Baseline Performance | PGO Performance | Expected Improvement | Why PGO Helps |
|-------------------|---------------------|-----------------|---------------------|---------------|
| **Single Request** | Good | Better | 10-20% faster | Branch prediction, cache warmup |
| **Medium (10-50)** | Degrades with contention | Much better | **20-40% faster** ⭐ | Optimized lock paths, better memory patterns |
| **High (100-200+)** | Significant degradation | Surprisingly good | **30-60% faster** ⭐ | Cache locality, reduced context switching |

### What Makes PGO Effective for Concurrency

1. **Hot Path Optimization**: Frequently executed code paths (lock acquisition, memory access) are optimized
2. **Branch Prediction**: Better prediction in async/await state machines
3. **Cache Line Optimization**: Data structure layouts optimized for concurrent access
4. **Memory Access Patterns**: Reduced cache misses under contention

## Key Metrics to Watch

### Latency Improvements
- **P50 latency**: Median response time
- **P95/P99 latency**: Tail latency improvements
- **Cold start time**: Model loading + first request

### Throughput Improvements
- **Requests per second**: Overall throughput
- **Concurrent request handling**: Scalability under load
- **Resource utilization**: CPU/memory efficiency

### Concurrency-Specific Benefits
- **Reduced lock contention**: Better synchronized access
- **Cache efficiency**: Improved memory access patterns
- **Branch prediction**: Fewer pipeline stalls in hot paths

## Expected Results

Based on typical PGO improvements:

| Scenario | Expected Improvement |
|----------|---------------------|
| Single request | 10-20% faster |
| Low concurrency (2-10) | 5-15% faster |
| Medium concurrency (20-100) | **20-40% faster** |
| High concurrency (200+) | **30-60% faster** |
| Cold start | 15-25% faster |

## Build System Integration

The benchmarking system includes automatic binary preparation:

### build.rs Integration
- Automatically detects when running benchmarks
- Builds baseline and PGO binaries as needed
- Validates model availability
- Provides helpful error messages

### Environment Variables
- `BENCH_MODEL_PATH`: Path to model file (required)
- `CARGO_BENCH_PGO_SKIP`: Skip PGO binary building
- `CARGO_BENCH_FAST`: Use fast PGO profiling mode

## Troubleshooting

### Missing Model Error
```bash
❌ Model not found at: /path/to/model.safetensors
```
**Solution**: Download a model or set `BENCH_MODEL_PATH`

### Missing PGO Binary
```bash
❌ PGO binary not found at ./target/release/inferno-pgo
```
**Solution**: Run `./scripts/build-pgo.sh` or let build.rs handle it

### High Memory Usage
The high concurrency tests (200+ workers) may use significant memory.
**Solution**: Reduce concurrency levels in benchmarks if needed

### Long Build Times
PGO builds include profiling which takes time.
**Solution**: Use `--fast` flag or let build.rs use fast mode

## Reading Results

### Criterion Reports
- HTML reports: `./target/criterion/report/index.html`
- Raw data: `./target/criterion/pgo_*/`

### Log Files
- Basic comparison: `./benchmark-results/criterion_pgo_*.log`
- Concurrency tests: `./benchmark-results/criterion_pgo_concurrency_*.log`

### Key Indicators
- **Lower is better**: Latency measurements
- **Higher is better**: Throughput measurements
- **Improvement percentage**: PGO vs baseline comparison

## Advanced Usage

### Custom Concurrency Levels
Edit `pgo_concurrency.rs` to test specific concurrency patterns:

```rust
let concurrency_levels = vec![10, 25, 50, 100, 200, 500, 1000];
```

### Custom Workloads
Edit `pgo_comparison.rs` to test domain-specific prompts:

```rust
let prompts = vec![
    "your custom prompt",
    "domain specific query",
    "workload representative text"
];
```

### BOLT Optimization
For even better performance, enable BOLT optimization:

```bash
./scripts/build-pgo.sh --bolt
```

This applies additional binary layout optimization after PGO.

## Integration with CI/CD

For automated performance regression testing:

```bash
# In CI pipeline
export BENCH_MODEL_PATH=/ci/models/test-model.safetensors
./scripts/bench-pgo.sh --output ./ci-benchmark-results

# Compare with baseline metrics
# (implement comparison logic based on your requirements)
```

## Further Reading

- [Rust PGO Documentation](https://doc.rust-lang.org/rustc/profile-guided-optimization.html)
- [LLVM PGO Guide](https://llvm.org/docs/HowToBuildWithPGO.html)
- [BOLT Optimization](https://github.com/llvm/llvm-project/blob/main/bolt/README.md)