# PGO Benchmarks

This is a dedicated crate for benchmarking Profile-Guided Optimization (PGO) performance improvements in Inferno inference.

**Location**: `crates/inference/pgo-benchmarks/` - Part of the inference crate's testing suite.

## Quick Start

```bash
# Set model path
export BENCH_MODEL_PATH=/path/to/your/model.safetensors

# Build the concurrent_inference example
cargo build --release --package inferno-inference --example concurrent_inference --features examples

# Run concurrent inference benchmarks
cargo bench --package pgo-benchmarks
```

## Fast and Simple

This benchmark system is now lightweight and fast:
- No shell scripts required
- No hanging builds during cargo operations
- Simply benchmarks the concurrent_inference example
- Build time: ~1 minute (only for concurrent_inference example)
- Benchmark build: ~30 seconds

## What This Does

The benchmark:

1. **Uses pre-built concurrent_inference example** (must be built manually)
2. **Runs comprehensive benchmarks** testing different concurrency levels
3. **Reports performance metrics** with statistical analysis

## Benchmark Types

### Single Request Performance
- Tests individual inference requests with different prompt lengths
- Benchmarks: short, medium, and long prompts

### Medium Concurrency (10-50 requests)
- Tests concurrent requests within single process
- Concurrency levels: 10, 25, 50 requests

### High Concurrency (100-200 requests)
- Stress tests with high concurrent load
- Concurrency levels: 100, 200 requests

## Why a Separate Crate?

- **Faster builds**: Minimal dependencies (only criterion)
- **Clean separation**: Concurrent inference benchmarks separate from general benchmarks
- **Focused scope**: Only builds what's needed for concurrent testing
- **Better caching**: Doesn't rebuild when other parts of project change

## Requirements

- Model file at `BENCH_MODEL_PATH`
- Rust toolchain
- No additional dependencies or scripts required

## Output

Results are saved to:
- HTML reports: `./target/criterion/*/`
- Detailed performance metrics in terminal output

## Troubleshooting

### Missing Model Error

```bash
❌ Model not found at: /path/to/model.safetensors
```
**Solution**: Download a model or set `BENCH_MODEL_PATH` environment variable.

### Missing Binary Error

```bash
❌ concurrent_inference binary not found at ./target/release/examples/concurrent_inference
```
**Solution**: The build.rs script should handle this automatically. If it fails, manually build:
```bash
cargo build --release --package inferno-inference --example concurrent_inference --features examples
```

### High Memory Usage

The high concurrency tests (100+ workers) may use significant memory.
**Solution**: Reduce concurrency levels by editing the benchmark files if your system has limited RAM.

The build.rs script will show helpful error messages about what's missing and how to fix it.