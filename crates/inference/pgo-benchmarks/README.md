# Inference Benchmarking

This directory contains benchmarking scripts and tools for the inference crate.

## Scripts

### `build-pgo-concurrent.sh`

Creates a PGO (Profile-Guided Optimization) optimized version of the `concurrent_inference` example.

**Prerequisites:**
```bash
# Build the concurrent_inference example first
cd crates/inference
cargo build --release --example concurrent_inference --features examples
```

**Usage:**
```bash
# Set model path (required)
export BENCH_MODEL_PATH=/path/to/your/model.safetensors

# Run PGO script (assumes binary already exists)
./crates/inference/benches/build-pgo-concurrent.sh
```

**What it does:**
1. **Checks** that `concurrent_inference` binary already exists (NO unnecessary builds!)
2. **Builds** minimal instrumented version for profiling only
3. **Profiles** with focused concurrency workloads (1, 5, 10, 25 workers)
4. **Creates** PGO-optimized binary as `concurrent_inference.pgo`
5. **Preserves** all three versions for benchmarking

**Key advantages:**
- ✅ **PGO-focused** - builds concurrent_inference with proper PGO instrumentation
- ✅ **Fast profiling** - focused workloads (15s timeouts per test)
- ✅ **Smart validation** - checks prerequisites upfront
- ⚠️ **Note**: RUSTFLAGS changes require dependency rebuilds (unavoidable with PGO)

**Expected build behavior:**
- First build: ~483 packages (instrumented version)
- Profiling: 4 quick tests with different concurrency levels
- Second build: ~483 packages again (PGO-optimized version)
- This is **normal** - PGO requires two complete builds with different compiler flags

**Output binaries for benchmarking:**
- `target/release/examples/concurrent_inference.original` - Baseline performance
- `target/release/examples/concurrent_inference.instrumented` - With profiling overhead
- `target/release/examples/concurrent_inference.pgo` - PGO-optimized version
- `target/release/examples/concurrent_inference-pgo` - Legacy name (same as .pgo)

**Requirements:**
- Model file at `$BENCH_MODEL_PATH`
- `llvm-profdata` (usually included with Rust toolchain)
- `examples` feature enabled

**Output:**
- PGO-optimized binary: `./target/release/examples/concurrent_inference-pgo`
- Can be used for performance comparisons against baseline

## Benchmarks

### `inference_benchmark.rs`

General inference benchmarking using Criterion.rs for performance regression testing.