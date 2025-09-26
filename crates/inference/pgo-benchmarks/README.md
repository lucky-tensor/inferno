# PGO Benchmarks

This is a dedicated crate for benchmarking Profile-Guided Optimization (PGO) performance improvements in Inferno inference.

**Location**: `crates/inference/pgo-benchmarks/` - Part of the inference crate's testing suite.

## Quick Start

```bash
# Set model path
export BENCH_MODEL_PATH=/path/to/your/model.safetensors

# Run PGO benchmarks (automatically builds PGO binaries)
cargo bench --package pgo-benchmarks
```

## Fast Development Mode

If the build is taking too long or you want to skip the heavy PGO build process:

```bash
# Skip PGO binary building (for faster iteration)
export SKIP_PGO_BUILD=1
cargo bench --package pgo-benchmarks

# Or just compile without running
export SKIP_PGO_BUILD=1
cargo check --package pgo-benchmarks

# For verbose build progress (see what build.rs is doing)
cargo bench --package pgo-benchmarks -vv
```

This will skip the expensive PGO build step and just check for existing binaries at runtime.

## What This Does

The benchmark automatically:

1. **Builds baseline binaries** (if missing)
2. **Runs PGO profiling** workloads to generate profile data
3. **Builds PGO-optimized** binaries using the profile data
4. **Runs comprehensive benchmarks** comparing baseline vs PGO performance
5. **Reports improvements** with statistical analysis

## Benchmark Types

### Single Request Performance
- Tests individual inference requests
- Expected improvements: 10-20%

### Medium Concurrency (10-50 requests)
- Tests concurrent requests within single process
- Expected improvements: **20-40%** ⭐

### High Concurrency (100-200 requests)
- Stress tests with high concurrent load
- Expected improvements: **30-60%** ⭐

## Why a Separate Crate?

- **Faster builds**: Minimal dependencies (only criterion)
- **Clean separation**: PGO benchmarks separate from general benchmarks
- **Focused scope**: Only builds what's needed for PGO testing
- **Better caching**: Doesn't rebuild when other parts of project change

## Requirements

- Model file at `BENCH_MODEL_PATH`
- Rust toolchain with llvm-profdata
- `scripts/build-pgo.sh` and `scripts/build-pgo-examples.sh`

## Output

Results are saved to:
- HTML reports: `./target/criterion/pgo_*/`
- Detailed logs in terminal with improvement percentages

## Troubleshooting

### Build Hanging

If the build hangs during PGO binary creation:

```bash
# Use verbose mode to see progress
cargo bench --package pgo-benchmarks -vv

# Or skip PGO build entirely for testing
export SKIP_PGO_BUILD=1
cargo bench --package pgo-benchmarks
```

### Missing Model Error

```bash
❌ Model not found at: /path/to/model.safetensors
```
**Solution**: Download a model or set `BENCH_MODEL_PATH` environment variable.

### Missing PGO Binaries

```bash
❌ PGO binary not found at ./target/release/inferno-pgo
```
**Solution**: The build.rs script should handle this automatically. If it fails, check that:
- `scripts/build-pgo.sh` exists
- `scripts/build-pgo-examples.sh` exists
- You have `llvm-profdata` installed

### Permission Errors

If you get permission errors during PGO profiling:
```bash
# Make sure scripts are executable
chmod +x scripts/build-pgo.sh scripts/build-pgo-examples.sh
```

### High Memory Usage

The high concurrency tests (100+ workers) may use significant memory.
**Solution**: Reduce concurrency levels by editing the benchmark files if your system has limited RAM.

If benchmarks fail to find binaries, the build.rs script will show helpful error messages about what's missing and how to fix it.