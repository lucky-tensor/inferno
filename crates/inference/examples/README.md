# Inferno Examples

This directory contains example applications demonstrating how to use the Inferno inference library.

## Simple Inference Example

The `simple-inference` example shows how to run inference on a safetensors model using the Inferno library.

### Usage

```bash
# Run with default settings (CUDA if available, otherwise CPU)
cargo run --bin simple-inference -- --prompt "What is 2+2?" --model-path /path/to/your/model

# Force CPU backend
cargo run --bin simple-inference -- --prompt "What is 2+2?" --model-path /path/to/your/model --cpu
```

## Concurrent Inference Example

The `concurrent-inference` example demonstrates **true concurrency** within a single process, perfect for PGO benchmarking.

### Usage

```bash
# Single request (same as simple-inference)
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "Hello" --model-path /path/to/model

# 10 concurrent requests in the same process
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "Hello" --model-path /path/to/model --concurrent 10

# High concurrency with verbose output
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "Hi" --model-path /path/to/model --concurrent 100 --verbose
```

### Key Features

- **Single Process**: All requests run in one process (important for PGO testing)
- **Async Concurrency**: Uses Tokio for efficient async request handling
- **Shared Model**: Single model loaded once, shared across all requests
- **Detailed Stats**: Shows throughput, latency distribution, and timing breakdown
- **Lock Contention**: Tests real concurrency bottlenecks

### Example Output

```
Model path: /path/to/model.safetensors
Prompt: hello
Concurrent requests: 50

Initializing inference engine... Using CandleCuda backend
Ready! (loaded in 2.1s)
Running 50 concurrent inference requests...

🎉 Concurrent inference completed!
Total time: 8.456s
Successful requests: 50/50
Total tokens generated: 1247

Individual request statistics:
  Mean:   4.123s
  Median: 4.089s
  Min:    3.876s
  Max:    4.445s

Throughput:
  5.91 requests/second
  147.42 tokens/second

Timing breakdown:
  Model loading: 2.100s
  Concurrent inference: 8.456s
```

## Building Examples

Examples are built as part of the inference crate:

```bash
# Build all examples
cargo build --release --package inferno-inference --examples --features examples

# Run specific examples
cargo run --package inferno-inference --example simple_inference --features examples -- --prompt "Hello" --model-path /path/to/model
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "Hello" --model-path /path/to/model --concurrent 10
```

## Features

- `examples`: Enable CLI dependencies for examples (required)
- `candle-cuda`: Enable CUDA acceleration (default)
- `candle-cpu`: Use CPU-only inference
- `candle-metal`: Use Metal acceleration (Apple Silicon)
- `burn-cpu`: Use Burn framework with CPU backend

## Environment Variables

- `INFERNO_MODEL_NAME`: Override the model name in config

## PGO Benchmarking

The concurrent-inference example is specifically designed for PGO (Profile-Guided Optimization) benchmarking.

### Quick PGO Benchmark

```bash
# Make sure you're in the project root directory first
cd ~/inferno

# Single command - automatically builds PGO binaries and runs benchmarks
export BENCH_MODEL_PATH=/path/to/your/model.safetensors
cargo bench --package pgo-benchmarks
```

### Manual Testing

```bash
# Test different concurrency levels manually
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "test" --model-path /path/to/model --concurrent 1
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "test" --model-path /path/to/model --concurrent 10
cargo run --package inferno-inference --example concurrent_inference --features examples -- --prompt "test" --model-path /path/to/model --concurrent 100
```

This tests real concurrency bottlenecks that PGO can optimize:
- Lock contention in shared model access
- Memory access patterns under high concurrency
- Branch prediction in hot async paths

See `docs/PGO_BENCHMARKING.md` for complete documentation.