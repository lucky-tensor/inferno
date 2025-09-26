#!/bin/bash

# PGO build script specifically for examples binary
# This creates PGO-optimized version of inferno-examples

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PGO_DATA_DIR="/tmp/inferno-examples-pgo-data"
MODEL_PATH="${BENCH_MODEL_PATH:-$HOME/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors}"
FAST_MODE=true  # Always use fast mode for examples

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model not found at: $MODEL_PATH"
    log_error "Set BENCH_MODEL_PATH environment variable"
    exit 1
fi

# Find llvm-profdata
RUSTC_SYSROOT=$(rustc --print sysroot)
RUST_LLVM_PROFDATA="$RUSTC_SYSROOT/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata"

if [ -f "$RUST_LLVM_PROFDATA" ]; then
    LLVM_PROFDATA="$RUST_LLVM_PROFDATA"
elif command -v llvm-profdata &> /dev/null; then
    LLVM_PROFDATA="llvm-profdata"
else
    log_error "llvm-profdata not found"
    exit 1
fi

log_info "Building PGO-optimized examples binary..."

# Clean existing data
rm -rf "$PGO_DATA_DIR"
mkdir -p "$PGO_DATA_DIR"

# Step 1: Build instrumented examples binary
log_info "Step 1/4: Building instrumented examples binary..."
RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR -C target-cpu=native" \
    cargo build --release --package inferno-inference --examples --features examples

# Step 2: Run profiling workloads on examples
log_info "Step 2/4: Running profiling workloads..."

# Test different concurrency levels with short prompts
CONCURRENT_TESTS=(
    "1 hi"
    "2 hello"
    "5 test"
    "10 quick"
)

for test in "${CONCURRENT_TESTS[@]}"; do
    read -r concurrency prompt <<< "$test"
    log_info "  Profiling concurrency=$concurrency prompt='$prompt'"

    timeout 30s ./target/release/examples/concurrent_inference \
        --prompt "$prompt" \
        --model-path "$MODEL_PATH" \
        --concurrent "$concurrency" >/dev/null 2>&1 || true
done

# Step 3: Merge profile data
log_info "Step 3/4: Merging profile data..."
"$LLVM_PROFDATA" merge -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR"

# Step 4: Build optimized examples binary
log_info "Step 4/4: Building PGO-optimized examples binary..."
RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata -C target-cpu=native" \
    cargo build --release --package inferno-inference --examples --features examples

# Copy to PGO location
cp ./target/release/examples/concurrent_inference ./target/release/examples/concurrent_inference-pgo

# Cleanup
rm -rf "$PGO_DATA_DIR"

log_info "✅ PGO concurrent_inference example created: ./target/release/examples/concurrent_inference-pgo"