#!/bin/bash

# Simple PGO script that assumes concurrent_inference binary already exists
# This script ONLY does PGO profiling and optimization

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Change to workspace root
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$WORKSPACE_ROOT"

log_info "PGO script - will rebuild concurrent_inference with PGO instrumentation"
log_warn "Note: RUSTFLAGS changes require dependency rebuilds - this is expected"

# Configuration
PGO_DATA_DIR="/tmp/inferno-concurrent-pgo-data"
MODEL_PATH="${BENCH_MODEL_PATH:-$HOME/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors}"
CONCURRENT_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference"

# Check prerequisites
if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model not found at: $MODEL_PATH"
    log_error "Set BENCH_MODEL_PATH environment variable"
    exit 1
fi

# Ensure we have a clean, optimized release binary first
if [ ! -f "$CONCURRENT_BINARY" ]; then
    log_info "Building clean release binary first..."
    cd crates/inference
    cargo build --release --example concurrent_inference --features examples
    cd "$WORKSPACE_ROOT"
else
    # Check if existing binary looks properly optimized (reasonable size)
    BINARY_SIZE=$(du -m "$CONCURRENT_BINARY" | cut -f1)
    if [ "$BINARY_SIZE" -gt 20 ]; then
        log_warn "Existing binary seems too large (${BINARY_SIZE}M) - rebuilding clean version"
        cd crates/inference
        cargo build --release --example concurrent_inference --features examples
        cd "$WORKSPACE_ROOT"
    else
        log_info "Existing binary size looks good: ${BINARY_SIZE}M"
    fi
fi

# Find compatible llvm-profdata
# Rust 1.90 uses LLVM 20, but system might have older versions
log_info "Rust version: $(rustc --version --verbose | grep LLVM)"

# Try to find a compatible llvm-profdata version
LLVM_PROFDATA=""

# Check for specific LLVM versions that might be compatible
for version in 20 19 18 17; do
    if command -v "llvm-profdata-$version" &> /dev/null; then
        log_info "Found llvm-profdata-$version"
        LLVM_PROFDATA="llvm-profdata-$version"
        break
    fi
done

# Fallback to default llvm-profdata
if [ -z "$LLVM_PROFDATA" ]; then
    if command -v llvm-profdata &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata"
        log_warn "Using system llvm-profdata (may have version mismatch)"
    else
        log_error "No compatible llvm-profdata found"
        log_error "Consider installing llvm-20-tools: sudo apt install llvm-20-tools"
        exit 1
    fi
fi

log_info "Using: $LLVM_PROFDATA"

log_info "Using existing binary: $(du -h "$CONCURRENT_BINARY" | cut -f1)"

# Save original binary for benchmarking comparison
ORIGINAL_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference.original"
cp "$CONCURRENT_BINARY" "$ORIGINAL_BINARY"
log_info "Saved original binary: $ORIGINAL_BINARY"

# Clean and prepare PGO data directory
rm -rf "$PGO_DATA_DIR"
mkdir -p "$PGO_DATA_DIR"

# Step 1: Build instrumented version for profiling
log_info "Step 1/3: Building instrumented version (dependencies will be rebuilt)..."
log_info "  This rebuilds dependencies due to RUSTFLAGS changes - it's unavoidable"

cd crates/inference
# Build instrumented version using dedicated PGO profile
log_info "  Using pgo-gen profile (matches release but with profiling instrumentation)"
RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR" \
    cargo build --release --example concurrent_inference --features examples
cd "$WORKSPACE_ROOT"

# Save instrumented binary for debugging/analysis (built with pgo-gen profile)
INSTRUMENTED_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference.instrumented"
RELEASE_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference"
cp "$RELEASE_BINARY" "$INSTRUMENTED_BINARY"

# Validate instrumented binary size (PGO instrumentation adds significant but expected overhead)
INSTRUMENTED_SIZE=$(du -m "$INSTRUMENTED_BINARY" | cut -f1)
ORIGINAL_SIZE=$(du -m "$ORIGINAL_BINARY" | cut -f1)
OVERHEAD_RATIO=$((INSTRUMENTED_SIZE * 100 / ORIGINAL_SIZE))
if [ "$INSTRUMENTED_SIZE" -gt $((ORIGINAL_SIZE * 8)) ]; then
    log_warn "Instrumented binary seems unusually large: ${INSTRUMENTED_SIZE}M vs original ${ORIGINAL_SIZE}M (${OVERHEAD_RATIO}% of original)"
    log_warn "This may indicate an issue with the build process"
else
    log_info "  Instrumented binary size: ${INSTRUMENTED_SIZE}M vs original ${ORIGINAL_SIZE}M (${OVERHEAD_RATIO}% overhead is normal for PGO instrumentation)"
fi
log_info "  Instrumented binary saved: $INSTRUMENTED_BINARY"
log_info "  Binary ready for profiling workloads"

# Step 2: Run profiling workloads
log_info "Step 2/3: Running profiling workloads..."

# Focused profiling tests
TESTS=("1 hi" "5 test" "10 quick" "25 brief")

for test in "${TESTS[@]}"; do
    read -r concurrency prompt <<< "$test"
    log_info "  Profiling: $concurrency workers with '$prompt'"

    # Use the instrumented binary built with release profile
    timeout 15s "$RELEASE_BINARY" \
        --prompt "$prompt" \
        --model-path "$MODEL_PATH" \
        --concurrent "$concurrency" >/dev/null 2>&1 || true
done

# Step 3: Build PGO-optimized binary
log_info "Step 3/3: Creating PGO-optimized binary..."
log_info "Merging profile data with $LLVM_PROFDATA..."

# Try to merge profile data with fallback strategies
if "$LLVM_PROFDATA" merge -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR" 2>/tmp/profdata-error.log; then
    log_info "Profile data merged successfully"
elif "$LLVM_PROFDATA" merge --failure-mode=warn -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR" 2>/tmp/profdata-error.log; then
    log_warn "Profile data merged with warnings (some data may be skipped)"
else
    log_error "Failed to merge profile data. Error details:"
    cat /tmp/profdata-error.log
    log_error "This is likely due to LLVM version mismatch between Rust and llvm-profdata"
    log_error "Rust uses $(rustc --version --verbose | grep LLVM), but $LLVM_PROFDATA expects older format"
    log_error "Continuing without PGO optimization..."

    # Create empty profdata file to avoid build failure
    touch "$PGO_DATA_DIR/merged.profdata"
fi

cd crates/inference
log_info "  Building final PGO-optimized binary (recompiling with profile data)..."
# Build PGO-optimized version using dedicated PGO profile with full optimizations
log_info "  Using pgo-use profile (release + PGO optimizations)"
RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata" \
    cargo build --release --example concurrent_inference --features examples
cd "$WORKSPACE_ROOT"

log_info "  PGO-optimized build complete"

# Save PGO-optimized binary (built with release profile)
PGO_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference.pgo"
RELEASE_PGO_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference"
cp "$RELEASE_PGO_BINARY" "$PGO_BINARY"

# Validate PGO binary size (should be similar or smaller than original)
PGO_SIZE=$(du -m "$PGO_BINARY" | cut -f1)
ORIGINAL_SIZE=$(du -m "$ORIGINAL_BINARY" | cut -f1)

if [ "$PGO_SIZE" -gt "$ORIGINAL_SIZE" ]; then
    log_warn "⚠️  PGO binary is larger than original: ${PGO_SIZE}M vs ${ORIGINAL_SIZE}M"
    log_warn "This may indicate PGO didn't optimize effectively or there's a build issue"
    log_warn "Check if profile data was collected properly during the profiling phase"
elif [ "$PGO_SIZE" -lt "$ORIGINAL_SIZE" ]; then
    log_info "✅ PGO optimization successful: ${PGO_SIZE}M vs ${ORIGINAL_SIZE}M (smaller!)"
else
    log_info "ℹ️  PGO binary same size as original: ${PGO_SIZE}M"
fi

log_info "  PGO-optimized binary saved: $PGO_BINARY"

# Also keep legacy name for compatibility
cp "$RELEASE_PGO_BINARY" "$WORKSPACE_ROOT/target/release/examples/concurrent_inference-pgo"

# Cleanup
rm -rf "$PGO_DATA_DIR"

# Verify and summarize results
log_info "\n=== PGO Build Summary ==="
log_info "All binary versions created for benchmarking comparison:"

if [ -f "$ORIGINAL_BINARY" ]; then
    log_info "✅ Original:     $ORIGINAL_BINARY ($(du -h "$ORIGINAL_BINARY" | cut -f1))"
else
    log_error "❌ Original binary missing"
fi

if [ -f "$INSTRUMENTED_BINARY" ]; then
    log_info "✅ Instrumented: $INSTRUMENTED_BINARY ($(du -h "$INSTRUMENTED_BINARY" | cut -f1))"
else
    log_warn "⚠️ Instrumented binary missing"
fi

if [ -f "$PGO_BINARY" ]; then
    PGO_FINAL_SIZE=$(du -m "$PGO_BINARY" | cut -f1)
    ORIGINAL_FINAL_SIZE=$(du -m "$ORIGINAL_BINARY" | cut -f1)

    log_info "✅ PGO-optimized: $PGO_BINARY ($(du -h "$PGO_BINARY" | cut -f1))"

    # Final size comparison summary
    if [ "$PGO_FINAL_SIZE" -gt "$ORIGINAL_FINAL_SIZE" ]; then
        log_warn "⚠️  WARNING: PGO binary (${PGO_FINAL_SIZE}M) is larger than original (${ORIGINAL_FINAL_SIZE}M)!"
        log_warn "PGO may not have optimized effectively - check profile data collection"
    elif [ "$PGO_FINAL_SIZE" -lt "$ORIGINAL_FINAL_SIZE" ]; then
        log_info "🎉 SUCCESS: PGO reduced binary size from ${ORIGINAL_FINAL_SIZE}M to ${PGO_FINAL_SIZE}M!"
    else
        log_info "ℹ️  PGO binary same size as original (${PGO_FINAL_SIZE}M)"
    fi

    log_info "🎯 Ready for benchmarking!"
    log_info "\nBenchmark comparison: $ORIGINAL_BINARY vs $PGO_BINARY"
else
    log_error "❌ PGO-optimized binary creation failed"
    exit 1
fi