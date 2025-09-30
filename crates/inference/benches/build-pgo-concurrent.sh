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
MODEL_PATH="${BENCH_MODEL_PATH:-$HOME/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0}"
CONCURRENT_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference"

# Check prerequisites
if [ ! -d "$MODEL_PATH" ]; then
    log_error "Model directory not found at: $MODEL_PATH"
    log_error "Set BENCH_MODEL_PATH environment variable to model directory"
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
# Extract LLVM version from rustc
RUSTC_LLVM_VERSION=$(rustc --version --verbose | grep LLVM | sed 's/LLVM version: \([0-9]*\).*/\1/')
log_info "Rust compiler uses LLVM version: $RUSTC_LLVM_VERSION"

# Try to find a compatible llvm-profdata version
LLVM_PROFDATA=""

# First, try to use rustc's bundled llvm-profdata (most reliable)
RUSTC_SYSROOT=$(rustc --print sysroot)
RUSTC_LLVM_PROFDATA="$RUSTC_SYSROOT/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata"
if [ -f "$RUSTC_LLVM_PROFDATA" ]; then
    log_info "Found rustc's bundled llvm-profdata (LLVM $RUSTC_LLVM_VERSION)"
    LLVM_PROFDATA="$RUSTC_LLVM_PROFDATA"
else
    log_warn "Rustc's bundled llvm-profdata not found, trying system versions"

    # Try exact version match
    if command -v "llvm-profdata-$RUSTC_LLVM_VERSION" &> /dev/null; then
        log_info "Found exact LLVM version match: llvm-profdata-$RUSTC_LLVM_VERSION"
        LLVM_PROFDATA="llvm-profdata-$RUSTC_LLVM_VERSION"
    else
        log_warn "No exact LLVM version match found for version $RUSTC_LLVM_VERSION"

        # Check for compatible versions (within 1-2 versions)
        for version in $((RUSTC_LLVM_VERSION-1)) $((RUSTC_LLVM_VERSION+1)) $((RUSTC_LLVM_VERSION-2)) 20 19 18 17; do
            if command -v "llvm-profdata-$version" &> /dev/null; then
                log_warn "Using potentially compatible llvm-profdata-$version (rustc uses LLVM $RUSTC_LLVM_VERSION)"
                LLVM_PROFDATA="llvm-profdata-$version"
                break
            fi
        done
    fi

    # Fallback to default llvm-profdata
    if [ -z "$LLVM_PROFDATA" ]; then
        if command -v llvm-profdata &> /dev/null; then
            LLVM_PROFDATA="llvm-profdata"
            PROFDATA_VERSION=$($LLVM_PROFDATA --version 2>/dev/null | head -n1 || echo "unknown")
            log_warn "Using system llvm-profdata (version: $PROFDATA_VERSION)"
            log_warn "This may cause version mismatch with rustc LLVM $RUSTC_LLVM_VERSION"
        else
            log_error "No compatible llvm-profdata found"
            log_error "Try: rustup component add llvm-tools-preview"
            exit 1
        fi
    fi
fi

log_info "Selected: $LLVM_PROFDATA"

log_info "Using existing binary: $(du -h "$CONCURRENT_BINARY" | cut -f1)"

# Clean and prepare PGO data directory
rm -rf "$PGO_DATA_DIR"
mkdir -p "$PGO_DATA_DIR"

# Save original binary for benchmarking comparison BEFORE any cleaning
# Use a temporary location that won't be affected by cargo clean
ORIGINAL_BINARY_TEMP="$PGO_DATA_DIR/concurrent_inference.original"
ORIGINAL_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference.original"
ORIGINAL_SHA_BEFORE=$(sha256sum "$CONCURRENT_BINARY" | cut -d' ' -f1)
cp "$CONCURRENT_BINARY" "$ORIGINAL_BINARY_TEMP"
log_info "Saved original binary to temp location: $ORIGINAL_BINARY_TEMP (SHA: $ORIGINAL_SHA_BEFORE)"

# Step 1: Build instrumented version for profiling
log_info "Step 1/3: Building instrumented version (dependencies will be rebuilt)..."
log_info "  This rebuilds dependencies due to RUSTFLAGS changes - it's unavoidable"

cd crates/inference
# Clean to ensure no cached artifacts interfere with PGO builds
log_info "  Cleaning cargo cache to ensure fresh build with new RUSTFLAGS"
cargo clean --release

# Build instrumented version using dedicated PGO profile
log_info "  Using pgo-gen profile (matches release but with profiling instrumentation)"
log_info "  RUSTFLAGS: -Cprofile-generate=$PGO_DATA_DIR"
RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR" \
    cargo build --release --example concurrent_inference --features examples
cd "$WORKSPACE_ROOT"

# Save instrumented binary for debugging/analysis (built with pgo-gen profile)
INSTRUMENTED_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference.instrumented"
RELEASE_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference"
cp "$RELEASE_BINARY" "$INSTRUMENTED_BINARY"

# Validate instrumented binary size (PGO instrumentation adds significant but expected overhead)
INSTRUMENTED_SIZE=$(du -m "$INSTRUMENTED_BINARY" | cut -f1)
if [ -f "$ORIGINAL_BINARY_TEMP" ]; then
    ORIGINAL_SIZE=$(du -m "$ORIGINAL_BINARY_TEMP" | cut -f1)
    if [ "$ORIGINAL_SIZE" -gt 0 ]; then
        OVERHEAD_RATIO=$((INSTRUMENTED_SIZE * 100 / ORIGINAL_SIZE))
        if [ "$INSTRUMENTED_SIZE" -gt $((ORIGINAL_SIZE * 8)) ]; then
            log_warn "Instrumented binary seems unusually large: ${INSTRUMENTED_SIZE}M vs original ${ORIGINAL_SIZE}M (${OVERHEAD_RATIO}% of original)"
            log_warn "This may indicate an issue with the build process"
        else
            log_info "  Instrumented binary size: ${INSTRUMENTED_SIZE}M vs original ${ORIGINAL_SIZE}M (${OVERHEAD_RATIO}% overhead is normal for PGO instrumentation)"
        fi
    else
        log_warn "Original binary size is 0 - cannot calculate overhead ratio"
        log_info "  Instrumented binary size: ${INSTRUMENTED_SIZE}M"
    fi
else
    log_warn "Original binary temp file missing - cannot calculate overhead ratio"
    log_info "  Instrumented binary size: ${INSTRUMENTED_SIZE}M"
fi
log_info "  Instrumented binary saved: $INSTRUMENTED_BINARY"
log_info "  Binary ready for profiling workloads"

# Step 2: Run profiling workloads
log_info "Step 2/3: Running profiling workloads..."
log_info "Profile data will be saved to: $PGO_DATA_DIR"

# Check that profile directory is writable
if [ ! -w "$PGO_DATA_DIR" ]; then
    log_error "Profile data directory not writable: $PGO_DATA_DIR"
    exit 1
fi

# Focused profiling tests
TESTS=("1 hi" "5 test" "10 quick" "25 brief")

for test in "${TESTS[@]}"; do
    read -r concurrency prompt <<< "$test"
    log_info "  Profiling: $concurrency workers with '$prompt'"

    # Check profile data before run
    PROFILE_COUNT_BEFORE=$(find "$PGO_DATA_DIR" -name "*.profraw" 2>/dev/null | wc -l)

    # Use the instrumented binary built with release profile
    PROFILE_OUTPUT=$(timeout 15s "$RELEASE_BINARY" \
        --prompt "$prompt" \
        --model-path "$MODEL_PATH" \
        --concurrent "$concurrency" 2>&1) || PROFILE_EXIT_CODE=$?

    # Check if profile data was generated
    PROFILE_COUNT_AFTER=$(find "$PGO_DATA_DIR" -name "*.profraw" 2>/dev/null | wc -l)
    PROFILE_DATA_SIZE=$(du -sh "$PGO_DATA_DIR" 2>/dev/null | cut -f1)

    if [ "$PROFILE_COUNT_AFTER" -gt "$PROFILE_COUNT_BEFORE" ]; then
        log_info "    ✅ Generated profile data ($PROFILE_DATA_SIZE total)"
    else
        log_warn "    ⚠️  No new profile data generated"
        if [ -n "$PROFILE_OUTPUT" ]; then
            log_warn "    Output: ${PROFILE_OUTPUT:0:200}..."
        fi
    fi
done

# Final profile data check
TOTAL_PROFILE_FILES=$(find "$PGO_DATA_DIR" -name "*.profraw" 2>/dev/null | wc -l)
TOTAL_PROFILE_SIZE=$(du -sh "$PGO_DATA_DIR" 2>/dev/null | cut -f1)
log_info "Total profile data collected: $TOTAL_PROFILE_FILES files ($TOTAL_PROFILE_SIZE)"

if [ "$TOTAL_PROFILE_FILES" -eq 0 ]; then
    log_error "❌ CRITICAL: No profile data collected!"
    log_error "   This explains why PGO binaries are identical - no optimization data available"
    log_error "   Check that the instrumented binary is working and model path is correct"
    exit 1
fi

# Step 3: Build PGO-optimized binary
log_info "Step 3/3: Creating PGO-optimized binary..."
log_info "Merging profile data with $LLVM_PROFDATA..."

# Try to merge profile data with fallback strategies
MERGE_SUCCESS=false
if "$LLVM_PROFDATA" merge -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR" 2>/tmp/profdata-error.log; then
    log_info "Profile data merged successfully"
    MERGE_SUCCESS=true
elif "$LLVM_PROFDATA" merge --failure-mode=warn -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR" 2>/tmp/profdata-error.log; then
    log_warn "Profile data merged with warnings (some data may be skipped)"
    MERGE_SUCCESS=true
else
    log_error "Failed to merge profile data. Error details:"
    cat /tmp/profdata-error.log
    log_error "This is likely due to LLVM version mismatch between Rust and llvm-profdata"
    log_error "Rust uses $(rustc --version --verbose | grep LLVM), but $LLVM_PROFDATA expects older format"
    log_error "Continuing without PGO optimization..."
    MERGE_SUCCESS=false
fi

# Verify merged profile data
if [ "$MERGE_SUCCESS" = true ] && [ -f "$PGO_DATA_DIR/merged.profdata" ]; then
    MERGED_SIZE=$(du -h "$PGO_DATA_DIR/merged.profdata" | cut -f1)
    if [ -s "$PGO_DATA_DIR/merged.profdata" ]; then
        log_info "✅ Merged profile data is valid ($MERGED_SIZE)"
    else
        log_error "❌ Merged profile data file is empty!"
        log_error "   This explains why PGO binaries are identical - no optimization data to use"
        MERGE_SUCCESS=false
    fi
else
    log_error "❌ No valid merged profile data available"
    MERGE_SUCCESS=false
fi

if [ "$MERGE_SUCCESS" = false ]; then
    log_error "❌ CRITICAL: No usable profile data for PGO optimization!"
    log_error "   Creating empty profdata file - PGO build will be identical to original"
    # Create empty profdata file to avoid build failure
    touch "$PGO_DATA_DIR/merged.profdata"
fi

cd crates/inference
log_info "  Building final PGO-optimized binary (recompiling with profile data)..."

# Clean again to ensure the PGO build uses the new RUSTFLAGS
log_info "  Cleaning cargo cache again for PGO-optimized build"
cargo clean --release

# Build PGO-optimized version using dedicated PGO profile with full optimizations
log_info "  Using pgo-use profile (release + PGO optimizations)"
log_info "  RUSTFLAGS: -Cprofile-use=$PGO_DATA_DIR/merged.profdata"
RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata" \
    cargo build --release --example concurrent_inference --features examples
cd "$WORKSPACE_ROOT"

log_info "  PGO-optimized build complete"

# Save PGO-optimized binary (built with release profile)
PGO_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference.pgo"
RELEASE_PGO_BINARY="$WORKSPACE_ROOT/target/release/examples/concurrent_inference"

# Verify that the PGO build actually produced a different binary
if [ -f "$RELEASE_PGO_BINARY" ]; then
    PGO_SHA_BEFORE_COPY=$(sha256sum "$RELEASE_PGO_BINARY" | cut -d' ' -f1)
    log_info "PGO build produced binary with SHA: $PGO_SHA_BEFORE_COPY"

    if [ "$PGO_SHA_BEFORE_COPY" = "$ORIGINAL_SHA_BEFORE" ]; then
        log_error "❌ CRITICAL: PGO build produced identical binary to original!"
        log_error "   Original SHA: $ORIGINAL_SHA_BEFORE"
        log_error "   PGO SHA:      $PGO_SHA_BEFORE_COPY"
        log_error "   This indicates RUSTFLAGS for PGO optimization didn't take effect"
        log_error "   Possible causes: cargo cache, RUSTFLAGS not applied, or profile data issues"
    fi

    cp "$RELEASE_PGO_BINARY" "$PGO_BINARY"
    log_info "Saved PGO binary: $PGO_BINARY (SHA: $PGO_SHA_BEFORE_COPY)"
else
    log_error "❌ PGO build failed - no binary produced at $RELEASE_PGO_BINARY"
    exit 1
fi

# Validate PGO binary size (should be similar or smaller than original)
PGO_SIZE=$(du -m "$PGO_BINARY" | cut -f1)
if [ -f "$ORIGINAL_BINARY_TEMP" ]; then
    ORIGINAL_SIZE=$(du -m "$ORIGINAL_BINARY_TEMP" | cut -f1)

    if [ "$PGO_SIZE" -gt "$ORIGINAL_SIZE" ]; then
        log_warn "⚠️  PGO binary is larger than original: ${PGO_SIZE}M vs ${ORIGINAL_SIZE}M"
        log_warn "This may indicate PGO didn't optimize effectively or there's a build issue"
        log_warn "Check if profile data was collected properly during the profiling phase"
    elif [ "$PGO_SIZE" -lt "$ORIGINAL_SIZE" ]; then
        log_info "✅ PGO optimization successful: ${PGO_SIZE}M vs ${ORIGINAL_SIZE}M (smaller!)"
    else
        log_info "ℹ️  PGO binary same size as original: ${PGO_SIZE}M"
    fi
else
    log_warn "Cannot compare PGO binary size - original binary temp missing"
    log_info "PGO binary size: ${PGO_SIZE}M"
fi

log_info "  PGO-optimized binary saved: $PGO_BINARY"

# Also keep legacy name for compatibility
cp "$RELEASE_PGO_BINARY" "$WORKSPACE_ROOT/target/release/examples/concurrent_inference-pgo"

# Restore original binary from temp location for comparison
if [ -f "$ORIGINAL_BINARY_TEMP" ]; then
    cp "$ORIGINAL_BINARY_TEMP" "$ORIGINAL_BINARY"
    log_info "Restored original binary: $ORIGINAL_BINARY"
else
    log_warn "Original binary temp file missing: $ORIGINAL_BINARY_TEMP"
fi

# Cleanup (but keep the original binary now that it's restored)
rm -rf "$PGO_DATA_DIR"

# Verify binaries are different using SHA checksums
log_info "\n=== Binary Verification ==="
if [ -f "$ORIGINAL_BINARY" ] && [ -f "$PGO_BINARY" ]; then
    ORIGINAL_SHA=$(sha256sum "$ORIGINAL_BINARY" | cut -d' ' -f1)
    PGO_SHA=$(sha256sum "$PGO_BINARY" | cut -d' ' -f1)

    if [ "$ORIGINAL_SHA" = "$PGO_SHA" ]; then
        log_error "❌ CRITICAL: Original and PGO binaries are identical (SHA: $ORIGINAL_SHA)!"
        log_error "   PGO optimization did not produce a different binary"
        log_error "   This indicates the PGO process failed - benchmarks would be meaningless"
        exit 1
    else
        log_info "✅ Binary verification passed: Original and PGO binaries are different"
        log_info "   Original SHA: $ORIGINAL_SHA"
        log_info "   PGO SHA:      $PGO_SHA"
    fi
else
    log_error "❌ Cannot verify binaries - one or both are missing"
fi

# Verify and summarize results
log_info "\n=== PGO Build Summary ==="
log_info "All binary versions created for benchmarking comparison:"

if [ -f "$ORIGINAL_BINARY" ]; then
    log_info "✅ Original:     $ORIGINAL_BINARY ($(du -h "$ORIGINAL_BINARY" | cut -f1))"
else
    log_error "❌ Original binary missing"
fi

# Note: Instrumented binary is intentionally cleaned up during the build process
# and is not needed for benchmarking, so we don't report it as missing

if [ -f "$PGO_BINARY" ]; then
    PGO_FINAL_SIZE=$(du -m "$PGO_BINARY" | cut -f1)

    log_info "✅ PGO-optimized: $PGO_BINARY ($(du -h "$PGO_BINARY" | cut -f1))"

    # Final size comparison summary
    if [ -f "$ORIGINAL_BINARY" ]; then
        ORIGINAL_FINAL_SIZE=$(du -m "$ORIGINAL_BINARY" | cut -f1)

        if [ "$PGO_FINAL_SIZE" -gt "$ORIGINAL_FINAL_SIZE" ]; then
            log_warn "⚠️  WARNING: PGO binary (${PGO_FINAL_SIZE}M) is larger than original (${ORIGINAL_FINAL_SIZE}M)!"
            log_warn "PGO may not have optimized effectively - check profile data collection"
        elif [ "$PGO_FINAL_SIZE" -lt "$ORIGINAL_FINAL_SIZE" ]; then
            log_info "🎉 SUCCESS: PGO reduced binary size from ${ORIGINAL_FINAL_SIZE}M to ${PGO_FINAL_SIZE}M!"
        else
            log_info "ℹ️  PGO binary same size as original (${PGO_FINAL_SIZE}M)"
        fi
    else
        log_warn "Cannot compare final sizes - original binary missing"
        log_info "PGO binary final size: ${PGO_FINAL_SIZE}M"
    fi

    log_info "🎯 Ready for benchmarking!"
    log_info "\nBenchmark comparison: $ORIGINAL_BINARY vs $PGO_BINARY"
else
    log_error "❌ PGO-optimized binary creation failed"
    exit 1
fi