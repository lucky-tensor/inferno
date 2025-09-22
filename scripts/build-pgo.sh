#!/bin/bash

# Profile-Guided Optimization (PGO) build script for Inferno
# This script builds an optimized binary using PGO with the specified entrypoint

set -e

# Configuration
PGO_DATA_DIR="/tmp/inferno-pgo-data"
ENTRYPOINT_COMMAND="inferno play --prompt \"what is python?\" --model-path ~/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors"
BINARY_NAME="inferno"
FAST_MODE=false
BOLT_ENABLED=false

# Inference-optimized CPU features for quantized operations and tokenization
INFERENCE_CPU_FEATURES="+avx2,+fma,+bmi2,+popcnt,+lzcnt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean up function
cleanup() {
    if [ -d "$PGO_DATA_DIR" ]; then
        log_info "Cleaning up PGO data directory..."
        rm -rf "$PGO_DATA_DIR"
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Try to find llvm-profdata from Rust toolchain first
    RUSTC_SYSROOT=$(rustc --print sysroot)
    RUST_LLVM_PROFDATA="$RUSTC_SYSROOT/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata"

    if [ -f "$RUST_LLVM_PROFDATA" ]; then
        LLVM_PROFDATA="$RUST_LLVM_PROFDATA"
        log_info "Using Rust's bundled llvm-profdata: $LLVM_PROFDATA"
    elif command -v llvm-profdata &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata"
        log_warn "Using system llvm-profdata (may have version mismatch issues)"
    else
        log_error "llvm-profdata not found. Please install LLVM tools or update Rust toolchain."
        log_error "Rust toolchain path checked: $RUST_LLVM_PROFDATA"
        exit 1
    fi

    if [ ! -f "Cargo.toml" ]; then
        log_error "Cargo.toml not found. Please run this script from the project root."
        exit 1
    fi

    # Check BOLT dependencies if enabled
    if [ "$BOLT_ENABLED" = true ]; then
        if ! command -v llvm-bolt &> /dev/null; then
            log_error "llvm-bolt not found. Please install LLVM BOLT tools or disable --bolt flag."
            exit 1
        fi

        if ! command -v perf &> /dev/null; then
            log_error "perf not found. Please install linux-perf-tools or disable --bolt flag."
            exit 1
        fi

        # Check for perf2bolt
        if ! command -v perf2bolt &> /dev/null; then
            log_warn "perf2bolt not found. Will try to use llvm-bolt directly."
        fi
    fi

    # Check if model file exists
    MODEL_PATH=$(echo "$ENTRYPOINT_COMMAND" | grep -o '~[^"]*\.safetensors' | head -1)
    if [ -n "$MODEL_PATH" ]; then
        EXPANDED_PATH="${MODEL_PATH/#\~/$HOME}"
        if [ ! -f "$EXPANDED_PATH" ]; then
            log_warn "Model file not found at: $EXPANDED_PATH"
            log_warn "Please ensure the model is downloaded before running PGO training."
        fi
    fi
}

# Step 1: Build instrumented binary
build_instrumented() {
    log_info "Step 1/4: Building instrumented binary for profiling..."

    # Clean any existing PGO data to avoid version conflicts
    if [ -d "$PGO_DATA_DIR" ]; then
        log_info "Cleaning existing PGO data..."
        rm -rf "$PGO_DATA_DIR"
    fi

    # Create PGO data directory
    mkdir -p "$PGO_DATA_DIR"

    # Build with profiling instrumentation optimized for inference workloads
    RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR -C target-cpu=native -C target-feature=$INFERENCE_CPU_FEATURES" \
        cargo build --release --bin "$BINARY_NAME"

    if [ $? -ne 0 ]; then
        log_error "Failed to build instrumented binary"
        exit 1
    fi

    log_info "Instrumented binary built successfully"
}

# Step 2: Run profiling workloads
run_profiling() {
    log_info "Step 2/4: Running comprehensive profiling workloads..."

    # Check if binary exists
    if [ ! -f "./target/release/$BINARY_NAME" ]; then
        log_error "Binary not found at ./target/release/$BINARY_NAME"
        exit 1
    fi

    # Extract model path for reuse
    MODEL_PATH=$(echo "$ENTRYPOINT_COMMAND" | grep -o '\~[^"]*\.safetensors' | head -1)
    EXPANDED_PATH=""
    if [ -n "$MODEL_PATH" ]; then
        EXPANDED_PATH="${MODEL_PATH/#\~/$HOME}"
        if [ ! -f "$EXPANDED_PATH" ]; then
            log_error "Model file not found at: $EXPANDED_PATH"
            log_error "Please ensure the model is downloaded before running PGO."
            exit 1
        fi
    fi

    # Profile different code paths with varying workload intensities
    log_info "Running diverse inference workloads for comprehensive profiling..."

    # 1. Quick startup/initialization patterns (cold start optimization)
    log_info "Profiling initialization patterns..."
    for i in {1..3}; do
        ./target/release/inferno --help >/dev/null 2>&1 || true
        ./target/release/inferno --version >/dev/null 2>&1 || true
    done

    if [ -n "$EXPANDED_PATH" ]; then
        # 2. Short prompts (common interactive usage)
        log_info "Profiling short interactive prompts..."

        # Inference-optimized profiling workloads targeting different computation patterns
        if [ "$FAST_MODE" = true ]; then
            # Fast mode: reduced workload focusing on core inference patterns
            SHORT_PROMPTS=(
                "hi"                    # Single token response
                "what is python?"       # Medium tokenization + reasoning
                "1+1"                   # Math reasoning
            )
        else
            # Comprehensive inference profiling targeting different computation patterns
            SHORT_PROMPTS=(
                "hi"                    # Minimal tokenization
                "hello"                 # Similar tokenization pattern
                "1+1"                   # Math/arithmetic reasoning
                "test"                  # Single word response
                "what is AI?"           # Question answering pattern
                "explain briefly"       # Instruction following
                "yes"                   # Single token
                "no"                    # Single token
                "thanks"               # Gratitude/social response
                "a"                    # Single character (tokenization edge case)
                "the"                  # Common word (frequent token)
                "python"               # Technical term (domain-specific token)
            )
        fi

        for prompt in "${SHORT_PROMPTS[@]}"; do
            log_info "  Running: '$prompt'"
            timeout 30s ./target/release/inferno play --prompt "$prompt" --model-path "$EXPANDED_PATH" 2>/dev/null || true
        done

        # Skip comprehensive profiling in fast mode
        if [ "$FAST_MODE" = false ]; then
            # 3. Medium-length prompts targeting different reasoning patterns
            log_info "Profiling medium-length prompts (attention mechanism stress test)..."
            MEDIUM_PROMPTS=(
                "what is python and why is it popular?"                           # Factual + reasoning
                "explain machine learning in simple terms"                        # Complex concept simplification
                "how do I write a for loop in rust with error handling?"         # Technical instruction
                "compare functional programming with object-oriented programming" # Comparison/contrast
                "describe the difference between stack and heap memory allocation" # Technical explanation
                "how does HTTP work and what are status codes?"                   # Multi-part technical question
                "what is the purpose of version control in software development?" # Domain-specific knowledge
                "solve this step by step: if x + 5 = 12, what is x?"            # Multi-step reasoning
            )

            for prompt in "${MEDIUM_PROMPTS[@]}"; do
                log_info "  Running: '$prompt'"
                timeout 45s ./target/release/inferno play --prompt "$prompt" --model-path "$EXPANDED_PATH" 2>/dev/null || true
            done

            # 4. Longer prompts (complex reasoning)
            log_info "Profiling complex reasoning prompts..."
            LONG_PROMPTS=(
                "explain the differences between procedural, object-oriented, and functional programming paradigms with examples"
                "walk me through the process of building a web application from scratch, including frontend, backend, and database considerations"
                "describe how neural networks work and explain the training process including backpropagation"
                "compare and contrast different sorting algorithms, their time complexities, and when to use each one"
            )

            for prompt in "${LONG_PROMPTS[@]}"; do
                log_info "  Running complex prompt: '${prompt:0:50}...'"
                timeout 60s ./target/release/inferno play --prompt "$prompt" --model-path "$EXPANDED_PATH" 2>/dev/null || true
            done

            # 5. Interactive session simulation (multiple rounds)
            log_info "Profiling interactive conversation patterns..."
            CONVERSATION_PROMPTS=(
                "hello, can you help me with coding?"
                "I need to learn about data structures"
                "start with arrays and lists"
                "now explain linked lists"
                "what about hash tables?"
                "give me an example in python"
                "thank you for the explanation"
            )

            for prompt in "${CONVERSATION_PROMPTS[@]}"; do
                log_info "  Conversation: '$prompt'"
                timeout 30s ./target/release/inferno play --prompt "$prompt" --model-path "$EXPANDED_PATH" 2>/dev/null || true
            done

            # 6. Inference stress patterns (quantized matrix operation hotpath optimization)
            log_info "Profiling quantized inference stress patterns..."

            # Repeated identical prompts (optimize for branch prediction and cache locality)
            for i in {1..5}; do
                timeout 20s ./target/release/inferno play --prompt "hello world" --model-path "$EXPANDED_PATH" 2>/dev/null || true
            done

            # Token count variations (stress different sequence lengths for attention)
            STRESS_PROMPTS=(
                "a"                                                    # 1 token
                "hello world"                                          # 2-3 tokens
                "the quick brown fox jumps over the lazy dog"          # ~10 tokens
                "this is a longer sentence with multiple words and concepts" # ~12 tokens
            )

            for prompt in "${STRESS_PROMPTS[@]}"; do
                # Run each pattern multiple times to hit hot paths
                for j in {1..3}; do
                    timeout 15s ./target/release/inferno play --prompt "$prompt" --model-path "$EXPANDED_PATH" 2>/dev/null || true
                done
            done

            # 7. Edge cases (empty/special characters)
            log_info "Profiling edge case inputs..."
            EDGE_CASES=(
                ""
                " "
                "?"
                "!!!"
                "123456789"
                "a very long single word: supercalifragilisticexpialidocious"
            )

            for prompt in "${EDGE_CASES[@]}"; do
                timeout 15s ./target/release/inferno play --prompt "$prompt" --model-path "$EXPANDED_PATH" 2>/dev/null || true
            done
        fi
    fi

    log_info "✅ Comprehensive profiling workloads completed"
    log_info "Total profile samples should now cover diverse execution paths"
}

# Step 3: Merge profile data
merge_profiles() {
    log_info "Step 3/4: Merging profile data..."

    # Check if profile data exists
    if [ ! -d "$PGO_DATA_DIR" ] || [ -z "$(ls -A "$PGO_DATA_DIR")" ]; then
        log_error "No profile data found in $PGO_DATA_DIR"
        exit 1
    fi

    # Merge all profile data
    "$LLVM_PROFDATA" merge -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR"

    if [ $? -ne 0 ]; then
        log_error "Failed to merge profile data"
        exit 1
    fi

    # Check if merged profile exists and has reasonable size
    if [ ! -f "$PGO_DATA_DIR/merged.profdata" ]; then
        log_error "Merged profile data not found"
        exit 1
    fi

    PROFILE_SIZE=$(stat -f%z "$PGO_DATA_DIR/merged.profdata" 2>/dev/null || stat -c%s "$PGO_DATA_DIR/merged.profdata" 2>/dev/null)
    if [ "$PROFILE_SIZE" -lt 1000 ]; then
        log_warn "Profile data seems small ($PROFILE_SIZE bytes). Results may not be optimal."
    fi

    log_info "Profile data merged successfully (${PROFILE_SIZE} bytes)"
}

# Step 4: Build optimized binary
build_optimized() {
    log_info "Step 4/4: Building PGO-optimized binary..."

    # Build with profile data optimized for inference hot paths (quantized ops, tokenization, attention)
    RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata -C target-cpu=native -C target-feature=$INFERENCE_CPU_FEATURES -C prefer-dynamic=no -Cllvm-args=-pgo-warn-missing-function -Cllvm-args=--enable-unsafe-fp-math" \
        cargo build --release --bin "$BINARY_NAME"

    if [ $? -ne 0 ]; then
        log_error "Failed to build optimized binary"
        exit 1
    fi

    log_info "PGO-optimized binary built successfully!"
}

# Step 5: Apply BOLT optimization (optional)
apply_bolt_optimization() {
    if [ "$BOLT_ENABLED" = false ]; then
        return
    fi

    log_info "Step 5/5: Applying BOLT optimization for better cache locality and branch prediction..."

    BOLT_DATA_DIR="/tmp/inferno-bolt-data"
    PERF_DATA_FILE="$BOLT_DATA_DIR/perf.data"
    BOLT_DATA_FILE="$BOLT_DATA_DIR/perf.fdata"
    ORIGINAL_BINARY="./target/release/$BINARY_NAME"
    BOLT_BINARY="./target/release/${BINARY_NAME}-bolt"

    # Create BOLT data directory
    mkdir -p "$BOLT_DATA_DIR"

    # Check if binary exists
    if [ ! -f "$ORIGINAL_BINARY" ]; then
        log_error "PGO binary not found at $ORIGINAL_BINARY"
        exit 1
    fi

    # Profile the PGO binary with perf for BOLT
    log_info "Profiling PGO binary for BOLT optimization..."

    # Extract model path for profiling
    MODEL_PATH=$(echo "$ENTRYPOINT_COMMAND" | grep -o '\~[^"]*\.safetensors' | head -1)
    EXPANDED_PATH=""
    if [ -n "$MODEL_PATH" ]; then
        EXPANDED_PATH="${MODEL_PATH/#\~/$HOME}"
    fi

    if [ -n "$EXPANDED_PATH" ] && [ -f "$EXPANDED_PATH" ]; then
        # Run representative inference workloads under perf
        log_info "Running inference workloads under perf profiling..."

        # Use perf record with branch sampling for BOLT
        perf record -e cycles:u -j any,u -o "$PERF_DATA_FILE" -- \
            "$ORIGINAL_BINARY" play --prompt "what is python and how is it used?" --model-path "$EXPANDED_PATH" 2>/dev/null || {
            log_error "Failed to profile binary with perf"
            return
        }

        # Convert perf data to BOLT format
        if command -v perf2bolt &> /dev/null; then
            log_info "Converting perf data to BOLT format..."
            perf2bolt -p "$PERF_DATA_FILE" -o "$BOLT_DATA_FILE" "$ORIGINAL_BINARY" || {
                log_error "Failed to convert perf data with perf2bolt"
                return
            }
        else
            # Try using llvm-bolt directly with perf.data
            log_info "Using perf data directly with llvm-bolt..."
            BOLT_DATA_FILE="$PERF_DATA_FILE"
        fi

        # Apply BOLT optimization
        log_info "Applying BOLT optimization to binary..."
        llvm-bolt "$ORIGINAL_BINARY" \
            -data="$BOLT_DATA_FILE" \
            -reorder-blocks=ext-tsp \
            -reorder-functions=hfsort \
            -split-functions=hot \
            -split-all-cold \
            -dyno-stats \
            -icf=1 \
            -use-gnu-stack \
            -o "$BOLT_BINARY" || {
            log_error "BOLT optimization failed"
            return
        }

        if [ -f "$BOLT_BINARY" ]; then
            log_info "✅ BOLT optimization completed successfully!"
            log_info "BOLT-optimized binary: $BOLT_BINARY"

            # Show optimization stats
            ORIGINAL_SIZE=$(stat -f%z "$ORIGINAL_BINARY" 2>/dev/null || stat -c%s "$ORIGINAL_BINARY" 2>/dev/null)
            BOLT_SIZE=$(stat -f%z "$BOLT_BINARY" 2>/dev/null || stat -c%s "$BOLT_BINARY" 2>/dev/null)
            ORIGINAL_MB=$((ORIGINAL_SIZE / 1024 / 1024))
            BOLT_MB=$((BOLT_SIZE / 1024 / 1024))

            log_info "Binary sizes:"
            log_info "  PGO:        ${ORIGINAL_MB}MB"
            log_info "  PGO+BOLT:   ${BOLT_MB}MB"

            # Replace main binary with BOLT version
            cp "$BOLT_BINARY" "$ORIGINAL_BINARY"
            log_info "Main binary updated with BOLT optimization"
        else
            log_error "BOLT binary not created successfully"
        fi

        # Cleanup BOLT data
        rm -rf "$BOLT_DATA_DIR"
    else
        log_error "Model file not found, cannot run BOLT profiling"
    fi
}

# Main execution
main() {
    log_info "Starting Profile-Guided Optimization (PGO) build for Inferno"
    log_info "Entrypoint: $ENTRYPOINT_COMMAND"
    echo

    check_dependencies
    build_instrumented
    run_profiling
    merge_profiles
    build_optimized
    apply_bolt_optimization

    echo
    log_info "✅ PGO optimization complete!"
    log_info "Optimized binary location: ./target/release/$BINARY_NAME"

    # Show binary size
    if [ -f "./target/release/$BINARY_NAME" ]; then
        BINARY_SIZE=$(stat -f%z "./target/release/$BINARY_NAME" 2>/dev/null || stat -c%s "./target/release/$BINARY_NAME" 2>/dev/null)
        BINARY_SIZE_MB=$((BINARY_SIZE / 1024 / 1024))
        log_info "Binary size: ${BINARY_SIZE_MB}MB"
    fi

    # Build baseline for comparison
    log_info "Building baseline (no PGO) for performance comparison..."
    cp "./target/release/$BINARY_NAME" "./target/release/${BINARY_NAME}-pgo"

    # Build standard release without PGO
    RUSTFLAGS="-C target-cpu=native" cargo build --release --bin "$BINARY_NAME" >/dev/null 2>&1
    cp "./target/release/$BINARY_NAME" "./target/release/${BINARY_NAME}-baseline"
    cp "./target/release/${BINARY_NAME}-pgo" "./target/release/$BINARY_NAME"

    echo
    log_info "Performance comparison binaries created:"
    log_info "  Baseline: ./target/release/${BINARY_NAME}-baseline"
    log_info "  PGO:      ./target/release/${BINARY_NAME}-pgo"
    echo
    log_info "To benchmark performance improvements:"

    # Extract just the play command part from ENTRYPOINT_COMMAND
    PLAY_COMMAND=$(echo "$ENTRYPOINT_COMMAND" | sed 's/^inferno //')

    log_info "  hyperfine --warmup 2 \\"
    log_info "    './target/release/${BINARY_NAME}-baseline $PLAY_COMMAND' \\"
    log_info "    './target/release/${BINARY_NAME}-pgo $PLAY_COMMAND'"
    echo

    # Show binary sizes
    if [ -f "./target/release/${BINARY_NAME}-baseline" ] && [ -f "./target/release/${BINARY_NAME}-pgo" ]; then
        BASELINE_SIZE=$(stat -f%z "./target/release/${BINARY_NAME}-baseline" 2>/dev/null || stat -c%s "./target/release/${BINARY_NAME}-baseline" 2>/dev/null)
        PGO_SIZE=$(stat -f%z "./target/release/${BINARY_NAME}-pgo" 2>/dev/null || stat -c%s "./target/release/${BINARY_NAME}-pgo" 2>/dev/null)
        BASELINE_MB=$((BASELINE_SIZE / 1024 / 1024))
        PGO_MB=$((PGO_SIZE / 1024 / 1024))

        log_info "Binary sizes:"
        log_info "  Baseline: ${BASELINE_MB}MB"
        log_info "  PGO:      ${PGO_MB}MB"

        if [ "$BASELINE_SIZE" -ne 0 ]; then
            SIZE_DIFF=$(((PGO_SIZE - BASELINE_SIZE) * 100 / BASELINE_SIZE))
            if [ "$SIZE_DIFF" -gt 0 ]; then
                log_info "  Size change: +${SIZE_DIFF}%"
            else
                SIZE_DIFF=$((SIZE_DIFF * -1))
                log_info "  Size change: -${SIZE_DIFF}%"
            fi
        fi
    fi
}

# Show help
show_help() {
    echo "Inferno PGO Build Script"
    echo
    echo "USAGE:"
    echo "    $0 [OPTIONS]"
    echo
    echo "OPTIONS:"
    echo "    -h, --help     Show this help message"
    echo "    -c, --clean    Clean build (remove target directory first)"
    echo "    -f, --fast     Fast profiling (reduced workload for development)"
    echo "    -b, --bolt     Apply BOLT optimization after PGO (requires llvm-bolt)"
    echo
    echo "DESCRIPTION:"
    echo "    Builds a Profile-Guided Optimization (PGO) binary for Inferno."
    echo "    Optionally applies BOLT (Binary Optimization and Layout Tool) for additional"
    echo "    cache locality and branch prediction improvements."
    echo "    The script will:"
    echo "      1. Build an instrumented binary"
    echo "      2. Run profiling workloads with the specified entrypoint"
    echo "      3. Merge profile data"
    echo "      4. Build the final optimized binary"
    echo "      5. Apply BOLT optimization (if --bolt flag is used)"
    echo
    echo "ENTRYPOINT:"
    echo "    $ENTRYPOINT_COMMAND"
    echo
    echo "REQUIREMENTS:"
    echo "    - LLVM tools (llvm-profdata)"
    echo "    - Model file at specified path"
    echo "    - Rust toolchain"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            log_info "Cleaning build directory..."
            cargo clean
            shift
            ;;
        -f|--fast)
            FAST_MODE=true
            log_info "Fast mode enabled - reduced profiling workload"
            shift
            ;;
        -b|--bolt)
            BOLT_ENABLED=true
            log_info "BOLT optimization enabled"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main