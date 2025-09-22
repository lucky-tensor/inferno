# Compilation Optimization Guide

This document covers various techniques for optimizing Rust compilation for specific hardware and workflows, ranging from basic CPU-specific optimizations to advanced profile-guided optimization (PGO) and binary layout optimization.

## 1. CPU-Specific Optimizations

The most straightforward way to optimize for your specific hardware is to enable CPU-specific instructions and optimizations.

### Basic Native Compilation

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables all CPU instructions supported by your build machine, including:
- AVX, AVX2, AVX-512
- BMI (Bit Manipulation Instructions)
- FMA (Fused Multiply-Add)
- Other CPU-specific extensions

### Manual CPU Targeting

For more control, specify exact CPU architectures:

```bash
# AMD Zen 4 architecture
RUSTFLAGS="-C target-cpu=znver4" cargo build --release

# Intel Skylake architecture
RUSTFLAGS="-C target-cpu=skylake" cargo build --release

# Generic modern x86-64 with common extensions
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

### Fine-Grained Feature Control

Enable specific CPU features manually:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma,+bmi2" cargo build --release
```

## 2. Profile-Guided Optimization (PGO)

PGO analyzes actual program execution to guide compiler optimizations. This typically yields 5-20% performance improvements.

### Step-by-Step PGO Workflow

1. **Build an instrumented binary:**
   ```bash
   RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
   ```

2. **Run representative workloads:**
   Execute your program with typical inputs to collect profile data:
   ```bash
   ./target/release/your_program < typical_input1.txt
   ./target/release/your_program --benchmark-mode
   ./target/release/your_program --stress-test
   ```

3. **Merge profile data:**
   ```bash
   llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
   ```

4. **Rebuild with profile data:**
   ```bash
   RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
       cargo build --release
   ```

### PGO with Cargo Configuration

Add to your `Cargo.toml` for reproducible PGO builds:

```toml
[profile.pgo-gen]
inherits = "release"
debug = 1  # Needed for profiling

[profile.pgo-use]
inherits = "release"
```

Then use:
```bash
# Step 1: Generate instrumented build
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --profile=pgo-gen

# Steps 2-3: Run workloads and merge profiles (same as above)

# Step 4: Final optimized build
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --profile=pgo-use
```

## 3. Binary Layout Optimization with BOLT

BOLT (Binary Optimization and Layout Tool) optimizes the layout of compiled binaries for better cache and branch prediction performance. BOLT is particularly effective for inference workloads with complex call patterns and branchy code.

### BOLT Benefits for ML Inference

- **Function Layout Optimization**: Groups frequently called functions together in memory
- **Hot Function Clustering**: Places related inference functions in the same cache lines
- **Cold Code Splitting**: Moves error handling and initialization code away from hot paths
- **Branch Prediction**: Optimizes branch target placement for better prediction
- **Cache Locality**: Improves instruction cache usage for sequential operations

### BOLT Installation

```bash
# Ubuntu/Debian
sudo apt install llvm linux-perf-tools

# Arch Linux
sudo pacman -S llvm linux-perf

# Check installation
llvm-bolt --version
perf --version
```

### BOLT Workflow

1. **Build with PGO (recommended):**
   Follow the PGO steps above first.

2. **Profile the binary:**
   ```bash
   # Using perf with branch sampling (essential for BOLT)
   perf record -e cycles:u -j any,u -- ./target/release/your_program input.txt

   # Convert perf data for BOLT
   perf2bolt -p perf.data -o perf.fdata ./target/release/your_program
   ```

3. **Optimize with BOLT:**
   ```bash
   llvm-bolt ./target/release/your_program \
     -data=perf.fdata \
     -reorder-blocks=ext-tsp \      # Optimal block layout algorithm
     -reorder-functions=hfsort \    # Hot function clustering
     -split-functions=hot \         # Split hot/cold within functions
     -split-all-cold \             # Aggressive cold code splitting
     -dyno-stats \                 # Show optimization statistics
     -icf=1 \                      # Identical code folding
     -use-gnu-stack \              # GNU stack compatibility
     -o ./target/release/your_program.bolt
   ```

### BOLT Flags Explained

- **`-reorder-blocks=ext-tsp`**: Uses the extended Traveling Salesman Problem algorithm for optimal basic block ordering within functions
- **`-reorder-functions=hfsort`**: Sorts functions by hotness and groups related functions together
- **`-split-functions=hot`**: Splits functions into hot and cold parts, keeping hot parts together
- **`-split-all-cold`**: Aggressively moves all cold code to separate sections
- **`-dyno-stats`**: Provides detailed statistics about the optimizations applied
- **`-icf=1`**: Enables identical code folding to reduce binary size

## 4. Advanced Cargo Profile Configuration

### Maximum Optimization Profile

Add to `Cargo.toml`:

```toml
[profile.release-max]
inherits = "release"
lto = "fat"                # Full link-time optimization
codegen-units = 1          # Single codegen unit for better optimization
opt-level = 3              # Maximum optimization level
panic = "abort"            # Smaller binaries, faster execution
strip = true               # Remove debug symbols
```

Build with:
```bash
cargo build --profile=release-max
```

### Size-Optimized Profile

For smaller binaries:

```toml
[profile.release-size]
inherits = "release"
opt-level = "s"            # Optimize for size
lto = "thin"               # Thin LTO (good size/compile-time balance)
codegen-units = 1
panic = "abort"
strip = true
```

## 5. Inference-Specific Optimizations

For ML inference workloads like the Inferno inference crate, additional optimizations target quantized operations, tokenization, and attention mechanisms.

### CPU Features for Inference

Enable specific CPU features optimized for neural network operations:

```bash
# Inference-optimized CPU features
INFERENCE_CPU_FEATURES="+avx2,+fma,+bmi2,+popcnt,+lzcnt"

RUSTFLAGS="-C target-cpu=native -C target-feature=$INFERENCE_CPU_FEATURES" \
    cargo build --release
```

**Feature explanations:**
- **AVX2**: 256-bit SIMD for matrix operations and vectorized computations
- **FMA**: Fused multiply-add instructions crucial for neural network layers
- **BMI2**: Bit manipulation instructions for quantization/dequantization operations
- **POPCNT**: Population count for sparse operations and bit counting
- **LZCNT**: Leading zero count for numerical computations and normalization

### LLVM Flags for ML Workloads

```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=$INFERENCE_CPU_FEATURES \
          -Cllvm-args=-pgo-warn-missing-function \
          -Cllvm-args=--enable-unsafe-fp-math"
```

- **`--enable-unsafe-fp-math`**: Enables aggressive floating-point optimizations that trade strict IEEE compliance for performance (safe for most ML workloads)

## 6. Automated PGO+BOLT Script

The Inferno project includes an automated script that handles the complete optimization pipeline with inference-specific workloads:

### Basic Usage

```bash
# Standard PGO optimization
./scripts/build-pgo.sh

# PGO with BOLT optimization (maximum performance)
./scripts/build-pgo.sh --bolt

# Fast development build with BOLT
./scripts/build-pgo.sh --fast --bolt

# Clean build with full optimization
./scripts/build-pgo.sh --clean --bolt
```

### Script Features

- **Inference-focused profiling**: Targets quantized LLaMA operations, tokenization, and attention mechanisms
- **Comprehensive workload coverage**: Tests various prompt lengths, reasoning patterns, and edge cases
- **Automatic BOLT integration**: Handles perf profiling and BOLT optimization seamlessly
- **Performance comparison**: Creates baseline and optimized binaries for benchmarking
- **Dependency checking**: Validates required tools (llvm-profdata, llvm-bolt, perf)

### Expected Performance Improvements

| Optimization Level | Typical Improvement | Use Case |
|-------------------|-------------------|----------|
| Baseline Release | 0% | Standard cargo build --release |
| + Native CPU | 2-5% | CPU-specific instructions |
| + PGO | 5-15% | Profile-guided optimization |
| + PGO + BOLT | 10-25% | Maximum optimization |

### Inference-Specific Profiling Workloads

The script includes specialized workloads that exercise different computational patterns:

1. **Tokenization patterns**: Single characters, common words, technical terms
2. **Sequence length variations**: 1, 3, 10, and 12+ token sequences to stress attention mechanisms
3. **Reasoning patterns**: Factual questions, multi-step reasoning, technical explanations
4. **Repetitive workloads**: Identical prompts for branch prediction and cache locality optimization
5. **Edge cases**: Empty strings, special characters, and boundary conditions

## 7. Manual Complete Optimization Workflow

For advanced users who want full control:

```bash
#!/bin/bash

# Inference-optimized CPU features
INFERENCE_CPU_FEATURES="+avx2,+fma,+bmi2,+popcnt,+lzcnt"

# Step 1: Build instrumented binary for PGO
echo "Building instrumented binary..."
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -C target-cpu=native -C target-feature=$INFERENCE_CPU_FEATURES" \
    cargo build --release

# Step 2: Run representative inference workloads
echo "Collecting profile data..."
./target/release/inferno play --prompt "what is python?" --model-path ~/.inferno/models/model.safetensors
./target/release/inferno play --prompt "explain machine learning" --model-path ~/.inferno/models/model.safetensors
./target/release/inferno play --prompt "1+1" --model-path ~/.inferno/models/model.safetensors
# Add more diverse workloads targeting different code paths...

# Step 3: Merge profile data
echo "Merging profile data..."
$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata merge \
    -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Build PGO-optimized binary
echo "Building PGO-optimized binary..."
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -C target-cpu=native -C target-feature=$INFERENCE_CPU_FEATURES \
          -Cllvm-args=-pgo-warn-missing-function -Cllvm-args=--enable-unsafe-fp-math" \
    cargo build --release

# Step 5: Apply BOLT optimization
echo "Profiling for BOLT..."
perf record -e cycles:u -j any,u -- ./target/release/inferno play --prompt "detailed explanation" --model-path ~/.inferno/models/model.safetensors

echo "Converting perf data..."
perf2bolt -p perf.data -o perf.fdata ./target/release/inferno

echo "Applying BOLT optimizations..."
llvm-bolt ./target/release/inferno \
    -data=perf.fdata \
    -reorder-blocks=ext-tsp \
    -reorder-functions=hfsort \
    -split-functions=hot \
    -split-all-cold \
    -dyno-stats \
    -icf=1 \
    -use-gnu-stack \
    -o ./target/release/inferno.bolt

echo "Optimization complete! Final binary: ./target/release/inferno.bolt"
```

## 8. Measuring Performance Improvements

Always benchmark your optimizations with realistic workloads:

### Automated Comparison

The PGO script automatically creates comparison binaries:

```bash
# Run the PGO script with BOLT
./scripts/build-pgo.sh --bolt

# The script will output a hyperfine command like:
hyperfine --warmup 2 \
    './target/release/inferno-baseline play --prompt "what is python?" --model-path ~/.inferno/models/model.safetensors' \
    './target/release/inferno-pgo play --prompt "what is python?" --model-path ~/.inferno/models/model.safetensors'
```

### Manual Benchmarking

```bash
# Baseline release build
cargo build --release
cp ./target/release/inferno ./target/release/inferno-baseline

# PGO+BOLT optimized build (after optimization)
cp ./target/release/inferno ./target/release/inferno-optimized

# Benchmark with realistic inference workloads
hyperfine --warmup 3 --runs 10 \
    './target/release/inferno-baseline play --prompt "explain neural networks" --model-path ~/.inferno/models/model.safetensors' \
    './target/release/inferno-optimized play --prompt "explain neural networks" --model-path ~/.inferno/models/model.safetensors'

# Test different prompt lengths
hyperfine --warmup 2 \
    './target/release/inferno-baseline play --prompt "hi" --model-path ~/.inferno/models/model.safetensors' \
    './target/release/inferno-optimized play --prompt "hi" --model-path ~/.inferno/models/model.safetensors'
```

### Performance Analysis Tools

```bash
# Profile the optimized binary to verify improvements
perf stat -e cycles,instructions,cache-misses,branch-misses \
    ./target/release/inferno play --prompt "test" --model-path ~/.inferno/models/model.safetensors

# Measure memory usage
/usr/bin/time -v ./target/release/inferno play --prompt "test" --model-path ~/.inferno/models/model.safetensors
```

## 9. Tips and Considerations

### PGO Best Practices
- **Representative workloads**: Use diverse prompts that cover your typical usage patterns
- **Model coverage**: Profile with the actual models you'll deploy in production
- **Multiple runs**: Include both cold start and warm-up scenarios in profiling
- **Edge cases**: Include error conditions and boundary cases in your profiling workloads

### BOLT Considerations
- **Profile data quality**: BOLT effectiveness depends on good branch sampling data from perf
- **Binary size**: BOLT typically increases binary size due to code layout changes
- **Compatibility**: Ensure your deployment environment supports the BOLT-optimized binary

### Build Time vs Performance
- **Development builds**: Use `--fast` flag for quicker profiling during development
- **Production builds**: Use full profiling with BOLT for maximum performance
- **CI/CD integration**: Consider caching optimized binaries and only rebuilding when core inference code changes

### Deployment Considerations
- **CPU compatibility**: Native optimizations may not work across different CPU architectures
- **Containerization**: Be careful with CPU-specific optimizations in containers
- **Fallback binaries**: Consider building both optimized and portable versions

## 10. Environment Variables Reference

### Inference-Optimized RUSTFLAGS

```bash
# Maximum inference performance (long compile time)
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma,+bmi2,+popcnt,+lzcnt -C opt-level=3 -C lto=fat -C codegen-units=1 -Cllvm-args=--enable-unsafe-fp-math"

# Balanced performance/compile-time for inference
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma -C opt-level=3 -C lto=thin"

# Development inference builds
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2 -C opt-level=2"

# Size-optimized inference builds
export RUSTFLAGS="-C target-cpu=native -C opt-level=s -C lto=thin -C strip=symbols"
```

### CPU Feature Detection

```bash
# Check what features your CPU supports
rustc --print target-features --target x86_64-unknown-linux-gnu | grep -E "(avx|fma|bmi)"

# Verify CPU capabilities
lscpu | grep -E "(avx|fma|bmi)"

# Check what features are being used in your build
RUSTFLAGS="-C target-cpu=native -v" cargo build --release 2>&1 | grep "target-feature"
```

### Troubleshooting

```bash
# If BOLT fails, check for debug symbols
readelf -S ./target/release/inferno | grep debug

# Verify perf data quality
perf report -i perf.data --stdio | head -20

# Check LLVM version compatibility
llvm-bolt --version
rustc --version --verbose | grep LLVM
```