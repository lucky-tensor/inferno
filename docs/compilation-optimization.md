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

BOLT (Binary Optimization and Layout Tool) optimizes the layout of compiled binaries for better cache and branch prediction performance.

### BOLT Workflow

1. **Build with PGO (recommended):**
   Follow the PGO steps above first.

2. **Profile the binary:**
   ```bash
   # Using perf
   perf record -e cycles:u -j any,u -- ./target/release/your_program input.txt

   # Convert perf data for BOLT
   perf2bolt -p perf.data -o perf.fdata ./target/release/your_program
   ```

3. **Optimize with BOLT:**
   ```bash
   llvm-bolt ./target/release/your_program -data=perf.fdata -reorder-blocks=ext-tsp \
     -reorder-functions=hfsort -split-functions -split-all-cold \
     -o ./target/release/your_program.bolt
   ```

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

## 5. Complete Optimization Workflow

Here's a complete script that combines all techniques:

```bash
#!/bin/bash

# Step 1: Build instrumented binary for PGO
echo "Building instrumented binary..."
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -C target-cpu=native" \
    cargo build --release

# Step 2: Run representative workloads
echo "Collecting profile data..."
./target/release/inferno --benchmark  # Replace with your typical workloads
./target/release/inferno --stress-test
./target/release/inferno < test_input.txt

# Step 3: Merge profile data
echo "Merging profile data..."
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Build final optimized binary
echo "Building PGO-optimized binary..."
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -C target-cpu=native -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release

# Step 5: Optional BOLT optimization
echo "Profiling for BOLT..."
perf record -e cycles:u -j any,u -- ./target/release/inferno typical_workload.txt
perf2bolt -p perf.data -o perf.fdata ./target/release/inferno

echo "Applying BOLT optimizations..."
llvm-bolt ./target/release/inferno -data=perf.fdata \
    -reorder-blocks=ext-tsp -reorder-functions=hfsort \
    -split-functions -split-all-cold \
    -o ./target/release/inferno.bolt

echo "Optimization complete! Final binary: ./target/release/inferno.bolt"
```

## 6. Measuring Performance Improvements

Always benchmark your optimizations:

```bash
# Baseline
cargo build --release
hyperfine './target/release/inferno benchmark'

# With native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
hyperfine './target/release/inferno benchmark'

# With PGO
# (after running PGO workflow)
hyperfine './target/release/inferno benchmark'
```

## 7. Tips and Considerations

- **Profile with representative workloads:** PGO is only as good as your training data
- **Compile time vs runtime tradeoffs:** These optimizations significantly increase build times
- **Binary size:** Some optimizations increase binary size
- **Reproducibility:** Document your optimization flags for consistent builds
- **CI/CD integration:** Consider separate optimization stages in your build pipeline

## 8. Environment Variables Reference

Common `RUSTFLAGS` for optimization:

```bash
# Maximum performance (long compile time)
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"

# Balanced performance/compile-time
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin"

# Size optimization
export RUSTFLAGS="-C target-cpu=native -C opt-level=s -C lto=thin -C strip=symbols"
```