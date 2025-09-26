#![allow(unused_imports, dead_code)]

fn main() {
    println!("cargo:rerun-if-env-changed=BENCH_MODEL_PATH");

    // Minimal build.rs - just report readiness
    // The benchmark itself will handle any necessary building
    println!("cargo:warning=📊 PGO benchmarks ready (dependencies will be checked at runtime)");
}
