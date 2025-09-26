#![allow(clippy::expect_used, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// Configuration for PGO comparison benchmarks
struct PGOBenchmarkConfig {
    model_path: PathBuf,
    original_binary: PathBuf,
    pgo_binary: PathBuf,
}

impl PGOBenchmarkConfig {
    fn new() -> Option<Self> {
        let model_path = std::env::var("BENCH_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                PathBuf::from(format!(
                    "{}/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0",
                    home
                ))
            });

        // Find workspace root using CARGO_MANIFEST_DIR
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR should be set when running via cargo");

        // pgo-benchmarks is at: workspace/crates/inference/pgo-benchmarks
        // So workspace root is: manifest_dir + "../../.."
        let workspace_root = PathBuf::from(&manifest_dir)
            .parent()  // crates/inference/
            .and_then(|p| p.parent())  // crates/
            .and_then(|p| p.parent())  // workspace root
            .map(|p| p.to_path_buf())
            .expect("Could not find workspace root from manifest directory");

        // Expected binary locations (built by PGO script)
        let original_binary = workspace_root.join("target/release/examples/concurrent_inference.original");
        let pgo_binary = workspace_root.join("target/release/examples/concurrent_inference.pgo");

        // Check if required files exist
        if !model_path.exists() {
            eprintln!("❌ Model not found at {:?}", model_path);
            eprintln!("   Set BENCH_MODEL_PATH environment variable or download the model:");
            eprintln!("   inferno download TinyLlama_TinyLlama-1.1B-Chat-v1.0");
            return None;
        }

        if !original_binary.exists() {
            eprintln!(
                "❌ Original binary not found at {:?}",
                original_binary
            );
            eprintln!("   Run the PGO script first from workspace root:");
            eprintln!("   ./crates/inference/benches/build-pgo-concurrent.sh");
            return None;
        }

        if !pgo_binary.exists() {
            eprintln!(
                "❌ PGO-optimized binary not found at {:?}",
                pgo_binary
            );
            eprintln!("   Run the PGO script first from workspace root:");
            eprintln!("   ./crates/inference/benches/build-pgo-concurrent.sh");
            return None;
        }

        Some(Self {
            model_path,
            original_binary,
            pgo_binary,
        })
    }
}

/// Run concurrent inference with the specified parameters
fn run_concurrent_inference(
    binary: &PathBuf,
    model_path: &PathBuf,
    prompt: &str,
    concurrency: usize,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let start = Instant::now();

    let output = Command::new(binary)
        .args([
            "--prompt",
            prompt,
            "--model-path",
            &model_path.to_string_lossy(),
            "--concurrent",
            &concurrency.to_string(),
        ])
        .output()?;

    let duration = start.elapsed();

    if !output.status.success() {
        return Err(format!(
            "Inference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(duration)
}

/// Benchmark single request performance - Original vs PGO
fn bench_single_request_pgo_comparison(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping single request benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("single_request_pgo_comparison");

    let prompts = vec![
        ("short", "Hi"),
        ("medium", "Explain machine learning briefly"),
        ("long", "Write a detailed explanation of neural networks, including forward propagation, backpropagation, and gradient descent"),
    ];

    for (name, prompt) in prompts {
        // Benchmark original binary
        group.bench_with_input(
            BenchmarkId::new("original", name),
            &prompt,
            |b, prompt| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        1, // Single request
                    )
                    .expect("Single request should succeed")
                });
            },
        );

        // Benchmark PGO-optimized binary
        group.bench_with_input(
            BenchmarkId::new("pgo", name),
            &prompt,
            |b, prompt| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        1, // Single request
                    )
                    .expect("Single request should succeed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark medium concurrency performance - Original vs PGO (10-50 requests)
fn bench_medium_concurrency_pgo_comparison(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping medium concurrency benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("medium_concurrency_pgo_comparison");
    group.throughput(Throughput::Elements(1));

    let concurrency_levels = vec![10, 25, 50];
    let prompt = "What is 2+2?";

    for concurrency in concurrency_levels {
        // Benchmark original binary
        group.bench_with_input(
            BenchmarkId::new("original", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("Medium concurrency should succeed")
                });
            },
        );

        // Benchmark PGO-optimized binary
        group.bench_with_input(
            BenchmarkId::new("pgo", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("Medium concurrency should succeed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark high concurrency performance - Original vs PGO (100+ requests)
fn bench_high_concurrency_pgo_comparison(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping high concurrency benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("high_concurrency_pgo_comparison");
    group.throughput(Throughput::Elements(1));
    group.sample_size(10); // Fewer samples for high concurrency tests

    let concurrency_levels = vec![100, 200];
    let prompt = "Hi";

    for concurrency in concurrency_levels {
        // Benchmark original binary
        group.bench_with_input(
            BenchmarkId::new("original", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("High concurrency should succeed")
                });
            },
        );

        // Benchmark PGO-optimized binary
        group.bench_with_input(
            BenchmarkId::new("pgo", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("High concurrency should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_request_pgo_comparison,
    bench_medium_concurrency_pgo_comparison,
    bench_high_concurrency_pgo_comparison
);
criterion_main!(benches);
