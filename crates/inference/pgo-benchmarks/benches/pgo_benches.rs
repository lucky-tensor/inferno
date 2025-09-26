#![allow(clippy::expect_used, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// Configuration for concurrent inference benchmarks
struct ConcurrentBenchmarkConfig {
    model_path: PathBuf,
    concurrent_binary: PathBuf,
}

impl ConcurrentBenchmarkConfig {
    fn new() -> Option<Self> {
        let model_path = std::env::var("BENCH_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                PathBuf::from(format!(
                    "{}/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors",
                    home
                ))
            });

        // Expected binary location (built by build.rs)
        let concurrent_binary = PathBuf::from("./target/release/examples/concurrent_inference");

        // Check if required files exist
        if !model_path.exists() {
            eprintln!("❌ Model not found at {:?}", model_path);
            eprintln!("   Set BENCH_MODEL_PATH environment variable");
            return None;
        }

        if !concurrent_binary.exists() {
            eprintln!(
                "❌ concurrent_inference binary not found at {:?}",
                concurrent_binary
            );
            eprintln!("   Run: cargo build --release --package inferno-inference --example concurrent_inference --features examples");
            return None;
        }

        Some(Self {
            model_path,
            concurrent_binary,
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

/// Benchmark single request performance
fn bench_single_request(c: &mut Criterion) {
    let config = match ConcurrentBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping single request benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("single_request");

    let prompts = vec![
        ("short", "Hi"),
        ("medium", "Explain machine learning briefly"),
        ("long", "Write a detailed explanation of neural networks, including forward propagation, backpropagation, and gradient descent"),
    ];

    for (name, prompt) in prompts {
        group.bench_with_input(
            BenchmarkId::new("concurrent_inference", name),
            &prompt,
            |b, prompt| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.concurrent_binary,
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

/// Benchmark medium concurrency performance (10-50 requests)
fn bench_medium_concurrency(c: &mut Criterion) {
    let config = match ConcurrentBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping medium concurrency benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("medium_concurrency");
    group.throughput(Throughput::Elements(1));

    let concurrency_levels = vec![10, 25, 50];
    let prompt = "What is 2+2?";

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_requests", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.concurrent_binary,
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

/// Benchmark high concurrency performance (100+ requests)
fn bench_high_concurrency(c: &mut Criterion) {
    let config = match ConcurrentBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping high concurrency benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("high_concurrency");
    group.throughput(Throughput::Elements(1));
    group.sample_size(10); // Fewer samples for high concurrency tests

    let concurrency_levels = vec![100, 200];
    let prompt = "Hi";

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("stress_test", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    run_concurrent_inference(
                        &config.concurrent_binary,
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
    bench_single_request,
    bench_medium_concurrency,
    bench_high_concurrency
);
criterion_main!(benches);
