use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// Configuration for PGO benchmarks
struct PGOBenchmarkConfig {
    model_path: PathBuf,
    baseline_binary: PathBuf,
    pgo_binary: PathBuf,
    baseline_concurrent_binary: PathBuf,
    pgo_concurrent_binary: PathBuf,
}

impl PGOBenchmarkConfig {
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

        // Expected binary locations (built by build.rs)
        let baseline_binary = PathBuf::from("./target/release/inferno-baseline");
        let pgo_binary = PathBuf::from("./target/release/inferno-pgo");
        let baseline_concurrent_binary =
            PathBuf::from("./target/release/examples/concurrent_inference-baseline");
        let pgo_concurrent_binary =
            PathBuf::from("./target/release/examples/concurrent_inference-pgo");

        // Check if all required files exist
        if !model_path.exists() {
            eprintln!("❌ Model not found at {:?}", model_path);
            eprintln!("   Set BENCH_MODEL_PATH environment variable");
            return None;
        }

        if !baseline_binary.exists() {
            eprintln!("❌ Baseline binary not found at {:?}", baseline_binary);
            eprintln!("   build.rs should have created this automatically");
            return None;
        }

        if !pgo_binary.exists() {
            eprintln!("❌ PGO binary not found at {:?}", pgo_binary);
            eprintln!("   build.rs should have created this automatically");
            return None;
        }

        if !baseline_concurrent_binary.exists() {
            eprintln!(
                "❌ Baseline concurrent example not found at {:?}",
                baseline_concurrent_binary
            );
            eprintln!("   build.rs should have created this automatically");
            return None;
        }

        if !pgo_concurrent_binary.exists() {
            eprintln!(
                "❌ PGO concurrent example not found at {:?}",
                pgo_concurrent_binary
            );
            eprintln!("   build.rs should have created this automatically");
            return None;
        }

        println!("🚀 All PGO benchmark binaries found:");
        println!("   Model: {:?}", model_path);
        println!("   Baseline CLI: {:?}", baseline_binary);
        println!("   PGO CLI: {:?}", pgo_binary);
        println!("   Baseline Concurrent: {:?}", baseline_concurrent_binary);
        println!("   PGO Concurrent: {:?}", pgo_concurrent_binary);

        Some(PGOBenchmarkConfig {
            model_path,
            baseline_binary,
            pgo_binary,
            baseline_concurrent_binary,
            pgo_concurrent_binary,
        })
    }
}

/// Run single inference request
fn run_single_inference(
    binary: &PathBuf,
    model_path: &PathBuf,
    prompt: &str,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let start = Instant::now();

    let output = Command::new(binary)
        .arg("play")
        .arg("--prompt")
        .arg(prompt)
        .arg("--model-path")
        .arg(model_path)
        .output()?;

    let duration = start.elapsed();

    if !output.status.success() {
        return Err(format!(
            "Single inference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(duration)
}

/// Run concurrent inference requests
fn run_concurrent_inference(
    binary: &PathBuf,
    model_path: &PathBuf,
    prompt: &str,
    concurrency: usize,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let start = Instant::now();

    let output = Command::new(binary)
        .arg("--prompt")
        .arg(prompt)
        .arg("--model-path")
        .arg(model_path)
        .arg("--concurrent")
        .arg(concurrency.to_string())
        .output()?;

    let duration = start.elapsed();

    if !output.status.success() {
        return Err(format!(
            "Concurrent inference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(duration)
}

/// Benchmark single request latency (CLI comparison)
fn bench_single_request_latency(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(config) => config,
        None => {
            eprintln!("⚠️  Skipping PGO benchmarks - missing binaries or model");
            return;
        }
    };

    let mut group = c.benchmark_group("pgo_single_request");
    group.sample_size(5);
    group.measurement_time(Duration::from_secs(120));

    let prompts = vec!["hi", "what is python?", "explain machine learning briefly"];

    for prompt in &prompts {
        // Benchmark baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", prompt),
            prompt,
            |b, &prompt| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        match run_single_inference(
                            &config.baseline_binary,
                            &config.model_path,
                            prompt,
                        ) {
                            Ok(duration) => total_duration += duration,
                            Err(e) => {
                                eprintln!("Baseline single request error: {}", e);
                                total_duration += Duration::from_secs(30);
                            }
                        }
                    }
                    total_duration
                });
            },
        );

        // Benchmark PGO
        group.bench_with_input(BenchmarkId::new("pgo", prompt), prompt, |b, &prompt| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::new(0, 0);
                for _ in 0..iters {
                    match run_single_inference(&config.pgo_binary, &config.model_path, prompt) {
                        Ok(duration) => total_duration += duration,
                        Err(e) => {
                            eprintln!("PGO single request error: {}", e);
                            total_duration += Duration::from_secs(30);
                        }
                    }
                }
                total_duration
            });
        });
    }

    group.finish();
}

/// Benchmark low-medium concurrency (where PGO starts to shine)
fn bench_concurrent_requests_medium(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(config) => config,
        None => return,
    };

    let mut group = c.benchmark_group("pgo_concurrent_medium");
    group.sample_size(3);
    group.measurement_time(Duration::from_secs(180));

    let concurrency_levels = vec![10, 25, 50];
    let test_prompt = "hello world";

    println!("🔥 Testing medium concurrency - where PGO really shines!");

    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));

        println!("   Testing {} concurrent requests...", concurrency);

        // Benchmark baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        match run_concurrent_inference(
                            &config.baseline_concurrent_binary,
                            &config.model_path,
                            test_prompt,
                            concurrency,
                        ) {
                            Ok(duration) => total_duration += duration,
                            Err(e) => {
                                eprintln!("Baseline concurrent error ({}): {}", concurrency, e);
                                total_duration += Duration::from_secs(60);
                            }
                        }
                    }
                    total_duration
                });
            },
        );

        // Benchmark PGO
        group.bench_with_input(
            BenchmarkId::new("pgo", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        match run_concurrent_inference(
                            &config.pgo_concurrent_binary,
                            &config.model_path,
                            test_prompt,
                            concurrency,
                        ) {
                            Ok(duration) => total_duration += duration,
                            Err(e) => {
                                eprintln!("PGO concurrent error ({}): {}", concurrency, e);
                                total_duration += Duration::from_secs(60);
                            }
                        }
                    }
                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark high concurrency (stress test)
fn bench_concurrent_requests_high(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(config) => config,
        None => return,
    };

    let mut group = c.benchmark_group("pgo_concurrent_high");
    group.sample_size(2);
    group.measurement_time(Duration::from_secs(300));

    let concurrency_levels = vec![100, 200];
    let test_prompt = "hi"; // Very short for stress test

    println!("⚡ Testing high concurrency - maximum PGO benefits!");

    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));

        println!("   Stress testing {} concurrent requests...", concurrency);

        // Benchmark baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        match run_concurrent_inference(
                            &config.baseline_concurrent_binary,
                            &config.model_path,
                            test_prompt,
                            concurrency,
                        ) {
                            Ok(duration) => total_duration += duration,
                            Err(e) => {
                                eprintln!(
                                    "Baseline high concurrent error ({}): {}",
                                    concurrency, e
                                );
                                total_duration += Duration::from_secs(120);
                            }
                        }
                    }
                    total_duration
                });
            },
        );

        // Benchmark PGO
        group.bench_with_input(
            BenchmarkId::new("pgo", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        match run_concurrent_inference(
                            &config.pgo_concurrent_binary,
                            &config.model_path,
                            test_prompt,
                            concurrency,
                        ) {
                            Ok(duration) => total_duration += duration,
                            Err(e) => {
                                eprintln!("PGO high concurrent error ({}): {}", concurrency, e);
                                total_duration += Duration::from_secs(120);
                            }
                        }
                    }
                    total_duration
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_request_latency,
    bench_concurrent_requests_medium,
    bench_concurrent_requests_high
);
criterion_main!(benches);
