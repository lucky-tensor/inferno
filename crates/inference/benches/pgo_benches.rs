#![allow(clippy::expect_used, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};
use regex::Regex;

/// Detailed performance metrics parsed from concurrent_inference output
#[derive(Debug, Clone)]
struct InferenceMetrics {
    wall_time: Duration,
    cold_start_time: Option<Duration>,
    backend_init_time: Option<Duration>,
    engine_create_time: Option<Duration>,
    model_load_time: Option<Duration>,
    mean_request_time: Option<Duration>,
    mean_inference_time: Option<Duration>,
    mean_lock_wait_time: Option<Duration>,
    total_tokens: Option<u32>,
    successful_requests: Option<u32>,
    tokens_per_sec: Option<f64>,
    requests_per_sec: Option<f64>,
    inference_percentage: Option<f64>,
    parallelism_efficiency: Option<f64>,
}

impl InferenceMetrics {
    fn new(wall_time: Duration) -> Self {
        Self {
            wall_time,
            cold_start_time: None,
            backend_init_time: None,
            engine_create_time: None,
            model_load_time: None,
            mean_request_time: None,
            mean_inference_time: None,
            mean_lock_wait_time: None,
            total_tokens: None,
            successful_requests: None,
            tokens_per_sec: None,
            requests_per_sec: None,
            inference_percentage: None,
            parallelism_efficiency: None,
        }
    }

    /// Parse detailed metrics from concurrent_inference output
    fn parse_from_output(wall_time: Duration, output: &str) -> Self {
        let mut metrics = Self::new(wall_time);

        // Parse cold start breakdown
        if let Some(captures) = Regex::new(r"Backend initialization:\s+([0-9.]+)s").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.backend_init_time = Some(Duration::from_secs_f64(secs));
            }
        }

        if let Some(captures) = Regex::new(r"Engine creation:\s+([0-9.]+)s").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.engine_create_time = Some(Duration::from_secs_f64(secs));
            }
        }

        if let Some(captures) = Regex::new(r"Model loading \+ GPU:\s+([0-9.]+)s").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.model_load_time = Some(Duration::from_secs_f64(secs));
            }
        }

        if let Some(captures) = Regex::new(r"Total cold start:\s+([0-9.]+)s").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.cold_start_time = Some(Duration::from_secs_f64(secs));
            }
        }

        // Parse concurrent performance metrics
        if let Some(captures) = Regex::new(r"Successful requests:\s+([0-9]+)/").unwrap().captures(output) {
            if let Ok(count) = captures[1].parse::<u32>() {
                metrics.successful_requests = Some(count);
            }
        }

        if let Some(captures) = Regex::new(r"Total tokens generated:\s+([0-9]+)").unwrap().captures(output) {
            if let Ok(count) = captures[1].parse::<u32>() {
                metrics.total_tokens = Some(count);
            }
        }

        if let Some(captures) = Regex::new(r"Mean:\s+([0-9.]+)s").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.mean_request_time = Some(Duration::from_secs_f64(secs));
            }
        }

        if let Some(captures) = Regex::new(r"Mean:\s+([0-9.]+)s \([0-9.]+% of total\)").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.mean_lock_wait_time = Some(Duration::from_secs_f64(secs));
            }
        }

        if let Some(captures) = Regex::new(r"Mean:\s+([0-9.]+)s \(([0-9.]+)% of total\)$").unwrap().captures(output) {
            if let Ok(secs) = captures[1].parse::<f64>() {
                metrics.mean_inference_time = Some(Duration::from_secs_f64(secs));
            }
            if let Ok(pct) = captures[2].parse::<f64>() {
                metrics.inference_percentage = Some(pct);
            }
        }

        if let Some(captures) = Regex::new(r"Average tokens/sec:\s+([0-9.]+)").unwrap().captures(output) {
            if let Ok(rate) = captures[1].parse::<f64>() {
                metrics.tokens_per_sec = Some(rate);
            }
        }

        if let Some(captures) = Regex::new(r"Requests/sec:\s+([0-9.]+)").unwrap().captures(output) {
            if let Ok(rate) = captures[1].parse::<f64>() {
                metrics.requests_per_sec = Some(rate);
            }
        }

        if let Some(captures) = Regex::new(r"Parallelism efficiency:\s+([0-9.]+)%").unwrap().captures(output) {
            if let Ok(eff) = captures[1].parse::<f64>() {
                metrics.parallelism_efficiency = Some(eff);
            }
        }

        metrics
    }
}

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

        // Verify that the binaries are actually different
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::fs;

        let original_hash = {
            let content = fs::read(&original_binary).expect("Failed to read original binary");
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            hasher.finish()
        };

        let pgo_hash = {
            let content = fs::read(&pgo_binary).expect("Failed to read PGO binary");
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            hasher.finish()
        };

        if original_hash == pgo_hash {
            eprintln!("❌ CRITICAL: Original and PGO binaries are identical!");
            eprintln!("   The PGO optimization process did not produce a different binary.");
            eprintln!("   Running benchmarks would be meaningless since they're the same binary.");
            eprintln!("   Please run the PGO script to generate proper optimized binaries:");
            eprintln!("   ./crates/inference/benches/build-pgo-concurrent.sh");
            return None;
        }

        eprintln!("✅ Binary verification passed: Original and PGO binaries are different");
        eprintln!("   Ready to run meaningful performance comparisons!");

        Some(Self {
            model_path,
            original_binary,
            pgo_binary,
        })
    }
}

/// Run concurrent inference with the specified parameters and collect detailed metrics
fn run_concurrent_inference(
    binary: &PathBuf,
    model_path: &PathBuf,
    prompt: &str,
    concurrency: usize,
) -> Result<InferenceMetrics, Box<dyn std::error::Error>> {
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

    // Parse detailed metrics from stdout and stderr
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined_output = format!("{}\n{}", stdout, stderr);

    Ok(InferenceMetrics::parse_from_output(duration, &combined_output))
}

/// Run a warm-up + measurement cycle to isolate CPU performance and collect detailed metrics
fn run_concurrent_inference_warm(
    binary: &PathBuf,
    model_path: &PathBuf,
    prompt: &str,
    concurrency: usize,
) -> Result<InferenceMetrics, Box<dyn std::error::Error>> {
    // First run: warm up GPU, load model, establish CUDA context
    let _warmup = Command::new(binary)
        .args([
            "--prompt",
            "warm up", // Short warmup prompt
            "--model-path",
            &model_path.to_string_lossy(),
            "--concurrent",
            "1", // Single request for warmup
        ])
        .output()?;

    // Second run: measure actual performance with warm GPU/model
    // Small delay to ensure GPU context is established
    std::thread::sleep(std::time::Duration::from_millis(100));

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

    // Parse detailed metrics from stdout and stderr
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined_output = format!("{}\n{}", stdout, stderr);

    Ok(InferenceMetrics::parse_from_output(duration, &combined_output))
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
    group.sample_size(10);

    let prompts = vec![
        ("short", "Hi"),
        ("medium", "Explain machine learning briefly"),
        ("long", "Write a detailed explanation of neural networks, including forward propagation, backpropagation, and gradient descent"),
    ];

    for (name, prompt) in &prompts {
        // Benchmark original binary
        group.bench_with_input(
            BenchmarkId::new("original", name),
            &prompt,
            |b, &prompt| {
                b.iter(|| {
                    let metrics = run_concurrent_inference(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        1, // Single request
                    )
                    .expect("Single request should succeed");

                    // Log detailed metrics for analysis
                    if let Some(cold_start) = metrics.cold_start_time {
                        eprintln!("ORIGINAL {} cold_start: {:.3}s", name, cold_start.as_secs_f64());
                    }
                    if let Some(tokens_per_sec) = metrics.tokens_per_sec {
                        eprintln!("ORIGINAL {} tokens/sec: {:.1}", name, tokens_per_sec);
                    }

                    metrics.wall_time
                });
            },
        );

        // Benchmark PGO-optimized binary
        group.bench_with_input(
            BenchmarkId::new("pgo", name),
            &prompt,
            |b, &prompt| {
                b.iter(|| {
                    let metrics = run_concurrent_inference(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        1, // Single request
                    )
                    .expect("Single request should succeed");

                    // Log detailed metrics for analysis
                    if let Some(cold_start) = metrics.cold_start_time {
                        eprintln!("PGO {} cold_start: {:.3}s", name, cold_start.as_secs_f64());
                    }
                    if let Some(tokens_per_sec) = metrics.tokens_per_sec {
                        eprintln!("PGO {} tokens/sec: {:.1}", name, tokens_per_sec);
                    }

                    metrics.wall_time
                });
            },
        );
    }

    group.finish();

    // Also benchmark WARM performance to isolate CPU optimizations
    let mut warm_group = c.benchmark_group("single_request_pgo_warm");
    warm_group.sample_size(10);

    for (name, prompt) in &prompts {
        // Benchmark original binary (warm)
        warm_group.bench_with_input(
            BenchmarkId::new("original_warm", name),
            &prompt,
            |b, &prompt| {
                b.iter(|| {
                    let metrics = run_concurrent_inference_warm(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        1, // Single request
                    )
                    .expect("Single warm request should succeed");

                    // Log inference-specific metrics (should be higher % since warm)
                    if let Some(inference_pct) = metrics.inference_percentage {
                        eprintln!("ORIGINAL_WARM {} inference_pct: {:.1}%", name, inference_pct);
                    }
                    if let Some(tokens_per_sec) = metrics.tokens_per_sec {
                        eprintln!("ORIGINAL_WARM {} tokens/sec: {:.1}", name, tokens_per_sec);
                    }

                    metrics.wall_time
                });
            },
        );

        // Benchmark PGO-optimized binary (warm)
        warm_group.bench_with_input(
            BenchmarkId::new("pgo_warm", name),
            &prompt,
            |b, &prompt| {
                b.iter(|| {
                    let metrics = run_concurrent_inference_warm(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        1, // Single request
                    )
                    .expect("Single warm request should succeed");

                    // Log inference-specific metrics (should be higher % since warm)
                    if let Some(inference_pct) = metrics.inference_percentage {
                        eprintln!("PGO_WARM {} inference_pct: {:.1}%", name, inference_pct);
                    }
                    if let Some(tokens_per_sec) = metrics.tokens_per_sec {
                        eprintln!("PGO_WARM {} tokens/sec: {:.1}", name, tokens_per_sec);
                    }

                    metrics.wall_time
                });
            },
        );
    }

    warm_group.finish();
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
    group.sample_size(10);

    let concurrency_levels = vec![10, 25, 50];
    let prompt = "What is 2+2?";

    for concurrency in concurrency_levels {
        // Benchmark original binary
        group.bench_with_input(
            BenchmarkId::new("original", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let metrics = run_concurrent_inference(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("Medium concurrency should succeed");

                    // Log concurrency-specific metrics
                    if let Some(req_per_sec) = metrics.requests_per_sec {
                        eprintln!("ORIGINAL concurrent={} req/sec: {:.1}", concurrency, req_per_sec);
                    }
                    if let Some(parallel_eff) = metrics.parallelism_efficiency {
                        eprintln!("ORIGINAL concurrent={} parallel_eff: {:.1}%", concurrency, parallel_eff);
                    }

                    metrics.wall_time
                });
            },
        );

        // Benchmark PGO-optimized binary
        group.bench_with_input(
            BenchmarkId::new("pgo", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let metrics = run_concurrent_inference(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("Medium concurrency should succeed");

                    // Log concurrency-specific metrics
                    if let Some(req_per_sec) = metrics.requests_per_sec {
                        eprintln!("PGO concurrent={} req/sec: {:.1}", concurrency, req_per_sec);
                    }
                    if let Some(parallel_eff) = metrics.parallelism_efficiency {
                        eprintln!("PGO concurrent={} parallel_eff: {:.1}%", concurrency, parallel_eff);
                    }

                    metrics.wall_time
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
    group.measurement_time(std::time::Duration::from_secs(20)); // Longer measurement time for high concurrency tests

    let concurrency_levels = vec![100, 200];
    let prompt = "Hi";

    for concurrency in concurrency_levels {
        // Benchmark original binary
        group.bench_with_input(
            BenchmarkId::new("original", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let metrics = run_concurrent_inference(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("High concurrency should succeed");

                    // Log high-concurrency specific metrics
                    if let Some(req_per_sec) = metrics.requests_per_sec {
                        eprintln!("ORIGINAL high_concurrent={} req/sec: {:.1}", concurrency, req_per_sec);
                    }
                    if let Some(parallel_eff) = metrics.parallelism_efficiency {
                        eprintln!("ORIGINAL high_concurrent={} parallel_eff: {:.1}%", concurrency, parallel_eff);
                    }

                    metrics.wall_time
                });
            },
        );

        // Benchmark PGO-optimized binary
        group.bench_with_input(
            BenchmarkId::new("pgo", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let metrics = run_concurrent_inference(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        concurrency,
                    )
                    .expect("High concurrency should succeed");

                    // Log high-concurrency specific metrics
                    if let Some(req_per_sec) = metrics.requests_per_sec {
                        eprintln!("PGO high_concurrent={} req/sec: {:.1}", concurrency, req_per_sec);
                    }
                    if let Some(parallel_eff) = metrics.parallelism_efficiency {
                        eprintln!("PGO high_concurrent={} parallel_eff: {:.1}%", concurrency, parallel_eff);
                    }

                    metrics.wall_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CPU-intensive scenarios where PGO should show more benefit
fn bench_cpu_intensive_pgo_comparison(c: &mut Criterion) {
    let config = match PGOBenchmarkConfig::new() {
        Some(c) => c,
        None => {
            eprintln!("⚠️ Skipping CPU-intensive benchmarks - configuration invalid");
            return;
        }
    };

    let mut group = c.benchmark_group("cpu_intensive_pgo_comparison");
    group.sample_size(10); // Minimum required by criterion
    group.measurement_time(std::time::Duration::from_secs(30)); // Longer measurement time for CPU intensive tests

    // Test scenarios that should stress CPU paths more
    let scenarios = vec![
        ("many_small_requests", "Hi", 50), // Many small concurrent requests
        ("medium_concurrent", "Explain the concept of machine learning in simple terms", 20),
        ("large_concurrent", "Write a detailed explanation of how neural networks work, including backpropagation", 10),
    ];

    for (name, prompt, concurrency) in scenarios {
        // Original binary
        group.bench_with_input(
            BenchmarkId::new("original_intensive", name),
            &(prompt, concurrency),
            |b, (prompt, concurrency)| {
                b.iter(|| {
                    let metrics = run_concurrent_inference_warm(
                        &config.original_binary,
                        &config.model_path,
                        prompt,
                        *concurrency,
                    )
                    .expect("CPU intensive request should succeed");

                    // Log CPU-intensive metrics - these should show PGO benefits
                    if let Some(inference_pct) = metrics.inference_percentage {
                        eprintln!("ORIGINAL_INTENSIVE {} inference_pct: {:.1}%", name, inference_pct);
                    }
                    if let Some(mean_inf) = metrics.mean_inference_time {
                        eprintln!("ORIGINAL_INTENSIVE {} mean_inference: {:.3}s", name, mean_inf.as_secs_f64());
                    }

                    metrics.wall_time
                });
            },
        );

        // PGO binary
        group.bench_with_input(
            BenchmarkId::new("pgo_intensive", name),
            &(prompt, concurrency),
            |b, (prompt, concurrency)| {
                b.iter(|| {
                    let metrics = run_concurrent_inference_warm(
                        &config.pgo_binary,
                        &config.model_path,
                        prompt,
                        *concurrency,
                    )
                    .expect("CPU intensive request should succeed");

                    // Log CPU-intensive metrics - these should show PGO benefits
                    if let Some(inference_pct) = metrics.inference_percentage {
                        eprintln!("PGO_INTENSIVE {} inference_pct: {:.1}%", name, inference_pct);
                    }
                    if let Some(mean_inf) = metrics.mean_inference_time {
                        eprintln!("PGO_INTENSIVE {} mean_inference: {:.3}s", name, mean_inf.as_secs_f64());
                    }

                    metrics.wall_time
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
    bench_high_concurrency_pgo_comparison,
    bench_cpu_intensive_pgo_comparison
);
criterion_main!(benches);
