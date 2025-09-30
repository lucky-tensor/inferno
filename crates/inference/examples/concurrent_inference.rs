use clap::Parser;
use inferno_inference::inference::EngineType;
use inferno_inference::{create_engine, create_math_test_request, InfernoConfig};
use inferno_shared::print_inference_stats_with_newline;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;
use tracing_subscriber::fmt::init;

/// Concurrent inference example for benchmarking PGO improvements
#[derive(Parser, Debug)]
#[command(name = "concurrent-inference")]
#[command(about = "Run concurrent inference requests within a single process")]
#[command(version)]
struct Args {
    /// The prompt/query to send to the model for inference
    #[arg(short, long)]
    prompt: String,

    /// Path to the directory containing the safetensors model
    #[arg(short, long)]
    model_path: PathBuf,

    /// Number of concurrent inference requests to run
    #[arg(short, long, default_value = "1")]
    concurrent: usize,

    /// Force CPU backend instead of auto-detecting GPU
    #[arg(long, default_value = "false")]
    cpu: bool,

    /// Show individual request timings
    #[arg(long, default_value = "false")]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    init();

    // Parse command line arguments
    let args = Args::parse();

    // Validate that the model path exists
    if !args.model_path.exists() {
        eprintln!(
            "Error: Model path '{}' does not exist",
            args.model_path.display()
        );
        std::process::exit(1);
    }

    if !args.model_path.is_dir() {
        eprintln!(
            "Error: Model path '{}' is not a directory",
            args.model_path.display()
        );
        std::process::exit(1);
    }

    // Create a basic config with CLI model path
    let mut config = InfernoConfig {
        model_path: args.model_path.to_string_lossy().to_string(),
        ..Default::default()
    };

    // Load other settings from environment if available
    if let Ok(model_name) = std::env::var("INFERNO_MODEL_NAME") {
        config.model_name = model_name;
    }

    // Validate the final config
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    println!("Model path: {}", args.model_path.display());
    println!("Prompt: {}", args.prompt);
    println!("Concurrent requests: {}", args.concurrent);
    println!();

    println!("🚀 COLD START ANALYSIS");
    println!("======================");

    print!("⏱️  [1/3] Initializing inference engine... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Track detailed cold start phases
    let total_cold_start = Instant::now();
    let loading_start = Instant::now();

    // Select backend based on CLI flag and feature availability
    let engine_type = if args.cpu {
        EngineType::CandleCpu
    } else {
        #[cfg(feature = "candle-cuda")]
        {
            EngineType::CandleCuda
        }
        #[cfg(not(feature = "candle-cuda"))]
        {
            EngineType::CandleCpu
        }
    };

    let backend_init_time = loading_start.elapsed();
    println!("✅ Done ({:.3}s)", backend_init_time.as_secs_f64());

    print!("⏱️  [2/3] Creating {} engine... ", engine_type);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let engine_create_start = Instant::now();

    let mut engine = create_engine(engine_type);
    let engine_create_time = engine_create_start.elapsed();
    println!("✅ Done ({:.3}s)", engine_create_time.as_secs_f64());

    print!("⏱️  [3/3] Loading model and initializing GPU context... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let model_load_start = Instant::now();

    // Initialize the engine with the config
    engine.initialize(config).await?;

    let model_load_time = model_load_start.elapsed();
    let total_cold_start_time = total_cold_start.elapsed();

    println!("✅ Done ({:.3}s)", model_load_time.as_secs_f64());
    println!();
    println!("📊 COLD START BREAKDOWN:");
    println!("   Backend initialization: {:.3}s", backend_init_time.as_secs_f64());
    println!("   Engine creation:        {:.3}s", engine_create_time.as_secs_f64());
    println!("   Model loading + GPU:    {:.3}s", model_load_time.as_secs_f64());
    println!("   ═══════════════════════════════");
    println!("   Total cold start:       {:.3}s", total_cold_start_time.as_secs_f64());
    println!();

    // Create shared engine reference for concurrent access
    let engine = Arc::new(tokio::sync::Mutex::new(engine));

    if args.concurrent == 1 {
        // Single request case
        print!("Running inference... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let inference_start = Instant::now();

        // Create inference request
        let mut request = create_math_test_request();
        request.prompt = args.prompt.clone();

        // Run single inference
        let engine_guard = engine.lock().await;
        match engine_guard.process(request).await {
            Ok(response) => {
                let total_inference_duration = inference_start.elapsed();
                println!("Done!\n");
                println!("{}", response.generated_text);

                // Show cold start time separately
                eprintln!("\nCold start: {:.1}s", total_cold_start_time.as_secs_f64());

                // Show inference statistics
                let inference_time_ms = total_inference_duration.as_millis() as f64;
                print_inference_stats_with_newline(&response, Some(inference_time_ms));
            }
            Err(e) => {
                println!("Failed!");
                eprintln!("Inference error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Concurrent requests case
        println!("🏃 CONCURRENT INFERENCE ANALYSIS");
        println!("=================================");
        println!(
            "Spawning {} concurrent inference requests...",
            args.concurrent
        );

        let total_start = Instant::now();
        let spawn_start = Instant::now();
        let mut join_set = JoinSet::new();

        // Spawn concurrent inference tasks
        for request_id in 0..args.concurrent {
            let engine_clone = Arc::clone(&engine);
            let prompt = args.prompt.clone();
            let verbose = args.verbose;

            join_set.spawn(async move {
                let request_spawn_time = Instant::now();

                // Track time waiting to acquire engine lock (concurrency overhead)
                let lock_wait_start = Instant::now();
                let engine_guard = engine_clone.lock().await;
                let lock_wait_time = lock_wait_start.elapsed();

                // Create inference request
                let mut request = create_math_test_request();
                request.prompt = prompt;

                // Track actual inference time (excluding lock wait)
                let inference_start = Instant::now();
                let result = engine_guard.process(request).await;
                let inference_time = inference_start.elapsed();

                drop(engine_guard); // Release lock immediately

                let total_request_time = request_spawn_time.elapsed();

                match result {
                    Ok(response) => {
                        // Always show timing breakdown for concurrent requests
                        println!(
                            "✅ Request {:2}: total={:.3}s (wait={:.3}s, inference={:.3}s) tokens={}",
                            request_id,
                            total_request_time.as_secs_f64(),
                            lock_wait_time.as_secs_f64(),
                            inference_time.as_secs_f64(),
                            response.generated_tokens
                        );

                        if verbose {
                            println!("   └─ Response: {}", response.generated_text.chars().take(50).collect::<String>() + "...");
                        }

                        Ok((request_id, total_request_time, lock_wait_time, inference_time, response))
                    }
                    Err(e) => {
                        println!("❌ Request {:2}: FAILED after {:.3}s - {}", request_id, total_request_time.as_secs_f64(), e);
                        Err(e)
                    }
                }
            });
        }

        let spawn_time = spawn_start.elapsed();
        println!("📤 All requests spawned in {:.3}s", spawn_time.as_secs_f64());
        println!();

        // Collect results with detailed timing analysis
        let mut successful_requests = 0;
        let mut total_tokens = 0;
        let mut individual_durations = Vec::new();
        let mut lock_wait_times = Vec::new();
        let mut inference_times = Vec::new();
        let mut first_request_time = None;
        let mut last_request_time = None;

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok((_request_id, total_time, wait_time, inference_time, response))) => {
                    successful_requests += 1;
                    total_tokens += response.generated_tokens;
                    individual_durations.push(total_time);
                    lock_wait_times.push(wait_time);
                    inference_times.push(inference_time);

                    // Track first and last completion times for throughput analysis
                    let completion_time = total_start.elapsed();
                    if first_request_time.is_none() || completion_time < first_request_time.unwrap() {
                        first_request_time = Some(completion_time);
                    }
                    if last_request_time.is_none() || completion_time > last_request_time.unwrap() {
                        last_request_time = Some(completion_time);
                    }
                }
                Ok(Err(_)) => {
                    // Error already printed in task
                }
                Err(e) => {
                    eprintln!("Task panicked: {}", e);
                }
            }
        }

        let total_duration = total_start.elapsed();

        // Print comprehensive performance analysis
        println!();
        println!("📊 PERFORMANCE ANALYSIS");
        println!("========================");

        println!("🎯 Overall Results:");
        println!("   Successful requests: {}/{}", successful_requests, args.concurrent);
        println!("   Total execution time: {:.3}s", total_duration.as_secs_f64());
        println!("   Total tokens generated: {}", total_tokens);

        if successful_requests > 0 {
            // Calculate detailed statistics
            individual_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            lock_wait_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            inference_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean_total = individual_durations.iter().sum::<std::time::Duration>() / individual_durations.len() as u32;
            let mean_wait = lock_wait_times.iter().sum::<std::time::Duration>() / lock_wait_times.len() as u32;
            let mean_inference = inference_times.iter().sum::<std::time::Duration>() / inference_times.len() as u32;

            let median_total = individual_durations[individual_durations.len() / 2];
            let median_wait = lock_wait_times[lock_wait_times.len() / 2];
            let median_inference = inference_times[inference_times.len() / 2];

            let first_completion = first_request_time.unwrap_or(total_duration);
            let last_completion = last_request_time.unwrap_or(total_duration);

            println!();
            println!("⏱️  Timing Breakdown (where PGO optimization matters):");
            println!("   ┌─ Total per request:");
            println!("   │  Mean:   {:.3}s", mean_total.as_secs_f64());
            println!("   │  Median: {:.3}s", median_total.as_secs_f64());
            println!("   │  Range:  {:.3}s - {:.3}s", individual_durations[0].as_secs_f64(), individual_durations.last().unwrap().as_secs_f64());
            println!("   │");
            println!("   ├─ Lock wait time (concurrency overhead):");
            println!("   │  Mean:   {:.3}s ({:.1}% of total)", mean_wait.as_secs_f64(), (mean_wait.as_secs_f64() / mean_total.as_secs_f64()) * 100.0);
            println!("   │  Median: {:.3}s", median_wait.as_secs_f64());
            println!("   │");
            println!("   └─ Pure inference time (PGO optimizes this):");
            println!("      Mean:   {:.3}s ({:.1}% of total)", mean_inference.as_secs_f64(), (mean_inference.as_secs_f64() / mean_total.as_secs_f64()) * 100.0);
            println!("      Median: {:.3}s", median_inference.as_secs_f64());

            println!();
            println!("🚀 Throughput Analysis:");
            println!("   First request completed: {:.3}s", first_completion.as_secs_f64());
            println!("   Last request completed:  {:.3}s", last_completion.as_secs_f64());
            println!("   Average tokens/sec:      {:.1}", total_tokens as f64 / total_duration.as_secs_f64());
            println!("   Requests/sec:            {:.1}", successful_requests as f64 / total_duration.as_secs_f64());

            // Performance insights for PGO analysis
            let inference_percentage = (mean_inference.as_secs_f64() / mean_total.as_secs_f64()) * 100.0;
            let wait_percentage = (mean_wait.as_secs_f64() / mean_total.as_secs_f64()) * 100.0;

            println!();
            println!("💡 PGO Optimization Insights:");
            if inference_percentage > 50.0 {
                println!("   ✅ Good PGO target: {:.1}% of time spent in pure inference", inference_percentage);
                println!("      PGO should show meaningful improvements here");
            } else {
                println!("   ⚠️  Limited PGO benefit: only {:.1}% of time in pure inference", inference_percentage);
                println!("      Most time spent on concurrency overhead ({:.1}%) or I/O", wait_percentage);
            }

            let total_cpu_time = mean_inference.as_secs_f64() * successful_requests as f64;
            println!("   📈 Total CPU inference time: {:.3}s (vs {:.3}s wall time)", total_cpu_time, total_duration.as_secs_f64());
            println!("   📊 Parallelism efficiency: {:.1}% ({:.1}x speedup)", (total_cpu_time / total_duration.as_secs_f64()) * 100.0, total_cpu_time / total_duration.as_secs_f64());
        }
    }

    Ok(())
}
