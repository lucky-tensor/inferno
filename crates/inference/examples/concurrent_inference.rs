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

    print!("Initializing inference engine... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Track model loading time
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

    println!("Using {} backend", engine_type);
    let mut engine = create_engine(engine_type);

    // Initialize the engine with the config
    engine.initialize(config).await?;

    let loading_duration = loading_start.elapsed();
    println!("Ready! (loaded in {:.1}s)", loading_duration.as_secs_f64());

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

                // Show loading time separately
                eprintln!("\nLoading: {:.1}s", loading_duration.as_secs_f64());

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
        println!(
            "Running {} concurrent inference requests...",
            args.concurrent
        );

        let total_start = Instant::now();
        let mut join_set = JoinSet::new();

        // Spawn concurrent inference tasks
        for request_id in 0..args.concurrent {
            let engine_clone = Arc::clone(&engine);
            let prompt = args.prompt.clone();
            let verbose = args.verbose;

            join_set.spawn(async move {
                let request_start = Instant::now();

                // Create inference request
                let mut request = create_math_test_request();
                request.prompt = prompt;

                // Acquire engine lock and run inference
                let engine_guard = engine_clone.lock().await;
                let result = engine_guard.process(request).await;
                drop(engine_guard); // Release lock immediately

                let request_duration = request_start.elapsed();

                match result {
                    Ok(response) => {
                        if verbose {
                            println!(
                                "Request {}: completed in {:.3}s",
                                request_id,
                                request_duration.as_secs_f64()
                            );
                        }
                        Ok((request_id, request_duration, response))
                    }
                    Err(e) => {
                        eprintln!("Request {} failed: {}", request_id, e);
                        Err(e)
                    }
                }
            });
        }

        // Collect results
        let mut successful_requests = 0;
        let mut total_tokens = 0;
        let mut individual_durations = Vec::new();

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok((request_id, duration, response))) => {
                    successful_requests += 1;
                    total_tokens += response.generated_tokens;
                    individual_durations.push(duration);

                    if args.verbose {
                        println!(
                            "✅ Request {}: {} tokens in {:.3}s",
                            request_id,
                            response.generated_tokens,
                            duration.as_secs_f64()
                        );
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

        // Print summary statistics
        println!("\n🎉 Concurrent inference completed!");
        println!("Total time: {:.3}s", total_duration.as_secs_f64());
        println!(
            "Successful requests: {}/{}",
            successful_requests, args.concurrent
        );

        if successful_requests > 0 {
            println!("Total tokens generated: {}", total_tokens);

            // Calculate statistics for individual request times
            individual_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean_duration = individual_durations.iter().sum::<std::time::Duration>()
                / individual_durations.len() as u32;
            let median_duration = individual_durations[individual_durations.len() / 2];
            let min_duration = individual_durations[0];
            let max_duration = individual_durations[individual_durations.len() - 1];

            println!("\nIndividual request statistics:");
            println!("  Mean:   {:.3}s", mean_duration.as_secs_f64());
            println!("  Median: {:.3}s", median_duration.as_secs_f64());
            println!("  Min:    {:.3}s", min_duration.as_secs_f64());
            println!("  Max:    {:.3}s", max_duration.as_secs_f64());

            // Calculate throughput
            let requests_per_second = successful_requests as f64 / total_duration.as_secs_f64();
            let tokens_per_second = total_tokens as f64 / total_duration.as_secs_f64();

            println!("\nThroughput:");
            println!("  {:.2} requests/second", requests_per_second);
            println!("  {:.2} tokens/second", tokens_per_second);

            // Show loading time separately
            println!("\nTiming breakdown:");
            println!("  Model loading: {:.3}s", loading_duration.as_secs_f64());
            println!(
                "  Concurrent inference: {:.3}s",
                total_duration.as_secs_f64()
            );
        }
    }

    Ok(())
}
