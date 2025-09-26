use clap::Parser;
use inferno_inference::inference::EngineType;
use inferno_inference::{create_engine, create_math_test_request, InfernoConfig};
use inferno_shared::print_inference_stats_with_newline;
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::fmt::init;

/// Simple CLI tool for running inference with safetensors models
#[derive(Parser, Debug)]
#[command(name = "simple-inference")]
#[command(about = "A simple example of running inference with safetensors models")]
#[command(version)]
struct Args {
    /// The prompt/query to send to the model for inference
    #[arg(short, long)]
    prompt: String,

    /// Path to the directory containing the safetensors model
    #[arg(short, long)]
    model_path: PathBuf,

    /// Force CPU backend instead of auto-detecting GPU
    #[arg(long, default_value = "false")]
    cpu: bool,
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

    // Create inference request using the utility function
    let mut request = create_math_test_request();
    request.prompt = args.prompt.clone();

    print!("Running inference... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Track pure inference time (separate from loading)
    let inference_start = Instant::now();

    // Run inference
    match engine.process(request).await {
        Ok(response) => {
            let total_inference_duration = inference_start.elapsed();
            println!("Done!\n");
            println!("{}", response.generated_text);

            // Show loading time separately
            eprintln!("\nLoading: {:.1}s", loading_duration.as_secs_f64());

            // Show inference statistics using shared utility
            let inference_time_ms = total_inference_duration.as_millis() as f64;
            print_inference_stats_with_newline(&response, Some(inference_time_ms));
        }
        Err(e) => {
            println!("Failed!");
            eprintln!("Inference error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
