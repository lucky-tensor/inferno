//! Test OpenAI model inference with CUDA
//!
//! Usage:
//!   cargo run --example test_openai_cuda --features candle-cuda --release -- /path/to/model

use candle_core::Device;
use inferno_inference::inference::candle::OpenAIEngine;

fn main() -> anyhow::Result<()> {
    // Get model path from args
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} ~/.inferno/models/gpt2", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("🚀 Inferno OpenAI Model Test (CUDA)");
    println!("=====================================\n");

    // Check CUDA availability
    println!("📊 Checking CUDA availability...");
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("✅ CUDA device 0 available!");
            println!("   Device: {:?}\n", device);

            // Load model
            println!("📦 Loading model from: {}", model_path);
            let mut engine = OpenAIEngine::load_from_safetensors(model_path, device)?;
            println!("✅ Model loaded successfully!\n");

            // Test prompts
            let prompts = vec![
                "Hello, my name is",
                "The capital of France is",
                "Rust programming language is",
            ];

            for prompt in prompts {
                println!("💭 Prompt: \"{}\"", prompt);
                println!("🤖 Generating...");

                let output = engine.generate(prompt, 30, 0.7)?;
                println!("📝 Output: {}\n", output);

                // Reset caches for next generation
                engine.reset_caches();
            }

            println!("✅ All tests passed!");
        }
        Err(e) => {
            eprintln!("❌ CUDA not available: {}", e);
            eprintln!("\nPlease ensure:");
            eprintln!("  1. NVIDIA GPU is installed");
            eprintln!("  2. CUDA toolkit is installed");
            eprintln!("  3. Candle is built with CUDA support");
            std::process::exit(1);
        }
    }

    Ok(())
}
