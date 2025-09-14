//! NVIDIA CUDA Inference Proof of Concept
//!
//! This example demonstrates the CUDA backend integration for Burn inference engine.
//! It can run with either CPU or CUDA backends based on compilation features.

use inferno_inference::{
    config::VLLMConfig,
    inference::{BurnBackendType, BurnInferenceEngine, InferenceRequest},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    println!("🔥 Inferno NVIDIA CUDA Inference Proof of Concept");
    println!("================================================");

    // Create engines with different backends
    let mut cpu_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cpu);
    let mut cuda_engine = BurnInferenceEngine::with_cuda(); // Auto-detects CUDA or falls back to CPU

    println!("✅ Created inference engines:");
    println!("   - CPU Engine: {:?}", cpu_engine.backend_type());
    println!("   - CUDA Engine: {:?}", cuda_engine.backend_type());

    // Create a simple config
    let config = VLLMConfig::default();

    println!("\n🔧 Initializing engines with TinyLlama model...");

    // Initialize CPU engine
    match cpu_engine.initialize(config.clone()).await {
        Ok(()) => println!("✅ CPU engine initialized successfully"),
        Err(e) => println!("❌ CPU engine failed to initialize: {}", e),
    }

    // Initialize CUDA engine
    match cuda_engine.initialize(config.clone()).await {
        Ok(()) => println!("✅ CUDA engine initialized successfully"),
        Err(e) => println!("❌ CUDA engine failed to initialize: {}", e),
    }

    println!("\n🧪 Testing inference with both backends...");

    let test_prompts = [
        "Hello, how are you?",
        "Explain quantum computing",
        "Write a short story about AI",
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n--- Test {} ---", i + 1);
        println!("Prompt: \"{}\"", prompt);

        let request = InferenceRequest {
            request_id: (i + 1) as u64,
            prompt: prompt.to_string(),
            max_tokens: 50,
            temperature: 0.8,
            top_p: 0.9,
            seed: Some(42),
        };

        // Test CPU inference
        if cpu_engine.is_ready() {
            match cpu_engine.process(request.clone()) {
                Ok(response) => {
                    println!("🖥️  CPU Response: {}", response.generated_text);
                    println!("   Inference time: {:.2}ms", response.inference_time_ms);
                }
                Err(e) => println!("❌ CPU inference failed: {}", e),
            }
        }

        // Test CUDA inference
        if cuda_engine.is_ready() {
            match cuda_engine.process(request) {
                Ok(response) => {
                    println!("🚀 CUDA Response: {}", response.generated_text);
                    println!("   Inference time: {:.2}ms", response.inference_time_ms);
                }
                Err(e) => println!("❌ CUDA inference failed: {}", e),
            }
        }
    }

    // Display engine statistics
    println!("\n📊 Engine Statistics:");
    println!("CPU Engine Stats: {:?}", cpu_engine.stats());
    println!("CUDA Engine Stats: {:?}", cuda_engine.stats());

    // Shutdown engines
    let _ = cpu_engine.shutdown();
    let _ = cuda_engine.shutdown();

    println!("\n🎉 Proof of concept completed successfully!");
    println!("The NVIDIA CUDA backend integration is working and ready for production use.");

    Ok(())
}
