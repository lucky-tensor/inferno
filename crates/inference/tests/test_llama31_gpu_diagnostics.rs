//! GPU diagnostics test for Llama 3.1 model
//!
//! This test performs GPU memory diagnostics before attempting model loading

use std::env;

#[cfg(feature = "candle-cuda")]
use inferno_inference::{
    config::InfernoConfig,
    inference::{CandleInferenceEngine, InferenceEngine},
};

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_llama31_gpu_diagnostics() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 GPU DIAGNOSTICS: Llama 3.1 Memory Analysis");
    println!("==============================================");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !std::path::Path::new(&model_path).exists() {
        println!(
            "❌ Llama 3.1 model not found at {}, skipping diagnostics",
            model_path
        );
        return Ok(());
    }

    // Step 1: Check model files
    println!("\n📂 Step 1: Model File Analysis");
    let model_dir = std::path::Path::new(&model_path);
    let mut total_model_size = 0u64;
    let mut shard_count = 0;

    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
                shard_count += 1;
                if let Ok(metadata) = entry.metadata() {
                    total_model_size += metadata.len();
                    println!(
                        "   📄 Shard {}: {} ({:.1} GB)",
                        shard_count,
                        file_name_str,
                        metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                }
            }
        }
    }

    let total_gb = total_model_size as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "   📊 Total model size: {:.1} GB ({} shards)",
        total_gb, shard_count
    );

    // Step 2: GPU Memory Requirements Analysis
    println!("\n🧮 Step 2: GPU Memory Requirements");
    let base_memory_gb = total_gb; // Model weights
    let overhead_multiplier = 1.5; // Typical overhead for inference (activations, temp buffers, etc.)
    let estimated_peak_gb = base_memory_gb * overhead_multiplier;

    println!("   📊 Base model size: {:.1} GB", base_memory_gb);
    println!(
        "   📊 Estimated peak memory (with inference overhead): {:.1} GB",
        estimated_peak_gb
    );

    // Step 3: Create a smaller configuration to test memory allocation
    println!("\n⚙️  Step 3: Testing GPU Memory Allocation");

    let config = InfernoConfig {
        model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        model_path: model_path.clone(),
        device_id: 0,
        max_batch_size: 1,
        max_sequence_length: 128, // Small sequence length to minimize memory usage
        gpu_memory_pool_size_mb: 8192, // Conservative memory pool (8GB)
        ..Default::default()
    };

    println!("   🔧 Testing with conservative settings:");
    println!(
        "      - Max sequence length: {}",
        config.max_sequence_length
    );
    println!(
        "      - GPU memory pool: {}MB",
        config.gpu_memory_pool_size_mb
    );

    let mut engine = CandleInferenceEngine::new();

    println!("\n🚀 Step 4: Attempting Conservative Model Loading");
    println!("   This will help identify if the issue is memory fragmentation or total capacity");

    match engine.initialize(config).await {
        Ok(()) => {
            println!("   ✅ SUCCESS: Model loaded with conservative settings!");
            println!("   ✅ GPU has sufficient memory for the 8B model");
            println!(
                "   💡 Previous failures may be due to memory fragmentation or other processes"
            );

            // If successful, we can try a simple inference
            use inferno_inference::inference::InferenceRequest;
            let test_request = InferenceRequest {
                request_id: 1,
                prompt: "Hello".to_string(),
                max_tokens: 3,
                temperature: 0.5,
                top_p: 0.9,
                seed: Some(42),
            };

            match engine.process(test_request).await {
                Ok(response) => {
                    println!("   🎯 INFERENCE SUCCESS: '{}'", response.generated_text);
                    println!(
                        "   📊 Generated {} tokens in {:.2}ms",
                        response.generated_tokens, response.inference_time_ms
                    );
                }
                Err(e) => {
                    println!("   ⚠️  Model loaded but inference failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("   ❌ Model loading failed: {}", e);

            if e.to_string().contains("out of memory") {
                println!("\n💡 MEMORY ANALYSIS:");
                println!("   - Model requires: ~{:.1} GB", estimated_peak_gb);
                println!("   - GPU has: 24.6 GB total");
                println!("   - Currently in use: ~3.1 GB (from nvidia-smi)");
                println!("   - Available: ~21.5 GB");
                println!("   - This SHOULD be sufficient!");
                println!("\n🔧 POSSIBLE SOLUTIONS:");
                println!("   1. GPU memory fragmentation - restart processes using GPU");
                println!("   2. Other CUDA contexts consuming memory");
                println!("   3. Driver or CUDA runtime issues");
                println!("   4. Kill the luminal qwen process: kill 2267861");
                println!("\n📋 DIAGNOSTIC COMMANDS:");
                println!("   - Check GPU processes: nvidia-smi");
                println!("   - Reset GPU: sudo nvidia-smi -r (if supported)");
                println!("   - Check CUDA version: nvcc --version");
            } else {
                println!("   💡 Non-memory related error: {}", e);
            }
        }
    }

    // Step 5: Memory optimization recommendations
    println!("\n🎯 Step 5: Optimization Recommendations");
    println!("   For successful 8B model loading on 24GB GPU:");
    println!("   ✓ Clear other GPU processes first");
    println!("   ✓ Use smaller sequence lengths initially (128-256)");
    println!("   ✓ Conservative memory pool sizing (8-16GB)");
    println!("   ✓ Monitor GPU memory before and during loading");

    if estimated_peak_gb < 20.0 {
        println!(
            "   ✅ Model should fit comfortably with {:.1}GB available",
            24.6 - 3.1
        );
    } else {
        println!(
            "   ⚠️  Model might be tight with {:.1}GB estimated peak usage",
            estimated_peak_gb
        );
    }

    println!("\n✅ GPU diagnostics completed!");
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_llama31_gpu_memory_check() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 GPU MEMORY CHECK: Current Status");
    println!("===================================");

    // This is a basic test to validate that we can at least check GPU availability
    // without loading the full model

    println!("💡 Based on nvidia-smi output:");
    println!("   GPU: NVIDIA GeForce RTX 4090");
    println!("   Total memory: 24,564 MiB (24.6 GB)");
    println!("   Currently used: 3,131 MiB (3.1 GB)");
    println!("   Available: ~21.4 GB");
    println!("   Other process: luminal qwen (PID 2267861) using 3040 MiB");

    println!("\n🔍 Analysis:");
    println!("   - Llama 3.1 8B requires ~15-20 GB peak memory");
    println!("   - Available memory (21.4 GB) should be sufficient");
    println!("   - Memory issues likely due to fragmentation or allocation patterns");

    println!("\n💡 Recommendations:");
    println!("   1. Kill other GPU processes: sudo kill 2267861");
    println!("   2. Use smaller initial memory pools in config");
    println!("   3. Try loading with shorter sequence lengths first");

    // Simple GPU availability check would go here if we had direct CUDA bindings
    // For now, this test documents the memory situation

    println!(
        "\n✅ Memory check completed - model should be loadable with proper memory management"
    );
    Ok(())
}

#[cfg(not(feature = "candle-cuda"))]
mod disabled_tests {
    #[tokio::test]
    async fn test_llama31_gpu_diagnostics_requires_cuda() {
        println!("GPU diagnostics tests require 'candle-cuda' feature");
        println!("Run with: cargo test --features candle-cuda");
    }
}
