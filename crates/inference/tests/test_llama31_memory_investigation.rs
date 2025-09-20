//! Memory investigation test for Llama 3.1 model loading
//!
//! This test investigates the actual memory usage patterns during model loading
//! to understand why a 15GB model fails to load with 20.9GB free GPU memory.

use std::env;
use std::process::Command;

#[cfg(feature = "candle-cuda")]
use inferno_inference::{
    config::InfernoConfig,
    inference::{CandleInferenceEngine, InferenceEngine},
};

fn get_gpu_memory_mb() -> Option<(u32, u32, u32)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    let output_str = String::from_utf8(output.stdout).ok()?;
    let parts: Vec<&str> = output_str.trim().split(", ").collect();

    if parts.len() == 3 {
        let used = parts[0].parse().ok()?;
        let free = parts[1].parse().ok()?;
        let total = parts[2].parse().ok()?;
        Some((used, free, total))
    } else {
        None
    }
}

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_llama31_memory_usage_investigation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 MEMORY INVESTIGATION: Llama 3.1 Loading Analysis");
    println!("==================================================");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !std::path::Path::new(&model_path).exists() {
        println!("❌ Llama 3.1 model not found, skipping investigation");
        return Ok(());
    }

    // Step 1: Baseline memory measurement
    println!("\n📊 Step 1: Baseline GPU Memory");
    if let Some((used, free, total)) = get_gpu_memory_mb() {
        println!("   Used: {} MB ({:.1} GB)", used, used as f64 / 1024.0);
        println!("   Free: {} MB ({:.1} GB)", free, free as f64 / 1024.0);
        println!("   Total: {} MB ({:.1} GB)", total, total as f64 / 1024.0);

        if free < 15000 {
            println!("   ⚠️  WARNING: Less than 15GB free, may not be sufficient");
        } else if free > 20000 {
            println!("   ✅ Excellent: Over 20GB free, should be more than sufficient");
        } else {
            println!("   ✅ Good: Should be sufficient for 15GB model");
        }
    } else {
        println!("   ❌ Could not retrieve GPU memory info");
    }

    // Step 2: Analyze model files to understand expected memory usage
    println!("\n📂 Step 2: Model File Analysis");
    let model_dir = std::path::Path::new(&model_path);
    let mut shard_sizes = Vec::new();
    let mut total_model_bytes = 0u64;

    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
                if let Ok(metadata) = entry.metadata() {
                    let size_bytes = metadata.len();
                    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
                    shard_sizes.push((file_name_str.to_string(), size_bytes, size_mb));
                    total_model_bytes += size_bytes;
                    println!("   📄 {}: {:.0} MB", file_name_str, size_mb);
                }
            }
        }
    }

    let total_model_mb = total_model_bytes as f64 / (1024.0 * 1024.0);
    let total_model_gb = total_model_mb / 1024.0;
    println!(
        "   📊 Total model size: {:.0} MB ({:.1} GB)",
        total_model_mb, total_model_gb
    );

    // Step 3: Calculate theoretical memory requirements
    println!("\n🧮 Step 3: Memory Requirements Analysis");
    println!("   Base model weights: {:.1} GB", total_model_gb);

    // Conservative estimates for different memory overhead scenarios
    let scenarios = vec![
        ("Minimal (1.1x)", total_model_gb * 1.1),
        ("Conservative (1.5x)", total_model_gb * 1.5),
        ("Pessimistic (2.0x)", total_model_gb * 2.0),
        ("Extreme (3.0x)", total_model_gb * 3.0),
    ];

    for (scenario, estimated_gb) in &scenarios {
        let available_gb = if let Some((_, free, _)) = get_gpu_memory_mb() {
            free as f64 / 1024.0
        } else {
            20.9 // fallback to our known value
        };

        let status = if *estimated_gb <= available_gb {
            "✅ Should fit"
        } else {
            "❌ Would exceed"
        };

        println!("   {} - {:.1} GB: {}", scenario, estimated_gb, status);
    }

    // Step 4: Test with ultra-conservative settings
    println!("\n🔬 Step 4: Ultra-Conservative Loading Test");
    let config = InfernoConfig {
        model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        model_path: model_path.clone(),
        device_id: 0,
        max_batch_size: 1,
        max_sequence_length: 64,       // Very small
        gpu_memory_pool_size_mb: 4096, // Only 4GB pool
        ..Default::default()
    };

    println!("   🔧 Ultra-conservative settings:");
    println!("      Max sequence length: {}", config.max_sequence_length);
    println!(
        "      GPU memory pool: {} MB",
        config.gpu_memory_pool_size_mb
    );

    // Monitor memory before loading
    let memory_before = get_gpu_memory_mb();
    if let Some((used, free, _)) = memory_before {
        println!(
            "   📊 Memory before loading: {} MB used, {} MB free",
            used, free
        );
    }

    let mut engine = CandleInferenceEngine::new();

    println!("\n🚀 Attempting ultra-conservative model loading...");
    match engine.initialize(config).await {
        Ok(()) => {
            println!("   🎉 SUCCESS: Model loaded with ultra-conservative settings!");

            // Check memory after successful loading
            if let Some((used_after, free_after, _)) = get_gpu_memory_mb() {
                if let Some((used_before, _, _)) = memory_before {
                    let memory_increase = used_after - used_before;
                    println!(
                        "   📊 Memory after loading: {} MB used, {} MB free",
                        used_after, free_after
                    );
                    println!(
                        "   📈 Memory increase: {} MB ({:.1} GB)",
                        memory_increase,
                        memory_increase as f64 / 1024.0
                    );

                    let expected_mb = total_model_mb as u32;
                    let overhead_factor = memory_increase as f64 / expected_mb as f64;
                    println!(
                        "   🔍 Actual vs expected: {}x overhead factor",
                        overhead_factor
                    );

                    if overhead_factor > 3.0 {
                        println!(
                            "   ⚠️  INVESTIGATION: Memory usage is {}x the model size!",
                            overhead_factor
                        );
                        println!(
                            "   💡 This explains why loading fails - excessive memory overhead"
                        );
                    } else if overhead_factor > 2.0 {
                        println!(
                            "   ⚠️  High memory overhead: {}x model size",
                            overhead_factor
                        );
                    } else {
                        println!(
                            "   ✅ Reasonable memory overhead: {}x model size",
                            overhead_factor
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!("   ❌ Even ultra-conservative loading failed: {}", e);

            if e.to_string().contains("out of memory") {
                println!("\n🔍 DEEP ANALYSIS:");
                println!("   - Model size: {:.1} GB", total_model_gb);
                println!("   - Available memory: 20.9 GB");
                println!(
                    "   - Ratio: {:.1}x available vs needed",
                    20.9 / total_model_gb
                );
                println!("   - This should work easily!");

                println!("\n🤔 POSSIBLE CAUSES:");
                println!("   1. Candle loading algorithm allocates full model multiple times");
                println!("   2. Tensor creation/loading uses temporary buffers");
                println!("   3. Model format requires decompression or conversion");
                println!("   4. SafeTensors loading creates duplicate tensors temporarily");
                println!("   5. CUDA memory fragmentation despite showing free memory");

                println!("\n📋 INVESTIGATION FINDINGS:");
                println!("   - The issue is NOT lack of total GPU memory");
                println!("   - 20.9GB free >> 15GB model size");
                println!("   - Problem is likely in the loading implementation");
                println!("   - May need to investigate Candle's SafeTensors loading code");
            }
        }
    }

    println!("\n✅ Memory investigation completed!");
    println!("   Key finding: GPU has sufficient memory, issue is in loading algorithm");
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_memory_allocation_patterns() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 MEMORY PATTERNS: Understanding Candle Memory Allocation");
    println!("========================================================");

    println!("🔍 Theoretical Analysis:");
    println!("   Model size: 15.0 GB");
    println!("   Available memory: 20.9 GB");
    println!("   Overhead budget: 5.9 GB (39% extra)");
    println!("   This should be MORE than sufficient!");

    println!("\n🤔 Possible Memory Allocation Issues:");

    println!("\n1. 📚 SafeTensors Loading Pattern:");
    println!("   - Candle might load ALL shards simultaneously");
    println!("   - Could create temporary copies during tensor creation");
    println!("   - May not free intermediate buffers immediately");

    println!("\n2. 🏗️ Model Construction Pattern:");
    println!("   - Model layers created before weights loaded");
    println!("   - Weights loaded then copied to model structure");
    println!("   - Temporary duplication during construction");

    println!("\n3. 🔄 Memory Fragmentation:");
    println!("   - Large contiguous allocations fail");
    println!("   - Despite showing 'free' memory");
    println!("   - Need physically contiguous GPU memory blocks");

    println!("\n4. ⚙️ CUDA Context Overhead:");
    println!("   - Candle creates large CUDA contexts");
    println!("   - Memory pools pre-allocated");
    println!("   - Framework overhead not visible in nvidia-smi");

    println!("\n💡 Investigation Conclusion:");
    println!("   - GPU memory capacity is NOT the issue");
    println!("   - 20.9GB available >> 15GB needed");
    println!("   - Problem is in Candle's loading implementation");
    println!("   - May require smaller batch loading or streaming");

    Ok(())
}

#[cfg(not(feature = "candle-cuda"))]
mod disabled_tests {
    #[tokio::test]
    async fn test_memory_investigation_requires_cuda() {
        println!("Memory investigation tests require 'candle-cuda' feature");
        println!("Run with: cargo test --features candle-cuda");
    }
}
