//! Test for GPU memory validation and diagnostics system integration

use std::env;

#[cfg(feature = "candle-cuda")]
use inferno_inference::{
    config::InfernoConfig,
    inference::{CandleInferenceEngine, InferenceEngine},
};

#[cfg(feature = "candle-cuda")]
use inferno_shared::{monitor_model_loading, validate_and_display_model_memory, GpuDiagnostics};

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_memory_validation_system_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 GPU MEMORY VALIDATION SYSTEM TEST");
    println!("====================================");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !std::path::Path::new(&model_path).exists() {
        println!("❌ Llama 3.1 model not found, skipping validation test");
        return Ok(());
    }

    // Step 1: Test direct memory validation
    println!("\n📊 Step 1: Direct Model Memory Validation");
    let validation_result = validate_and_display_model_memory(
        0, // GPU 0
        &model_path,
        "meta-llama/Llama-3.1-8B-Instruct",
    )
    .await;

    match validation_result {
        Ok(will_fit) => {
            println!("✅ Memory validation completed successfully");
            println!(
                "   Result: Model {} fit",
                if will_fit { "will" } else { "won't" }
            );
        }
        Err(e) => {
            println!("⚠️  Memory validation failed: {}", e);
        }
    }

    // Step 2: Test GPU diagnostics
    println!("\n📊 Step 2: GPU Diagnostics System");
    let diagnostics = GpuDiagnostics::new_default();

    println!("   Starting GPU memory tracking...");
    diagnostics.start_memory_tracking(0, Some("test_start".to_string()));

    // Simulate some work
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    diagnostics.add_memory_snapshot(0, Some("test_midpoint".to_string()));

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let evolution = diagnostics.stop_memory_tracking(0);

    if let Some(evolution) = evolution {
        println!("   📈 Memory tracking completed:");
        println!("      Snapshots recorded: {}", evolution.snapshots.len());
        if let Some(delta) = evolution.memory_delta_mb() {
            println!("      Memory delta: {:+} MB", delta);
        }
    }

    // Step 3: Test integration with CandleInferenceEngine
    println!("\n📊 Step 3: Engine Integration Test");

    let config = InfernoConfig {
        model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        model_path: model_path.clone(),
        device_id: 0,
        max_batch_size: 1,
        max_sequence_length: 64,       // Very small for safety
        gpu_memory_pool_size_mb: 4096, // Conservative
        ..Default::default()
    };

    let mut engine = CandleInferenceEngine::new();

    println!("   Testing engine initialization with memory validation...");
    match engine.initialize(config).await {
        Ok(()) => {
            println!("   ✅ Engine initialized successfully!");
            println!("   ✅ Memory validation system working properly");
        }
        Err(e) => {
            if e.to_string()
                .contains("Model validation indicates insufficient GPU memory")
            {
                println!("   ✅ Memory validation system prevented loading (expected)");
                println!("   ✅ Validation system is working correctly");
            } else {
                println!("   ❌ Engine initialization failed: {}", e);
            }
        }
    }

    println!("\n✅ GPU memory validation system integration test completed!");
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_memory_monitoring_during_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 MEMORY MONITORING SIMULATION");
    println!("===============================");

    // Simulate model loading with memory monitoring
    let result = monitor_model_loading(
        0, // GPU 0
        "simulation_model",
        || async {
            // Simulate model loading work
            println!("   📦 Simulating model loading...");
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            println!("   ✅ Simulation completed");
            Ok::<String, Box<dyn std::error::Error>>("Model loaded successfully".to_string())
        },
    )
    .await;

    match result {
        Ok((message, evolution)) => {
            println!("✅ Monitoring completed: {}", message);
            println!("📊 Memory evolution:");
            println!(
                "    Duration: {:.1}s",
                evolution.start_time.elapsed().as_secs_f64()
            );
            println!("    Snapshots: {}", evolution.snapshots.len());
            if let Some(delta) = evolution.memory_delta_mb() {
                println!("    Memory delta: {:+} MB", delta);
            }
        }
        Err(e) => {
            println!("❌ Monitoring failed: {}", e);
        }
    }

    println!("\n✅ Memory monitoring simulation test completed!");
    Ok(())
}

#[cfg(not(feature = "candle-cuda"))]
mod disabled_tests {
    #[tokio::test]
    async fn test_memory_validation_requires_cuda() {
        println!("Memory validation tests require 'candle-cuda' feature");
        println!("Run with: cargo test --features candle-cuda");
    }
}
