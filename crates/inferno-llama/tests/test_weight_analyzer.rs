//! # Weight Analyzer Tests
//!
//! Tests for real SafeTensors weight analysis functionality that should replace
//! the stub implementation in WeightAnalyzer::analyze_weights.

use inferno_llama::diagnostic::{QuantizationScheme, WeightAnalyzer};
use std::path::Path;

const MODEL_PATH: &str = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";
const QUANTIZED_MODEL_PATH: &str =
    "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8";

#[tokio::test]
async fn test_analyze_weights_real_model() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test - model not found at {}", MODEL_PATH);
        return;
    }

    // This should fail initially because analyze_weights returns an error
    let result = WeightAnalyzer::analyze_weights(MODEL_PATH).await;
    assert!(
        result.is_ok(),
        "analyze_weights should work with real model: {:?}",
        result.err()
    );

    let analysis = result.unwrap();

    // Validate the analysis results
    assert_eq!(
        analysis.primary_dtype,
        candle_core::DType::BF16,
        "Should detect BF16 as primary dtype"
    );
    assert!(
        analysis.total_params > 7_000_000_000,
        "Should detect ~8B parameters, got {}",
        analysis.total_params
    );
    assert!(analysis.is_sharded, "Llama 3.1 8B should be sharded");
    assert!(
        analysis.num_shards >= 2,
        "Should have multiple shards, got {}",
        analysis.num_shards
    );
    assert!(
        analysis.estimated_memory_bytes > 10_000_000_000,
        "Should estimate >10GB memory"
    );
}

#[tokio::test]
async fn test_analyze_weights_quantized_model() {
    let model_path = Path::new(QUANTIZED_MODEL_PATH);
    if !model_path.exists() {
        eprintln!(
            "Skipping test - quantized model not found at {}",
            QUANTIZED_MODEL_PATH
        );
        return;
    }

    let result = WeightAnalyzer::analyze_weights(QUANTIZED_MODEL_PATH).await;
    assert!(
        result.is_ok(),
        "analyze_weights should work with quantized model: {:?}",
        result.err()
    );

    let analysis = result.unwrap();

    // Validate quantized model properties
    assert_eq!(
        analysis.quantization.scheme,
        QuantizationScheme::W8A8,
        "Should detect W8A8 quantization"
    );
    assert!(
        analysis.total_params > 1_000_000_000,
        "Should detect ~1B parameters, got {}",
        analysis.total_params
    );
    assert!(
        analysis.estimated_memory_bytes < analysis.total_params * 2,
        "Quantized model should use less memory"
    );
}

#[tokio::test]
async fn test_detect_quantization_from_path() {
    // Test W8A8 detection
    let result = WeightAnalyzer::detect_quantization(QUANTIZED_MODEL_PATH).await;
    assert!(result.is_ok(), "Should detect quantization from path");

    let config = result.unwrap();
    assert_eq!(config.scheme, QuantizationScheme::W8A8);
    assert!(config.symmetric);
}

#[tokio::test]
async fn test_is_sharded_detection() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test - model not found at {}", MODEL_PATH);
        return;
    }

    let result = WeightAnalyzer::is_sharded(MODEL_PATH).await;
    assert!(result.is_ok(), "Should detect sharding: {:?}", result.err());

    let (is_sharded, num_shards) = result.unwrap();
    assert!(is_sharded, "Llama 3.1 8B should be detected as sharded");
    assert!(
        num_shards >= 2,
        "Should have at least 2 shards, got {}",
        num_shards
    );
}

#[tokio::test]
async fn test_weight_file_enumeration() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test - model not found at {}", MODEL_PATH);
        return;
    }

    // Test that we can enumerate all weight files
    let result = WeightAnalyzer::analyze_weights(MODEL_PATH).await;
    assert!(result.is_ok());

    let analysis = result.unwrap();

    // Should find all the expected .safetensors files
    assert!(analysis.num_shards >= 2, "Should find multiple shard files");

    // Memory estimation should be reasonable for 8B model
    let memory_gb = analysis.estimated_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!(
        memory_gb > 10.0 && memory_gb < 30.0,
        "Memory estimate should be 10-30GB, got {:.1}GB",
        memory_gb
    );
}
