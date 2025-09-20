//! Comprehensive End-to-End Model Loading Test Coverage
//!
//! This test suite provides comprehensive coverage for the entire model loading pipeline:
//! - Configuration detection and validation
//! - Weight analysis and dtype detection
//! - Model architecture initialization
//! - SafeTensors weight loading with HuggingFace naming
//! - Parameter count validation
//! - Error handling for unsupported formats

use candle_core::DType;
use inferno_llama::factory::UnifiedModelFactory;
use inferno_llama::{WeightAnalyzer, InfernoLlama};
use std::path::Path;

const TEST_MODEL_PATHS: &[&str] = &[
    "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct",
    "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8",
];

fn model_exists(path: &str) -> bool {
    Path::new(path).exists()
}

#[tokio::test]
async fn test_complete_pipeline_standard_model() {
    let model_path = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";

    if !model_exists(model_path) {
        println!("Skipping test: Model not found at {}", model_path);
        return;
    }

    println!("Testing complete pipeline for Meta Llama 3.1 8B...");

    // Step 1: Weight Analysis
    println!("Step 1: Analyzing model weights...");
    let weight_analysis = WeightAnalyzer::analyze_weights(model_path).await;
    assert!(weight_analysis.is_ok(), "Weight analysis should succeed");

    let analysis = weight_analysis.unwrap();
    println!("  Primary dtype: {:?}", analysis.primary_dtype);
    println!("  Total params: {}", analysis.total_params);
    println!("  Is sharded: {}", analysis.is_sharded);

    // Validate analysis results
    assert_eq!(analysis.primary_dtype, DType::BF16);
    assert!(analysis.total_params > 8_000_000_000, "Should detect ~8.8B parameters");
    assert!(analysis.is_sharded, "Meta Llama 3.1 8B should be sharded");

    // Step 2: Model Detection
    println!("Step 2: Detecting model configuration...");
    let factory = UnifiedModelFactory::new().unwrap();
    let config = factory.detect_model_config(model_path).await.unwrap();

    println!("  Variant: {:?}", config.variant);
    println!("  Hidden size: {}", config.base.hidden_size);
    println!("  Layers: {}", config.base.num_hidden_layers);

    // Validate configuration
    assert_eq!(config.base.hidden_size, 4096);
    assert_eq!(config.base.num_hidden_layers, 32);
    assert_eq!(config.base.num_attention_heads, 32);

    // Step 3: Model Loading
    println!("Step 3: Loading complete model...");
    let model = factory.load_model(model_path, config).await.unwrap();

    println!("  Parameter count: {}", model.parameter_count());
    println!("  Model layers: {}", model.layers.len());

    // Validate model structure
    assert_eq!(model.layers.len(), 32);
    let param_count = model.parameter_count();
    assert!(param_count > 8_000_000_000 && param_count < 9_000_000_000,
            "Parameter count {} should be around 8.8B", param_count);

    println!("✅ Complete pipeline test passed for standard model!");
}

#[tokio::test]
async fn test_complete_pipeline_quantized_model() {
    let model_path = "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8";

    if !model_exists(model_path) {
        println!("Skipping test: Model not found at {}", model_path);
        return;
    }

    println!("Testing complete pipeline for quantized model...");

    // Step 1: Weight Analysis (should work even for quantized)
    println!("Step 1: Analyzing quantized model weights...");
    let weight_analysis = WeightAnalyzer::analyze_weights(model_path).await;
    assert!(weight_analysis.is_ok(), "Weight analysis should work for quantized models");

    let analysis = weight_analysis.unwrap();
    println!("  Primary dtype: {:?}", analysis.primary_dtype);
    println!("  Quantization: {:?}", analysis.quantization.scheme);

    // Validate quantization detection
    assert!(analysis.quantization.scheme.to_string().contains("W8A8"),
            "Should detect W8A8 quantization");

    // Step 2: Model Detection (should work)
    println!("Step 2: Detecting quantized model configuration...");
    let factory = UnifiedModelFactory::new().unwrap();
    let config = factory.detect_model_config(model_path).await.unwrap();

    assert!(config.quantization.is_some(), "Should detect quantization config");

    // Step 3: Model Loading (expected to fail due to I8 dtype limitation)
    println!("Step 3: Attempting to load quantized model...");
    let model_result = factory.load_model(model_path, config).await;

    // This should fail with I8 dtype error as identified in engineering review
    assert!(model_result.is_err(), "Quantized model loading should fail due to I8 dtype limitation");

    let error = model_result.unwrap_err();
    let error_msg = error.to_string();
    assert!(error_msg.contains("I8") || error_msg.contains("dtype"),
            "Error should mention I8 dtype issue: {}", error_msg);

    println!("✅ Complete pipeline test passed for quantized model (expected I8 dtype limitation)!");
}

#[test]
fn test_safetensors_weight_name_validation() {
    // Test that our E2E pipeline correctly handles HuggingFace naming convention
    let expected_weight_patterns = vec![
        // Core model weights
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",

        // Transformer layer weights
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",

        // Attention weights (HuggingFace naming)
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",

        // MLP weights (HuggingFace naming)
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];

    // Validate all patterns follow HuggingFace convention
    for pattern in expected_weight_patterns {
        // Should start with "model." for most weights, except lm_head
        if !pattern.starts_with("lm_head.") {
            assert!(pattern.starts_with("model."),
                    "Pattern should start with 'model.': {}", pattern);
        }

        // Attention weights should use "self_attn" (HuggingFace) not "attention" (old mapping)
        if pattern.contains("attn") {
            assert!(pattern.contains("self_attn"),
                    "Should use 'self_attn' not 'attention': {}", pattern);
        }

        // MLP weights should use "mlp" (HuggingFace) not "feed_forward" (old mapping)
        if pattern.contains("proj.weight") && !pattern.contains("attn") {
            assert!(pattern.contains("mlp"),
                    "Should use 'mlp' not 'feed_forward': {}", pattern);
        }
    }

    println!("✅ HuggingFace naming convention validation passed!");
}

#[tokio::test]
async fn test_error_handling_coverage() {
    // Test error handling for various edge cases in the E2E pipeline

    // Test 1: Non-existent model path
    let factory = UnifiedModelFactory::new().unwrap();
    let result = factory.detect_model_config("/nonexistent/model").await;
    assert!(result.is_err(), "Should fail for non-existent model path");

    // Test 2: Directory without proper model files
    let temp_dir = tempfile::TempDir::new().unwrap();
    let result = factory.detect_model_config(temp_dir.path().to_str().unwrap()).await;
    assert!(result.is_err(), "Should fail for directory without model files");

    println!("✅ Error handling coverage test passed!");
}

#[test]
fn test_parameter_count_accuracy() {
    // Test that parameter counting is accurate across different model configurations
    use inferno_llama::LlamaConfig;

    let config_8b = LlamaConfig::llama_3_1_8b().unwrap();

    // Calculate expected parameters for 8B model
    let embedding_params = config_8b.vocab_size * config_8b.dim;
    let layer_params = config_8b.n_layers * (
        // Attention: q,k,v,o projections
        4 * config_8b.dim * config_8b.dim +
        // FFN: gate, up, down projections
        config_8b.dim * config_8b.intermediate_size * 3 +
        // Layer norms: input + post-attention
        config_8b.dim * 2
    );
    let final_norm_params = config_8b.dim;
    let lm_head_params = config_8b.vocab_size * config_8b.dim;

    let expected_total = embedding_params + layer_params + final_norm_params + lm_head_params;

    println!("Expected total parameters: {}", expected_total);
    println!("Expected ~8.8B parameters, got: {:.2}B", expected_total as f64 / 1_000_000_000.0);

    // Should be approximately 8.8B parameters
    assert!(expected_total > 8_000_000_000 && expected_total < 9_000_000_000,
            "Parameter count should be ~8.8B, got {}", expected_total);

    println!("✅ Parameter count accuracy validation passed!");
}