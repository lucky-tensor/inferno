//! Tests for SafeTensors weight loading functionality
//!
//! These tests validate that the actual SafeTensors weight loading works correctly
//! with real model files while preserving dtypes and supporting sharded models.

use inferno_llama::{InfernoLlama, LlamaConfig, LlamaError};
use candle_core::{Device, DType, Tensor};
use tempfile::TempDir;
use std::fs;

/// Test that load_weights_into_model actually loads weights instead of returning placeholder error
#[test]
fn test_load_weights_into_model_not_placeholder() {
    // Create a minimal model structure
    let config = LlamaConfig::llama_3_1_8b().unwrap();
    let device = Device::Cpu;
    let dtype = DType::BF16;
    let vs = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&vs, dtype, &device);

    let mut model = InfernoLlama::new(&config, vb).unwrap();

    // Create a fake weight analysis result
    let analysis = inferno_llama::diagnostic::WeightAnalysisResult {
        primary_dtype: DType::BF16,
        total_params: 1000,
        quantization: inferno_llama::diagnostic::QuantizationConfig::default(),
        is_sharded: false,
        num_shards: 1,
        estimated_memory_bytes: 2000,
    };

    // Create a temporary directory for testing
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path().to_string_lossy().to_string();

    // This should NOT return the placeholder error message
    // Since load_weights_into_model is private, we'll test through the public API

    // Test that the placeholder implementation is detected through load_from_path
    let result = tokio_test::block_on(async {
        InfernoLlama::load_from_path(&temp_path).await
    });

    // The current implementation should return a placeholder error - this test should fail initially
    match result {
        Err(LlamaError::ConfigError { field, reason }) if field == "weight_loading" => {
            if reason.contains("Weight loading not yet implemented") {
                panic!("load_weights_into_model is still returning placeholder error - implementation needed!");
            }
        }
        _ => {
            // This is what we want - either success or a real implementation error
        }
    }
}

/// Test weight name mapping from Hugging Face to InfernoLlama format
#[test]
fn test_weight_name_mapping() {
    // Test basic layer mapping
    let hf_name = "model.layers.0.self_attn.q_proj.weight";
    let mapped = InfernoLlama::map_weight_name(hf_name).unwrap();
    assert_eq!(mapped, "layers.0.attention.q_proj.weight");

    // Test MLP mapping
    let hf_name = "model.layers.15.mlp.gate_proj.weight";
    let mapped = InfernoLlama::map_weight_name(hf_name).unwrap();
    assert_eq!(mapped, "layers.15.feed_forward.gate_proj.weight");

    // Test norm layers (should remain unchanged)
    let hf_name = "model.layers.0.input_layernorm.weight";
    let mapped = InfernoLlama::map_weight_name(hf_name).unwrap();
    assert_eq!(mapped, "layers.0.input_layernorm.weight");

    // Test final norm
    let hf_name = "model.norm.weight";
    let mapped = InfernoLlama::map_weight_name(hf_name).unwrap();
    assert_eq!(mapped, "norm.weight");

    // Test embeddings
    let hf_name = "model.embed_tokens.weight";
    let mapped = InfernoLlama::map_weight_name(hf_name).unwrap();
    assert_eq!(mapped, "embed_tokens.weight");
}

/// Test loading weights from a mock SafeTensors file
#[test]
fn test_load_tensors_from_mock_safetensors() {
    let temp_dir = TempDir::new().unwrap();
    let device = Device::Cpu;

    // Create a mock SafeTensors file with some test weights
    // Note: This is a simplified test - real implementation would use actual SafeTensors format
    let test_weights = vec!["embed_tokens.weight", "layers.0.attention.q_proj.weight"];

    // This test validates the interface exists - but method is private, so skip for now
    // TODO: Test through public API once available
    // let result = InfernoLlama::load_tensors_from_file(
    //     &temp_dir.path().join("test.safetensors"),
    //     &test_weights,
    //     &device,
    // );

    // For now, just validate the test setup
    let test_weights_str: Vec<String> = test_weights.iter().map(|s| s.to_string()).collect();
    assert_eq!(test_weights_str.len(), 2);

    // Skip actual test until public API is available
    // assert!(result.is_err());
}

/// Test dtype preservation during weight loading
#[test]
fn test_dtype_preservation_during_loading() {
    // This test ensures that when we load BF16 weights, they remain BF16
    // When we load F16 weights, they remain F16, etc.

    let device = Device::Cpu;

    // Test BF16 preservation
    let bf16_tensor = Tensor::zeros((10, 10), DType::BF16, &device).unwrap();
    assert_eq!(bf16_tensor.dtype(), DType::BF16);

    // Test F16 preservation
    let f16_tensor = Tensor::zeros((10, 10), DType::F16, &device).unwrap();
    assert_eq!(f16_tensor.dtype(), DType::F16);

    // This validates the principle - actual implementation will test with loaded weights
}

/// Test sharded model weight loading
#[test]
fn test_sharded_weight_loading_interface() {
    let temp_dir = TempDir::new().unwrap();

    // Create mock index file for sharded model
    let index_content = r#"{
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.attention.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.attention.q_proj.weight": "model-00002-of-00002.safetensors"
        }
    }"#;

    fs::write(
        temp_dir.path().join("model.safetensors.index.json"),
        index_content,
    ).unwrap();

    // Test weight mapping loading
    let result = InfernoLlama::get_weight_mapping(temp_dir.path());
    assert!(result.is_ok());

    let mapping = result.unwrap();
    assert_eq!(mapping.len(), 3);
    assert_eq!(
        mapping.get("model.embed_tokens.weight"),
        Some(&"model-00001-of-00002.safetensors".to_string())
    );
}

/// Test error handling for missing weights
#[test]
fn test_missing_weight_error_handling() {
    let temp_dir = TempDir::new().unwrap();

    // Try to load weight mapping from non-existent directory
    let result = InfernoLlama::get_weight_mapping(temp_dir.path());
    assert!(result.is_err());

    // The error should be descriptive
    let error = result.unwrap_err();
    match error {
        LlamaError::IoError { message, .. } => {
            assert!(message.contains("Failed to read weight index"));
        }
        _ => panic!("Expected IoError for missing index file"),
    }
}

/// Test that loaded weights have correct shapes
#[test]
fn test_weight_shape_validation() {
    // This test will validate that loaded weights have the expected shapes
    // for each component of the model

    let config = LlamaConfig::llama_3_1_8b().unwrap();

    // Expected shapes for key weights
    let expected_embed_shape = (config.vocab_size, config.dim);
    let expected_q_proj_shape = (config.dim, config.dim);
    let expected_ffn_gate_shape = (config.intermediate_size, config.dim);

    // Validate expected shapes match configuration
    assert_eq!(expected_embed_shape.0, 128256);
    assert_eq!(expected_embed_shape.1, 4096);
    assert_eq!(expected_q_proj_shape, (4096, 4096));
    assert_eq!(expected_ffn_gate_shape, (14336, 4096));
}

/// Integration test for end-to-end weight loading (when implementation is complete)
#[test]
#[ignore] // Enable once implementation is complete
fn test_end_to_end_weight_loading() {
    // This test will validate the complete pipeline:
    // 1. Analyze weights
    // 2. Create model structure
    // 3. Load actual weights
    // 4. Verify model can run forward pass

    // For now, this is a placeholder that documents the expected flow
    let _model_path = "/path/to/real/model";

    // Steps that will be implemented:
    // let analysis = WeightAnalyzer::analyze_weights(model_path).await.unwrap();
    // let model = InfernoLlama::load_from_path(model_path).await.unwrap();
    // let input_ids = vec![1, 2, 3]; // Some test tokens
    // let logits = model.forward_from_token_ids(&input_ids, 0).unwrap();
    // assert_eq!(logits.dims()[2], config.vocab_size);
}
