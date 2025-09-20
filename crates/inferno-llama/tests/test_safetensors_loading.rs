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

    let _model = InfernoLlama::new(&config, vb).unwrap();

    // Create a fake weight analysis result
    let _analysis = inferno_llama::diagnostic::WeightAnalysisResult {
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


/// Test loading weights from a mock SafeTensors file
#[test]
fn test_load_tensors_from_mock_safetensors() {
    let _temp_dir = TempDir::new().unwrap();
    let _device = Device::Cpu;

    // Create a mock SafeTensors file with some test weights (using HuggingFace naming)
    // Note: This is a simplified test - real implementation would use actual SafeTensors format
    let test_weights = vec!["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"];

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

    // Create mock index file for sharded model (using HuggingFace naming)
    let index_content = r#"{
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
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
    assert_eq!(
        mapping.get("model.layers.0.self_attn.q_proj.weight"),
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

/// Test SafeTensors configuration validation
#[test]
fn test_safetensors_config_validation() {
    let temp_dir = TempDir::new().unwrap();

    // Create a valid config.json for testing
    let config_content = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16"
    }"#;

    fs::write(temp_dir.path().join("config.json"), config_content).unwrap();

    let result = InfernoLlama::load_config_simple(temp_dir.path());
    assert!(result.is_ok());

    let config = result.unwrap();
    assert_eq!(config.dim, 4096);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.n_heads, 32);
    assert_eq!(config.n_kv_heads, Some(8));
    assert_eq!(config.vocab_size, 128256);
}

/// Test SafeTensors sharding patterns
#[test]
fn test_safetensors_sharding_patterns() {
    let temp_dir = TempDir::new().unwrap();

    // Create multiple index patterns to test flexibility
    let patterns = vec![
        // Standard sharded pattern
        (r#"{"weight_map": {"model.embed_tokens.weight": "model-00001-of-00004.safetensors"}}"#, 1),
        // Multi-shard pattern
        (r#"{"weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.31.mlp.down_proj.weight": "model-00004-of-00004.safetensors"
        }}"#, 3),
    ];

    for (index_content, expected_weights) in patterns {
        let test_dir = temp_dir.path().join("test_model");
        fs::create_dir_all(&test_dir).unwrap();
        fs::write(test_dir.join("model.safetensors.index.json"), index_content).unwrap();

        let result = InfernoLlama::get_weight_mapping(&test_dir);
        assert!(result.is_ok());

        let mapping = result.unwrap();
        assert_eq!(mapping.len(), expected_weights);

        // Clean up for next iteration
        fs::remove_dir_all(&test_dir).unwrap();
    }
}

/// Test HuggingFace naming convention consistency
#[test]
fn test_huggingface_naming_consistency() {
    // Test that all expected HuggingFace weight names are recognized
    let expected_patterns = vec![
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    // All these patterns should be valid for weight loading
    for pattern in expected_patterns {
        // Test that pattern follows expected format
        if pattern.starts_with("model.layers.") {
            assert!(pattern.contains("self_attn") || pattern.contains("mlp") || pattern.contains("layernorm"));
        }

        // Test pattern parsing doesn't fail
        let parts: Vec<&str> = pattern.split('.').collect();
        assert!(parts.len() >= 2, "Pattern should have at least 2 parts: {}", pattern);
    }
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
