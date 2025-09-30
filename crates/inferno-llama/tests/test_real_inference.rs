//! Real inference tests for InfernoLlama
//!
//! These tests validate actual tensor operations and forward pass functionality.
//! All tests should initially fail until the real inference engine is implemented.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarMap;
use inferno_llama::*;

/// Helper function to create a test VarBuilder with minimal memory allocation
fn create_test_var_builder(device: &Device, dtype: DType) -> candle_nn::VarBuilder<'_> {
    let vs = VarMap::new();
    candle_nn::VarBuilder::from_varmap(&vs, dtype, device)
}

/// Helper function to create a minimal test config for faster testing
fn create_minimal_test_config() -> Result<LlamaConfig> {
    // Create a very small model for fast testing
    let config = LlamaConfig {
        dim: 256,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: Some(2),
        vocab_size: 1000,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        ..Default::default()
    };

    config.validate()?;
    Ok(config)
}

#[tokio::test]
async fn test_real_forward_pass_with_actual_tensors() {
    // This test should fail until we implement real forward pass functionality
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::F32);

    // Create model - this should work since we have the structure
    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Create real input tensor with valid token IDs
    let batch_size = 1;
    let seq_len = 5;
    let token_ids = vec![1u32, 2, 3, 4, 5]; // Simple token sequence

    let input_tensor = Tensor::from_slice(&token_ids, (batch_size, seq_len), &device)
        .expect("Input tensor creation should succeed");

    // This should fail until we implement real forward pass
    let result = model.forward(&input_tensor, 0);

    match result {
        Ok(logits) => {
            // If this passes, check that the output has correct shape
            let expected_shape = &[batch_size, seq_len, config.vocab_size];
            assert_eq!(
                logits.dims(),
                expected_shape,
                "Output logits should have shape [batch_size, seq_len, vocab_size]"
            );

            // Verify logits are real numbers (not NaN/Inf)
            let logits_vec = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                logits_vec.iter().all(|&x| x.is_finite()),
                "All logits should be finite numbers"
            );
        }
        Err(e) => {
            panic!(
                "Forward pass failed - this is expected until real inference is implemented: {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_inference_with_different_sequence_lengths() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::F32);

    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Test different sequence lengths
    let test_sequences = [
        vec![1u32],                             // Single token
        vec![1u32, 2, 3],                       // Short sequence
        vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10], // Longer sequence
    ];

    for (i, token_ids) in test_sequences.iter().enumerate() {
        let batch_size = 1;
        let seq_len = token_ids.len();

        let input_tensor = Tensor::from_slice(token_ids, (batch_size, seq_len), &device)
            .expect("Input tensor creation should succeed");

        let result = model.forward(&input_tensor, 0);

        match result {
            Ok(logits) => {
                let expected_shape = &[batch_size, seq_len, config.vocab_size];
                assert_eq!(
                    logits.dims(),
                    expected_shape,
                    "Sequence {} should produce correct output shape",
                    i
                );
            }
            Err(e) => {
                panic!(
                    "Forward pass failed for sequence {}: {} - implement real inference",
                    i, e
                );
            }
        }
    }
}

#[tokio::test]
async fn test_batched_inference() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::F32);

    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Test batch processing
    let batch_size = 3;
    let seq_len = 4;
    let token_ids = vec![
        1u32, 2, 3, 4, // First sequence
        5, 6, 7, 8, // Second sequence
        9, 10, 1, 2, // Third sequence
    ];

    let input_tensor = Tensor::from_slice(&token_ids, (batch_size, seq_len), &device)
        .expect("Batched input tensor creation should succeed");

    let result = model.forward(&input_tensor, 0);

    match result {
        Ok(logits) => {
            let expected_shape = &[batch_size, seq_len, config.vocab_size];
            assert_eq!(
                logits.dims(),
                expected_shape,
                "Batched inference should produce correct output shape"
            );
        }
        Err(e) => {
            panic!(
                "Batched inference failed: {} - implement real batched forward pass",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_kv_caching_with_start_pos() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::F32);

    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Test KV caching by running inference with different start positions
    let token_ids = vec![1u32, 2, 3];
    let batch_size = 1;
    let seq_len = token_ids.len();

    let input_tensor = Tensor::from_slice(&token_ids, (batch_size, seq_len), &device)
        .expect("Input tensor creation should succeed");

    // First inference from position 0
    let result1 = model.forward(&input_tensor, 0);

    // Second inference from position 3 (as if extending the sequence)
    let result2 = model.forward(&input_tensor, 3);

    match (result1, result2) {
        (Ok(logits1), Ok(logits2)) => {
            // Both should have same output shape
            assert_eq!(
                logits1.dims(),
                logits2.dims(),
                "KV caching should maintain consistent output shape"
            );
        }
        (Err(e), _) | (_, Err(e)) => {
            panic!(
                "KV caching inference failed: {} - implement position encoding",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_precision_handling_bf16() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");

    // Test with F16 precision (BF16 has limited CPU support in candle-core)
    let vb = create_test_var_builder(&device, DType::F16);

    let model_result = InfernoLlama::new(&config, vb);

    match model_result {
        Ok(model) => {
            let token_ids = vec![1u32, 2, 3];
            let batch_size = 1;
            let seq_len = token_ids.len();

            let input_tensor = Tensor::from_slice(&token_ids, (batch_size, seq_len), &device)
                .expect("Input tensor creation should succeed");

            let result = model.forward(&input_tensor, 0);

            match result {
                Ok(logits) => {
                    // Verify output maintains F16 precision
                    assert_eq!(
                        logits.dtype(),
                        DType::F16,
                        "Output should maintain F16 precision"
                    );
                }
                Err(e) => {
                    panic!("F16 inference failed: {} - implement precision handling", e);
                }
            }
        }
        Err(e) => {
            panic!("F16 model creation failed: {} - implement F16 support", e);
        }
    }
}

#[tokio::test]
async fn test_text_generation_interface() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::F32);

    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Test the text generation interface
    let input_tokens = vec![1u32, 2, 3]; // Start tokens

    let result = model.forward_from_token_ids(&input_tokens, 0);

    match result {
        Ok(logits) => {
            // Should be able to get next token probabilities
            let batch_size = 1;
            let seq_len = input_tokens.len();
            let expected_shape = &[batch_size, seq_len, config.vocab_size];
            assert_eq!(logits.dims(), expected_shape);

            // Test getting the next token (greedy sampling)
            let last_logits = logits.i((0, seq_len - 1, ..)).unwrap();
            let next_token = last_logits.argmax(0).unwrap();

            // Should get a valid token ID
            let token_id = next_token.to_scalar::<u32>().unwrap();
            assert!(
                token_id < config.vocab_size as u32,
                "Next token should be within vocabulary"
            );
        }
        Err(e) => {
            panic!(
                "Text generation interface failed: {} - implement sampling",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_memory_efficiency() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::BF16);

    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Test memory usage estimation
    let batch_size = 1;
    let seq_len = 100;
    let estimated_memory = model.estimated_memory_usage(batch_size, seq_len);

    // Should provide reasonable memory estimates
    assert!(estimated_memory > 0, "Memory estimate should be positive");
    assert!(
        estimated_memory < 1_000_000_000,
        "Memory estimate should be reasonable for tiny model"
    );

    // Test actual parameter count
    let param_count = model.parameter_count();
    assert!(param_count > 0, "Parameter count should be positive");

    // For our minimal config, parameter count should be reasonable
    let expected_range = 100_000..10_000_000;
    assert!(
        expected_range.contains(&param_count),
        "Parameter count {} should be in reasonable range for minimal model",
        param_count
    );
}

#[tokio::test]
async fn test_error_handling_invalid_inputs() {
    let device = Device::Cpu;
    let config = create_minimal_test_config().expect("Config creation should succeed");
    let vb = create_test_var_builder(&device, DType::F32);

    let model = InfernoLlama::new(&config, vb).expect("Model creation should succeed");

    // Test with invalid token IDs (out of vocabulary)
    let invalid_tokens = vec![config.vocab_size as u32 + 1000]; // Way out of range
    let batch_size = 1;
    let seq_len = invalid_tokens.len();

    let input_tensor = Tensor::from_slice(&invalid_tokens, (batch_size, seq_len), &device)
        .expect("Input tensor creation should succeed");

    let result = model.forward(&input_tensor, 0);

    // Should either handle gracefully or provide clear error
    match result {
        Ok(_) => {
            // If it passes, the model should handle invalid tokens gracefully
            println!("Model handles invalid tokens gracefully");
        }
        Err(e) => {
            // Should provide clear error about invalid tokens
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("token")
                    || error_msg.contains("vocab")
                    || error_msg.contains("index"),
                "Error should mention token/vocab/index issue: {}",
                error_msg
            );
        }
    }
}
