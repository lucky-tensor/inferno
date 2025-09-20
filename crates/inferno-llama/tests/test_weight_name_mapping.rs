//! Comprehensive Weight Name Mapping Tests
//!
//! This test suite validates the mapping between Hugging Face SafeTensors weight names
//! and InfernoLlama internal weight names. Based on actual SafeTensors file analysis
//! from real model files.
//!
//! Test data comes from analysis of:
//! - RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8 (259 tensors, W8A8 quantized)
//! - Meta Llama 3.1/3.2 models (BF16/F16 standard precision)
//! - TinyLlama models (various precisions)

use inferno_llama::InfernoLlama;

/// Test basic component name mapping
#[test]
fn test_basic_weight_name_mapping() {
    // Test embedding weights
    assert_eq!(
        InfernoLlama::map_weight_name("model.embed_tokens.weight").unwrap(),
        "embed_tokens.weight"
    );

    // Test output head (no model. prefix)
    assert_eq!(
        InfernoLlama::map_weight_name("lm_head.weight").unwrap(),
        "lm_head.weight"
    );

    // Test final normalization
    assert_eq!(
        InfernoLlama::map_weight_name("model.norm.weight").unwrap(),
        "norm.weight"
    );
}

/// Test attention weight mapping
#[test]
fn test_attention_weight_mapping() {
    // Query projection
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.self_attn.q_proj.weight").unwrap(),
        "layers.0.attention.q_proj.weight"
    );

    // Key projection
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.15.self_attn.k_proj.weight").unwrap(),
        "layers.15.attention.k_proj.weight"
    );

    // Value projection
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.self_attn.v_proj.weight").unwrap(),
        "layers.0.attention.v_proj.weight"
    );

    // Output projection
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.12.self_attn.o_proj.weight").unwrap(),
        "layers.12.attention.o_proj.weight"
    );
}

/// Test MLP/feed-forward weight mapping
#[test]
fn test_mlp_weight_mapping() {
    // Gate projection (SwiGLU activation)
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.mlp.gate_proj.weight").unwrap(),
        "layers.0.feed_forward.gate_proj.weight"
    );

    // Up projection
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.9.mlp.up_proj.weight").unwrap(),
        "layers.9.feed_forward.up_proj.weight"
    );

    // Down projection
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.mlp.down_proj.weight").unwrap(),
        "layers.0.feed_forward.down_proj.weight"
    );
}

/// Test layer normalization weight mapping
#[test]
fn test_normalization_weight_mapping() {
    // Input layer normalization (pre-attention)
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.input_layernorm.weight").unwrap(),
        "layers.0.input_layernorm.weight"
    );

    // Post-attention layer normalization
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.5.post_attention_layernorm.weight").unwrap(),
        "layers.5.post_attention_layernorm.weight"
    );
}

/// Test quantized model weight mapping
///
/// Quantized models have additional _scale tensors for W8A8 quantization
#[test]
fn test_quantized_weight_mapping() {
    // Query projection scale (for quantization)
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.self_attn.q_proj.weight_scale").unwrap(),
        "layers.0.attention.q_proj.weight_scale"
    );

    // Key projection scale
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.10.self_attn.k_proj.weight_scale").unwrap(),
        "layers.10.attention.k_proj.weight_scale"
    );

    // Value projection scale
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.self_attn.v_proj.weight_scale").unwrap(),
        "layers.0.attention.v_proj.weight_scale"
    );

    // Output projection scale
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.2.self_attn.o_proj.weight_scale").unwrap(),
        "layers.2.attention.o_proj.weight_scale"
    );

    // MLP gate projection scale
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.mlp.gate_proj.weight_scale").unwrap(),
        "layers.0.feed_forward.gate_proj.weight_scale"
    );

    // MLP down projection scale
    assert_eq!(
        InfernoLlama::map_weight_name("model.layers.0.mlp.down_proj.weight_scale").unwrap(),
        "layers.0.feed_forward.down_proj.weight_scale"
    );
}

/// Test edge cases and error conditions
#[test]
fn test_weight_mapping_edge_cases() {
    // Empty weight name should error
    assert!(InfernoLlama::map_weight_name("").is_err());

    // Invalid layer format
    assert!(InfernoLlama::map_weight_name("model.layers").is_err());
    assert!(InfernoLlama::map_weight_name("model.layers.").is_err());
    assert!(InfernoLlama::map_weight_name("model.layers.0").is_err());

    // Unknown component (should pass through unchanged after model. removal)
    assert_eq!(
        InfernoLlama::map_weight_name("model.unknown.weight").unwrap(),
        "unknown.weight"
    );

    // Weight name without model. prefix (should pass through)
    assert_eq!(
        InfernoLlama::map_weight_name("some.other.weight").unwrap(),
        "some.other.weight"
    );
}

/// Test comprehensive mapping for a full layer
///
/// This tests all expected weights for a single transformer layer
/// based on actual SafeTensors analysis
#[test]
fn test_complete_layer_mapping() {
    let layer_idx = 7;

    let test_cases = vec![
        // Layer normalization
        (format!("model.layers.{}.input_layernorm.weight", layer_idx),
         format!("layers.{}.input_layernorm.weight", layer_idx)),
        (format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
         format!("layers.{}.post_attention_layernorm.weight", layer_idx)),

        // Attention weights (standard)
        (format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
         format!("layers.{}.attention.q_proj.weight", layer_idx)),
        (format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
         format!("layers.{}.attention.k_proj.weight", layer_idx)),
        (format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
         format!("layers.{}.attention.v_proj.weight", layer_idx)),
        (format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
         format!("layers.{}.attention.o_proj.weight", layer_idx)),

        // Attention scales (quantized)
        (format!("model.layers.{}.self_attn.q_proj.weight_scale", layer_idx),
         format!("layers.{}.attention.q_proj.weight_scale", layer_idx)),
        (format!("model.layers.{}.self_attn.k_proj.weight_scale", layer_idx),
         format!("layers.{}.attention.k_proj.weight_scale", layer_idx)),
        (format!("model.layers.{}.self_attn.v_proj.weight_scale", layer_idx),
         format!("layers.{}.attention.v_proj.weight_scale", layer_idx)),
        (format!("model.layers.{}.self_attn.o_proj.weight_scale", layer_idx),
         format!("layers.{}.attention.o_proj.weight_scale", layer_idx)),

        // MLP weights (standard)
        (format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
         format!("layers.{}.feed_forward.gate_proj.weight", layer_idx)),
        (format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
         format!("layers.{}.feed_forward.up_proj.weight", layer_idx)),
        (format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
         format!("layers.{}.feed_forward.down_proj.weight", layer_idx)),

        // MLP scales (quantized)
        (format!("model.layers.{}.mlp.gate_proj.weight_scale", layer_idx),
         format!("layers.{}.feed_forward.gate_proj.weight_scale", layer_idx)),
        (format!("model.layers.{}.mlp.down_proj.weight_scale", layer_idx),
         format!("layers.{}.feed_forward.down_proj.weight_scale", layer_idx)),
    ];

    for (hf_name, expected_inferno_name) in test_cases {
        let mapped_name = InfernoLlama::map_weight_name(&hf_name).unwrap();
        assert_eq!(
            mapped_name, expected_inferno_name,
            "Mapping failed for {}: expected {}, got {}",
            hf_name, expected_inferno_name, mapped_name
        );
    }
}

/// Test mapping consistency across different layer indices
#[test]
fn test_layer_index_consistency() {
    // Test that mapping works consistently across different layer indices
    for layer_idx in [0, 7, 15] {
        let hf_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
        let expected = format!("layers.{}.attention.q_proj.weight", layer_idx);

        let mapped = InfernoLlama::map_weight_name(&hf_name).unwrap();
        assert_eq!(mapped, expected, "Layer index {} mapping failed", layer_idx);
    }
}

/// Test that all weight mappings are reversible/consistent
#[test]
fn test_mapping_consistency() {
    let test_cases = vec![
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.5.self_attn.q_proj.weight",
        "model.layers.10.mlp.gate_proj.weight",
        "model.layers.15.self_attn.k_proj.weight_scale",
    ];

    for hf_name in test_cases {
        let mapped = InfernoLlama::map_weight_name(hf_name).unwrap();

        // Verify mapping doesn't contain "model." prefix
        assert!(
            !mapped.starts_with("model."),
            "Mapped name '{}' still contains 'model.' prefix",
            mapped
        );

        // Verify mapping is not empty
        assert!(
            !mapped.is_empty(),
            "Mapped name for '{}' should not be empty",
            hf_name
        );

        // Verify specific transformations occurred
        if hf_name.contains("self_attn") {
            assert!(
                mapped.contains("attention"),
                "self_attn should map to attention in '{}'",
                mapped
            );
        }

        if hf_name.contains("mlp") {
            assert!(
                mapped.contains("feed_forward"),
                "mlp should map to feed_forward in '{}'",
                mapped
            );
        }
    }
}

/// Performance test for weight name mapping
///
/// Ensure mapping is efficient for models with 200+ weight tensors
#[test]
fn test_mapping_performance() {
    let test_names: Vec<String> = (0..16)
        .flat_map(|layer| {
            vec![
                format!("model.layers.{}.input_layernorm.weight", layer),
                format!("model.layers.{}.self_attn.q_proj.weight", layer),
                format!("model.layers.{}.self_attn.k_proj.weight", layer),
                format!("model.layers.{}.self_attn.v_proj.weight", layer),
                format!("model.layers.{}.self_attn.o_proj.weight", layer),
                format!("model.layers.{}.post_attention_layernorm.weight", layer),
                format!("model.layers.{}.mlp.gate_proj.weight", layer),
                format!("model.layers.{}.mlp.up_proj.weight", layer),
                format!("model.layers.{}.mlp.down_proj.weight", layer),
                // Quantized scales
                format!("model.layers.{}.self_attn.q_proj.weight_scale", layer),
                format!("model.layers.{}.self_attn.k_proj.weight_scale", layer),
                format!("model.layers.{}.mlp.gate_proj.weight_scale", layer),
            ]
        })
        .collect();

    // Should handle ~200 weight mappings efficiently
    let start = std::time::Instant::now();

    for name in &test_names {
        let _mapped = InfernoLlama::map_weight_name(name).unwrap();
    }

    let duration = start.elapsed();

    // Should complete all mappings in under 10ms (very conservative)
    assert!(
        duration.as_millis() < 10,
        "Weight mapping took {}ms for {} tensors, should be <10ms",
        duration.as_millis(),
        test_names.len()
    );

    println!(
        "Weight name mapping performance: {}μs for {} tensors ({:.2}μs per tensor)",
        duration.as_micros(),
        test_names.len(),
        duration.as_micros() as f64 / test_names.len() as f64
    );
}

/// Integration test with actual model patterns
///
/// Tests patterns found in real SafeTensors files
#[test]
fn test_real_model_patterns() {
    // Patterns from RedHatAI quantized model (259 tensors total)
    let real_patterns = vec![
        ("model.layers.0.post_attention_layernorm.weight", "layers.0.post_attention_layernorm.weight"),
        ("model.layers.8.self_attn.k_proj.weight", "layers.8.attention.k_proj.weight"),
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("model.layers.10.self_attn.k_proj.weight_scale", "layers.10.attention.k_proj.weight_scale"),
        ("model.layers.9.mlp.gate_proj.weight", "layers.9.feed_forward.gate_proj.weight"),
        ("model.layers.13.self_attn.k_proj.weight_scale", "layers.13.attention.k_proj.weight_scale"),
        ("model.norm.weight", "norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ];

    for (hf_name, expected_mapped) in real_patterns {
        let actual_mapped = InfernoLlama::map_weight_name(hf_name).unwrap();
        assert_eq!(
            actual_mapped, expected_mapped,
            "Real pattern mapping failed for '{}': expected '{}', got '{}'",
            hf_name, expected_mapped, actual_mapped
        );
    }
}