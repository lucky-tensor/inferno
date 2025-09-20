//! # Weight Loading Tests
//!
//! These tests specifically focus on SafeTensors weight loading capabilities:
//! 1. Loading and parsing SafeTensors files from real Llama 3.1 8B model
//! 2. Weight name mapping from HuggingFace format to InfernoLlama format
//! 3. Maintaining BF16 precision during weight loading
//! 4. Validating tensor shapes and content

use inferno_llama::InfernoLlama;
use std::path::Path;

const MODEL_PATH: &str = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";

#[test]
fn test_safetensors_weight_loading() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        panic!("Model directory not found at {}", MODEL_PATH);
    }

    // Load weights from SafeTensors files
    let result = InfernoLlama::load_weights_from_safetensors(MODEL_PATH);
    assert!(
        result.is_ok(),
        "Should load SafeTensors weights: {:?}",
        result.err()
    );

    let tensors = result.unwrap();

    // Verify we loaded all 291 expected tensors
    assert!(
        tensors.len() >= 291,
        "Should load 291+ weight tensors, got {}",
        tensors.len()
    );

    // Verify key tensor shapes and types
    let embed_tensor = &tensors["embed_tokens.weight"];
    assert_eq!(embed_tensor.dims(), &[128256, 4096]);
    assert_eq!(embed_tensor.dtype(), candle_core::DType::BF16);

    let q_proj = &tensors["layers.0.attention.q_proj.weight"];
    assert_eq!(q_proj.dims(), &[4096, 4096]);
    assert_eq!(q_proj.dtype(), candle_core::DType::BF16);

    println!(
        "✅ SafeTensors weight loading: {} tensors loaded",
        tensors.len()
    );
}

#[test]
fn test_weight_name_mapping() {
    // Test HuggingFace to InfernoLlama weight name mapping
    let test_cases = vec![
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "layers.0.attention.q_proj.weight",
        ),
        (
            "model.layers.15.mlp.gate_proj.weight",
            "layers.15.feed_forward.gate_proj.weight",
        ),
        ("model.norm.weight", "norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ];

    for (hf_name, expected_inferno_name) in test_cases {
        let mapped_name = InfernoLlama::map_weight_name(hf_name).unwrap();
        assert_eq!(
            mapped_name, expected_inferno_name,
            "Mapping failed for {}: expected {}, got {}",
            hf_name, expected_inferno_name, mapped_name
        );
    }

    println!("✅ Weight name mapping: HF format → InfernoLlama format");
}

#[test]
fn test_weight_index_parsing() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        panic!("Model directory not found at {}", MODEL_PATH);
    }

    // Test parsing the SafeTensors index file
    let result = InfernoLlama::get_weight_mapping(MODEL_PATH);
    assert!(
        result.is_ok(),
        "Should parse SafeTensors index: {:?}",
        result.err()
    );

    let mapping = result.unwrap();
    assert!(mapping.len() > 290, "Should map 290+ weights to files");

    // Verify specific mappings
    assert!(mapping.contains_key("model.embed_tokens.weight"));
    assert!(mapping.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(mapping.contains_key("lm_head.weight"));

    println!(
        "✅ SafeTensors index parsing: {} weight mappings",
        mapping.len()
    );
}

#[test]
fn test_bf16_precision_preservation() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        panic!("Model directory not found at {}", MODEL_PATH);
    }

    // Load weights and verify BF16 precision is maintained
    let tensors = InfernoLlama::load_weights_from_safetensors(MODEL_PATH).unwrap();

    let bf16_count = tensors
        .values()
        .filter(|t| t.dtype() == candle_core::DType::BF16)
        .count();

    // Most tensors should be BF16 (the original problem we're solving)
    assert!(
        bf16_count > 200,
        "Should preserve BF16 precision: {} BF16 tensors out of {}",
        bf16_count,
        tensors.len()
    );

    println!(
        "✅ BF16 precision preserved: {}/{} tensors",
        bf16_count,
        tensors.len()
    );
}
