//! # Simple Loading Tests
//!
//! Basic configuration loading and file existence tests.
//! These are the simplest possible tests that verify the model directory
//! structure and configuration parsing without loading weights.

use inferno_llama::InfernoLlama;
use std::path::Path;

const MODEL_PATH: &str = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";

#[test]
fn test_model_directory_exists() {
    let model_path = Path::new(MODEL_PATH);
    assert!(
        model_path.exists(),
        "Model directory should exist at {}",
        MODEL_PATH
    );
    assert!(model_path.is_dir(), "Model path should be a directory");
}

#[test]
fn test_required_files_exist() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        panic!("Model directory not found at {}", MODEL_PATH);
    }

    // Check for required files
    let config_file = model_path.join("config.json");
    assert!(config_file.exists(), "config.json should exist");

    let index_file = model_path.join("model.safetensors.index.json");
    assert!(index_file.exists(), "SafeTensors index should exist");

    let tokenizer_file = model_path.join("tokenizer.json");
    assert!(tokenizer_file.exists(), "tokenizer.json should exist");

    // Check at least one SafeTensors shard exists
    let shard1 = model_path.join("model-00001-of-00004.safetensors");
    assert!(shard1.exists(), "First SafeTensors shard should exist");
}

#[test]
fn test_config_json_parsing() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        panic!("Model directory not found at {}", MODEL_PATH);
    }

    // Test our config parsing implementation
    let result = InfernoLlama::load_config_simple(model_path);
    assert!(
        result.is_ok(),
        "Config parsing should succeed: {:?}",
        result.err()
    );

    let config = result.unwrap();

    // Verify Llama 3.1 8B specifications
    assert_eq!(config.dim, 4096, "Hidden dimension should be 4096");
    assert_eq!(config.n_layers, 32, "Should have 32 layers");
    assert_eq!(config.n_heads, 32, "Should have 32 attention heads");
    assert_eq!(config.n_kv_heads, Some(8), "Should have 8 KV heads (GQA)");
    assert_eq!(
        config.vocab_size, 128256,
        "Vocabulary size should be 128256"
    );

    println!(
        "✅ Config loaded: Llama 3.1 8B ({} layers, {} dim)",
        config.n_layers, config.dim
    );
}

#[test]
fn test_empty_model_creation() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        panic!("Model directory not found at {}", MODEL_PATH);
    }

    // Test creating empty model structure (no weights loaded)
    let result = InfernoLlama::load_from_path(MODEL_PATH);
    assert!(
        result.is_ok(),
        "Empty model creation should succeed: {:?}",
        result.err()
    );

    let model = result.unwrap();
    assert_eq!(
        model.layers.len(),
        32,
        "Should create 32 transformer layers"
    );
    assert_eq!(model.config.dim, 4096, "Should preserve config dimensions");

    println!(
        "✅ Empty model structure created: {} layers, {} parameters",
        model.layers.len(),
        model.parameter_count()
    );
}
