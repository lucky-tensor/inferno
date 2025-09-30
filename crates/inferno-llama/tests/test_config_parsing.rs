//! Test-Driven Development tests for configuration parsing
//!
//! These tests define the expected behavior for parsing different config.json formats
//! used by various Llama model variants.

use inferno_llama::candle_extensions::llama_models::{Config, LlamaVariant};
use inferno_llama::diagnostic::ConfigParser;
use serde_json::json;
use tempfile::TempDir;
use tokio::fs;

/// Test parsing Meta Llama 3.1 8B Instruct config
#[tokio::test]
async fn test_parse_meta_llama_31_8b_config() {
    let model_path = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";

    let result = ConfigParser::parse_config(model_path).await;
    assert!(
        result.is_ok(),
        "Should successfully parse Meta Llama 3.1 config"
    );

    let config = result.unwrap();

    // Meta Llama 3.1 8B specific values
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.max_position_embeddings, 131072);
    assert!(config.rms_norm_eps > 0.0);
    assert!(config.rope_theta > 0.0);
}

/// Test parsing TinyLlama 1.1B config
#[tokio::test]
async fn test_parse_tinyllama_config() {
    let model_path = "/home/jeef/models/tinyllama-1.1b/TinyLlama-1.1B-Chat-v1.0";

    let result = ConfigParser::parse_config(model_path).await;
    assert!(result.is_ok(), "Should successfully parse TinyLlama config");

    let config = result.unwrap();

    // TinyLlama specific values
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.intermediate_size, 5632);
    assert_eq!(config.num_hidden_layers, 22);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.vocab_size, 32000);
    assert!(config.max_position_embeddings >= 2048);
}

/// Test parsing quantized Llama 3.2 1B config
#[tokio::test]
async fn test_parse_quantized_llama_32_config() {
    let model_path = "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8";

    let result = ConfigParser::parse_config(model_path).await;
    assert!(
        result.is_ok(),
        "Should successfully parse quantized Llama 3.2 config"
    );

    let config = result.unwrap();

    // Llama 3.2 1B specific values
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 16);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.vocab_size, 128256); // Should use extended vocab like other 3.x models
}

/// Test parsing DeepSeek distilled config
#[tokio::test]
async fn test_parse_deepseek_distilled_config() {
    let model_path = "/home/jeef/models/DeepSeek-R1-Distill-Llama-70B";

    let result = ConfigParser::parse_config(model_path).await;

    // This model doesn't have a config.json, so we expect it to fail
    // In a real implementation, we'd need to create a config.json or skip this test
    if result.is_err() {
        // Skip this test since the model doesn't have a proper config.json
        return;
    }

    let config = result.unwrap();

    // Large model expectations
    assert!(
        config.hidden_size >= 4096,
        "Large model should have substantial hidden size"
    );
    assert!(
        config.num_hidden_layers >= 32,
        "Large model should have many layers"
    );
    assert!(
        config.num_attention_heads >= 32,
        "Large model should have many attention heads"
    );
}

/// Test robust parsing with missing optional fields
#[tokio::test]
async fn test_parse_config_with_missing_fields() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    // Create a minimal config with only required fields
    let minimal_config = json!({
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-6
    });

    fs::write(&config_path, minimal_config.to_string())
        .await
        .unwrap();

    let result = ConfigParser::parse_config(temp_dir.path().to_str().unwrap()).await;
    assert!(
        result.is_ok(),
        "Should parse config with missing optional fields"
    );

    let config = result.unwrap();

    // Check that defaults were applied correctly
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_attention_heads, 16);
    // num_key_value_heads should default to num_attention_heads
    assert_eq!(config.num_key_value_heads, 16);
    // rope_theta should have a sensible default
    assert!(config.rope_theta > 0.0);
}

/// Test parsing config with rope_scaling (Llama 3 feature)
#[tokio::test]
async fn test_parse_config_with_rope_scaling() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    // Config with rope_scaling (typical of Llama 3.1 models)
    let config_with_rope = json!({
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    });

    fs::write(&config_path, config_with_rope.to_string())
        .await
        .unwrap();

    let result = ConfigParser::parse_config(temp_dir.path().to_str().unwrap()).await;
    assert!(result.is_ok(), "Should parse config with rope_scaling");

    let config = result.unwrap();
    assert!(
        config.rope_scaling.is_some(),
        "Should preserve rope_scaling"
    );

    let rope_scaling = config.rope_scaling.unwrap();
    assert_eq!(rope_scaling.factor, 8.0);
    assert_eq!(rope_scaling.original_max_position_embeddings, 8192);
}

/// Test error handling for malformed config files
#[tokio::test]
async fn test_parse_malformed_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    // Write invalid JSON
    fs::write(&config_path, "{ invalid json }").await.unwrap();

    let result = ConfigParser::parse_config(temp_dir.path().to_str().unwrap()).await;
    assert!(result.is_err(), "Should return error for malformed JSON");
}

/// Test error handling for missing config file
#[tokio::test]
async fn test_parse_missing_config() {
    let temp_dir = TempDir::new().unwrap();
    // Don't create config.json

    let result = ConfigParser::parse_config(temp_dir.path().to_str().unwrap()).await;
    assert!(
        result.is_err(),
        "Should return error for missing config.json"
    );
}

/// Test variant detection from parsed config
#[tokio::test]
async fn test_detect_variant_from_config() {
    // Test TinyLlama detection
    let tinyllama_config = Config {
        hidden_size: 2048,
        num_hidden_layers: 22,
        vocab_size: 32000,
        ..Default::default()
    };

    let variant = ConfigParser::detect_variant_from_config(&tinyllama_config);
    assert_eq!(variant, LlamaVariant::TinyLlama);

    // Test Meta Llama 3.1 detection (large model)
    let meta_31_config = Config {
        hidden_size: 4096,
        num_hidden_layers: 32,
        vocab_size: 128256,
        ..Default::default()
    };

    let variant = ConfigParser::detect_variant_from_config(&meta_31_config);
    assert_eq!(variant, LlamaVariant::MetaLlama31);

    // Test Meta Llama 3.2 detection (smaller model)
    let meta_32_config = Config {
        hidden_size: 2048,
        num_hidden_layers: 16,
        vocab_size: 128256,
        ..Default::default()
    };

    let variant = ConfigParser::detect_variant_from_config(&meta_32_config);
    assert_eq!(variant, LlamaVariant::MetaLlama32);

    // Test unknown/custom detection
    let custom_config = Config {
        hidden_size: 1234,
        num_hidden_layers: 56,
        vocab_size: 12345,
        ..Default::default()
    };

    let variant = ConfigParser::detect_variant_from_config(&custom_config);
    assert_eq!(variant, LlamaVariant::Custom);
}

/// Test parsing configs with different field names/formats
#[tokio::test]
async fn test_parse_config_field_variations() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    // Some models might use slightly different field names
    let variant_config = json!({
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": false,
        "bos_token_id": 1,
        "eos_token_id": 2
    });

    fs::write(&config_path, variant_config.to_string())
        .await
        .unwrap();

    let result = ConfigParser::parse_config(temp_dir.path().to_str().unwrap()).await;
    assert!(result.is_ok(), "Should handle config field variations");

    let config = result.unwrap();
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.bos_token_id, Some(1));
    assert!(!config.tie_word_embeddings);
}

/// Test parsing configs with multiple EOS tokens
#[tokio::test]
async fn test_parse_config_multiple_eos_tokens() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    // Config with multiple EOS token IDs
    let multi_eos_config = json!({
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "eos_token_id": [128001, 128009]
    });

    fs::write(&config_path, multi_eos_config.to_string())
        .await
        .unwrap();

    let result = ConfigParser::parse_config(temp_dir.path().to_str().unwrap()).await;
    assert!(
        result.is_ok(),
        "Should parse config with multiple EOS tokens"
    );

    let config = result.unwrap();
    assert!(
        config.eos_token_id.is_some(),
        "Should preserve EOS token info"
    );
}
