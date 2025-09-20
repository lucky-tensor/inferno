//! Configuration file parsing for different Llama model variants
//!
//! Handles the variety of config.json formats used by different model variants

use crate::candle_extensions::llama_models::{Config, LlamaVariant};

/// Configuration parser for different model formats
pub struct ConfigParser;

impl ConfigParser {
    /// Parse config.json from a model directory
    pub async fn parse_config(
        model_path: &str,
    ) -> std::result::Result<Config, Box<dyn std::error::Error>> {
        use std::path::Path;
        use tokio::fs;

        let config_path = Path::new(model_path).join("config.json");

        // Check if config.json exists
        if !tokio::fs::try_exists(&config_path).await.unwrap_or(false) {
            return Err(format!("config.json not found in {}", model_path).into());
        }

        // Read and parse config.json
        let config_content = fs::read_to_string(&config_path).await?;
        let json_value: serde_json::Value = serde_json::from_str(&config_content)?;

        // Parse the configuration with robust handling
        Self::parse_config_json(&json_value)
    }

    /// Parse configuration from JSON value with robust field handling
    fn parse_config_json(
        json: &serde_json::Value,
    ) -> std::result::Result<Config, Box<dyn std::error::Error>> {
        let obj = json.as_object().ok_or("Config JSON must be an object")?;

        // Required fields
        let hidden_size = obj
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .ok_or("hidden_size is required")? as usize;

        let intermediate_size = obj
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .ok_or("intermediate_size is required")? as usize;

        let num_hidden_layers = obj
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .ok_or("num_hidden_layers is required")? as usize;

        let num_attention_heads = obj
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .ok_or("num_attention_heads is required")? as usize;

        let vocab_size = obj
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .ok_or("vocab_size is required")? as usize;

        // Optional fields with defaults
        let num_key_value_heads = obj
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let rms_norm_eps = obj
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6);

        let rope_theta = obj
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(10000.0);

        let max_position_embeddings = obj
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2048);

        let tie_word_embeddings = obj
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Optional token IDs
        let bos_token_id = obj
            .get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        // Handle both single and multiple EOS tokens
        let eos_token_id = obj.get("eos_token_id").map(|v| {
            if let Some(single) = v.as_u64() {
                use crate::candle_extensions::llama_models::LlamaEosToks;
                LlamaEosToks::Single(single as u32)
            } else if let Some(array) = v.as_array() {
                use crate::candle_extensions::llama_models::LlamaEosToks;
                let tokens: Vec<u32> = array
                    .iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as u32)
                    .collect();
                LlamaEosToks::Multiple(tokens)
            } else {
                use crate::candle_extensions::llama_models::LlamaEosToks;
                LlamaEosToks::Single(2) // Default EOS
            }
        });

        // Parse rope_scaling if present
        let rope_scaling = obj
            .get("rope_scaling")
            .and_then(|v| Self::parse_rope_scaling(v).ok());

        Ok(Config {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            use_flash_attn: false, // Will be set externally based on hardware
            rms_norm_eps,
            rope_theta,
            bos_token_id,
            eos_token_id,
            rope_scaling,
            max_position_embeddings,
            tie_word_embeddings,
        })
    }

    /// Parse rope_scaling configuration
    fn parse_rope_scaling(
        json: &serde_json::Value,
    ) -> std::result::Result<
        crate::candle_extensions::llama_models::Llama3RopeConfig,
        Box<dyn std::error::Error>,
    > {
        use crate::candle_extensions::llama_models::{Llama3RopeConfig, Llama3RopeType};

        let obj = json.as_object().ok_or("rope_scaling must be an object")?;

        let factor = obj
            .get("factor")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0);

        let low_freq_factor = obj
            .get("low_freq_factor")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0);

        let high_freq_factor = obj
            .get("high_freq_factor")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0);

        let original_max_position_embeddings = obj
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2048);

        let rope_type = obj
            .get("rope_type")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "llama3" => Llama3RopeType::Llama3,
                _ => Llama3RopeType::Default,
            })
            .unwrap_or(Llama3RopeType::Default);

        Ok(Llama3RopeConfig {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
            rope_type,
        })
    }

    /// Detect model variant from config parameters
    pub fn detect_variant_from_config(config: &Config) -> LlamaVariant {
        // TinyLlama detection heuristics
        if config.hidden_size == 2048 && config.num_hidden_layers == 22 {
            return LlamaVariant::TinyLlama;
        }

        // Meta Llama 3.2 detection (typically smaller models)
        if config.vocab_size >= 128000 && config.hidden_size <= 3072 {
            return LlamaVariant::MetaLlama32;
        }

        // Meta Llama 3.1 detection (typically larger models)
        if config.vocab_size >= 128000 && config.hidden_size >= 4096 {
            return LlamaVariant::MetaLlama31;
        }

        // Default to custom for unrecognized configurations
        LlamaVariant::Custom
    }
}
