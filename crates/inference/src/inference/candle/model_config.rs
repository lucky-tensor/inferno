//! Model configuration and loading utilities for Candle

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use crate::inference::InferenceError;
use serde::{Deserialize, Serialize};

/// Model configuration loaded from HuggingFace config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl CandleModelConfig {
    /// Load model configuration from config.json file
    pub async fn load_from_path(model_path: &str) -> Result<Self, InferenceError> {
        let config_path = std::path::Path::new(model_path).join("config.json");

        if !config_path.exists() {
            return Err(InferenceError::InvalidArgument(format!(
                "Config file not found: {}",
                config_path.display()
            )));
        }

        let config_content = tokio::fs::read_to_string(&config_path).await.map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to read config.json: {}", e))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to parse config.json: {}", e))
        })?;

        Ok(Self {
            hidden_size: config["hidden_size"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(5632) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
            num_key_value_heads: config["num_key_value_heads"].as_u64().map(|v| v as usize),
            vocab_size: config["vocab_size"].as_u64().unwrap_or(32000) as usize,
            max_position_embeddings: config["max_position_embeddings"]
                .as_u64()
                .unwrap_or(2048) as usize,
            rms_norm_eps: config["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            rope_theta: config["rope_theta"].as_f64().unwrap_or(10000.0),
            tie_word_embeddings: config["tie_word_embeddings"].as_bool(),
            bos_token_id: config["bos_token_id"].as_u64().map(|v| v as u32),
            eos_token_id: config["eos_token_id"]
                .as_u64()
                .map(|v| v as u32)
                .or_else(|| {
                    // Handle array of EOS tokens - take the first one
                    config["eos_token_id"]
                        .as_array()
                        .and_then(|arr| arr.get(0))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                }),
        })
    }

    /// Estimate model parameters for display
    pub fn estimate_parameters(&self) -> String {
        let embedding_params = self.vocab_size * self.hidden_size;
        let layer_params = self.num_hidden_layers
            * (
                // Multi-head attention
                4 * self.hidden_size * self.hidden_size +
            // Feed-forward network
            3 * self.hidden_size * self.intermediate_size +
            // Layer norms
            2 * self.hidden_size
            );
        let total_params = embedding_params + layer_params;

        if total_params >= 1_000_000_000 {
            format!("{:.1}B", total_params as f64 / 1_000_000_000.0)
        } else {
            format!("{:.1}M", total_params as f64 / 1_000_000.0)
        }
    }
}
