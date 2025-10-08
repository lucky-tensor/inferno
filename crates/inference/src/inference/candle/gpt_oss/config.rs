use serde::{Deserialize, Serialize};

/// GPT-OSS model configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GptOssConfig {
    /// Activation function (defaults to "silu" for SwiGLU)
    #[serde(default)]
    pub hidden_act: Option<String>,

    /// Hidden dimension size (e.g., 3072 for gpt-oss-20b)
    pub hidden_size: usize,

    /// MLP intermediate size per expert
    pub intermediate_size: usize,

    /// Dimension of each attention head
    pub head_dim: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for Grouped Query Attention)
    pub num_key_value_heads: usize,

    /// RMSNorm epsilon
    pub rms_norm_eps: f64,

    /// RoPE theta base frequency
    pub rope_theta: f32,

    /// Maximum position embeddings (context length)
    pub max_position_embeddings: usize,

    /// RoPE scaling configuration (optional)
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Whether to tie word embeddings with output layer
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Number of experts in each MoE layer
    pub num_local_experts: usize,

    /// Number of experts activated per token (Top-K)
    pub num_experts_per_tok: usize,

    /// Sliding window size for attention
    pub sliding_window: usize,

    /// Whether attention projections have bias
    #[serde(default)]
    pub attention_bias: bool,

    /// Per-layer attention types: "sliding_window" or "full"
    pub layer_types: Vec<String>,
}

/// RoPE scaling configuration for extended context
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeScalingConfig {
    /// Scaling factor
    pub factor: f32,

    /// Scaling type (e.g., "yarn", "linear")
    #[serde(rename = "type")]
    pub scaling_type: String,

    /// Additional parameters
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl GptOssConfig {
    /// Get the activation function name
    pub fn hidden_act(&self) -> &str {
        self.hidden_act.as_deref().unwrap_or("silu")
    }

    /// Check if a layer uses sliding window attention
    pub fn is_sliding_window(&self, layer_idx: usize) -> bool {
        if layer_idx >= self.layer_types.len() {
            false
        } else {
            self.layer_types[layer_idx] == "sliding_window"
        }
    }

    /// Get the number of KV groups for Grouped Query Attention
    pub fn n_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let config_json = r#"{
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "head_dim": 128,
            "vocab_size": 201088,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 150000.0,
            "max_position_embeddings": 131072,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "sliding_window": 128,
            "attention_bias": false,
            "layer_types": ["full", "sliding_window", "full"]
        }"#;

        let config: GptOssConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_local_experts, 32);
        assert_eq!(config.num_experts_per_tok, 4);
        assert_eq!(config.n_kv_groups(), 8);
        assert!(!config.is_sliding_window(0));
        assert!(config.is_sliding_window(1));
    }
}
