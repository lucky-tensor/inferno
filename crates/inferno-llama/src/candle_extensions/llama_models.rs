//! Generic Llama model implementations forked and extended from candle-transformers
//!
//! This module provides a unified interface for different Llama model variants
//! while leveraging Candle's proven tensor operations and memory management.
//!
//! Forked from: candle-transformers/src/models/llama.rs
//! Extensions: Multi-variant support, quantization, enhanced precision handling

use serde::{Deserialize, Serialize};

/// Llama model variants supported by the generic engine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlamaVariant {
    /// Meta Llama 3.1 models (8B, 70B, 405B)
    MetaLlama31,
    /// Meta Llama 3.2 models (1B, 3B, 11B, 90B)
    MetaLlama32,
    /// TinyLlama distilled models (~1B parameters)
    TinyLlama,
    /// DeepSeek distilled models
    DeepSeekDistilled,
    /// Custom/Unknown variant - requires manual configuration
    Custom,
}

/// Enhanced configuration combining Candle's base config with our extensions
#[derive(Debug, Clone)]
pub struct GenericLlamaConfig {
    /// Base candle-transformers configuration
    pub base: Config,
    /// Detected model variant
    pub variant: LlamaVariant,
    /// Quantization configuration if present
    pub quantization: Option<crate::diagnostic::QuantizationConfig>,
    /// Memory layout information
    pub memory_layout: crate::diagnostic::ModelMemoryLayout,
}

/// Core configuration structure (forked from candle-transformers)
///
/// This maintains compatibility with Candle's Config while allowing extensions
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

/// EOS token configuration (from candle-transformers)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

/// Llama 3 RoPE configuration (from candle-transformers)
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

/// RoPE type variants (from candle-transformers)
#[derive(Debug, Clone, Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

impl GenericLlamaConfig {
    /// Auto-detect model variant and configuration from a model directory
    ///
    /// This is the main entry point for model variant detection.
    /// It analyzes config files, weight files, and directory structure
    /// to determine the model type and optimal configuration.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory containing config.json and weights
    ///
    /// # Returns
    /// * `Result<GenericLlamaConfig, Box<dyn std::error::Error>>` - Detected configuration or error
    ///
    /// # Examples
    /// ```rust,no_run
    /// use inferno_llama::candle_extensions::GenericLlamaConfig;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let config = GenericLlamaConfig::detect_variant("/path/to/llama/model").await?;
    /// println!("Detected model variant: {:?}", config.variant);
    /// # Result::<(), Box<dyn std::error::Error>>::Ok(())
    /// # }).unwrap();
    /// ```
    pub async fn detect_variant(
        model_path: &str,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        // Use the diagnostic system to detect the model variant
        crate::diagnostic::ModelDetector::detect_variant(model_path).await
    }

    /// Create a GenericLlamaConfig from a manually specified variant and path
    ///
    /// Useful when auto-detection fails or for custom model configurations.
    pub async fn from_variant(
        model_path: &str,
        _variant: LlamaVariant,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        // Use the detector to perform the actual detection
        // This will be implemented when the detector is complete
        crate::diagnostic::ModelDetector::detect_variant(model_path).await
    }

    /// Validate that the configuration is consistent and usable
    ///
    /// Performs sanity checks on configuration values and ensures
    /// compatibility between variant type and configuration parameters.
    pub fn validate(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Basic validation rules that apply to all variants
        if self.base.hidden_size == 0 {
            return Err("Hidden size must be positive".into());
        }

        if self.base.num_hidden_layers == 0 {
            return Err("Number of layers must be positive".into());
        }

        if self.base.num_attention_heads == 0 {
            return Err("Number of attention heads must be positive".into());
        }

        if self.base.vocab_size == 0 {
            return Err("Vocabulary size must be positive".into());
        }

        // Variant-specific validation
        match self.variant {
            LlamaVariant::TinyLlama => {
                if self.base.hidden_size != 2048 {
                    return Err("TinyLlama models should have hidden_size=2048".into());
                }
            }
            LlamaVariant::MetaLlama31 | LlamaVariant::MetaLlama32 => {
                if self.base.vocab_size < 128000 {
                    return Err("Meta Llama 3.x models should have vocab_size >= 128000".into());
                }
            }
            _ => {} // No specific validation for other variants yet
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn: false,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(LlamaEosToks::Single(2)),
            rope_scaling: None,
            max_position_embeddings: 2048,
            tie_word_embeddings: false,
        }
    }
}

impl Config {
    /// Convert from candle-transformers LlamaConfig to our Config
    ///
    /// This provides compatibility with existing candle-transformers code
    /// while allowing our extensions.
    pub fn from_candle_config(candle_config: &candle_transformers::models::llama::Config) -> Self {
        Self {
            hidden_size: candle_config.hidden_size,
            intermediate_size: candle_config.intermediate_size,
            vocab_size: candle_config.vocab_size,
            num_hidden_layers: candle_config.num_hidden_layers,
            num_attention_heads: candle_config.num_attention_heads,
            num_key_value_heads: candle_config.num_key_value_heads,
            use_flash_attn: candle_config.use_flash_attn,
            rms_norm_eps: candle_config.rms_norm_eps,
            rope_theta: candle_config.rope_theta,
            bos_token_id: candle_config.bos_token_id,
            // Note: This requires matching enum variants - we'll handle conversion
            eos_token_id: None, // TODO: Convert from candle's LlamaEosToks
            rope_scaling: None, // TODO: Convert from candle's Llama3RopeConfig
            max_position_embeddings: candle_config.max_position_embeddings,
            tie_word_embeddings: candle_config.tie_word_embeddings,
        }
    }

    /// Convert to candle-transformers Config for compatibility
    ///
    /// Allows us to use existing candle model implementations
    /// with our enhanced configuration.
    pub fn to_candle_config(&self) -> candle_transformers::models::llama::Config {
        todo!("Conversion to candle Config not yet implemented")
    }
}
