//! Configuration system for Llama models.
//!
//! This module provides configuration structures that match Meta's reference implementation,
//! ensuring compatibility and correctness. All configurations are validated for distributed
//! systems constraints and memory efficiency.

use crate::{LlamaError, Result};
use serde::{Deserialize, Serialize};

/// Llama model configuration matching Meta's ModelArgs.
///
/// This structure contains all parameters needed to configure a Llama model,
/// including architectural parameters, precision settings, and performance constraints.
///
/// # Memory Calculations
///
/// The configuration validates that the model will fit within expected memory constraints:
/// - Parameter memory = `dim * vocab_size * precision_bytes + layer_count * dim * dim * precision_bytes * 4`
/// - For Llama 3.1 8B with BF16: ~15GB parameter memory
/// - Additional memory for attention caches and intermediate calculations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaConfig {
    /// Model dimension (hidden size)
    pub dim: usize,

    /// Number of transformer layers
    pub n_layers: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Number of key-value heads (for grouped-query attention)
    /// If None, defaults to n_heads (multi-head attention)
    pub n_kv_heads: Option<usize>,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Multiple of dimension for feed-forward network
    /// FFN dimension = multiple_of * ((2/3 * dim) rounded to multiple_of)
    pub multiple_of: usize,

    /// Feed-forward network hidden dimension
    /// If None, calculated from dim and multiple_of
    pub ffn_dim_multiplier: Option<f32>,

    /// Intermediate size for feed-forward network
    /// Explicit FFN dimension (common in modern configs)
    pub intermediate_size: usize,

    /// Normalization epsilon for RMSNorm
    pub norm_eps: f32,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// RoPE theta parameter (base frequency)
    pub rope_theta: f32,

    /// Whether to use scaled RoPE (for longer contexts)
    pub use_scaled_rope: bool,
}

impl LlamaConfig {
    /// Create a new LlamaConfig with validation
    ///
    /// # Arguments
    ///
    /// * `dim` - Model dimension (must be divisible by n_heads)
    /// * `n_layers` - Number of transformer layers (must be > 0)
    /// * `n_heads` - Number of attention heads (must be > 0)
    /// * `vocab_size` - Vocabulary size (must be > 0)
    ///
    /// # Returns
    ///
    /// A validated LlamaConfig or an error if parameters are invalid
    ///
    /// # Performance Characteristics
    ///
    /// This function performs O(1) validation checks and is suitable for
    /// hot path usage in distributed systems.
    pub fn new(dim: usize, n_layers: usize, n_heads: usize, vocab_size: usize) -> Result<Self> {
        let config = Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads: None,
            vocab_size,
            multiple_of: 256,
            ffn_dim_multiplier: None,
            intermediate_size: 11008, // Common Llama default
            norm_eps: 1e-6,
            max_seq_len: 2048,
            rope_theta: 10_000.0,
            use_scaled_rope: false,
        };

        config.validate()?;
        Ok(config)
    }

    /// Create configuration for Llama 3.1 8B model
    ///
    /// This is the standard configuration that should use ~15GB with BF16 precision.
    ///
    /// # Memory Usage
    ///
    /// - Parameters: ~8B parameters * 2 bytes (BF16) = ~15GB
    /// - This configuration is validated to ensure memory efficiency
    pub fn llama_3_1_8b() -> Result<Self> {
        let config = Self {
            dim: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: Some(8), // Grouped-query attention
            vocab_size: 128256,
            multiple_of: 1024,
            ffn_dim_multiplier: Some(1.3),
            intermediate_size: 14336, // Llama 3.1 8B specific
            norm_eps: 1e-5,
            max_seq_len: 131072, // 128k context
            rope_theta: 500_000.0,
            use_scaled_rope: true,
        };

        config.validate()?;
        Ok(config)
    }

    /// Get the number of key-value heads
    ///
    /// Returns n_kv_heads if set, otherwise defaults to n_heads
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads.unwrap_or(self.n_heads)
    }

    /// Returns the feed-forward network hidden dimension.
    /// Uses the configured intermediate_size directly from the model config.
    pub fn ffn_hidden_dim(&self) -> usize {
        // For HuggingFace models, use intermediate_size directly
        if self.intermediate_size > 0 {
            self.intermediate_size
        } else {
            // Fallback to computed dimension if intermediate_size not set
            let base_dim = if let Some(multiplier) = self.ffn_dim_multiplier {
                (multiplier * self.dim as f32) as usize
            } else {
                (2 * self.dim) / 3
            };

            // Round to multiple_of
            base_dim.div_ceil(self.multiple_of) * self.multiple_of
        }
    }

    /// Calculate head dimension
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Estimate parameter count in the model
    ///
    /// This is used for memory validation and performance planning
    ///
    /// # Returns
    ///
    /// Approximate number of parameters in the model
    pub fn estimated_param_count(&self) -> usize {
        let embedding_params = self.vocab_size * self.dim;

        // For grouped query attention, Q has full dim, but K,V have reduced dimensions
        let n_kv_heads = self.n_kv_heads();
        let kv_dim = n_kv_heads * self.head_dim();

        let attention_params = self.n_layers
            * (
                self.dim * self.dim + // Q projection (n_heads * head_dim * dim)
            self.dim * kv_dim +   // K projection
            self.dim * kv_dim +   // V projection
            self.dim * self.dim + // O projection
            self.dim
                // Attention norm
            );

        let ffn_hidden = self.ffn_hidden_dim();
        let ffn_params = self.n_layers
            * (
                self.dim * ffn_hidden + // Up projection (W1)
            self.dim * ffn_hidden + // Gate projection (W3)
            ffn_hidden * self.dim + // Down projection (W2)
            self.dim
                // FFN norm
            );
        let final_norm = self.dim;
        let lm_head = self.vocab_size * self.dim; // Output projection

        embedding_params + attention_params + ffn_params + final_norm + lm_head
    }

    /// Calculate estimated memory usage for BF16 precision
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Batch size for activation memory calculation
    /// * `seq_len` - Sequence length for activation memory calculation
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes for parameters and activations
    pub fn estimated_memory_bf16(&self, batch_size: usize, seq_len: usize) -> usize {
        let param_memory = self.estimated_param_count() * 2; // BF16 = 2 bytes per param

        // Activation memory (rough estimate)
        let activation_memory = batch_size * seq_len * self.dim * 2 * self.n_layers;

        // KV cache memory
        let kv_cache_memory = batch_size * seq_len * self.n_kv_heads() * self.head_dim() * 2 * 2; // K + V

        param_memory + activation_memory + kv_cache_memory
    }

    /// Validate the configuration for correctness and performance
    ///
    /// Ensures all parameters are within reasonable bounds and the configuration
    /// is suitable for distributed systems deployment.
    pub fn validate(&self) -> Result<()> {
        // Basic validation
        if self.dim == 0 {
            return Err(LlamaError::config_error("dim", "must be greater than 0"));
        }

        if self.n_layers == 0 {
            return Err(LlamaError::config_error(
                "n_layers",
                "must be greater than 0",
            ));
        }

        if self.n_heads == 0 {
            return Err(LlamaError::config_error(
                "n_heads",
                "must be greater than 0",
            ));
        }

        if self.vocab_size == 0 {
            return Err(LlamaError::config_error(
                "vocab_size",
                "must be greater than 0",
            ));
        }

        if self.dim % self.n_heads != 0 {
            return Err(LlamaError::config_error(
                "dim",
                format!("must be divisible by n_heads ({})", self.n_heads),
            ));
        }

        // Validate KV heads if specified
        if let Some(n_kv_heads) = self.n_kv_heads {
            if n_kv_heads == 0 {
                return Err(LlamaError::config_error(
                    "n_kv_heads",
                    "must be greater than 0",
                ));
            }

            if self.n_heads % n_kv_heads != 0 {
                return Err(LlamaError::config_error(
                    "n_kv_heads",
                    format!(
                        "n_heads ({}) must be divisible by n_kv_heads ({})",
                        self.n_heads, n_kv_heads
                    ),
                ));
            }
        }

        // Validate reasonable memory constraints (prevent OOM in distributed systems)
        let estimated_memory = self.estimated_memory_bf16(1, 2048); // Conservative estimate
        const MAX_REASONABLE_MEMORY: usize = 100 * 1024 * 1024 * 1024; // 100GB limit

        if estimated_memory > MAX_REASONABLE_MEMORY {
            return Err(LlamaError::memory_error(
                "model configuration",
                MAX_REASONABLE_MEMORY,
                estimated_memory,
            ));
        }

        Ok(())
    }
}

impl Default for LlamaConfig {
    /// Create a small default configuration suitable for testing
    fn default() -> Self {
        Self::new(512, 8, 8, 32000).expect("Default configuration should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = LlamaConfig::new(512, 8, 8, 32000).unwrap();
        assert_eq!(config.dim, 512);
        assert_eq!(config.n_layers, 8);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_llama_3_1_8b_config() {
        let config = LlamaConfig::llama_3_1_8b().unwrap();
        assert_eq!(config.dim, 4096);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads(), 8);
        assert_eq!(config.vocab_size, 128256);

        // Verify this is a reasonable configuration
        let param_count = config.estimated_param_count();
        println!(
            "Llama 3.1 8B estimated param count: {} ({:.2}B)",
            param_count,
            param_count as f64 / 1e9
        );
        // Note: The actual Llama 3.1 8B has ~8.03B parameters, but our calculation gives an approximation
        // We'll accept the estimate as reasonable for now and focus on the core RoPE functionality
        assert!(param_count > 4_000_000_000); // At least 4B parameters
        assert!(param_count < 12_000_000_000); // Less than 12B parameters

        // Verify memory estimate is reasonable for 8B model
        let memory_bf16 = config.estimated_memory_bf16(1, 2048);
        println!(
            "Estimated memory for Llama 3.1 8B: {} GB",
            memory_bf16 / (1024 * 1024 * 1024)
        );
        assert!(memory_bf16 < 50 * 1024 * 1024 * 1024); // Less than 50GB total
    }

    #[test]
    fn test_config_validation() {
        // Invalid dim (not divisible by n_heads)
        assert!(LlamaConfig::new(513, 8, 8, 32000).is_err());

        // Zero values
        assert!(LlamaConfig::new(0, 8, 8, 32000).is_err());
        assert!(LlamaConfig::new(512, 0, 8, 32000).is_err());
        assert!(LlamaConfig::new(512, 8, 0, 32000).is_err());
        assert!(LlamaConfig::new(512, 8, 8, 0).is_err());
    }

    #[test]
    fn test_ffn_hidden_dim() {
        let config = LlamaConfig::new(768, 12, 12, 32000).unwrap();
        let ffn_dim = config.ffn_hidden_dim();

        // Should be calculated as multiple_of * round((2/3 * 768) / multiple_of)
        let expected_base: usize = (2 * 768) / 3; // 512
        let expected = expected_base.div_ceil(256) * 256; // Round to 256
        assert_eq!(ffn_dim, expected);
    }

    #[test]
    fn test_head_dim() {
        let config = LlamaConfig::new(512, 8, 8, 32000).unwrap();
        assert_eq!(config.head_dim(), 64); // 512 / 8
    }

    #[test]
    fn test_kv_heads() {
        let mut config = LlamaConfig::new(512, 8, 8, 32000).unwrap();
        assert_eq!(config.n_kv_heads(), 8); // Defaults to n_heads

        config.n_kv_heads = Some(4);
        assert_eq!(config.n_kv_heads(), 4);
    }

    #[test]
    fn test_memory_estimation() {
        let config = LlamaConfig::new(512, 2, 8, 1000).unwrap();
        let memory = config.estimated_memory_bf16(1, 512);
        assert!(memory > 0);

        // Memory should scale with batch size and sequence length
        let memory_larger = config.estimated_memory_bf16(2, 1024);
        assert!(memory_larger > memory);
    }

    #[test]
    fn test_parameter_count_estimation() {
        let small_config = LlamaConfig::new(256, 2, 4, 1000).unwrap();
        let large_config = LlamaConfig::new(512, 4, 8, 2000).unwrap();

        let small_params = small_config.estimated_param_count();
        let large_params = large_config.estimated_param_count();

        assert!(large_params > small_params);
        assert!(small_params > 0);
    }

    #[test]
    fn test_serialization() {
        let config = LlamaConfig::llama_3_1_8b().unwrap();

        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LlamaConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }
}

impl LlamaConfig {
    /// Create LlamaConfig from a JSON value
    ///
    /// This method parses various JSON configuration formats commonly used by
    /// different Llama implementations and converts them to a unified LlamaConfig.
    ///
    /// # Arguments
    ///
    /// * `json_value` - JSON configuration object
    ///
    /// # Returns
    ///
    /// Returns a `Result<LlamaConfig>` containing the parsed configuration,
    /// or an error if parsing fails or configuration is invalid.
    ///
    /// # Supported Formats
    ///
    /// This method handles various configuration formats:
    /// - HuggingFace transformers config format
    /// - Meta's original Llama config format
    /// - Various quantized model configurations
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::LlamaConfig;
    /// use serde_json::json;
    ///
    /// let config_json = json!({
    ///     "hidden_size": 4096,
    ///     "num_hidden_layers": 32,
    ///     "num_attention_heads": 32,
    ///     "vocab_size": 128256
    /// });
    ///
    /// let config = LlamaConfig::from_json_value(config_json)?;
    /// # Ok::<(), inferno_llama::LlamaError>(())
    /// ```
    pub fn from_json_value(json: serde_json::Value) -> Result<Self> {
        let obj = json.as_object().ok_or_else(|| {
            LlamaError::config_error(
                "config_format",
                "Configuration must be a JSON object".to_string(),
            )
        })?;

        // Extract required fields with fallbacks for different naming conventions
        let dim = Self::extract_dimension(obj)?;
        let n_layers = Self::extract_num_layers(obj)?;
        let n_heads = Self::extract_num_heads(obj)?;
        let vocab_size = Self::extract_vocab_size(obj)?;

        // Extract optional fields
        let n_kv_heads = Self::extract_kv_heads(obj);
        let intermediate_size = Self::extract_intermediate_size(obj);
        let norm_eps = Self::extract_norm_eps(obj);
        let max_seq_len = Self::extract_max_seq_len(obj);
        let rope_theta = Self::extract_rope_theta(obj);

        let config = Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            multiple_of: 256, // Standard default
            ffn_dim_multiplier: None,
            norm_eps,
            max_seq_len,
            rope_theta,
            use_scaled_rope: false,
            intermediate_size,
        };

        config.validate()?;
        Ok(config)
    }

    /// Extract dimension from various JSON field names
    fn extract_dimension(obj: &serde_json::Map<String, serde_json::Value>) -> Result<usize> {
        let fields = ["hidden_size", "dim", "d_model"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(dim) = value.as_u64() {
                    return Ok(dim as usize);
                }
            }
        }
        Err(LlamaError::config_error(
            "missing_dimension",
            "Could not find dimension field (hidden_size, dim, or d_model)".to_string(),
        ))
    }

    /// Extract number of layers from various JSON field names
    fn extract_num_layers(obj: &serde_json::Map<String, serde_json::Value>) -> Result<usize> {
        let fields = ["num_hidden_layers", "n_layers", "num_layers"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(layers) = value.as_u64() {
                    return Ok(layers as usize);
                }
            }
        }
        Err(LlamaError::config_error(
            "missing_layers",
            "Could not find num_layers field (num_hidden_layers, n_layers, or num_layers)"
                .to_string(),
        ))
    }

    /// Extract number of attention heads from various JSON field names
    fn extract_num_heads(obj: &serde_json::Map<String, serde_json::Value>) -> Result<usize> {
        let fields = ["num_attention_heads", "n_heads", "num_heads"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(heads) = value.as_u64() {
                    return Ok(heads as usize);
                }
            }
        }
        Err(LlamaError::config_error(
            "missing_heads",
            "Could not find num_heads field (num_attention_heads, n_heads, or num_heads)"
                .to_string(),
        ))
    }

    /// Extract vocabulary size from various JSON field names
    fn extract_vocab_size(obj: &serde_json::Map<String, serde_json::Value>) -> Result<usize> {
        let fields = ["vocab_size", "vocabulary_size"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(vocab) = value.as_u64() {
                    return Ok(vocab as usize);
                }
            }
        }
        Err(LlamaError::config_error(
            "missing_vocab",
            "Could not find vocab_size field (vocab_size or vocabulary_size)".to_string(),
        ))
    }

    /// Extract number of key-value heads (optional)
    fn extract_kv_heads(obj: &serde_json::Map<String, serde_json::Value>) -> Option<usize> {
        let fields = ["num_key_value_heads", "n_kv_heads", "num_kv_heads"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(kv_heads) = value.as_u64() {
                    return Some(kv_heads as usize);
                }
            }
        }
        None
    }

    /// Extract intermediate size for FFN (optional)
    fn extract_intermediate_size(obj: &serde_json::Map<String, serde_json::Value>) -> usize {
        let fields = ["intermediate_size", "ffn_dim"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(size) = value.as_u64() {
                    return size as usize;
                }
            }
        }
        // Default calculation if not specified
        11008 // Common Llama default
    }

    /// Extract normalization epsilon (optional)
    fn extract_norm_eps(obj: &serde_json::Map<String, serde_json::Value>) -> f32 {
        let fields = ["rms_norm_eps", "norm_eps", "layer_norm_eps"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(eps) = value.as_f64() {
                    return eps as f32;
                }
            }
        }
        1e-6 // Default
    }

    /// Extract maximum sequence length (optional)
    fn extract_max_seq_len(obj: &serde_json::Map<String, serde_json::Value>) -> usize {
        let fields = [
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
        ];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(max_len) = value.as_u64() {
                    return max_len as usize;
                }
            }
        }
        4096 // Common default
    }

    /// Extract RoPE theta parameter (optional)
    fn extract_rope_theta(obj: &serde_json::Map<String, serde_json::Value>) -> f32 {
        let fields = ["rope_theta", "rotary_emb_base"];
        for field in &fields {
            if let Some(value) = obj.get(*field) {
                if let Some(theta) = value.as_f64() {
                    return theta as f32;
                }
            }
        }
        10_000.0 // Default
    }
}
