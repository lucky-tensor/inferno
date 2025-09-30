//! Model variant detection implementation
//!
//! This module analyzes model directories to automatically detect the specific
//! Llama variant (Meta Llama 3.1/3.2, TinyLlama, DeepSeek distilled, etc.)

use crate::candle_extensions::llama_models::GenericLlamaConfig;
use std::path::Path;

/// Main model detection engine
#[derive(Debug)]
pub struct ModelDetector;

impl Default for ModelDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDetector {
    /// Create a new ModelDetector
    pub fn new() -> Self {
        Self
    }

    /// Detect model variant from a model directory
    ///
    /// Analyzes config.json, model files, and directory structure
    /// to determine the specific Llama model variant.
    pub async fn detect_variant(
        model_path: &str,
    ) -> Result<GenericLlamaConfig, Box<dyn std::error::Error>> {
        // Check if directory contains a valid model
        if !Self::is_valid_model_directory(model_path).await {
            return Err(format!("Invalid model directory: {}", model_path).into());
        }

        // Parse configuration
        let config = crate::diagnostic::ConfigParser::parse_config(model_path).await?;

        // Detect variant from config
        let variant = crate::diagnostic::ConfigParser::detect_variant_from_config(&config);

        // Analyze weights for quantization detection
        let quantization =
            crate::diagnostic::WeightAnalyzer::detect_quantization(model_path).await?;

        // Analyze model memory layout
        let (is_sharded, num_shards) =
            crate::diagnostic::WeightAnalyzer::is_sharded(model_path).await?;

        // Estimate parameters (basic calculation)
        let total_params = Self::estimate_parameters(&config);
        let estimated_memory_bytes = Self::estimate_memory(&config, &quantization);

        let memory_layout = crate::diagnostic::ModelMemoryLayout {
            total_params,
            primary_dtype: Self::detect_primary_dtype(&quantization),
            is_sharded,
            num_shards,
            estimated_memory_bytes,
            layer_memory_map: std::collections::HashMap::new(), // TODO: Implement detailed layer analysis
            optimization_hints: Vec::new(),                     // TODO: Generate optimization hints
        };

        Ok(GenericLlamaConfig {
            base: config,
            variant,
            quantization: if quantization.scheme == crate::diagnostic::QuantizationScheme::None {
                None
            } else {
                Some(quantization)
            },
            memory_layout,
        })
    }

    /// Check if a directory contains a valid Llama model
    pub async fn is_valid_model_directory(path: &str) -> bool {
        let path = Path::new(path);

        // Check for config.json
        if !tokio::fs::try_exists(path.join("config.json"))
            .await
            .unwrap_or(false)
        {
            return false;
        }

        // Check for at least one model file
        let model_extensions = ["safetensors", "bin", "pth"];

        if let Ok(mut entries) = tokio::fs::read_dir(path).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Some(file_ext) = entry.path().extension() {
                    let ext_str = file_ext.to_string_lossy().to_lowercase();
                    for model_ext in &model_extensions {
                        if ext_str == *model_ext {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Estimate total parameters from configuration
    fn estimate_parameters(config: &crate::candle_extensions::llama_models::Config) -> u64 {
        // Basic calculation based on transformer architecture
        let embedding_params = config.vocab_size * config.hidden_size;
        let attention_params_per_layer = 4 * config.hidden_size * config.hidden_size;
        let ffn_params_per_layer = 3 * config.hidden_size * config.intermediate_size;
        let norm_params_per_layer = config.hidden_size;

        let layer_params =
            (attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer)
                * config.num_hidden_layers;

        (embedding_params + layer_params) as u64
    }

    /// Estimate memory usage from configuration and quantization
    fn estimate_memory(
        config: &crate::candle_extensions::llama_models::Config,
        quantization: &crate::diagnostic::QuantizationConfig,
    ) -> u64 {
        let total_params = Self::estimate_parameters(config);
        let bytes_per_param = match quantization.scheme {
            crate::diagnostic::QuantizationScheme::W8A8 => 1,
            crate::diagnostic::QuantizationScheme::W4A16 => 1, // Rounded up for W4
            _ => 2,                                            // Default to FP16
        };

        total_params * bytes_per_param
    }

    /// Detect primary data type from quantization scheme
    fn detect_primary_dtype(
        quantization: &crate::diagnostic::QuantizationConfig,
    ) -> candle_core::DType {
        match quantization.scheme {
            crate::diagnostic::QuantizationScheme::W8A8 => candle_core::DType::U8,
            crate::diagnostic::QuantizationScheme::W4A16 => candle_core::DType::U8, // We'll store W4 as U8
            _ => candle_core::DType::BF16, // Default to BF16 for modern models
        }
    }
}
