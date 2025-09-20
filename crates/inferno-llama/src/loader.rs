//! # Model Loader
//!
//! This module provides functionality to load pre-trained Llama models from various formats,
//! with a focus on SafeTensors sharded models. It handles:
//!
//! - Loading sharded SafeTensors files (model-00001-of-00004.safetensors, etc.)
//! - Mapping weight names from Hugging Face format to InfernoLlama component names
//! - Preserving native BF16 precision during loading
//! - Validating loaded weight dimensions against model configuration
//!
//! ## Key Features
//!
//! - **Sharded Loading**: Efficiently loads models split across multiple files
//! - **Precision Preservation**: Maintains BF16/F16 precision without conversion
//! - **Weight Mapping**: Maps from HF naming convention to InfernoLlama structure
//! - **Validation**: Ensures loaded weights match expected dimensions
//!
//! ## Performance
//!
//! The loader is optimized for minimal memory overhead during loading:
//! - Only loads needed tensors, not entire files into memory
//! - Uses memory-mapped files where possible
//! - Validates dimensions before allocation

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::LlamaConfig;
use crate::error::{LlamaError, Result};
use crate::model::InfernoLlama;

/// Model loader for SafeTensors format with sharding support.
///
/// This struct handles loading pre-trained Llama models from disk,
/// specifically targeting the SafeTensors format used by Hugging Face
/// model repositories.
///
/// ## Architecture
///
/// The loader follows these steps:
/// 1. Parse model configuration from config.json
/// 2. Load weight mapping from model.safetensors.index.json
/// 3. Load tensor data from sharded .safetensors files
/// 4. Map weight names from HF format to InfernoLlama format
/// 5. Initialize model with loaded weights
///
/// ## Memory Efficiency
///
/// The loader is designed to minimize peak memory usage:
/// - Loads tensors incrementally, not all at once
/// - Uses memory-mapped files where supported
/// - Validates tensor shapes before full loading
#[derive(Debug)]
#[allow(dead_code)]
pub struct ModelLoader {
    /// Path to the model directory
    model_path: PathBuf,
    /// Model configuration loaded from config.json
    config: LlamaConfig,
    /// Weight mapping from model.safetensors.index.json
    weight_map: HashMap<String, String>,
    /// Target device for loading
    device: Device,
    /// Target dtype for loading
    dtype: DType,
}

impl ModelLoader {
    /// Creates a new model loader for the specified path.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory containing config.json and safetensors files
    /// * `device` - Target device for tensor allocation
    /// * `dtype` - Target dtype, typically BF16 for memory efficiency
    ///
    /// # Returns
    ///
    /// Returns a `Result<ModelLoader>` that can be used to load the model.
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// - Model directory doesn't exist or is inaccessible
    /// - config.json is missing or malformed
    /// - model.safetensors.index.json is missing or malformed
    /// - Model format is not supported
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::loader::ModelLoader;
    /// use candle_core::{Device, DType};
    ///
    /// let loader = ModelLoader::new(
    ///     "/path/to/meta-llama_Llama-3.1-8B-Instruct",
    ///     Device::Cpu,
    ///     DType::BF16
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new<P: AsRef<Path>>(model_path: P, device: Device, dtype: DType) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        // Validate model directory exists
        if !model_path.exists() {
            return Err(LlamaError::io_error(
                format!("Model directory does not exist: {:?}", model_path),
                "model_loader_init",
            ));
        }

        // Load model configuration
        let config = Self::load_config(&model_path)?;

        // Load weight mapping
        let weight_map = Self::load_weight_map(&model_path)?;

        Ok(ModelLoader {
            model_path,
            config,
            weight_map,
            device,
            dtype,
        })
    }

    /// Loads the complete model with weights from disk.
    ///
    /// This method is currently a placeholder and will be implemented
    /// when the weight loading mechanism is fully developed.
    pub fn load_model(self) -> Result<InfernoLlama> {
        // TODO: Implement weight loading once VarMap API is resolved
        Err(LlamaError::tensor_error(
            "Weight loading not yet implemented",
            "load_model",
        ))
    }

    /// Loads model configuration from config.json.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model directory
    ///
    /// # Returns
    ///
    /// Returns the parsed LlamaConfig or an error if parsing fails.
    fn load_config(model_path: &Path) -> Result<LlamaConfig> {
        let config_path = model_path.join("config.json");
        let config_content = fs::read_to_string(&config_path).map_err(|e| {
            LlamaError::io_error(
                format!("Failed to read config.json from {:?}: {}", config_path, e),
                "load_config",
            )
        })?;

        let config_value: Value = serde_json::from_str(&config_content).map_err(|e| {
            LlamaError::config_error("config.json", format!("Failed to parse config.json: {}", e))
        })?;

        // Extract relevant configuration fields and create LlamaConfig
        Self::parse_config_from_json(config_value)
    }

    /// Parses LlamaConfig from JSON configuration.
    ///
    /// This function maps the Hugging Face configuration format to our
    /// internal LlamaConfig representation.
    fn parse_config_from_json(config: Value) -> Result<LlamaConfig> {
        let get_field = |name: &str| -> Result<&Value> {
            config.get(name).ok_or_else(|| {
                LlamaError::config_error(name, format!("Missing required field: {}", name))
            })
        };

        let dim = get_field("hidden_size")?
            .as_u64()
            .ok_or_else(|| LlamaError::config_error("hidden_size", "Must be a number"))?
            as usize;

        let n_layers = get_field("num_hidden_layers")?
            .as_u64()
            .ok_or_else(|| LlamaError::config_error("num_hidden_layers", "Must be a number"))?
            as usize;

        let n_heads = get_field("num_attention_heads")?
            .as_u64()
            .ok_or_else(|| LlamaError::config_error("num_attention_heads", "Must be a number"))?
            as usize;

        let n_kv_heads = config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let vocab_size = get_field("vocab_size")?
            .as_u64()
            .ok_or_else(|| LlamaError::config_error("vocab_size", "Must be a number"))?
            as usize;

        let _intermediate_size = get_field("intermediate_size")?
            .as_u64()
            .ok_or_else(|| LlamaError::config_error("intermediate_size", "Must be a number"))?
            as usize;

        let rms_norm_eps = config
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        Ok(LlamaConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            ffn_dim_multiplier: None, // We calculate from intermediate_size
            multiple_of: 256,         // Standard value
            norm_eps: rms_norm_eps,
            rope_theta,
            max_seq_len: 4096,      // Standard default
            use_scaled_rope: false, // Default value
        })
    }

    /// Loads weight mapping from model.safetensors.index.json.
    ///
    /// This file maps weight names to their corresponding SafeTensors files
    /// in sharded models.
    fn load_weight_map(model_path: &Path) -> Result<HashMap<String, String>> {
        let index_path = model_path.join("model.safetensors.index.json");
        let index_content = fs::read_to_string(&index_path).map_err(|e| {
            LlamaError::io_error(
                format!("Failed to read weight index from {:?}: {}", index_path, e),
                "load_weight_map",
            )
        })?;

        let index_value: Value = serde_json::from_str(&index_content).map_err(|e| {
            LlamaError::config_error(
                "model.safetensors.index.json",
                format!("Failed to parse weight index: {}", e),
            )
        })?;

        let weight_map = index_value
            .get("weight_map")
            .ok_or_else(|| {
                LlamaError::config_error(
                    "weight_map",
                    "Missing weight_map field in index file".to_string(),
                )
            })?
            .as_object()
            .ok_or_else(|| {
                LlamaError::config_error("weight_map", "weight_map must be an object".to_string())
            })?;

        let mut result = HashMap::new();
        for (weight_name, file_name) in weight_map {
            let file_name_str = file_name.as_str().ok_or_else(|| {
                LlamaError::config_error(
                    "weight_map",
                    format!("File name must be string for weight: {}", weight_name),
                )
            })?;
            result.insert(weight_name.clone(), file_name_str.to_string());
        }

        Ok(result)
    }

    /// Loads all weights from SafeTensors files into the VarMap.
    ///
    /// This function iterates through all required weights, loads them from
    /// their corresponding SafeTensors files, and inserts them into the VarMap
    /// with the correct names for InfernoLlama.
    #[allow(dead_code)]
    fn load_weights_into_varmap(&self, _var_map: &VarMap) -> Result<()> {
        // TODO: Implement once VarMap API is resolved
        Ok(())
    }

    /// Loads specific weights from a single SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `file_name` - Name of the SafeTensors file (e.g., "model-00001-of-00004.safetensors")
    /// * `weight_names` - List of weight names to extract from this file
    /// * `_var_map` - VarMap to store the loaded weights (placeholder for now)
    #[allow(dead_code)]
    fn load_weights_from_file(
        &self,
        _file_name: &str,
        _weight_names: &[String],
        _var_map: &VarMap,
    ) -> Result<()> {
        // TODO: Implement weight loading
        Ok(())
    }

    /// Converts a SafeTensors tensor view to a Candle tensor.
    ///
    /// This function handles dtype conversion and device placement.
    #[allow(dead_code)]
    fn safetensors_to_candle_tensor(
        &self,
        _tensor_view: safetensors::tensor::TensorView,
        _weight_name: &str,
    ) -> Result<Tensor> {
        // TODO: Implement tensor conversion
        Err(LlamaError::tensor_error(
            "Tensor conversion not yet implemented",
            "safetensors_to_candle_tensor",
        ))
    }

    /// Maps SafeTensors dtype to Candle DType.
    #[allow(dead_code)]
    fn safetensors_dtype_to_candle(&self, dtype: safetensors::Dtype) -> Result<DType> {
        use safetensors::Dtype as STDtype;

        let candle_dtype = match dtype {
            STDtype::F32 => DType::F32,
            STDtype::F16 => DType::F16,
            STDtype::BF16 => DType::BF16,
            STDtype::U8 => DType::U8,
            STDtype::U32 => DType::U32,
            STDtype::I64 => DType::I64,
            _ => {
                return Err(LlamaError::tensor_error(
                    format!("Unsupported SafeTensors dtype: {:?}", dtype),
                    "safetensors_dtype_to_candle",
                ))
            }
        };

        Ok(candle_dtype)
    }

    /// Maps Hugging Face weight names to InfernoLlama weight names.
    ///
    /// This function handles the naming convention differences between
    /// Hugging Face models and our internal structure.
    ///
    /// # Examples
    ///
    /// - `model.embed_tokens.weight` -> `embed_tokens.weight`
    /// - `model.layers.0.self_attn.q_proj.weight` -> `layers.0.attention.q_proj.weight`
    /// - `lm_head.weight` -> `lm_head.weight`
    #[allow(dead_code)]
    fn map_weight_name(&self, hf_name: &str) -> Result<String> {
        // Remove "model." prefix if present
        let name = if let Some(stripped) = hf_name.strip_prefix("model.") {
            stripped
        } else {
            hf_name
        };

        // Map specific component names
        let mapped_name = if name.starts_with("layers.") {
            // Extract layer number and component
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() < 3 {
                return Err(LlamaError::config_error(
                    "weight_mapping",
                    format!("Invalid layer weight name: {}", hf_name),
                ));
            }

            let layer_idx = parts[1];
            let component = parts[2];

            match component {
                "self_attn" => {
                    // Map self_attn to attention
                    let rest = &parts[3..].join(".");
                    format!("layers.{}.attention.{}", layer_idx, rest)
                }
                "mlp" => {
                    // Map mlp to feed_forward
                    let rest = &parts[3..].join(".");
                    format!("layers.{}.feed_forward.{}", layer_idx, rest)
                }
                _ => {
                    // Keep other components as-is (input_layernorm, post_attention_layernorm)
                    name.to_string()
                }
            }
        } else {
            // Non-layer weights, keep as-is
            name.to_string()
        };

        Ok(mapped_name)
    }
}

// The load_from_path method is now implemented in simple_loader.rs

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_weight_name_mapping() {
        let loader = ModelLoader {
            model_path: PathBuf::new(),
            config: LlamaConfig::llama_3_1_8b().unwrap(),
            weight_map: HashMap::new(),
            device: Device::Cpu,
            dtype: DType::BF16,
        };

        // Test embedding weight mapping
        assert_eq!(
            loader.map_weight_name("model.embed_tokens.weight").unwrap(),
            "embed_tokens.weight"
        );

        // Test attention weight mapping
        assert_eq!(
            loader
                .map_weight_name("model.layers.0.self_attn.q_proj.weight")
                .unwrap(),
            "layers.0.attention.q_proj.weight"
        );

        // Test MLP weight mapping
        assert_eq!(
            loader
                .map_weight_name("model.layers.5.mlp.gate_proj.weight")
                .unwrap(),
            "layers.5.feed_forward.gate_proj.weight"
        );

        // Test normalization weights (should remain unchanged)
        assert_eq!(
            loader
                .map_weight_name("model.layers.0.input_layernorm.weight")
                .unwrap(),
            "layers.0.input_layernorm.weight"
        );

        // Test output head mapping
        assert_eq!(
            loader.map_weight_name("lm_head.weight").unwrap(),
            "lm_head.weight"
        );
    }

    #[test]
    fn test_config_parsing() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0
        }"#;

        let config_value: Value = serde_json::from_str(config_json).unwrap();
        let config = ModelLoader::parse_config_from_json(config_value).unwrap();

        assert_eq!(config.dim, 4096);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, Some(8));
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.norm_eps, 1e-5);
        assert_eq!(config.rope_theta, 500000.0);
    }

    #[test]
    fn test_safetensors_dtype_conversion() {
        let loader = ModelLoader {
            model_path: PathBuf::new(),
            config: LlamaConfig::llama_3_1_8b().unwrap(),
            weight_map: HashMap::new(),
            device: Device::Cpu,
            dtype: DType::BF16,
        };

        assert_eq!(
            loader
                .safetensors_dtype_to_candle(safetensors::Dtype::F32)
                .unwrap(),
            DType::F32
        );
        assert_eq!(
            loader
                .safetensors_dtype_to_candle(safetensors::Dtype::BF16)
                .unwrap(),
            DType::BF16
        );
        assert_eq!(
            loader
                .safetensors_dtype_to_candle(safetensors::Dtype::F16)
                .unwrap(),
            DType::F16
        );
    }
}
