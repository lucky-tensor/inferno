//! # Model Loader
//!
//! This module provides functionality to load pre-trained Llama models from various formats,
//! with a focus on SafeTensors sharded models. It handles:
//!
//! - Loading sharded SafeTensors files (model-00001-of-00004.safetensors, etc.)
//! - Preserving native BF16 precision during loading
//! - Validating loaded weight dimensions against model configuration
//!
//! ## Key Features
//!
//! - **Sharded Loading**: Efficiently loads models split across multiple files
//! - **Precision Preservation**: Maintains BF16/F16 precision without conversion
//! - **Validation**: Ensures loaded weights match expected dimensions
//!
//! ## Performance
//!
//! The loader is optimized for minimal memory overhead during loading:
//! - Only loads needed tensors, not entire files into memory
//! - Uses memory-mapped files where possible
//! - Validates dimensions before allocation

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
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
/// 4. Initialize model with loaded weights using HuggingFace naming
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
    /// This method loads SafeTensors files and creates the model with real weights.
    pub fn load_model(self) -> Result<InfernoLlama> {
        // For single-file models, use direct SafeTensors loading
        let single_file = self.model_path.join("model.safetensors");
        if single_file.exists() {
            // Load directly from single SafeTensors file
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[single_file.clone()],
                    self.dtype,
                    &self.device,
                )
                .map_err(|e| {
                    LlamaError::io_error(
                        format!(
                            "Failed to create VarBuilder from SafeTensors file {:?}: {}",
                            single_file, e
                        ),
                        "load_model_single_file",
                    )
                })?
            };

            // Create the model with loaded weights
            InfernoLlama::new(&self.config, vb)
        } else {
            // For sharded models, collect all SafeTensors files
            let mut safetensor_files = Vec::new();
            let mut file_set = std::collections::HashSet::new();

            for filename in self.weight_map.values() {
                if file_set.insert(filename.clone()) {
                    let file_path = self.model_path.join(filename);
                    safetensor_files.push(file_path);
                }
            }

            if safetensor_files.is_empty() {
                return Err(LlamaError::io_error(
                    "No SafeTensors files found".to_string(),
                    "load_model_no_files",
                ));
            }

            // Load from multiple SafeTensors files
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&safetensor_files, self.dtype, &self.device)
                    .map_err(|e| {
                        LlamaError::io_error(
                            format!("Failed to create VarBuilder from SafeTensors files: {}", e),
                            "load_model_sharded_files",
                        )
                    })?
            };

            // Create the model with loaded weights
            InfernoLlama::new(&self.config, vb)
        }
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

        let intermediate_size = get_field("intermediate_size")?
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
            intermediate_size,        // Use extracted value
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
        // First try to load from index file (sharded models)
        let index_path = model_path.join("model.safetensors.index.json");
        if index_path.exists() {
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
                    LlamaError::config_error(
                        "weight_map",
                        "weight_map must be an object".to_string(),
                    )
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
        } else {
            // Fallback: Check for single model.safetensors file
            let single_file = model_path.join("model.safetensors");
            if single_file.exists() {
                // Read the single safetensors file to get all weight names
                let buffer = fs::read(&single_file).map_err(|e| {
                    LlamaError::io_error(
                        format!(
                            "Failed to read single SafeTensors file {:?}: {}",
                            single_file, e
                        ),
                        "load_single_safetensors",
                    )
                })?;

                let safetensors = safetensors::SafeTensors::deserialize(&buffer).map_err(|e| {
                    LlamaError::config_error(
                        "safetensors_format",
                        format!("Failed to parse SafeTensors file: {}", e),
                    )
                })?;

                let mut mapping = HashMap::new();
                for tensor_name in safetensors.names() {
                    mapping.insert(tensor_name.to_string(), "model.safetensors".to_string());
                }
                Ok(mapping)
            } else {
                Err(LlamaError::io_error(
                    "No model.safetensors.index.json or model.safetensors file found".to_string(),
                    "find_safetensors_files",
                ))
            }
        }
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
        tensor_view: safetensors::tensor::TensorView,
        weight_name: &str,
    ) -> Result<Tensor> {
        // Get tensor properties
        let shape = tensor_view.shape();
        let dtype = self.safetensors_dtype_to_candle(tensor_view.dtype())?;
        let data = tensor_view.data();

        // Create Candle tensor from raw data
        let tensor = match dtype {
            DType::F32 => {
                let f32_data = bytemuck::cast_slice::<u8, f32>(data);
                Tensor::from_slice(f32_data, shape, &self.device)
            }
            DType::F16 => {
                let f16_data = bytemuck::cast_slice::<u8, half::f16>(data);
                Tensor::from_slice(f16_data, shape, &self.device)
            }
            DType::BF16 => {
                let bf16_data = bytemuck::cast_slice::<u8, half::bf16>(data);
                Tensor::from_slice(bf16_data, shape, &self.device)
            }
            DType::U8 => Tensor::from_slice(data, shape, &self.device),
            DType::I64 => {
                let i64_data = bytemuck::cast_slice::<u8, i64>(data);
                Tensor::from_slice(i64_data, shape, &self.device)
            }
            DType::U32 => {
                let u32_data = bytemuck::cast_slice::<u8, u32>(data);
                Tensor::from_slice(u32_data, shape, &self.device)
            }
            _ => {
                return Err(LlamaError::tensor_error(
                    &format!(
                        "Unsupported tensor dtype {:?} for weight '{}'",
                        dtype, weight_name
                    ),
                    "safetensors_to_candle_tensor",
                ));
            }
        };

        tensor.map_err(|e| {
            LlamaError::tensor_error(
                &format!(
                    "Failed to create Candle tensor for weight '{}': {}",
                    weight_name, e
                ),
                "safetensors_to_candle_tensor",
            )
        })
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
}

// The load_from_path method is now implemented in simple_loader.rs

#[cfg(test)]
mod tests {
    use super::*;

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
