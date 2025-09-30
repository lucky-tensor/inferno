//! # Unified Model Factory
//!
//! The UnifiedModelFactory provides a single interface for auto-detecting and loading
//! any supported Llama model variant using the diagnostic system.
//!
//! ## Key Features
//!
//! - **Auto-detection**: Automatically detects model variant from directory structure
//! - **Unified API**: Single interface for all model types (Meta Llama, TinyLlama, quantized)
//! - **Weight Loading**: Handles sharded models, quantized models, and different formats
//! - **Configuration**: Automatically parses and validates configuration for each variant
//! - **Error Handling**: Provides clear error messages for debugging
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferno_llama::factory::UnifiedModelFactory;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let factory = UnifiedModelFactory::new()?;
//!
//! // Auto-detect and load any Llama variant
//! let config = factory.detect_model_config("/path/to/model").await?;
//! let model = factory.load_model("/path/to/model", config).await?;
//!
//! // Or load with integrated tokenizer for text generation
//! let model_with_tokenizer = factory.load_model_with_tokenizer("/path/to/model").await?;
//! let generated = model_with_tokenizer.generate_text("Hello", 10).await?;
//! # Result::<(), Box<dyn std::error::Error>>::Ok(())
//! # }).unwrap();
//! ```
//!
//! ## Supported Models
//!
//! - **Meta Llama 3.1/3.2**: Standard and sharded models
//! - **TinyLlama**: Distilled 1B parameter models
//! - **Quantized Models**: w8a8 and compressed-tensor formats
//! - **Custom Models**: Through manual configuration

use candle_core::{DType, Device};

use crate::candle_extensions::GenericLlamaConfig;
use crate::diagnostic::ModelDetector;
use crate::error::{LlamaError, Result};
use crate::loader::ModelLoader;
use crate::model::InfernoLlama;
use crate::precision::PrecisionConfig;
use crate::tokenizer::TokenizedInfernoLlama;

/// Unified factory for auto-detecting and loading any supported Llama model variant.
///
/// The UnifiedModelFactory combines the diagnostic system with model loading
/// to provide a single interface for working with different Llama model types.
///
/// ## Architecture
///
/// ```text
/// Model Directory → [Detect] → [Parse Config] → [Load Weights] → [Create Model]
///       ↓              ↓            ↓              ↓              ↓
///   File Analysis → Variant ID → Generic Config → Tensors → InfernoLlama
/// ```
///
/// ## Performance Characteristics
///
/// - **Detection**: O(1) with caching - subsequent detections of same model are instant
/// - **Loading**: O(model_size) - scales with number of parameters and precision
/// - **Memory**: Efficient BF16/F16 loading respects theoretical memory limits
///
/// ## Error Handling
///
/// The factory provides detailed error information for:
/// - Unsupported model types
/// - Missing or corrupted files
/// - Configuration parsing errors
/// - Memory allocation failures
/// - Weight loading issues
#[derive(Debug)]
pub struct UnifiedModelFactory {
    /// Model detector for variant identification
    #[allow(dead_code)]
    detector: ModelDetector,
    /// Default device for model loading
    device: Device,
    /// Default precision configuration
    default_precision: PrecisionConfig,
}

impl UnifiedModelFactory {
    /// Creates a new UnifiedModelFactory with default settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result<UnifiedModelFactory>` configured with:
    /// - CPU device (GPU auto-detection planned for future)
    /// - BF16 precision for memory efficiency
    /// - Diagnostic caching enabled
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// - Device initialization fails
    /// - Default precision configuration is invalid
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::factory::UnifiedModelFactory;
    ///
    /// let factory = UnifiedModelFactory::new()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new() -> Result<Self> {
        let device = Device::Cpu; // TODO: Add GPU auto-detection
        let default_precision = PrecisionConfig::default(); // BF16 for efficiency
        let detector = ModelDetector::new();

        Ok(Self {
            detector,
            device,
            default_precision,
        })
    }

    /// Creates a factory with custom device and precision settings.
    ///
    /// # Arguments
    ///
    /// * `device` - Target device for model loading
    /// * `precision` - Default precision configuration
    ///
    /// # Returns
    ///
    /// Returns a configured `UnifiedModelFactory`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use candle_core::Device;
    /// use inferno_llama::factory::UnifiedModelFactory;
    /// use inferno_llama::precision::PrecisionConfig;
    ///
    /// let device = Device::Cpu;
    /// let precision = PrecisionConfig::bf16();
    /// let factory = UnifiedModelFactory::with_config(device, precision)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_config(device: Device, precision: PrecisionConfig) -> Result<Self> {
        let detector = ModelDetector::new();

        Ok(Self {
            detector,
            device,
            default_precision: precision,
        })
    }

    /// Auto-detects model configuration from a directory path.
    ///
    /// This method analyzes the model directory to determine:
    /// - Model variant (Meta Llama, TinyLlama, etc.)
    /// - Architecture parameters (layers, heads, dimensions)
    /// - Quantization scheme if present
    /// - Memory layout and optimization hints
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory
    ///
    /// # Returns
    ///
    /// Returns `Result<GenericLlamaConfig>` containing the detected configuration,
    /// or an error if detection fails.
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// - Model directory does not exist
    /// - No valid configuration files found
    /// - Model variant is unsupported
    /// - Configuration is malformed or invalid
    ///
    /// # Performance
    ///
    /// - First call: O(file_count) - analyzes all files in directory
    /// - Subsequent calls: O(1) - uses cached results
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::factory::UnifiedModelFactory;
    /// use inferno_llama::candle_extensions::LlamaVariant;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let factory = UnifiedModelFactory::new()?;
    /// let config = factory.detect_model_config("/path/to/llama").await?;
    ///
    /// match config.variant {
    ///     LlamaVariant::MetaLlama31 => println!("Detected Meta Llama 3.1"),
    ///     LlamaVariant::TinyLlama => println!("Detected TinyLlama"),
    ///     _ => println!("Detected other variant"),
    /// }
    /// # Result::<(), Box<dyn std::error::Error>>::Ok(())
    /// # }).unwrap();
    /// ```
    pub async fn detect_model_config(&self, model_path: &str) -> Result<GenericLlamaConfig> {
        // Use the diagnostic system for detection
        ModelDetector::detect_variant(model_path)
            .await
            .map_err(|e| {
                LlamaError::config_error(
                    "model_detection",
                    format!("Failed to detect model variant at '{}': {}", model_path, e),
                )
            })
    }

    /// Loads a model from the specified path using the provided configuration.
    ///
    /// This method handles the complete model loading pipeline:
    /// 1. Validates configuration compatibility
    /// 2. Creates appropriate VarBuilder for weight loading
    /// 3. Loads weights using the correct loader for the model type
    /// 4. Constructs and returns the InfernoLlama model
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory
    /// * `config` - Pre-detected model configuration
    ///
    /// # Returns
    ///
    /// Returns `Result<InfernoLlama>` containing the loaded model,
    /// or an error if loading fails.
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// - Model weights are missing or corrupted
    /// - Configuration is incompatible with weights
    /// - Insufficient memory for model loading
    /// - Device initialization fails
    ///
    /// # Performance
    ///
    /// - Time complexity: O(model_size) - depends on parameter count and precision
    /// - Memory usage: Respects theoretical limits (e.g., ~15GB for Llama 3.1 8B in BF16)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::factory::UnifiedModelFactory;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let factory = UnifiedModelFactory::new()?;
    /// let config = factory.detect_model_config("/path/to/llama").await?;
    /// let model = factory.load_model("/path/to/llama", config).await?;
    ///
    /// println!("Loaded model with {} parameters", model.parameter_count());
    /// # Result::<(), Box<dyn std::error::Error>>::Ok(())
    /// # }).unwrap();
    /// ```
    pub async fn load_model(
        &self,
        model_path: &str,
        config: GenericLlamaConfig,
    ) -> Result<InfernoLlama> {
        // Validate configuration
        config.validate().map_err(|e| {
            LlamaError::config_error("config_validation", format!("Invalid configuration: {}", e))
        })?;

        // Determine precision from config and quantization
        let dtype = self.determine_dtype(&config)?;

        // Use the existing ModelLoader which handles the complete loading pipeline
        let loader = ModelLoader::new(model_path, self.device.clone(), dtype)?;
        let model = loader.load_model()?;

        Ok(model)
    }

    /// Loads a model with integrated tokenizer for end-to-end text processing.
    ///
    /// This is a convenience method that combines model loading with tokenizer
    /// initialization, providing a complete text-to-text interface.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory
    ///
    /// # Returns
    ///
    /// Returns `Result<TokenizedInfernoLlama>` containing the model with tokenizer,
    /// or an error if loading fails.
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// - Model loading fails (see `load_model` errors)
    /// - Tokenizer files are missing or invalid
    /// - Tokenizer initialization fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::factory::UnifiedModelFactory;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let factory = UnifiedModelFactory::new()?;
    /// let model = factory.load_model_with_tokenizer("/path/to/llama").await?;
    ///
    /// // Can now generate text directly
    /// let generated = model.generate_text("Hello", 10).await?;
    /// println!("Generated: {}", generated);
    /// # Result::<(), Box<dyn std::error::Error>>::Ok(())
    /// # }).unwrap();
    /// ```
    pub async fn load_model_with_tokenizer(
        &self,
        model_path: &str,
    ) -> Result<TokenizedInfernoLlama> {
        // Use the existing TokenizedInfernoLlama implementation
        TokenizedInfernoLlama::load_from_path(model_path).await
    }

    /// Helper method to determine the appropriate DType for loading
    fn determine_dtype(&self, config: &GenericLlamaConfig) -> Result<DType> {
        // Check if quantization is specified
        if let Some(quantization) = &config.quantization {
            match quantization.scheme {
                crate::diagnostic::QuantizationScheme::W8A8 => {
                    // For quantized models, we typically load in F32 and dequantize
                    Ok(DType::F32)
                }
                crate::diagnostic::QuantizationScheme::CompressedTensors(_) => {
                    // Compressed tensors often use F16
                    Ok(DType::F16)
                }
                _ => {
                    // Default to BF16 for efficiency
                    Ok(DType::BF16)
                }
            }
        } else {
            // For non-quantized models, use the default precision
            Ok(self.default_precision.to_dtype())
        }
    }
}

impl Default for UnifiedModelFactory {
    fn default() -> Self {
        Self::new().expect("Default factory creation should not fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_initialization() {
        let factory = UnifiedModelFactory::new();
        assert!(factory.is_ok(), "Factory initialization should succeed");

        let _factory = factory.unwrap();
        // Device comparison not supported by candle-core, just verify creation succeeded
    }

    #[test]
    fn test_factory_with_custom_config() {
        let device = Device::Cpu;
        let precision = PrecisionConfig::from_dtype(DType::F16).unwrap();

        let factory = UnifiedModelFactory::with_config(device.clone(), precision.clone());
        assert!(
            factory.is_ok(),
            "Custom factory configuration should succeed"
        );

        let factory = factory.unwrap();
        // Device comparison not supported, just verify creation succeeded
        assert_eq!(factory.default_precision, precision);
    }

    #[test]
    fn test_dtype_determination() {
        let factory = UnifiedModelFactory::new().unwrap();

        // Test non-quantized model
        let config = GenericLlamaConfig {
            base: crate::candle_extensions::llama_models::Config::default(),
            variant: crate::candle_extensions::LlamaVariant::MetaLlama31,
            quantization: None,
            memory_layout: crate::diagnostic::ModelMemoryLayout::default(),
        };

        let dtype = factory.determine_dtype(&config).unwrap();
        assert_eq!(dtype, DType::BF16); // Default precision

        // Test quantized model
        let mut config_quantized = config.clone();
        config_quantized.quantization = Some(crate::diagnostic::QuantizationConfig {
            scheme: crate::diagnostic::QuantizationScheme::W8A8,
            per_layer_config: None,
            global_params: None,
            symmetric: true,
            zero_point: None,
            scale: None,
        });

        let dtype_quantized = factory.determine_dtype(&config_quantized).unwrap();
        assert_eq!(dtype_quantized, DType::F32); // For W8A8 dequantization
    }

    #[test]
    fn test_factory_basic_functionality() {
        let factory = UnifiedModelFactory::new().unwrap();

        // Test that the factory can determine dtypes correctly
        let generic_config = GenericLlamaConfig {
            base: crate::candle_extensions::llama_models::Config {
                hidden_size: 4096,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                vocab_size: 128256,
                intermediate_size: 11008,
                use_flash_attn: false,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                bos_token_id: Some(1),
                eos_token_id: None,
                rope_scaling: None,
                max_position_embeddings: 4096,
                tie_word_embeddings: false,
            },
            variant: crate::candle_extensions::LlamaVariant::MetaLlama31,
            quantization: None,
            memory_layout: crate::diagnostic::ModelMemoryLayout::default(),
        };

        let dtype = factory.determine_dtype(&generic_config);
        assert!(dtype.is_ok(), "DType determination should succeed");
        assert_eq!(dtype.unwrap(), DType::BF16);
    }
}
