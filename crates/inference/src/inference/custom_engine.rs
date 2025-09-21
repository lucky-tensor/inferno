//! Custom inference engine using our SafeTensors-compatible Llama implementation

use super::{InferenceEngine, InferenceError, InferenceRequest, InferenceResponse};
use crate::config::InfernoConfig;
use crate::models::{CustomLlama, CustomLlamaConfig};
use burn::{backend::ndarray::NdArray, tensor::Device};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use tracing::{info, warn};

// Type alias for our backend
type Backend = NdArray<f32>;

/// Custom inference engine with SafeTensors support
pub struct CustomInferenceEngine {
    /// Whether the engine is initialized
    initialized: bool,
    /// Model configuration
    config: Option<InfernoConfig>,
    /// Loaded Custom Llama model - wrapped in Mutex for interior mutability
    model: Option<Mutex<CustomLlama<Backend>>>,
    /// Model ready for inference
    model_ready: bool,
    /// Request count for statistics
    request_count: Mutex<u64>,
    /// Total inference time for averaging
    total_inference_time: Mutex<f64>,
    /// Device for tensor operations
    device: Device<Backend>,
}

impl CustomInferenceEngine {
    /// Create a new custom inference engine
    pub fn new() -> Self {
        Self {
            initialized: false,
            config: None,
            model: None,
            model_ready: false,
            request_count: Mutex::new(0),
            total_inference_time: Mutex::new(0.0),
            device: Device::<Backend>::default(),
        }
    }

    /// Initialize the engine with configuration
    pub async fn initialize(&mut self, config: InfernoConfig) -> Result<(), InferenceError> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing CustomInferenceEngine with SafeTensors support");

        self.config = Some(config.clone());

        // Load model from SafeTensors
        let model_path = if config.model_path.is_empty() {
            return Err(InferenceError::InitializationError("Model path is empty".to_string()));
        } else {
            PathBuf::from(&config.model_path)
        };

        info!("Loading model from: {:?}", model_path);

        // Load model configuration from config.json
        let model_config = self.load_huggingface_config(&model_path)?;
        info!("Loaded model config: {:?}", model_config);

        // Create custom Llama model
        let mut model = CustomLlama::new(model_config, &self.device);

        // Load SafeTensors weights
        let safetensors_path = model_path.join("model.safetensors");
        match model.load_safetensors_weights(&safetensors_path, &self.device) {
            Ok(()) => {
                info!("✅ Successfully loaded SafeTensors weights!");
                self.model_ready = true;
            }
            Err(e) => {
                warn!("Failed to load SafeTensors weights: {}. Using initialized weights.", e);
                self.model_ready = false;
            }
        }

        self.model = Some(Mutex::new(model));
        self.initialized = true;

        info!("CustomInferenceEngine initialized successfully!");
        Ok(())
    }

    /// Load HuggingFace config.json
    fn load_huggingface_config(&self, model_path: &std::path::Path) -> Result<CustomLlamaConfig, InferenceError> {
        let config_path = model_path.join("config.json");

        if !config_path.exists() {
            info!("config.json not found, using TinyLlama defaults");
            return Ok(CustomLlamaConfig::default());
        }

        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| InferenceError::InitializationError(format!("Failed to read config.json: {}", e)))?;

        let hf_config: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| InferenceError::InitializationError(format!("Failed to parse config.json: {}", e)))?;

        let config = CustomLlamaConfig {
            vocab_size: hf_config["vocab_size"].as_u64().unwrap_or(32000) as usize,
            hidden_size: hf_config["hidden_size"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: hf_config["intermediate_size"].as_u64().unwrap_or(5632) as usize,
            num_hidden_layers: hf_config["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
            num_attention_heads: hf_config["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: hf_config["num_key_value_heads"].as_u64().map(|v| v as usize),
            max_position_embeddings: hf_config["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
            rms_norm_eps: hf_config["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            rope_theta: hf_config["rope_theta"].as_f64().unwrap_or(10000.0),
        };

        Ok(config)
    }

    /// Process a single inference request
    pub fn process_sync(&self, mut request: InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        // Ensure request has an ID
        if request.request_id == 0 {
            let count = self.request_count.lock().unwrap();
            request.request_id = *count + 1;
        }

        if !self.initialized {
            return Err(InferenceError::ProcessingError("Engine not initialized".to_string()));
        }

        let start_time = Instant::now();
        {
            let mut count = self.request_count.lock().unwrap();
            *count += 1;
        }

        info!("Processing inference request with CustomLlama: {}", request.prompt);

        // Real inference with Custom Llama model
        let response_text = {
            if let Some(ref model_mutex) = self.model {
                let mut model = model_mutex.lock().unwrap();

                // Perform REAL neural network text generation
                match model.generate(
                    &request.prompt,
                    request.max_tokens as usize,
                    request.temperature,
                ) {
                    Ok(generated_text) => {
                        info!("✅ CustomLlama generated {} characters", generated_text.len());
                        generated_text
                    }
                    Err(e) => {
                        warn!("CustomLlama generation failed: {}", e);
                        return Err(InferenceError::ProcessingError(format!("Generation failed: {}", e)));
                    }
                }
            } else {
                return Err(InferenceError::ProcessingError("Model not loaded".to_string()));
            }
        };

        let inference_time = start_time.elapsed().as_secs_f64();

        // Update statistics
        {
            let mut total_time = self.total_inference_time.lock().unwrap();
            *total_time += inference_time;
        }

        info!("CustomLlama inference completed in {:.3}s", inference_time);

        // Calculate generated tokens (rough approximation)
        let generated_tokens = response_text.split_whitespace().count();

        Ok(InferenceResponse {
            request_id: request.request_id,
            generated_text: response_text,
            generated_tokens: generated_tokens as u32,
            inference_time_ms: inference_time * 1000.0,
            time_to_first_token_ms: None,
            is_finished: true,
            error: None,
        })
    }

    /// Check if the engine is ready for inference
    pub fn is_ready(&self) -> bool {
        self.initialized && self.model_ready
    }

    /// Shutdown the engine
    pub fn shutdown(&mut self) -> Result<(), InferenceError> {
        info!("Shutting down CustomInferenceEngine");
        self.initialized = false;
        self.model_ready = false;
        self.model = None;
        Ok(())
    }
}

impl Default for CustomInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Implement the InferenceEngine trait for CustomInferenceEngine
#[async_trait::async_trait]
impl InferenceEngine for CustomInferenceEngine {
    type Error = InferenceError;

    async fn initialize(&mut self, config: InfernoConfig) -> Result<(), Self::Error> {
        self.initialize(config).await
    }

    async fn process(&self, request: InferenceRequest) -> Result<InferenceResponse, Self::Error> {
        // Call the sync process method
        self.process_sync(request)
    }
}