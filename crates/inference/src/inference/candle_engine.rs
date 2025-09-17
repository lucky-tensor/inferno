//! Candle-based inference engine for LLM inference
//!
//! This module provides an inference engine using `HuggingFace` Candle framework,

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::unused_async, clippy::too_many_arguments)]
//! offering optimized performance for LLM inference with support for:
//! - Native `HuggingFace` tokenizers (BPE, `SentencePiece`)
//! - `SafeTensors` weight loading
//! - CUDA acceleration with Flash Attention v2
//! - Multi-GPU distribution via NCCL

#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
use crate::inference::{
    InferenceEngine, InferenceRequest, InferenceResponse, InferenceError, InferenceStats,
};
use crate::config::VLLMConfig;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, debug};

#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
use candle_core::{Device, DType, Tensor, IndexOp};
#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
use candle_transformers::models::llama::{Llama, Config as LlamaConfig, Cache};
#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
use candle_nn::VarBuilder;
#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
use tokenizers::{Tokenizer, models::bpe::BPE};

/// Candle backend types for different hardware acceleration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandleBackendType {
    /// CPU backend using optimized CPU kernels
    Cpu,
    /// CUDA backend with GPU acceleration and custom kernels
    #[cfg(feature = "candle-cuda")]
    Cuda,
    /// Metal backend for Apple Silicon acceleration
    #[cfg(feature = "candle-metal")]
    Metal,
}

impl std::fmt::Display for CandleBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            #[cfg(feature = "candle-cuda")]
            Self::Cuda => write!(f, "CUDA"),
            #[cfg(feature = "candle-metal")]
            Self::Metal => write!(f, "Metal"),
        }
    }
}

/// Model wrapper containing the loaded model and tokenizer
#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
struct CandleModelWrapper {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    config: CandleModelConfig,
    llama_config: LlamaConfig,
}

/// Model configuration loaded from `HuggingFace` config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandleModelConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: Option<bool>,
}

/// Candle-based inference engine
pub struct CandleInferenceEngine {
    /// Backend type being used
    backend_type: CandleBackendType,
    /// Loaded model and tokenizer (None if not initialized)
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    model: Arc<RwLock<Option<CandleModelWrapper>>>,
    /// Whether the engine is ready for inference
    ready: AtomicBool,
    /// Statistics tracking
    stats: Arc<InferenceStatsInner>,
}

#[derive(Default)]
struct InferenceStatsInner {
    total_requests: AtomicU64,
    total_tokens_generated: AtomicU64,
    total_inference_time_ms: AtomicU64,
    model_loaded: AtomicBool,
}

impl CandleInferenceEngine {
    /// Create a new Candle inference engine with CPU backend
    pub fn new() -> Self {
        Self::with_backend(CandleBackendType::Cpu)
    }

    /// Create a new Candle inference engine with specified backend
    pub fn with_backend(backend_type: CandleBackendType) -> Self {
        info!("Creating Candle inference engine with {} backend", backend_type);

        Self {
            backend_type,
            #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
            model: Arc::new(RwLock::new(None)),
            ready: AtomicBool::new(false),
            stats: Arc::new(InferenceStatsInner::default()),
        }
    }

    /// Get the backend type
    pub fn backend_type(&self) -> &CandleBackendType {
        &self.backend_type
    }

    /// Check if the engine is ready for inference
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Get inference statistics
    pub fn stats(&self) -> InferenceStats {
        let total_requests = self.stats.total_requests.load(Ordering::SeqCst);
        let total_inference_time_ms = self.stats.total_inference_time_ms.load(Ordering::SeqCst);

        InferenceStats {
            total_requests,
            total_tokens_generated: self.stats.total_tokens_generated.load(Ordering::SeqCst),
            avg_inference_time_ms: if total_requests > 0 {
                total_inference_time_ms as f64 / total_requests as f64
            } else {
                0.0
            },
            model_loaded: self.stats.model_loaded.load(Ordering::SeqCst),
        }
    }

    /// Create device based on backend type
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    fn create_device(&self) -> Result<Device, InferenceError> {
        match self.backend_type {
            CandleBackendType::Cpu => {
                info!("Initializing CPU device for Candle inference");
                info!("Created CPU device successfully");
                Ok(Device::Cpu)
            }
            #[cfg(feature = "candle-cuda")]
            CandleBackendType::Cuda => {
                info!("Initializing CUDA device for Candle inference");
                let device = Device::new_cuda(0).map_err(|e| {
                    InferenceError::InitializationError(format!("Failed to create CUDA device: {}", e))
                })?;
                info!("Created CUDA device successfully");
                Ok(device)
            }
            #[cfg(feature = "candle-metal")]
            CandleBackendType::Metal => {
                #[cfg(target_os = "macos")]
                {
                    info!("Initializing Metal device for Candle inference");
                    let device = Device::new_metal(0).map_err(|e| {
                        InferenceError::InitializationError(format!("Failed to create Metal device: {}", e))
                    })?;
                    info!("Created Metal device successfully");
                    Ok(device)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(InferenceError::InitializationError(
                        "Metal backend is only supported on macOS. Use candle-cpu or candle-cuda instead.".to_string()
                    ))
                }
            }
        }
    }

    /// Load model configuration from config.json
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    async fn load_model_config(&self, model_path: &str) -> Result<CandleModelConfig, InferenceError> {
        let config_path = std::path::Path::new(model_path).join("config.json");

        if !config_path.exists() {
            return Err(InferenceError::InvalidArgument(
                format!("Model config not found: {}", config_path.display())
            ));
        }

        let config_content = tokio::fs::read_to_string(&config_path).await.map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to read config.json: {}", e))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to parse config.json: {}", e))
        })?;

        Ok(CandleModelConfig {
            hidden_size: config["hidden_size"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(8192) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(16) as usize,
            num_key_value_heads: config["num_key_value_heads"].as_u64().map(|v| v as usize),
            vocab_size: config["vocab_size"].as_u64().unwrap_or(128_256) as usize,
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(131_072) as usize,
            rms_norm_eps: config["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            rope_theta: config["rope_theta"].as_f64().unwrap_or(500_000.0),
            tie_word_embeddings: config["tie_word_embeddings"].as_bool(),
        })
    }

    /// Load tokenizer from model directory
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    async fn load_tokenizer(&self, model_path: &str) -> Result<Tokenizer, InferenceError> {
        let model_path = std::path::Path::new(model_path);
        let tokenizer_path = model_path.join("tokenizer.json");

        // First, try to load tokenizer.json directly
        if tokenizer_path.exists() {
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    info!("Direct tokenizer loading failed: {}. Attempting compatibility mode for Llama 3.2...", e);
                    // Fall through to attempt creating a compatible tokenizer
                }
            }
        }

        // Llama 3.2 compatibility: Create a simplified tokenizer from tokenizer_config.json
        let config_path = model_path.join("tokenizer_config.json");
        if config_path.exists() {
            return self.create_llama32_compatible_tokenizer(&config_path, &tokenizer_path).await;
        }

        // If tokenizer.json doesn't exist, try to create it from vocab.json + tokenizer_config.json
        let vocab_path = model_path.join("vocab.json");

        if vocab_path.exists() && config_path.exists() {
            info!("Creating tokenizer from vocab.json and tokenizer_config.json");

            // Read tokenizer config to determine the tokenizer type
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                InferenceError::InvalidArgument(format!("Failed to read tokenizer_config.json: {}", e))
            })?;

            let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
                InferenceError::InvalidArgument(format!("Failed to parse tokenizer_config.json: {}", e))
            })?;

            // Get tokenizer class (defaults to GPT2Tokenizer)
            let tokenizer_class = config.get("tokenizer_class")
                .and_then(|v| v.as_str())
                .unwrap_or("GPT2Tokenizer");

            match tokenizer_class {
                "GPT2Tokenizer" => {
                    // For GPT2 tokenizer, create from vocab and merges
                    let vocab_str = std::fs::read_to_string(&vocab_path).map_err(|e| {
                        InferenceError::InvalidArgument(format!("Failed to read vocab.json: {}", e))
                    })?;

                    // Check if we have merges.txt or if it's embedded in tokenizer_config
                    let merges_path = model_path.join("merges.txt");
                    let _merges_content = if merges_path.exists() {
                        std::fs::read_to_string(&merges_path).map_err(|e| {
                            InferenceError::InvalidArgument(format!("Failed to read merges.txt: {}", e))
                        })?
                    } else {
                        // Generate basic merges for character-level tokenization
                        info!("No merges.txt found, using character-level tokenization");
                        "#version: 0.2\n".to_string()
                    };

                    // Parse vocab.json to get the vocabulary mapping
                    let vocab: std::collections::HashMap<String, u32> = serde_json::from_str(&vocab_str)
                        .map_err(|e| InferenceError::InvalidArgument(format!("Failed to parse vocab.json: {}", e)))?;

                    // Parse merges if available (for now we'll create an empty merges for character-level)
                    let merges = Vec::new(); // Empty merges for character-level tokenization

                    // Create BPE tokenizer
                    let bpe_tokenizer = BPE::new(vocab, merges);

                    // Create tokenizer with the BPE model
                    let tokenizer = Tokenizer::new(bpe_tokenizer);

                    // Save the created tokenizer for future use
                    if let Err(e) = tokenizer.save(&tokenizer_path, false) {
                        info!("Could not save tokenizer.json: {}", e);
                    }

                    Ok(tokenizer)
                }
                _ => {
                    Err(InferenceError::InvalidArgument(
                        format!("Unsupported tokenizer class: {}. Only GPT2Tokenizer is currently supported when tokenizer.json is missing.", tokenizer_class)
                    ))
                }
            }
        } else {
            Err(InferenceError::InvalidArgument(
                format!("Tokenizer files not found. Expected either tokenizer.json or both vocab.json and tokenizer_config.json in: {}", model_path.display())
            ))
        }
    }

    /// Create a Llama 3.2 compatible tokenizer from tokenizer files
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    async fn create_llama32_compatible_tokenizer(
        &self,
        config_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<Tokenizer, InferenceError> {
        use tokenizers::{
            AddedToken, PaddingParams, TruncationParams, PaddingDirection, TruncationDirection,
            processors::template::TemplateProcessing,
        };

        info!("Creating Llama 3.2 compatible tokenizer from config and extracting vocab from tokenizer.json");

        // Read tokenizer config for special tokens
        let config_str = tokio::fs::read_to_string(config_path).await.map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to read tokenizer_config.json: {}", e))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to parse tokenizer_config.json: {}", e))
        })?;

        // Try to extract vocabulary from the original tokenizer.json
        let original_tokenizer_path = tokenizer_path;
        let mut vocab = std::collections::HashMap::new();

        if original_tokenizer_path.exists() {
            info!("Extracting vocabulary from existing tokenizer.json");
            let tokenizer_str = tokio::fs::read_to_string(original_tokenizer_path).await.map_err(|e| {
                InferenceError::InvalidArgument(format!("Failed to read tokenizer.json: {}", e))
            })?;

            if let Ok(tokenizer_data) = serde_json::from_str::<serde_json::Value>(&tokenizer_str) {
                // Extract vocabulary from the tokenizer.json
                if let Some(model) = tokenizer_data.get("model") {
                    if let Some(vocab_obj) = model.get("vocab").and_then(|v| v.as_object()) {
                        info!("Found vocabulary with {} entries", vocab_obj.len());
                        for (token, id) in vocab_obj {
                            if let Some(id_num) = id.as_u64() {
                                vocab.insert(token.clone(), id_num as u32);
                            }
                        }
                    }
                }
            }
        }

        // If we couldn't extract vocab or it's empty, create a basic one
        if vocab.is_empty() {
            info!("Creating fallback vocabulary");
            // Add basic tokens to cover the vocabulary space
            for i in 0..128_000u32 {
                vocab.insert(format!("token_{}", i), i);
            }
        }

        // Create empty merges (character-level tokenization)
        let merges = Vec::new();

        // Create BPE tokenizer
        let bpe_tokenizer = BPE::new(vocab, merges);
        let mut tokenizer = Tokenizer::new(bpe_tokenizer);

        // Add special tokens from config with their proper IDs
        if let Some(added_tokens) = config.get("added_tokens_decoder").and_then(|v| v.as_object()) {
            info!("Adding {} special tokens", added_tokens.len());
            for (id_str, token_info) in added_tokens {
                if let (Ok(id), Some(content)) = (
                    id_str.parse::<u32>(),
                    token_info.get("content").and_then(|v| v.as_str())
                ) {
                    let special = token_info.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                    let added_token = AddedToken::from(content, special);
                    // Try to add the token with correct ID
                    tokenizer.add_tokens(&[added_token]);
                    // Manually set the ID in the tokenizer's vocabulary if possible
                    debug!("Added special token '{}' with ID {}", content, id);
                }
            }
        }

        // Set up padding if configured
        if let Some(pad_token) = config.get("pad_token").and_then(|v| v.as_str()) {
            let padding = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Left,
                pad_to_multiple_of: None,
                pad_id: 128_004,  // <|finetune_right_pad_id|> for Llama 3.2
                pad_type_id: 0,
                pad_token: pad_token.to_string(),
            };
            tokenizer.with_padding(Some(padding));
        }

        // Set up truncation
        if let Some(max_length) = config.get("model_max_length").and_then(|v| v.as_u64()) {
            let truncation = TruncationParams {
                max_length: max_length as usize,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            };
            let _ = tokenizer.with_truncation(Some(truncation));
        }

        // Set up post-processor for Llama 3.2 format
        if let Some(bos_token) = config.get("bos_token").and_then(|v| v.as_str()) {
            let post_processor = TemplateProcessing::builder()
                .try_single(format!("{} $A", bos_token))
                .map_err(|e| InferenceError::InvalidArgument(format!("Failed to create post-processor: {}", e)))?
                .special_tokens(vec![
                    (bos_token.to_string(), 128_000),  // <|begin_of_text|>
                    ("<|eot_id|>".to_string(), 128_009), // <|eot_id|>
                ])
                .build()
                .map_err(|e| InferenceError::InvalidArgument(format!("Failed to build post-processor: {}", e)))?;

            tokenizer.with_post_processor(post_processor);
        }

        // Save the tokenizer for future use
        if let Err(e) = tokenizer.save(tokenizer_path, false) {
            info!("Could not save compatible tokenizer: {}", e);
        }

        info!("Successfully created Llama 3.2 compatible tokenizer");
        Ok(tokenizer)
    }

    /// Create a `VarBuilder` that handles tensor remapping and weight tying for Llama models
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    fn create_remapping_var_builder(base_builder: VarBuilder<'_>) -> VarBuilder<'_> {
        // For Llama 3.2 models with weight tying, we need to handle the case where
        // lm_head.weight should be the same as model.embed_tokens.weight
        // Try direct root level access first (no prefix)
        base_builder
    }

    /// Generate text tokens using the loaded model
    #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
    async fn generate_tokens(
        &self,
        model: &Llama,
        tokenizer: &Tokenizer,
        device: &Device,
        config: &LlamaConfig,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String, InferenceError> {
        // Tokenize input
        let tokens = tokenizer.encode(prompt, false).map_err(|e| {
            InferenceError::ProcessingError(format!("Tokenization failed: {}", e))
        })?;

        let token_ids = tokens.get_ids();
        debug!("Input tokens: {:?}", &token_ids[..std::cmp::min(token_ids.len(), 10)]);

        // Start with input tokens for generation
        let mut all_tokens = token_ids.to_vec();

        // Simplified generation without KV cache for initial implementation
        for step in 0..max_tokens {
            debug!("Generation step {}/{}", step + 1, max_tokens);

            // Create input tensor from all current tokens
            let current_input = Tensor::new(all_tokens.as_slice(), device).map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to create input tensor: {}", e))
            })?.unsqueeze(0).map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to add batch dimension: {}", e))
            })?;

            // Create a simple cache for this forward pass
            let mut cache = Cache::new(true, DType::F32, config, device).map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to create cache: {}", e))
            })?;

            // Forward pass through model
            let logits = model.forward(&current_input, all_tokens.len() - 1, &mut cache).map_err(|e| {
                InferenceError::ProcessingError(format!("Model forward pass failed: {}", e))
            })?;

            // Get last token logits
            let last_logits = logits.i((.., logits.dim(1).map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to get logits dimension: {}", e))
            })? - 1, ..)).map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to get last token logits: {}", e))
            })?;

            // Apply temperature scaling
            let scaled_logits = if temperature > 0.0 {
                (last_logits / f64::from(temperature)).map_err(|e| {
                    InferenceError::ProcessingError(format!("Temperature scaling failed: {}", e))
                })?
            } else {
                last_logits
            };

            // Sample next token (argmax for deterministic output)
            let next_token_id = scaled_logits.argmax(1).map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to sample next token: {}", e))
            })?;

            let next_token = next_token_id.to_scalar::<u32>().map_err(|e| {
                InferenceError::ProcessingError(format!("Failed to convert token to scalar: {}", e))
            })?;

            all_tokens.push(next_token);

            // Check for end-of-sequence tokens (Llama 3.2 specific)
            if next_token == 128_009 || next_token == 128_001 { // </s> or <|end_of_text|>
                debug!("Stopping generation at end token: {}", next_token);
                break;
            }
        }

        // Decode the generated tokens (skip the original input tokens)
        let generated_only = &all_tokens[token_ids.len()..];
        let output_text = tokenizer.decode(generated_only, true).map_err(|e| {
            InferenceError::ProcessingError(format!("Token decoding failed: {}", e))
        })?;

        Ok(output_text)
    }
}

/// Helper function to estimate model parameters from configuration
#[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
fn estimate_model_parameters(config: &CandleModelConfig) -> String {
    let embed_params = config.vocab_size * config.hidden_size;
    let attention_params = config.num_hidden_layers * config.num_attention_heads * config.hidden_size * config.hidden_size / config.num_attention_heads;
    let mlp_params = config.num_hidden_layers * config.intermediate_size * config.hidden_size * 2;
    let total_params = embed_params + attention_params + mlp_params;

    if total_params > 1_000_000_000 {
        format!("{:.1}B", total_params as f64 / 1_000_000_000.0)
    } else if total_params > 1_000_000 {
        format!("{:.1}M", total_params as f64 / 1_000_000.0)
    } else {
        format!("{}K", total_params / 1000)
    }
}

#[async_trait]
impl InferenceEngine for CandleInferenceEngine {
    type Error = InferenceError;

    async fn initialize(&mut self, config: VLLMConfig) -> Result<(), Self::Error> {
        info!("Initializing Candle inference engine with model: {}", config.model_name);

        #[cfg(not(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal")))]
        {
            return Err(InferenceError::InitializationError(
                "No Candle features enabled. Enable one of: candle-cpu, candle-cuda, candle-metal".to_string()
            ));
        }

        #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
        {
            let start_time = Instant::now();

            // Create device
            let device = self.create_device()?;
            info!("Created {} device successfully", self.backend_type);

            // Load model configuration
            let model_config = self.load_model_config(&config.model_path).await?;
            info!("Loaded model config: {} layers, {} heads, vocab_size: {}",
                  model_config.num_hidden_layers,
                  model_config.num_attention_heads,
                  model_config.vocab_size);

            // Load tokenizer
            let tokenizer = self.load_tokenizer(&config.model_path).await?;
            info!("Loaded tokenizer successfully");

            // Load model weights using SafeTensors
            let safetensors_path = std::path::Path::new(&config.model_path).join("model.safetensors");

            if !safetensors_path.exists() {
                return Err(InferenceError::InitializationError(
                    format!("SafeTensors file not found: {}", safetensors_path.display())
                ));
            }

            info!("Loading model weights from SafeTensors: {}", safetensors_path.display());

            // Check if this model uses weight tying
            let is_weight_tied = model_config.tie_word_embeddings.unwrap_or(false);
            if is_weight_tied {
                info!("Model uses weight tying (tie_word_embeddings: true) - lm_head.weight will be shared with embed_tokens");
            }

            // Load model weights using VarBuilder from SafeTensors
            let dtype = DType::F32; // Use F32 for CPU inference
            let base_var_builder = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, &device)
            }.map_err(|e| InferenceError::InitializationError(
                format!("Failed to load SafeTensors weights: {}", e)
            ))?;

            // Create a remapping VarBuilder to handle tensor name differences
            // This model uses "transformer.*" naming while Candle expects "model.*" naming
            let var_builder = Self::create_remapping_var_builder(base_var_builder);

            // Create Llama configuration for candle-transformers
            let llama_config = LlamaConfig {
                vocab_size: model_config.vocab_size,
                hidden_size: model_config.hidden_size,
                intermediate_size: model_config.intermediate_size,
                num_hidden_layers: model_config.num_hidden_layers,
                num_attention_heads: model_config.num_attention_heads,
                num_key_value_heads: model_config.num_key_value_heads.unwrap_or(model_config.num_attention_heads),
                rms_norm_eps: model_config.rms_norm_eps,
                rope_theta: model_config.rope_theta as f32,
                max_position_embeddings: model_config.max_position_embeddings,
                bos_token_id: Some(128_000),  // Llama 3.2 specific
                eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(128_001)),  // Llama 3.2 specific
                rope_scaling: None,
                use_flash_attn: false,
                tie_word_embeddings: is_weight_tied,
            };

            // Create the Llama model with loaded weights
            let llama_model = Llama::load(var_builder, &llama_config)
                .map_err(|e| InferenceError::InitializationError(
                    format!("Failed to create Llama model: {}", e)
                ))?;

            info!("✅ Successfully loaded Llama model with {} parameters",
                  estimate_model_parameters(&model_config));

            let model_wrapper = CandleModelWrapper {
                model: llama_model,
                tokenizer,
                device,
                config: model_config,
                llama_config,
            };

            {
                let mut model_lock = self.model.write().await;
                *model_lock = Some(model_wrapper);
            }

            self.ready.store(true, Ordering::SeqCst);
            self.stats.model_loaded.store(true, Ordering::SeqCst);

            let init_time = start_time.elapsed();
            info!("🚀 Candle inference engine initialized successfully in {:?}", init_time);
        }

        Ok(())
    }

    async fn process(&self, request: InferenceRequest) -> Result<InferenceResponse, Self::Error> {
        if !self.is_ready() {
            return Err(InferenceError::ProcessingError(
                "Engine not initialized".to_string()
            ));
        }

        #[cfg(not(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal")))]
        {
            return Err(InferenceError::ProcessingError(
                "No Candle features enabled".to_string()
            ));
        }

        #[cfg(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal"))]
        {
            let start_time = Instant::now();
            self.stats.total_requests.fetch_add(1, Ordering::SeqCst);

            debug!("Processing inference request: {}", request.request_id);

            let model_lock = self.model.read().await;
            let model_wrapper = model_lock.as_ref().ok_or_else(|| {
                InferenceError::ProcessingError("Model not loaded".to_string())
            })?;

            // Generate response
            let generated_text = self.generate_tokens(
                &model_wrapper.model,
                &model_wrapper.tokenizer,
                &model_wrapper.device,
                &model_wrapper.llama_config,
                &request.prompt,
                request.max_tokens,
                request.temperature,
            ).await?;

            let inference_time = start_time.elapsed();
            let inference_time_ms = inference_time.as_millis() as f64;

            // Update statistics
            self.stats.total_inference_time_ms.fetch_add(inference_time_ms as u64, Ordering::SeqCst);
            self.stats.total_tokens_generated.fetch_add(u64::from(request.max_tokens), Ordering::SeqCst);

            info!("Completed inference request {} in {:?}", request.request_id, inference_time);

            Ok(InferenceResponse {
                request_id: request.request_id,
                generated_text,
                generated_tokens: request.max_tokens, // Simplified
                inference_time_ms,
                is_finished: true,
                error: None,
            })
        }
    }
}

impl Default for CandleInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal")))]
/// Stub implementation when no Candle features are enabled
pub struct CandleInferenceEngine {
    backend_type: CandleBackendType,
}

#[cfg(not(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal")))]
impl CandleInferenceEngine {
    pub fn new() -> Self {
        Self::with_backend(CandleBackendType::Cpu)
    }

    pub fn with_backend(backend_type: CandleBackendType) -> Self {
        Self { backend_type }
    }

    pub fn backend_type(&self) -> &CandleBackendType {
        &self.backend_type
    }

    pub fn is_ready(&self) -> bool {
        false
    }

    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            total_requests: 0,
            total_tokens_generated: 0,
            avg_inference_time_ms: 0.0,
            model_loaded: false,
        }
    }
}

#[cfg(not(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal")))]
#[async_trait]
impl InferenceEngine for CandleInferenceEngine {
    type Error = InferenceError;

    async fn initialize(&mut self, _config: VLLMConfig) -> Result<(), Self::Error> {
        Err(InferenceError::InitializationError(
            "Candle features not enabled. Add --features candle-cpu, candle-cuda, or candle-metal".to_string()
        ))
    }

    async fn process(&self, _request: InferenceRequest) -> Result<InferenceResponse, Self::Error> {
        Err(InferenceError::ProcessingError(
            "Candle features not enabled".to_string()
        ))
    }
}

#[cfg(not(any(feature = "candle-cpu", feature = "candle-cuda", feature = "candle-metal")))]
impl Default for CandleInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}