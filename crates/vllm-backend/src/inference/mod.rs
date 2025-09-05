//! CPU-based inference engine implementation
//!
//! This module implements a basic CPU inference pipeline using Candle for tensor operations
//! and Hugging Face models. It provides deterministic inference for mathematical queries.

use crate::config::VLLMConfig;
use crate::error::VLLMResult;

#[cfg(not(feature = "cpu-only"))]
use crate::error::VLLMError;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[cfg(feature = "cpu-only")]
mod cpu_engine;

#[cfg(feature = "lmrs")]
mod llama_engine;

#[cfg(feature = "cpu-only")]
pub use cpu_engine::*;

#[cfg(feature = "lmrs")]
pub use llama_engine::*;

/// Request for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique identifier for this request
    pub request_id: u64,
    /// Input prompt text
    pub prompt: String,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Optional seed for reproducible results
    pub seed: Option<u64>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            request_id: 0,
            prompt: String::new(),
            max_tokens: 50,
            temperature: 0.0, // Deterministic by default
            top_p: 1.0,
            seed: Some(42), // Fixed seed for reproducibility
        }
    }
}

/// Response from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID this response corresponds to
    pub request_id: u64,
    /// Generated text
    pub generated_text: String,
    /// Number of tokens generated
    pub generated_tokens: usize,
    /// Time taken for inference in milliseconds
    pub inference_time_ms: f64,
    /// Whether generation is finished
    pub is_finished: bool,
    /// Error information if any
    pub error: Option<String>,
}

impl Default for InferenceResponse {
    fn default() -> Self {
        Self {
            request_id: 0,
            generated_text: String::new(),
            generated_tokens: 0,
            inference_time_ms: 0.0,
            is_finished: true,
            error: None,
        }
    }
}

/// Main inference engine trait
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Initialize the engine with a model
    async fn initialize(&mut self, config: &VLLMConfig, models_dir: &str) -> VLLMResult<()>;

    /// Check if the engine is ready for inference
    fn is_ready(&self) -> bool;

    /// Process a single inference request
    async fn infer(&self, request: InferenceRequest) -> VLLMResult<InferenceResponse>;

    /// Get engine statistics
    async fn get_stats(&self) -> EngineStats;

    /// Shutdown the engine
    async fn shutdown(&mut self) -> VLLMResult<()>;
}

/// Engine performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Current memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Whether model is loaded
    pub model_loaded: bool,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            avg_inference_time_ms: 0.0,
            memory_usage_bytes: 0,
            model_loaded: false,
        }
    }
}

/// Create a new inference engine based on configuration
pub async fn create_engine(config: &VLLMConfig) -> VLLMResult<Arc<RwLock<dyn InferenceEngine>>> {
    // Prefer Llama 3.2 1B engine when available
    #[cfg(feature = "lmrs")]
    {
        if config.model_path.contains("llama") || config.model_path.contains("3.2") {
            info!("Creating Llama 3.2 1B inference engine");
            let mut engine = LlamaInferenceEngine::new();
            engine.initialize(config, "./models").await?;
            return Ok(Arc::new(RwLock::new(engine)));
        }
    }

    // Fallback to pattern matching engine
    #[cfg(feature = "cpu-only")]
    {
        info!("Creating CPU-based inference engine (pattern matching)");
        let mut engine = CpuInferenceEngine::new();
        engine.initialize(config, "./models").await?;
        Ok(Arc::new(RwLock::new(engine)))
    }

    #[cfg(not(feature = "cpu-only"))]
    {
        Err(VLLMError::InvalidArgument(
            "No inference engine available. Enable cpu-only or cuda feature".to_string(),
        ))
    }
}

/// Utility function to create a deterministic math test request
pub fn create_math_test_request() -> InferenceRequest {
    InferenceRequest {
        request_id: 1,
        prompt: "What is 2+2? Answer with number only:".to_string(),
        max_tokens: 5,
        temperature: 0.0, // Deterministic
        top_p: 1.0,
        seed: Some(42),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let request = create_math_test_request();
        assert_eq!(request.prompt, "What is 2+2? Answer with number only:");
        assert!((request.temperature - 0.0).abs() < f32::EPSILON);
        assert!(request.seed.is_some());
    }

    #[test]
    fn test_deterministic_defaults() {
        let request = InferenceRequest::default();
        assert!((request.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(request.seed, Some(42));
    }
}
