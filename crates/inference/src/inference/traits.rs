//! Common inference traits and types for llama-burn engine

use crate::config::InfernoConfig;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Inference engine errors
#[derive(Error, Debug, Clone)]
pub enum InferenceError {
    /// Engine initialization failed
    #[error("Initialization error: {0}")]
    InitializationError(String),
    /// Processing request failed
    #[error("Processing error: {0}")]
    ProcessingError(String),
    /// Invalid argument provided
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    /// Model file not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    /// IO operation failed
    #[error("IO error: {0}")]
    IoError(String),
}

/// Request for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique identifier for this request
    pub request_id: u64,
    /// Input prompt text
    pub prompt: String,
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
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
    pub generated_tokens: u32,
    /// Time taken for inference in milliseconds
    pub inference_time_ms: f64,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: Option<f64>,
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
            time_to_first_token_ms: None,
            is_finished: true,
            error: None,
        }
    }
}

/// Engine performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Whether model is loaded
    pub model_loaded: bool,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_tokens_generated: 0,
            avg_inference_time_ms: 0.0,
            model_loaded: false,
        }
    }
}

/// Common trait for inference engines
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Engine-specific error type
    type Error: std::error::Error + Send + Sync + 'static;

    /// Initialize the engine with a model configuration
    async fn initialize(&mut self, config: InfernoConfig) -> Result<(), Self::Error>;

    /// Process a single inference request
    async fn process(&self, request: InferenceRequest) -> Result<InferenceResponse, Self::Error>;
}

/// Utility function to create a test inference request
pub fn create_test_request(prompt: &str) -> InferenceRequest {
    InferenceRequest {
        request_id: 1,
        prompt: prompt.to_string(),
        max_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(42),
    }
}

/// Utility function to create a math test request
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
    fn test_inference_request_creation() {
        let request = create_math_test_request();
        assert_eq!(request.prompt, "What is 2+2? Answer with number only:");
        assert!((request.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(request.seed, Some(42));
    }

    #[test]
    fn test_inference_request_defaults() {
        let request = InferenceRequest::default();
        assert!((request.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.max_tokens, 50);
    }

    #[test]
    fn test_inference_response_defaults() {
        let response = InferenceResponse::default();
        assert_eq!(response.request_id, 0);
        assert!(response.is_finished);
        assert!(response.error.is_none());
    }
}