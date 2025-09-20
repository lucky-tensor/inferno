//! Llama-burn inference engine implementation
//!
//! This module implements inference using the Burn framework with llama-burn models.
//! This is the only supported inference backend.

pub mod traits;
pub use traits::*;

pub mod burn_engine;
pub use burn_engine::BurnInferenceEngine;

/// Create the llama-burn inference engine
pub fn create_engine() -> Box<dyn InferenceEngine<Error = InferenceError>> {
    Box::new(BurnInferenceEngine::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let _engine = create_engine();
    }

    #[test]
    fn test_inference_types() {
        let request = create_test_request("Hello, world!");
        assert_eq!(request.prompt, "Hello, world!");
        assert_eq!(request.max_tokens, 50);

        let math_request = create_math_test_request();
        assert_eq!(math_request.prompt, "What is 2+2? Answer with number only:");
        assert!((math_request.temperature - 0.0).abs() < f32::EPSILON);
    }
}
