//! Inference Tests
//!
//! Tests for the AI inference engine functionality.

use inferno_backend::inference::InferenceEngine;

#[test]
fn test_inference_engine_creation() {
    let engine = InferenceEngine::new("path/to/model.bin".to_string()).unwrap();
    // Engine is ready for inference
    assert!(engine.model_path().contains("model.bin"));
}
