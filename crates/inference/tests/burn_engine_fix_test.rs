//! Test to verify the BurnInferenceEngine fix without requiring real models

use inferno_inference::{
    config::InfernoConfig,
    inference::{InferenceRequest, BurnInferenceEngine},
};

#[tokio::test]
async fn test_burn_engine_response_fix() {
    println!("🧪 Testing BurnInferenceEngine response handling...");

    // Create engine without initializing (to avoid needing real model)
    let engine = BurnInferenceEngine::new();

    let request = InferenceRequest {
        request_id: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(42),
    };

    // This should fail with EngineNotInitialized, not hang
    let result = engine.process_sync(request);

    match result {
        Err(inferno_inference::error::InfernoError::EngineNotInitialized) => {
            println!("✅ Engine correctly returns EngineNotInitialized");
        }
        Ok(_) => {
            panic!("❌ Engine should not succeed without initialization");
        }
        Err(e) => {
            panic!("❌ Unexpected error: {}", e);
        }
    }

    println!("✅ BurnInferenceEngine response fix test passed");
}