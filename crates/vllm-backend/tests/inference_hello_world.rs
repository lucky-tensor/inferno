//! Hello World inference tests for CPU pipeline
//!
//! These tests validate the basic CPU inference functionality with deterministic
//! mathematical queries like "2+2=4". This serves as the foundation for our
//! progressive testing strategy before adding more complex features.

use inferno_vllm_backend::{
    create_engine, create_math_test_request, CpuInferenceEngine, InferenceEngine, InferenceRequest,
    VLLMConfig,
};
use std::time::Duration;
use tokio::time::timeout;

/// Test the basic CPU inference engine creation and initialization
#[tokio::test]
async fn test_cpu_engine_creation() {
    let mut engine = CpuInferenceEngine::new();
    assert!(!engine.is_ready(), "Engine should not be ready initially");

    let config = VLLMConfig {
        model_path: "test-math-model".to_string(),
        model_name: "test-model".to_string(),
        device_id: -1, // CPU-only
        ..Default::default()
    };

    let result = engine.initialize(&config, "./models").await;
    assert!(result.is_ok(), "Engine initialization should succeed");
    assert!(
        engine.is_ready(),
        "Engine should be ready after initialization"
    );
}

/// Test deterministic "2+2=4" response - our core hello world test
#[tokio::test]
async fn test_hello_world_math_inference() {
    let config = VLLMConfig {
        model_path: "pattern-matching-model".to_string(),
        model_name: "math-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let request = create_math_test_request();
    assert_eq!(
        request.prompt, "What is 2+2? Answer with number only:",
        "Test request should have correct prompt"
    );

    let response = engine_guard.infer(request).await;
    assert!(response.is_ok(), "Inference should succeed");

    let response = response.unwrap();
    assert_eq!(response.generated_text, "4", "Should respond with '4'");
    assert!(
        response.is_finished,
        "Response should be marked as finished"
    );
    assert!(response.error.is_none(), "Should not have any errors");
    assert!(
        response.inference_time_ms >= 0.0,
        "Should have non-negative inference time (got: {:.3}ms)",
        response.inference_time_ms
    );
}

/// Test deterministic behavior - same input should produce same output
#[tokio::test]
async fn test_deterministic_responses() {
    let config = VLLMConfig {
        model_path: "deterministic-test-model".to_string(),
        model_name: "consistency-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let request = InferenceRequest {
        request_id: 1,
        prompt: "2+2".to_string(),
        max_tokens: 5,
        temperature: 0.0, // Deterministic
        top_p: 1.0,
        seed: Some(42),
    };

    // Run the same inference multiple times
    let response1 = engine_guard.infer(request.clone()).await.unwrap();
    let response2 = engine_guard.infer(request.clone()).await.unwrap();
    let response3 = engine_guard.infer(request).await.unwrap();

    // All responses should be identical
    assert_eq!(response1.generated_text, response2.generated_text);
    assert_eq!(response2.generated_text, response3.generated_text);
    assert_eq!(response1.generated_text, "4");
}

/// Test various mathematical expressions
#[tokio::test]
async fn test_mathematical_expressions() {
    let config = VLLMConfig {
        model_path: "math-expressions-model".to_string(),
        model_name: "math-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let test_cases = vec![
        ("What is 2+2?", "4"),
        ("2+2", "4"),
        ("What is 1+1?", "2"),
        ("1+1", "2"),
        ("What is 3+3?", "6"),
        ("3+3", "6"),
    ];

    for (prompt, expected) in test_cases {
        let request = InferenceRequest {
            request_id: 1,
            prompt: prompt.to_string(),
            max_tokens: 5,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        };

        let response = engine_guard.infer(request).await.unwrap();
        assert_eq!(
            response.generated_text, expected,
            "For prompt '{}', expected '{}' but got '{}'",
            prompt, expected, response.generated_text
        );
    }
}

/// Test engine performance and response times
#[tokio::test]
async fn test_inference_performance() {
    let config = VLLMConfig {
        model_path: "performance-test-model".to_string(),
        model_name: "perf-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let request = create_math_test_request();

    // Test that inference completes within reasonable time (5 seconds)
    let result = timeout(Duration::from_secs(5), engine_guard.infer(request)).await;

    assert!(result.is_ok(), "Inference should complete within 5 seconds");

    let response = result.unwrap().unwrap();
    assert!(
        response.inference_time_ms < 1000.0,
        "Inference should complete in under 1 second, got {:.2}ms",
        response.inference_time_ms
    );
}

/// Test engine statistics and monitoring
#[tokio::test]
async fn test_engine_statistics() {
    let config = VLLMConfig {
        model_path: "stats-test-model".to_string(),
        model_name: "stats-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let stats = engine_guard.get_stats().await;

    assert!(stats.model_loaded, "Model should be marked as loaded");
    assert!(
        stats.memory_usage_bytes > 0,
        "Should report some memory usage"
    );
    assert_eq!(stats.total_requests, 0, "Should start with zero requests");
}

/// Test error handling for invalid configurations
#[tokio::test]
async fn test_error_handling() {
    // Test empty model path
    let mut engine = CpuInferenceEngine::new();
    let invalid_config = VLLMConfig {
        model_path: "".to_string(), // Empty path should fail
        ..Default::default()
    };

    let result = engine.initialize(&invalid_config, "./models").await;
    assert!(result.is_err(), "Should fail with empty model path");

    // Test inference on uninitialized engine
    let request = create_math_test_request();
    let result = engine.infer(request).await;
    assert!(result.is_err(), "Should fail on uninitialized engine");
}

/// Integration test with realistic workflow
#[tokio::test]
async fn test_complete_workflow() {
    // 1. Create and initialize engine
    let config = VLLMConfig {
        model_path: "workflow-test-model".to_string(),
        model_name: "workflow".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Engine creation failed");

    // 2. Verify engine is ready
    {
        let engine_guard = engine.read().await;
        assert!(engine_guard.is_ready(), "Engine should be ready");
    }

    // 3. Perform inference
    let request = InferenceRequest {
        request_id: 42,
        prompt: "Hello, what is 2+2?".to_string(),
        max_tokens: 10,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(12345),
    };

    let response = {
        let engine_guard = engine.read().await;
        engine_guard.infer(request).await.expect("Inference failed")
    };

    // 4. Validate response
    assert_eq!(response.request_id, 42);
    assert_eq!(response.generated_text, "4");
    assert!(response.is_finished);
    assert!(response.error.is_none());

    // 5. Shutdown engine
    {
        let mut engine_guard = engine.write().await;
        let shutdown_result = engine_guard.shutdown().await;
        assert!(shutdown_result.is_ok(), "Shutdown should succeed");
        assert!(
            !engine_guard.is_ready(),
            "Engine should not be ready after shutdown"
        );
    }
}
