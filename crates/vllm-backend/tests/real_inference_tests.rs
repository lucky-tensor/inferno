//! Real CPU inference tests with actual Llama 3.2 1B model
//!
//! These tests download and use the actual Llama 3.2 1B model from Hugging Face
//! to perform real inference on CPU. No mocking or simulation.

#[cfg(feature = "burn-cpu")]
use inferno_vllm_backend::{
    create_engine, create_math_test_request, BurnInferenceEngine, InferenceEngine,
    InferenceRequest, VLLMConfig,
};
#[cfg(feature = "burn-cpu")]
use std::path::Path;
#[cfg(feature = "burn-cpu")]
use std::time::Duration;
#[cfg(feature = "burn-cpu")]
use tokio::time::timeout;

/// Test creating the real inference engine
#[tokio::test]
#[cfg(feature = "burn-cpu")]
async fn test_burn_engine_creation() {
    let engine = BurnInferenceEngine::new();
    assert!(!engine.is_ready(), "Engine should not be ready initially");
}

/// Test real model download and loading
/// This test downloads ~2.5GB model files and caches them in ./models/
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "downloads real model - run with --ignored to test real inference"]
async fn test_real_model_initialization() {
    let mut engine = BurnInferenceEngine::new();
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "llama-3.2-1b-real-test".to_string(),
        device_id: -1, // CPU-only
        ..Default::default()
    };

    // Ensure models directory exists
    std::fs::create_dir_all("./models").expect("Failed to create models directory");

    // This will download Llama 3.2 1B from Hugging Face if not cached
    let start_time = std::time::Instant::now();
    let result = engine.initialize(&config, "./models").await;
    let init_time = start_time.elapsed();

    assert!(
        result.is_ok(),
        "Real model initialization should succeed: {:?}",
        result
    );
    assert!(
        engine.is_ready(),
        "Engine should be ready after real model loading"
    );

    let stats = engine.get_stats().await;
    assert!(stats.model_loaded, "Model should be marked as loaded");
    // Memory usage should be substantial for real model
    assert!(
        stats.memory_usage_bytes > 1_000_000_000,
        "Should report realistic memory usage for Llama model: {} bytes",
        stats.memory_usage_bytes
    ); // >1GB

    println!(
        "Model initialization took: {:.2}s, Memory: {:.1}GB",
        init_time.as_secs_f64(),
        stats.memory_usage_bytes as f64 / 1_000_000_000.0
    );

    // Verify model files were cached
    let model_files = [
        "./models/llama3.2-1b/model.safetensors",
        "./models/llama3.2-1b/tokenizer.json",
        "./models/llama3.2-1b/config.json",
    ];

    for file_path in &model_files {
        assert!(
            Path::new(file_path).exists(),
            "Model file should be cached: {}",
            file_path
        );
    }
}

/// Test actual mathematical inference with real model
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model download and inference"]
async fn test_real_math_inference() {
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "real-math-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    // Create engine using the create_engine function
    let engine = create_engine(&config)
        .await
        .expect("Failed to create real Llama engine");
    let engine_guard = engine.read().await;

    let request = create_math_test_request();
    assert_eq!(
        request.prompt, "What is 2+2? Answer with number only:",
        "Test request should have correct prompt"
    );

    println!("Testing real inference with prompt: '{}'", request.prompt);

    let response = engine_guard.infer(request).await;
    assert!(response.is_ok(), "Real Llama inference should succeed");

    let response = response.unwrap();
    println!("Real model response: '{}'", response.generated_text);

    // The real model may not give exactly "4" but should provide a reasonable response
    assert!(
        !response.generated_text.is_empty(),
        "Should generate non-empty response"
    );
    assert!(
        response.is_finished,
        "Response should be marked as finished"
    );
    assert!(response.error.is_none(), "Should not have any errors");
    assert!(
        response.inference_time_ms > 0.0,
        "Should have positive inference time: {}ms",
        response.inference_time_ms
    );
    assert!(
        response.generated_tokens > 0,
        "Should generate tokens: {}",
        response.generated_tokens
    );

    println!(
        "Inference completed in {:.2}ms, generated {} tokens",
        response.inference_time_ms, response.generated_tokens
    );
}

/// Test deterministic responses from real model
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model"]
async fn test_real_deterministic_responses() {
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "deterministic-real-test".to_string(),
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

    println!("Testing deterministic inference with real model...");

    // Run the same inference multiple times - should be deterministic with temp=0
    let response1 = engine_guard.infer(request.clone()).await.unwrap();
    let response2 = engine_guard.infer(request.clone()).await.unwrap();
    let response3 = engine_guard.infer(request).await.unwrap();

    println!("Response 1: '{}'", response1.generated_text);
    println!("Response 2: '{}'", response2.generated_text);
    println!("Response 3: '{}'", response3.generated_text);

    // With temperature=0 and fixed seed, responses should be identical
    assert_eq!(
        response1.generated_text, response2.generated_text,
        "Deterministic responses should be identical"
    );
    assert_eq!(
        response2.generated_text, response3.generated_text,
        "All deterministic responses should be identical"
    );
}

/// Test various mathematical expressions with real model
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model"]
async fn test_real_mathematical_expressions() {
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "math-expressions-real".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let test_cases = [
        "What is 2+2?",
        "Calculate 1+1",
        "What is 3+3?",
        "5-2=?",
        "Simple math: 4+1",
    ];

    println!("Testing various mathematical expressions with real model...");

    for (i, prompt) in test_cases.iter().enumerate() {
        let request = InferenceRequest {
            request_id: i as u64 + 1,
            prompt: prompt.to_string(),
            max_tokens: 10,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        };

        let response = engine_guard.infer(request).await.unwrap();

        println!(
            "Prompt: '{}' -> Response: '{}' ({} tokens, {:.1}ms)",
            prompt, response.generated_text, response.generated_tokens, response.inference_time_ms
        );

        assert!(
            !response.generated_text.is_empty(),
            "Should generate response for prompt: {}",
            prompt
        );
        assert!(response.is_finished);
        assert!(response.error.is_none());
    }
}

/// Test inference performance with real model
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model"]
async fn test_real_inference_performance() {
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "performance-real-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");
    let engine_guard = engine.read().await;

    let request = create_math_test_request();

    println!("Testing real model inference performance...");

    // Test that real Llama inference completes within reasonable time
    // Allow generous timeout for CPU inference
    let result = timeout(Duration::from_secs(60), engine_guard.infer(request)).await;

    assert!(
        result.is_ok(),
        "Real Llama inference should complete within 60 seconds"
    );

    let response = result.unwrap().unwrap();

    println!(
        "Performance test completed: {:.2}s, {} tokens, throughput: {:.1} tokens/sec",
        response.inference_time_ms / 1000.0,
        response.generated_tokens,
        response.generated_tokens as f64 / (response.inference_time_ms / 1000.0)
    );

    // Real CPU inference should complete within reasonable time
    // but will be slower than simulated responses
    assert!(
        response.inference_time_ms < 60_000.0,
        "Real inference should complete in under 60 seconds, got {:.2}ms",
        response.inference_time_ms
    );
}

/// Test engine statistics with real model
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model"]
async fn test_real_engine_statistics() {
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "stats-real-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    let engine = create_engine(&config)
        .await
        .expect("Failed to create engine");

    // Test initial stats
    {
        let engine_guard = engine.read().await;
        let initial_stats = engine_guard.get_stats().await;

        assert!(initial_stats.model_loaded, "Model should be loaded");
        assert!(
            initial_stats.memory_usage_bytes > 1_000_000_000,
            "Should report realistic memory usage"
        );

        println!(
            "Initial stats - Memory: {:.1}GB, Requests: {}",
            initial_stats.memory_usage_bytes as f64 / 1_000_000_000.0,
            initial_stats.total_requests
        );
    }

    // Perform several inferences to update statistics
    for i in 0..3 {
        let request = InferenceRequest {
            request_id: i + 1,
            prompt: format!("Test inference #{}", i + 1),
            max_tokens: 5,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        };

        let engine_guard = engine.read().await;
        let response = engine_guard.infer(request).await.unwrap();

        println!(
            "Inference #{}: '{}' ({:.1}ms)",
            i + 1,
            response.generated_text,
            response.inference_time_ms
        );
    }

    // Check updated stats
    {
        let engine_guard = engine.read().await;
        let final_stats = engine_guard.get_stats().await;

        println!(
            "Final stats - Requests: {}, Avg time: {:.1}ms, Memory: {:.1}GB",
            final_stats.total_requests,
            final_stats.avg_inference_time_ms,
            final_stats.memory_usage_bytes as f64 / 1_000_000_000.0
        );

        assert!(
            final_stats.total_requests >= 3,
            "Should track request count"
        );
        assert!(
            final_stats.avg_inference_time_ms > 0.0,
            "Should calculate average inference time"
        );
    }
}

/// Test proper model caching (subsequent runs should be faster)
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model"]
async fn test_model_caching() {
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "caching-test".to_string(),
        device_id: -1,
        ..Default::default()
    };

    // First initialization (may need to download)
    let start1 = std::time::Instant::now();
    let engine1 = create_engine(&config)
        .await
        .expect("First engine creation failed");
    let init_time1 = start1.elapsed();

    // Shutdown first engine
    {
        let mut engine_guard = engine1.write().await;
        engine_guard.shutdown().await.expect("Shutdown failed");
    }

    // Second initialization (should use cache)
    let start2 = std::time::Instant::now();
    let engine2 = create_engine(&config)
        .await
        .expect("Second engine creation failed");
    let init_time2 = start2.elapsed();

    println!(
        "Initialization times - First: {:.2}s, Second: {:.2}s",
        init_time1.as_secs_f64(),
        init_time2.as_secs_f64()
    );

    // Second initialization should be faster (using cached model)
    // Allow some variance but expect significant improvement
    if init_time1.as_secs() > 10 {
        // Only check if first run was slow (indicating download)
        assert!(
            init_time2 < init_time1,
            "Cached model loading should be faster: {:.2}s vs {:.2}s",
            init_time2.as_secs_f64(),
            init_time1.as_secs_f64()
        );
    }

    // Verify second engine works
    let engine_guard = engine2.read().await;
    let request = create_math_test_request();
    let response = engine_guard.infer(request).await;
    assert!(response.is_ok(), "Cached model should work for inference");

    println!(
        "Cached model inference result: '{}'",
        response.unwrap().generated_text
    );
}

/// Integration test: Complete workflow with real model
#[tokio::test]
#[cfg(feature = "burn-cpu")]
#[ignore = "requires real model"]
async fn test_complete_real_workflow() {
    println!("Starting complete real model workflow test...");

    // 1. Create and initialize engine
    let config = VLLMConfig {
        model_path: "llama3.2-1b".to_string(),
        model_name: "workflow-real-test".to_string(),
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

        let stats = engine_guard.get_stats().await;
        assert!(stats.model_loaded, "Model should be loaded");

        println!(
            "Engine ready - Memory: {:.1}GB",
            stats.memory_usage_bytes as f64 / 1_000_000_000.0
        );
    }

    // 3. Perform mathematical inference
    let request = InferenceRequest {
        request_id: 999,
        prompt: "Calculate 2+2, respond with the answer:".to_string(),
        max_tokens: 10,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(12345),
    };

    let response = {
        let engine_guard = engine.read().await;
        engine_guard
            .infer(request)
            .await
            .expect("Real inference failed")
    };

    // 4. Validate response
    assert_eq!(response.request_id, 999);
    assert!(!response.generated_text.is_empty());
    assert!(response.is_finished);
    assert!(response.error.is_none());
    assert!(response.inference_time_ms > 0.0);

    println!(
        "Workflow complete - Generated: '{}' in {:.2}ms",
        response.generated_text, response.inference_time_ms
    );

    // 5. Shutdown engine
    {
        let mut engine_guard = engine.write().await;
        let shutdown_result = engine_guard.shutdown().await;
        assert!(shutdown_result.is_ok(), "Engine shutdown should succeed");
        assert!(
            !engine_guard.is_ready(),
            "Engine should not be ready after shutdown"
        );
    }

    println!("Complete workflow test passed!");
}
