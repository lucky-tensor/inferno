//! Basic integration tests for VLLM backend
//!
//! These tests verify core functionality without requiring CUDA.

use inferno_vllm_backend::{VLLMBackend, VLLMConfig, VLLMConfigBuilder, VLLMEngine, VLLMServer};
use std::env;

#[test]
fn test_config_creation_and_validation() {
    // Test default configuration fails validation (no model path)
    let config = VLLMConfig::default();
    assert!(
        config.validate().is_err(),
        "Default config should fail validation"
    );

    // Test builder pattern with validation
    let result = VLLMConfigBuilder::new()
        .model_path("/tmp/test_model")
        .model_name("test-model")
        .device_id(-1) // Use CPU-only mode
        .max_batch_size(4)
        .port(8081)
        .build();

    assert!(
        result.is_err(),
        "Should fail due to non-existent model path"
    );

    // Test server address generation
    let mut config = VLLMConfig::default();
    config.server.host = "localhost".to_string();
    config.server.port = 9000;
    assert_eq!(config.server_address(), "localhost:9000");
}

#[test]
fn test_config_from_env() {
    // Set some environment variables
    env::set_var("VLLM_MODEL_NAME", "test-env-model");
    env::set_var("VLLM_DEVICE_ID", "-1");
    env::set_var("VLLM_MAX_BATCH_SIZE", "16");
    env::set_var("VLLM_PORT", "8090");
    env::set_var("VLLM_HOST", "0.0.0.0");

    let result = VLLMConfig::from_env();
    // Will fail due to missing model path, but should have loaded other values
    assert!(result.is_err(), "Should fail due to missing model path");

    // Clean up
    env::remove_var("VLLM_MODEL_NAME");
    env::remove_var("VLLM_DEVICE_ID");
    env::remove_var("VLLM_MAX_BATCH_SIZE");
    env::remove_var("VLLM_PORT");
    env::remove_var("VLLM_HOST");
}

#[tokio::test]
async fn test_engine_lifecycle() {
    let config = VLLMConfig {
        model_path: "/tmp/fake_model".to_string(), // Non-existent is OK for this test
        device_id: -1,                             // CPU-only mode
        ..Default::default()
    };

    // Test engine creation
    let engine = VLLMEngine::new(&config);
    assert!(engine.is_ok(), "Engine creation should succeed");

    let engine = engine.unwrap();

    // Test initial state
    assert!(
        !engine.is_running().await,
        "Engine should not be running initially"
    );

    // Test start
    let result = engine.start().await;
    assert!(result.is_ok(), "Engine start should succeed");
    assert!(
        engine.is_running().await,
        "Engine should be running after start"
    );

    // Test stop
    let result = engine.stop().await;
    assert!(result.is_ok(), "Engine stop should succeed");
    assert!(
        !engine.is_running().await,
        "Engine should not be running after stop"
    );
}

#[tokio::test]
async fn test_backend_integration() {
    let config = VLLMConfig {
        model_path: "/tmp/fake_model".to_string(),
        device_id: -1, // CPU-only mode
        ..Default::default()
    };

    // Test backend creation
    let backend = VLLMBackend::new(config);
    assert!(backend.is_ok(), "Backend creation should succeed");

    let backend = backend.unwrap();

    // Test backend start/stop
    assert!(
        backend.start().await.is_ok(),
        "Backend start should succeed"
    );
    assert!(backend.stop().await.is_ok(), "Backend stop should succeed");
}

#[test]
fn test_server_configuration() {
    let mut config = VLLMConfig::default();
    config.server.host = "localhost".to_string();
    config.server.port = 8080;
    config.server.enable_cors = true;
    config.server.enable_metrics = true;

    let server = VLLMServer::new(config.clone());
    assert_eq!(server.config().host, "localhost");
    assert_eq!(server.config().port, 8080);
    assert!(server.config().enable_cors);
    assert!(server.config().enable_metrics);
}

#[test]
fn test_memory_allocator_interface() {
    use inferno_vllm_backend::{CudaMemoryPool, MemoryTracker};

    // Test memory pool creation
    let pool = CudaMemoryPool::new(-1); // CPU-only device
    assert!(pool.is_ok(), "Memory pool creation should succeed");

    let pool = pool.unwrap();
    assert_eq!(pool.device_id(), -1, "Device ID should match");

    // Test memory tracker
    let tracker = MemoryTracker::new(-1);
    let stats = tracker.get_stats();
    assert_eq!(stats.device_id, -1, "Tracker device ID should match");
}

#[test]
fn test_configuration_serialization() {
    let config = VLLMConfig {
        model_path: "/tmp/test".to_string(),
        model_name: "test-model".to_string(),
        ..Default::default()
    };

    // Test JSON serialization
    let json_str = serde_json::to_string(&config);
    assert!(json_str.is_ok(), "JSON serialization should work");

    let deserialized: Result<VLLMConfig, _> = serde_json::from_str(&json_str.unwrap());
    assert!(deserialized.is_ok(), "JSON deserialization should work");

    // Test TOML serialization
    let toml_str = toml::to_string(&config);
    assert!(toml_str.is_ok(), "TOML serialization should work");

    let deserialized: Result<VLLMConfig, _> = toml::from_str(&toml_str.unwrap());
    assert!(deserialized.is_ok(), "TOML deserialization should work");
}
