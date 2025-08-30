//! Configuration Tests
//!
//! Tests for backend configuration loading and validation.

use inferno_backend::{BackendCliOptions, BackendConfig};
use inferno_shared::{HealthCheckOptions, LoggingOptions, MetricsOptions, ServiceDiscoveryOptions};
use std::path::PathBuf;

#[test]
fn test_configuration_from_env_defaults() {
    // Falls back to defaults when no env vars are set
    let config = BackendConfig::from_env().unwrap();
    assert_eq!(config.service_name, "inferno-backend");
}

#[test]
fn test_default_configuration() {
    let config = BackendConfig::default();
    assert_eq!(config.listen_addr.port(), 3000);
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.service_name, "inferno-backend");
    assert!(config.enable_metrics);
}

#[test]
fn test_cli_to_config_conversion() {
    let cli = BackendCliOptions {
        listen_addr: "127.0.0.1:4000".parse().unwrap(),
        model_path: PathBuf::from("/path/to/model.bin"),
        model_type: "llama".to_string(),
        max_batch_size: 64,
        gpu_device_id: 0,
        max_context_length: 4096,
        memory_pool_mb: 2048,
        discovery_lb: Some("127.0.0.1:8080,127.0.0.1:8081".to_string()),
        enable_cache: false,
        cache_ttl_seconds: 7200,
        logging: LoggingOptions {
            log_level: "debug".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: Some("127.0.0.1:6101".parse().unwrap()),
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/status".to_string(),
        },
        service_discovery: ServiceDiscoveryOptions {
            service_name: Some("custom-backend".to_string()),
            registration_endpoint: Some("http://127.0.0.1:8080/register".to_string()),
        },
    };

    let config = cli.to_config().unwrap();

    assert_eq!(config.listen_addr, "127.0.0.1:4000".parse().unwrap());
    assert_eq!(config.model_path, PathBuf::from("/path/to/model.bin"));
    assert_eq!(config.model_type, "llama");
    assert_eq!(config.max_batch_size, 64);
    assert_eq!(config.gpu_device_id, 0);
    assert_eq!(config.max_context_length, 4096);
    assert_eq!(config.memory_pool_mb, 2048);
    assert_eq!(config.discovery_lb.len(), 2);
    assert!(!config.enable_cache);
    assert_eq!(config.cache_ttl_seconds, 7200);
    assert_eq!(config.health_check_path, "/status");
    assert_eq!(config.service_name, "custom-backend");
}

#[test]
fn test_cli_to_config_with_defaults() {
    let cli = BackendCliOptions {
        listen_addr: "127.0.0.1:3000".parse().unwrap(),
        model_path: PathBuf::from("model.bin"),
        model_type: "auto".to_string(),
        max_batch_size: 32,
        gpu_device_id: -1,
        max_context_length: 2048,
        memory_pool_mb: 1024,
        discovery_lb: None,
        enable_cache: true,
        cache_ttl_seconds: 3600,
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: None,
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        service_discovery: ServiceDiscoveryOptions {
            service_name: None,
            registration_endpoint: None,
        },
    };

    let config = cli.to_config().unwrap();

    assert_eq!(config.listen_addr, "127.0.0.1:3000".parse().unwrap());
    assert_eq!(config.model_path, PathBuf::from("model.bin"));
    assert_eq!(config.model_type, "auto");
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.gpu_device_id, -1);
    assert_eq!(config.max_context_length, 2048);
    assert_eq!(config.memory_pool_mb, 1024);
    assert!(config.discovery_lb.is_empty());
    assert!(config.enable_cache);
    assert_eq!(config.cache_ttl_seconds, 3600);
    assert_eq!(config.health_check_path, "/health");
    assert_eq!(config.service_name, "inferno-backend");
}

#[tokio::test]
async fn test_backend_cli_run_method() {
    let cli = BackendCliOptions {
        listen_addr: "127.0.0.1:3001".parse().unwrap(),
        model_path: PathBuf::from("test_model.bin"),
        model_type: "auto".to_string(),
        max_batch_size: 32,
        gpu_device_id: -1,
        max_context_length: 2048,
        memory_pool_mb: 1024,
        discovery_lb: None,
        enable_cache: true,
        cache_ttl_seconds: 3600,
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: false, // Disable to avoid port conflicts
            operations_addr: None,
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        service_discovery: ServiceDiscoveryOptions {
            service_name: None,
            registration_endpoint: None,
        },
    };

    // Since run() waits for ctrl+c, we need to test it in a timeout
    let run_future = cli.run();

    // Test that it starts correctly (within a timeout)
    let result = tokio::time::timeout(std::time::Duration::from_millis(100), run_future).await;

    // The run should timeout (which means it started correctly and is waiting for shutdown)
    assert!(
        result.is_err(),
        "Backend run method should timeout while waiting for shutdown signal"
    );
}
