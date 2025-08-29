//! Integration Tests
//!
//! These tests verify that different components work together correctly
//! and that the overall system behaves as expected.

use inferno_proxy::{ProxyConfig, ProxyServer};
use inferno_shared::InfernoError;
use std::time::Duration;

#[tokio::test]
async fn test_server_with_metrics_integration() {
    let config = ProxyConfig::default();
    let server = ProxyServer::new(config).await.unwrap();

    // Test that server and metrics are properly integrated
    let metrics = server.metrics();

    // Simulate some activity
    metrics.record_request();
    metrics.record_response(200);
    metrics.record_request_duration(Duration::from_millis(15));

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_requests, 1);
    assert_eq!(snapshot.total_responses, 1);
    assert_eq!(snapshot.status_2xx, 1);
    assert_eq!(snapshot.active_requests, 0);
    assert!(snapshot.duration_histogram[3] > 0); // Should be in < 50ms bucket

    // Test success rate calculation
    assert_eq!(snapshot.success_rate(), 1.0); // 100% success
    assert_eq!(snapshot.error_rate(), 0.0); // 0% errors
}

#[tokio::test]
async fn test_config_validation_integration() {
    // Test that invalid configurations are properly rejected
    let config = ProxyConfig {
        timeout: Duration::from_secs(0), // Invalid
        ..Default::default()
    };

    let result = ProxyServer::new(config).await;
    assert!(result.is_err());

    let error = result.unwrap_err();
    assert!(matches!(error, InfernoError::Configuration { .. }));
    assert_eq!(error.to_http_status(), 500);
    assert!(!error.is_temporary());
}

#[tokio::test]
async fn test_environment_config_integration() {
    // Test environment variable configuration
    std::env::set_var("INFERNO_LISTEN_ADDR", "127.0.0.1:8888");
    std::env::set_var("INFERNO_MAX_CONNECTIONS", "15000");

    let config = ProxyConfig::from_env().unwrap();
    let server = ProxyServer::new(config).await.unwrap();

    assert_eq!(server.local_addr().port(), 8888);
    assert_eq!(server.config().max_connections, 15000);

    // Clean up
    std::env::remove_var("INFERNO_LISTEN_ADDR");
    std::env::remove_var("INFERNO_MAX_CONNECTIONS");
}

#[tokio::test]
async fn test_concurrent_server_creation() {
    // Test that multiple servers can be created concurrently without issues
    let mut handles = vec![];

    for i in 0..5 {
        let handle = tokio::spawn(async move {
            let config = ProxyConfig {
                listen_addr: format!("127.0.0.1:{}", 9000 + i).parse().unwrap(),
                backend_addr: format!("127.0.0.1:{}", 3000 + i).parse().unwrap(),
                ..Default::default()
            };

            ProxyServer::new(config).await
        });
        handles.push(handle);
    }

    // All servers should be created successfully
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}