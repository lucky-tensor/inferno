//! Server Tests
//!
//! Tests for server lifecycle and management functionality.
//! These tests verify server creation, configuration, lifecycle management,
//! and proper resource cleanup.

use inferno_proxy::{ProxyConfig, ProxyServer};
use std::time::Duration;

#[tokio::test]
async fn test_server_creation() {
    let config = ProxyConfig::default();
    let result = ProxyServer::new(config).await;

    assert!(result.is_ok());
    let server = result.unwrap();
    assert_eq!(server.local_addr().port(), 8080);
}

#[tokio::test]
async fn test_server_with_custom_config() {
    let config = ProxyConfig {
        listen_addr: "127.0.0.1:9999".parse().unwrap(),
        backend_addr: "127.0.0.1:4000".parse().unwrap(),
        max_connections: 5000,
        timeout: Duration::from_secs(60),
        ..Default::default()
    };

    let result = ProxyServer::new(config).await;
    assert!(result.is_ok());

    let server = result.unwrap();
    assert_eq!(server.local_addr().port(), 9999);
    assert_eq!(server.config().backend_addr.port(), 4000);
    assert_eq!(server.config().max_connections, 5000);
    assert_eq!(server.config().timeout, Duration::from_secs(60));
}

#[tokio::test]
async fn test_server_with_invalid_config() {
    let mut config = ProxyConfig::default();
    config.backend_addr = config.listen_addr; // Invalid: same as listen addr

    let result = ProxyServer::new(config).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("cannot be the same"));
}

#[tokio::test]
async fn test_server_metrics_access() {
    let config = ProxyConfig::default();
    let server = ProxyServer::new(config).await.unwrap();

    let metrics = server.metrics();
    let snapshot = metrics.snapshot();

    assert_eq!(snapshot.total_requests, 0);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.total_responses, 0);
    assert_eq!(snapshot.total_errors, 0);
}

#[tokio::test]
async fn test_server_config_access() {
    let config = ProxyConfig {
        max_connections: 7500,
        enable_health_check: false,
        ..Default::default()
    };

    let server = ProxyServer::new(config).await.unwrap();
    let server_config = server.config();

    assert_eq!(server_config.max_connections, 7500);
    assert!(!server_config.enable_health_check);
}

#[tokio::test]
async fn test_server_shutdown_signal() {
    let config = ProxyConfig::default();
    let mut server = ProxyServer::new(config).await.unwrap();

    // Test shutdown signal
    let result = server.shutdown().await;
    assert!(result.is_ok());

    // Second shutdown should fail since shutdown_tx was consumed
    let result = server.shutdown().await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not running"));
}

#[tokio::test]
async fn test_server_startup_timeout() {
    let config = ProxyConfig::default();
    let _server = ProxyServer::new(config).await.unwrap();

    // Test that server creation completes quickly
    let start = std::time::Instant::now();
    let _server = ProxyServer::new(ProxyConfig::default()).await.unwrap();
    let duration = start.elapsed();

    // Server creation should be fast (< 100ms)
    assert!(duration < Duration::from_millis(100));
}