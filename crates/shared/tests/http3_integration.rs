//! Integration tests for HTTP/3 service discovery client
//!
//! These tests verify the HTTP/3 client functionality for service discovery
//! operations, including registration, updates, peer discovery, and health checks.

use inferno_shared::service_discovery::http3_client::{
    Http3ClientConfig, Http3Metrics, Http3ServiceDiscoveryClient,
};
use inferno_shared::service_discovery::{NodeInfo, NodeType};
use std::time::Duration;

#[tokio::test]
async fn test_http3_client_creation() {
    let _client = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
        .await
        .expect("Failed to create HTTP/3 client");

    // Client should be created successfully
}

#[tokio::test]
async fn test_http3_client_with_config() {
    let config = Http3ClientConfig::new(Duration::from_secs(10));
    let _client = Http3ServiceDiscoveryClient::with_config(config)
        .await
        .expect("Failed to create HTTP/3 client with config");
}

#[tokio::test]
async fn test_http3_client_high_throughput_config() {
    let config = Http3ClientConfig::high_throughput(Duration::from_secs(5));
    assert_eq!(config.max_concurrent_streams, 1000);
    assert_eq!(config.initial_max_stream_data, 1024 * 1024);

    let _client = Http3ServiceDiscoveryClient::with_config(config)
        .await
        .expect("Failed to create high-throughput client");
}

#[tokio::test]
async fn test_http3_client_testing_config() {
    let config = Http3ClientConfig::for_testing();
    assert!(config.accept_invalid_certs);
    assert_eq!(config.max_concurrent_streams, 10);

    let _client = Http3ServiceDiscoveryClient::with_config(config)
        .await
        .expect("Failed to create testing client");
}

#[tokio::test]
async fn test_http3_register_network_error() {
    let client = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
        .await
        .expect("Failed to create client");

    let node = NodeInfo::new(
        "test-node".to_string(),
        "127.0.0.1:8080".to_string(),
        8080,
        NodeType::Backend,
    );

    let result = client
        .register_with_peer("https://localhost:9999", &node)
        .await;

    // Should return network error (no server running)
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        inferno_shared::service_discovery::ServiceDiscoveryError::NetworkError { .. }
    ));
}

#[tokio::test]
async fn test_http3_update_network_error() {
    let client = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
        .await
        .expect("Failed to create client");

    let node = NodeInfo::new(
        "test-node".to_string(),
        "127.0.0.1:8080".to_string(),
        8080,
        NodeType::Backend,
    );

    let result = client
        .update_with_peer("https://localhost:9999", &node)
        .await;

    // Should return network error (no server running)
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        inferno_shared::service_discovery::ServiceDiscoveryError::NetworkError { .. }
    ));
}

#[tokio::test]
async fn test_http3_discover_peers_network_error() {
    let client = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
        .await
        .expect("Failed to create client");

    let result = client.discover_peers("https://localhost:9999").await;

    // Should return network error (no server running)
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        inferno_shared::service_discovery::ServiceDiscoveryError::NetworkError { .. }
    ));
}

#[tokio::test]
async fn test_http3_health_check_network_error() {
    let client = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
        .await
        .expect("Failed to create client");

    let result = client.check_peer_health("https://localhost:9999").await;

    // Should return network error (no server running)
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        inferno_shared::service_discovery::ServiceDiscoveryError::NetworkError { .. }
    ));
}

#[test]
fn test_http3_config_fields() {
    let config = Http3ClientConfig::new(Duration::from_secs(5));

    assert_eq!(config.request_timeout, Duration::from_secs(5));
    assert_eq!(config.max_concurrent_streams, 100);
    assert_eq!(config.keep_alive_interval, Duration::from_secs(30));
    assert_eq!(config.initial_max_stream_data, 256 * 1024);
    assert_eq!(config.idle_timeout, Duration::from_secs(300));
    assert!(!config.accept_invalid_certs);
}

#[test]
fn test_http3_config_cloning() {
    let config1 = Http3ClientConfig::new(Duration::from_secs(5));
    let config2 = config1.clone();

    assert_eq!(config1.request_timeout, config2.request_timeout);
    assert_eq!(
        config1.max_concurrent_streams,
        config2.max_concurrent_streams
    );
    assert_eq!(config1.keep_alive_interval, config2.keep_alive_interval);
}

#[test]
fn test_http3_metrics_initialization() {
    let metrics = Http3Metrics::default();

    assert_eq!(metrics.zero_rtt_connections, 0);
    assert_eq!(metrics.connection_migrations, 0);
    assert_eq!(metrics.stream_resets, 0);
    assert_eq!(metrics.avg_rtt_us, 0);
    assert_eq!(metrics.retransmitted_packets, 0);
    assert_eq!(metrics.active_connections, 0);
    assert_eq!(metrics.bytes_sent, 0);
    assert_eq!(metrics.bytes_received, 0);
}

#[test]
fn test_http3_metrics_update() {
    let metrics = Http3Metrics {
        zero_rtt_connections: 10,
        connection_migrations: 5,
        bytes_sent: 1024,
        bytes_received: 2048,
        ..Default::default()
    };

    assert_eq!(metrics.zero_rtt_connections, 10);
    assert_eq!(metrics.connection_migrations, 5);
    assert_eq!(metrics.bytes_sent, 1024);
    assert_eq!(metrics.bytes_received, 2048);
}

#[test]
fn test_http3_metrics_cloning() {
    let metrics1 = Http3Metrics {
        zero_rtt_connections: 15,
        avg_rtt_us: 500,
        ..Default::default()
    };

    let metrics2 = metrics1.clone();

    assert_eq!(metrics1.zero_rtt_connections, metrics2.zero_rtt_connections);
    assert_eq!(metrics1.avg_rtt_us, metrics2.avg_rtt_us);
}

#[tokio::test]
async fn test_multiple_clients_creation() {
    let _client1 = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
        .await
        .expect("Failed to create client 1");

    let _client2 = Http3ServiceDiscoveryClient::new(Duration::from_secs(10))
        .await
        .expect("Failed to create client 2");

    let config = Http3ClientConfig::for_testing();
    let _client3 = Http3ServiceDiscoveryClient::with_config(config)
        .await
        .expect("Failed to create client 3");

    // All clients should be created successfully
}
