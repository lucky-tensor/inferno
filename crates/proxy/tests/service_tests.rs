//! Service Tests
//!
//! Tests for proxy service functionality.
//! These tests verify proxy service creation, configuration handling,
//! and basic service operations.

use inferno_proxy::{ProxyConfig, ProxyService};
use inferno_shared::MetricsCollector;
use std::sync::Arc;

#[test]
fn test_proxy_service_creation() {
    let config = Arc::new(ProxyConfig::default());
    let metrics = Arc::new(MetricsCollector::new());

    let service = ProxyService::new(config, metrics);

    // Verify service was created successfully
    assert_eq!(service.config().listen_addr.port(), 8080);
    assert_eq!(service.config().backend_addr.port(), 3000);
}

#[test]
fn test_proxy_service_metrics_access() {
    let config = Arc::new(ProxyConfig::default());
    let metrics = Arc::new(MetricsCollector::new());

    let service = ProxyService::new(Arc::clone(&config), Arc::clone(&metrics));

    // Test that service shares the same metrics instance
    metrics.record_request();
    let snapshot = service.metrics().snapshot();
    assert_eq!(snapshot.total_requests, 1);
}

#[test]
fn test_proxy_service_with_multiple_backends() {
    let config = ProxyConfig {
        backend_servers: vec![
            "192.168.1.1:8080".parse().unwrap(),
            "192.168.1.2:8080".parse().unwrap(),
        ],
        ..Default::default()
    };

    let config = Arc::new(config);
    let metrics = Arc::new(MetricsCollector::new());
    let service = ProxyService::new(config, metrics);

    assert!(service.config().has_multiple_backends());
    assert_eq!(service.config().effective_backends().len(), 2);
}
