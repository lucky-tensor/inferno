//! Performance Tests
//!
//! These tests verify that performance characteristics remain within
//! acceptable bounds and catch performance regressions.

use inferno_proxy::{ProxyConfig, ProxyServer};
use inferno_shared::{InfernoError, MetricsCollector};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use inferno_shared::test_utils::get_random_port_addr;

#[test]
fn test_config_creation_performance() {
    let start = Instant::now();

    for _ in 0..1000 {
        let _config = ProxyConfig::default();
    }

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(10),
        "Config creation too slow: {:?}",
        duration
    );
}

#[test]
fn test_metrics_update_performance() {
    let collector = MetricsCollector::new();
    let start = Instant::now();

    for _ in 0..10000 {
        collector.record_request();
        collector.record_response(200);
        collector.record_request_duration(Duration::from_millis(5));
    }

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(50),
        "Metrics updates too slow: {:?}",
        duration
    );
}

#[test]
fn test_error_creation_performance() {
    let start = Instant::now();

    for i in 0..1000 {
        let _error = InfernoError::network(
            format!("192.168.1.{}:8080", i % 255),
            "Connection refused",
            None,
        );
    }

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(20),
        "Error creation too slow: {:?}",
        duration
    );
}

#[tokio::test]
async fn test_server_creation_performance() {
    let start = Instant::now();

    for _i in 0..10 {
        let config = ProxyConfig {
            listen_addr: get_random_port_addr(),
            backend_addr: get_random_port_addr(),
            ..Default::default()
        };

        let _server = ProxyServer::new(config).await.unwrap();
    }

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(2000),
        "Server creation too slow: {:?}",
        duration
    );
}

#[test]
fn test_metrics_snapshot_performance() {
    let collector = MetricsCollector::new();

    // Add substantial data
    for _ in 0..1000 {
        collector.record_request();
        collector.record_response(200);
        collector.record_request_duration(Duration::from_millis(10));
    }

    let start = Instant::now();

    for _ in 0..100 {
        let _snapshot = collector.snapshot();
    }

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(100),
        "Snapshot creation too slow: {:?}",
        duration
    );
}

#[test]
fn test_prometheus_format_performance() {
    let collector = MetricsCollector::new();

    // Add test data
    for _ in 0..1000 {
        collector.record_request();
        collector.record_response(200);
    }

    let snapshot = collector.snapshot();
    let start = Instant::now();

    for _ in 0..100 {
        let _prometheus = snapshot.to_prometheus_format();
    }

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(500),
        "Prometheus format too slow: {:?}",
        duration
    );
}
