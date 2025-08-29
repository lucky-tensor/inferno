//! Metrics Tests
//!
//! Tests for metrics collection and calculation functionality.
//! These tests verify that metrics are collected accurately, calculations
//! are correct, and performance is maintained under various load patterns.

use inferno_shared::{MetricsCollector, MetricsSnapshot};
use std::time::{Duration, SystemTime};

#[test]
fn test_metrics_collector_creation() {
    let collector = MetricsCollector::new();
    let snapshot = collector.snapshot();

    assert_eq!(snapshot.total_requests, 0);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.total_responses, 0);
    assert_eq!(snapshot.total_errors, 0);
    assert_eq!(snapshot.status_2xx, 0);
    assert_eq!(snapshot.status_3xx, 0);
    assert_eq!(snapshot.status_4xx, 0);
    assert_eq!(snapshot.status_5xx, 0);
    assert_eq!(snapshot.backend_connections, 0);
    assert_eq!(snapshot.backend_connection_errors, 0);
    assert_eq!(snapshot.active_backend_connections, 0);
    assert_eq!(snapshot.duration_histogram, [0; 9]);
    assert_eq!(snapshot.average_upstream_selection_time_us, 0);
    assert!(snapshot.uptime < Duration::from_millis(100)); // Should be very small
}

#[test]
fn test_request_response_tracking() {
    let collector = MetricsCollector::new();

    // Record several requests and responses
    collector.record_request();
    collector.record_request();
    collector.record_request();

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.total_requests, 3);
    assert_eq!(snapshot.active_requests, 3);

    // Complete some responses
    collector.record_response(200);
    collector.record_response(404);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.total_responses, 2);
    assert_eq!(snapshot.active_requests, 1);
    assert_eq!(snapshot.status_2xx, 1);
    assert_eq!(snapshot.status_4xx, 1);
}

#[test]
fn test_error_tracking() {
    let collector = MetricsCollector::new();

    collector.record_request();
    collector.record_request();
    collector.record_error();

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.total_errors, 1);
    assert_eq!(snapshot.active_requests, 1); // One request still active
}

#[test]
fn test_status_code_classification() {
    let collector = MetricsCollector::new();

    // Test various status codes
    collector.record_response(200); // 2xx
    collector.record_response(201); // 2xx
    collector.record_response(301); // 3xx
    collector.record_response(302); // 3xx
    collector.record_response(400); // 4xx
    collector.record_response(404); // 4xx
    collector.record_response(500); // 5xx
    collector.record_response(503); // 5xx

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.status_2xx, 2);
    assert_eq!(snapshot.status_3xx, 2);
    assert_eq!(snapshot.status_4xx, 2);
    assert_eq!(snapshot.status_5xx, 2);
    assert_eq!(snapshot.total_responses, 8);
}

#[test]
fn test_upstream_selection_timing() {
    let collector = MetricsCollector::new();

    collector.record_upstream_selection_time(Duration::from_micros(50));
    collector.record_upstream_selection_time(Duration::from_micros(30));
    collector.record_upstream_selection_time(Duration::from_micros(70));

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.average_upstream_selection_time_us, 50); // (50+30+70)/3 = 50
}

#[test]
fn test_backend_connection_tracking() {
    let collector = MetricsCollector::new();

    collector.record_backend_connection();
    collector.record_backend_connection();
    collector.record_backend_connection_error();

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.backend_connections, 2);
    assert_eq!(snapshot.backend_connection_errors, 1);
    assert_eq!(snapshot.active_backend_connections, 1); // 2 started, 1 errored
}

#[test]
fn test_metrics_calculations() {
    let snapshot = MetricsSnapshot {
        total_requests: 100,
        active_requests: 5,
        total_responses: 95,
        total_errors: 2,
        status_2xx: 80,
        status_3xx: 10,
        status_4xx: 3,
        status_5xx: 2,
        backend_connections: 100,
        backend_connection_errors: 5,
        active_backend_connections: 10,
        duration_histogram: [20, 30, 25, 15, 5, 0, 0, 0, 0],
        average_upstream_selection_time_us: 75,
        uptime: Duration::from_secs(3600), // 1 hour
        timestamp: SystemTime::now(),
    };

    // Test success rate calculation
    assert_eq!(snapshot.success_rate(), 0.9473684210526315); // (80+10)/(80+10+3+2) = 90/95

    // Test error rate calculation
    assert_eq!(snapshot.error_rate(), 0.04); // (2+2)/100 = 4/100

    // Test backend connection success rate
    assert_eq!(snapshot.backend_connection_success_rate(), 0.95); // (100-5)/100 = 95/100

    // Test requests per second
    assert_eq!(snapshot.requests_per_second(), 100.0 / 3600.0); // ~0.0278

    // Test P95 estimation (rough)
    let p95 = snapshot.p95_response_time_ms();
    assert!(p95 > 0); // Should be some reasonable value
}

#[test]
fn test_metrics_reset() {
    let collector = MetricsCollector::new();

    // Add some data
    collector.record_request();
    collector.record_response(200);
    collector.record_request_duration(Duration::from_millis(10));

    assert_eq!(collector.snapshot().total_requests, 1);

    // Reset and verify
    collector.reset();
    let snapshot = collector.snapshot();

    assert_eq!(snapshot.total_requests, 0);
    assert_eq!(snapshot.total_responses, 0);
    assert_eq!(snapshot.status_2xx, 0);
    assert_eq!(snapshot.duration_histogram[3], 0); // < 50ms bucket
}

#[test]
fn test_prometheus_format_generation() {
    let collector = MetricsCollector::new();

    // Add some test data
    for _ in 0..100 {
        collector.record_request();
        collector.record_response(200);
    }
    collector.record_response(404);
    collector.record_response(500);

    let snapshot = collector.snapshot();
    let prometheus = snapshot.to_prometheus_format();

    // Verify key metrics are present
    assert!(prometheus.contains("inferno_requests_total"));
    assert!(prometheus.contains("inferno_responses_by_status_total"));
    assert!(prometheus.contains("inferno_request_duration_ms"));
    assert!(prometheus.contains("inferno_success_rate"));

    // Verify values are correct
    assert!(prometheus.contains("inferno_requests_total 100"));
    assert!(prometheus.contains("inferno_responses_by_status_total{status=\"2xx\"} 100"));
    assert!(prometheus.contains("inferno_responses_by_status_total{status=\"4xx\"} 1"));
    assert!(prometheus.contains("inferno_responses_by_status_total{status=\"5xx\"} 1"));
}
