//! Metrics Tests
//!
//! Tests for metrics collection, calculation, and Prometheus format generation.

use inferno_shared::metrics::{MetricsCollector, MetricsSnapshot};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::thread;

#[test]
fn test_metrics_collector_new() {
    let collector = MetricsCollector::new();
    let snapshot = collector.snapshot();

    assert_eq!(snapshot.total_requests, 0);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.total_responses, 0);
    assert_eq!(snapshot.total_errors, 0);
}

#[test]
fn test_record_request() {
    let collector = MetricsCollector::new();

    collector.record_request();
    let snapshot = collector.snapshot();

    assert_eq!(snapshot.total_requests, 1);
    assert_eq!(snapshot.active_requests, 1);
}

#[test]
fn test_record_response() {
    let collector = MetricsCollector::new();

    collector.record_request();
    collector.record_response(200);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.total_responses, 1);
    assert_eq!(snapshot.status_2xx, 1);
    assert_eq!(snapshot.active_requests, 0);
}

#[test]
fn test_record_error() {
    let collector = MetricsCollector::new();

    collector.record_request();
    collector.record_error();

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.total_errors, 1);
    assert_eq!(snapshot.active_requests, 0);
}

#[test]
fn test_duration_histogram() {
    let collector = MetricsCollector::new();

    // Test different duration buckets
    collector.record_request_duration(Duration::from_micros(500)); // < 1ms
    collector.record_request_duration(Duration::from_millis(3)); // < 5ms
    collector.record_request_duration(Duration::from_millis(25)); // < 50ms
    collector.record_request_duration(Duration::from_millis(200)); // < 500ms
    collector.record_request_duration(Duration::from_secs(2)); // < 5s

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.duration_histogram[0], 1); // < 1ms
    assert_eq!(snapshot.duration_histogram[1], 1); // < 5ms
    assert_eq!(snapshot.duration_histogram[3], 1); // < 50ms
    assert_eq!(snapshot.duration_histogram[5], 1); // < 500ms
    assert_eq!(snapshot.duration_histogram[7], 1); // < 5s
}

#[test]
fn test_success_rate_calculation() {
    let snapshot = MetricsSnapshot {
        total_requests: 100,
        active_requests: 0,
        total_responses: 100,
        total_errors: 0,
        status_2xx: 80,
        status_3xx: 15,
        status_4xx: 4,
        status_5xx: 1,
        backend_connections: 0,
        backend_connection_errors: 0,
        active_backend_connections: 0,
        duration_histogram: [0; 9],
        average_upstream_selection_time_us: 0,
        uptime: Duration::from_secs(3600),
        timestamp: SystemTime::now(),
    };

    assert_eq!(snapshot.success_rate(), 0.95); // 95 out of 100
}

#[test]
fn test_error_rate_calculation() {
    let snapshot = MetricsSnapshot {
        total_requests: 100,
        active_requests: 0,
        total_responses: 98,
        total_errors: 2,
        status_2xx: 85,
        status_3xx: 10,
        status_4xx: 2,
        status_5xx: 1,
        backend_connections: 0,
        backend_connection_errors: 0,
        active_backend_connections: 0,
        duration_histogram: [0; 9],
        average_upstream_selection_time_us: 0,
        uptime: Duration::from_secs(3600),
        timestamp: SystemTime::now(),
    };

    assert_eq!(snapshot.error_rate(), 0.03); // 3 errors out of 100 requests
}

#[test]
fn test_requests_per_second() {
    let snapshot = MetricsSnapshot {
        total_requests: 3600,
        active_requests: 0,
        total_responses: 3600,
        total_errors: 0,
        status_2xx: 3600,
        status_3xx: 0,
        status_4xx: 0,
        status_5xx: 0,
        backend_connections: 0,
        backend_connection_errors: 0,
        active_backend_connections: 0,
        duration_histogram: [0; 9],
        average_upstream_selection_time_us: 0,
        uptime: Duration::from_secs(3600), // 1 hour
        timestamp: SystemTime::now(),
    };

    assert_eq!(snapshot.requests_per_second(), 1.0); // 1 RPS
}

#[test]
fn test_prometheus_format() {
    let snapshot = MetricsSnapshot {
        total_requests: 100,
        active_requests: 5,
        total_responses: 95,
        total_errors: 0,
        status_2xx: 90,
        status_3xx: 5,
        status_4xx: 0,
        status_5xx: 0,
        backend_connections: 100,
        backend_connection_errors: 1,
        active_backend_connections: 10,
        duration_histogram: [50, 30, 10, 4, 1, 0, 0, 0, 0],
        average_upstream_selection_time_us: 25,
        uptime: Duration::from_secs(60),
        timestamp: SystemTime::now(),
    };

    let prometheus = snapshot.to_prometheus_format();

    assert!(prometheus.contains("inferno_requests_total 100"));
    assert!(prometheus.contains("inferno_requests_active 5"));
    assert!(prometheus.contains("inferno_responses_by_status_total{status=\"2xx\"} 90"));
    assert!(prometheus.contains("inferno_success_rate 1.000000"));
}

#[test]
fn test_concurrent_metrics_updates() {
    let collector = Arc::new(MetricsCollector::new());
    let mut handles = vec![];

    // Spawn multiple threads updating metrics concurrently
    for _ in 0..10 {
        let collector_clone = Arc::clone(&collector);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                collector_clone.record_request();
                collector_clone.record_response(200);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.total_requests, 10000);
    assert_eq!(snapshot.total_responses, 10000);
    assert_eq!(snapshot.status_2xx, 10000);
    assert_eq!(snapshot.active_requests, 0);
}