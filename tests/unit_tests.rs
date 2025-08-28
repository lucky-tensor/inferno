//! # Unit Tests for Pingora Proxy Demo
//!
//! Comprehensive unit tests for all components of the proxy system.
//! These tests focus on individual component functionality, performance
//! characteristics, and edge case handling.
//!
//! ## Test Categories
//!
//! 1. **Configuration Tests**: Configuration parsing, validation, and environment handling
//! 2. **Metrics Tests**: Metrics collection, calculation, and export functionality
//! 3. **Error Tests**: Error creation, classification, and conversion
//! 4. **Server Tests**: Server lifecycle, configuration, and management
//! 5. **Integration Tests**: Component interaction and end-to-end scenarios
//! 6. **Performance Tests**: Performance characteristics and regression detection
//!
//! ## Performance Requirements
//!
//! All unit tests should complete within reasonable time bounds:
//! - Individual tests: < 1 second
//! - Full test suite: < 30 seconds
//! - Memory usage: < 100MB total
//! - No memory leaks or resource leakage

use inferno_proxy::metrics::{MetricsCollector, MetricsSnapshot};
use inferno_proxy::{ProxyConfig, ProxyError, ProxyServer, ProxyService};
use serial_test::serial;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Tests for configuration management functionality
///
/// These tests verify that configuration loading, validation, and
/// environment variable handling work correctly under various scenarios.
mod config_tests {
    use super::*;

    #[test]
    fn test_default_configuration() {
        let config = ProxyConfig::default();

        assert_eq!(config.listen_addr.port(), 8080);
        assert_eq!(config.backend_addr.port(), 3000);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_connections, 10000);
        assert!(config.enable_health_check);
        assert_eq!(config.health_check_interval, Duration::from_secs(30));
        assert_eq!(config.health_check_path, "/health");
        assert!(!config.enable_tls);
        assert_eq!(config.log_level, "info");
        assert!(config.enable_metrics);
        assert_eq!(config.load_balancing_algorithm, "round_robin");
        assert!(config.backend_servers.is_empty());
    }

    #[test]
    fn test_configuration_validation_success() {
        let config = ProxyConfig::default();
        let result = ProxyConfig::new(config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_configuration_validation_same_addresses() {
        let mut config = ProxyConfig::default();
        config.backend_addr = config.listen_addr;

        let result = ProxyConfig::new(config);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("cannot be the same"));
        assert_eq!(error.to_http_status(), 500); // Internal error for config issues
    }

    #[test]
    fn test_configuration_validation_invalid_timeout() {
        let config = ProxyConfig {
            timeout: Duration::from_secs(0),
            ..Default::default()
        };

        let result = ProxyConfig::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timeout must be"));
    }

    #[test]
    fn test_configuration_validation_invalid_connections() {
        let mut config = ProxyConfig {
            max_connections: 0,
            ..Default::default()
        };

        let result = ProxyConfig::new(config.clone());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_connections"));

        config.max_connections = 2_000_000;
        let result = ProxyConfig::new(config.clone());
        assert!(result.is_err());
    }

    #[test]
    fn test_configuration_tls_validation() {
        let config = ProxyConfig {
            enable_tls: true,
            ..Default::default()
        };
        // tls_cert_path and tls_key_path remain None

        let result = ProxyConfig::new(config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("are required when"));
    }

    #[test]
    fn test_configuration_invalid_log_level() {
        let config = ProxyConfig {
            log_level: "invalid_level".to_string(),
            ..Default::default()
        };

        let result = ProxyConfig::new(config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid log_level"));
    }

    #[test]
    fn test_configuration_invalid_load_balancing_algorithm() {
        let config = ProxyConfig {
            load_balancing_algorithm: "invalid_algorithm".to_string(),
            ..Default::default()
        };

        let result = ProxyConfig::new(config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid load_balancing_algorithm"));
    }

    #[test]
    fn test_configuration_health_check_path_validation() {
        let config = ProxyConfig {
            health_check_path: "invalid_path".to_string(), // Doesn't start with /
            ..Default::default()
        };

        let result = ProxyConfig::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must start with"));
    }

    #[test]
    fn test_effective_backends_single() {
        let config = ProxyConfig::default();
        let backends = config.effective_backends();

        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0], config.backend_addr);
        assert!(!config.has_multiple_backends());
    }

    #[test]
    fn test_effective_backends_multiple() {
        let config = ProxyConfig {
            backend_servers: vec![
                "192.168.1.1:8080".parse().unwrap(),
                "192.168.1.2:8080".parse().unwrap(),
                "192.168.1.3:8080".parse().unwrap(),
            ],
            ..Default::default()
        };

        let backends = config.effective_backends();
        assert_eq!(backends.len(), 3);
        assert_eq!(backends, config.backend_servers);
        assert!(config.has_multiple_backends());
    }

    #[test]
    #[serial]
    fn test_configuration_from_env_defaults() {
        // Clear any existing environment variables
        for (key, _) in std::env::vars() {
            if key.starts_with("PINGORA_") {
                std::env::remove_var(key);
            }
        }

        let config = ProxyConfig::from_env().unwrap();
        let default_config = ProxyConfig::default();

        assert_eq!(config.listen_addr, default_config.listen_addr);
        assert_eq!(config.backend_addr, default_config.backend_addr);
        assert_eq!(config.timeout, default_config.timeout);
        assert_eq!(config.max_connections, default_config.max_connections);
    }

    #[test]
    #[serial]
    fn test_configuration_from_env_custom_values() {
        // Clear any existing environment variables first
        for (key, _) in std::env::vars() {
            if key.starts_with("PINGORA_") {
                std::env::remove_var(key);
            }
        }

        std::env::set_var("PINGORA_LISTEN_ADDR", "127.0.0.1:9091");
        std::env::set_var("PINGORA_BACKEND_ADDR", "127.0.0.1:4000");
        std::env::set_var("PINGORA_TIMEOUT_SECONDS", "60");
        std::env::set_var("PINGORA_MAX_CONNECTIONS", "5000");
        std::env::set_var("PINGORA_LOG_LEVEL", "debug");
        std::env::set_var(
            "PINGORA_BACKEND_SERVERS",
            "192.168.1.1:8080,192.168.1.2:8080",
        );

        let config = ProxyConfig::from_env().unwrap();

        assert_eq!(config.listen_addr.port(), 9091);
        assert_eq!(config.backend_addr.port(), 4000);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_connections, 5000);
        assert_eq!(config.log_level, "debug");
        assert_eq!(config.backend_servers.len(), 2);

        // Clean up
        std::env::remove_var("PINGORA_LISTEN_ADDR");
        std::env::remove_var("PINGORA_BACKEND_ADDR");
        std::env::remove_var("PINGORA_TIMEOUT_SECONDS");
        std::env::remove_var("PINGORA_MAX_CONNECTIONS");
        std::env::remove_var("PINGORA_LOG_LEVEL");
        std::env::remove_var("PINGORA_BACKEND_SERVERS");
    }

    #[test]
    #[serial]
    fn test_configuration_from_env_invalid_values() {
        // Clear any existing environment variables first
        for (key, _) in std::env::vars() {
            if key.starts_with("PINGORA_") {
                std::env::remove_var(key);
            }
        }

        std::env::set_var("PINGORA_LISTEN_ADDR", "invalid_address");

        let result = ProxyConfig::from_env();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid PINGORA_LISTEN_ADDR"));

        std::env::remove_var("PINGORA_LISTEN_ADDR");
    }
}

/// Tests for metrics collection and calculation functionality
///
/// These tests verify that metrics are collected accurately, calculations
/// are correct, and performance is maintained under various load patterns.
mod metrics_tests {
    use super::*;

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
    fn test_duration_histogram() {
        let collector = MetricsCollector::new();

        // Test different duration buckets
        collector.record_request_duration(Duration::from_micros(500)); // < 1ms
        collector.record_request_duration(Duration::from_millis(3)); // < 5ms
        collector.record_request_duration(Duration::from_millis(7)); // < 10ms
        collector.record_request_duration(Duration::from_millis(25)); // < 50ms
        collector.record_request_duration(Duration::from_millis(75)); // < 100ms
        collector.record_request_duration(Duration::from_millis(200)); // < 500ms
        collector.record_request_duration(Duration::from_millis(800)); // < 1s
        collector.record_request_duration(Duration::from_secs(2)); // < 5s
        collector.record_request_duration(Duration::from_secs(10)); // >= 5s

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.duration_histogram[0], 1); // < 1ms
        assert_eq!(snapshot.duration_histogram[1], 1); // < 5ms
        assert_eq!(snapshot.duration_histogram[2], 1); // < 10ms
        assert_eq!(snapshot.duration_histogram[3], 1); // < 50ms
        assert_eq!(snapshot.duration_histogram[4], 1); // < 100ms
        assert_eq!(snapshot.duration_histogram[5], 1); // < 500ms
        assert_eq!(snapshot.duration_histogram[6], 1); // < 1s
        assert_eq!(snapshot.duration_histogram[7], 1); // < 5s
        assert_eq!(snapshot.duration_histogram[8], 1); // >= 5s
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
        assert!(prometheus.contains("proxy_requests_total"));
        assert!(prometheus.contains("proxy_responses_by_status_total"));
        assert!(prometheus.contains("proxy_request_duration_ms"));
        assert!(prometheus.contains("proxy_success_rate"));

        // Verify values are correct
        assert!(prometheus.contains("proxy_requests_total 100"));
        assert!(prometheus.contains("proxy_responses_by_status_total{status=\"2xx\"} 100"));
        assert!(prometheus.contains("proxy_responses_by_status_total{status=\"4xx\"} 1"));
        assert!(prometheus.contains("proxy_responses_by_status_total{status=\"5xx\"} 1"));
    }

    #[test]
    fn test_concurrent_metrics_updates() {
        let collector = Arc::new(MetricsCollector::new());
        let mut handles = vec![];

        // Spawn multiple threads updating metrics concurrently
        for _ in 0..4 {
            let collector_clone: Arc<MetricsCollector> = Arc::clone(&collector);
            let handle = std::thread::spawn(move || {
                for _ in 0..250 {
                    collector_clone.record_request();
                    collector_clone.record_response(200);
                    collector_clone.record_request_duration(Duration::from_millis(5));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.total_requests, 1000);
        assert_eq!(snapshot.total_responses, 1000);
        assert_eq!(snapshot.status_2xx, 1000);
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.duration_histogram[1], 1000); // All in < 5ms bucket
    }
}

/// Tests for error handling and classification functionality
///
/// These tests verify that errors are created correctly, classified properly,
/// and converted to appropriate HTTP status codes.
mod error_tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let config_err = ProxyError::configuration("Invalid config", None);
        assert!(matches!(config_err, ProxyError::Configuration { .. }));

        let network_err = ProxyError::network("127.0.0.1:8080", "Connection refused", None);
        assert!(matches!(network_err, ProxyError::Network { .. }));

        let backend_err = ProxyError::backend("api.example.com", 500, "Internal error");
        assert!(matches!(backend_err, ProxyError::Backend { .. }));

        let timeout_err = ProxyError::timeout(Duration::from_secs(30), "connect");
        assert!(matches!(timeout_err, ProxyError::Timeout { .. }));

        let resource_err = ProxyError::resource_exhausted("memory", "OOM");
        assert!(matches!(resource_err, ProxyError::ResourceExhausted { .. }));

        let validation_err =
            ProxyError::request_validation("Bad header", Some("POST /api".to_string()));
        assert!(matches!(
            validation_err,
            ProxyError::RequestValidation { .. }
        ));

        let internal_err = ProxyError::internal("Unexpected error", None);
        assert!(matches!(internal_err, ProxyError::Internal { .. }));

        let unavailable_err = ProxyError::service_unavailable("Maintenance", Some(300));
        assert!(matches!(
            unavailable_err,
            ProxyError::ServiceUnavailable { .. }
        ));
    }

    #[test]
    fn test_http_status_mapping() {
        assert_eq!(
            ProxyError::configuration("test", None).to_http_status(),
            500
        );
        assert_eq!(
            ProxyError::network("host", "error", None).to_http_status(),
            502
        );
        assert_eq!(
            ProxyError::backend("host", 404, "not found").to_http_status(),
            404
        );
        assert_eq!(ProxyError::backend("host", 200, "ok").to_http_status(), 502); // Invalid success code mapped to 502
        assert_eq!(
            ProxyError::backend("host", 999, "invalid").to_http_status(),
            502
        ); // Invalid code mapped to 502
        assert_eq!(
            ProxyError::timeout(Duration::from_secs(1), "op").to_http_status(),
            504
        );
        assert_eq!(
            ProxyError::resource_exhausted("mem", "oom").to_http_status(),
            503
        );
        assert_eq!(
            ProxyError::request_validation("bad", None).to_http_status(),
            400
        );
        assert_eq!(ProxyError::internal("bug", None).to_http_status(), 500);
        assert_eq!(
            ProxyError::service_unavailable("down", None).to_http_status(),
            503
        );
    }

    #[test]
    fn test_temporary_error_classification() {
        // Temporary errors (retriable)
        assert!(ProxyError::network("host", "connection failed", None).is_temporary());
        assert!(ProxyError::backend("host", 500, "server error").is_temporary());
        assert!(ProxyError::backend("host", 502, "bad gateway").is_temporary());
        assert!(ProxyError::backend("host", 503, "unavailable").is_temporary());
        assert!(ProxyError::timeout(Duration::from_secs(1), "op").is_temporary());
        assert!(ProxyError::resource_exhausted("connections", "pool full").is_temporary());
        assert!(ProxyError::service_unavailable("maintenance", Some(60)).is_temporary());

        // Permanent errors (not retriable)
        assert!(!ProxyError::configuration("invalid", None).is_temporary());
        assert!(!ProxyError::backend("host", 400, "bad request").is_temporary());
        assert!(!ProxyError::backend("host", 401, "unauthorized").is_temporary());
        assert!(!ProxyError::backend("host", 404, "not found").is_temporary());
        assert!(!ProxyError::request_validation("bad header", None).is_temporary());
        assert!(!ProxyError::internal("logic error", None).is_temporary());
    }

    #[test]
    fn test_error_display() {
        let config_err = ProxyError::configuration("Invalid listen address", None);
        assert!(config_err.to_string().contains("Configuration error"));
        assert!(config_err.to_string().contains("Invalid listen address"));

        let network_err = ProxyError::network("192.168.1.1:8080", "Connection timeout", None);
        assert!(network_err.to_string().contains("Network error"));
        assert!(network_err.to_string().contains("192.168.1.1:8080"));
        assert!(network_err.to_string().contains("Connection timeout"));

        let backend_err = ProxyError::backend("api.example.com:443", 404, "Not Found");
        assert!(backend_err.to_string().contains("Backend error"));
        assert!(backend_err.to_string().contains("api.example.com:443"));
        assert!(backend_err.to_string().contains("HTTP 404"));
        assert!(backend_err.to_string().contains("Not Found"));

        let timeout_err = ProxyError::timeout(Duration::from_millis(5000), "database query");
        assert!(timeout_err.to_string().contains("Operation timed out"));
        assert!(timeout_err.to_string().contains("5000ms"));
        assert!(timeout_err.to_string().contains("database query"));
    }

    #[test]
    fn test_error_source_chain() {
        let io_error = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
        let proxy_error = ProxyError::network(
            "127.0.0.1:8080",
            "Failed to connect",
            Some(Box::new(io_error)),
        );

        assert!(proxy_error.source().is_some());

        let config_error = ProxyError::configuration("Invalid port", None);
        assert!(config_error.source().is_none());
    }

    #[test]
    fn test_error_conversion_from_io() {
        let timeout_io_error = std::io::Error::from(std::io::ErrorKind::TimedOut);
        let proxy_error: ProxyError = timeout_io_error.into();
        assert!(matches!(proxy_error, ProxyError::Timeout { .. }));

        let connection_io_error = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
        let proxy_error: ProxyError = connection_io_error.into();
        assert!(matches!(proxy_error, ProxyError::Network { .. }));

        let not_found_io_error = std::io::Error::from(std::io::ErrorKind::NotFound);
        let proxy_error: ProxyError = not_found_io_error.into();
        assert!(matches!(proxy_error, ProxyError::Network { .. }));

        let other_io_error = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let proxy_error: ProxyError = other_io_error.into();
        assert!(matches!(proxy_error, ProxyError::Internal { .. }));
    }

    #[test]
    fn test_error_conversion_from_addr_parse() {
        let addr_parse_error: std::net::AddrParseError = "invalid_address"
            .parse::<std::net::SocketAddr>()
            .unwrap_err();
        let proxy_error: ProxyError = addr_parse_error.into();
        assert!(matches!(proxy_error, ProxyError::Configuration { .. }));
        assert!(proxy_error.to_string().contains("Invalid network address"));
    }
}

/// Tests for server lifecycle and management functionality
///
/// These tests verify server creation, configuration, lifecycle management,
/// and proper resource cleanup.
mod server_tests {
    use super::*;

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
}

/// Tests for proxy service functionality
///
/// These tests verify proxy service creation, configuration handling,
/// and basic service operations.
mod service_tests {
    use super::*;

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
}

/// Integration tests for component interactions
///
/// These tests verify that different components work together correctly
/// and that the overall system behaves as expected.
mod integration_tests {
    use super::*;

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
        assert!(matches!(error, ProxyError::Configuration { .. }));
        assert_eq!(error.to_http_status(), 500);
        assert!(!error.is_temporary());
    }

    #[tokio::test]
    async fn test_environment_config_integration() {
        // Test environment variable configuration
        std::env::set_var("PINGORA_LISTEN_ADDR", "127.0.0.1:8888");
        std::env::set_var("PINGORA_MAX_CONNECTIONS", "15000");

        let config = ProxyConfig::from_env().unwrap();
        let server = ProxyServer::new(config).await.unwrap();

        assert_eq!(server.local_addr().port(), 8888);
        assert_eq!(server.config().max_connections, 15000);

        // Clean up
        std::env::remove_var("PINGORA_LISTEN_ADDR");
        std::env::remove_var("PINGORA_MAX_CONNECTIONS");
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
}

/// Performance regression tests
///
/// These tests verify that performance characteristics remain within
/// acceptable bounds and catch performance regressions.
mod performance_tests {
    use super::*;
    use std::time::Instant;

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
            let _error = ProxyError::network(
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

        for i in 0..10 {
            let config = ProxyConfig {
                listen_addr: format!("127.0.0.1:{}", 8000 + i).parse().unwrap(),
                backend_addr: format!("127.0.0.1:{}", 3000 + i).parse().unwrap(),
                ..Default::default()
            };

            let _server = ProxyServer::new(config).await.unwrap();
        }

        let duration = start.elapsed();
        assert!(
            duration < Duration::from_millis(500),
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
}
