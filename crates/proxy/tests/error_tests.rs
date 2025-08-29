//! Error Tests
//!
//! Tests for error handling and classification functionality.
//! These tests verify that errors are created correctly, classified properly,
//! and converted to appropriate HTTP status codes.

use inferno_shared::InfernoError;
use std::error::Error;
use std::time::Duration;

#[test]
fn test_error_creation() {
    let config_err = InfernoError::configuration("Invalid config", None);
    assert!(matches!(config_err, InfernoError::Configuration { .. }));

    let network_err = InfernoError::network("127.0.0.1:8080", "Connection refused", None);
    assert!(matches!(network_err, InfernoError::Network { .. }));

    let backend_err = InfernoError::backend("api.example.com", 500, "Internal error");
    assert!(matches!(backend_err, InfernoError::Backend { .. }));

    let timeout_err = InfernoError::timeout(Duration::from_secs(30), "connect");
    assert!(matches!(timeout_err, InfernoError::Timeout { .. }));

    let resource_err = InfernoError::resource_exhausted("memory", "OOM");
    assert!(matches!(
        resource_err,
        InfernoError::ResourceExhausted { .. }
    ));

    let validation_err =
        InfernoError::request_validation("Bad header", Some("POST /api".to_string()));
    assert!(matches!(
        validation_err,
        InfernoError::RequestValidation { .. }
    ));

    let internal_err = InfernoError::internal("Unexpected error", None);
    assert!(matches!(internal_err, InfernoError::Internal { .. }));

    let unavailable_err = InfernoError::service_unavailable("Maintenance", Some(300));
    assert!(matches!(
        unavailable_err,
        InfernoError::ServiceUnavailable { .. }
    ));
}

#[test]
fn test_http_status_mapping() {
    assert_eq!(
        InfernoError::configuration("test", None).to_http_status(),
        500
    );
    assert_eq!(
        InfernoError::network("host", "error", None).to_http_status(),
        502
    );
    assert_eq!(
        InfernoError::backend("host", 404, "not found").to_http_status(),
        404
    );
    assert_eq!(
        InfernoError::backend("host", 200, "ok").to_http_status(),
        502
    ); // Invalid success code mapped to 502
    assert_eq!(
        InfernoError::backend("host", 999, "invalid").to_http_status(),
        502
    ); // Invalid code mapped to 502
    assert_eq!(
        InfernoError::timeout(Duration::from_secs(1), "op").to_http_status(),
        504
    );
    assert_eq!(
        InfernoError::resource_exhausted("mem", "oom").to_http_status(),
        503
    );
    assert_eq!(
        InfernoError::request_validation("bad", None).to_http_status(),
        400
    );
    assert_eq!(InfernoError::internal("bug", None).to_http_status(), 500);
    assert_eq!(
        InfernoError::service_unavailable("down", None).to_http_status(),
        503
    );
}

#[test]
fn test_temporary_error_classification() {
    // Temporary errors (retriable)
    assert!(InfernoError::network("host", "connection failed", None).is_temporary());
    assert!(InfernoError::backend("host", 500, "server error").is_temporary());
    assert!(InfernoError::backend("host", 502, "bad gateway").is_temporary());
    assert!(InfernoError::backend("host", 503, "unavailable").is_temporary());
    assert!(InfernoError::timeout(Duration::from_secs(1), "op").is_temporary());
    assert!(InfernoError::resource_exhausted("connections", "pool full").is_temporary());
    assert!(InfernoError::service_unavailable("maintenance", Some(60)).is_temporary());

    // Permanent errors (not retriable)
    assert!(!InfernoError::configuration("invalid", None).is_temporary());
    assert!(!InfernoError::backend("host", 400, "bad request").is_temporary());
    assert!(!InfernoError::backend("host", 401, "unauthorized").is_temporary());
    assert!(!InfernoError::backend("host", 404, "not found").is_temporary());
    assert!(!InfernoError::request_validation("bad header", None).is_temporary());
    assert!(!InfernoError::internal("logic error", None).is_temporary());
}

#[test]
fn test_error_display() {
    let config_err = InfernoError::configuration("Invalid listen address", None);
    assert!(config_err.to_string().contains("Configuration error"));
    assert!(config_err.to_string().contains("Invalid listen address"));

    let network_err = InfernoError::network("192.168.1.1:8080", "Connection timeout", None);
    assert!(network_err.to_string().contains("Network error"));
    assert!(network_err.to_string().contains("192.168.1.1:8080"));
    assert!(network_err.to_string().contains("Connection timeout"));

    let backend_err = InfernoError::backend("api.example.com:443", 404, "Not Found");
    assert!(backend_err.to_string().contains("Backend error"));
    assert!(backend_err.to_string().contains("api.example.com:443"));
    assert!(backend_err.to_string().contains("HTTP 404"));
    assert!(backend_err.to_string().contains("Not Found"));

    let timeout_err = InfernoError::timeout(Duration::from_millis(5000), "database query");
    assert!(timeout_err.to_string().contains("Operation timed out"));
    assert!(timeout_err.to_string().contains("5000ms"));
    assert!(timeout_err.to_string().contains("database query"));
}

#[test]
fn test_error_source_chain() {
    let io_error = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
    let proxy_error = InfernoError::network(
        "127.0.0.1:8080",
        "Failed to connect",
        Some(Box::new(io_error)),
    );

    assert!(proxy_error.source().is_some());

    let config_error = InfernoError::configuration("Invalid port", None);
    assert!(config_error.source().is_none());
}

#[test]
fn test_error_conversion_from_io() {
    let timeout_io_error = std::io::Error::from(std::io::ErrorKind::TimedOut);
    let proxy_error: InfernoError = timeout_io_error.into();
    assert!(matches!(proxy_error, InfernoError::Timeout { .. }));

    let connection_io_error = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
    let proxy_error: InfernoError = connection_io_error.into();
    assert!(matches!(proxy_error, InfernoError::Network { .. }));

    let not_found_io_error = std::io::Error::from(std::io::ErrorKind::NotFound);
    let proxy_error: InfernoError = not_found_io_error.into();
    assert!(matches!(proxy_error, InfernoError::Network { .. }));

    let other_io_error = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
    let proxy_error: InfernoError = other_io_error.into();
    assert!(matches!(proxy_error, InfernoError::Internal { .. }));
}

#[test]
fn test_error_conversion_from_addr_parse() {
    let addr_parse_error: std::net::AddrParseError = "invalid_address"
        .parse::<std::net::SocketAddr>()
        .unwrap_err();
    let proxy_error: InfernoError = addr_parse_error.into();
    assert!(matches!(proxy_error, InfernoError::Configuration { .. }));
    assert!(proxy_error.to_string().contains("Invalid network address"));
}
