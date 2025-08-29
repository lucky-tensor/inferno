//! Error Tests
//!
//! Tests for error types, classification, and conversion.

use inferno_shared::error::{InfernoError, ProxyError};
use std::time::Duration;

#[test]
fn test_error_construction() {
    let config_err = InfernoError::configuration("test", None);
    assert!(matches!(config_err, InfernoError::Configuration { .. }));

    let network_err = InfernoError::network("127.0.0.1:8080", "connection refused", None);
    assert!(matches!(network_err, InfernoError::Network { .. }));

    let backend_err = InfernoError::backend("api.example.com", 500, "error");
    assert!(matches!(backend_err, InfernoError::Backend { .. }));

    let timeout_err = InfernoError::timeout(Duration::from_secs(30), "connect");
    assert!(matches!(timeout_err, InfernoError::Timeout { .. }));
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
        InfernoError::backend("host", 999, "invalid").to_http_status(),
        502
    );
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
fn test_temporary_classification() {
    assert!(!InfernoError::configuration("test", None).is_temporary());
    assert!(InfernoError::network("host", "error", None).is_temporary());
    assert!(!InfernoError::backend("host", 400, "bad request").is_temporary());
    assert!(InfernoError::backend("host", 500, "server error").is_temporary());
    assert!(InfernoError::timeout(Duration::from_secs(1), "op").is_temporary());
    assert!(InfernoError::resource_exhausted("mem", "oom").is_temporary());
    assert!(!InfernoError::request_validation("bad", None).is_temporary());
    assert!(!InfernoError::internal("bug", None).is_temporary());
    assert!(InfernoError::service_unavailable("down", None).is_temporary());
}

#[test]
fn test_error_conversion() {
    let addr_err: InfernoError = "invalid:address:format"
        .parse::<std::net::SocketAddr>()
        .unwrap_err()
        .into();
    assert!(matches!(addr_err, InfernoError::Configuration { .. }));

    let io_err: InfernoError = std::io::Error::from(std::io::ErrorKind::TimedOut).into();
    assert!(matches!(io_err, InfernoError::Timeout { .. }));
}

#[test]
fn test_proxy_error_alias() {
    let proxy_err: ProxyError = InfernoError::configuration("test", None);
    assert!(matches!(proxy_err, InfernoError::Configuration { .. }));
}
