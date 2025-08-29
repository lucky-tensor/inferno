//! CLI Tests
//!
//! Tests for CLI options parsing and validation.

use inferno_shared::cli::{
    parse_socket_addrs, parse_string_list, LoggingOptions, MetricsOptions, ServiceDiscoveryOptions,
};
use tracing::Level;

#[test]
fn test_parse_log_level() {
    let opts = LoggingOptions {
        log_level: "debug".to_string(),
    };
    assert_eq!(opts.parse_log_level(), Level::DEBUG);

    let opts = LoggingOptions {
        log_level: "ERROR".to_string(),
    };
    assert_eq!(opts.parse_log_level(), Level::ERROR);

    let opts = LoggingOptions {
        log_level: "invalid".to_string(),
    };
    assert_eq!(opts.parse_log_level(), Level::INFO);
}

#[test]
fn test_get_metrics_addr() {
    let opts = MetricsOptions {
        enable_metrics: true,
        metrics_addr: Some("192.168.1.1:8080".parse().unwrap()),
    };
    assert_eq!(
        opts.get_metrics_addr(9090),
        "192.168.1.1:8080".parse().unwrap()
    );

    let opts = MetricsOptions {
        enable_metrics: true,
        metrics_addr: None,
    };
    assert_eq!(
        opts.get_metrics_addr(9090),
        "127.0.0.1:9090".parse().unwrap()
    );
}

#[test]
fn test_parse_socket_addrs() {
    let addrs = parse_socket_addrs("127.0.0.1:8080,192.168.1.1:9090");
    assert_eq!(addrs.len(), 2);
    assert_eq!(addrs[0], "127.0.0.1:8080".parse().unwrap());
    assert_eq!(addrs[1], "192.168.1.1:9090".parse().unwrap());

    let addrs = parse_socket_addrs("127.0.0.1:8080,invalid,192.168.1.1:9090");
    assert_eq!(addrs.len(), 2);
}

#[test]
fn test_parse_string_list() {
    let list = parse_string_list("aws,gcp,azure");
    assert_eq!(list, vec!["aws", "gcp", "azure"]);

    let list = parse_string_list(" aws , gcp , azure ");
    assert_eq!(list, vec!["aws", "gcp", "azure"]);

    let list = parse_string_list("aws,,gcp");
    assert_eq!(list, vec!["aws", "gcp"]);
}

#[test]
fn test_get_service_name() {
    let opts = ServiceDiscoveryOptions {
        service_name: Some("custom-service".to_string()),
        registration_endpoint: None,
    };
    assert_eq!(opts.get_service_name("default"), "custom-service");

    let opts = ServiceDiscoveryOptions {
        service_name: None,
        registration_endpoint: None,
    };
    assert_eq!(opts.get_service_name("default"), "default");
}
