//! Configuration Tests
//!
//! Tests for configuration management functionality.
//! These tests verify that configuration loading, validation, and
//! environment variable handling work correctly under various scenarios.

use inferno_proxy::{ProxyCliOptions, ProxyConfig};
use inferno_shared::{HealthCheckOptions, LoggingOptions, MetricsOptions};
use serial_test::serial;
use std::time::Duration;

use inferno_shared::test_utils::get_random_port_addr;

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
        if key.starts_with("INFERNO_") {
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
        if key.starts_with("INFERNO_") {
            std::env::remove_var(key);
        }
    }

    let listen_addr = get_random_port_addr();
    let backend_addr = get_random_port_addr();
    std::env::set_var("INFERNO_LISTEN_ADDR", listen_addr.to_string());
    std::env::set_var("INFERNO_BACKEND_ADDR", backend_addr.to_string());
    std::env::set_var("INFERNO_TIMEOUT_SECONDS", "60");
    std::env::set_var("INFERNO_MAX_CONNECTIONS", "5000");
    std::env::set_var("INFERNO_LOG_LEVEL", "debug");
    std::env::set_var(
        "INFERNO_BACKEND_SERVERS",
        "192.168.1.1:8080,192.168.1.2:8080",
    );

    let config = ProxyConfig::from_env().unwrap();

    assert_eq!(config.listen_addr, listen_addr);
    assert_eq!(config.backend_addr, backend_addr);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.max_connections, 5000);
    assert_eq!(config.log_level, "debug");
    assert_eq!(config.backend_servers.len(), 2);

    // Clean up
    std::env::remove_var("INFERNO_LISTEN_ADDR");
    std::env::remove_var("INFERNO_BACKEND_ADDR");
    std::env::remove_var("INFERNO_TIMEOUT_SECONDS");
    std::env::remove_var("INFERNO_MAX_CONNECTIONS");
    std::env::remove_var("INFERNO_LOG_LEVEL");
    std::env::remove_var("INFERNO_BACKEND_SERVERS");
}

#[test]
#[serial]
fn test_configuration_from_env_invalid_values() {
    // Clear any existing environment variables first
    for (key, _) in std::env::vars() {
        if key.starts_with("INFERNO_") {
            std::env::remove_var(key);
        }
    }

    std::env::set_var("INFERNO_LISTEN_ADDR", "invalid_address");

    let result = ProxyConfig::from_env();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid INFERNO_LISTEN_ADDR"));

    std::env::remove_var("INFERNO_LISTEN_ADDR");
}

#[test]
fn test_proxy_cli_to_config_conversion() {
    let listen_addr = get_random_port_addr();
    let backend_addr = get_random_port_addr();
    let operations_addr = get_random_port_addr();

    let cli = ProxyCliOptions {
        listen_addr,
        backend_addr: Some(backend_addr),
        backend_servers: Some("192.168.1.1:8080,192.168.1.2:8080".to_string()),
        max_connections: 5000,
        timeout_seconds: 60,
        enable_health_check: false,
        health_check_interval: 60,
        health_check: HealthCheckOptions {
            health_check_path: "/status".to_string(),
        },
        logging: LoggingOptions {
            log_level: "debug".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: Some(operations_addr),
            metrics_addr: None,
        },
        enable_tls: true,
        load_balancing_algorithm: "least_connections".to_string(),
    };

    let config = cli.to_config().unwrap();

    assert_eq!(config.listen_addr, listen_addr);
    assert_eq!(config.backend_addr, backend_addr);
    assert_eq!(config.backend_servers.len(), 2);
    assert_eq!(config.max_connections, 5000);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert!(!config.enable_health_check);
    assert_eq!(config.health_check_interval, Duration::from_secs(60));
    assert_eq!(config.health_check_path, "/status");
    assert_eq!(config.log_level, "debug");
    assert!(config.enable_metrics);
    assert_eq!(config.operations_addr, operations_addr);
    assert!(config.enable_tls);
    assert_eq!(config.load_balancing_algorithm, "least_connections");
}

#[test]
fn test_proxy_cli_to_config_with_defaults() {
    let listen_addr = get_random_port_addr();

    let cli = ProxyCliOptions {
        listen_addr,
        backend_addr: None,
        backend_servers: None,
        max_connections: 10000,
        timeout_seconds: 30,
        enable_health_check: true,
        health_check_interval: 30,
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: None,
            metrics_addr: None,
        },
        enable_tls: false,
        load_balancing_algorithm: "round_robin".to_string(),
    };

    let config = cli.to_config().unwrap();

    assert_eq!(config.listen_addr, listen_addr);
    assert_eq!(config.backend_addr, "127.0.0.1:3000".parse().unwrap()); // Default
    assert!(config.backend_servers.is_empty());
    assert_eq!(config.max_connections, 10000);
    assert_eq!(config.timeout, Duration::from_secs(30));
    assert!(config.enable_health_check);
    assert_eq!(config.health_check_interval, Duration::from_secs(30));
    assert_eq!(config.health_check_path, "/health");
    assert_eq!(config.log_level, "info");
    assert!(config.enable_metrics);
    assert!(!config.enable_tls);
    assert_eq!(config.load_balancing_algorithm, "round_robin");
}

#[tokio::test]
async fn test_proxy_cli_run_method() {
    let cli = ProxyCliOptions {
        listen_addr: get_random_port_addr(),
        backend_addr: Some(get_random_port_addr()),
        backend_servers: None,
        max_connections: 1000,
        timeout_seconds: 30,
        enable_health_check: true,
        health_check_interval: 30,
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: false, // Disable to avoid port conflicts
            operations_addr: None,
            metrics_addr: None,
        },
        enable_tls: false,
        load_balancing_algorithm: "round_robin".to_string(),
    };

    // Since run() starts a server and waits for connections, we need to test with a timeout
    let run_future = cli.run();

    // Test that it starts correctly (within a timeout)
    let result = tokio::time::timeout(std::time::Duration::from_millis(100), run_future).await;

    // The run should timeout (which means it started correctly and is waiting for connections)
    assert!(
        result.is_err(),
        "Proxy run method should timeout while waiting for connections"
    );
}
