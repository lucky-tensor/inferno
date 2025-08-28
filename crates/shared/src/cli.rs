//! Shared CLI functionality for all Inferno components
//!
//! This module provides common CLI options and utilities that are shared
//! across the proxy, backend, governator, and CLI components to reduce
//! code duplication and ensure consistency.

use clap::Args;
use std::net::SocketAddr;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// Common logging options shared across all components
#[derive(Args, Debug, Clone)]
pub struct LoggingOptions {
    /// Logging level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info", env = "INFERNO_LOG_LEVEL")]
    pub log_level: String,
}

/// Common metrics options shared across all components
#[derive(Args, Debug, Clone)]
pub struct MetricsOptions {
    /// Enable metrics collection
    #[arg(long, default_value_t = true, env = "INFERNO_ENABLE_METRICS")]
    pub enable_metrics: bool,

    /// Metrics server address
    #[arg(long, env = "INFERNO_METRICS_ADDR")]
    pub metrics_addr: Option<SocketAddr>,
}

/// Common health check options shared across all components
#[derive(Args, Debug, Clone)]
pub struct HealthCheckOptions {
    /// Health check endpoint path
    #[arg(long, default_value = "/health", env = "INFERNO_HEALTH_CHECK_PATH")]
    pub health_check_path: String,
}

/// Common service discovery options
#[derive(Args, Debug, Clone)]
pub struct ServiceDiscoveryOptions {
    /// Service name for registration
    #[arg(long, env = "INFERNO_SERVICE_NAME")]
    pub service_name: Option<String>,

    /// Registration endpoint for service discovery
    #[arg(long, env = "INFERNO_REGISTRATION_ENDPOINT")]
    pub registration_endpoint: Option<String>,
}

impl LoggingOptions {
    /// Initialize logging with the configured level
    pub fn init_logging(&self) {
        let level = self.parse_log_level();

        let subscriber = FmtSubscriber::builder()
            .with_max_level(level)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set logging subscriber");
    }

    /// Parse the log level string into a tracing Level
    pub fn parse_log_level(&self) -> Level {
        match self.log_level.to_lowercase().as_str() {
            "error" => Level::ERROR,
            "warn" => Level::WARN,
            "info" => Level::INFO,
            "debug" => Level::DEBUG,
            "trace" => Level::TRACE,
            _ => Level::INFO,
        }
    }
}

impl MetricsOptions {
    /// Get the metrics address with component-specific defaults
    pub fn get_metrics_addr(&self, default_port: u16) -> SocketAddr {
        self.metrics_addr.unwrap_or_else(|| {
            format!("127.0.0.1:{}", default_port)
                .parse()
                .expect("Default metrics address should be valid")
        })
    }
}

impl ServiceDiscoveryOptions {
    /// Get the service name with a default fallback
    pub fn get_service_name(&self, default: &str) -> String {
        self.service_name
            .clone()
            .unwrap_or_else(|| default.to_string())
    }
}

/// Utility function to parse comma-separated socket addresses
pub fn parse_socket_addrs(input: &str) -> Vec<SocketAddr> {
    input
        .split(',')
        .filter_map(|addr| addr.trim().parse::<SocketAddr>().ok())
        .collect()
}

/// Utility function to parse comma-separated strings
pub fn parse_string_list(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
