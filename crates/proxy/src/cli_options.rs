//! CLI options for the Inferno Proxy
//!
//! This module defines the command-line interface options for the proxy server,
//! which can be used both standalone and integrated into the unified CLI.

use crate::{ProxyConfig, ProxyServer};
use clap::Parser;
use inferno_shared::{HealthCheckOptions, InfernoError, LoggingOptions, MetricsOptions, Result};
use std::net::SocketAddr;
use std::time::Duration;
use tracing::{error, info};

/// Inferno Proxy - High-performance HTTP reverse proxy
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct ProxyCliOptions {
    /// Address to listen on
    #[arg(
        short,
        long,
        default_value = "127.0.0.1:8080",
        env = "INFERNO_LISTEN_ADDR"
    )]
    pub listen_addr: SocketAddr,

    /// Backend server address
    #[arg(short, long, env = "INFERNO_BACKEND_ADDR")]
    pub backend_addr: Option<SocketAddr>,

    /// Comma-separated list of backend servers for load balancing
    #[arg(short = 's', long, env = "INFERNO_BACKEND_SERVERS")]
    pub backend_servers: Option<String>,

    /// Maximum concurrent connections
    #[arg(long, default_value_t = 10000, env = "INFERNO_MAX_CONNECTIONS")]
    pub max_connections: usize,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 30, env = "INFERNO_TIMEOUT_SECONDS")]
    pub timeout_seconds: u64,

    /// Enable health checking for backends
    #[arg(long, default_value_t = true, env = "INFERNO_ENABLE_HEALTH_CHECK")]
    pub enable_health_check: bool,

    /// Health check interval in seconds
    #[arg(
        long,
        default_value_t = 30,
        env = "INFERNO_HEALTH_CHECK_INTERVAL_SECONDS"
    )]
    pub health_check_interval: u64,

    #[command(flatten)]
    pub health_check: HealthCheckOptions,

    #[command(flatten)]
    pub logging: LoggingOptions,

    #[command(flatten)]
    pub metrics: MetricsOptions,

    /// Enable TLS/SSL
    #[arg(long, default_value_t = false, env = "INFERNO_ENABLE_TLS")]
    pub enable_tls: bool,

    /// Load balancing algorithm (round_robin, least_connections, weighted)
    #[arg(
        long,
        default_value = "round_robin",
        env = "INFERNO_LOAD_BALANCING_ALGORITHM"
    )]
    pub load_balancing_algorithm: String,
}

impl ProxyCliOptions {
    /// Run the proxy server with the configured options
    pub async fn run(self) -> Result<()> {
        info!("Starting Inferno Proxy");

        // Convert CLI options to ProxyConfig
        let config = self.to_config()?;

        info!(
            listen_addr = %config.listen_addr,
            backend_addr = %config.backend_addr,
            backend_servers = ?config.backend_servers,
            max_connections = config.max_connections,
            health_check_enabled = config.enable_health_check,
            tls_enabled = config.enable_tls,
            metrics_enabled = config.enable_metrics,
            "Configuration loaded successfully"
        );

        // Create and run the proxy server
        let server = ProxyServer::new(config).await?;

        info!("Proxy server initialized successfully");
        info!("Server is ready to handle connections");

        // Run the server
        match server.run().await {
            Ok(_) => {
                info!("Server stopped normally");
                Ok(())
            }
            Err(e) => {
                error!(error = %e, "Server encountered an error");
                Err(e)
            }
        }
    }

    /// Convert CLI options to ProxyConfig
    pub fn to_config(&self) -> Result<ProxyConfig> {
        // Parse backend servers if provided
        let backend_servers = self
            .backend_servers
            .as_ref()
            .map(|s| {
                s.split(',')
                    .map(|addr| {
                        addr.trim()
                            .parse()
                            .map_err(|e| InfernoError::Configuration {
                                message: format!(
                                    "Invalid backend server address '{}': {}",
                                    addr, e
                                ),
                                source: None,
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        // Use backend_addr or default if not specified
        let backend_addr = self.backend_addr.unwrap_or_else(|| {
            "127.0.0.1:3000"
                .parse()
                .expect("Default backend address should be valid")
        });

        Ok(ProxyConfig {
            listen_addr: self.listen_addr,
            backend_addr,
            timeout: Duration::from_secs(self.timeout_seconds),
            max_connections: self.max_connections as u32,
            enable_health_check: self.enable_health_check,
            health_check_interval: Duration::from_secs(self.health_check_interval),
            health_check_path: self.health_check.health_check_path.clone(),
            health_check_timeout: Duration::from_secs(5), // Default timeout
            enable_tls: self.enable_tls,
            tls_cert_path: None, // Would need to add to CLI options if needed
            tls_key_path: None,  // Would need to add to CLI options if needed
            log_level: self.logging.log_level.clone(),
            enable_metrics: self.metrics.enable_metrics,
            operations_addr: self.metrics.get_operations_addr(6100),
            load_balancing_algorithm: self.load_balancing_algorithm.clone(),
            backend_servers,
            service_discovery_auth_mode: "open".to_string(),
            service_discovery_shared_secret: None,
        })
    }
}
