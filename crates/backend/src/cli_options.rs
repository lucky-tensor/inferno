//! CLI options for the Inferno Backend
//!
//! This module defines the command-line interface options for the backend server,
//! which can be used both standalone and integrated into the unified CLI.

use crate::BackendConfig;
use clap::Parser;
use inferno_shared::{
    HealthCheckOptions, InfernoError, LoggingOptions, MetricsOptions, Result,
    ServiceDiscoveryOptions,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::{info, warn};

/// Inferno Backend - AI inference backend server
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct BackendCliOptions {
    /// Address to listen on
    #[arg(
        short,
        long,
        default_value = "127.0.0.1:3000",
        env = "INFERNO_BACKEND_LISTEN_ADDR"
    )]
    pub listen_addr: SocketAddr,

    /// Path to the AI model file
    #[arg(short, long, default_value = "model.bin", env = "INFERNO_MODEL_PATH")]
    pub model_path: PathBuf,

    /// Model type/format (e.g., llama, gguf, onnx)
    #[arg(long, default_value = "auto", env = "INFERNO_MODEL_TYPE")]
    pub model_type: String,

    /// Maximum batch size for inference
    #[arg(long, default_value_t = 32, env = "INFERNO_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// GPU device ID to use (-1 for CPU)
    #[arg(long, default_value_t = -1, env = "INFERNO_GPU_DEVICE_ID")]
    pub gpu_device_id: i32,

    /// Maximum context length
    #[arg(long, default_value_t = 2048, env = "INFERNO_MAX_CONTEXT_LENGTH")]
    pub max_context_length: usize,

    /// Memory pool size in MB
    #[arg(long, default_value_t = 1024, env = "INFERNO_MEMORY_POOL_MB")]
    pub memory_pool_mb: usize,

    /// Discovery load balancer addresses (comma-separated)
    #[arg(short = 'd', long, env = "INFERNO_DISCOVERY_LB")]
    pub discovery_lb: Option<String>,

    /// Enable request caching
    #[arg(long, default_value_t = true, env = "INFERNO_ENABLE_CACHE")]
    pub enable_cache: bool,

    /// Cache TTL in seconds
    #[arg(long, default_value_t = 3600, env = "INFERNO_CACHE_TTL_SECONDS")]
    pub cache_ttl_seconds: u64,

    #[command(flatten)]
    pub logging: LoggingOptions,

    #[command(flatten)]
    pub metrics: MetricsOptions,

    #[command(flatten)]
    pub health_check: HealthCheckOptions,

    #[command(flatten)]
    pub service_discovery: ServiceDiscoveryOptions,
}

impl BackendCliOptions {
    /// Run the backend server with the configured options
    pub async fn run(self) -> Result<()> {
        info!("Starting Inferno Backend");

        // Convert CLI options to BackendConfig
        let config = self.to_config()?;

        info!(
            listen_addr = %config.listen_addr,
            model_path = ?config.model_path,
            gpu_device_id = config.gpu_device_id,
            max_batch_size = config.max_batch_size,
            "Backend server starting"
        );

        // TODO: Implement actual backend server functionality
        // This would typically:
        // 1. Load the AI model
        // 2. Initialize the inference engine
        // 3. Start the HTTP server
        // 4. Register with service discovery
        // 5. Begin serving inference requests

        info!("Backend server is running");

        // Perform service registration if configured
        if let Some(registration_endpoint) = self.service_discovery.registration_endpoint.as_ref() {
            info!(
                "Attempting to register with service discovery at: {}",
                registration_endpoint
            );

            // Create registration manager
            let lb_addrs = self
                .discovery_lb
                .as_ref()
                .map(|s| {
                    s.split(',')
                        .filter_map(|addr| addr.trim().parse::<SocketAddr>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let registration =
                crate::registration::ServiceRegistration::new(self.listen_addr, lb_addrs);

            // Attempt registration
            if let Err(e) = registration.register().await {
                warn!("Failed to register with service discovery: {}", e);
            } else {
                info!("Successfully registered backend service");
            }
        }

        // Keep the server running until interrupted
        tokio::signal::ctrl_c()
            .await
            .map_err(|e| InfernoError::Configuration {
                message: format!("Failed to listen for shutdown signal: {}", e),
                source: None,
            })?;

        info!("Shutdown signal received, stopping backend server");
        Ok(())
    }

    /// Convert CLI options to BackendConfig
    fn to_config(&self) -> Result<BackendConfig> {
        let discovery_lb = self.discovery_lb.as_ref().map(|s| {
            s.split(',')
                .filter_map(|addr| addr.trim().parse::<SocketAddr>().ok())
                .collect::<Vec<_>>()
        });

        let registration_endpoint = self
            .service_discovery
            .registration_endpoint
            .as_ref()
            .and_then(|s| s.parse().ok());

        Ok(BackendConfig {
            listen_addr: self.listen_addr,
            model_path: self.model_path.clone(),
            model_type: self.model_type.clone(),
            max_batch_size: self.max_batch_size,
            gpu_device_id: self.gpu_device_id,
            max_context_length: self.max_context_length,
            memory_pool_mb: self.memory_pool_mb,
            discovery_lb: discovery_lb.unwrap_or_default(),
            enable_cache: self.enable_cache,
            cache_ttl_seconds: self.cache_ttl_seconds,
            enable_metrics: self.metrics.enable_metrics,
            metrics_addr: self.metrics.get_metrics_addr(9091),
            health_check_path: self.health_check.health_check_path.clone(),
            registration_endpoint,
            service_name: self.service_discovery.get_service_name("inferno-backend"),
        })
    }
}
