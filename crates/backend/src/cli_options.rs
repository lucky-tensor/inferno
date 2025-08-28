//! CLI options for the Inferno Backend
//!
//! This module defines the command-line interface options for the backend server,
//! which can be used both standalone and integrated into the unified CLI.

use crate::BackendConfig;
use clap::Parser;
use inferno_shared::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::info;

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

    /// Logging level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info", env = "INFERNO_LOG_LEVEL")]
    pub log_level: String,

    /// Enable metrics collection
    #[arg(long, default_value_t = true, env = "INFERNO_ENABLE_METRICS")]
    pub enable_metrics: bool,

    /// Metrics server address
    #[arg(long, default_value = "127.0.0.1:9091", env = "INFERNO_METRICS_ADDR")]
    pub metrics_addr: SocketAddr,

    /// Health check endpoint path
    #[arg(long, default_value = "/health", env = "INFERNO_HEALTH_CHECK_PATH")]
    pub health_check_path: String,

    /// Registration endpoint for service discovery
    #[arg(long, env = "INFERNO_REGISTRATION_ENDPOINT")]
    pub registration_endpoint: Option<String>,

    /// Service name for registration
    #[arg(long, default_value = "inferno-backend", env = "INFERNO_SERVICE_NAME")]
    pub service_name: String,
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

        info!("Backend server is running (placeholder implementation)");

        // For now, just sleep to keep the process running
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

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
            enable_metrics: self.enable_metrics,
            metrics_addr: self.metrics_addr,
            health_check_path: self.health_check_path.clone(),
            registration_endpoint,
            service_name: self.service_name.clone(),
        })
    }
}
