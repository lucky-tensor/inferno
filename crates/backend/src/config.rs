//! Backend configuration management

use inferno_shared::Result;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Configuration for the backend server
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendConfig {
    /// Address to bind the backend server to
    pub listen_addr: SocketAddr,
    /// Path to the AI model file
    pub model_path: PathBuf,
    /// Model type/format
    pub model_type: String,
    /// Maximum batch size for inference
    pub max_batch_size: usize,
    /// GPU device ID (-1 for CPU)
    pub gpu_device_id: i32,
    /// Maximum context length
    pub max_context_length: usize,
    /// Memory pool size in MB
    pub memory_pool_mb: usize,
    /// Discovery load balancer addresses
    pub discovery_lb: Vec<SocketAddr>,
    /// Enable request caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Operations server address (metrics, health, registration)
    pub operations_addr: SocketAddr,
    /// Health check endpoint path
    pub health_check_path: String,
    /// Registration endpoint for service discovery
    pub registration_endpoint: Option<SocketAddr>,
    /// Service name for registration
    pub service_name: String,
}

impl Default for BackendConfig {
    /// Creates a default backend configuration with sensible defaults
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:3000".parse().unwrap(),
            model_path: PathBuf::from("model.bin"),
            model_type: "auto".to_string(),
            max_batch_size: 32,
            gpu_device_id: -1,
            max_context_length: 2048,
            memory_pool_mb: 1024,
            discovery_lb: Vec::new(),
            enable_cache: true,
            cache_ttl_seconds: 3600,
            enable_metrics: true,
            operations_addr: "127.0.0.1:6100".parse().unwrap(),
            health_check_path: "/health".to_string(),
            registration_endpoint: None,
            service_name: "inferno-backend".to_string(),
        }
    }
}

impl BackendConfig {
    /// Creates configuration from environment variables, falling back to defaults
    pub fn from_env() -> Result<Self> {
        // TODO: Implement proper environment variable loading
        Ok(Self::default())
    }
}
