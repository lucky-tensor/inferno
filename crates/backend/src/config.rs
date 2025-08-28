//! Backend configuration management

use inferno_shared::Result;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Configuration for the backend server
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendConfig {
    /// Address to bind the backend server to
    pub listen_addr: SocketAddr,
    /// Model to load for inference
    pub model_path: String,
    /// Maximum number of concurrent requests
    pub max_connections: u32,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:3000".parse().unwrap(),
            model_path: "model.bin".to_string(),
            max_connections: 1000,
        }
    }
}

impl BackendConfig {
    pub fn from_env() -> Result<Self> {
        // Placeholder implementation
        Ok(Self::default())
    }
}
