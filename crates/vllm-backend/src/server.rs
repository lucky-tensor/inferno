//! HTTP server for VLLM backend

use crate::config::{ServerConfig, VLLMConfig};
use crate::error::VLLMResult;

/// VLLM HTTP server
pub struct VLLMServer {
    config: ServerConfig,
}

impl VLLMServer {
    /// Create a new server
    pub fn new(config: VLLMConfig) -> Self {
        Self {
            config: config.server,
        }
    }

    /// Start the server
    pub async fn start(&self) -> VLLMResult<()> {
        // TODO: Implement HTTP server with axum
        // - /v1/completions endpoint
        // - /v1/chat/completions endpoint
        // - /health endpoint
        // - /metrics endpoint
        tracing::info!("VLLM server would start on {}", self.config.host);
        Ok(())
    }

    /// Stop the server
    pub async fn stop(&self) -> VLLMResult<()> {
        tracing::info!("VLLM server stopping");
        Ok(())
    }

    /// Get server configuration
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }
}
