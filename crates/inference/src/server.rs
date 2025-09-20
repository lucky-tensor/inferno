//! HTTP server for Inferno backend

use crate::config::{InfernoConfig, ServerConfig};
use crate::error::InfernoResult;

/// Inferno HTTP server
pub struct InfernoServer {
    config: ServerConfig,
}

impl InfernoServer {
    /// Create a new server
    #[must_use]
    pub fn new(config: InfernoConfig) -> Self {
        Self {
            config: config.server,
        }
    }

    /// Start the server
    pub fn start(&self) -> InfernoResult<()> {
        // TODO: Implement HTTP server with axum
        // - /v1/completions endpoint
        // - /v1/chat/completions endpoint
        // - /health endpoint
        // - /metrics endpoint
        tracing::info!("Inferno server would start on {}", self.config.host);
        Ok(())
    }

    /// Stop the server
    pub fn stop(&self) -> InfernoResult<()> {
        tracing::info!("Inferno server stopping");
        Ok(())
    }

    /// Get server configuration
    #[must_use]
    pub const fn config(&self) -> &ServerConfig {
        &self.config
    }
}
