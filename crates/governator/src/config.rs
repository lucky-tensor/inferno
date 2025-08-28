//! Governator configuration management

use inferno_shared::Result;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Configuration for the governator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GovernatorConfig {
    /// Address to bind the governator API server to
    pub listen_addr: SocketAddr,
    /// Database connection string
    pub database_url: String,
    /// Cloud providers to monitor
    pub providers: Vec<String>,
}

impl Default for GovernatorConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:4000".parse().unwrap(),
            database_url: "postgresql://localhost/inferno".to_string(),
            providers: vec!["aws".to_string(), "gcp".to_string()],
        }
    }
}

impl GovernatorConfig {
    pub fn from_env() -> Result<Self> {
        // Placeholder implementation
        Ok(Self::default())
    }
}
