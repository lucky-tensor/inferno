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
    /// Cost alert threshold in USD
    pub cost_alert_threshold: f64,
    /// Cost optimization check interval in seconds
    pub optimization_interval: u64,
    /// Maximum allowed instances per region
    pub max_instances_per_region: usize,
    /// Auto-scaling enabled
    pub enable_autoscaling: bool,
    /// Minimum instances to maintain
    pub min_instances: usize,
    /// Maximum instances allowed
    pub max_instances: usize,
    /// Scale up threshold (CPU percentage)
    pub scale_up_threshold: f64,
    /// Scale down threshold (CPU percentage)
    pub scale_down_threshold: f64,
    /// Metrics endpoint for monitoring
    pub metrics_endpoint: Option<SocketAddr>,
    /// Alert webhook URL
    pub alert_webhook: Option<SocketAddr>,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics server address
    pub metrics_addr: SocketAddr,
    /// Health check endpoint path
    pub health_check_path: String,
    /// Budget enforcement mode
    pub budget_mode: String,
    /// Monthly budget limit in USD
    pub monthly_budget: f64,
}

impl Default for GovernatorConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:4000".parse().unwrap(),
            database_url: "sqlite://governator.db".to_string(),
            providers: vec!["aws".to_string(), "gcp".to_string()],
            cost_alert_threshold: 1000.0,
            optimization_interval: 300,
            max_instances_per_region: 100,
            enable_autoscaling: true,
            min_instances: 1,
            max_instances: 100,
            scale_up_threshold: 80.0,
            scale_down_threshold: 20.0,
            metrics_endpoint: None,
            alert_webhook: None,
            enable_metrics: true,
            metrics_addr: "127.0.0.1:9092".parse().unwrap(),
            health_check_path: "/health".to_string(),
            budget_mode: "warn".to_string(),
            monthly_budget: 10000.0,
        }
    }
}

impl GovernatorConfig {
    pub fn from_env() -> Result<Self> {
        // TODO: Implement proper environment variable loading
        Ok(Self::default())
    }
}
