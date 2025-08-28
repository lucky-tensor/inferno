//! CLI options for the Inferno Governator
//!
//! This module defines the command-line interface options for the governator server,
//! which can be used both standalone and integrated into the unified CLI.

use crate::GovernatorConfig;
use clap::Parser;
use inferno_shared::{HealthCheckOptions, LoggingOptions, MetricsOptions, Result};
use std::net::SocketAddr;
use tracing::info;

/// Inferno Governator - Cost optimization and resource governance server
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct GovernatorCliOptions {
    /// Address to listen on
    #[arg(
        short,
        long,
        default_value = "127.0.0.1:4000",
        env = "INFERNO_GOVERNATOR_LISTEN_ADDR"
    )]
    pub listen_addr: SocketAddr,

    /// Database connection URL
    #[arg(
        long,
        default_value = "sqlite://governator.db",
        env = "INFERNO_DATABASE_URL"
    )]
    pub database_url: String,

    /// Cloud providers to monitor (comma-separated: aws,gcp,azure)
    #[arg(short, long, default_value = "aws,gcp", env = "INFERNO_PROVIDERS")]
    pub providers: String,

    /// Cost alert threshold in USD
    #[arg(long, default_value_t = 1000.0, env = "INFERNO_COST_ALERT_THRESHOLD")]
    pub cost_alert_threshold: f64,

    /// Cost optimization check interval in seconds
    #[arg(long, default_value_t = 300, env = "INFERNO_OPTIMIZATION_INTERVAL")]
    pub optimization_interval: u64,

    /// Maximum allowed instances per region
    #[arg(long, default_value_t = 100, env = "INFERNO_MAX_INSTANCES_PER_REGION")]
    pub max_instances_per_region: usize,

    /// Auto-scaling enabled
    #[arg(long, default_value_t = true, env = "INFERNO_ENABLE_AUTOSCALING")]
    pub enable_autoscaling: bool,

    /// Minimum instances to maintain
    #[arg(long, default_value_t = 1, env = "INFERNO_MIN_INSTANCES")]
    pub min_instances: usize,

    /// Maximum instances allowed
    #[arg(long, default_value_t = 100, env = "INFERNO_MAX_INSTANCES")]
    pub max_instances: usize,

    /// Scale up threshold (CPU percentage)
    #[arg(long, default_value_t = 80.0, env = "INFERNO_SCALE_UP_THRESHOLD")]
    pub scale_up_threshold: f64,

    /// Scale down threshold (CPU percentage)
    #[arg(long, default_value_t = 20.0, env = "INFERNO_SCALE_DOWN_THRESHOLD")]
    pub scale_down_threshold: f64,

    /// Metrics endpoint for monitoring
    #[arg(
        short,
        long,
        default_value = "prometheus:9090",
        env = "INFERNO_METRICS_ENDPOINT"
    )]
    pub metrics_endpoint: String,

    /// Alert webhook URL
    #[arg(long, env = "INFERNO_ALERT_WEBHOOK")]
    pub alert_webhook: Option<String>,

    #[command(flatten)]
    pub logging: LoggingOptions,

    #[command(flatten)]
    pub metrics: MetricsOptions,

    #[command(flatten)]
    pub health_check: HealthCheckOptions,

    /// Budget enforcement mode (warn, enforce, off)
    #[arg(long, default_value = "warn", env = "INFERNO_BUDGET_MODE")]
    pub budget_mode: String,

    /// Monthly budget limit in USD
    #[arg(long, default_value_t = 10000.0, env = "INFERNO_MONTHLY_BUDGET")]
    pub monthly_budget: f64,
}

impl GovernatorCliOptions {
    /// Run the governator server with the configured options
    pub async fn run(self) -> Result<()> {
        info!("Starting Inferno Governator");

        // Convert CLI options to GovernatorConfig
        let config = self.to_config()?;

        info!(
            listen_addr = %config.listen_addr,
            database_url = %config.database_url,
            providers = ?config.providers,
            cost_alert_threshold = config.cost_alert_threshold,
            budget_mode = %config.budget_mode,
            monthly_budget = config.monthly_budget,
            "Governator server starting"
        );

        // TODO: Implement actual governator server functionality
        // This would typically:
        // 1. Connect to the database
        // 2. Initialize cloud provider clients
        // 3. Start monitoring loops
        // 4. Launch the HTTP API server
        // 5. Begin cost optimization analysis

        info!("Governator server is running (placeholder implementation)");

        // For now, just sleep to keep the process running
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        Ok(())
    }

    /// Convert CLI options to GovernatorConfig
    fn to_config(&self) -> Result<GovernatorConfig> {
        let providers = self
            .providers
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let alert_webhook = self.alert_webhook.as_ref().and_then(|s| s.parse().ok());

        let metrics_endpoint = self.metrics_endpoint.parse().ok();

        Ok(GovernatorConfig {
            listen_addr: self.listen_addr,
            database_url: self.database_url.clone(),
            providers,
            cost_alert_threshold: self.cost_alert_threshold,
            optimization_interval: self.optimization_interval,
            max_instances_per_region: self.max_instances_per_region,
            enable_autoscaling: self.enable_autoscaling,
            min_instances: self.min_instances,
            max_instances: self.max_instances,
            scale_up_threshold: self.scale_up_threshold,
            scale_down_threshold: self.scale_down_threshold,
            metrics_endpoint,
            alert_webhook,
            enable_metrics: self.metrics.enable_metrics,
            metrics_addr: self.metrics.get_metrics_addr(9092),
            health_check_path: self.health_check.health_check_path.clone(),
            budget_mode: self.budget_mode.clone(),
            monthly_budget: self.monthly_budget,
        })
    }
}
