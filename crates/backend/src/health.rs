//! HTTP metrics server for backend services
//!
//! Provides HTTP endpoints for metrics monitoring and service discovery.
//! The server exposes NodeVitals data in JSON format following the
//! service discovery specification.

use inferno_shared::{MetricsCollector, MetricsServer, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};

/// HTTP metrics server for backend health monitoring
///
/// This service provides HTTP endpoints for monitoring backend health
/// and performance metrics. It serves the NodeVitals format required
/// for service discovery and load balancing.
///
/// ## Endpoints
///
/// - `GET /metrics` - Returns NodeVitals JSON with current service metrics
/// - `GET /health` - Simple health check endpoint returning 200 OK
///
/// ## Usage
///
/// The server is typically started as a background task during backend
/// initialization and runs for the lifetime of the service.
pub struct HealthService {
    /// Shared metrics collector for accessing performance data
    metrics: Arc<MetricsCollector>,
    /// Address to bind the HTTP metrics server
    metrics_addr: SocketAddr,
    /// Service name for NodeVitals responses
    service_name: String,
    /// Version string for NodeVitals responses
    version: String,
}

impl HealthService {
    /// Creates a new health service instance
    ///
    /// # Arguments
    ///
    /// * `metrics` - Shared metrics collector for accessing service data
    /// * `metrics_addr` - Address to bind the HTTP metrics server
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_backend::health::HealthService;
    /// use inferno_shared::MetricsCollector;
    /// use std::sync::Arc;
    ///
    /// let metrics = Arc::new(MetricsCollector::new());
    /// let addr = "127.0.0.1:9090".parse().unwrap();
    /// let health_service = HealthService::new(metrics, addr);
    /// ```
    pub fn new(metrics: Arc<MetricsCollector>, metrics_addr: SocketAddr) -> Self {
        Self {
            metrics,
            metrics_addr,
            service_name: "inferno-backend".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Creates a health service with custom service information
    ///
    /// # Arguments
    ///
    /// * `metrics` - Shared metrics collector for accessing service data
    /// * `metrics_addr` - Address to bind the HTTP metrics server
    /// * `service_name` - Custom service name for NodeVitals responses
    /// * `version` - Custom version string for NodeVitals responses
    pub fn with_service_info(
        metrics: Arc<MetricsCollector>,
        metrics_addr: SocketAddr,
        service_name: String,
        version: String,
    ) -> Self {
        Self {
            metrics,
            metrics_addr,
            service_name,
            version,
        }
    }

    /// Starts the HTTP metrics server
    ///
    /// This method starts the HTTP server and blocks until it's shut down
    /// or encounters an error. The server provides metrics endpoints for
    /// service discovery and monitoring.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the server shuts down gracefully, or an
    /// `InfernoError` if startup or runtime errors occur.
    ///
    /// # Blocking Behavior
    ///
    /// This method blocks the current async task until the server
    /// shuts down. For non-blocking operation, spawn in a separate task.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use inferno_backend::health::HealthService;
    /// use inferno_shared::MetricsCollector;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let metrics = Arc::new(MetricsCollector::new());
    ///     let addr = "127.0.0.1:9090".parse().unwrap();
    ///     let health_service = HealthService::new(metrics, addr);
    ///     
    ///     health_service.start().await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn start(&self) -> Result<()> {
        info!(
            metrics_addr = %self.metrics_addr,
            service_name = %self.service_name,
            version = %self.version,
            "Starting backend HTTP metrics server"
        );

        let server = MetricsServer::with_service_info(
            Arc::clone(&self.metrics),
            self.metrics_addr,
            self.service_name.clone(),
            self.version.clone(),
        );

        if let Err(e) = server.start().await {
            error!(
                error = %e,
                metrics_addr = %self.metrics_addr,
                "Backend HTTP metrics server failed"
            );
            return Err(e);
        }

        info!("Backend HTTP metrics server shut down");
        Ok(())
    }

    /// Returns the configured metrics server address
    ///
    /// # Returns
    ///
    /// Returns the SocketAddr where the metrics server is bound
    pub fn metrics_addr(&self) -> SocketAddr {
        self.metrics_addr
    }

    /// Returns the service name used in NodeVitals responses
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Returns the version string used in NodeVitals responses
    pub fn version(&self) -> &str {
        &self.version
    }
}
