//! Health checking and monitoring for service discovery
//!
//! This module provides health checking functionality for service discovery,
//! including HTTP health checks, vitals monitoring, and health status tracking.

use super::errors::ServiceDiscoveryResult;
use super::types::NodeInfo;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

/// Health check result for individual backends
///
/// This enum represents the possible outcomes of a health check operation
/// on a backend node.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthCheckResult {
    /// Backend is healthy and ready to serve requests
    Healthy,

    /// Backend is unhealthy and should not receive requests
    Unhealthy(String),

    /// Health check failed due to network or protocol error
    Failed(String),
}

impl HealthCheckResult {
    /// Returns whether this result indicates a healthy backend
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::HealthCheckResult;
    ///
    /// assert!(HealthCheckResult::Healthy.is_healthy());
    /// assert!(!HealthCheckResult::Unhealthy("error".to_string()).is_healthy());
    /// assert!(!HealthCheckResult::Failed("timeout".to_string()).is_healthy());
    /// ```
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Returns the error message if the result is not healthy
    ///
    /// # Returns
    ///
    /// Returns `None` for healthy results, `Some(message)` for unhealthy or failed results.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::HealthCheckResult;
    ///
    /// assert_eq!(HealthCheckResult::Healthy.error_message(), None);
    /// assert_eq!(
    ///     HealthCheckResult::Unhealthy("down".to_string()).error_message(),
    ///     Some("down")
    /// );
    /// ```
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Healthy => None,
            Self::Unhealthy(msg) | Self::Failed(msg) => Some(msg),
        }
    }
}

/// Node vitals and health information
///
/// This structure contains health and status information extracted from
/// a backend's metrics endpoint response.
///
/// # Health Determination
///
/// Health is determined by the `ready` flag in the metrics response.
/// Additional vitals provide context for monitoring and debugging.
///
/// # Performance Monitoring
///
/// CPU, memory, and request metrics enable capacity planning and
/// load balancing decisions based on backend performance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeVitals {
    /// Whether the backend is ready to serve requests
    pub ready: bool,

    /// CPU usage percentage (0.0 to 100.0)
    pub cpu_usage: Option<f64>,

    /// Memory usage percentage (0.0 to 100.0)  
    pub memory_usage: Option<f64>,

    /// Number of active requests being processed
    pub active_requests: Option<u64>,

    /// Average response time in milliseconds
    pub avg_response_time_ms: Option<f64>,

    /// Error rate as percentage (0.0 to 100.0)
    pub error_rate: Option<f64>,

    /// Custom status message from the backend
    pub status_message: Option<String>,
}

impl NodeVitals {
    /// Creates healthy node vitals with default values
    ///
    /// # Returns
    ///
    /// Returns NodeVitals with ready=true and no performance metrics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeVitals;
    ///
    /// let vitals = NodeVitals::healthy();
    /// assert!(vitals.ready);
    /// assert_eq!(vitals.cpu_usage, None);
    /// ```
    pub fn healthy() -> Self {
        Self {
            ready: true,
            cpu_usage: None,
            memory_usage: None,
            active_requests: None,
            avg_response_time_ms: None,
            error_rate: None,
            status_message: None,
        }
    }

    /// Creates unhealthy node vitals with a status message
    ///
    /// # Arguments
    ///
    /// * `message` - Status message describing why the node is unhealthy
    ///
    /// # Returns
    ///
    /// Returns NodeVitals with ready=false and the provided status message.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeVitals;
    ///
    /// let vitals = NodeVitals::unhealthy("Database connection failed");
    /// assert!(!vitals.ready);
    /// assert_eq!(vitals.status_message, Some("Database connection failed".to_string()));
    /// ```
    pub fn unhealthy(message: impl Into<String>) -> Self {
        Self {
            ready: false,
            cpu_usage: None,
            memory_usage: None,
            active_requests: None,
            avg_response_time_ms: None,
            error_rate: None,
            status_message: Some(message.into()),
        }
    }

    /// Returns whether the node is overloaded based on performance metrics
    ///
    /// # Returns
    ///
    /// Returns `true` if CPU usage > 90%, memory usage > 95%, or error rate > 10%.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeVitals;
    ///
    /// let mut vitals = NodeVitals::healthy();
    /// vitals.cpu_usage = Some(95.0);
    /// assert!(vitals.is_overloaded());
    ///
    /// vitals.cpu_usage = Some(50.0);
    /// assert!(!vitals.is_overloaded());
    /// ```
    pub fn is_overloaded(&self) -> bool {
        if let Some(cpu) = self.cpu_usage {
            if cpu > 90.0 {
                return true;
            }
        }

        if let Some(memory) = self.memory_usage {
            if memory > 95.0 {
                return true;
            }
        }

        if let Some(error_rate) = self.error_rate {
            if error_rate > 10.0 {
                return true;
            }
        }

        false
    }

    /// Calculates a health score from 0.0 to 1.0 based on vitals
    ///
    /// # Returns
    ///
    /// Returns a score where 1.0 is perfect health and 0.0 is completely unhealthy.
    /// Score considers ready state, CPU/memory usage, and error rate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeVitals;
    ///
    /// let healthy = NodeVitals::healthy();
    /// assert_eq!(healthy.health_score(), 1.0);
    ///
    /// let unhealthy = NodeVitals::unhealthy("down");
    /// assert_eq!(unhealthy.health_score(), 0.0);
    /// ```
    pub fn health_score(&self) -> f64 {
        if !self.ready {
            return 0.0;
        }

        let mut score = 1.0;

        // Reduce score based on CPU usage
        if let Some(cpu) = self.cpu_usage {
            score *= (100.0 - cpu) / 100.0;
        }

        // Reduce score based on memory usage
        if let Some(memory) = self.memory_usage {
            score *= (100.0 - memory) / 100.0;
        }

        // Reduce score based on error rate
        if let Some(error_rate) = self.error_rate {
            score *= (100.0 - error_rate) / 100.0;
        }

        score.clamp(0.0, 1.0)
    }
}

/// Trait for health checking implementations
///
/// This trait defines the interface for different health checking strategies.
/// Implementations can use HTTP, TCP, custom protocols, or mock checks
/// for different implementations (HTTP, custom protocols, etc.)
/// and easier testing with mock implementations.
#[async_trait]
pub trait HealthChecker: Send + Sync {
    /// Performs a health check on the specified node
    ///
    /// # Arguments
    ///
    /// * `node_info` - Information about the node to health check
    ///
    /// # Returns
    ///
    /// Returns the health check result indicating success, failure, or error.
    ///
    /// # Errors
    ///
    /// Returns a ServiceDiscoveryError if the health check operation fails
    /// due to network issues, invalid responses, or other errors.
    async fn check_health(&self, node_info: &NodeInfo)
        -> ServiceDiscoveryResult<HealthCheckResult>;

    /// Returns the name of this health checker implementation
    fn name(&self) -> &'static str;
}

/// HTTP-based health checker implementation
///
/// This implementation performs health checks by making HTTP GET requests
/// to the `/metrics` endpoint of backend nodes and parsing the JSON response.
#[derive(Debug)]
pub struct HttpHealthChecker {
    client: Client,
    timeout: Duration,
}

impl HttpHealthChecker {
    /// Creates a new HTTP health checker
    ///
    /// # Arguments
    ///
    /// * `timeout` - Timeout for health check requests
    ///
    /// # Returns
    ///
    /// Returns a new HttpHealthChecker with the specified timeout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{HttpHealthChecker, HealthChecker};
    /// use std::time::Duration;
    ///
    /// let checker = HttpHealthChecker::new(Duration::from_secs(5));
    /// assert_eq!(checker.name(), "http");
    /// ```
    pub fn new(timeout: Duration) -> Self {
        Self {
            client: Client::new(),
            timeout,
        }
    }
}

#[async_trait]
impl HealthChecker for HttpHealthChecker {
    async fn check_health(
        &self,
        node_info: &NodeInfo,
    ) -> ServiceDiscoveryResult<HealthCheckResult> {
        let url = node_info.metrics_url();

        debug!("Performing health check for {} at {}", node_info.id, url);

        match tokio::time::timeout(self.timeout, self.client.get(&url).send()).await {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    match response.json::<NodeVitals>().await {
                        Ok(vitals) => {
                            if vitals.ready {
                                Ok(HealthCheckResult::Healthy)
                            } else {
                                let message = vitals
                                    .status_message
                                    .unwrap_or_else(|| "Backend not ready".to_string());
                                Ok(HealthCheckResult::Unhealthy(message))
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse health response from {}: {}",
                                node_info.id, e
                            );
                            Ok(HealthCheckResult::Failed(format!(
                                "Invalid response format: {}",
                                e
                            )))
                        }
                    }
                } else {
                    Ok(HealthCheckResult::Unhealthy(format!(
                        "HTTP {}",
                        response.status()
                    )))
                }
            }
            Ok(Err(e)) => {
                debug!("Health check request failed for {}: {}", node_info.id, e);
                Ok(HealthCheckResult::Failed(format!("Request failed: {}", e)))
            }
            Err(_) => {
                debug!("Health check timeout for {}", node_info.id);
                Ok(HealthCheckResult::Failed("Request timeout".to_string()))
            }
        }
    }

    fn name(&self) -> &'static str {
        "http"
    }
}
