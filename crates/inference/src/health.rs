//! Health monitoring for Inferno backend

use crate::config::HealthConfig;
use crate::error::InfernoResult;
use crate::memory::{CudaMemoryPool, GpuAllocator, MemoryTracker};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    /// Service is operating normally
    Healthy,
    /// Service is operational but with degraded performance
    Degraded(String),
    /// Service is not operational
    Unhealthy(String),
}

/// Simple health checker trait for Inferno backend
#[async_trait]
pub trait HealthChecker: Send + Sync {
    /// Health status type returned by this checker
    type Status;

    /// Perform a health check
    async fn check_health(&self) -> InfernoResult<Self::Status>;

    /// Get the name of this health checker
    fn name(&self) -> &'static str;
}

/// Inferno-specific health checker
pub struct InfernoHealthChecker {
    config: HealthConfig,
    memory_pool: Option<Arc<CudaMemoryPool>>,
    memory_tracker: Option<Arc<MemoryTracker>>,
    last_check: std::sync::Mutex<Option<Instant>>,
}

impl InfernoHealthChecker {
    /// Create a new health checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HealthConfig::default(),
            memory_pool: None,
            memory_tracker: None,
            last_check: std::sync::Mutex::new(None),
        }
    }

    /// Create a new health checker with configuration
    #[must_use]
    pub const fn with_config(config: HealthConfig) -> Self {
        Self {
            config,
            memory_pool: None,
            memory_tracker: None,
            last_check: std::sync::Mutex::new(None),
        }
    }

    /// Set memory pool for health checking
    #[must_use]
    pub fn with_memory_pool(mut self, pool: Arc<CudaMemoryPool>) -> Self {
        self.memory_pool = Some(pool);
        self
    }

    /// Set memory tracker for health checking
    #[must_use]
    pub fn with_memory_tracker(mut self, tracker: Arc<MemoryTracker>) -> Self {
        self.memory_tracker = Some(tracker);
        self
    }
}

impl Default for InfernoHealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HealthChecker for InfernoHealthChecker {
    type Status = HealthStatus;

    async fn check_health(&self) -> InfernoResult<Self::Status> {
        if !self.config.enabled {
            return Ok(HealthStatus::Healthy);
        }

        // Update last check timestamp
        if let Ok(mut last_check) = self.last_check.lock() {
            *last_check = Some(Instant::now());
        }

        let mut issues = Vec::new();

        // Check memory usage if memory components are available
        if let Some(memory_tracker) = &self.memory_tracker {
            let stats = memory_tracker.get_stats();

            if stats.utilization_percentage > self.config.gpu_memory_threshold {
                issues.push(format!(
                    "Memory usage {:.1}% exceeds threshold {:.1}%",
                    stats.utilization_percentage * 100.0,
                    self.config.gpu_memory_threshold * 100.0
                ));
            }

            // Check for potential memory leaks (simplified)
            if stats.num_allocations > 10000 {
                issues.push(format!(
                    "High allocation count: {} (potential memory leak)",
                    stats.num_allocations
                ));
            }
        }

        // Perform basic memory pool health check
        if let Some(memory_pool) = &self.memory_pool {
            match tokio::time::timeout(
                Duration::from_millis(self.config.timeout_secs * 1000),
                memory_pool.get_stats(),
            )
            .await
            {
                Ok(Ok(_stats)) => {
                    // Memory pool is responsive
                }
                Ok(Err(e)) => {
                    issues.push(format!("Memory pool error: {e}"));
                }
                Err(_) => {
                    issues.push("Memory pool health check timeout".to_string());
                }
            }
        }

        // Simple inference latency test (placeholder)
        let inference_start = Instant::now();
        tokio::time::sleep(Duration::from_millis(1)).await; // Simulate minimal inference
        #[allow(clippy::cast_precision_loss)] // Duration conversion for metrics
        let inference_latency = inference_start.elapsed().as_millis() as f64;

        if inference_latency > self.config.inference_latency_threshold_ms {
            issues.push(format!(
                "Inference latency {:.1}ms exceeds threshold {:.1}ms",
                inference_latency, self.config.inference_latency_threshold_ms
            ));
        }

        // Determine overall health status
        match issues.len() {
            0 => Ok(HealthStatus::Healthy),
            1..=2 => Ok(HealthStatus::Degraded(issues.join("; "))),
            _ => Ok(HealthStatus::Unhealthy(issues.join("; "))),
        }
    }

    fn name(&self) -> &'static str {
        "inferno-backend"
    }
}

impl InfernoHealthChecker {
    /// Get the last health check timestamp
    pub fn last_check_time(&self) -> Option<Instant> {
        self.last_check.lock().ok().and_then(|guard| *guard)
    }

    /// Check if health checking is enabled
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get health configuration
    pub const fn config(&self) -> &HealthConfig {
        &self.config
    }
}

/// Health check result with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall health status
    pub status: HealthStatus,
    /// Timestamp of the check
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration of the health check in milliseconds
    pub check_duration_ms: u64,
    /// Optional detailed metrics
    pub metrics: Option<HealthMetrics>,
}

/// Detailed health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Inference latency in milliseconds
    pub inference_latency_ms: f64,
    /// Whether GPU is available
    pub gpu_available: bool,
}

impl InfernoHealthChecker {
    /// Perform detailed health check with metrics
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub async fn check_health_detailed(&self) -> InfernoResult<HealthCheckResult> {
        let start_time = Instant::now();
        let status = self.check_health().await?;
        let duration = start_time.elapsed();

        let metrics = self.memory_tracker.as_ref().map(|memory_tracker| {
            let stats = memory_tracker.get_stats();
            HealthMetrics {
                memory_utilization: stats.utilization_percentage,
                active_allocations: stats.num_allocations,
                inference_latency_ms: 1.0, // Placeholder
                gpu_available: stats.device_id >= 0,
            }
        });

        Ok(HealthCheckResult {
            status,
            timestamp: chrono::Utc::now(),
            check_duration_ms: duration.as_millis() as u64,
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{CudaMemoryPool, DeviceMemory, MemoryTracker};
    use std::sync::Arc;

    #[test]
    fn test_health_checker_creation() {
        let checker = InfernoHealthChecker::new();
        assert_eq!(checker.name(), "inferno-backend");
        assert!(checker.is_enabled()); // Default should be enabled
    }

    #[test]
    fn test_health_checker_with_config() {
        let config = HealthConfig {
            enabled: false,
            ..Default::default()
        };

        let checker = InfernoHealthChecker::with_config(config);
        assert!(!checker.is_enabled());
    }

    #[tokio::test]
    async fn test_basic_health_check() {
        let checker = InfernoHealthChecker::new();
        let result = checker.check_health().await;
        assert!(result.is_ok());

        match result.unwrap() {
            HealthStatus::Healthy => {}
            _ => panic!("Expected healthy status with default checker"),
        }
    }

    #[tokio::test]
    async fn test_health_check_with_disabled() {
        let config = HealthConfig {
            enabled: false,
            ..Default::default()
        };

        let checker = InfernoHealthChecker::with_config(config);
        let result = checker.check_health().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), HealthStatus::Healthy); // Should return healthy when disabled
    }

    #[tokio::test]
    async fn test_health_check_with_memory_components() {
        let memory_pool = Arc::new(CudaMemoryPool::new(0).unwrap());
        let memory_tracker = Arc::new(MemoryTracker::new(0));

        let checker = InfernoHealthChecker::new()
            .with_memory_pool(memory_pool)
            .with_memory_tracker(memory_tracker);

        let result = checker.check_health().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_detailed_health_check() {
        let memory_tracker = Arc::new(MemoryTracker::new(0));

        let checker = InfernoHealthChecker::new().with_memory_tracker(memory_tracker);

        let result = checker.check_health_detailed().await;
        assert!(result.is_ok());

        let health_result = result.unwrap();
        assert!(health_result.check_duration_ms < 1000); // Should be fast
        assert!(health_result.metrics.is_some());

        let metrics = health_result.metrics.unwrap();
        assert!(!metrics.gpu_available); // CPU mode
    }

    #[tokio::test]
    async fn test_memory_threshold_check() {
        let memory_tracker = Arc::new(MemoryTracker::new(0));

        // Simulate high memory usage by tracking a large allocation
        let large_memory = DeviceMemory {
            ptr: std::ptr::null_mut(),
            size: 8 * 1024 * 1024 * 1024, // 8GB - will exceed threshold
            device_id: 0,
            allocation_id: 1,
        };
        memory_tracker.track_allocation(&large_memory);

        let config = HealthConfig {
            gpu_memory_threshold: 0.5, // 50% threshold
            ..Default::default()
        };

        let checker = InfernoHealthChecker::with_config(config).with_memory_tracker(memory_tracker);

        let result = checker.check_health().await;
        assert!(result.is_ok());

        // Should be degraded due to high memory usage
        match result.unwrap() {
            HealthStatus::Degraded(msg) => {
                assert!(msg.contains("Memory usage"));
            }
            _ => panic!("Expected degraded status due to high memory usage"),
        }
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus::Healthy;
        let json = serde_json::to_string(&status);
        assert!(json.is_ok());

        let deserialized: HealthStatus = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(deserialized, HealthStatus::Healthy);

        let status = HealthStatus::Degraded("test issue".to_string());
        let json = serde_json::to_string(&status);
        assert!(json.is_ok());
    }
}
