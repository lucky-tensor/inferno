//! # Inferno Shared Library
//!
//! Shared utilities, types, and protocols for Inferno distributed systems.
//! This crate provides common functionality used across all Inferno components
//! including error handling, metrics collection, and service discovery.
//!
//! ## Features
//!
//! - **Error Handling**: Comprehensive error types with HTTP status mapping
//! - **Metrics Collection**: High-performance, lock-free metrics system
//! - **Service Discovery**: Minimalist service registration protocol
//! - **Common Types**: Shared data structures and validation utilities
//!
//! ## Design Principles
//!
//! - **Performance First**: All operations optimized for distributed systems
//! - **Zero Allocation**: Hot paths avoid memory allocation where possible
//! - **Thread Safety**: All types are designed for concurrent access
//! - **Observability**: Built-in metrics and structured logging support

pub mod cli;
pub mod error;
pub mod metrics;
// TODO: Fix operations_server for hyper 1.x API changes
// pub mod operations_server;
pub mod service_discovery;

// Re-export commonly used types for convenience
pub use cli::{HealthCheckOptions, LoggingOptions, MetricsOptions, ServiceDiscoveryOptions};
pub use error::{InfernoError, ProxyError, Result};
pub use metrics::{MetricsCollector, MetricsSnapshot};
// TODO: Fix operations_server exports after hyper 1.x migration
// pub use operations_server::OperationsServer;
// Backward compatibility alias - provide stub until hyper 1.x migration complete

use std::net::SocketAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// Temporary stub for MetricsServer until hyper 1.x migration is complete
pub struct MetricsServer {
    connected_peers: Arc<AtomicUsize>,
}

impl MetricsServer {
    /// Stub for with_service_info
    pub fn with_service_info(
        _metrics: Arc<MetricsCollector>,
        _addr: SocketAddr,
        _service_name: String,
        _version: String,
    ) -> Self {
        MetricsServer {
            connected_peers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Stub for with_service_discovery (old API with 5 params for compatibility)
    #[allow(clippy::too_many_arguments)]
    pub fn with_service_discovery(
        _metrics: Arc<MetricsCollector>,
        _addr: SocketAddr,
        _service_name: String,
        _version: String,
        _discovery: Arc<ServiceDiscovery>,
    ) -> Self {
        MetricsServer {
            connected_peers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Stub for start
    pub async fn start(self) -> Result<()> {
        // TODO: Implement once operations_server is migrated to hyper 1.x
        Err(InfernoError::internal(
            "MetricsServer not yet implemented for hyper 1.x",
            None,
        ))
    }

    /// Stub for run
    pub async fn run(self) -> Result<()> {
        self.start().await
    }

    /// Stub for connected_peers_handle
    pub fn connected_peers_handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.connected_peers)
    }
}
pub use service_discovery::{
    AuthMode, BackendRegistration, HealthCheckResult, HealthChecker, HttpHealthChecker, NodeInfo,
    NodeType, NodeVitals, PeerInfo, ServiceDiscovery, ServiceDiscoveryConfig,
    ServiceDiscoveryError, ServiceDiscoveryResult,
};
