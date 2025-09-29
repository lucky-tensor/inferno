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
pub mod operations_server;
pub mod paths;
pub mod service_discovery;

pub mod test_utils;

// Re-export commonly used types for convenience
pub use cli::{HealthCheckOptions, LoggingOptions, MetricsOptions, ServiceDiscoveryOptions};
pub use error::{InfernoError, ProxyError, Result};
pub use metrics::{MetricsCollector, MetricsSnapshot};
pub use operations_server::OperationsServer;
// Backward compatibility alias
pub use operations_server::OperationsServer as MetricsServer;
pub use paths::{
    default_models_dir, default_models_dir_string, expand_home_dir, resolve_models_path,
};
pub use service_discovery::{
    AuthMode, BackendRegistration, HealthCheckResult, HealthChecker, HttpHealthChecker, NodeInfo,
    NodeType, NodeVitals, PeerInfo, ServiceDiscovery, ServiceDiscoveryConfig,
    ServiceDiscoveryError, ServiceDiscoveryResult,
};
