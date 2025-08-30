//! Service Discovery Module
//!
//! Minimalist service discovery protocol for Inferno distributed systems.
//! Implements the self-healing cloud pattern where backends register themselves
//! with load balancers and are monitored via their metrics endpoints.
//!
//! ## Protocol Overview
//!
//! 1. **Backend Registration**: Backends announce themselves to load balancers
//! 2. **Health Monitoring**: Load balancers check backend `/metrics` endpoints
//! 3. **Auto-Recovery**: Failed backends are removed, recovered backends re-register
//! 4. **Metrics-Based Health**: Health determined by `ready` flag in metrics response
//!
//! ## Design Principles
//!
//! - **Zero Dependencies**: No external service discovery required
//! - **High Performance**: Lock-free data structures, minimal overhead
//! - **Self-Healing**: Automatic failure detection and recovery
//! - **Simple Protocol**: JSON-based, human-readable
//! - **Fault Tolerant**: Graceful handling of network partitions
//!
//! ## Performance Characteristics
//!
//! - Backend registration: < 100ms
//! - Health check cycle: < 5s (configurable)
//! - Backend list access: < 1μs (lock-free reads)
//! - Memory overhead: < 1KB per backend
//!
//! ## Modular Architecture
//!
//! The service discovery system is organized into focused modules:
//! - `types`: Core data structures (NodeType, NodeInfo, PeerInfo, BackendRegistration)
//! - `auth`: Authentication modes (Open, SharedSecret) with Bearer token support
//! - `config`: Configuration structures with environment variable support
//! - `errors`: Specialized error types and comprehensive error handling
//! - `health`: Health checking, vitals monitoring, and status reporting
//! - `registration`: Enhanced registration protocol with peer information sharing
//! - `consensus`: Consensus algorithms for distributed peer resolution
//! - `client`: HTTP client for peer-to-peer communication
//! - `server`: HTTP server endpoints and request handlers
//! - `updates`: Self-sovereign update propagation with retry logic
//! - `retry`: Exponential backoff and retry management with persistence
//! - `validation`: Input validation, sanitization, and security hardening
//!
//! ## Usage Example
//!
//! ```rust
//! use inferno_shared::service_discovery::{
//!     ServiceDiscoveryConfig, NodeInfo, NodeType, AuthMode,
//!     validate_and_sanitize_node_info, validate_address
//! };
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configuration with authentication and security hardening
//! let config = ServiceDiscoveryConfig::with_shared_secret("secret123".to_string());
//!
//! // Create and validate node information with input sanitization
//! let mut node = NodeInfo::new(
//!     "backend-1".to_string(),
//!     "10.0.1.5:3000".to_string(),
//!     9090,
//!     NodeType::Backend
//! );
//!
//! // Validate and sanitize input data for security
//! let sanitized_node = validate_and_sanitize_node_info(node)?;
//! println!("Node {} capabilities: {:?}", sanitized_node.id, sanitized_node.capabilities);
//!
//! // Address validation example
//! let valid = validate_address("192.168.1.100:8080").is_ok();
//! println!("Address is valid: {}", valid);
//! # Ok(())
//! # }
//! ```

// Public module declarations
pub mod auth;
pub mod client;
pub mod config;
pub mod consensus;
pub mod errors;
pub mod health;
pub mod registration;
pub mod retry;
pub mod server;
pub mod service;
pub mod types;
pub mod updates;
pub mod validation;

#[cfg(test)]
pub mod tests;

// Re-export commonly used types for convenience
pub use auth::AuthMode;
pub use client::{ClientConfig, ServiceDiscoveryClient};
pub use config::ServiceDiscoveryConfig;
pub use consensus::{ConsensusMetrics, ConsensusResolver};
pub use errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
pub use health::{HealthCheckResult, HealthChecker, HttpHealthChecker, NodeVitals};
pub use registration::{
    RegistrationAction, RegistrationHandler, RegistrationRequest, RegistrationResponse,
};
pub use retry::{RetryConfig, RetryManager, RetryMetrics};
pub use server::ServiceDiscoveryServer;
pub use service::ServiceDiscovery;
pub use types::{BackendRegistration, NodeInfo, NodeType, PeerInfo};
pub use updates::{NodeUpdate, UpdatePropagator, UpdateResult};
pub use validation::{
    validate_address, validate_and_sanitize_node_info, validate_and_sanitize_peer_info,
    validate_capabilities, validate_node_id, validate_port, ValidationError, ValidationResult,
};

// Legacy compatibility - re-export old structure names
pub use health::NodeVitals as BackendVitals;
pub use types::NodeInfo as BackendInfo;

/// Current version of the service discovery protocol
pub const PROTOCOL_VERSION: &str = "1.0";

/// Default port for service discovery HTTP endpoints
pub const DEFAULT_SERVICE_PORT: u16 = 8080;

/// Default port for metrics and health endpoints
pub const DEFAULT_METRICS_PORT: u16 = 9090;

/// Default health check interval in seconds
pub const DEFAULT_HEALTH_CHECK_INTERVAL_SECS: u64 = 5;

/// Default health check timeout in seconds
pub const DEFAULT_HEALTH_CHECK_TIMEOUT_SECS: u64 = 2;

/// Maximum number of backends that can be registered
pub const MAX_REGISTERED_BACKENDS: usize = 1000;

/// Maximum size of capability strings
pub const MAX_CAPABILITY_LENGTH: usize = 64;

/// Maximum size of node IDs
pub const MAX_NODE_ID_LENGTH: usize = 128;

/// Service discovery protocol paths
pub mod protocol {
    /// Registration endpoint path
    pub const REGISTER_PATH: &str = "/service-discovery/register";

    /// Peer discovery endpoint path
    pub const PEERS_PATH: &str = "/service-discovery/peers";

    /// Health check path for service discovery health
    pub const HEALTH_PATH: &str = "/service-discovery/health";

    /// Status endpoint for service discovery information
    pub const STATUS_PATH: &str = "/service-discovery/status";
}

/// HTTP headers used in service discovery protocol
pub mod headers {
    /// Authorization header for authentication
    pub const AUTHORIZATION: &str = "Authorization";

    /// Content-Type header (always application/json)
    pub const CONTENT_TYPE: &str = "Content-Type";

    /// Protocol version header
    pub const PROTOCOL_VERSION: &str = "X-Service-Discovery-Version";

    /// Node ID header for identification
    pub const NODE_ID: &str = "X-Node-ID";

    /// Node type header
    pub const NODE_TYPE: &str = "X-Node-Type";
}

/// Content types used in service discovery
pub mod content_types {
    /// JSON content type for all service discovery messages
    pub const JSON: &str = "application/json";
}
