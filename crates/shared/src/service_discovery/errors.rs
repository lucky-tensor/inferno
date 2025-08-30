//! Error types for service discovery operations
//!
//! This module defines specific error types for service discovery operations,
//! providing detailed error handling and context for failures.

use crate::error::InfernoError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Errors that can occur during service discovery operations
///
/// This enum provides specific error types for different failure modes
/// in the service discovery system, allowing for targeted error handling
/// and better debugging information.
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::ServiceDiscoveryError;
///
/// let error = ServiceDiscoveryError::AuthenticationFailed(
///     "Invalid Bearer token".to_string()
/// );
/// println!("Auth error: {}", error);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServiceDiscoveryError {
    /// Authentication failed during service discovery operation
    ///
    /// This error occurs when:
    /// - Missing Authorization header in SharedSecret mode
    /// - Invalid Bearer token format
    /// - Token doesn't match expected shared secret
    AuthenticationFailed(String),

    /// Backend registration failed
    ///
    /// This error occurs when:
    /// - Invalid registration data provided
    /// - Network connectivity issues during registration
    /// - Timeout waiting for registration response
    RegistrationFailed { backend_id: String, reason: String },

    /// Health check operation failed
    ///
    /// This error occurs when:
    /// - Health check endpoint unreachable
    /// - Invalid health check response format
    /// - Health check request timeout
    HealthCheckFailed { backend_id: String, reason: String },

    /// Backend not found in the registry
    ///
    /// This error occurs when:
    /// - Requested backend ID doesn't exist
    /// - Backend was removed due to health failures
    BackendNotFound(String),

    /// Invalid node information provided
    ///
    /// This error occurs when:
    /// - Required fields missing from NodeInfo
    /// - Invalid network address format
    /// - Node type doesn't match capabilities
    InvalidNodeInfo { reason: String },

    /// Consensus operation failed
    ///
    /// This error occurs when:
    /// - Peer disagreement on node state
    /// - Network partition during consensus
    /// - Conflicting node information updates
    ConsensusFailed { operation: String, reason: String },

    /// Consensus algorithm error
    ///
    /// This error occurs when:
    /// - Insufficient peer responses for consensus
    /// - No valid peer information found
    /// - Internal consistency validation failures
    ConsensusError { reason: String },

    /// Configuration validation failed
    ///
    /// This error occurs when:
    /// - Invalid configuration parameters
    /// - Missing required configuration fields
    /// - Inconsistent configuration state
    ConfigValidationFailed(String),

    /// Network communication error
    ///
    /// This error occurs when:
    /// - HTTP request/response failures
    /// - Network connectivity issues
    /// - Timeout during network operations
    NetworkError { operation: String, error: String },

    /// Serialization/deserialization error
    ///
    /// This error occurs when:
    /// - Invalid JSON in requests/responses
    /// - Schema mismatch between nodes
    /// - Malformed protocol messages
    SerializationError(String),

    /// Generic service discovery error
    ///
    /// Used for errors that don't fit other specific categories
    /// but are related to service discovery operations.
    Other(String),
}

impl ServiceDiscoveryError {
    /// Creates an authentication failed error
    ///
    /// # Arguments
    ///
    /// * `reason` - Description of the authentication failure
    ///
    /// # Returns
    ///
    /// Returns a new AuthenticationFailed error with the provided reason.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let error = ServiceDiscoveryError::auth_failed("Missing Bearer token");
    /// assert!(matches!(error, ServiceDiscoveryError::AuthenticationFailed(_)));
    /// ```
    pub fn auth_failed(reason: impl Into<String>) -> Self {
        Self::AuthenticationFailed(reason.into())
    }

    /// Creates a registration failed error
    ///
    /// # Arguments
    ///
    /// * `backend_id` - ID of the backend that failed to register
    /// * `reason` - Description of the registration failure
    ///
    /// # Returns
    ///
    /// Returns a new RegistrationFailed error with backend ID and reason.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let error = ServiceDiscoveryError::registration_failed(
    ///     "backend-1",
    ///     "Connection timeout"
    /// );
    /// ```
    pub fn registration_failed(backend_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::RegistrationFailed {
            backend_id: backend_id.into(),
            reason: reason.into(),
        }
    }

    /// Creates a health check failed error
    ///
    /// # Arguments
    ///
    /// * `backend_id` - ID of the backend that failed health check
    /// * `reason` - Description of the health check failure
    ///
    /// # Returns
    ///
    /// Returns a new HealthCheckFailed error with backend ID and reason.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let error = ServiceDiscoveryError::health_check_failed(
    ///     "backend-2",
    ///     "Endpoint returned 503"
    /// );
    /// ```
    pub fn health_check_failed(backend_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::HealthCheckFailed {
            backend_id: backend_id.into(),
            reason: reason.into(),
        }
    }

    /// Creates a consensus failed error
    ///
    /// # Arguments
    ///
    /// * `operation` - The consensus operation that failed
    /// * `reason` - Description of the consensus failure
    ///
    /// # Returns
    ///
    /// Returns a new ConsensusFailed error with operation and reason.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let error = ServiceDiscoveryError::consensus_failed(
    ///     "peer_update",
    ///     "Conflicting timestamps"
    /// );
    /// ```
    pub fn consensus_failed(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConsensusFailed {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Creates a network error
    ///
    /// # Arguments
    ///
    /// * `operation` - The network operation that failed
    /// * `error` - Description of the network error
    ///
    /// # Returns
    ///
    /// Returns a new NetworkError with operation and error details.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let error = ServiceDiscoveryError::network_error(
    ///     "health_check",
    ///     "Connection refused"
    /// );
    /// ```
    pub fn network_error(operation: impl Into<String>, error: impl Into<String>) -> Self {
        Self::NetworkError {
            operation: operation.into(),
            error: error.into(),
        }
    }

    /// Returns whether this error is retryable
    ///
    /// # Returns
    ///
    /// Returns `true` if the operation that caused this error can be retried,
    /// `false` if the error indicates a permanent failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let network_error = ServiceDiscoveryError::network_error("request", "timeout");
    /// assert!(network_error.is_retryable());
    ///
    /// let auth_error = ServiceDiscoveryError::auth_failed("invalid token");
    /// assert!(!auth_error.is_retryable());
    /// ```
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::AuthenticationFailed(_) => false,
            Self::RegistrationFailed { .. } => true,
            Self::HealthCheckFailed { .. } => true,
            Self::BackendNotFound(_) => false,
            Self::InvalidNodeInfo { .. } => false,
            Self::ConsensusFailed { .. } => true,
            Self::ConsensusError { .. } => true,
            Self::ConfigValidationFailed(_) => false,
            Self::NetworkError { .. } => true,
            Self::SerializationError(_) => false,
            Self::Other(_) => false,
        }
    }

    /// Returns the error category for logging and metrics
    ///
    /// # Returns
    ///
    /// Returns a string category for this error type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::ServiceDiscoveryError;
    ///
    /// let error = ServiceDiscoveryError::auth_failed("invalid");
    /// assert_eq!(error.category(), "authentication");
    /// ```
    pub fn category(&self) -> &'static str {
        match self {
            Self::AuthenticationFailed(_) => "authentication",
            Self::RegistrationFailed { .. } => "registration",
            Self::HealthCheckFailed { .. } => "health_check",
            Self::BackendNotFound(_) => "backend_lookup",
            Self::InvalidNodeInfo { .. } => "validation",
            Self::ConsensusFailed { .. } => "consensus",
            Self::ConsensusError { .. } => "consensus",
            Self::ConfigValidationFailed(_) => "configuration",
            Self::NetworkError { .. } => "network",
            Self::SerializationError(_) => "serialization",
            Self::Other(_) => "other",
        }
    }
}

impl fmt::Display for ServiceDiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthenticationFailed(reason) => {
                write!(f, "Authentication failed: {}", reason)
            }
            Self::RegistrationFailed { backend_id, reason } => {
                write!(
                    f,
                    "Registration failed for backend {}: {}",
                    backend_id, reason
                )
            }
            Self::HealthCheckFailed { backend_id, reason } => {
                write!(
                    f,
                    "Health check failed for backend {}: {}",
                    backend_id, reason
                )
            }
            Self::BackendNotFound(backend_id) => {
                write!(f, "Backend not found: {}", backend_id)
            }
            Self::InvalidNodeInfo { reason } => {
                write!(f, "Invalid node information: {}", reason)
            }
            Self::ConsensusFailed { operation, reason } => {
                write!(
                    f,
                    "Consensus failed for operation {}: {}",
                    operation, reason
                )
            }
            Self::ConsensusError { reason } => {
                write!(f, "Consensus algorithm error: {}", reason)
            }
            Self::ConfigValidationFailed(reason) => {
                write!(f, "Configuration validation failed: {}", reason)
            }
            Self::NetworkError { operation, error } => {
                write!(f, "Network error during {}: {}", operation, error)
            }
            Self::SerializationError(reason) => {
                write!(f, "Serialization error: {}", reason)
            }
            Self::Other(reason) => {
                write!(f, "Service discovery error: {}", reason)
            }
        }
    }
}

impl std::error::Error for ServiceDiscoveryError {}

/// Converts a ServiceDiscoveryError into an InfernoError
impl From<ServiceDiscoveryError> for InfernoError {
    fn from(error: ServiceDiscoveryError) -> Self {
        match error {
            ServiceDiscoveryError::AuthenticationFailed(msg) => {
                InfernoError::request_validation(format!("Authentication failed: {}", msg), None)
            }
            ServiceDiscoveryError::ConfigValidationFailed(msg) => InfernoError::configuration(
                format!("Service discovery configuration error: {}", msg),
                None,
            ),
            ServiceDiscoveryError::NetworkError { operation, error } => InfernoError::network(
                "service_discovery",
                format!("{} failed: {}", operation, error),
                None,
            ),
            _ => InfernoError::internal(format!("Service discovery error: {}", error), None),
        }
    }
}

/// Result type alias for service discovery operations
pub type ServiceDiscoveryResult<T> = std::result::Result<T, ServiceDiscoveryError>;
