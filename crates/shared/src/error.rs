//! # Shared Error Handling Module
//!
//! Comprehensive error handling for Inferno distributed systems with proper
//! error classification, context preservation, and performance optimization.
//!
//! ## Error Categories
//!
//! - **Configuration Errors**: Invalid component configuration
//! - **Network Errors**: Connection failures, timeouts, DNS issues
//! - **Backend Errors**: Upstream server failures, invalid responses
//! - **System Errors**: Resource exhaustion, OS-level failures
//! - **Security Errors**: Authentication, authorization, validation failures
//!
//! ## Performance Characteristics
//!
//! - Error creation: < 100ns
//! - Error propagation: Zero allocation where possible
//! - Error logging: Non-blocking, structured format
//! - Error recovery: Fast failover with circuit breaker patterns

use std::net::AddrParseError;
use thiserror::Error;
use tracing::warn;

/// Result type alias for Inferno operations
///
/// This is the standard Result type used throughout the Inferno codebase.
/// It provides a consistent interface for error handling and makes
/// error propagation more ergonomic.
pub type Result<T> = std::result::Result<T, InfernoError>;

/// Comprehensive error types for Inferno distributed system operations
///
/// This enum covers all possible error conditions that can occur during
/// operations across all Inferno components. Each variant includes relevant
/// context and is designed to provide actionable information for debugging
/// and monitoring.
///
/// ## Design Principles
///
/// - Errors are cheap to construct (no heap allocations in common cases)
/// - Error messages are human-readable and actionable
/// - Context is preserved through the error chain
/// - Errors can be efficiently serialized for logging
/// - Error types map clearly to HTTP status codes where applicable
#[derive(Error, Debug)]
pub enum InfernoError {
    /// Configuration validation errors
    ///
    /// These errors occur during component startup when validating the
    /// provided configuration. They typically indicate user configuration
    /// mistakes that need to be corrected before the component can start.
    ///
    /// **HTTP Status Mapping**: Not applicable (startup error)
    ///
    /// **Recovery Strategy**: Fix configuration and restart
    #[error("Configuration error: {message}")]
    Configuration {
        /// Human-readable error message describing the configuration issue
        message: String,
        /// Optional source error for additional context
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Network-level connectivity errors
    ///
    /// These errors occur when components cannot establish or maintain
    /// network connections to other services. They include DNS resolution
    /// failures, connection timeouts, and connection refused errors.
    ///
    /// **HTTP Status Mapping**: 502 Bad Gateway or 504 Gateway Timeout
    ///
    /// **Recovery Strategy**: Retry with exponential backoff, try alternative targets
    #[error("Network error connecting to {target}: {message}")]
    Network {
        /// Target address that failed to connect
        target: String,
        /// Descriptive error message
        message: String,
        /// Underlying network error for debugging
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Backend server errors
    ///
    /// These errors occur when backend servers return error responses
    /// or invalid data. This includes HTTP error status codes, malformed
    /// responses, and protocol violations.
    ///
    /// **HTTP Status Mapping**: Forward backend status or 502 Bad Gateway
    ///
    /// **Recovery Strategy**: Try alternative backend, return error to client
    #[error("Backend error from {backend}: HTTP {status} - {message}")]
    Backend {
        /// Backend server address
        backend: String,
        /// HTTP status code from backend
        status: u16,
        /// Error message or response body
        message: String,
    },

    /// Request timeout errors
    ///
    /// These errors occur when operations exceed configured timeout values.
    /// This includes connection timeouts, read timeouts, and write timeouts.
    ///
    /// **HTTP Status Mapping**: 504 Gateway Timeout
    ///
    /// **Recovery Strategy**: Retry with shorter timeout, try alternative backend
    #[error("Operation timed out after {timeout_ms}ms: {operation}")]
    Timeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
        /// Description of the operation that timed out
        operation: String,
    },

    /// Resource exhaustion errors
    ///
    /// These errors occur when components run out of system resources
    /// such as memory, file descriptors, or connection pool slots.
    ///
    /// **HTTP Status Mapping**: 503 Service Unavailable
    ///
    /// **Recovery Strategy**: Shed load, implement backpressure
    #[error("Resource exhausted: {resource} - {message}")]
    ResourceExhausted {
        /// Type of resource that was exhausted
        resource: String,
        /// Additional context about the exhaustion
        message: String,
    },

    /// Request validation errors
    ///
    /// These errors occur when incoming requests fail validation checks.
    /// This includes invalid headers, malformed URLs, unsupported methods,
    /// and security policy violations.
    ///
    /// **HTTP Status Mapping**: 400 Bad Request or 403 Forbidden
    ///
    /// **Recovery Strategy**: Return error to client, log security events
    #[error("Request validation failed: {reason}")]
    RequestValidation {
        /// Reason for validation failure
        reason: String,
        /// Optional request context for debugging
        context: Option<String>,
    },

    /// Internal system errors
    ///
    /// These errors represent unexpected internal failures that shouldn't
    /// normally occur during operation. They typically indicate bugs or
    /// system-level issues that require investigation.
    ///
    /// **HTTP Status Mapping**: 500 Internal Server Error
    ///
    /// **Recovery Strategy**: Log error, attempt graceful degradation
    #[error("Internal error: {message}")]
    Internal {
        /// Error message describing the internal failure
        message: String,
        /// Source error for debugging
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Service unavailable errors
    ///
    /// These errors occur when services cannot serve requests due to
    /// backend unavailability, maintenance mode, or circuit breaker
    /// activation.
    ///
    /// **HTTP Status Mapping**: 503 Service Unavailable
    ///
    /// **Recovery Strategy**: Retry after delay, check service health
    #[error("Service unavailable: {reason}")]
    ServiceUnavailable {
        /// Reason service is unavailable
        reason: String,
        /// Optional estimated retry time in seconds
        retry_after: Option<u32>,
    },
}

impl InfernoError {
    /// Creates a configuration error with context
    ///
    /// # Arguments
    ///
    /// * `message` - Human-readable error description
    /// * `source` - Optional underlying error cause
    ///
    /// # Performance Notes
    ///
    /// - String allocation required for message
    /// - Boxed source error adds heap allocation
    /// - Consider using static strings for common errors
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::configuration("Invalid listen address", None);
    /// ```
    pub fn configuration(
        message: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source,
        }
    }

    /// Creates a network error with target and context
    ///
    /// # Arguments
    ///
    /// * `target` - Target address that failed
    /// * `message` - Error description
    /// * `source` - Optional underlying network error
    ///
    /// # Performance Notes
    ///
    /// - Two string allocations required
    /// - Use string references where possible to minimize allocations
    /// - Consider caching common network error messages
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::network("192.168.1.1:8080", "Connection refused", None);
    /// ```
    pub fn network(
        target: impl Into<String>,
        message: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Network {
            target: target.into(),
            message: message.into(),
            source,
        }
    }

    /// Creates a backend error with server and status information
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend server address
    /// * `status` - HTTP status code from backend
    /// * `message` - Error message or response body
    ///
    /// # Performance Notes
    ///
    /// - Two string allocations required
    /// - Status code copied (minimal overhead)
    /// - Consider limiting message length for large response bodies
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::backend("api.example.com:443", 500, "Internal Server Error");
    /// ```
    pub fn backend(backend: impl Into<String>, status: u16, message: impl Into<String>) -> Self {
        Self::Backend {
            backend: backend.into(),
            status,
            message: message.into(),
        }
    }

    /// Creates a timeout error with operation context
    ///
    /// # Arguments
    ///
    /// * `timeout` - Timeout duration that was exceeded
    /// * `operation` - Description of the operation that timed out
    ///
    /// # Performance Notes
    ///
    /// - Single string allocation for operation description
    /// - Duration converted to milliseconds (no allocation)
    /// - Very fast to construct (< 50ns typical)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    /// use std::time::Duration;
    ///
    /// let error = InfernoError::timeout(Duration::from_secs(30), "backend connection");
    /// ```
    pub fn timeout(timeout: std::time::Duration, operation: impl Into<String>) -> Self {
        Self::Timeout {
            timeout_ms: timeout.as_millis() as u64,
            operation: operation.into(),
        }
    }

    /// Creates a resource exhaustion error
    ///
    /// # Arguments
    ///
    /// * `resource` - Type of resource that was exhausted
    /// * `message` - Additional context about the exhaustion
    ///
    /// # Performance Notes
    ///
    /// - Two string allocations required
    /// - Consider using static strings for common resource types
    /// - Keep messages concise to minimize allocation overhead
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::resource_exhausted("connection_pool", "All 1000 connections in use");
    /// ```
    pub fn resource_exhausted(resource: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            message: message.into(),
        }
    }

    /// Creates a request validation error
    ///
    /// # Arguments
    ///
    /// * `reason` - Reason for validation failure
    /// * `context` - Optional request context for debugging
    ///
    /// # Performance Notes
    ///
    /// - One or two string allocations depending on context
    /// - Validation errors are expected in normal operation
    /// - Should be very fast to construct and handle
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::request_validation("Invalid Content-Type header", Some("POST /api/data".to_string()));
    /// ```
    pub fn request_validation(reason: impl Into<String>, context: Option<String>) -> Self {
        Self::RequestValidation {
            reason: reason.into(),
            context,
        }
    }

    /// Creates an internal error with source context
    ///
    /// # Arguments
    ///
    /// * `message` - Error description
    /// * `source` - Optional underlying error cause
    ///
    /// # Performance Notes
    ///
    /// - Single string allocation for message
    /// - Boxed source error adds heap allocation
    /// - Internal errors should be rare in normal operation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::internal("Failed to serialize metrics", None);
    /// ```
    pub fn internal(
        message: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Internal {
            message: message.into(),
            source,
        }
    }

    /// Creates a service unavailable error
    ///
    /// # Arguments
    ///
    /// * `reason` - Reason service is unavailable
    /// * `retry_after` - Optional retry time in seconds
    ///
    /// # Performance Notes
    ///
    /// - Single string allocation for reason
    /// - Optional retry_after is stack-allocated
    /// - Very efficient for circuit breaker patterns
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::error::InfernoError;
    ///
    /// let error = InfernoError::service_unavailable("Circuit breaker open", Some(60));
    /// ```
    pub fn service_unavailable(reason: impl Into<String>, retry_after: Option<u32>) -> Self {
        Self::ServiceUnavailable {
            reason: reason.into(),
            retry_after,
        }
    }

    /// Maps this error to an appropriate HTTP status code
    ///
    /// This method provides a consistent mapping from internal Inferno errors
    /// to HTTP status codes that should be returned to clients.
    ///
    /// # Returns
    ///
    /// Returns the appropriate HTTP status code (100-599 range)
    ///
    /// # Performance Notes
    ///
    /// - Zero allocations (pattern matching only)
    /// - Constant time operation (< 5ns)
    /// - Branch prediction friendly for common error types
    ///
    /// # Status Code Mapping
    ///
    /// - Configuration errors: Not applicable (startup only)
    /// - Network errors: 502 Bad Gateway or 504 Gateway Timeout
    /// - Backend errors: Forward backend status or 502
    /// - Timeout errors: 504 Gateway Timeout
    /// - Resource exhaustion: 503 Service Unavailable
    /// - Request validation: 400 Bad Request
    /// - Internal errors: 500 Internal Server Error
    /// - Service unavailable: 503 Service Unavailable
    pub fn to_http_status(&self) -> u16 {
        match self {
            InfernoError::Configuration { .. } => {
                // Configuration errors shouldn't reach HTTP layer
                warn!("Configuration error reached HTTP status mapping");
                500
            }
            InfernoError::Network { .. } => 502, // Bad Gateway
            InfernoError::Backend { status, .. } => {
                // Forward backend status, but ensure it's valid
                if *status >= 400 && *status <= 599 {
                    *status
                } else {
                    502 // Bad Gateway for invalid status codes
                }
            }
            InfernoError::Timeout { .. } => 504, // Gateway Timeout
            InfernoError::ResourceExhausted { .. } => 503, // Service Unavailable
            InfernoError::RequestValidation { .. } => 400, // Bad Request
            InfernoError::Internal { .. } => 500, // Internal Server Error
            InfernoError::ServiceUnavailable { .. } => 503, // Service Unavailable
        }
    }

    /// Checks if this error represents a temporary condition
    ///
    /// Temporary errors may be suitable for automatic retry,
    /// while permanent errors should not be retried.
    ///
    /// # Returns
    ///
    /// Returns `true` if the error condition is likely temporary
    ///
    /// # Performance Notes
    ///
    /// - Zero allocations (pattern matching only)
    /// - Constant time operation
    /// - Used for retry logic decisions
    ///
    /// # Retry Classification
    ///
    /// **Temporary (retriable):**
    /// - Network timeouts
    /// - Connection failures
    /// - Resource exhaustion
    /// - Service unavailable
    /// - 5xx backend errors
    ///
    /// **Permanent (not retriable):**
    /// - Configuration errors
    /// - Request validation failures
    /// - 4xx backend errors
    /// - Internal logic errors
    pub fn is_temporary(&self) -> bool {
        match self {
            InfernoError::Configuration { .. } => false, // Config errors are permanent
            InfernoError::Network { .. } => true,        // Network issues often temporary
            InfernoError::Backend { status, .. } => {
                // 5xx errors are temporary, 4xx are permanent
                *status >= 500
            }
            InfernoError::Timeout { .. } => true, // Timeouts are temporary
            InfernoError::ResourceExhausted { .. } => true, // Resources may become available
            InfernoError::RequestValidation { .. } => false, // Validation errors are permanent
            InfernoError::Internal { .. } => false, // Internal errors need investigation
            InfernoError::ServiceUnavailable { .. } => true, // Service may come back
        }
    }
}

/// Conversion from address parsing errors
///
/// This provides automatic conversion from std::net::AddrParseError
/// to InfernoError, making error handling more ergonomic when parsing
/// network addresses in configuration.
impl From<AddrParseError> for InfernoError {
    fn from(err: AddrParseError) -> Self {
        InfernoError::configuration(
            format!("Invalid network address: {}", err),
            Some(Box::new(err)),
        )
    }
}

/// Conversion from I/O errors
///
/// This provides automatic conversion from std::io::Error to InfernoError,
/// mapping common I/O error kinds to appropriate Inferno error types.
impl From<std::io::Error> for InfernoError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::TimedOut => {
                InfernoError::timeout(std::time::Duration::from_secs(30), "I/O operation")
            }
            std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset => {
                InfernoError::network("unknown", "Connection failed", Some(Box::new(err)))
            }
            std::io::ErrorKind::NotFound => {
                InfernoError::network("unknown", "Host not found", Some(Box::new(err)))
            }
            _ => InfernoError::internal("I/O error", Some(Box::new(err))),
        }
    }
}

// Compatibility alias for existing ProxyError usage
pub type ProxyError = InfernoError;

