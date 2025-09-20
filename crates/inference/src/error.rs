//! Error types for the Inferno inference backend

use thiserror::Error;

/// Result type for Inferno inference operations
pub type InfernoResult<T> = Result<T, InfernoError>;

/// Main error type for Inferno inference backend operations
#[derive(Error, Debug)]
pub enum InfernoError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] InfernoConfigError),

    /// Engine-related errors
    #[error("Engine error: {0}")]
    Engine(#[from] InfernoEngineError),

    /// Memory allocation errors
    #[error("Memory allocation error: {0}")]
    Allocation(#[from] AllocationError),

    /// Service registration errors
    #[error("Service registration error: {0}")]
    ServiceRegistration(#[from] ServiceRegistrationError),

    /// Invalid argument provided
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// CUDA-related errors
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Model loading/unloading failed
    #[error("Model operation failed: {0}")]
    ModelLoadFailed(String),

    /// Inference operation failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Engine initialization failed
    #[error("Engine initialization failed: {0}")]
    InitializationFailed(String),

    /// Engine not initialized
    #[error("Engine not initialized")]
    EngineNotInitialized,

    /// Health check failed
    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),

    /// Generic operation failed
    #[error("Operation failed: {0}")]
    OperationFailed(String),

    /// CUDA not available
    #[error("CUDA support not available - compile with 'cuda' feature")]
    CudaNotAvailable,

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP-related errors
    #[error("HTTP error: {0}")]
    Http(String),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Thread join errors
    #[error("Thread join error")]
    ThreadJoin,

    /// Channel communication errors
    #[error("Channel error: {0}")]
    Channel(String),
}

/// Configuration-specific errors
#[derive(Error, Debug)]
pub enum InfernoConfigError {
    /// Missing required configuration field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid configuration value provided
    #[error("Invalid value for {field}: {value} (reason: {reason})")]
    InvalidValue {
        /// Configuration field name
        field: String,
        /// Invalid value provided
        value: String,
        /// Reason why value is invalid
        reason: String,
    },

    /// Configuration validation failed
    #[error("Configuration validation failed: {0}")]
    ValidationFailed(String),

    /// Environment variable access error
    #[error("Environment variable error: {0}")]
    Environment(String),

    /// File read/write error
    #[error("File read error: {0}")]
    FileRead(String),

    /// Configuration parsing error
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Engine-specific errors
#[derive(Error, Debug)]
pub enum InfernoEngineError {
    /// Engine has not been started
    #[error("Engine not started")]
    NotStarted,

    /// Engine is already running
    #[error("Engine already running")]
    AlreadyRunning,

    /// Model has not been loaded into engine
    #[error("Model not loaded")]
    ModelNotLoaded,

    /// Request queue has reached capacity
    #[error("Request queue full")]
    QueueFull,

    /// Request with given ID not found
    #[error("Request not found: {0}")]
    RequestNotFound(u64),

    /// Batch processing operation failed
    #[error("Batch processing error: {0}")]
    BatchProcessing(String),

    /// Request scheduling error
    #[error("Scheduling error: {0}")]
    Scheduling(String),

    /// System resources exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// Memory allocation errors
#[derive(Error, Debug)]
pub enum AllocationError {
    /// Insufficient memory available for allocation
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Number of bytes requested
        requested: usize,
        /// Number of bytes available
        available: usize,
    },

    /// Memory alignment requirement not met
    #[error("Invalid alignment: {0} (must be power of 2)")]
    InvalidAlignment(usize),

    /// Requested memory block exceeds maximum size
    #[error("Block size too large: {0} bytes")]
    BlockTooLarge(usize),

    /// Memory too fragmented for allocation
    #[error(
        "Memory fragmentation: cannot allocate {size} bytes (largest free block: {largest_free})"
    )]
    Fragmentation {
        /// Requested allocation size in bytes
        size: usize,
        /// Largest available contiguous block in bytes
        largest_free: usize,
    },

    /// GPU/device memory operation failed
    #[error("Device memory error: {0}")]
    DeviceMemory(String),

    /// Memory pool management error
    #[error("Memory pool error: {0}")]
    MemoryPool(String),

    /// Unsupported device type (GPU only)
    #[error("Unsupported device: {0}")]
    UnsupportedDevice(String),
}

/// Service registration errors
#[derive(Error, Debug)]
pub enum ServiceRegistrationError {
    /// Service discovery backend is unavailable
    #[error("Service discovery unavailable")]
    DiscoveryUnavailable,

    /// Service registration attempt failed
    #[error("Registration failed: {0}")]
    RegistrationFailed(String),

    /// Health check endpoint registration failed
    #[error("Health check registration failed: {0}")]
    HealthCheckFailed(String),

    /// Service is already registered
    #[error("Service already registered")]
    AlreadyRegistered,

    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),
}

/// Convert from inferno-shared errors
impl From<inferno_shared::error::InfernoError> for InfernoError {
    fn from(err: inferno_shared::error::InfernoError) -> Self {
        match err {
            inferno_shared::error::InfernoError::Configuration { message, .. } => {
                Self::Configuration(InfernoConfigError::ValidationFailed(message))
            }
            inferno_shared::error::InfernoError::Network { message, .. } => {
                Self::ServiceRegistration(ServiceRegistrationError::Network(message))
            }
            _ => Self::OperationFailed(err.to_string()),
        }
    }
}

/// Convert from validator errors
impl From<validator::ValidationErrors> for InfernoError {
    fn from(err: validator::ValidationErrors) -> Self {
        let messages: Vec<String> = err
            .field_errors()
            .iter()
            .flat_map(|(field, errors)| {
                errors.iter().map(move |error| {
                    format!(
                        "{}: {}",
                        field,
                        error
                            .message
                            .as_ref()
                            .unwrap_or(&std::borrow::Cow::Borrowed("validation error"))
                    )
                })
            })
            .collect();

        Self::Configuration(InfernoConfigError::ValidationFailed(messages.join(", ")))
    }
}

/// Convert common standard library errors
impl From<tokio::task::JoinError> for InfernoError {
    fn from(_: tokio::task::JoinError) -> Self {
        Self::ThreadJoin
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for InfernoError {
    fn from(err: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Self::Channel(format!("Send error: {err}"))
    }
}

impl From<tokio::sync::oneshot::error::RecvError> for InfernoError {
    fn from(err: tokio::sync::oneshot::error::RecvError) -> Self {
        Self::Channel(format!("Receive error: {err}"))
    }
}

/// HTTP status code mapping for API responses
impl InfernoError {
    /// Convert error to appropriate HTTP status code
    #[must_use]
    pub const fn to_status_code(&self) -> u16 {
        match self {
            Self::InvalidArgument(_) | Self::Configuration(_) => 400,
            Self::EngineNotInitialized | Self::HealthCheckFailed(_) => 503,
            Self::OutOfMemory(_) | Self::Engine(InfernoEngineError::ResourceExhausted(_)) => 507,
            Self::Timeout(_) => 408,
            Self::Engine(InfernoEngineError::QueueFull) => 429,
            Self::Engine(InfernoEngineError::RequestNotFound(_)) => 404,
            Self::CudaNotAvailable => 501,
            Self::ModelLoadFailed(_) => 422,
            _ => 500,
        }
    }

    /// Get user-friendly error message
    #[must_use]
    pub fn user_message(&self) -> String {
        match self {
            Self::InvalidArgument(msg) => format!("Invalid request: {msg}"),
            Self::OutOfMemory(_) => "Server is out of memory. Please try again later.".to_string(),
            Self::Timeout(_) => "Request timed out. Please try again.".to_string(),
            Self::Engine(InfernoEngineError::QueueFull) => {
                "Server is busy. Please try again later.".to_string()
            }
            Self::CudaNotAvailable => "GPU acceleration is not available.".to_string(),
            Self::EngineNotInitialized => "Service is temporarily unavailable.".to_string(),
            Self::HealthCheckFailed(_) => "Service is unhealthy.".to_string(),
            _ => "An internal error occurred.".to_string(),
        }
    }
}

/// Structured error response for HTTP APIs
#[derive(Debug, serde::Serialize)]
pub struct ErrorResponse {
    /// Error details for API consumers
    pub error: ErrorDetails,
}

/// Detailed error information for structured API responses
#[derive(Debug, serde::Serialize)]
pub struct ErrorDetails {
    /// Error type identifier
    pub r#type: String,
    /// Human-readable error message
    pub message: String,
    /// Optional error code for programmatic handling
    pub code: Option<String>,
    /// Optional additional error details
    pub details: Option<serde_json::Value>,
}

impl From<InfernoError> for ErrorResponse {
    fn from(err: InfernoError) -> Self {
        let error_type = match &err {
            InfernoError::Configuration(_) => "configuration_error",
            InfernoError::Engine(_) => "engine_error",
            InfernoError::Allocation(_) => "memory_error",
            InfernoError::InvalidArgument(_) => "validation_error",
            InfernoError::OutOfMemory(_) => "resource_error",
            InfernoError::CudaError(_) => "cuda_error",
            InfernoError::ModelLoadFailed(_) => "model_error",
            InfernoError::InferenceFailed(_) => "inference_error",
            InfernoError::Timeout(_) => "timeout_error",
            InfernoError::CudaNotAvailable => "feature_unavailable",
            _ => "internal_error",
        };

        Self {
            error: ErrorDetails {
                r#type: error_type.to_string(),
                message: err.user_message(),
                code: None,
                details: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            InfernoError::InvalidArgument("test".to_string()).to_status_code(),
            400
        );
        assert_eq!(
            InfernoError::OutOfMemory("test".to_string()).to_status_code(),
            507
        );
        assert_eq!(InfernoError::CudaNotAvailable.to_status_code(), 501);
        assert_eq!(InfernoError::EngineNotInitialized.to_status_code(), 503);
    }

    #[test]
    fn test_user_messages() {
        let err = InfernoError::InvalidArgument("missing field".to_string());
        assert_eq!(err.user_message(), "Invalid request: missing field");

        let err = InfernoError::CudaNotAvailable;
        assert_eq!(err.user_message(), "GPU acceleration is not available.");
    }

    #[test]
    fn test_error_response_conversion() {
        let err = InfernoError::OutOfMemory("GPU memory full".to_string());
        let response = ErrorResponse::from(err);

        assert_eq!(response.error.r#type, "resource_error");
        assert!(!response.error.message.is_empty());
    }
}
