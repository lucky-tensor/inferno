//! Error types for the Inferno Llama crate.
//!
//! This module provides comprehensive error handling for all Llama operations,
//! with specific attention to precision-related errors and memory constraints.

use thiserror::Error;

/// Result type alias for Llama operations
pub type Result<T> = std::result::Result<T, LlamaError>;

/// Comprehensive error types for Llama operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LlamaError {
    /// Tensor operation errors, particularly related to dtype mismatches
    #[error("Tensor operation failed: {message}. Context: {context}")]
    TensorError { message: String, context: String },

    /// Configuration validation errors
    #[error("Invalid configuration: {field} - {reason}")]
    ConfigError { field: String, reason: String },

    /// Memory constraint violations
    #[error("Memory constraint violated: {operation} would exceed {limit} bytes (requested: {requested})")]
    MemoryError {
        operation: String,
        limit: usize,
        requested: usize,
    },

    /// Precision/dtype related errors
    #[error("Precision error: {operation} failed with dtype {dtype}. {details}")]
    PrecisionError {
        operation: String,
        dtype: String,
        details: String,
    },

    /// Dimension mismatch errors
    #[error("Dimension mismatch in {operation}: expected {expected:?}, got {actual:?}")]
    DimensionError {
        operation: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Model loading and I/O errors
    #[error("Model I/O error: {message}")]
    IoError { message: String },

    /// Mathematical operation errors (NaN, overflow, etc.)
    #[error("Mathematical error in {operation}: {reason}")]
    MathError { operation: String, reason: String },
}

impl LlamaError {
    /// Create a tensor error with context
    pub fn tensor_error(message: impl Into<String>, context: impl Into<String>) -> Self {
        Self::TensorError {
            message: message.into(),
            context: context.into(),
        }
    }

    /// Create a configuration error
    pub fn config_error(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigError {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create a memory constraint error
    pub fn memory_error(operation: impl Into<String>, limit: usize, requested: usize) -> Self {
        Self::MemoryError {
            operation: operation.into(),
            limit,
            requested,
        }
    }

    /// Create a precision/dtype error
    pub fn precision_error(
        operation: impl Into<String>,
        dtype: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self::PrecisionError {
            operation: operation.into(),
            dtype: dtype.into(),
            details: details.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_error(
        operation: impl Into<String>,
        expected: Vec<usize>,
        actual: Vec<usize>,
    ) -> Self {
        Self::DimensionError {
            operation: operation.into(),
            expected,
            actual,
        }
    }

    /// Create a mathematical error
    pub fn math_error(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::MathError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create an I/O error
    pub fn io_error(message: impl Into<String>, _context: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }
}

// Convert from candle_core::Error to LlamaError
impl From<candle_core::Error> for LlamaError {
    fn from(err: candle_core::Error) -> Self {
        LlamaError::tensor_error(format!("Candle error: {}", err), "candle tensor operation")
    }
}

// Convert from LlamaError to candle_core::Error
impl From<LlamaError> for candle_core::Error {
    fn from(err: LlamaError) -> Self {
        match err {
            LlamaError::TensorError { message, context } => {
                candle_core::Error::Msg(format!("Tensor error: {} ({})", message, context))
            }
            LlamaError::ConfigError { field, reason } => {
                candle_core::Error::Msg(format!("Config error in {}: {}", field, reason))
            }
            LlamaError::MemoryError {
                operation,
                limit,
                requested,
            } => candle_core::Error::Msg(format!(
                "Memory error in {}: requested {} bytes, limit {} bytes",
                operation, requested, limit
            )),
            LlamaError::PrecisionError {
                operation,
                dtype,
                details,
            } => candle_core::Error::Msg(format!(
                "Precision error in {} with {}: {}",
                operation, dtype, details
            )),
            LlamaError::DimensionError {
                operation,
                expected,
                actual,
            } => candle_core::Error::Msg(format!(
                "Dimension error in {}: expected {:?}, got {:?}",
                operation, expected, actual
            )),
            LlamaError::IoError { message } => {
                candle_core::Error::Msg(format!("IO error: {}", message))
            }
            LlamaError::MathError { operation, reason } => {
                candle_core::Error::Msg(format!("Math error in {}: {}", operation, reason))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let tensor_err = LlamaError::tensor_error("test message", "test context");
        assert!(matches!(tensor_err, LlamaError::TensorError { .. }));

        let config_err = LlamaError::config_error("dim", "invalid value");
        assert!(matches!(config_err, LlamaError::ConfigError { .. }));

        let memory_err = LlamaError::memory_error("allocation", 1000, 2000);
        assert!(matches!(memory_err, LlamaError::MemoryError { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = LlamaError::precision_error("rope", "BF16", "unsupported operation");
        let display = format!("{}", err);
        assert!(display.contains("Precision error"));
        assert!(display.contains("rope"));
        assert!(display.contains("BF16"));
    }

    #[test]
    fn test_dimension_error() {
        let err = LlamaError::dimension_error("matmul", vec![2, 3], vec![2, 4]);
        match err {
            LlamaError::DimensionError {
                operation,
                expected,
                actual,
            } => {
                assert_eq!(operation, "matmul");
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(actual, vec![2, 4]);
            }
            _ => panic!("Expected DimensionError"),
        }
    }
}
