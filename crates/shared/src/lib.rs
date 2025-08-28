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

pub mod error;
pub mod metrics;

// Re-export commonly used types for convenience
pub use error::{InfernoError, ProxyError, Result};
pub use metrics::{MetricsCollector, MetricsSnapshot};
