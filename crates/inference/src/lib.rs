//! # Inferno Inference Engine
//!
//! A high-performance inference engine implementation for the Inferno platform.
//! This crate provides Burn framework-based inference with multi-backend support,
//! advanced batching, and memory management capabilities.
//!
//! ## Features
//!
//! - **Burn Framework**: Multi-backend inference (CPU/CUDA/ROCm/Metal/WebGPU)
//! - **Real Model Support**: Actual LLM inference with Hugging Face model downloads
//! - **Memory Management**: Efficient memory pooling and tracking
//! - **Service Discovery**: Integration with Inferno's service discovery system
//! - **Health Monitoring**: Comprehensive health checks and metrics
//! - **No Mocking**: Only real model inference, no pattern matching fallbacks
//!
//! ## Usage
//!
//! ```rust,no_run
//! use inferno_inference::{InfernoBackend, InfernoConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = InfernoConfig::from_env()?;
//!     let backend = InfernoBackend::new(config)?;
//!     backend.start().await?;
//!     Ok(())
//! }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(
    missing_docs,
    rust_2018_idioms,
    unreachable_pub,
    clippy::all,
    clippy::pedantic,
    clippy::nursery
)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::similar_names,
    // Allow for FFI code which follows different conventions
    clippy::wildcard_imports,
    clippy::missing_const_for_fn,
    clippy::uninlined_format_args,
    clippy::borrow_as_ptr,
    clippy::must_use_candidate,
    clippy::ptr_as_ptr,
    clippy::needless_return,
    clippy::manual_assert,
    clippy::non_send_fields_in_send_ty,
    clippy::significant_drop_tightening,
    clippy::unsafe_derive_deserialize,
    clippy::needless_pass_by_value,
    dead_code
)]

// Core modules
pub mod config;
pub mod error;

// Engine components
pub mod engine;
pub mod inference;
pub mod memory;
pub mod models;

// Service integration
pub mod health;
pub mod server;
pub mod service;

// Test modules
mod tokenizer_tests;

// Re-export core types and traits
pub use config::{
    HealthConfig, InfernoConfig, InfernoConfigBuilder, LoggingConfig, ServerConfig,
    ServiceDiscoveryConfig,
};
pub use engine::{InfernoBackend, InfernoEngine};
pub use error::{
    AllocationError, InfernoConfigError, InfernoEngineError, InfernoError, InfernoResult,
    ServiceRegistrationError,
};
pub use health::{HealthStatus, InfernoHealthChecker};
pub use inference::{
    create_engine, create_math_test_request, BurnInferenceEngine, InferenceEngine, InferenceRequest, InferenceResponse,
};

pub use server::InfernoServer;
pub use service::InfernoServiceRegistration;

// Memory management exports
pub use memory::{CudaMemoryPool, DeviceMemory, GpuAllocator, MemoryStats, MemoryTracker};

/// Current version of the Inferno inference backend
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum supported CUDA version
pub const MIN_CUDA_VERSION: (u32, u32) = (11, 8);

/// Default HTTP server port
pub const DEFAULT_PORT: u16 = 8000;

/// Default batch size for inference
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// Default GPU memory pool size (in MB)
pub const DEFAULT_MEMORY_POOL_SIZE_MB: usize = 4096;
