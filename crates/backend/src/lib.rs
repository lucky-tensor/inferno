//! # Inferno Backend
//!
//! AI inference backend component for the Inferno distributed systems platform.
//! Handles model loading, request batching, and health monitoring.
//!
//! ## Features
//!
//! - Model loading and management
//! - Request batching and optimization
//! - Health metrics HTTP server
//! - Service registration with load balancers
//! - Resource utilization monitoring

pub mod cli_options;
pub mod config;
pub mod health;
pub mod inference;
pub mod registration;

pub use cli_options::BackendCliOptions;
pub use config::BackendConfig;
