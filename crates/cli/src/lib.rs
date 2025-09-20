//! # Inferno CLI
//!
//! Unified command-line interface for Inferno distributed systems.
//! Provides subcommands for starting and managing different components.
//!
//! ## Usage
//!
//! ```bash
//! # Start proxy
//! inferno proxy --listen-addr 0.0.0.0:8080 --backend-addr backend1:3000
//!
//! # Start backend
//! inferno backend --model-path model.bin --listen-addr 0.0.0.0:3000
//!
//! # Start governator
//! inferno governator --providers aws,gcp --database-url postgresql://localhost/inferno
//! ```

pub mod cli_options;
pub mod doctor;
pub mod model_downloader;
pub mod models;
pub mod play;

#[cfg(test)]
mod xet_tests;

pub use cli_options::{Cli, Commands};
