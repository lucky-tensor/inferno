//! # Inferno CLI
//!
//! Unified command-line interface for Inferno distributed systems.
//! Provides subcommands for starting and managing different components.
//!
//! ## Usage
//!
//! ```bash
//! # Start proxy
//! inferno proxy --port 8080 --backends backend1:3000,backend2:3000
//!
//! # Start backend
//! inferno backend --model llama2 --discovery-lb lb1:8080,lb2:8080  
//!
//! # Start governator
//! inferno governator --providers aws,gcp --metrics prometheus:9090
//! ```

pub mod commands;

pub use commands::*;
