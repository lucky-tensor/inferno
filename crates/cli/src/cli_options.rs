//! CLI options for the unified Inferno command-line interface
//!
//! This module re-exports and organizes CLI options from all Inferno components.

use clap::{Parser, Subcommand};
use inferno_shared::Result;

// Import CLI options from other crates
use inferno_backend::BackendCliOptions;
use inferno_governator::GovernatorCliOptions;
use inferno_proxy::ProxyCliOptions;

/// Inferno - Unified command interface for distributed AI inference platform
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available Inferno commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the proxy server
    Proxy(ProxyCliOptions),

    /// Start the backend inference server
    Backend(BackendCliOptions),

    /// Start the governator cost optimization server
    Governator(GovernatorCliOptions),
}

impl Cli {
    /// Run the selected command
    pub async fn run(self) -> Result<()> {
        match self.command {
            Commands::Proxy(opts) => opts.run().await,
            Commands::Backend(opts) => opts.run().await,
            Commands::Governator(opts) => opts.run().await,
        }
    }
}
