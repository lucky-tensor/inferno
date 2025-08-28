//! # Inferno CLI - Main Entry Point
//!
//! Unified command-line interface for managing Inferno distributed systems components.

use clap::Parser;
use inferno_cli::Cli;
use inferno_shared::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize basic logging for the CLI
    tracing_subscriber::fmt().with_target(false).init();

    // Parse and run the CLI
    let cli = Cli::parse();
    cli.run().await
}
