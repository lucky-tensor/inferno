//! # Inferno CLI - Main Entry Point
//!
//! Unified command-line interface for managing Inferno distributed systems components.

use clap::Parser;
use inferno_cli::Cli;
use inferno_shared::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with INFERNO_LOG environment variable, defaulting to warn
    let log_level = std::env::var("INFERNO_LOG").unwrap_or_else(|_| "warn".to_string());
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(log_level)
        .init();

    // Parse and run the CLI
    let cli = Cli::parse();
    cli.run().await
}
