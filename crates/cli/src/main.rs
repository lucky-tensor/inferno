//! # Inferno CLI - Main Entry Point
//!
//! Unified command-line interface for managing Inferno distributed systems components.

use clap::Parser;
use inferno_cli::Cli;
use inferno_shared::Result;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting Inferno CLI");

    let cli = Cli::parse();

    if let Err(e) = cli.run().await {
        error!("Command failed: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
