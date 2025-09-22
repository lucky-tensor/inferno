//! # Inferno CLI - Main Entry Point
//!
//! Unified command-line interface for managing Inferno distributed systems components.

use clap::Parser;
use inferno_shared::Result;

mod cli_options;
mod doctor;
mod model_downloader;
mod models;
mod play;

use cli_options::Cli;

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
