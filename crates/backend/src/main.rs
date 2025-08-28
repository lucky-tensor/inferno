//! # Inferno Backend - Main Entry Point
//!
//! AI inference backend server for the Inferno distributed systems platform.

use clap::Parser;
use inferno_backend::BackendCliOptions;
use inferno_shared::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line options
    let cli_opts = BackendCliOptions::parse();

    // Initialize logging with the specified level
    cli_opts.logging.init_logging();

    // Run the backend with the parsed options
    cli_opts.run().await
}
