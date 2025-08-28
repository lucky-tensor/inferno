//! # Inferno Governator - Main Entry Point
//!
//! Cost optimization and resource governance server for Inferno distributed systems.

use clap::Parser;
use inferno_governator::GovernatorCliOptions;
use inferno_shared::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line options
    let cli_opts = GovernatorCliOptions::parse();

    // Initialize logging with the specified level
    cli_opts.logging.init_logging();

    // Run the governator with the parsed options
    cli_opts.run().await
}
