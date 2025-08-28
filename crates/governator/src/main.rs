//! # Inferno Governator - Main Entry Point
//!
//! Cost optimization and resource governance server for Inferno distributed systems.

use clap::Parser;
use inferno_governator::GovernatorCliOptions;
use inferno_shared::Result;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line options
    let cli_opts = GovernatorCliOptions::parse();

    // Initialize logging with the specified level
    init_logging(&cli_opts.log_level);

    // Run the governator with the parsed options
    cli_opts.run().await
}

/// Initialize logging with a specific level
fn init_logging(level_str: &str) {
    let level = match level_str.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set logging subscriber");
}
