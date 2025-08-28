//! # Inferno Proxy - Main Entry Point
//!
//! High-performance HTTP reverse proxy built with Cloudflare's Pingora framework.

use clap::Parser;
use inferno_proxy::ProxyCliOptions;
use inferno_shared::Result;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line options
    let cli_opts = ProxyCliOptions::parse();

    // Initialize logging with the specified level
    cli_opts.logging.init_logging();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        authors = env!("CARGO_PKG_AUTHORS"),
        "Starting Inferno Proxy"
    );

    // Run the proxy with the parsed options
    cli_opts.run().await
}
