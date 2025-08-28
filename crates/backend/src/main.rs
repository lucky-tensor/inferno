//! # Inferno Backend - Main Entry Point
//!
//! AI inference backend server for the Inferno distributed systems platform.

use inferno_backend::BackendConfig;
use inferno_shared::Result;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting Inferno Backend");

    // Load configuration
    let config = BackendConfig::from_env()?;

    info!(
        listen_addr = %config.listen_addr,
        model_path = %config.model_path,
        "Backend server starting"
    );

    // TODO: Implement actual backend server functionality
    info!("Backend server is running (placeholder implementation)");

    // For now, just sleep to keep the process running
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    Ok(())
}
