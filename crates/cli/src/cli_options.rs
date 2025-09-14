//! CLI options for the unified Inferno command-line interface
//!
//! This module re-exports and organizes CLI options from all Inferno components.

use clap::{Args, Parser, Subcommand};
use inferno_shared::Result;

// Import CLI options from other crates
use inferno_backend::BackendCliOptions;
use inferno_governator::GovernatorCliOptions;
use inferno_proxy::ProxyCliOptions;

use crate::model_downloader;

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

    /// Download models from Hugging Face Hub
    Download(DownloadCliOptions),
}

/// CLI options for downloading models from Hugging Face Hub
#[derive(Args, Debug)]
pub struct DownloadCliOptions {
    /// Hugging Face model ID (e.g., "microsoft/DialoGPT-medium")
    #[arg(short = 'i', long = "model-id", value_name = "HF_MODEL_ID")]
    pub model_id: String,

    /// Output directory for downloaded model
    #[arg(
        short = 'o',
        long = "output-dir",
        value_name = "PATH",
        default_value = "./models"
    )]
    pub output_dir: String,

    /// Hugging Face API token (for gated/private models). Can also use HF_TOKEN environment variable
    #[arg(long = "hf-token", value_name = "TOKEN")]
    pub hf_token: Option<String>,

    /// Resume interrupted download
    #[arg(long = "resume")]
    pub resume: bool,

    /// Use Git LFS instead of the default xet backend
    #[arg(long = "use-lfs")]
    pub use_lfs: bool,
}

impl Cli {
    /// Run the selected command
    pub async fn run(self) -> Result<()> {
        match self.command {
            Commands::Proxy(opts) => opts.run().await,
            Commands::Backend(opts) => opts.run().await,
            Commands::Governator(opts) => opts.run().await,
            Commands::Download(opts) => opts.run().await,
        }
    }
}

impl DownloadCliOptions {
    /// Run the download command
    pub async fn run(self) -> Result<()> {
        println!("Downloading model: {}", self.model_id);
        println!("Output directory: {}", self.output_dir);
        if self.hf_token.is_some() {
            println!("Using Hugging Face API token for authentication");
        }
        if self.resume {
            println!("Resume mode: Will attempt to resume interrupted downloads");
        }
        if self.use_lfs {
            println!("LFS mode: Will use Git LFS instead of default xet backend");
        }
        println!("---");

        model_downloader::download_model(
            &self.model_id,
            &self.output_dir,
            self.hf_token.as_ref(),
            self.resume,
            !self.use_lfs, // Xet is default, LFS when flag is set
        )
        .await
        .map_err(|e| {
            inferno_shared::InfernoError::internal(format!("Model download failed: {}", e), None)
        })?;

        // Download completion messages are handled by the download functions

        Ok(())
    }
}
