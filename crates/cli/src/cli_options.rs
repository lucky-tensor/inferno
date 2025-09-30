//! CLI options for the unified Inferno command-line interface
//!
//! This module re-exports and organizes CLI options from all Inferno components.

use clap::{Args, Parser, Subcommand};
use inferno_shared::Result;

// Import CLI options from other crates
use inferno_backend::BackendCliOptions;
use inferno_governator::GovernatorCliOptions;
use inferno_proxy::ProxyCliOptions;

use crate::doctor;
use crate::model_downloader;

/// Determine the best default inference engine based on compiled features
#[allow(unreachable_code)]
fn default_engine() -> String {
    // Priority order: GPU engines first (faster), then CPU engines
    // Always prefer CUDA if available
    "candle-cuda".to_string()
}

/// CLI options for play mode
#[derive(Parser, Debug, Clone)]
pub struct PlayCliOptions {
    /// Model directory or specific model file path
    #[arg(
        short = 'm',
        long = "model-path",
        value_name = "PATH",
        default_value_t = inferno_shared::default_models_dir_string()
    )]
    pub model_path: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 50)]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 = deterministic)
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// Top-p sampling parameter
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    /// Random seed for reproducible results
    #[arg(long)]
    pub seed: Option<u64>,

    /// Run in headless mode with a single prompt
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Option<String>,

    /// Internal: Inference engine (automatically selected based on compiled features)
    #[arg(skip = default_engine())]
    pub engine: String,
}

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

    /// Run system diagnostics and check hardware compatibility
    Doctor(DoctorCliOptions),

    /// Interactive Q&A mode for testing inference
    Play(PlayCliOptions),
}

/// CLI options for system diagnostics and hardware compatibility checking
#[derive(Args, Debug)]
pub struct DoctorCliOptions {
    /// Model directory to scan for compatibility
    #[arg(
        short = 'm',
        long = "model-dir",
        value_name = "PATH",
        default_value_t = inferno_shared::default_models_dir_string()
    )]
    pub model_dir: String,

    /// Enable verbose output with detailed diagnostics
    #[arg(short = 'v', long = "verbose")]
    pub verbose: bool,

    /// Output format (table, json, yaml)
    #[arg(
        short = 'f',
        long = "format",
        value_name = "FORMAT",
        default_value = "table"
    )]
    pub format: String,
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
        default_value_t = inferno_shared::default_models_dir_string()
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
            Commands::Doctor(opts) => opts.run().await,
            Commands::Play(opts) => crate::play::run_play_mode(opts).await,
        }
    }
}

impl DownloadCliOptions {
    /// Run the download command
    pub async fn run(self) -> Result<()> {
        // Resolve the models path to handle ~ expansion
        let resolved_output_dir = inferno_shared::resolve_models_path(&self.output_dir);
        let output_dir_str = resolved_output_dir.to_string_lossy().to_string();

        println!("Downloading model: {}", self.model_id);
        println!("Output directory: {}", output_dir_str);
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
            &output_dir_str,
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

impl DoctorCliOptions {
    /// Run the doctor command
    pub async fn run(self) -> Result<()> {
        doctor::run_diagnostics(self).await
    }
}
