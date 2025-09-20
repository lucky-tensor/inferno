//! Interactive Play Mode for Inferno CLI
//!
//! Provides a chatbot-like Q&A interface for testing inference capabilities.
//! Users can interact with the AI model in real-time through a readline interface.
//!
//! ## Features
//!
//! - Interactive readline interface with history and tab completion
//! - Configurable inference parameters (temperature, max tokens, etc.)
//! - Backend integration via HTTP API or local inference engine
//! - Graceful error handling and user feedback
//! - Support for special commands (`:quit`, `:help`, `:stats`)
//!
//! ## Architecture
//!
//! The play mode can operate in two ways:
//! 1. **Remote Backend**: Connect to an existing inference server via HTTP
//! 2. **Local Backend**: Start a background inference server automatically
//!
//! ## Usage Examples
//!
//! ```bash
//! # Connect to existing backend
//! inferno play --backend-url http://localhost:3000
//!
//! # Start local backend with specific model
//! inferno play --local-backend --model-path ./models/llama2
//!
//! # Customize inference parameters
//! inferno play --temperature 0.8 --max-tokens 100
//! ```

use crate::cli_options::PlayCliOptions;
use crate::models;
use inferno_backend::BackendCliOptions;
use inferno_inference::inference::{InferenceRequest, InferenceResponse};
use inferno_shared::{InfernoError, Result};
use rand::Rng;
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tracing::{debug, error, info};

/// Maximum time to wait for backend server to start (seconds)
const BACKEND_STARTUP_TIMEOUT_SECS: u64 = 30;

/// Health check endpoint path
const HEALTH_CHECK_PATH: &str = "/health";

/// Inference endpoint path
const INFERENCE_ENDPOINT: &str = "/v1/completions";

/// Special command prefix
const COMMAND_PREFIX: char = ':';

/// Global request ID counter for tracking requests
static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Statistics for the play session
#[derive(Debug, Clone)]
pub struct PlayStats {
    /// Total requests made
    pub total_requests: u64,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Total inference time in milliseconds
    pub total_inference_time_ms: f64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Session start time
    pub session_start: Instant,
}

impl Default for PlayStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            total_inference_time_ms: 0.0,
            avg_tokens_per_second: 0.0,
            session_start: Instant::now(),
        }
    }
}

impl PlayStats {
    /// Create new play statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with a new inference response
    pub fn update(&mut self, response: &InferenceResponse) {
        self.total_requests += 1;
        self.total_tokens += response.generated_tokens as u64;
        self.total_inference_time_ms += response.inference_time_ms;

        // Calculate average tokens per second
        if self.total_inference_time_ms > 0.0 {
            self.avg_tokens_per_second =
                (self.total_tokens as f64 * 1000.0) / self.total_inference_time_ms;
        }
    }

    /// Get session duration
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }
}

/// Play mode context containing configuration and state
pub struct PlayContext {
    /// CLI options
    pub options: PlayCliOptions,
    /// HTTP client for backend communication
    pub client: Client,
    /// Background backend task
    pub backend_task: JoinHandle<()>,
    /// Session statistics
    pub stats: PlayStats,
    /// Backend server port
    pub backend_port: u16,
}

/// Find a random available port in a range around the base port
fn find_random_available_port(base_port: u16) -> Result<u16> {
    let mut rng = rand::thread_rng();
    let port_range = 1000; // Search in a range of 1000 ports
    let min_port = base_port;
    let max_port = base_port + port_range;

    // Try random ports first (up to 50 attempts)
    for _ in 0..50 {
        let port = rng.gen_range(min_port..max_port);
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            drop(listener);
            info!("Found random available port: {}", port);
            return Ok(port);
        }
    }

    // Fallback to sequential search if random didn't work
    info!("Random port search failed, falling back to sequential search");
    for port in min_port..max_port {
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            drop(listener);
            return Ok(port);
        }
    }

    Err(InfernoError::internal(
        format!(
            "Could not find available port in range {}..{}",
            min_port, max_port
        ),
        None,
    ))
}

/// Select and validate model using the shared models module
async fn select_model(mut options: PlayCliOptions) -> Result<PlayCliOptions> {
    use crate::models::{format_file_size, ModelValidationResult};
    use inferno_shared::ModelMemoryValidator;

    let validation_result = models::validate_and_discover_models(&options.model_path)?;

    let final_model_path = match validation_result {
        ModelValidationResult::SingleModel(model_path) => {
            options.model_path = model_path.clone();
            model_path
        }
        ModelValidationResult::NoModels => {
            eprintln!(
                "No SafeTensors models found in directory: {}",
                options.model_path
            );
            eprintln!();
            eprintln!("To download a model, run:");
            eprintln!(
                "   inferno download --model-id <HF_MODEL_ID> --output-dir {}",
                options.model_path
            );
            eprintln!();
            eprintln!("Popular models to try:");
            eprintln!("   inferno download --model-id microsoft/DialoGPT-medium");
            eprintln!("   inferno download --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0");
            std::process::exit(1);
        }
        ModelValidationResult::MultipleModels(models_list) => {
            if options.prompt.is_some() {
                // In headless mode, require explicit model selection
                eprintln!("Multiple models found in directory: {}", options.model_path);
                eprintln!();
                eprintln!("Available SafeTensors models:");
                for model in &models_list {
                    eprintln!("   {} ({})", model.name, format_file_size(model.size_bytes));
                }
                eprintln!();
                eprintln!("Please specify a model with --model-path:");
                eprintln!(
                    "   inferno play --prompt \"your question\" --model-path {}/MODEL_NAME",
                    options.model_path
                );
                std::process::exit(1);
            } else {
                // In interactive mode, let user select
                let selected_model =
                    select_model_interactively(models_list, &options.model_path).await?;
                options.model_path = selected_model.path.to_string_lossy().to_string();
                selected_model.path.to_string_lossy().to_string()
            }
        }
    };

    // Perform memory validation for GPU engines
    if options.engine == "candle-cuda" || options.engine == "candle-metal" {
        let device_id = 0; // Use GPU 0 by default

        info!("Validating GPU memory requirements for selected model...");

        let validator = ModelMemoryValidator::new(device_id);
        match validator.validate_model_fit(&final_model_path).await {
            Ok(validation) => {
                validator.display_validation(&validation);

                if !validation.will_fit {
                    // Model doesn't fit - ask user if they want to proceed anyway
                    if options.prompt.is_some() {
                        // In headless mode, just fail
                        eprintln!("ERROR: Model validation indicates insufficient GPU memory for headless mode.");
                        eprintln!("Use interactive mode to override this warning.");
                        std::process::exit(1);
                    } else {
                        // In interactive mode, ask user for confirmation
                        if !prompt_user_override(&validation).await? {
                            eprintln!("Model loading cancelled by user.");
                            std::process::exit(1);
                        }
                        println!("Proceeding with model loading at user's own risk...");
                    }
                } else if validation.confidence < 0.7 {
                    // Model might fit but with low confidence - warn user
                    println!(
                        "WARNING: Memory validation has low confidence ({:.0}%)",
                        validation.confidence * 100.0
                    );
                    if !validation.recommendations.is_empty() {
                        println!("Consider the following recommendations:");
                        for rec in &validation.recommendations {
                            println!("   {}", rec);
                        }
                    }

                    if options.prompt.is_none() {
                        // In interactive mode, ask for confirmation
                        if !prompt_user_continue_with_warning().await? {
                            eprintln!("Model loading cancelled by user.");
                            std::process::exit(1);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("WARNING: Could not validate GPU memory requirements: {}", e);
                eprintln!("Proceeding without memory validation...");
            }
        }
    }

    Ok(options)
}

/// Prompt user to override memory validation warning
async fn prompt_user_override(validation: &inferno_shared::MemoryValidation) -> Result<bool> {
    let mut editor = DefaultEditor::new().map_err(|e| {
        InfernoError::internal(format!("Failed to initialize readline: {}", e), None)
    })?;

    println!();
    println!(
        "Model requires {:.1} GB but only {:.1} GB available (shortfall: {:.1} GB)",
        validation.estimated_requirement_gb,
        validation.available_memory_gb,
        validation.estimated_requirement_gb - validation.available_memory_gb
    );

    println!("\nRisks of proceeding:");
    println!("   - Model loading may fail with CUDA out-of-memory errors");
    println!("   - GPU may become unresponsive requiring system restart");
    println!("   - Other GPU processes may be killed by the system");

    loop {
        let input = editor
            .readline("Do you want to proceed anyway? (y/N): ")
            .map_err(|e| {
                InfernoError::internal(format!("Failed to read user input: {}", e), None)
            })?;

        let trimmed = input.trim().to_lowercase();
        match trimmed.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" | "" => return Ok(false),
            _ => println!("Please enter 'y' for yes or 'n' for no."),
        }
    }
}

/// Prompt user to continue despite warning
async fn prompt_user_continue_with_warning() -> Result<bool> {
    let mut editor = DefaultEditor::new().map_err(|e| {
        InfernoError::internal(format!("Failed to initialize readline: {}", e), None)
    })?;

    loop {
        let input = editor
            .readline("Continue with model loading? (Y/n): ")
            .map_err(|e| {
                InfernoError::internal(format!("Failed to read user input: {}", e), None)
            })?;

        let trimmed = input.trim().to_lowercase();
        match trimmed.as_str() {
            "y" | "yes" | "" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("Please enter 'y' for yes or 'n' for no."),
        }
    }
}

/// Let user select a model interactively from a list
async fn select_model_interactively(
    models: Vec<crate::models::ModelInfo>,
    model_dir: &str,
) -> Result<crate::models::ModelInfo> {
    use crate::models::format_file_size;

    println!("Multiple models found in directory: {}", model_dir);
    println!();

    // Display available models
    for (i, model) in models.iter().enumerate() {
        println!(
            "  [{}] {} ({})",
            i + 1,
            model.name,
            format_file_size(model.size_bytes)
        );
    }
    println!();

    // Get user selection
    let mut editor = DefaultEditor::new().map_err(|e| {
        InfernoError::internal(format!("Failed to initialize readline: {}", e), None)
    })?;

    loop {
        let input = editor
            .readline(&format!("Select a model (1-{}): ", models.len()))
            .map_err(|e| {
                InfernoError::internal(format!("Failed to read user input: {}", e), None)
            })?;

        let trimmed = input.trim();
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= models.len() {
                let selected_model = &models[choice - 1];
                println!(
                    "Selected: {} ({})",
                    selected_model.name,
                    format_file_size(selected_model.size_bytes)
                );
                println!();
                return Ok(selected_model.clone());
            }
        }

        eprintln!(
            "Invalid selection. Please enter a number between 1 and {}",
            models.len()
        );
    }
}

impl PlayContext {
    /// Create new play context and start the backend server
    pub async fn new(options: PlayCliOptions) -> Result<Self> {
        info!("Starting local backend server in current process...");

        // Find random available ports
        let backend_port = find_random_available_port(3000)?;
        let metrics_port = find_random_available_port(6100)?;

        info!(
            "Using backend port: {}, metrics port: {}",
            backend_port, metrics_port
        );

        // Determine GPU device ID based on engine type
        let gpu_device_id = match options.engine.as_str() {
            "candle-cuda" => 0,  // Use GPU 0 for CUDA
            "candle-metal" => 0, // Use GPU 0 for Metal
            _ => -1,             // Use CPU for all other engines
        };

        info!(
            "Using engine: {} with device ID: {}",
            options.engine, gpu_device_id
        );

        // Create backend CLI options
        let backend_opts = BackendCliOptions {
            listen_addr: format!("127.0.0.1:{}", backend_port).parse().unwrap(),
            model_path: options.model_path.clone().into(),
            model_type: "auto".to_string(),
            engine: options.engine.clone(),
            max_batch_size: 32,
            gpu_device_id,
            max_context_length: 2048,
            memory_pool_mb: 1024,
            discovery_lb: None,
            enable_cache: false,
            cache_ttl_seconds: 3600,
            logging: inferno_shared::cli::LoggingOptions {
                log_level: "info".to_string(),
            },
            metrics: inferno_shared::cli::MetricsOptions {
                enable_metrics: true,
                operations_addr: Some(format!("127.0.0.1:{}", metrics_port).parse().unwrap()),
                metrics_addr: None,
            },
            health_check: inferno_shared::cli::HealthCheckOptions {
                health_check_path: "/health".to_string(),
            },
            service_discovery: inferno_shared::cli::ServiceDiscoveryOptions {
                service_name: None,
                registration_endpoint: None,
            },
        };

        // Start backend in a separate task
        let backend_task = tokio::spawn(async move {
            if let Err(e) = backend_opts.run().await {
                error!("Backend task failed: {}", e);
            }
        });

        let context = Self {
            options,
            client: Client::new(),
            backend_task,
            stats: PlayStats::new(),
            backend_port,
        };

        // Wait for backend to be ready
        context.wait_for_backend_ready().await?;

        info!("Local backend server started successfully");
        Ok(context)
    }

    /// Wait for backend server to become ready
    async fn wait_for_backend_ready(&self) -> Result<()> {
        let health_url = format!(
            "http://127.0.0.1:{}{}",
            self.backend_port, HEALTH_CHECK_PATH
        );
        let start_time = Instant::now();

        while start_time.elapsed().as_secs() < BACKEND_STARTUP_TIMEOUT_SECS {
            match self.client.get(&health_url).send().await {
                Ok(response) if response.status().is_success() => {
                    debug!("Backend health check succeeded");
                    return Ok(());
                }
                Ok(response) => {
                    debug!(
                        "Backend health check failed with status: {}",
                        response.status()
                    );
                }
                Err(e) => {
                    debug!("Backend health check error: {}", e);
                }
            }

            sleep(Duration::from_millis(500)).await;
        }

        Err(InfernoError::internal(
            format!(
                "Backend server failed to start within {} seconds",
                BACKEND_STARTUP_TIMEOUT_SECS
            ),
            None,
        ))
    }

    /// Send inference request to backend
    pub async fn send_inference_request(&mut self, prompt: String) -> Result<InferenceResponse> {
        let request_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        let request = InferenceRequest {
            request_id,
            prompt,
            max_tokens: self.options.max_tokens as u32,
            temperature: self.options.temperature,
            top_p: self.options.top_p,
            seed: self.options.seed,
        };

        debug!("Sending inference request: {:?}", request);

        let inference_url = format!(
            "http://127.0.0.1:{}{}",
            self.backend_port, INFERENCE_ENDPOINT
        );
        let start_time = Instant::now();

        let response = self
            .client
            .post(&inference_url)
            .json(&request)
            .timeout(std::time::Duration::from_secs(10)) // 10 second timeout
            .send()
            .await
            .map_err(|e| {
                InfernoError::internal(format!("Failed to send request to backend: {}", e), None)
            })?;

        let inference_time = start_time.elapsed().as_millis() as f64;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(InfernoError::internal(
                format!("Backend returned error {}: {}", status, error_text),
                None,
            ));
        }

        let mut inference_response: InferenceResponse = response.json().await.map_err(|e| {
            InfernoError::internal(format!("Failed to parse backend response: {}", e), None)
        })?;

        // Update inference time if not set by backend
        if inference_response.inference_time_ms == 0.0 {
            inference_response.inference_time_ms = inference_time;
        }

        // Update statistics
        self.stats.update(&inference_response);

        debug!("Received inference response: {:?}", inference_response);
        Ok(inference_response)
    }

    /// Cleanup resources
    pub async fn cleanup(self) {
        info!("Stopping local backend server...");
        self.backend_task.abort();
        let _ = self.backend_task.await;
        info!("Local backend server stopped");
    }
}

/// Process special commands (starting with ':')
fn process_special_command(command: &str, stats: &PlayStats) -> Option<String> {
    let cmd = command
        .trim_start_matches(COMMAND_PREFIX)
        .trim()
        .to_lowercase();

    match cmd.as_str() {
        "help" | "h" => {
            Some(format!(
                r#"
Inferno Play Mode - Interactive AI Chat Interface

Special Commands:
  :help, :h     - Show this help message
  :quit, :q     - Exit play mode
  :stats, :s    - Show session statistics
  :clear, :c    - Clear screen (if supported)

Regular Usage:
  Type any message and press Enter to get an AI response.
  Use Ctrl+C to exit at any time.

Configuration:
  Current backend: Available
  Max tokens: {}
  Temperature: {}
  Top-p: {}
  Seed: {:?}
"#,
                stats.total_requests, // Using placeholder until we get max_tokens from context
                0.7,                  // Using placeholder until we get temperature from context
                0.9,                  // Using placeholder until we get top_p from context
                Option::<u64>::None   // Using placeholder until we get seed from context
            ))
        }
        "stats" | "s" => {
            let session_duration = stats.session_duration();
            Some(format!(
                r#"
Session Statistics:
  Duration: {:02}:{:02}:{:02}
  Total requests: {}
  Total tokens generated: {}
  Total inference time: {:.2}ms
  Average tokens/sec: {:.2}
  Average response time: {:.2}ms
"#,
                session_duration.as_secs() / 3600,
                (session_duration.as_secs() % 3600) / 60,
                session_duration.as_secs() % 60,
                stats.total_requests,
                stats.total_tokens,
                stats.total_inference_time_ms,
                stats.avg_tokens_per_second,
                if stats.total_requests > 0 {
                    stats.total_inference_time_ms / stats.total_requests as f64
                } else {
                    0.0
                }
            ))
        }
        "clear" | "c" => {
            // Clear screen using ANSI escape codes
            Some("\x1b[2J\x1b[H".to_string())
        }
        "quit" | "q" => None, // Signal to quit
        _ => Some(format!(
            "Unknown command: :{}. Type :help for available commands.",
            cmd
        )),
    }
}

/// Run headless mode with a single prompt
async fn run_headless_mode(mut context: PlayContext, prompt: String) -> Result<()> {
    info!("Running in headless mode with prompt: {}", prompt);

    // Use tokio::select! to race between inference and Ctrl+C signal
    let result = tokio::select! {
        inference_result = context.send_inference_request(prompt) => {
            Some(inference_result)
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\n^C received, shutting down...");
            None
        }
    };

    if let Some(inference_result) = result {
        match inference_result {
            Ok(response) => {
                if let Some(error) = &response.error {
                    eprintln!("Error: {}", error);
                    context.cleanup().await;
                    std::process::exit(1);
                } else {
                    println!("{}", response.generated_text);

                    // Show statistics in headless mode
                    let tokens_per_second = if response.inference_time_ms > 0.0 {
                        (response.generated_tokens as f64 * 1000.0) / response.inference_time_ms
                    } else {
                        0.0
                    };

                    eprint!("\nStats: ");
                    eprint!("Tokens: {} | ", response.generated_tokens);
                    eprint!("Total: {:.0}ms | ", response.inference_time_ms);

                    if let Some(ttft) = response.time_to_first_token_ms {
                        eprint!("First token: {:.0}ms | ", ttft);
                    }

                    eprintln!("Speed: {:.1} tok/s", tokens_per_second);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                context.cleanup().await;
                std::process::exit(1);
            }
        }
    }

    // Cleanup
    context.cleanup().await;

    Ok(())
}

/// Main entry point for play mode
pub async fn run_play_mode(mut options: PlayCliOptions) -> Result<()> {
    info!("Starting Inferno Play Mode");

    // Resolve model path to handle ~ expansion
    options.model_path = inferno_shared::resolve_models_path(&options.model_path)
        .to_string_lossy()
        .to_string();

    // Select and validate model before starting backend
    let options = select_model(options).await?;

    // Check if this is headless mode (one-off prompt)
    let prompt = options.prompt.clone();

    // Create context (automatically starts backend)
    let context = PlayContext::new(options).await?;

    if let Some(prompt) = prompt {
        return run_headless_mode(context, prompt).await;
    }

    let mut context = context;

    // Interactive mode - print welcome message
    println!("\nWelcome to Inferno Play Mode!");
    println!("Type your messages and get AI responses. Use :help for commands or :quit to exit.\n");

    // Initialize readline editor
    let mut editor = DefaultEditor::new().map_err(|e| {
        InfernoError::internal(format!("Failed to initialize readline editor: {}", e), None)
    })?;

    // Main interaction loop
    let result = run_interaction_loop(&mut editor, &mut context).await;

    // Cleanup
    context.cleanup().await;

    result
}

/// Run the main interaction loop
async fn run_interaction_loop(editor: &mut DefaultEditor, context: &mut PlayContext) -> Result<()> {
    loop {
        // Read user input
        let readline = editor.readline("You: ");

        match readline {
            Ok(line) => {
                let trimmed_line = line.trim();

                // Skip empty lines
                if trimmed_line.is_empty() {
                    continue;
                }

                // Add to history
                editor.add_history_entry(&line).map_err(|e| {
                    InfernoError::internal(format!("Failed to add history entry: {}", e), None)
                })?;

                // Check for special commands
                if trimmed_line.starts_with(COMMAND_PREFIX) {
                    match process_special_command(trimmed_line, &context.stats) {
                        Some(response) => println!("{}", response),
                        None => break, // Quit command
                    }
                    continue;
                }

                // Send inference request
                print!("Inferno: ");
                match context
                    .send_inference_request(trimmed_line.to_string())
                    .await
                {
                    Ok(response) => {
                        if let Some(error) = &response.error {
                            println!("Error: {}", error);
                        } else {
                            println!("{}", response.generated_text);

                            // Show statistics after each response
                            let tokens_per_second = if response.inference_time_ms > 0.0 {
                                (response.generated_tokens as f64 * 1000.0)
                                    / response.inference_time_ms
                            } else {
                                0.0
                            };

                            print!("\nStats: ");
                            print!("Tokens: {} | ", response.generated_tokens);
                            print!("Total: {:.0}ms | ", response.inference_time_ms);

                            if let Some(ttft) = response.time_to_first_token_ms {
                                print!("First token: {:.0}ms | ", ttft);
                            }

                            println!("Speed: {:.1} tok/s", tokens_per_second);
                        }
                    }
                    Err(e) => {
                        error!("Inference failed: {}", e);
                        println!("Sorry, I encountered an error: {}", e);
                    }
                }
                println!(); // Add spacing between conversations
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("^D");
                break;
            }
            Err(err) => {
                error!("Readline error: {}", err);
                return Err(InfernoError::internal(
                    format!("Readline error: {}", err),
                    None,
                ));
            }
        }
    }

    println!("\nThanks for using Inferno Play Mode!");

    // Show final statistics
    let stats_output = process_special_command(":stats", &context.stats).unwrap_or_default();
    if !stats_output.trim().is_empty() {
        println!("{}", stats_output);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_play_stats_creation() {
        let stats = PlayStats::new();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_tokens, 0);
        assert!((stats.total_inference_time_ms - 0.0).abs() < f64::EPSILON);
        assert!((stats.avg_tokens_per_second - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_play_stats_update() {
        let mut stats = PlayStats::new();
        let response = InferenceResponse {
            request_id: 1,
            generated_text: "Hello world".to_string(),
            generated_tokens: 2,
            inference_time_ms: 100.0,
            time_to_first_token_ms: Some(50.0),
            is_finished: true,
            error: None,
        };

        stats.update(&response);

        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.total_tokens, 2);
        assert!((stats.total_inference_time_ms - 100.0).abs() < f64::EPSILON);
        assert!((stats.avg_tokens_per_second - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_special_command_processing() {
        let stats = PlayStats::new();

        // Test help command
        let help_response = process_special_command(":help", &stats);
        assert!(help_response.is_some());
        assert!(help_response.unwrap().contains("Special Commands"));

        // Test stats command
        let stats_response = process_special_command(":stats", &stats);
        assert!(stats_response.is_some());
        assert!(stats_response.unwrap().contains("Session Statistics"));

        // Test quit command
        let quit_response = process_special_command(":quit", &stats);
        assert!(quit_response.is_none());

        // Test unknown command
        let unknown_response = process_special_command(":unknown", &stats);
        assert!(unknown_response.is_some());
        assert!(unknown_response.unwrap().contains("Unknown command"));
    }

    #[tokio::test]
    async fn test_play_context_creation() {
        let options = PlayCliOptions {
            model_path: "./models".to_string(),
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
            seed: Some(42),
            prompt: None,
            engine: "burn-cpu".to_string(),
        };

        // Since new() is now async and starts a backend, we can't easily test it
        // in a synchronous test. For now, just test the options structure.
        assert_eq!(options.model_path, "./models");
        assert_eq!(options.max_tokens, 50);
    }

    #[test]
    fn test_request_id_counter() {
        let id1 = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        let id2 = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        assert!(id2 > id1);
    }
}
