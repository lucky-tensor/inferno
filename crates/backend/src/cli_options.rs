//! CLI options for the Inferno Backend
//!
//! This module defines the command-line interface options for the backend server,
//! which can be used both standalone and integrated into the unified CLI.

use crate::health::HealthService;
use crate::BackendConfig;
use clap::Parser;
use http_body_util::{combinators::BoxBody, BodyExt, Empty, Full};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{body::Incoming, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use inferno_inference::{
    config::InfernoConfig,
    inference::{create_engine, InferenceEngine, InferenceRequest},
};
use inferno_shared::{
    HealthCheckOptions, InfernoError, LoggingOptions, MetricsCollector, MetricsOptions, Result,
    ServiceDiscoveryOptions,
};
use serde_json;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, warn};

/// Engine is always llama-burn
fn default_engine() -> String {
    "llama-burn".to_string()
}

/// Inferno Backend - AI inference backend server
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct BackendCliOptions {
    /// Address to listen on
    #[arg(
        short,
        long,
        default_value = "127.0.0.1:3000",
        env = "INFERNO_BACKEND_LISTEN_ADDR"
    )]
    pub listen_addr: SocketAddr,

    /// Path to the AI model file or directory
    #[arg(
        short,
        long,
        default_value = "",  // Will be handled in Default impl
        env = "INFERNO_MODEL_PATH"
    )]
    pub model_path: PathBuf,

    /// Model type/format (e.g., llama, gguf, onnx)
    #[arg(long, default_value = "auto", env = "INFERNO_MODEL_TYPE")]
    pub model_type: String,

    /// Inference engine (always llama-burn)
    #[arg(skip = default_engine())]
    pub engine: String,

    /// Maximum batch size for inference
    #[arg(long, default_value_t = 32, env = "INFERNO_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// GPU device ID to use (GPU only - no CPU support)
    #[arg(long, default_value_t = 0, env = "INFERNO_GPU_DEVICE_ID")]
    pub gpu_device_id: i32,

    /// Maximum context length
    #[arg(long, default_value_t = 2048, env = "INFERNO_MAX_CONTEXT_LENGTH")]
    pub max_context_length: usize,

    /// Memory pool size in MB
    #[arg(long, default_value_t = 1024, env = "INFERNO_MEMORY_POOL_MB")]
    pub memory_pool_mb: usize,

    /// Discovery load balancer addresses (comma-separated)
    #[arg(short = 'd', long, env = "INFERNO_DISCOVERY_LB")]
    pub discovery_lb: Option<String>,

    /// Enable request caching
    #[arg(long, default_value_t = true, env = "INFERNO_ENABLE_CACHE")]
    pub enable_cache: bool,

    /// Cache TTL in seconds
    #[arg(long, default_value_t = 3600, env = "INFERNO_CACHE_TTL_SECONDS")]
    pub cache_ttl_seconds: u64,

    #[command(flatten)]
    pub logging: LoggingOptions,

    #[command(flatten)]
    pub metrics: MetricsOptions,

    #[command(flatten)]
    pub health_check: HealthCheckOptions,

    #[command(flatten)]
    pub service_discovery: ServiceDiscoveryOptions,
}

impl BackendCliOptions {
    /// Run the backend server with the configured options
    pub async fn run(mut self) -> Result<()> {
        info!("Starting Inferno Backend");

        // Set default model path if empty
        if self.model_path.as_os_str().is_empty() {
            self.model_path = inferno_shared::default_models_dir();
        }

        // Convert CLI options to BackendConfig
        let config = self.to_config()?;

        info!(
            listen_addr = %config.listen_addr,
            model_path = ?config.model_path,
            gpu_device_id = config.gpu_device_id,
            max_batch_size = config.max_batch_size,
            "Backend server starting"
        );

        // Initialize the inference engine with the model
        info!("Initializing inference engine...");

        // Convert backend config to inference config format
        // If model_path is a file, use its parent directory as models dir
        let (models_dir, model_name) = if config.model_path.is_file() {
            let parent = config
                .model_path
                .parent()
                .ok_or_else(|| InfernoError::Configuration {
                    message: "Model file has no parent directory".to_string(),
                    source: None,
                })?;
            // For specific SafeTensors files, we want to use the directory containing the file
            // and let the inference engine auto-discover, rather than treating the filename as a model name
            (parent.to_string_lossy().to_string(), String::new())
        } else {
            // Assume it's a directory and auto-discover
            (
                config.model_path.to_string_lossy().to_string(),
                String::new(),
            )
        };

        let inference_config = InfernoConfig {
            model_path: models_dir,
            model_name,
            device_id: config.gpu_device_id,
            max_batch_size: config.max_batch_size,
            max_sequence_length: config.max_context_length,
            ..Default::default()
        };

        info!("Using llama-burn inference engine");

        // Create and initialize the inference engine
        let mut engine = create_engine();

        if let Err(e) = engine.initialize(inference_config).await {
            warn!("Failed to initialize inference engine: {}", e);
            return Err(InfernoError::Configuration {
                message: format!("Inference engine initialization failed: {}", e),
                source: None,
            });
        }

        info!("  Inference engine ready to receive requests!");

        // Start HTTP inference server
        let inference_server_task = {
            let listen_addr = config.listen_addr;
            let engine = Arc::new(tokio::sync::Mutex::new(engine));

            tokio::spawn(async move {
                if let Err(e) = start_inference_server(listen_addr, engine).await {
                    warn!("HTTP inference server failed: {}", e);
                }
            })
        };

        info!("Backend server is running");

        // Start HTTP metrics server if enabled
        let metrics_task = if config.enable_metrics {
            let metrics = Arc::new(MetricsCollector::new());
            let health_service = HealthService::new(Arc::clone(&metrics), config.operations_addr);

            info!(
                operations_addr = %config.operations_addr,
                "Starting HTTP operations server"
            );

            Some(tokio::spawn(async move {
                if let Err(e) = health_service.start().await {
                    warn!(
                        error = %e,
                        "HTTP metrics server failed"
                    );
                }
            }))
        } else {
            None
        };

        // Perform service registration if configured
        if let Some(registration_endpoint) = self.service_discovery.registration_endpoint.as_ref() {
            info!(
                "Attempting to register with service discovery at: {}",
                registration_endpoint
            );

            // Create registration manager
            let lb_addrs = self
                .discovery_lb
                .as_ref()
                .map(|s| {
                    s.split(',')
                        .filter_map(|addr| addr.trim().parse::<SocketAddr>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let registration = crate::registration::ServiceRegistration::new(
                self.listen_addr,
                lb_addrs,
                config.operations_addr.port(),
                self.service_discovery.service_name.clone(),
            );

            // Attempt registration
            if let Err(e) = registration.register().await {
                warn!("Failed to register with service discovery: {}", e);
            } else {
                info!("Successfully registered backend service");
            }
        }

        // Keep the server running until interrupted
        tokio::signal::ctrl_c()
            .await
            .map_err(|e| InfernoError::Configuration {
                message: format!("Failed to listen for shutdown signal: {}", e),
                source: None,
            })?;

        info!("Shutdown signal received, stopping backend server");

        // Clean up the metrics server task if it was started
        if let Some(task) = metrics_task {
            task.abort();
            let _ = task.await;
        }

        // Clean up the inference server task
        inference_server_task.abort();
        let _ = inference_server_task.await;

        Ok(())
    }

    /// Convert CLI options to BackendConfig
    pub fn to_config(&self) -> Result<BackendConfig> {
        let discovery_lb = self.discovery_lb.as_ref().map(|s| {
            s.split(',')
                .filter_map(|addr| addr.trim().parse::<SocketAddr>().ok())
                .collect::<Vec<_>>()
        });

        let registration_endpoint = self
            .service_discovery
            .registration_endpoint
            .as_ref()
            .and_then(|s| s.parse().ok());

        Ok(BackendConfig {
            listen_addr: self.listen_addr,
            model_path: inferno_shared::resolve_models_path(&self.model_path),
            model_type: self.model_type.clone(),
            max_batch_size: self.max_batch_size,
            gpu_device_id: self.gpu_device_id,
            max_context_length: self.max_context_length,
            memory_pool_mb: self.memory_pool_mb,
            discovery_lb: discovery_lb.unwrap_or_default(),
            enable_cache: self.enable_cache,
            cache_ttl_seconds: self.cache_ttl_seconds,
            enable_metrics: self.metrics.enable_metrics,
            operations_addr: self.metrics.get_operations_addr(6100),
            health_check_path: self.health_check.health_check_path.clone(),
            registration_endpoint,
            service_name: self.service_discovery.get_service_name("inferno-backend"),
            service_discovery_auth_mode: "open".to_string(),
            service_discovery_shared_secret: None,
        })
    }
}

/// Start the HTTP inference server
async fn start_inference_server(
    addr: SocketAddr,
    engine: Arc<
        tokio::sync::Mutex<
            Box<dyn InferenceEngine<Error = inferno_inference::inference::InferenceError>>,
        >,
    >,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await.map_err(|e| {
        InfernoError::internal(
            format!("Failed to bind inference server to {}: {}", addr, e),
            None,
        )
    })?;

    info!("🌐 HTTP inference server listening on {}", addr);

    loop {
        // This will be automatically cancelled when the task is aborted
        let (stream, _) = match listener.accept().await {
            Ok(connection) => connection,
            Err(e) => {
                // If we get an error during shutdown, it's likely because the listener was closed
                warn!("Accept error (possibly during shutdown): {}", e);
                break;
            }
        };

        let io = TokioIo::new(stream);
        let engine_clone = Arc::clone(&engine);

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req| handle_inference_request(req, Arc::clone(&engine_clone))),
                )
                .await
            {
                warn!("Error serving connection: {:?}", err);
            }
        });
    }

    Ok(())
}

/// Handle individual HTTP inference requests
async fn handle_inference_request(
    req: Request<Incoming>,
    engine: Arc<
        tokio::sync::Mutex<
            Box<dyn InferenceEngine<Error = inferno_inference::inference::InferenceError>>,
        >,
    >,
) -> std::result::Result<Response<BoxBody<bytes::Bytes, hyper::Error>>, hyper::Error> {
    let response = match (req.method(), req.uri().path()) {
        (&hyper::Method::POST, "/v1/completions") | (&hyper::Method::POST, "/inference") => {
            // Read request body
            let body_bytes = match req.into_body().collect().await {
                Ok(collected) => collected.to_bytes(),
                Err(e) => {
                    warn!("Failed to read request body: {}", e);
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(
                            Full::new("Failed to read request body".into())
                                .map_err(|never| match never {})
                                .boxed(),
                        )
                        .unwrap());
                }
            };

            // Parse JSON request
            let inference_req: InferenceRequest = match serde_json::from_slice(&body_bytes) {
                Ok(req) => req,
                Err(e) => {
                    warn!("Failed to parse inference request: {}", e);
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(
                            Full::new(format!("Invalid JSON: {}", e).into())
                                .map_err(|never| match never {})
                                .boxed(),
                        )
                        .unwrap());
                }
            };

            // Process inference
            let inference_response = {
                let engine_guard = engine.lock().await;
                match engine_guard.process(inference_req).await {
                    Ok(response) => response,
                    Err(e) => {
                        warn!("Inference failed: {}", e);
                        return Ok(Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(
                                Full::new(format!("Inference error: {}", e).into())
                                    .map_err(|never| match never {})
                                    .boxed(),
                            )
                            .unwrap());
                    }
                }
            };

            // Return JSON response
            let json_response = match serde_json::to_string(&inference_response) {
                Ok(json) => json,
                Err(e) => {
                    warn!("Failed to serialize response: {}", e);
                    return Ok(Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(
                            Full::new("Serialization error".into())
                                .map_err(|never| match never {})
                                .boxed(),
                        )
                        .unwrap());
                }
            };

            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(
                    Full::new(json_response.into())
                        .map_err(|never| match never {})
                        .boxed(),
                )
                .unwrap()
        }
        (&hyper::Method::GET, "/health") => Response::builder()
            .status(StatusCode::OK)
            .body(
                Full::new("OK".into())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap(),
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(
                Empty::<bytes::Bytes>::new()
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap(),
    };

    Ok(response)
}
