//! Configuration management for Inferno inference backend

use crate::error::{InfernoConfigError, InfernoResult};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use validator::{Validate, ValidationError};

/// Main configuration for the Inferno inference backend
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct InfernoConfig {
    /// Model configuration
    #[validate(length(min = 1, message = "Model path cannot be empty"))]
    pub model_path: String,

    /// Name identifier for the loaded model
    #[validate(length(min = 1, message = "Model name cannot be empty"))]
    pub model_name: String,

    /// CUDA device ID to use for inference (use 0 for CPU)
    #[validate(range(min = 0, message = "Device ID must be non-negative"))]
    pub device_id: i32,

    /// Inference parameters
    #[validate(range(
        min = 1,
        max = 1024,
        message = "Max batch size must be between 1 and 1024"
    ))]
    pub max_batch_size: usize,

    /// Maximum sequence length supported by the model
    #[validate(range(
        min = 1,
        max = 32768,
        message = "Max sequence length must be between 1 and 32768"
    ))]
    pub max_sequence_length: usize,

    /// Maximum number of tokens to generate per request
    #[validate(range(min = 1, max = 8192, message = "Max tokens must be between 1 and 8192"))]
    pub max_tokens: usize,

    /// Memory configuration
    #[validate(range(
        min = 512,
        max = 65536,
        message = "GPU memory pool size must be between 512 and 65536 MB"
    ))]
    pub gpu_memory_pool_size_mb: usize,

    /// Maximum number of sequences to process concurrently
    #[validate(range(
        min = 1,
        max = 2048,
        message = "Max num sequences must be between 1 and 2048"
    ))]
    pub max_num_seqs: usize,

    /// Performance settings
    #[validate(range(
        min = 0.1,
        max = 2.0,
        message = "Temperature must be between 0.1 and 2.0"
    ))]
    pub temperature: f32,

    /// Top-p sampling parameter for nucleus sampling
    #[validate(range(min = 0.1, max = 1.0, message = "Top-p must be between 0.1 and 1.0"))]
    pub top_p: f32,

    /// Top-k sampling parameter (0 disables top-k sampling)
    #[validate(range(min = 0, max = 1000, message = "Top-k must be 0 or between 1 and 1000"))]
    pub top_k: i32,

    /// Threading and async
    #[validate(range(min = 1, max = 64, message = "Worker threads must be between 1 and 64"))]
    pub worker_threads: usize,

    /// Enable asynchronous request processing
    pub enable_async_processing: bool,

    /// HTTP server configuration
    pub server: ServerConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Health check configuration
    pub health: HealthConfig,

    /// Service discovery configuration
    pub service_discovery: ServiceDiscoveryConfig,
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServerConfig {
    /// Server host address to bind to
    #[validate(length(min = 1, message = "Host cannot be empty"))]
    pub host: String,

    /// Server port number to bind to
    #[validate(range(
        min = 1024,
        max = 65535,
        message = "Port must be between 1024 and 65535"
    ))]
    pub port: u16,

    /// Request timeout in seconds
    #[validate(range(
        min = 1,
        max = 3600,
        message = "Request timeout must be between 1 and 3600 seconds"
    ))]
    pub request_timeout_secs: u64,

    /// Maximum number of concurrent requests to handle
    #[validate(range(
        min = 1,
        max = 10000,
        message = "Max concurrent requests must be between 1 and 10000"
    ))]
    pub max_concurrent_requests: usize,

    /// Enable CORS headers for cross-origin requests
    pub enable_cors: bool,
    /// Enable metrics collection and endpoint
    pub enable_metrics: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    #[validate(custom(function = "validate_log_level"))]
    pub level: String,

    /// Log format (json, text, compact)
    #[validate(custom(function = "validate_log_format"))]
    pub format: String,

    /// Enable logging to file
    pub enable_file_logging: bool,
    /// Path to log file (if file logging is enabled)
    pub log_file_path: Option<String>,

    /// Maximum log file size in megabytes
    #[validate(range(
        min = 1,
        max = 1000,
        message = "Max log file size must be between 1 and 1000 MB"
    ))]
    pub max_log_file_size_mb: usize,

    /// Number of days to retain log files
    #[validate(range(
        min = 1,
        max = 100,
        message = "Log file retention must be between 1 and 100 days"
    ))]
    pub log_file_retention_days: u32,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HealthConfig {
    /// Enable health checking
    pub enabled: bool,

    /// Interval between health checks in seconds
    #[validate(range(
        min = 1,
        max = 300,
        message = "Check interval must be between 1 and 300 seconds"
    ))]
    pub check_interval_secs: u64,

    /// Health check timeout in seconds
    #[validate(range(
        min = 1,
        max = 60,
        message = "Timeout must be between 1 and 60 seconds"
    ))]
    pub timeout_secs: u64,

    /// GPU memory threshold for health checks (0.00.0)
    pub gpu_memory_threshold: f64,
    /// Inference latency threshold in milliseconds
    pub inference_latency_threshold_ms: f64,
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServiceDiscoveryConfig {
    /// Enable service discovery registration
    pub enabled: bool,

    /// Service name to register with discovery service
    #[validate(length(min = 1, message = "Service name cannot be empty"))]
    pub service_name: String,

    /// TTL for service registration in seconds
    pub registration_ttl_secs: u64,
    /// Interval between heartbeat updates in seconds
    pub heartbeat_interval_secs: u64,

    /// Service capabilities advertised to discovery
    pub capabilities: Vec<String>,
    /// Additional metadata for service registration
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for InfernoConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            model_name: "default".to_string(),
            device_id: 0,
            max_batch_size: crate::DEFAULT_BATCH_SIZE,
            max_sequence_length: 4096,
            max_tokens: 100,
            gpu_memory_pool_size_mb: crate::DEFAULT_MEMORY_POOL_SIZE_MB,
            max_num_seqs: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            worker_threads: num_cpus::get().min(8),
            enable_async_processing: true,
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
            health: HealthConfig::default(),
            service_discovery: ServiceDiscoveryConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: crate::DEFAULT_PORT,
            request_timeout_secs: 300,
            max_concurrent_requests: 1000,
            enable_cors: true,
            enable_metrics: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            enable_file_logging: false,
            log_file_path: None,
            max_log_file_size_mb: 100,
            log_file_retention_days: 7,
        }
    }
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_secs: 30,
            timeout_secs: 5,
            gpu_memory_threshold: 0.9,
            inference_latency_threshold_ms: 1000.0,
        }
    }
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            service_name: "inferno-backend".to_string(),
            registration_ttl_secs: 60,
            heartbeat_interval_secs: 30,
            capabilities: vec!["text-generation".to_string()],
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Configuration builder for fluent configuration construction
#[derive(Debug)]
pub struct InfernoConfigBuilder {
    config: InfernoConfig,
}

impl InfernoConfigBuilder {
    /// Create a new configuration builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: InfernoConfig::default(),
        }
    }

    /// Set the model path
    #[must_use]
    pub fn model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_string_lossy().to_string();
        self
    }

    /// Set the model name
    #[must_use]
    pub fn model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the device ID
    #[must_use]
    pub const fn device_id(mut self, device_id: i32) -> Self {
        self.config.device_id = device_id;
        self
    }

    /// Set the maximum batch size
    #[must_use]
    pub const fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Set the maximum sequence length
    #[must_use]
    pub const fn max_sequence_length(mut self, length: usize) -> Self {
        self.config.max_sequence_length = length;
        self
    }

    /// Set the GPU memory pool size
    #[must_use]
    pub const fn gpu_memory_pool_size_mb(mut self, size_mb: usize) -> Self {
        self.config.gpu_memory_pool_size_mb = size_mb;
        self
    }

    /// Set the server port
    #[must_use]
    pub const fn port(mut self, port: u16) -> Self {
        self.config.server.port = port;
        self
    }

    /// Set the server host
    #[must_use]
    pub fn host<S: Into<String>>(mut self, host: S) -> Self {
        self.config.server.host = host.into();
        self
    }

    /// Enable or disable service discovery
    #[must_use]
    pub const fn service_discovery(mut self, enabled: bool) -> Self {
        self.config.service_discovery.enabled = enabled;
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> InfernoResult<InfernoConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for InfernoConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InfernoConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> InfernoResult<Self> {
        let mut config = Self::default();

        // Load from environment variables with INFERNO_ prefix
        if let Ok(model_path) = env::var("INFERNO_MODEL_PATH") {
            config.model_path = model_path;
        }

        if let Ok(model_name) = env::var("INFERNO_MODEL_NAME") {
            config.model_name = model_name;
        }

        if let Ok(device_id) = env::var("INFERNO_DEVICE_ID") {
            config.device_id = device_id
                .parse()
                .map_err(|_| InfernoConfigError::InvalidValue {
                    field: "device_id".to_string(),
                    value: device_id,
                    reason: "must be a valid integer".to_string(),
                })?;
        }

        if let Ok(batch_size) = env::var("INFERNO_MAX_BATCH_SIZE") {
            config.max_batch_size =
                batch_size
                    .parse()
                    .map_err(|_| InfernoConfigError::InvalidValue {
                        field: "max_batch_size".to_string(),
                        value: batch_size,
                        reason: "must be a valid positive integer".to_string(),
                    })?;
        }

        if let Ok(memory_size) = env::var("INFERNO_GPU_MEMORY_POOL_SIZE_MB") {
            config.gpu_memory_pool_size_mb =
                memory_size
                    .parse()
                    .map_err(|_| InfernoConfigError::InvalidValue {
                        field: "gpu_memory_pool_size_mb".to_string(),
                        value: memory_size,
                        reason: "must be a valid positive integer".to_string(),
                    })?;
        }

        if let Ok(port) = env::var("INFERNO_PORT") {
            config.server.port = port.parse().map_err(|_| InfernoConfigError::InvalidValue {
                field: "port".to_string(),
                value: port,
                reason: "must be a valid port number".to_string(),
            })?;
        }

        if let Ok(host) = env::var("INFERNO_HOST") {
            config.server.host = host;
        }

        if let Ok(log_level) = env::var("INFERNO_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        if let Ok(service_name) = env::var("INFERNO_SERVICE_NAME") {
            config.service_discovery.service_name = service_name;
        }

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> InfernoResult<Self> {
        let content = fs::read_to_string(path.as_ref()).map_err(|e| {
            InfernoConfigError::FileRead(format!("Failed to read config file: {e}"))
        })?;

        let config: Self = toml::from_str(&content)
            .map_err(|e| InfernoConfigError::Parse(format!("Failed to parse TOML: {e}")))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> InfernoResult<Self> {
        let content = fs::read_to_string(path.as_ref()).map_err(|e| {
            InfernoConfigError::FileRead(format!("Failed to read config file: {e}"))
        })?;

        let config: Self = serde_json::from_str(&content)
            .map_err(|e| InfernoConfigError::Parse(format!("Failed to parse JSON: {e}")))?;

        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> InfernoResult<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| InfernoConfigError::Parse(format!("Failed to serialize TOML: {e}")))?;

        fs::write(path.as_ref(), content).map_err(|e| {
            InfernoConfigError::FileRead(format!("Failed to write config file: {e}"))
        })?;

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> InfernoResult<()> {
        // Use validator crate for basic validation
        Validate::validate(self)?;

        // Custom validation logic
        if self.model_path.is_empty() {
            return Err(InfernoConfigError::MissingField("model_path".to_string()).into());
        }

        // Note: Model path existence is checked at runtime during model loading

        // Validate memory configuration
        if self.gpu_memory_pool_size_mb * self.max_batch_size > 65536 {
            return Err(InfernoConfigError::ValidationFailed(
                "Total memory requirement exceeds system limits".to_string(),
            )
            .into());
        }

        // Validate worker thread configuration
        if self.worker_threads > num_cpus::get() * 2 {
            tracing::warn!(
                "Worker threads ({}) exceeds 2x CPU count ({}), this may cause performance issues",
                self.worker_threads,
                num_cpus::get()
            );
        }

        Ok(())
    }

    /// Get the full server address
    #[must_use]
    pub fn server_address(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }

    /// Check if CUDA is required
    #[must_use]
    pub const fn requires_cuda(&self) -> bool {
        self.device_id >= 0
    }
}

// Custom validators
fn validate_log_level(level: &str) -> Result<(), ValidationError> {
    match level.to_lowercase().as_str() {
        "trace" | "debug" | "info" | "warn" | "error" => Ok(()),
        _ => Err(ValidationError::new("Invalid log level")),
    }
}

fn validate_log_format(format: &str) -> Result<(), ValidationError> {
    match format.to_lowercase().as_str() {
        "json" | "text" | "compact" => Ok(()),
        _ => Err(ValidationError::new("Invalid log format")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = InfernoConfig::default();
        // Default config should not pass validation without model_path
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = InfernoConfigBuilder::new()
            .model_path("/tmp/test_model")
            .model_name("test-model")
            .device_id(0)
            .max_batch_size(4)
            .port(8080)
            .build();

        assert!(config.is_ok()); // Should succeed as path validation is now at runtime
    }

    #[test]
    fn test_config_validation() {
        let config = InfernoConfig {
            model_path: "test-model".to_string(),
            gpu_memory_pool_size_mb: 32768, // 32GB
            max_batch_size: 4,              // This will exceed 65536 limit (32768 * 4 = 131072)
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err()); // Should fail due to memory limit exceeded
    }

    #[test]
    fn test_server_address() {
        let config = InfernoConfig::default();
        assert_eq!(config.server_address(), "0.0.0.0:8000");
    }

    #[test]
    fn test_cuda_requirement() {
        let config = InfernoConfig::default();
        assert!(config.requires_cuda()); // device_id = 0 by default

        let mut config = config;
        config.device_id = 0;
        assert!(!config.requires_cuda());
    }

    #[test]
    fn test_config_serialization() {
        let config = InfernoConfig::default();

        // Test TOML serialization
        let toml_str = toml::to_string(&config).unwrap();
        let _deserialized: InfernoConfig = toml::from_str(&toml_str).unwrap();

        // Test JSON serialization
        let json_str = serde_json::to_string(&config).unwrap();
        let _deserialized: InfernoConfig = serde_json::from_str(&json_str).unwrap();
    }
}
