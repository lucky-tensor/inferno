//! VLLM inference engine implementation

use crate::config::VLLMConfig;
use crate::error::{VLLMEngineError, VLLMError, VLLMResult};
use crate::health::VLLMHealthChecker;
use crate::memory::{CudaMemoryPool, MemoryTracker};
use crate::service::VLLMServiceRegistration;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main VLLM backend interface
pub struct VLLMBackend {
    engine: Arc<VLLMEngine>,
    config: VLLMConfig,
    memory_pool: Arc<CudaMemoryPool>,
    memory_tracker: Arc<MemoryTracker>,
    health_checker: Arc<VLLMHealthChecker>,
    service_registration: Arc<VLLMServiceRegistration>,
}

impl VLLMBackend {
    /// Create a new VLLM backend
    pub fn new(config: VLLMConfig) -> VLLMResult<Self> {
        // Initialize memory management
        let memory_pool = Arc::new(CudaMemoryPool::new(config.device_id)?);
        let memory_tracker = Arc::new(MemoryTracker::new(config.device_id));

        // Initialize health checker
        let health_checker = Arc::new(
            VLLMHealthChecker::with_config(config.health.clone())
                .with_memory_pool(Arc::clone(&memory_pool))
                .with_memory_tracker(Arc::clone(&memory_tracker)),
        );

        // Initialize service registration
        let service_registration = Arc::new(
            VLLMServiceRegistration::new(config.clone())
                .with_health_checker(Arc::clone(&health_checker)),
        );

        // Initialize engine
        let engine = Arc::new(VLLMEngine::new(&config)?);

        Ok(Self {
            engine,
            config,
            memory_pool,
            memory_tracker,
            health_checker,
            service_registration,
        })
    }

    /// Start the backend
    pub async fn start(&self) -> VLLMResult<()> {
        tracing::info!("Starting VLLM backend...");

        // Start the engine first
        self.engine.start().await?;

        // Register with service discovery
        if let Err(e) = self.service_registration.register().await {
            tracing::warn!("Service registration failed: {}", e);
            // Don't fail startup if registration fails
        }

        tracing::info!("VLLM backend started successfully");
        Ok(())
    }

    /// Stop the backend
    pub async fn stop(&self) -> VLLMResult<()> {
        tracing::info!("Stopping VLLM backend...");

        // Unregister from service discovery first
        if let Err(e) = self.service_registration.unregister().await {
            tracing::warn!("Service unregistration failed: {}", e);
        }

        // Stop the engine
        self.engine.stop().await?;

        tracing::info!("VLLM backend stopped successfully");
        Ok(())
    }

    /// Get the engine reference
    #[must_use]
    pub fn engine(&self) -> &VLLMEngine {
        &self.engine
    }

    /// Get the configuration
    #[must_use]
    pub const fn config(&self) -> &VLLMConfig {
        &self.config
    }

    /// Get the memory pool
    #[must_use]
    pub const fn memory_pool(&self) -> &Arc<CudaMemoryPool> {
        &self.memory_pool
    }

    /// Get the memory tracker
    #[must_use]
    pub const fn memory_tracker(&self) -> &Arc<MemoryTracker> {
        &self.memory_tracker
    }

    /// Get the health checker
    #[must_use]
    pub const fn health_checker(&self) -> &Arc<VLLMHealthChecker> {
        &self.health_checker
    }

    /// Get the service registration
    #[must_use]
    pub const fn service_registration(&self) -> &Arc<VLLMServiceRegistration> {
        &self.service_registration
    }
}

/// Core VLLM inference engine
pub struct VLLMEngine {
    state: RwLock<EngineState>,
}

#[derive(Debug, Clone)]
enum EngineState {
    NotStarted,
    Starting,
    Running,
    Stopping,
    Stopped,
    #[allow(dead_code)]
    Error(String),
}

impl VLLMEngine {
    /// Create a new engine
    pub fn new(_config: &VLLMConfig) -> VLLMResult<Self> {
        Ok(Self {
            state: RwLock::new(EngineState::NotStarted),
        })
    }

    /// Start the engine
    pub async fn start(&self) -> VLLMResult<()> {
        let mut state = self.state.write().await;

        match &*state {
            EngineState::Running => return Err(VLLMError::Engine(VLLMEngineError::AlreadyRunning)),
            EngineState::Starting => return Ok(()),
            _ => {}
        }

        *state = EngineState::Starting;

        // TODO: Initialize VLLM engine here

        *state = EngineState::Running;
        drop(state);
        Ok(())
    }

    /// Stop the engine
    pub async fn stop(&self) -> VLLMResult<()> {
        let mut state = self.state.write().await;

        match &*state {
            EngineState::NotStarted | EngineState::Stopped | EngineState::Stopping => return Ok(()),
            _ => {}
        }

        *state = EngineState::Stopping;

        // TODO: Cleanup VLLM engine here

        *state = EngineState::Stopped;
        drop(state);
        Ok(())
    }

    /// Check if engine is running
    pub async fn is_running(&self) -> bool {
        matches!(*self.state.read().await, EngineState::Running)
    }

    /// Get the current engine state
    pub async fn get_state(&self) -> String {
        match &*self.state.read().await {
            EngineState::NotStarted => "not_started".to_string(),
            EngineState::Starting => "starting".to_string(),
            EngineState::Running => "running".to_string(),
            EngineState::Stopping => "stopping".to_string(),
            EngineState::Stopped => "stopped".to_string(),
            EngineState::Error(msg) => format!("error: {msg}"),
        }
    }
}
