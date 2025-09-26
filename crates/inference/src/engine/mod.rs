//! Inferno inference engine implementation

use crate::config::InfernoConfig;
use crate::error::{InfernoEngineError, InfernoError, InfernoResult};
use crate::health::InfernoHealthChecker;
use crate::memory::{CudaMemoryPool, MemoryTracker};
use crate::service::InfernoServiceRegistration;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main Inferno backend interface
pub struct InfernoBackend {
    engine: Arc<InfernoEngine>,
    config: InfernoConfig,
    memory_pool: Arc<CudaMemoryPool>,
    memory_tracker: Arc<MemoryTracker>,
    health_checker: Arc<InfernoHealthChecker>,
    service_registration: Arc<InfernoServiceRegistration>,
}

impl InfernoBackend {
    /// Create a new Inferno backend
    pub fn new(config: InfernoConfig) -> InfernoResult<Self> {
        // Initialize memory management
        let memory_pool = Arc::new(CudaMemoryPool::new(config.device_id)?);
        let memory_tracker = Arc::new(MemoryTracker::new(config.device_id));

        // Initialize health checker
        let health_checker = Arc::new(
            InfernoHealthChecker::with_config(config.health.clone())
                .with_memory_pool(Arc::clone(&memory_pool))
                .with_memory_tracker(Arc::clone(&memory_tracker)),
        );

        // Initialize service registration
        let service_registration = Arc::new(
            InfernoServiceRegistration::new(config.clone())
                .with_health_checker(Arc::clone(&health_checker)),
        );

        // Initialize engine
        let engine = Arc::new(InfernoEngine::new(&config));

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
    pub async fn start(&self) -> InfernoResult<()> {
        tracing::info!("Starting Inferno backend...");

        // Start the engine first
        self.engine.start().await?;

        // Register with service discovery
        if let Err(e) = self.service_registration.register().await {
            tracing::warn!("Service registration failed: {}", e);
            // Don't fail startup if registration fails
        }

        tracing::info!("Inferno backend started successfully");
        Ok(())
    }

    /// Stop the backend
    pub async fn stop(&self) -> InfernoResult<()> {
        tracing::info!("Stopping Inferno backend...");

        // Unregister from service discovery first
        if let Err(e) = self.service_registration.unregister().await {
            tracing::warn!("Service unregistration failed: {}", e);
        }

        // Stop the engine
        self.engine.stop().await?;

        tracing::info!("Inferno backend stopped successfully");
        Ok(())
    }

    /// Get the engine reference
    #[must_use]
    pub fn engine(&self) -> &InfernoEngine {
        &self.engine
    }

    /// Get the configuration
    #[must_use]
    pub const fn config(&self) -> &InfernoConfig {
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
    pub const fn health_checker(&self) -> &Arc<InfernoHealthChecker> {
        &self.health_checker
    }

    /// Get the service registration
    #[must_use]
    pub const fn service_registration(&self) -> &Arc<InfernoServiceRegistration> {
        &self.service_registration
    }
}

/// Core Inferno inference engine
pub struct InfernoEngine {
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

impl InfernoEngine {
    /// Create a new engine
    pub fn new(_config: &InfernoConfig) -> Self {
        Self {
            state: RwLock::new(EngineState::NotStarted),
        }
    }

    /// Start the engine
    pub async fn start(&self) -> InfernoResult<()> {
        let mut state = self.state.write().await;

        match &*state {
            EngineState::Running => {
                return Err(InfernoError::Engine(InfernoEngineError::AlreadyRunning))
            }
            EngineState::Starting => return Ok(()),
            _ => {}
        }

        *state = EngineState::Starting;

        // TODO: Initialize Inferno engine here

        *state = EngineState::Running;
        drop(state);
        Ok(())
    }

    /// Stop the engine
    pub async fn stop(&self) -> InfernoResult<()> {
        let mut state = self.state.write().await;

        match &*state {
            EngineState::NotStarted | EngineState::Stopped | EngineState::Stopping => return Ok(()),
            _ => {}
        }

        *state = EngineState::Stopping;

        // TODO: Cleanup Inferno engine here

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
        let state = self.state.read().await;
        match &*state {
            EngineState::NotStarted => "not_started".to_string(),
            EngineState::Starting => "starting".to_string(),
            EngineState::Running => "running".to_string(),
            EngineState::Stopping => "stopping".to_string(),
            EngineState::Stopped => "stopped".to_string(),
            EngineState::Error(msg) => format!("error: {msg}"),
        }
    }
}
