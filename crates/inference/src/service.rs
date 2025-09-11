//! Service discovery integration

use crate::config::{ServiceDiscoveryConfig, VLLMConfig};
use crate::error::{ServiceRegistrationError, VLLMResult};
use crate::health::{HealthChecker, VLLMHealthChecker};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Service registration for VLLM backend
pub struct VLLMServiceRegistration {
    config: ServiceDiscoveryConfig,
    health_checker: Option<Arc<VLLMHealthChecker>>,
    registration_status: Arc<RwLock<RegistrationStatus>>,
    last_heartbeat: Arc<RwLock<Option<Instant>>>,
}

/// Service registration status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RegistrationStatus {
    /// Not yet registered
    NotRegistered,
    /// Registration in progress
    Registering,
    /// Successfully registered
    Registered,
    /// Registration failed
    Failed(String),
    /// Unregistration in progress
    Unregistering,
}

impl VLLMServiceRegistration {
    /// Create a new service registration
    #[must_use]
    pub fn new(config: VLLMConfig) -> Self {
        Self {
            config: config.service_discovery,
            health_checker: None,
            registration_status: Arc::new(RwLock::new(RegistrationStatus::NotRegistered)),
            last_heartbeat: Arc::new(RwLock::new(None)),
        }
    }

    /// Set health checker for service registration
    #[must_use]
    pub fn with_health_checker(mut self, health_checker: Arc<VLLMHealthChecker>) -> Self {
        self.health_checker = Some(health_checker);
        self
    }

    /// Register the service
    #[allow(clippy::cognitive_complexity)]
    pub async fn register(&self) -> VLLMResult<()> {
        if !self.config.enabled {
            tracing::debug!("Service discovery disabled, skipping registration");
            return Ok(());
        }

        // Check current registration status
        {
            let status = self.registration_status.read().await;
            match *status {
                RegistrationStatus::Registered => {
                    return Err(ServiceRegistrationError::AlreadyRegistered.into());
                }
                RegistrationStatus::Registering => {
                    tracing::debug!("Registration already in progress");
                    return Ok(());
                }
                _ => {}
            }
        }

        // Update status to registering
        *self.registration_status.write().await = RegistrationStatus::Registering;

        // Perform health check before registration
        if let Some(health_checker) = &self.health_checker {
            match health_checker.check_health().await {
                Ok(health_status) => {
                    tracing::info!("Pre-registration health check: {:?}", health_status);
                }
                Err(e) => {
                    let error_msg = format!("Health check failed before registration: {e}");
                    *self.registration_status.write().await =
                        RegistrationStatus::Failed(error_msg.clone());
                    return Err(ServiceRegistrationError::HealthCheckFailed(error_msg).into());
                }
            }
        }

        // TODO: Implement actual service registration with inferno-shared
        // This would typically involve:
        // 1. Creating a service registration request
        // 2. Sending it to the service discovery system
        // 3. Handling the response

        tracing::info!(
            "Registering service '{}' with capabilities: {:?}",
            self.config.service_name,
            self.config.capabilities
        );

        // Simulate registration success
        *self.registration_status.write().await = RegistrationStatus::Registered;
        *self.last_heartbeat.write().await = Some(Instant::now());

        tracing::info!("Service registration completed successfully");
        Ok(())
    }

    /// Unregister the service
    #[allow(clippy::cognitive_complexity)]
    pub async fn unregister(&self) -> VLLMResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check current registration status
        {
            let status = self.registration_status.read().await;
            match *status {
                RegistrationStatus::NotRegistered => {
                    tracing::debug!("Service not registered, nothing to unregister");
                    return Ok(());
                }
                RegistrationStatus::Unregistering => {
                    tracing::debug!("Unregistration already in progress");
                    return Ok(());
                }
                _ => {}
            }
        }

        // Update status to unregistering
        *self.registration_status.write().await = RegistrationStatus::Unregistering;

        // TODO: Implement actual service unregistration with inferno-shared
        tracing::info!("Unregistering service: {}", self.config.service_name);

        // Simulate unregistration success
        *self.registration_status.write().await = RegistrationStatus::NotRegistered;
        *self.last_heartbeat.write().await = None;

        tracing::info!("Service unregistration completed successfully");
        Ok(())
    }

    /// Send heartbeat
    #[allow(clippy::cognitive_complexity)]
    pub async fn heartbeat(&self) -> VLLMResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check if service is registered
        {
            let status = self.registration_status.read().await;
            if *status != RegistrationStatus::Registered {
                tracing::warn!("Cannot send heartbeat - service not registered");
                return Err(ServiceRegistrationError::RegistrationFailed(
                    "Service not registered".to_string(),
                )
                .into());
            }
        }

        // Perform health check as part of heartbeat
        let health_result = if let Some(health_checker) = &self.health_checker {
            match health_checker.check_health_detailed().await {
                Ok(result) => Some(result),
                Err(e) => {
                    tracing::warn!("Health check failed during heartbeat: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // TODO: Send heartbeat to service discovery with health information
        tracing::debug!(
            "Sending heartbeat for service '{}' with health: {:?}",
            self.config.service_name,
            health_result.as_ref().map(|r| &r.status)
        );

        // Update last heartbeat timestamp
        *self.last_heartbeat.write().await = Some(Instant::now());

        Ok(())
    }

    /// Get service configuration
    #[must_use]
    pub const fn config(&self) -> &ServiceDiscoveryConfig {
        &self.config
    }

    /// Get current registration status
    pub async fn get_registration_status(&self) -> RegistrationStatus {
        self.registration_status.read().await.clone()
    }

    /// Get last heartbeat timestamp
    pub async fn get_last_heartbeat(&self) -> Option<Instant> {
        *self.last_heartbeat.read().await
    }

    /// Check if service is currently registered
    pub async fn is_registered(&self) -> bool {
        matches!(
            *self.registration_status.read().await,
            RegistrationStatus::Registered
        )
    }

    /// Start periodic heartbeat task
    pub fn start_heartbeat_task(self: Arc<Self>) -> VLLMResult<tokio::task::JoinHandle<()>> {
        if !self.config.enabled {
            return Err(ServiceRegistrationError::DiscoveryUnavailable.into());
        }

        let interval_duration = Duration::from_secs(self.config.heartbeat_interval_secs);
        let service = Arc::clone(&self);

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_duration);

            loop {
                interval.tick().await;

                if let Err(e) = service.heartbeat().await {
                    tracing::error!("Heartbeat failed: {}", e);
                    // Continue trying - don't break the loop on heartbeat failure
                }
            }
        });

        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::VLLMHealthChecker;

    #[test]
    fn test_service_registration_creation() {
        let config = VLLMConfig::default();
        let registration = VLLMServiceRegistration::new(config);
        assert_eq!(registration.config().service_name, "vllm-backend");
    }

    #[tokio::test]
    async fn test_registration_status() {
        let config = VLLMConfig::default();
        let registration = VLLMServiceRegistration::new(config);

        let status = registration.get_registration_status().await;
        assert_eq!(status, RegistrationStatus::NotRegistered);

        assert!(!registration.is_registered().await);
        assert!(registration.get_last_heartbeat().await.is_none());
    }

    #[tokio::test]
    async fn test_registration_lifecycle() {
        let config = VLLMConfig::default();
        let registration = VLLMServiceRegistration::new(config);

        // Test registration
        let result = registration.register().await;
        assert!(result.is_ok());

        let status = registration.get_registration_status().await;
        assert_eq!(status, RegistrationStatus::Registered);
        assert!(registration.is_registered().await);
        assert!(registration.get_last_heartbeat().await.is_some());

        // Test duplicate registration should fail
        let result = registration.register().await;
        assert!(result.is_err());

        // Test heartbeat
        let result = registration.heartbeat().await;
        assert!(result.is_ok());

        // Test unregistration
        let result = registration.unregister().await;
        assert!(result.is_ok());

        let status = registration.get_registration_status().await;
        assert_eq!(status, RegistrationStatus::NotRegistered);
        assert!(!registration.is_registered().await);
    }

    #[tokio::test]
    async fn test_disabled_service_discovery() {
        let mut config = VLLMConfig::default();
        config.service_discovery.enabled = false;

        let registration = VLLMServiceRegistration::new(config);

        // All operations should succeed but do nothing
        assert!(registration.register().await.is_ok());
        assert!(registration.heartbeat().await.is_ok());
        assert!(registration.unregister().await.is_ok());
    }

    #[tokio::test]
    async fn test_heartbeat_without_registration() {
        let config = VLLMConfig::default();
        let registration = VLLMServiceRegistration::new(config);

        // Heartbeat should fail if not registered
        let result = registration.heartbeat().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_service_with_health_checker() {
        let config = VLLMConfig::default();
        let health_checker = Arc::new(VLLMHealthChecker::new());

        let registration = VLLMServiceRegistration::new(config).with_health_checker(health_checker);

        // Registration should include health check
        let result = registration.register().await;
        assert!(result.is_ok());

        // Heartbeat should include health check
        let result = registration.heartbeat().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_registration_status_serialization() {
        let status = RegistrationStatus::Registered;
        let json = serde_json::to_string(&status);
        assert!(json.is_ok());

        let deserialized: RegistrationStatus = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(deserialized, RegistrationStatus::Registered);

        let status = RegistrationStatus::Failed("test error".to_string());
        let json = serde_json::to_string(&status);
        assert!(json.is_ok());
    }
}
