//! Configuration structures for service discovery
//!
//! This module contains configuration data structures and validation
//! logic for the service discovery system.

use super::auth::AuthMode;
use crate::error::{InfernoError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Service discovery configuration
///
/// Contains all configuration parameters for the service discovery system
/// including health checking, authentication, and timeout settings.
///
/// # Authentication
///
/// The configuration supports two authentication modes:
/// - `AuthMode::Open`: No authentication required (default)
/// - `AuthMode::SharedSecret`: Require Bearer token authentication
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::{ServiceDiscoveryConfig, AuthMode};
/// use std::time::Duration;
///
/// // Default configuration (open authentication)
/// let config = ServiceDiscoveryConfig::default();
/// assert_eq!(config.auth_mode, AuthMode::Open);
///
/// // Secure configuration with shared secret
/// let secure_config = ServiceDiscoveryConfig::with_shared_secret(
///     "my-secure-token-123".to_string()
/// );
/// assert_eq!(secure_config.auth_mode, AuthMode::SharedSecret);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDiscoveryConfig {
    /// Interval between health check cycles
    pub health_check_interval: Duration,

    /// Timeout for individual health check requests
    pub health_check_timeout: Duration,

    /// Number of consecutive failures before marking backend unhealthy
    pub failure_threshold: u32,

    /// Number of consecutive successes needed to mark backend healthy again
    pub recovery_threshold: u32,

    /// Maximum time to wait for backend registration
    pub registration_timeout: Duration,

    /// Enable detailed health check logging
    pub enable_health_check_logging: bool,

    /// Authentication mode for service discovery operations
    ///
    /// - `AuthMode::Open`: No authentication required (default)
    /// - `AuthMode::SharedSecret`: Require Bearer token authentication
    ///
    /// # Security Note
    ///
    /// Open mode provides no security and should only be used in trusted
    /// environments. For production deployments, use SharedSecret mode.
    pub auth_mode: AuthMode,

    /// Shared secret for authentication (required if auth_mode is SharedSecret)
    ///
    /// This token must be included as a Bearer token in the Authorization
    /// header for all service discovery operations when using SharedSecret mode.
    ///
    /// # Security Requirements
    ///
    /// - Use a cryptographically strong, randomly generated secret
    /// - Rotate the secret regularly in production
    /// - Protect the secret during storage and transmission
    /// - Use HTTPS to prevent token interception
    ///
    /// # Format
    ///
    /// The secret should be a string that will be used in Authorization headers
    /// as: `Authorization: Bearer <shared_secret>`
    pub shared_secret: Option<String>,
}

impl ServiceDiscoveryConfig {
    /// Creates a new configuration with shared secret authentication
    ///
    /// # Arguments
    ///
    /// * `shared_secret` - Secret token for Bearer authentication
    ///
    /// # Returns
    ///
    /// Returns a new configuration with SharedSecret auth mode and the
    /// provided secret, using default values for all other settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{ServiceDiscoveryConfig, AuthMode};
    ///
    /// let config = ServiceDiscoveryConfig::with_shared_secret(
    ///     "my-secure-token-123".to_string()
    /// );
    ///
    /// assert_eq!(config.auth_mode, AuthMode::SharedSecret);
    /// assert_eq!(config.shared_secret, Some("my-secure-token-123".to_string()));
    /// ```
    pub fn with_shared_secret(shared_secret: String) -> Self {
        Self {
            auth_mode: AuthMode::SharedSecret,
            shared_secret: Some(shared_secret),
            ..Default::default()
        }
    }

    /// Validates the configuration for consistency
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the configuration is valid, or an error
    /// describing the validation failure.
    ///
    /// # Validation Rules
    ///
    /// - If auth_mode is SharedSecret, shared_secret must be provided
    /// - Shared secret should not be empty if provided
    /// - Health check intervals and thresholds must be reasonable
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{ServiceDiscoveryConfig, AuthMode};
    ///
    /// // Valid configuration
    /// let config = ServiceDiscoveryConfig::with_shared_secret("secret123".to_string());
    /// assert!(config.validate().is_ok());
    ///
    /// // Invalid configuration (missing secret)
    /// let mut invalid_config = ServiceDiscoveryConfig::default();
    /// invalid_config.auth_mode = AuthMode::SharedSecret;
    /// assert!(invalid_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<()> {
        // Validate auth mode and secret consistency
        match self.auth_mode {
            AuthMode::SharedSecret => match &self.shared_secret {
                Some(secret) => {
                    if secret.is_empty() {
                        return Err(InfernoError::configuration(
                            "Shared secret cannot be empty when using SharedSecret auth mode",
                            None,
                        ));
                    }
                }
                None => {
                    return Err(InfernoError::configuration(
                        "Shared secret is required when using SharedSecret auth mode",
                        None,
                    ));
                }
            },
            AuthMode::Open => {
                // Open mode doesn't require a shared secret, but warn if one is provided
                if self.shared_secret.is_some() {
                    tracing::warn!(
                        "Shared secret provided but auth_mode is Open - secret will be ignored"
                    );
                }
            }
        }

        // Validate health check parameters
        if self.health_check_interval.as_secs() == 0 {
            return Err(InfernoError::configuration(
                "Health check interval must be greater than zero",
                None,
            ));
        }

        if self.health_check_timeout >= self.health_check_interval {
            return Err(InfernoError::configuration(
                "Health check timeout must be less than health check interval",
                None,
            ));
        }

        if self.failure_threshold == 0 {
            return Err(InfernoError::configuration(
                "Failure threshold must be greater than zero",
                None,
            ));
        }

        if self.recovery_threshold == 0 {
            return Err(InfernoError::configuration(
                "Recovery threshold must be greater than zero",
                None,
            ));
        }

        if self.registration_timeout.as_secs() == 0 {
            return Err(InfernoError::configuration(
                "Registration timeout must be greater than zero",
                None,
            ));
        }

        Ok(())
    }

    /// Returns the Authorization header value for the current auth mode
    ///
    /// # Returns
    ///
    /// - `None` for Open mode
    /// - `Some("Bearer <secret>")` for SharedSecret mode
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{ServiceDiscoveryConfig, AuthMode};
    ///
    /// let open_config = ServiceDiscoveryConfig::default();
    /// assert_eq!(open_config.auth_header(), None);
    ///
    /// let secure_config = ServiceDiscoveryConfig::with_shared_secret("secret123".to_string());
    /// assert_eq!(secure_config.auth_header(), Some("Bearer secret123".to_string()));
    /// ```
    pub fn auth_header(&self) -> Option<String> {
        self.auth_mode.auth_header(self.shared_secret.as_deref())
    }

    /// Validates an incoming Authorization header against this configuration
    ///
    /// # Arguments
    ///
    /// * `auth_header` - Authorization header value from request
    ///
    /// # Returns
    ///
    /// Returns `true` if authentication is valid for the current mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{ServiceDiscoveryConfig, AuthMode};
    ///
    /// let open_config = ServiceDiscoveryConfig::default();
    /// assert!(open_config.validate_auth(None));
    /// assert!(open_config.validate_auth(Some("Bearer anything")));
    ///
    /// let secure_config = ServiceDiscoveryConfig::with_shared_secret("secret123".to_string());
    /// assert!(secure_config.validate_auth(Some("Bearer secret123")));
    /// assert!(!secure_config.validate_auth(Some("Bearer wrong")));
    /// assert!(!secure_config.validate_auth(None));
    /// ```
    pub fn validate_auth(&self, auth_header: Option<&str>) -> bool {
        self.auth_mode
            .validate_auth(auth_header, self.shared_secret.as_deref())
    }
}

impl Default for ServiceDiscoveryConfig {
    /// Default configuration with reasonable production-ready settings
    ///
    /// # Default Values
    ///
    /// - Health check interval: 5 seconds
    /// - Health check timeout: 2 seconds  
    /// - Failure threshold: 3 consecutive failures
    /// - Recovery threshold: 2 consecutive successes
    /// - Registration timeout: 30 seconds
    /// - Health check logging: disabled
    /// - Authentication: Open mode (no authentication)
    /// - Shared secret: None
    ///
    /// # Security Warning
    ///
    /// The default configuration uses Open authentication mode which provides
    /// no security. For production deployments, use `with_shared_secret()` or
    /// configure SharedSecret mode manually.
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(5),
            health_check_timeout: Duration::from_secs(2),
            failure_threshold: 3,
            recovery_threshold: 2,
            registration_timeout: Duration::from_secs(30),
            enable_health_check_logging: false,
            auth_mode: AuthMode::Open,
            shared_secret: None,
        }
    }
}
