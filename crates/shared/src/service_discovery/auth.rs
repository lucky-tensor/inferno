//! Authentication and authorization for service discovery
//!
//! This module handles authentication modes and validation for the
//! service discovery system.

use serde::{Deserialize, Serialize};

/// Authentication mode for service discovery operations
///
/// This enum defines the authentication requirements for service discovery
/// operations including registration, peer discovery, and consensus operations.
///
/// # Authentication Modes
///
/// - **Open**: No authentication required. Any node can register with any other node.
///   Suitable for trusted internal networks or development environments.
///
/// - **SharedSecret**: Authentication using a shared secret token. All nodes must
///   present the same Bearer token in the Authorization header to participate
///   in service discovery operations.
///
/// # Security Considerations
///
/// - **Open mode** provides no security and should only be used in trusted environments
/// - **SharedSecret mode** provides basic authentication but tokens are transmitted
///   in headers (should use HTTPS in production)
/// - The shared secret should be cryptographically strong and rotated regularly
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::AuthMode;
///
/// // No authentication required
/// let open = AuthMode::Open;
///
/// // Shared secret authentication
/// let secure = AuthMode::SharedSecret;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthMode {
    /// Open authentication - no credentials required
    ///
    /// In this mode, any node can register with any other node without
    /// providing authentication credentials. This is suitable for:
    /// - Development and testing environments
    /// - Trusted internal networks with physical security
    /// - Scenarios where service discovery is behind other authentication layers
    ///
    /// **Security Warning**: This mode provides no authentication protection.
    Open,

    /// Shared secret authentication using Bearer tokens
    ///
    /// In this mode, all nodes must present a valid Bearer token in the
    /// Authorization header for all service discovery operations. The token
    /// must match the configured shared secret. This provides:
    /// - Basic authentication for service discovery operations
    /// - Protection against unauthorized registration attempts
    /// - Simple token-based access control
    ///
    /// **Usage**: Set Authorization header to "Bearer <shared_secret>"
    ///
    /// **Security Note**: Use HTTPS to protect tokens in transit.
    SharedSecret,
}

impl AuthMode {
    /// Returns whether this authentication mode requires credentials
    ///
    /// # Returns
    ///
    /// - `false` for Open mode (no credentials required)
    /// - `true` for SharedSecret mode (credentials required)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::AuthMode;
    ///
    /// assert!(!AuthMode::Open.requires_auth());
    /// assert!(AuthMode::SharedSecret.requires_auth());
    /// ```
    pub fn requires_auth(&self) -> bool {
        matches!(self, AuthMode::SharedSecret)
    }

    /// Returns whether this mode is secure (requires authentication)
    ///
    /// This is an alias for `requires_auth()` to make security implications clear.
    ///
    /// # Returns
    ///
    /// - `false` for Open mode (insecure)
    /// - `true` for SharedSecret mode (secure)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::AuthMode;
    ///
    /// assert!(!AuthMode::Open.is_secure());
    /// assert!(AuthMode::SharedSecret.is_secure());
    /// ```
    pub fn is_secure(&self) -> bool {
        self.requires_auth()
    }

    /// Returns the string representation used in protocol messages
    ///
    /// # Returns
    ///
    /// Returns the lowercase string representation of the auth mode
    /// as used in JSON serialization and configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::AuthMode;
    ///
    /// assert_eq!(AuthMode::Open.as_str(), "open");
    /// assert_eq!(AuthMode::SharedSecret.as_str(), "sharedsecret");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            AuthMode::Open => "open",
            AuthMode::SharedSecret => "sharedsecret",
        }
    }

    /// Parses an auth mode from a string
    ///
    /// # Arguments
    ///
    /// * `s` - String representation of the auth mode (case-insensitive)
    ///
    /// # Returns
    ///
    /// Returns `Some(AuthMode)` if the string is recognized, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::AuthMode;
    ///
    /// assert_eq!(AuthMode::parse("open"), Some(AuthMode::Open));
    /// assert_eq!(AuthMode::parse("SHAREDSECRET"), Some(AuthMode::SharedSecret));
    /// assert_eq!(AuthMode::parse("shared_secret"), Some(AuthMode::SharedSecret));
    /// assert_eq!(AuthMode::parse("invalid"), None);
    /// ```
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('_', "").as_str() {
            "open" => Some(AuthMode::Open),
            "sharedsecret" | "shared_secret" => Some(AuthMode::SharedSecret),
            _ => None,
        }
    }

    /// Returns the expected Authorization header format for this auth mode
    ///
    /// # Arguments
    ///
    /// * `secret` - Optional shared secret (required for SharedSecret mode)
    ///
    /// # Returns
    ///
    /// Returns the complete Authorization header value, or None for Open mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::AuthMode;
    ///
    /// assert_eq!(AuthMode::Open.auth_header(None), None);
    /// assert_eq!(
    ///     AuthMode::SharedSecret.auth_header(Some("mysecret")),
    ///     Some("Bearer mysecret".to_string())
    /// );
    /// ```
    pub fn auth_header(&self, secret: Option<&str>) -> Option<String> {
        match self {
            AuthMode::Open => None,
            AuthMode::SharedSecret => secret.map(|s| format!("Bearer {}", s)),
        }
    }

    /// Validates an Authorization header against this auth mode
    ///
    /// # Arguments
    ///
    /// * `auth_header` - Authorization header value from request
    /// * `expected_secret` - Expected shared secret (required for SharedSecret mode)
    ///
    /// # Returns
    ///
    /// Returns `true` if authentication is valid for this mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::AuthMode;
    ///
    /// // Open mode accepts any header (or no header)
    /// assert!(AuthMode::Open.validate_auth(None, None));
    /// assert!(AuthMode::Open.validate_auth(Some("Bearer token"), None));
    ///
    /// // SharedSecret mode requires valid Bearer token
    /// assert!(AuthMode::SharedSecret.validate_auth(
    ///     Some("Bearer secret123"),
    ///     Some("secret123")
    /// ));
    /// assert!(!AuthMode::SharedSecret.validate_auth(
    ///     Some("Bearer wrong"),
    ///     Some("secret123")
    /// ));
    /// assert!(!AuthMode::SharedSecret.validate_auth(None, Some("secret123")));
    /// ```
    pub fn validate_auth(&self, auth_header: Option<&str>, expected_secret: Option<&str>) -> bool {
        match self {
            AuthMode::Open => true, // Open mode accepts anything
            AuthMode::SharedSecret => {
                if let (Some(header), Some(expected)) = (auth_header, expected_secret) {
                    // Check for "Bearer <token>" format
                    if let Some(token) = header.strip_prefix("Bearer ") {
                        token == expected
                    } else {
                        false
                    }
                } else {
                    false // SharedSecret mode requires both header and expected secret
                }
            }
        }
    }
}

impl std::fmt::Display for AuthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for AuthMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        AuthMode::parse(s).ok_or_else(|| format!("Invalid authentication mode: {}", s))
    }
}

impl Default for AuthMode {
    /// Default authentication mode is Open for backward compatibility
    ///
    /// # Security Note
    ///
    /// The default Open mode provides no authentication. For production
    /// deployments, explicitly configure SharedSecret mode with a strong secret.
    fn default() -> Self {
        AuthMode::Open
    }
}
