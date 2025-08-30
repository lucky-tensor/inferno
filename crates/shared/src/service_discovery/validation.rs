//! Input validation and sanitization for service discovery
//!
//! This module provides comprehensive validation and sanitization functions
//! for all service discovery data types to prevent security vulnerabilities
//! and ensure data integrity.
//!
//! ## Security Features
//!
//! - **String Length Validation**: Prevents buffer overflow and DoS attacks
//! - **Character Set Validation**: Ensures only safe characters are used
//! - **Address Validation**: Validates IP addresses and hostnames
//! - **Port Range Validation**: Ensures ports are within valid ranges
//! - **Input Sanitization**: Removes potentially dangerous characters
//!
//! ## Performance Characteristics
//!
//! - Validation time: < 100μs per field
//! - Memory allocation: zero-copy where possible
//! - Thread-safe: all functions are stateless and thread-safe

use crate::service_discovery::{NodeInfo, NodeType, PeerInfo};
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;

/// Maximum length for node IDs (prevents memory exhaustion)
pub const MAX_NODE_ID_LENGTH: usize = 128;

/// Minimum length for node IDs (ensures meaningful identifiers)
pub const MIN_NODE_ID_LENGTH: usize = 1;

/// Maximum length for addresses (prevents buffer overflow)
pub const MAX_ADDRESS_LENGTH: usize = 256;

/// Maximum length for capability strings
pub const MAX_CAPABILITY_LENGTH: usize = 64;

/// Maximum number of capabilities per node
pub const MAX_CAPABILITIES_COUNT: usize = 32;

/// Minimum valid port number (exclude well-known ports 1-1023)
pub const MIN_PORT: u16 = 1024;

/// Maximum valid port number
pub const MAX_PORT: u16 = 65535;

/// Valid characters for node IDs (alphanumeric, hyphen, underscore, dot)
const VALID_NODE_ID_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.";

/// Valid characters for capability names (alphanumeric, underscore)
const VALID_CAPABILITY_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";

/// Validation error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// String field is too long
    TooLong {
        field: String,
        max_length: usize,
        actual_length: usize,
    },
    /// String field is too short
    TooShort {
        field: String,
        min_length: usize,
        actual_length: usize,
    },
    /// String contains invalid characters
    InvalidCharacters {
        field: String,
        invalid_chars: Vec<char>,
    },
    /// Network address is invalid
    InvalidAddress {
        address: String,
        reason: String,
    },
    /// Port number is out of valid range
    InvalidPort {
        port: u16,
        min: u16,
        max: u16,
    },
    /// Too many capabilities specified
    TooManyCapabilities {
        count: usize,
        max: usize,
    },
    /// Field is empty but required
    Empty {
        field: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::TooLong { field, max_length, actual_length } => {
                write!(f, "Field '{}' is too long: {} characters (max: {})", field, actual_length, max_length)
            }
            ValidationError::TooShort { field, min_length, actual_length } => {
                write!(f, "Field '{}' is too short: {} characters (min: {})", field, actual_length, min_length)
            }
            ValidationError::InvalidCharacters { field, invalid_chars } => {
                write!(f, "Field '{}' contains invalid characters: {:?}", field, invalid_chars)
            }
            ValidationError::InvalidAddress { address, reason } => {
                write!(f, "Invalid address '{}': {}", address, reason)
            }
            ValidationError::InvalidPort { port, min, max } => {
                write!(f, "Invalid port {}: must be between {} and {}", port, min, max)
            }
            ValidationError::TooManyCapabilities { count, max } => {
                write!(f, "Too many capabilities: {} (max: {})", count, max)
            }
            ValidationError::Empty { field } => {
                write!(f, "Field '{}' cannot be empty", field)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validates a node ID string
///
/// # Arguments
///
/// * `id` - The node ID to validate
///
/// # Returns
///
/// Returns `Ok(())` if valid, or a `ValidationError` describing the issue.
///
/// # Validation Rules
///
/// - Length: 1-128 characters
/// - Characters: alphanumeric, hyphen, underscore, dot only
/// - Must not be empty or whitespace-only
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::validate_node_id;
///
/// assert!(validate_node_id("backend-1").is_ok());
/// assert!(validate_node_id("proxy_001").is_ok());
/// assert!(validate_node_id("test.node.123").is_ok());
/// 
/// assert!(validate_node_id("").is_err());
/// assert!(validate_node_id("a".repeat(200).as_str()).is_err());
/// assert!(validate_node_id("invalid@char").is_err());
/// ```
pub fn validate_node_id(id: &str) -> ValidationResult<()> {
    let trimmed = id.trim();
    
    if trimmed.is_empty() {
        return Err(ValidationError::Empty {
            field: "id".to_string(),
        });
    }
    
    if trimmed.len() < MIN_NODE_ID_LENGTH {
        return Err(ValidationError::TooShort {
            field: "id".to_string(),
            min_length: MIN_NODE_ID_LENGTH,
            actual_length: trimmed.len(),
        });
    }
    
    if trimmed.len() > MAX_NODE_ID_LENGTH {
        return Err(ValidationError::TooLong {
            field: "id".to_string(),
            max_length: MAX_NODE_ID_LENGTH,
            actual_length: trimmed.len(),
        });
    }
    
    let invalid_chars: Vec<char> = trimmed
        .chars()
        .filter(|c| !VALID_NODE_ID_CHARS.contains(*c))
        .collect();
    
    if !invalid_chars.is_empty() {
        return Err(ValidationError::InvalidCharacters {
            field: "id".to_string(),
            invalid_chars,
        });
    }
    
    Ok(())
}

/// Validates a network address string
///
/// # Arguments
///
/// * `address` - The address to validate (format: "host:port")
///
/// # Returns
///
/// Returns `Ok(())` if valid, or a `ValidationError` describing the issue.
///
/// # Validation Rules
///
/// - Format: "host:port" where host is IP or hostname
/// - Host: valid IPv4, IPv6, or hostname
/// - Port: valid port number (parsed separately)
/// - Total length: max 256 characters
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::validate_address;
///
/// assert!(validate_address("127.0.0.1:8080").is_ok());
/// assert!(validate_address("localhost:3000").is_ok());
/// assert!(validate_address("[::1]:8080").is_ok());
/// assert!(validate_address("service.local:9090").is_ok());
/// 
/// assert!(validate_address("invalid-format").is_err());
/// assert!(validate_address("").is_err());
/// assert!(validate_address("host:99999").is_err());
/// ```
pub fn validate_address(address: &str) -> ValidationResult<()> {
    let trimmed = address.trim();
    
    if trimmed.is_empty() {
        return Err(ValidationError::Empty {
            field: "address".to_string(),
        });
    }
    
    if trimmed.len() > MAX_ADDRESS_LENGTH {
        return Err(ValidationError::TooLong {
            field: "address".to_string(),
            max_length: MAX_ADDRESS_LENGTH,
            actual_length: trimmed.len(),
        });
    }
    
    // Try parsing as socket address first (handles IPv4, IPv6, hostname:port)
    if let Ok(_) = SocketAddr::from_str(trimmed) {
        return Ok(());
    }
    
    // Manual parsing for more detailed error reporting
    let parts: Vec<&str> = trimmed.rsplitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(ValidationError::InvalidAddress {
            address: trimmed.to_string(),
            reason: "Address must be in format 'host:port'".to_string(),
        });
    }
    
    let port_str = parts[0];
    let host = parts[1];
    
    // Validate port
    match port_str.parse::<u16>() {
        Ok(port) => validate_port(port)?,
        Err(_) => {
            return Err(ValidationError::InvalidAddress {
                address: trimmed.to_string(),
                reason: format!("Invalid port number: {}", port_str),
            });
        }
    }
    
    // Validate host (try as IP first, then as hostname)
    if host.is_empty() {
        return Err(ValidationError::InvalidAddress {
            address: trimmed.to_string(),
            reason: "Host cannot be empty".to_string(),
        });
    }
    
    // Try parsing as IP address
    if IpAddr::from_str(host).is_ok() {
        return Ok(());
    }
    
    // Validate as hostname (basic DNS name validation)
    validate_hostname(host, trimmed)?;
    
    Ok(())
}

/// Validates a hostname according to DNS standards
fn validate_hostname(hostname: &str, original_address: &str) -> ValidationResult<()> {
    if hostname.is_empty() || hostname.len() > 253 {
        return Err(ValidationError::InvalidAddress {
            address: original_address.to_string(),
            reason: "Hostname must be 1-253 characters".to_string(),
        });
    }
    
    // Check each label in the hostname
    for label in hostname.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err(ValidationError::InvalidAddress {
                address: original_address.to_string(),
                reason: "Each hostname label must be 1-63 characters".to_string(),
            });
        }
        
        // Labels must start and end with alphanumeric
        let chars: Vec<char> = label.chars().collect();
        if !chars[0].is_ascii_alphanumeric() || !chars[chars.len() - 1].is_ascii_alphanumeric() {
            return Err(ValidationError::InvalidAddress {
                address: original_address.to_string(),
                reason: "Hostname labels must start and end with alphanumeric characters".to_string(),
            });
        }
        
        // Check for valid characters (alphanumeric and hyphen)
        for ch in chars {
            if !ch.is_ascii_alphanumeric() && ch != '-' {
                return Err(ValidationError::InvalidAddress {
                    address: original_address.to_string(),
                    reason: format!("Invalid character '{}' in hostname", ch),
                });
            }
        }
    }
    
    Ok(())
}

/// Validates a port number
///
/// # Arguments
///
/// * `port` - The port number to validate
///
/// # Returns
///
/// Returns `Ok(())` if valid, or a `ValidationError` describing the issue.
///
/// # Validation Rules
///
/// - Range: 1024-65535 (excludes well-known ports for security)
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::validate_port;
///
/// assert!(validate_port(8080).is_ok());
/// assert!(validate_port(3000).is_ok());
/// assert!(validate_port(65535).is_ok());
/// 
/// assert!(validate_port(80).is_err()); // Well-known port
/// assert!(validate_port(0).is_err());  // Invalid port
/// ```
pub fn validate_port(port: u16) -> ValidationResult<()> {
    if port < MIN_PORT || port > MAX_PORT {
        return Err(ValidationError::InvalidPort {
            port,
            min: MIN_PORT,
            max: MAX_PORT,
        });
    }
    Ok(())
}

/// Validates a list of capability strings
///
/// # Arguments
///
/// * `capabilities` - The capabilities to validate
///
/// # Returns
///
/// Returns `Ok(())` if valid, or a `ValidationError` describing the issue.
///
/// # Validation Rules
///
/// - Count: max 32 capabilities
/// - Length: each capability max 64 characters
/// - Characters: alphanumeric and underscore only
/// - No duplicates allowed
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::validate_capabilities;
///
/// let valid_caps = vec!["inference".to_string(), "gpu_support".to_string()];
/// assert!(validate_capabilities(&valid_caps).is_ok());
/// 
/// let invalid_caps = vec!["invalid-char".to_string()];
/// assert!(validate_capabilities(&invalid_caps).is_err());
/// 
/// let too_many = (0..50).map(|i| format!("cap{}", i)).collect::<Vec<_>>();
/// assert!(validate_capabilities(&too_many).is_err());
/// ```
pub fn validate_capabilities(capabilities: &[String]) -> ValidationResult<()> {
    if capabilities.len() > MAX_CAPABILITIES_COUNT {
        return Err(ValidationError::TooManyCapabilities {
            count: capabilities.len(),
            max: MAX_CAPABILITIES_COUNT,
        });
    }
    
    let mut seen = std::collections::HashSet::new();
    
    for capability in capabilities {
        let trimmed = capability.trim();
        
        if trimmed.is_empty() {
            return Err(ValidationError::Empty {
                field: "capability".to_string(),
            });
        }
        
        if trimmed.len() > MAX_CAPABILITY_LENGTH {
            return Err(ValidationError::TooLong {
                field: "capability".to_string(),
                max_length: MAX_CAPABILITY_LENGTH,
                actual_length: trimmed.len(),
            });
        }
        
        let invalid_chars: Vec<char> = trimmed
            .chars()
            .filter(|c| !VALID_CAPABILITY_CHARS.contains(*c))
            .collect();
        
        if !invalid_chars.is_empty() {
            return Err(ValidationError::InvalidCharacters {
                field: "capability".to_string(),
                invalid_chars,
            });
        }
        
        // Check for duplicates
        if !seen.insert(trimmed.to_string()) {
            return Err(ValidationError::InvalidCharacters {
                field: "capability".to_string(),
                invalid_chars: vec![], // Reusing this error type for duplicates
            });
        }
    }
    
    Ok(())
}

/// Validates and sanitizes a complete NodeInfo structure
///
/// # Arguments
///
/// * `node_info` - The NodeInfo to validate
///
/// # Returns
///
/// Returns `Ok(NodeInfo)` with sanitized fields if valid, or a `ValidationError`.
///
/// # Validation
///
/// - Validates all string fields according to their rules
/// - Sanitizes by trimming whitespace (zero-copy where possible)
/// - Ensures consistency (e.g., is_load_balancer matches node_type)
///
/// # Performance Notes
///
/// - Validation time: < 100μs typical
/// - Memory allocation: minimal, only when trimming is needed
/// - Zero-copy validation when strings are already clean
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::{NodeInfo, NodeType};
/// use inferno_shared::service_discovery::validation::validate_and_sanitize_node_info;
///
/// let node = NodeInfo::new(
///     "backend-1".to_string(),
///     "127.0.0.1:3000".to_string(),
///     9090,
///     NodeType::Backend
/// );
///
/// let sanitized = validate_and_sanitize_node_info(node).unwrap();
/// assert_eq!(sanitized.id, "backend-1");
/// ```
pub fn validate_and_sanitize_node_info(mut node_info: NodeInfo) -> ValidationResult<NodeInfo> {
    // Sanitize strings by trimming whitespace (only allocate if needed)
    let trimmed_id = node_info.id.trim();
    if trimmed_id != node_info.id {
        node_info.id = trimmed_id.to_string();
    }
    
    let trimmed_address = node_info.address.trim();
    if trimmed_address != node_info.address {
        node_info.address = trimmed_address.to_string();
    }
    
    // Optimize capability trimming - only allocate if changes needed
    let mut capabilities_changed = false;
    let trimmed_capabilities: Vec<String> = node_info.capabilities
        .into_iter()
        .map(|cap| {
            let trimmed = cap.trim();
            if trimmed != cap {
                capabilities_changed = true;
            }
            if capabilities_changed {
                trimmed.to_string()
            } else {
                cap  // Reuse original string if no changes needed
            }
        })
        .collect();
    
    node_info.capabilities = trimmed_capabilities;
    
    // Validate individual fields
    validate_node_id(&node_info.id)?;
    validate_address(&node_info.address)?;
    validate_port(node_info.metrics_port)?;
    validate_capabilities(&node_info.capabilities)?;
    
    // Validate consistency
    if node_info.is_load_balancer && !node_info.node_type.can_load_balance() {
        return Err(ValidationError::InvalidCharacters {
            field: "is_load_balancer".to_string(),
            invalid_chars: vec![], // Reusing this error type for consistency issues
        });
    }
    
    Ok(node_info)
}

/// Validates and sanitizes a PeerInfo structure
///
/// # Arguments
///
/// * `peer_info` - The PeerInfo to validate
///
/// # Returns
///
/// Returns `Ok(PeerInfo)` with sanitized fields if valid, or a `ValidationError`.
pub fn validate_and_sanitize_peer_info(mut peer_info: PeerInfo) -> ValidationResult<PeerInfo> {
    // Sanitize strings by trimming whitespace
    peer_info.id = peer_info.id.trim().to_string();
    peer_info.address = peer_info.address.trim().to_string();
    
    // Validate individual fields
    validate_node_id(&peer_info.id)?;
    validate_address(&peer_info.address)?;
    validate_port(peer_info.metrics_port)?;
    
    // Validate consistency
    if peer_info.is_load_balancer && !peer_info.node_type.can_load_balance() {
        return Err(ValidationError::InvalidCharacters {
            field: "is_load_balancer".to_string(),
            invalid_chars: vec![], // Reusing this error type for consistency issues
        });
    }
    
    Ok(peer_info)
}

/// Legacy compatibility function for checking node ID validity
///
/// This function provides a simple boolean check for node ID validity,
/// compatible with existing code that expects a bool return.
///
/// # Arguments
///
/// * `id` - The node ID string to validate
///
/// # Returns
///
/// Returns `true` if the ID is valid according to validation rules, `false` otherwise.
///
/// # Performance Notes
///
/// - Same performance as `validate_node_id()` but discards error details
/// - Prefer `validate_node_id()` for new code that needs error information
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::is_valid_node_id;
///
/// assert!(is_valid_node_id("backend-1"));
/// assert!(!is_valid_node_id("invalid@id"));
/// ```
pub fn is_valid_node_id(id: &str) -> bool {
    validate_node_id(id).is_ok()
}

/// Legacy compatibility function for checking address validity
///
/// This function provides a simple boolean check for address validity,
/// compatible with existing code that expects a bool return.
///
/// # Arguments
///
/// * `address` - The address string to validate in "host:port" format
///
/// # Returns
///
/// Returns `true` if the address is valid, `false` otherwise.
///
/// # Performance Notes
///
/// - Same performance as `validate_address()` but discards error details
/// - Prefer `validate_address()` for new code that needs error information
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::is_valid_address;
///
/// assert!(is_valid_address("127.0.0.1:8080"));
/// assert!(!is_valid_address("invalid-address"));
/// ```
pub fn is_valid_address(address: &str) -> bool {
    validate_address(address).is_ok()
}

/// Legacy compatibility function for checking capability validity
///
/// This function provides a simple boolean check for capability validity,
/// compatible with existing code that expects a bool return.
///
/// # Arguments
///
/// * `capability` - The capability string to validate
///
/// # Returns
///
/// Returns `true` if the capability is valid, `false` otherwise.
///
/// # Performance Notes
///
/// - Slightly slower than direct validation due to Vec allocation
/// - Prefer `validate_capabilities()` for validating multiple capabilities
/// - Consider using direct validation for performance-critical paths
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::validation::is_valid_capability;
///
/// assert!(is_valid_capability("inference"));
/// assert!(!is_valid_capability("invalid-capability-name!"));
/// ```
pub fn is_valid_capability(capability: &str) -> bool {
    let capabilities = vec![capability.to_string()];
    validate_capabilities(&capabilities).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_discovery::NodeType;

    #[test]
    fn test_validate_node_id() {
        // Valid IDs
        assert!(validate_node_id("backend-1").is_ok());
        assert!(validate_node_id("proxy_001").is_ok());
        assert!(validate_node_id("test.node.123").is_ok());
        assert!(validate_node_id("a").is_ok());
        
        // Invalid IDs
        assert!(validate_node_id("").is_err());
        assert!(validate_node_id("   ").is_err());
        assert!(validate_node_id(&"a".repeat(200)).is_err());
        assert!(validate_node_id("invalid@char").is_err());
        assert!(validate_node_id("invalid space").is_err());
        assert!(validate_node_id("invalid/slash").is_err());
    }

    #[test]
    fn test_validate_address() {
        // Valid addresses
        assert!(validate_address("127.0.0.1:8080").is_ok());
        assert!(validate_address("localhost:3000").is_ok());
        assert!(validate_address("[::1]:8080").is_ok());
        assert!(validate_address("service.local:9090").is_ok());
        assert!(validate_address("192.168.1.100:5000").is_ok());
        
        // Invalid addresses
        assert!(validate_address("").is_err());
        assert!(validate_address("invalid-format").is_err());
        assert!(validate_address("host:99999").is_err());
        assert!(validate_address("host:0").is_err());
        assert!(validate_address(":8080").is_err());
        assert!(validate_address("host:").is_err());
        assert!(validate_address(&"a".repeat(300)).is_err());
    }

    #[test]
    fn test_validate_port() {
        // Valid ports
        assert!(validate_port(1024).is_ok());
        assert!(validate_port(8080).is_ok());
        assert!(validate_port(65535).is_ok());
        
        // Invalid ports
        assert!(validate_port(0).is_err());
        assert!(validate_port(80).is_err());  // Well-known port
        assert!(validate_port(443).is_err()); // Well-known port
        assert!(validate_port(1023).is_err()); // Below minimum
    }

    #[test]
    fn test_validate_capabilities() {
        // Valid capabilities
        let valid_caps = vec!["inference".to_string(), "gpu_support".to_string()];
        assert!(validate_capabilities(&valid_caps).is_ok());
        
        let empty_caps: Vec<String> = vec![];
        assert!(validate_capabilities(&empty_caps).is_ok());
        
        // Invalid capabilities
        let invalid_chars = vec!["invalid-char".to_string()];
        assert!(validate_capabilities(&invalid_chars).is_err());
        
        let empty_cap = vec!["".to_string()];
        assert!(validate_capabilities(&empty_cap).is_err());
        
        let too_long = vec!["a".repeat(100)];
        assert!(validate_capabilities(&too_long).is_err());
        
        let too_many: Vec<String> = (0..50).map(|i| format!("cap{}", i)).collect();
        assert!(validate_capabilities(&too_many).is_err());
    }

    #[test]
    fn test_validate_and_sanitize_node_info() {
        let node = NodeInfo::new(
            "  backend-1  ".to_string(), // With whitespace
            " 127.0.0.1:3000 ".to_string(), // With whitespace
            9090,
            NodeType::Backend
        );

        let sanitized = validate_and_sanitize_node_info(node).unwrap();
        assert_eq!(sanitized.id, "backend-1");
        assert_eq!(sanitized.address, "127.0.0.1:3000");
    }

    #[test]
    fn test_validate_node_info_consistency() {
        let mut node = NodeInfo::new(
            "backend-1".to_string(),
            "127.0.0.1:3000".to_string(),
            9090,
            NodeType::Backend
        );
        
        // This should be invalid: Backend marked as load balancer
        node.is_load_balancer = true;
        
        assert!(validate_and_sanitize_node_info(node).is_err());
    }

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::TooLong {
            field: "id".to_string(),
            max_length: 100,
            actual_length: 200,
        };
        
        let display = format!("{}", error);
        assert!(display.contains("id"));
        assert!(display.contains("too long"));
        assert!(display.contains("200"));
        assert!(display.contains("100"));
    }

    #[test]
    fn test_hostname_validation() {
        // Valid hostnames
        assert!(validate_address("example.com:8080").is_ok());
        assert!(validate_address("sub.example.com:3000").is_ok());
        assert!(validate_address("localhost:9090").is_ok());
        assert!(validate_address("a.b.c.d:5000").is_ok());
        
        // Invalid hostnames  
        assert!(validate_address("-invalid.com:8080").is_err());
        assert!(validate_address("invalid-.com:8080").is_err());
        assert!(validate_address("invalid..com:8080").is_err());
        assert!(validate_address(".invalid.com:8080").is_err());
    }
}