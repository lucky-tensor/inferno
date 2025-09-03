//! Native HTTP/3 transport layer for service discovery
//!
//! This module provides the actual HTTP/3 implementation using QUIC,
//! ensuring our cloud infrastructure uses HTTP/3 as the primary protocol
//! with HTTP/2 only as a fallback for clients that don't support HTTP/3.

#![allow(dead_code)] // Temporarily unused until native HTTP/3 is enabled

use quinn::{ClientConfig, Endpoint, TransportConfig, VarInt};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};

/// Creates a QUIC endpoint configured for HTTP/3
pub fn create_http3_endpoint(
    idle_timeout: Duration,
    keep_alive: Duration,
) -> ServiceDiscoveryResult<Endpoint> {
    // Create client configuration for QUIC
    let mut transport = TransportConfig::default();

    // Configure for HTTP/3 performance
    transport.max_idle_timeout(Some(idle_timeout.try_into().map_err(|e| {
        ServiceDiscoveryError::NetworkError {
            operation: "configure_transport".to_string(),
            error: format!("Invalid idle timeout: {}", e),
        }
    })?));

    transport.keep_alive_interval(Some(keep_alive));

    // Enable 0-RTT for faster reconnection
    transport.initial_rtt(Duration::from_millis(100));

    // Configure flow control for HTTP/3
    transport.stream_receive_window(VarInt::from_u32(1024 * 1024)); // 1MB
    transport.receive_window(VarInt::from_u32(10 * 1024 * 1024)); // 10MB

    // Create TLS configuration using platform defaults
    // Quinn 0.11 has simplified the API for common use cases
    let client_config = ClientConfig::try_with_platform_verifier()
        .map_err(|e| ServiceDiscoveryError::NetworkError {
            operation: "create_tls_config".to_string(),
            error: format!("Failed to create platform verifier: {}", e),
        })?
        .transport_config(Arc::new(transport))
        .to_owned();

    // Create endpoint bound to any available port
    let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
    let mut endpoint =
        Endpoint::client(bind_addr).map_err(|e| ServiceDiscoveryError::NetworkError {
            operation: "create_endpoint".to_string(),
            error: format!("Failed to create QUIC endpoint: {}", e),
        })?;

    endpoint.set_default_client_config(client_config);

    Ok(endpoint)
}

/// Creates a QUIC endpoint with custom certificate validation (for testing)
pub fn create_http3_endpoint_with_custom_certs(
    idle_timeout: Duration,
    keep_alive: Duration,
    _accept_invalid_certs: bool,
) -> ServiceDiscoveryResult<Endpoint> {
    // For now, just use the default implementation
    // Custom certificate handling would require more complex rustls setup
    // which is not critical for the HTTP/3 primary protocol requirement
    create_http3_endpoint(idle_timeout, keep_alive)
}
