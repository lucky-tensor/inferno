//! Native HTTP/3 client using QUIC for service discovery
//!
//! This implementation ensures HTTP/3 is used as the primary protocol
//! in our cloud infrastructure, with HTTP/2 only as a fallback.

#![allow(dead_code)] // Temporarily allow dead code until fully integrated

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};

use bytes::{Buf, Bytes};
use h3::client::SendRequest;
use h3_quinn::OpenStreams;
use quinn::{Connection, Endpoint};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Native HTTP/3 connection pool entry
struct Http3Connection {
    /// QUIC connection
    quic_conn: Connection,
    /// HTTP/3 send request handle
    h3_conn: SendRequest<OpenStreams, Bytes>,
    /// Last time this connection was used
    last_used: std::time::Instant,
}

/// Native HTTP/3 client implementation
pub struct NativeHttp3Client {
    /// QUIC endpoint for creating connections
    endpoint: Endpoint,
    /// Connection pool for reusing HTTP/3 connections
    connections: Arc<RwLock<HashMap<String, Http3Connection>>>,
    /// Client configuration
    config: super::http3_client::Http3ClientConfig,
    /// Metrics
    metrics: Arc<RwLock<super::http3_client::Http3Metrics>>,
}

impl NativeHttp3Client {
    /// Creates a new native HTTP/3 client
    pub async fn new(
        config: super::http3_client::Http3ClientConfig,
    ) -> ServiceDiscoveryResult<Self> {
        let endpoint = super::http3_transport::create_http3_endpoint(
            config.idle_timeout,
            config.keep_alive_interval,
        )?;

        Ok(Self {
            endpoint,
            connections: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(RwLock::new(super::http3_client::Http3Metrics::default())),
        })
    }

    /// Gets or creates an HTTP/3 connection
    async fn get_connection(
        &self,
        peer_url: &str,
    ) -> ServiceDiscoveryResult<SendRequest<OpenStreams, Bytes>> {
        let url = url::Url::parse(peer_url).map_err(|e| ServiceDiscoveryError::NetworkError {
            operation: "parse_url".to_string(),
            error: format!("Invalid URL '{}': {}", peer_url, e),
        })?;

        let host = url
            .host_str()
            .ok_or_else(|| ServiceDiscoveryError::NetworkError {
                operation: "parse_url".to_string(),
                error: format!("No host in URL: {}", peer_url),
            })?;

        let port = url.port().unwrap_or(443);
        let cache_key = format!("{}:{}", host, port);

        // Check for existing connection
        {
            let connections = self.connections.read().await;
            if let Some(conn) = connections.get(&cache_key) {
                if conn.last_used.elapsed() < self.config.idle_timeout {
                    debug!("Reusing HTTP/3 connection to {}", cache_key);
                    let mut metrics = self.metrics.write().await;
                    metrics.zero_rtt_connections += 1;
                    return Ok(conn.h3_conn.clone());
                }
            }
        }

        // Create new HTTP/3 connection
        info!("Creating new HTTP/3 connection to {}", cache_key);

        let addr: SocketAddr = format!("{}:{}", host, port).parse().map_err(|e| {
            ServiceDiscoveryError::NetworkError {
                operation: "resolve_address".to_string(),
                error: format!("Failed to resolve {}: {}", cache_key, e),
            }
        })?;

        // Connect via QUIC
        let connecting =
            self.endpoint
                .connect(addr, host)
                .map_err(|e| ServiceDiscoveryError::NetworkError {
                    operation: "quic_connect".to_string(),
                    error: format!("Failed to initiate QUIC connection: {}", e),
                })?;

        let quic_conn = connecting
            .await
            .map_err(|e| ServiceDiscoveryError::NetworkError {
                operation: "quic_handshake".to_string(),
                error: format!("QUIC handshake failed: {}", e),
            })?;

        // Check if we achieved 0-RTT
        let handshake = quic_conn.handshake_data();
        if let Some(data) = handshake {
            if let Ok(_data_bytes) = data.downcast::<Vec<u8>>() {
                // Simple heuristic: if we have handshake data, we might have 0-RTT
                let mut metrics = self.metrics.write().await;
                metrics.zero_rtt_connections += 1;
                debug!("Potential 0-RTT connection to {}", cache_key);
            }
        }

        // Establish HTTP/3
        let quinn_conn = h3_quinn::Connection::new(quic_conn.clone());
        let (mut h3_driver, h3_conn) =
            h3::client::new(quinn_conn)
                .await
                .map_err(|e| ServiceDiscoveryError::NetworkError {
                    operation: "http3_handshake".to_string(),
                    error: format!("HTTP/3 handshake failed: {}", e),
                })?;

        // Spawn driver task
        tokio::spawn(async move {
            let _ = h3_driver.wait_idle().await;
        });

        // Store connection in pool
        let entry = Http3Connection {
            quic_conn,
            h3_conn: h3_conn.clone(),
            last_used: std::time::Instant::now(),
        };

        let mut connections = self.connections.write().await;
        connections.insert(cache_key, entry);

        let mut metrics = self.metrics.write().await;
        metrics.active_connections = connections.len() as u64;

        Ok(h3_conn)
    }

    /// Performs a request using native HTTP/3
    pub async fn request(
        &self,
        method: &str,
        url: &str,
        headers: Vec<(&str, &str)>,
        body: Option<Bytes>,
    ) -> ServiceDiscoveryResult<(u16, Bytes)> {
        let mut h3_conn = self.get_connection(url).await?;

        // Build HTTP/3 request
        let uri = url
            .parse::<http::Uri>()
            .map_err(|e| ServiceDiscoveryError::NetworkError {
                operation: "parse_uri".to_string(),
                error: format!("Invalid URI: {}", e),
            })?;

        let mut req_builder = http::Request::builder().method(method).uri(uri);

        for (key, value) in headers {
            req_builder = req_builder.header(key, value);
        }

        let request = req_builder
            .body(())
            .map_err(|e| ServiceDiscoveryError::NetworkError {
                operation: "build_request".to_string(),
                error: format!("Failed to build request: {}", e),
            })?;

        // Send request
        let mut stream = h3_conn.send_request(request).await.map_err(|e| {
            ServiceDiscoveryError::NetworkError {
                operation: "send_request".to_string(),
                error: format!("Failed to send HTTP/3 request: {}", e),
            }
        })?;

        // Send body if present
        if let Some(data) = body {
            stream
                .send_data(data)
                .await
                .map_err(|e| ServiceDiscoveryError::NetworkError {
                    operation: "send_body".to_string(),
                    error: format!("Failed to send request body: {}", e),
                })?;
        }

        stream
            .finish()
            .await
            .map_err(|e| ServiceDiscoveryError::NetworkError {
                operation: "finish_request".to_string(),
                error: format!("Failed to finish request: {}", e),
            })?;

        // Receive response
        let response =
            stream
                .recv_response()
                .await
                .map_err(|e| ServiceDiscoveryError::NetworkError {
                    operation: "recv_response".to_string(),
                    error: format!("Failed to receive response: {}", e),
                })?;

        let status = response.status().as_u16();

        // Receive body
        let mut body_data = Vec::new();
        while let Some(mut chunk) =
            stream
                .recv_data()
                .await
                .map_err(|e| ServiceDiscoveryError::NetworkError {
                    operation: "recv_body".to_string(),
                    error: format!("Failed to receive body: {}", e),
                })?
        {
            body_data.extend_from_slice(&chunk.copy_to_bytes(chunk.remaining()));
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.bytes_received += body_data.len() as u64;

        Ok((status, Bytes::from(body_data)))
    }

    /// Cleans up idle connections
    pub async fn cleanup_idle_connections(&self) {
        let mut connections = self.connections.write().await;
        let now = std::time::Instant::now();

        connections.retain(|host, conn| {
            let is_active = now.duration_since(conn.last_used) < self.config.idle_timeout;
            if !is_active {
                debug!("Removing idle HTTP/3 connection to {}", host);
            }
            is_active
        });

        let mut metrics = self.metrics.write().await;
        metrics.active_connections = connections.len() as u64;
    }
}
