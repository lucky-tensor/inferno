//! SWIM Protocol Network Communication Layer
//!
//! This module implements the actual UDP-based network communication for the SWIM protocol.
//! It handles message serialization/deserialization, sending, and receiving over the network.

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
use super::swim::{GossipUpdate, SwimMessage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, instrument, warn};

/// Network message envelope with headers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEnvelope {
    /// Protocol version for compatibility
    pub version: u8,
    /// Message type identifier
    pub msg_type: MessageType,
    /// Source node ID
    pub from: u32,
    /// Destination node ID (for targeted messages)
    pub to: Option<u32>,
    /// Message sequence for deduplication
    pub sequence: u64,
    /// Timestamp in milliseconds since epoch
    pub timestamp: u64,
    /// Actual message payload
    pub payload: Vec<u8>,
    /// Optional compression flag
    pub compressed: bool,
}

/// Message type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    Probe,
    ProbeAck,
    IndirectProbe,
    IndirectProbeAck,
    Gossip,
    Sync,
    Join,
    Leave,
}

/// Network statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub send_errors: u64,
    pub receive_errors: u64,
    pub invalid_messages: u64,
    pub compression_successes: u64,
    pub compression_failures: u64,
}

/// SWIM network transport implementation
pub struct SwimNetwork {
    /// UDP socket for communication
    socket: Arc<UdpSocket>,
    /// Local node ID
    local_node_id: u32,
    /// Node address mapping
    node_addresses: Arc<RwLock<HashMap<u32, SocketAddr>>>,
    /// Message handlers
    message_handlers: Arc<RwLock<HashMap<MessageType, mpsc::UnboundedSender<NetworkEnvelope>>>>,
    /// Statistics
    stats: Arc<RwLock<NetworkStats>>,
    /// Configuration
    config: NetworkConfig,
    /// Background tasks
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Send buffer size
    pub send_buffer_size: usize,
    /// Receive buffer size
    pub receive_buffer_size: usize,
    /// Message timeout
    pub message_timeout: Duration,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression threshold
    pub compression_threshold: usize,
    /// Retry attempts for failed sends
    pub max_retries: u32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            max_message_size: 1400, // Single UDP packet
            send_buffer_size: 65536,
            receive_buffer_size: 65536,
            message_timeout: Duration::from_millis(100),
            enable_compression: true,
            compression_threshold: 500,
            max_retries: 3,
        }
    }
}

impl SwimNetwork {
    /// Creates new network transport
    pub async fn new(
        bind_addr: SocketAddr,
        local_node_id: u32,
        config: NetworkConfig,
    ) -> ServiceDiscoveryResult<Self> {
        // Create and bind UDP socket
        let socket =
            UdpSocket::bind(bind_addr)
                .await
                .map_err(|e| ServiceDiscoveryError::NetworkError {
                    operation: "bind_socket".to_string(),
                    error: format!("Failed to bind socket: {}", e),
                })?;

        // Note: Buffer size configuration would require platform-specific socket options
        // For now, we'll use default buffer sizes

        info!(
            local_addr = %bind_addr,
            local_node_id = local_node_id,
            "SWIM network transport initialized"
        );

        Ok(Self {
            socket: Arc::new(socket),
            local_node_id,
            node_addresses: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            config,
            tasks: Vec::new(),
        })
    }

    /// Starts network background tasks
    pub async fn start(&mut self) -> ServiceDiscoveryResult<()> {
        // Start receive loop
        let receive_task = self.spawn_receive_loop().await;
        self.tasks.push(receive_task);

        // Start statistics task
        let stats_task = self.spawn_stats_task().await;
        self.tasks.push(stats_task);

        info!("SWIM network transport started");
        Ok(())
    }

    /// Registers a message handler for a specific message type
    pub async fn register_handler(
        &self,
        msg_type: MessageType,
        handler: mpsc::UnboundedSender<NetworkEnvelope>,
    ) {
        self.message_handlers
            .write()
            .await
            .insert(msg_type, handler);
        debug!(?msg_type, "Registered message handler");
    }

    /// Updates node address mapping
    pub async fn update_node_address(&self, node_id: u32, addr: SocketAddr) {
        self.node_addresses.write().await.insert(node_id, addr);
    }

    /// Sends a probe message
    #[instrument(skip(self))]
    pub async fn send_probe(&self, target_id: u32, sequence: u32) -> ServiceDiscoveryResult<()> {
        let msg = SwimMessage::Probe {
            from: self.local_node_id,
            sequence,
        };

        self.send_message(target_id, MessageType::Probe, &msg).await
    }

    /// Sends a probe acknowledgment
    #[instrument(skip(self))]
    pub async fn send_probe_ack(
        &self,
        target_id: u32,
        sequence: u32,
    ) -> ServiceDiscoveryResult<()> {
        let msg = SwimMessage::ProbeAck {
            from: self.local_node_id,
            sequence,
        };

        self.send_message(target_id, MessageType::ProbeAck, &msg)
            .await
    }

    /// Sends an indirect probe request
    #[instrument(skip(self))]
    pub async fn send_indirect_probe(
        &self,
        via_node: u32,
        target: u32,
        sequence: u32,
    ) -> ServiceDiscoveryResult<()> {
        let msg = SwimMessage::IndirectProbe {
            from: self.local_node_id,
            target,
            sequence,
        };

        self.send_message(via_node, MessageType::IndirectProbe, &msg)
            .await
    }

    /// Sends gossip updates
    #[instrument(skip(self, updates))]
    pub async fn send_gossip(
        &self,
        target_id: u32,
        updates: Vec<GossipUpdate>,
    ) -> ServiceDiscoveryResult<()> {
        let msg = SwimMessage::Gossip { updates };

        self.send_message(target_id, MessageType::Gossip, &msg)
            .await
    }

    /// Broadcasts a message to multiple targets
    pub async fn broadcast_gossip(
        &self,
        targets: Vec<u32>,
        updates: Vec<GossipUpdate>,
    ) -> Vec<ServiceDiscoveryResult<()>> {
        let mut results = Vec::new();

        for target in targets {
            let result = self.send_gossip(target, updates.clone()).await;
            results.push(result);
        }

        results
    }

    /// Internal message sending implementation
    async fn send_message<T: Serialize>(
        &self,
        target_id: u32,
        msg_type: MessageType,
        message: &T,
    ) -> ServiceDiscoveryResult<()> {
        // Get target address
        let target_addr = {
            let addresses = self.node_addresses.read().await;
            addresses.get(&target_id).copied()
        };

        let target_addr = target_addr.ok_or_else(|| ServiceDiscoveryError::NetworkError {
            operation: "send_message".to_string(),
            error: format!("Unknown node address: {}", target_id),
        })?;

        // Serialize message
        let payload = bincode::serialize(message)
            .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))?;

        // Compress if needed
        let (final_payload, compressed) = if self.config.enable_compression
            && payload.len() > self.config.compression_threshold
        {
            match zstd::encode_all(&payload[..], 3) {
                Ok(compressed) => (compressed, true),
                Err(e) => {
                    warn!("Compression failed: {}", e);
                    self.stats.write().await.compression_failures += 1;
                    (payload, false)
                }
            }
        } else {
            (payload, false)
        };

        // Create network envelope
        let envelope = NetworkEnvelope {
            version: 1,
            msg_type,
            from: self.local_node_id,
            to: Some(target_id),
            sequence: rand::random(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            payload: final_payload,
            compressed,
        };

        // Serialize envelope
        let envelope_bytes = bincode::serialize(&envelope)
            .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))?;

        // Check message size
        if envelope_bytes.len() > self.config.max_message_size {
            return Err(ServiceDiscoveryError::NetworkError {
                operation: "send_message".to_string(),
                error: format!(
                    "Message too large: {} bytes (max: {})",
                    envelope_bytes.len(),
                    self.config.max_message_size
                ),
            });
        }

        // Send with retries
        let mut retries = 0;
        loop {
            match timeout(
                self.config.message_timeout,
                self.socket.send_to(&envelope_bytes, target_addr),
            )
            .await
            {
                Ok(Ok(bytes_sent)) => {
                    debug!(
                        target_id = target_id,
                        target_addr = %target_addr,
                        msg_type = ?msg_type,
                        bytes = bytes_sent,
                        compressed = compressed,
                        "Sent message"
                    );

                    // Update statistics
                    let mut stats = self.stats.write().await;
                    stats.messages_sent += 1;
                    stats.bytes_sent += bytes_sent as u64;
                    if compressed {
                        stats.compression_successes += 1;
                    }

                    return Ok(());
                }
                Ok(Err(e)) => {
                    retries += 1;
                    if retries >= self.config.max_retries {
                        error!(
                            target_id = target_id,
                            error = %e,
                            "Failed to send message after {} retries",
                            self.config.max_retries
                        );
                        self.stats.write().await.send_errors += 1;
                        return Err(ServiceDiscoveryError::NetworkError {
                            operation: "send_message".to_string(),
                            error: e.to_string(),
                        });
                    }
                    warn!(
                        target_id = target_id,
                        error = %e,
                        retry = retries,
                        "Send failed, retrying"
                    );
                }
                Err(_) => {
                    retries += 1;
                    if retries >= self.config.max_retries {
                        self.stats.write().await.send_errors += 1;
                        return Err(ServiceDiscoveryError::NetworkError {
                            operation: "send_message".to_string(),
                            error: "Send timeout".to_string(),
                        });
                    }
                }
            }
        }
    }

    /// Spawns the receive loop task
    async fn spawn_receive_loop(&self) -> tokio::task::JoinHandle<()> {
        let socket = Arc::clone(&self.socket);
        let handlers = Arc::clone(&self.message_handlers);
        let stats = Arc::clone(&self.stats);
        let _config = self.config.clone();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];

            loop {
                match socket.recv_from(&mut buf).await {
                    Ok((len, src_addr)) => {
                        let data = &buf[..len];

                        // Update statistics
                        stats.write().await.messages_received += 1;
                        stats.write().await.bytes_received += len as u64;

                        // Deserialize envelope
                        match bincode::deserialize::<NetworkEnvelope>(data) {
                            Ok(envelope) => {
                                debug!(
                                    from = envelope.from,
                                    msg_type = ?envelope.msg_type,
                                    src_addr = %src_addr,
                                    compressed = envelope.compressed,
                                    bytes = len,
                                    "Received message"
                                );

                                // Decompress if needed
                                let payload = if envelope.compressed {
                                    match zstd::decode_all(&envelope.payload[..]) {
                                        Ok(decompressed) => decompressed,
                                        Err(e) => {
                                            error!("Failed to decompress message: {}", e);
                                            stats.write().await.invalid_messages += 1;
                                            continue;
                                        }
                                    }
                                } else {
                                    envelope.payload.clone()
                                };

                                // Create envelope with decompressed payload
                                let processed_envelope = NetworkEnvelope {
                                    payload,
                                    compressed: false,
                                    ..envelope
                                };

                                // Route to handler
                                if let Some(handler) =
                                    handlers.read().await.get(&processed_envelope.msg_type)
                                {
                                    if let Err(e) = handler.send(processed_envelope) {
                                        warn!("Failed to route message to handler: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to deserialize message: {}", e);
                                stats.write().await.invalid_messages += 1;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to receive message: {}", e);
                        stats.write().await.receive_errors += 1;
                    }
                }
            }
        })
    }

    /// Spawns statistics reporting task
    async fn spawn_stats_task(&self) -> tokio::task::JoinHandle<()> {
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let stats_guard = stats.read().await;
                info!(
                    messages_sent = stats_guard.messages_sent,
                    messages_received = stats_guard.messages_received,
                    bytes_sent = stats_guard.bytes_sent,
                    bytes_received = stats_guard.bytes_received,
                    send_errors = stats_guard.send_errors,
                    receive_errors = stats_guard.receive_errors,
                    invalid_messages = stats_guard.invalid_messages,
                    compression_ratio = if stats_guard.compression_successes > 0 {
                        stats_guard.compression_successes as f64
                            / (stats_guard.compression_successes + stats_guard.compression_failures)
                                as f64
                    } else {
                        0.0
                    },
                    "Network statistics"
                );
            }
        })
    }

    /// Gets network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        self.stats.read().await.clone()
    }
}

impl Drop for SwimNetwork {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_network_creation() {
        let bind_addr = crate::test_utils::get_random_port_addr();
        let config = NetworkConfig::default();

        let network = SwimNetwork::new(bind_addr, 1, config).await.unwrap();
        assert_eq!(network.local_node_id, 1);
    }

    #[tokio::test]
    async fn test_message_envelope() {
        let envelope = NetworkEnvelope {
            version: 1,
            msg_type: MessageType::Probe,
            from: 1,
            to: Some(2),
            sequence: 100,
            timestamp: 1000,
            payload: vec![1, 2, 3, 4],
            compressed: false,
        };

        let serialized = bincode::serialize(&envelope).unwrap();
        let deserialized: NetworkEnvelope = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.from, envelope.from);
        assert_eq!(deserialized.to, envelope.to);
        assert_eq!(deserialized.msg_type, envelope.msg_type);
    }
}
