//! SWIM Protocol Bootstrap and Discovery
//!
//! This module implements the bootstrap mechanism for SWIM clusters, allowing nodes
//! to discover and join existing clusters or form new ones.

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
use super::swim::{SwimCluster, SwimConfig10k};
use super::swim_network::{NetworkConfig, SwimNetwork};
use super::types::PeerInfo;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, instrument};

/// Bootstrap configuration for cluster formation
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// List of seed nodes to contact for cluster discovery
    pub seed_nodes: Vec<SocketAddr>,

    /// Maximum time to wait for cluster discovery
    pub discovery_timeout: Duration,

    /// Interval between discovery attempts
    pub discovery_interval: Duration,

    /// Minimum nodes required to form a cluster
    pub min_cluster_size: usize,

    /// Enable automatic cluster formation
    pub auto_form_cluster: bool,

    /// Join timeout for individual nodes
    pub join_timeout: Duration,

    /// Maximum join retry attempts
    pub max_join_retries: u32,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            seed_nodes: Vec::new(),
            discovery_timeout: Duration::from_secs(30),
            discovery_interval: Duration::from_secs(2),
            min_cluster_size: 1,
            auto_form_cluster: true,
            join_timeout: Duration::from_secs(5),
            max_join_retries: 3,
        }
    }
}

/// Bootstrap messages for cluster discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BootstrapMessage {
    /// Discovery request from a new node
    DiscoveryRequest {
        node_id: String,
        node_addr: SocketAddr,
        capabilities: Vec<String>,
    },

    /// Discovery response with cluster information
    DiscoveryResponse {
        cluster_id: String,
        leader_addr: SocketAddr,
        members: Vec<PeerInfo>,
        swim_config: SwimConfig10k,
    },

    /// Join request to cluster
    JoinRequest {
        node_info: PeerInfo,
        cluster_id: String,
    },

    /// Join response
    JoinResponse {
        success: bool,
        member_id: u32,
        members: Vec<PeerInfo>,
        reason: Option<String>,
    },

    /// Seed node announcement
    SeedAnnouncement {
        seed_addr: SocketAddr,
        cluster_info: ClusterInfo,
    },
}

/// Cluster information for discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub cluster_id: String,
    pub creation_time: SystemTime,
    pub member_count: usize,
    pub leader_addr: SocketAddr,
    pub swim_config: SwimConfig10k,
}

/// Bootstrap state machine states
#[derive(Debug, Clone, PartialEq)]
enum BootstrapState {
    /// Initial state, discovering clusters
    Discovering,

    /// Joining an existing cluster
    Joining,

    /// Forming a new cluster
    Forming,

    /// Successfully joined/formed cluster
    Connected,

    /// Bootstrap failed
    #[allow(dead_code)]
    Failed,
}

/// SWIM cluster bootstrap manager
pub struct SwimBootstrap {
    /// Bootstrap configuration
    config: BootstrapConfig,

    /// Local node information
    local_node_id: String,
    local_addr: SocketAddr,

    /// Current bootstrap state
    state: Arc<RwLock<BootstrapState>>,

    /// Discovered clusters
    discovered_clusters: Arc<RwLock<Vec<ClusterInfo>>>,

    /// Network transport
    network: Option<SwimNetwork>,

    /// Event channel for bootstrap events
    event_sender: mpsc::UnboundedSender<BootstrapEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<BootstrapEvent>>,

    /// Background tasks
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// Bootstrap events
#[derive(Debug, Clone)]
pub enum BootstrapEvent {
    /// Cluster discovered
    ClusterDiscovered(ClusterInfo),

    /// Successfully joined cluster
    JoinedCluster {
        cluster_id: String,
        member_count: usize,
    },

    /// Formed new cluster
    FormedCluster { cluster_id: String },

    /// Bootstrap failed
    BootstrapFailed { reason: String },
}

impl SwimBootstrap {
    /// Creates new bootstrap manager
    pub fn new(local_node_id: String, local_addr: SocketAddr, config: BootstrapConfig) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        Self {
            config,
            local_node_id,
            local_addr,
            state: Arc::new(RwLock::new(BootstrapState::Discovering)),
            discovered_clusters: Arc::new(RwLock::new(Vec::new())),
            network: None,
            event_sender,
            event_receiver: Some(event_receiver),
            tasks: Vec::new(),
        }
    }

    /// Starts the bootstrap process
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> ServiceDiscoveryResult<SwimCluster> {
        info!(
            node_id = %self.local_node_id,
            addr = %self.local_addr,
            seed_nodes = ?self.config.seed_nodes,
            "Starting SWIM cluster bootstrap"
        );

        // Initialize network transport
        let network_config = NetworkConfig::default();
        let mut network = SwimNetwork::new(self.local_addr, 0, network_config).await?;
        network.start().await?;
        self.network = Some(network);

        // Start discovery task
        let discovery_task = self.spawn_discovery_task().await;
        self.tasks.push(discovery_task);

        // Wait for bootstrap to complete
        let result = timeout(self.config.discovery_timeout, self.wait_for_bootstrap()).await;

        match result {
            Ok(Ok(cluster)) => {
                info!(
                    node_id = %self.local_node_id,
                    "Successfully bootstrapped SWIM cluster"
                );
                Ok(cluster)
            }
            Ok(Err(e)) => {
                error!(error = %e, "Bootstrap failed");
                Err(e)
            }
            Err(_) => {
                error!("Bootstrap timeout");
                Err(ServiceDiscoveryError::NetworkError {
                    operation: "bootstrap".to_string(),
                    error: "Bootstrap timeout".to_string(),
                })
            }
        }
    }

    /// Waits for bootstrap to complete
    async fn wait_for_bootstrap(&mut self) -> ServiceDiscoveryResult<SwimCluster> {
        let mut event_receiver = self.event_receiver.take().ok_or_else(|| {
            ServiceDiscoveryError::InternalError("Event receiver already taken".to_string())
        })?;

        while let Some(event) = event_receiver.recv().await {
            match event {
                BootstrapEvent::JoinedCluster {
                    cluster_id,
                    member_count,
                } => {
                    info!(
                        cluster_id = %cluster_id,
                        member_count = member_count,
                        "Joined existing cluster"
                    );

                    // Create cluster instance
                    let swim_config = SwimConfig10k::default();
                    let (cluster, _events) =
                        SwimCluster::new(self.local_node_id.clone(), self.local_addr, swim_config)
                            .await?;

                    return Ok(cluster);
                }

                BootstrapEvent::FormedCluster { cluster_id } => {
                    info!(
                        cluster_id = %cluster_id,
                        "Formed new cluster"
                    );

                    // Create cluster instance as leader
                    let swim_config = SwimConfig10k::default();
                    let (cluster, _events) =
                        SwimCluster::new(self.local_node_id.clone(), self.local_addr, swim_config)
                            .await?;

                    return Ok(cluster);
                }

                BootstrapEvent::BootstrapFailed { reason } => {
                    return Err(ServiceDiscoveryError::NetworkError {
                        operation: "bootstrap".to_string(),
                        error: reason,
                    });
                }

                _ => continue,
            }
        }

        Err(ServiceDiscoveryError::NetworkError {
            operation: "bootstrap".to_string(),
            error: "Bootstrap event channel closed".to_string(),
        })
    }

    /// Spawns the discovery task
    async fn spawn_discovery_task(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let state = Arc::clone(&self.state);
        let discovered = Arc::clone(&self.discovered_clusters);
        let event_sender = self.event_sender.clone();
        let local_node_id = self.local_node_id.clone();
        let local_addr = self.local_addr;

        tokio::spawn(async move {
            let mut interval = interval(config.discovery_interval);

            loop {
                interval.tick().await;

                let current_state = state.read().await.clone();
                if current_state != BootstrapState::Discovering {
                    break;
                }

                // Contact seed nodes
                for seed_addr in &config.seed_nodes {
                    debug!(seed_addr = %seed_addr, "Contacting seed node");

                    // Send discovery request
                    let _request = BootstrapMessage::DiscoveryRequest {
                        node_id: local_node_id.clone(),
                        node_addr: local_addr,
                        capabilities: vec!["swim".to_string()],
                    };

                    // TODO: Send actual network message to seed node
                    // For now, simulate discovery
                    if rand::random::<f64>() > 0.5 {
                        let cluster_info = ClusterInfo {
                            cluster_id: format!("cluster-{}", seed_addr.port()),
                            creation_time: SystemTime::now(),
                            member_count: 10,
                            leader_addr: *seed_addr,
                            swim_config: SwimConfig10k::default(),
                        };

                        discovered.write().await.push(cluster_info.clone());

                        let _ = event_sender.send(BootstrapEvent::ClusterDiscovered(cluster_info));
                    }
                }

                // Check if we should join or form cluster
                let discovered_count = discovered.read().await.len();

                if discovered_count > 0 {
                    // Join existing cluster
                    *state.write().await = BootstrapState::Joining;

                    // Attempt to join the first discovered cluster
                    if let Some(cluster) = discovered.read().await.first() {
                        info!(
                            cluster_id = %cluster.cluster_id,
                            leader_addr = %cluster.leader_addr,
                            "Attempting to join cluster"
                        );

                        // TODO: Send actual join request
                        // For now, simulate successful join
                        let _ = event_sender.send(BootstrapEvent::JoinedCluster {
                            cluster_id: cluster.cluster_id.clone(),
                            member_count: cluster.member_count + 1,
                        });

                        *state.write().await = BootstrapState::Connected;
                        break;
                    }
                } else if config.auto_form_cluster {
                    // No clusters found, form new one
                    info!("No clusters discovered, forming new cluster");

                    *state.write().await = BootstrapState::Forming;

                    let cluster_id = format!("cluster-{}", uuid::Uuid::new_v4());

                    let _ = event_sender.send(BootstrapEvent::FormedCluster { cluster_id });

                    *state.write().await = BootstrapState::Connected;
                    break;
                }
            }
        })
    }

    /// Handles incoming bootstrap messages
    pub async fn handle_bootstrap_message(
        &self,
        message: BootstrapMessage,
    ) -> ServiceDiscoveryResult<Option<BootstrapMessage>> {
        match message {
            BootstrapMessage::DiscoveryRequest {
                node_id, node_addr, ..
            } => {
                debug!(
                    node_id = %node_id,
                    node_addr = %node_addr,
                    "Received discovery request"
                );

                // If we're connected to a cluster, send cluster info
                if *self.state.read().await == BootstrapState::Connected {
                    // TODO: Get actual cluster info
                    let response = BootstrapMessage::DiscoveryResponse {
                        cluster_id: "test-cluster".to_string(),
                        leader_addr: self.local_addr,
                        members: Vec::new(),
                        swim_config: SwimConfig10k::default(),
                    };

                    return Ok(Some(response));
                }
            }

            BootstrapMessage::JoinRequest {
                node_info,
                cluster_id,
            } => {
                debug!(
                    node_id = %node_info.id,
                    cluster_id = %cluster_id,
                    "Received join request"
                );

                // TODO: Validate and process join request
                let response = BootstrapMessage::JoinResponse {
                    success: true,
                    member_id: rand::random(),
                    members: Vec::new(),
                    reason: None,
                };

                return Ok(Some(response));
            }

            _ => {}
        }

        Ok(None)
    }
}

impl Drop for SwimBootstrap {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Helper function to create bootstrap configuration from environment
pub fn bootstrap_config_from_env() -> BootstrapConfig {
    let mut config = BootstrapConfig::default();

    // Parse seed nodes from environment
    if let Ok(seed_nodes_str) = std::env::var("SWIM_SEED_NODES") {
        config.seed_nodes = seed_nodes_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
    }

    // Parse other configuration from environment
    if let Ok(timeout) = std::env::var("SWIM_DISCOVERY_TIMEOUT") {
        if let Ok(secs) = timeout.parse::<u64>() {
            config.discovery_timeout = Duration::from_secs(secs);
        }
    }

    if let Ok(min_size) = std::env::var("SWIM_MIN_CLUSTER_SIZE") {
        if let Ok(size) = min_size.parse() {
            config.min_cluster_size = size;
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bootstrap_creation() {
        let local_addr = crate::test_utils::get_random_port_addr();
        let config = BootstrapConfig::default();

        let bootstrap = SwimBootstrap::new("test-node".to_string(), local_addr, config);

        assert_eq!(bootstrap.local_node_id, "test-node");
        assert_eq!(bootstrap.local_addr, local_addr);
    }

    #[tokio::test]
    async fn test_bootstrap_config_from_env() {
        std::env::set_var("SWIM_SEED_NODES", "127.0.0.1:8000,127.0.0.1:8001");
        std::env::set_var("SWIM_DISCOVERY_TIMEOUT", "60");
        std::env::set_var("SWIM_MIN_CLUSTER_SIZE", "3");

        let config = bootstrap_config_from_env();

        assert_eq!(config.seed_nodes.len(), 2);
        assert_eq!(config.discovery_timeout, Duration::from_secs(60));
        assert_eq!(config.min_cluster_size, 3);

        // Clean up
        std::env::remove_var("SWIM_SEED_NODES");
        std::env::remove_var("SWIM_DISCOVERY_TIMEOUT");
        std::env::remove_var("SWIM_MIN_CLUSTER_SIZE");
    }
}
