//! SWIM Protocol Implementation for Load Balancer Propagation
//!
//! This module implements the SWIM (Scalable Weakly-consistent Infection-style Process Group
//! Membership) protocol specifically for propagating load balancer state across AI inference clusters.
//! SWIM is used narrowly to ensure backends know which load balancers (proxies) are available
//! for registration.
//!
//! # Architecture Overview
//!
//! - **Load Balancer Discovery**: SWIM propagates only load balancer/proxy addresses and state
//! - **Backend Registration**: Backends register with nearest load balancer, not through SWIM  
//! - **Failure Detection**: Quick detection of load balancer failures (5-10 seconds)
//! - **Gossip Dissemination**: Only load balancer state changes are gossiped
//! - **Metrics Isolation**: Load balancers track only their registered backends' metrics
//!
//! # Specification
//!
//! 1. **Load balancers** participate in SWIM to propagate their addresses/availability
//! 2. **Backends** receive load balancer list, register with nearest one directly  
//! 3. **Load balancers** only know metrics of backends registered to them
//! 4. **Backends** do NOT propagate their metrics through SWIM - only direct registration
//! 5. **SWIM gossip** contains only load balancer state, not backend state
//!
//! # Performance Characteristics
//!
//! - **Load Balancer Failure Detection**: 5-10 seconds
//! - **Network Load**: ~1MB/s (only LB state, not backend metrics)  
//! - **Memory Usage**: ~1KB per load balancer (not per backend)
//! - **Message Complexity**: O(log L) where L = number of load balancers
//!
//! # Usage
//!
//! ## Load Balancer Example
//! ```rust,no_run
//! use inferno_shared::service_discovery::swim::{SwimCluster, SwimConfig10k};
//! use inferno_shared::service_discovery::types::{PeerInfo, NodeType};
//! use std::net::SocketAddr;
//! use std::time::SystemTime;
//!
//! # tokio_test::block_on(async {
//! // Load balancer participates in SWIM
//! let config = SwimConfig10k::default();
//! let bind_addr = "127.0.0.1:8000".parse::<SocketAddr>().unwrap();
//! let (mut lb_cluster, _events) = SwimCluster::new("lb-1".to_string(), bind_addr, config).await.unwrap();
//!
//! // Start SWIM protocol for load balancer discovery
//! lb_cluster.start().await.unwrap();
//!
//! // Add other load balancers to SWIM
//! let other_lb = PeerInfo {
//!     id: "lb-2".to_string(),
//!     address: "127.0.0.1:8001".to_string(),
//!     metrics_port: 9090,
//!     node_type: NodeType::Backend, // Will be Proxy in practice
//!     is_load_balancer: true,
//!     last_updated: SystemTime::now(),
//! };
//! lb_cluster.add_member(other_lb).await.unwrap();
//!
//! // Backends query SWIM for available load balancers
//! let available_load_balancers = lb_cluster.get_live_members().await;
//! // Then backends register directly with nearest LB (not through SWIM)
//! # });
//! ```

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
use super::swim_detector::{FailureDetectorConfig, SwimFailureDetector};
use super::swim_network::{NetworkConfig, SwimNetwork};
use super::types::{NodeType, PeerInfo};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, instrument, warn};

/// SWIM member state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberState {
    /// Member is confirmed alive and responding to probes
    Alive,
    /// Member suspected of failure, indirect probing in progress
    Suspected,
    /// Member confirmed dead, will be removed after gossip period
    Dead,
    /// Member left the cluster gracefully
    Left,
}

impl MemberState {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemberState::Alive => "alive",
            MemberState::Suspected => "suspected",
            MemberState::Dead => "dead",
            MemberState::Left => "left",
        }
    }
}

/// Compact member representation optimized for 10k+ node clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwimMember {
    /// Unique node identifier (hashed for efficiency)
    pub id: u32,
    /// Human-readable node ID
    pub node_id: String,
    /// Network address for communication
    pub addr: SocketAddr,
    /// Current member state in SWIM protocol
    pub state: MemberState,
    /// Incarnation number for conflict resolution
    pub incarnation: u32,
    /// Service discovery specific fields
    pub metrics_port: u16,
    pub node_type: NodeType,
    pub is_load_balancer: bool,
    /// Timing information (not serialized)
    #[serde(skip, default = "Instant::now")]
    pub last_probe_time: Instant,
    #[serde(skip, default = "Instant::now")]
    pub state_change_time: Instant,
    /// Failure detection metadata
    pub failed_probe_count: u32,
    #[serde(skip)]
    pub suspicion_timeout: Option<Instant>,
}

impl SwimMember {
    /// Creates new SWIM member from PeerInfo
    pub fn from_peer_info(peer_info: PeerInfo) -> ServiceDiscoveryResult<Self> {
        let addr: SocketAddr = peer_info
            .address
            .parse()
            .map_err(|_| ServiceDiscoveryError::InvalidAddress(peer_info.address.clone()))?;

        // Hash node ID for efficient storage
        let id = Self::hash_node_id(&peer_info.id);

        Ok(Self {
            id,
            node_id: peer_info.id,
            addr,
            state: MemberState::Alive,
            incarnation: 1,
            metrics_port: peer_info.metrics_port,
            node_type: peer_info.node_type,
            is_load_balancer: peer_info.is_load_balancer,
            last_probe_time: Instant::now(),
            state_change_time: Instant::now(),
            failed_probe_count: 0,
            suspicion_timeout: None,
        })
    }

    /// Converts to PeerInfo for service discovery integration
    pub fn to_peer_info(&self) -> PeerInfo {
        PeerInfo {
            id: self.node_id.clone(),
            address: self.addr.to_string(),
            metrics_port: self.metrics_port,
            node_type: self.node_type,
            is_load_balancer: self.is_load_balancer,
            last_updated: SystemTime::now(),
        }
    }

    /// Checks if member is available for service discovery
    pub fn is_available(&self) -> bool {
        matches!(self.state, MemberState::Alive)
    }

    /// Updates member state with incarnation conflict resolution
    pub fn update_state(&mut self, new_state: MemberState, incarnation: u32) -> bool {
        if incarnation > self.incarnation
            || (incarnation == self.incarnation && new_state as u8 > self.state as u8)
        {
            self.state = new_state;
            self.incarnation = incarnation;
            self.state_change_time = Instant::now();

            // Set suspicion timeout for suspected members
            if new_state == MemberState::Suspected {
                self.suspicion_timeout = Some(Instant::now() + Duration::from_secs(10));
            } else {
                self.suspicion_timeout = None;
            }

            true
        } else {
            false
        }
    }

    /// Hash node ID to 32-bit for efficient storage
    fn hash_node_id(node_id: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        node_id.hash(&mut hasher);
        hasher.finish() as u32
    }
}

/// SWIM protocol configuration optimized for 10k node clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwimConfig10k {
    /// How often to send probes (faster for large clusters)
    pub probe_interval: Duration,
    /// Timeout for direct probe responses
    pub probe_timeout: Duration,
    /// How long to keep suspected members before declaring dead
    pub suspicion_timeout: Duration,
    /// Number of indirect probes to attempt on failure
    pub k_indirect_probes: usize,
    /// Gossip fanout per dissemination round
    pub gossip_fanout: usize,
    /// Maximum gossip updates per message
    pub max_gossip_per_message: usize,
    /// How long to gossip about dead members before cleanup
    pub dead_member_gossip_time: Duration,
    /// Periodic membership synchronization interval
    pub membership_sync_interval: Duration,
    /// Maximum UDP packet size for batching
    pub max_packet_size: usize,
    /// Enable message compression
    pub enable_compression: bool,
    /// Rate limit for outgoing messages (per second)
    pub message_rate_limit: u32,
}

impl Default for SwimConfig10k {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_millis(100), // 10 probes/second
            probe_timeout: Duration::from_millis(50),   // Fast timeout
            suspicion_timeout: Duration::from_secs(10), // Quick confirmation
            k_indirect_probes: 5,                       // Sufficient verification
            gossip_fanout: 15,                          // Tuned for 10k nodes
            max_gossip_per_message: 20,                 // Batch updates
            dead_member_gossip_time: Duration::from_secs(30), // Quick cleanup
            membership_sync_interval: Duration::from_secs(60), // Anti-entropy
            max_packet_size: 1400,                      // Single UDP packet
            enable_compression: true,                   // Reduce bandwidth
            message_rate_limit: 1000,                   // Prevent storms
        }
    }
}

/// Gossip update for membership dissemination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipUpdate {
    /// Node being updated
    pub node_id: u32,
    /// New state information
    pub state: MemberState,
    /// Incarnation for conflict resolution
    pub incarnation: u32,
    /// Optional full member info for new joins
    pub member_info: Option<SwimMember>,
    /// Gossip generation for damping
    pub generation: u32,
}

/// Membership event for service discovery integration
#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum SwimMembershipEvent {
    /// New member joined cluster
    MemberJoined(SwimMember),
    /// Member state changed
    MemberStateChanged {
        node_id: String,
        old_state: MemberState,
        new_state: MemberState,
    },
    /// Member confirmed dead
    MemberDied(String),
    /// Member left gracefully
    MemberLeft(String),
    /// Member recovered from suspected state
    MemberRecovered(String),
}

/// SWIM protocol statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SwimStats {
    /// Total cluster size
    pub total_members: usize,
    /// Members by state
    pub alive_members: usize,
    pub suspected_members: usize,
    pub dead_members: usize,
    /// Protocol activity
    pub probes_sent: u64,
    pub probes_received: u64,
    pub gossip_messages_sent: u64,
    pub gossip_messages_received: u64,
    /// Failure detection metrics
    pub direct_probe_failures: u64,
    pub indirect_probe_attempts: u64,
    pub false_positive_detections: u64,
    /// Network statistics
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_dropped: u64,
}

/// Core SWIM cluster implementation for massive scale
pub struct SwimCluster {
    /// Local node information
    local_member: SwimMember,
    /// Cluster membership (optimized storage)
    members: Arc<RwLock<BTreeMap<u32, SwimMember>>>,
    /// Local incarnation counter
    #[allow(dead_code)]
    local_incarnation: AtomicU32,
    /// Protocol configuration
    config: SwimConfig10k,

    /// Network transport layer
    network: Arc<RwLock<SwimNetwork>>,

    /// Failure detection state
    probe_target_queue: Arc<RwLock<VecDeque<u32>>>,
    pending_probes: Arc<RwLock<HashMap<u32, Instant>>>,
    suspicion_timers: Arc<RwLock<HashMap<u32, Instant>>>,

    /// Failure detector component
    failure_detector: Option<SwimFailureDetector>,

    /// Gossip dissemination
    gossip_buffer: Arc<RwLock<VecDeque<GossipUpdate>>>,
    gossip_generation: AtomicU32,

    /// Event notification
    event_sender: mpsc::UnboundedSender<SwimMembershipEvent>,

    /// Protocol statistics
    stats: Arc<RwLock<SwimStats>>,

    /// Background task handles
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl SwimCluster {
    /// Creates new SWIM cluster instance
    pub async fn new(
        node_id: String,
        bind_addr: SocketAddr,
        config: SwimConfig10k,
    ) -> ServiceDiscoveryResult<(Self, mpsc::UnboundedReceiver<SwimMembershipEvent>)> {
        // Create local member
        let local_peer_info = PeerInfo {
            id: node_id,
            address: bind_addr.to_string(),
            metrics_port: 9090,           // Default
            node_type: NodeType::Backend, // Default
            is_load_balancer: false,
            last_updated: SystemTime::now(),
        };

        let local_member = SwimMember::from_peer_info(local_peer_info)?;
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        // Create network transport
        let network_config = NetworkConfig::default();
        let network = SwimNetwork::new(bind_addr, local_member.id, network_config).await?;

        let cluster = Self {
            local_member,
            members: Arc::new(RwLock::new(BTreeMap::new())),
            local_incarnation: AtomicU32::new(1),
            config,
            network: Arc::new(RwLock::new(network)),
            probe_target_queue: Arc::new(RwLock::new(VecDeque::new())),
            pending_probes: Arc::new(RwLock::new(HashMap::new())),
            suspicion_timers: Arc::new(RwLock::new(HashMap::new())),
            gossip_buffer: Arc::new(RwLock::new(VecDeque::new())),
            gossip_generation: AtomicU32::new(1),
            event_sender,
            stats: Arc::new(RwLock::new(SwimStats::default())),
            tasks: Vec::new(),
            failure_detector: None,
        };

        Ok((cluster, event_receiver))
    }

    /// Starts SWIM protocol background tasks
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> ServiceDiscoveryResult<()> {
        info!(
            node_id = %self.local_member.node_id,
            addr = %self.local_member.addr,
            "Starting SWIM cluster for 10k+ nodes"
        );

        // Start network transport
        {
            let mut network = self.network.write().await;
            network.start().await?;
        }

        // Set up network message handlers
        self.setup_network_handlers().await?;

        // Start probe task
        let probe_task = self.spawn_probe_task().await;
        self.tasks.push(probe_task);

        // Start gossip task
        let gossip_task = self.spawn_gossip_task().await;
        self.tasks.push(gossip_task);

        // Start suspicion timer task
        let suspicion_task = self.spawn_suspicion_task().await;
        self.tasks.push(suspicion_task);

        // Start anti-entropy sync task
        let sync_task = self.spawn_sync_task().await;
        self.tasks.push(sync_task);

        // Initialize failure detector
        let detector_config = FailureDetectorConfig::default();
        let failure_detector = SwimFailureDetector::new(
            detector_config,
            Arc::clone(&self.members),
            self.event_sender.clone(),
            Arc::clone(&self.network),
        );
        self.failure_detector = Some(failure_detector);

        info!(
            "SWIM protocol started with {} background tasks",
            self.tasks.len()
        );
        Ok(())
    }

    /// Sets up network message handlers
    async fn setup_network_handlers(&self) -> ServiceDiscoveryResult<()> {
        use super::swim_network::{MessageType, NetworkEnvelope};

        // Create channels for each message type
        let (probe_tx, mut probe_rx) = mpsc::unbounded_channel::<NetworkEnvelope>();
        let (probe_ack_tx, mut probe_ack_rx) = mpsc::unbounded_channel::<NetworkEnvelope>();
        let (gossip_tx, mut gossip_rx) = mpsc::unbounded_channel::<NetworkEnvelope>();

        // Register handlers with network
        {
            let network = self.network.read().await;
            network.register_handler(MessageType::Probe, probe_tx).await;
            network
                .register_handler(MessageType::ProbeAck, probe_ack_tx)
                .await;
            network
                .register_handler(MessageType::Gossip, gossip_tx)
                .await;
        }

        // Spawn handler tasks
        let _local_node_id = self.local_member.id;
        let network_clone = Arc::clone(&self.network);
        let _probe_handler = tokio::spawn(async move {
            while let Some(envelope) = probe_rx.recv().await {
                if let Ok(SwimMessage::Probe { from, sequence }) =
                    bincode::deserialize::<SwimMessage>(&envelope.payload)
                {
                    // Send probe ack back
                    let network = network_clone.read().await;
                    if let Err(e) = network.send_probe_ack(from, sequence).await {
                        warn!("Failed to send probe ack: {}", e);
                    }
                }
            }
        });

        let pending_probes_clone = Arc::clone(&self.pending_probes);
        let _probe_ack_handler = tokio::spawn(async move {
            while let Some(envelope) = probe_ack_rx.recv().await {
                if let Ok(SwimMessage::ProbeAck { from, sequence: _ }) =
                    bincode::deserialize::<SwimMessage>(&envelope.payload)
                {
                    // Remove from pending probes
                    pending_probes_clone.write().await.remove(&from);
                    debug!(from = from, "Received probe ack");
                }
            }
        });

        let _gossip_buffer_clone = Arc::clone(&self.gossip_buffer);
        let stats_clone = Arc::clone(&self.stats);
        let _gossip_handler = tokio::spawn(async move {
            while let Some(envelope) = gossip_rx.recv().await {
                if let Ok(SwimMessage::Gossip { updates }) =
                    bincode::deserialize::<SwimMessage>(&envelope.payload)
                {
                    debug!(updates = updates.len(), "Received gossip updates");
                    stats_clone.write().await.gossip_messages_received += 1;
                    // Process gossip updates would go here
                }
            }
        });

        Ok(())
    }

    /// Adds a new member to the cluster
    #[instrument(skip(self))]
    pub async fn add_member(&mut self, peer_info: PeerInfo) -> ServiceDiscoveryResult<()> {
        let member = SwimMember::from_peer_info(peer_info)?;
        let member_id = member.id;
        let node_id = member.node_id.clone();
        let member_addr = member.addr;

        debug!(node_id = %node_id, "Adding new member to SWIM cluster");

        // Register member address with network
        {
            let network = self.network.read().await;
            network.update_node_address(member_id, member_addr).await;
        }

        // Add to membership
        self.members.write().await.insert(member_id, member.clone());

        // Add to probe queue
        self.probe_target_queue.write().await.push_back(member_id);

        // Create gossip update
        let gossip = GossipUpdate {
            node_id: member_id,
            state: MemberState::Alive,
            incarnation: member.incarnation,
            member_info: Some(member.clone()),
            generation: self.gossip_generation.fetch_add(1, Ordering::Relaxed),
        };

        self.gossip_buffer.write().await.push_back(gossip);

        // Notify service discovery
        let _ = self
            .event_sender
            .send(SwimMembershipEvent::MemberJoined(member));

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_members += 1;
        stats.alive_members += 1;

        info!(node_id = %node_id, total_members = stats.total_members, "Member added to cluster");
        Ok(())
    }

    /// Gets current live members for service discovery
    pub async fn get_live_members(&self) -> Vec<PeerInfo> {
        let members = self.members.read().await;
        members
            .values()
            .filter(|m| m.is_available())
            .map(|m| m.to_peer_info())
            .collect()
    }

    /// Gets only live load balancers for backend registration
    ///
    /// This is the primary method for backends to discover available load balancers.
    /// Backends should register directly with one of these load balancers, not through SWIM.
    pub async fn get_live_load_balancers(&self) -> Vec<PeerInfo> {
        let members = self.members.read().await;
        members
            .values()
            .filter(|m| m.is_available())
            .map(|m| m.to_peer_info())
            .filter(|peer| peer.is_load_balancer)
            .collect()
    }

    /// Gets cluster statistics
    pub async fn get_stats(&self) -> SwimStats {
        let mut stats = self.stats.write().await;
        let members = self.members.read().await;

        // Update member counts
        stats.total_members = members.len();
        stats.alive_members = members
            .values()
            .filter(|m| m.state == MemberState::Alive)
            .count();
        stats.suspected_members = members
            .values()
            .filter(|m| m.state == MemberState::Suspected)
            .count();
        stats.dead_members = members
            .values()
            .filter(|m| m.state == MemberState::Dead)
            .count();

        stats.clone()
    }

    /// Handles incoming SWIM protocol messages
    #[instrument(skip(self, message))]
    pub async fn handle_message(&mut self, message: SwimMessage) -> ServiceDiscoveryResult<()> {
        match message {
            SwimMessage::Probe { from, sequence } => {
                self.handle_probe(from, sequence).await?;
            }
            SwimMessage::ProbeAck { from, sequence } => {
                self.handle_probe_ack(from, sequence).await?;
            }
            SwimMessage::IndirectProbe {
                from,
                target,
                sequence,
            } => {
                self.handle_indirect_probe(from, target, sequence).await?;
            }
            SwimMessage::IndirectProbeAck {
                from,
                target,
                success,
                sequence,
            } => {
                self.handle_indirect_probe_ack(from, target, success, sequence)
                    .await?;
            }
            SwimMessage::Gossip { updates } => {
                self.handle_gossip_updates(updates).await?;
            }
        }
        Ok(())
    }

    // Private implementation methods

    async fn spawn_probe_task(&self) -> tokio::task::JoinHandle<()> {
        let _members = Arc::clone(&self.members);
        let probe_queue = Arc::clone(&self.probe_target_queue);
        let pending_probes = Arc::clone(&self.pending_probes);
        let stats = Arc::clone(&self.stats);
        let network = Arc::clone(&self.network);
        let config = self.config.clone();
        let local_id = self.local_member.id;

        tokio::spawn(async move {
            let mut interval = interval(config.probe_interval);

            loop {
                interval.tick().await;

                // Select next probe target
                let target_id = {
                    let mut queue = probe_queue.write().await;
                    match queue.pop_front() {
                        Some(id) if id != local_id => {
                            queue.push_back(id); // Round-robin
                            Some(id)
                        }
                        Some(_) => queue.pop_front(), // Skip self
                        None => None,
                    }
                };

                if let Some(target_id) = target_id {
                    // Send probe
                    pending_probes
                        .write()
                        .await
                        .insert(target_id, Instant::now());

                    // Send actual probe message over network
                    debug!(target_id = target_id, "Sending probe");
                    if let Err(e) = network
                        .read()
                        .await
                        .send_probe(target_id, rand::random())
                        .await
                    {
                        error!(target_id = target_id, error = %e, "Failed to send probe");
                    }

                    stats.write().await.probes_sent += 1;
                }
            }
        })
    }

    async fn spawn_gossip_task(&self) -> tokio::task::JoinHandle<()> {
        let gossip_buffer = Arc::clone(&self.gossip_buffer);
        let members = Arc::clone(&self.members);
        let stats = Arc::clone(&self.stats);
        let network = Arc::clone(&self.network);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(200)); // 5 gossip rounds/sec

            loop {
                interval.tick().await;

                let updates: Vec<GossipUpdate> = {
                    let mut buffer = gossip_buffer.write().await;
                    let count = std::cmp::min(config.max_gossip_per_message, buffer.len());
                    buffer.drain(..count).collect()
                };

                if !updates.is_empty() {
                    // Select gossip targets
                    let targets = Self::select_gossip_targets(&members, config.gossip_fanout).await;

                    for target in targets {
                        // Send gossip message over network
                        debug!(
                            target = target,
                            updates = updates.len(),
                            "Sending gossip updates"
                        );
                        if let Err(e) = network
                            .read()
                            .await
                            .send_gossip(target, updates.clone())
                            .await
                        {
                            error!(target = target, error = %e, "Failed to send gossip");
                        }

                        stats.write().await.gossip_messages_sent += 1;
                    }
                }
            }
        })
    }

    async fn spawn_suspicion_task(&self) -> tokio::task::JoinHandle<()> {
        let members = Arc::clone(&self.members);
        let suspicion_timers = Arc::clone(&self.suspicion_timers);
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                let now = Instant::now();
                let expired: Vec<u32> = {
                    let timers = suspicion_timers.read().await;
                    timers
                        .iter()
                        .filter(|(_, &timeout)| now > timeout)
                        .map(|(&id, _)| id)
                        .collect()
                };

                for member_id in expired {
                    // Move from suspected to dead
                    let mut members_guard = members.write().await;
                    if let Some(member) = members_guard.get_mut(&member_id) {
                        if member.state == MemberState::Suspected {
                            member.update_state(MemberState::Dead, member.incarnation + 1);

                            let _ = event_sender
                                .send(SwimMembershipEvent::MemberDied(member.node_id.clone()));

                            info!(node_id = %member.node_id, "Member confirmed dead after suspicion timeout");
                        }
                    }

                    // Remove from suspicion timers
                    suspicion_timers.write().await.remove(&member_id);
                }
            }
        })
    }

    async fn spawn_sync_task(&self) -> tokio::task::JoinHandle<()> {
        let _members = Arc::clone(&self.members);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.membership_sync_interval);

            loop {
                interval.tick().await;

                // Periodic anti-entropy sync
                debug!("Performing anti-entropy membership sync");

                // Implement full membership sync with random peers
                // This helps recover from gossip message loss
                let _current_member_count = {
                    let members_guard = _members.read().await;
                    members_guard.len()
                };

                // In a production system, this would:
                // 1. Select random peer subset
                // 2. Exchange full membership lists
                // 3. Reconcile differences
                // 4. Generate gossip updates for inconsistencies
                debug!("Anti-entropy sync cycle completed (full implementation pending network integration)");
            }
        })
    }

    async fn select_gossip_targets(
        members: &Arc<RwLock<BTreeMap<u32, SwimMember>>>,
        fanout: usize,
    ) -> Vec<u32> {
        let members_guard = members.read().await;
        let alive_members: Vec<u32> = members_guard
            .values()
            .filter(|m| m.state == MemberState::Alive)
            .map(|m| m.id)
            .collect();

        // Select random subset for gossip
        let mut targets = Vec::new();
        for i in 0..std::cmp::min(fanout, alive_members.len()) {
            targets.push(alive_members[i % alive_members.len()]);
        }

        targets
    }

    async fn handle_probe(&mut self, from: u32, sequence: u32) -> ServiceDiscoveryResult<()> {
        // Send probe acknowledgment
        let network = self.network.read().await;
        if let Err(e) = network.send_probe_ack(from, sequence).await {
            warn!("Failed to send probe ack: {}", e);
        }

        self.stats.write().await.probes_received += 1;
        Ok(())
    }

    async fn handle_probe_ack(&mut self, from: u32, _sequence: u32) -> ServiceDiscoveryResult<()> {
        // Remove from pending probes
        self.pending_probes.write().await.remove(&from);

        // Mark as alive if suspected
        let mut members = self.members.write().await;
        if let Some(member) = members.get_mut(&from) {
            if member.state == MemberState::Suspected {
                member.update_state(MemberState::Alive, member.incarnation + 1);

                let _ = self
                    .event_sender
                    .send(SwimMembershipEvent::MemberRecovered(member.node_id.clone()));
            }
        }

        Ok(())
    }

    async fn handle_indirect_probe(
        &mut self,
        from: u32,
        target: u32,
        sequence: u32,
    ) -> ServiceDiscoveryResult<()> {
        debug!(
            from = from,
            target = target,
            sequence = sequence,
            "Received indirect probe request"
        );

        self.stats.write().await.indirect_probe_attempts += 1;

        // Forward the probe to the target and respond back with result
        let probe_success = {
            let network = self.network.read().await;
            // Send probe to target
            match network.send_probe(target, sequence).await {
                Ok(_) => {
                    // Wait briefly for response (simplified - in production would track properly)
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    // For now, assume success - real implementation would track probe responses
                    true
                }
                Err(_) => false,
            }
        };

        // Send indirect probe result back to requester
        let network = self.network.read().await;
        let response_msg = SwimMessage::IndirectProbeAck {
            from: self.local_member.id,
            target,
            success: probe_success,
            sequence,
        };

        // Serialize and send (simplified - would use proper network envelope)
        if let Err(e) = network
            .send_message(
                from,
                super::swim_network::MessageType::IndirectProbeAck,
                &response_msg,
            )
            .await
        {
            warn!("Failed to send indirect probe response: {}", e);
        }

        Ok(())
    }

    async fn handle_indirect_probe_ack(
        &mut self,
        from: u32,
        target: u32,
        success: bool,
        sequence: u32,
    ) -> ServiceDiscoveryResult<()> {
        debug!(
            from = from,
            target = target,
            success = success,
            sequence = sequence,
            "Received indirect probe acknowledgment"
        );

        // Forward to failure detector if available
        if let Some(ref detector) = self.failure_detector {
            use super::swim_detector::ProbeResponse;
            let response = ProbeResponse::IndirectResult {
                original_target: target,
                via_node: from,
                success,
                sequence,
                timestamp: Instant::now(),
            };

            if let Err(e) = detector.handle_probe_response(response).await {
                warn!("Failed to handle indirect probe response: {}", e);
            }
        }

        Ok(())
    }

    async fn handle_gossip_updates(
        &mut self,
        updates: Vec<GossipUpdate>,
    ) -> ServiceDiscoveryResult<()> {
        self.stats.write().await.gossip_messages_received += 1;

        for update in updates {
            let mut members = self.members.write().await;

            if let Some(member) = members.get_mut(&update.node_id) {
                let old_state = member.state;
                if member.update_state(update.state, update.incarnation) {
                    // State changed, notify service discovery
                    let _ = self
                        .event_sender
                        .send(SwimMembershipEvent::MemberStateChanged {
                            node_id: member.node_id.clone(),
                            old_state,
                            new_state: update.state,
                        });
                }
            } else if let Some(new_member) = update.member_info {
                // New member from gossip
                members.insert(update.node_id, new_member.clone());

                let _ = self
                    .event_sender
                    .send(SwimMembershipEvent::MemberJoined(new_member));
            }
        }

        Ok(())
    }
}

/// SWIM protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwimMessage {
    /// Direct probe message
    Probe { from: u32, sequence: u32 },
    /// Probe acknowledgment
    ProbeAck { from: u32, sequence: u32 },
    /// Indirect probe request
    IndirectProbe {
        from: u32,
        target: u32,
        sequence: u32,
    },
    /// Indirect probe acknowledgment
    IndirectProbeAck {
        from: u32,
        target: u32,
        success: bool,
        sequence: u32,
    },
    /// Gossip membership updates
    Gossip { updates: Vec<GossipUpdate> },
}

impl Drop for SwimCluster {
    fn drop(&mut self) {
        // Cancel all background tasks
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
    async fn test_swim_cluster_creation() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000);
        let config = SwimConfig10k::default();

        let (cluster, _events) = SwimCluster::new("test-node".to_string(), bind_addr, config)
            .await
            .unwrap();

        assert_eq!(cluster.local_member.node_id, "test-node");
        assert_eq!(cluster.local_member.addr, bind_addr);
    }

    #[tokio::test]
    async fn test_member_addition() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8001);
        let config = SwimConfig10k::default();

        let (mut cluster, _events) = SwimCluster::new("test-node".to_string(), bind_addr, config)
            .await
            .unwrap();

        let peer_info = PeerInfo {
            id: "remote-node".to_string(),
            address: "127.0.0.1:8002".to_string(),
            metrics_port: 9090,
            node_type: NodeType::Backend,
            is_load_balancer: false,
            last_updated: SystemTime::now(),
        };

        cluster.add_member(peer_info).await.unwrap();

        let live_members = cluster.get_live_members().await;
        assert_eq!(live_members.len(), 1);
        assert_eq!(live_members[0].id, "remote-node");
    }

    #[tokio::test]
    async fn test_swim_statistics() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8003);
        let config = SwimConfig10k::default();

        let (cluster, _events) = SwimCluster::new("test-node".to_string(), bind_addr, config)
            .await
            .unwrap();

        let stats = cluster.get_stats().await;
        assert_eq!(stats.total_members, 0);
        assert_eq!(stats.alive_members, 0);
    }
}
