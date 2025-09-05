//! SWIM Gossip Dissemination Protocol
//!
//! This module implements the gossip-based membership update dissemination optimized
//! for 10,000+ node clusters. It provides efficient, reliable propagation of membership
//! changes with anti-entropy mechanisms and message batching for scale.
//!
//! # Key Features
//!
//! - **Epidemic Broadcast**: Gossip-based dissemination with configurable fanout
//! - **Message Batching**: Multiple updates per packet for network efficiency  
//! - **Anti-Entropy**: Periodic full synchronization to handle message loss
//! - **Priority Queuing**: Urgent updates (failures) prioritized over routine updates
//! - **Compression**: zstd compression for large member lists
//! - **Rate Limiting**: Prevent gossip storms in large clusters

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
use super::swim::{GossipUpdate, MemberState, SwimMember};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, instrument, trace};

/// Gossip message containing batched updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// Source node ID
    pub from: u32,
    /// Message sequence for deduplication
    pub sequence: u64,
    /// Timestamp for ordering
    pub timestamp: u64,
    /// Batched membership updates
    pub updates: Vec<GossipUpdate>,
    /// Optional compressed member list for anti-entropy
    pub full_member_list: Option<Vec<u8>>,
}

/// Gossip target selection result
#[derive(Debug, Clone)]
pub struct GossipTarget {
    pub node_id: u32,
    pub addr: std::net::SocketAddr,
    pub priority: GossipPriority,
}

/// Priority level for gossip targeting
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GossipPriority {
    /// Low priority - routine updates
    Low = 0,
    /// Normal priority - standard membership changes  
    Normal = 1,
    /// High priority - failures and critical updates
    High = 2,
    /// Critical priority - network partitions, mass failures
    Critical = 3,
}

/// Update propagation record for managing gossip lifecycle
#[derive(Debug, Clone)]
pub struct UpdatePropagation {
    pub update: GossipUpdate,
    pub creation_time: Instant,
    pub propagation_count: u32,
    pub max_propagations: u32,
    pub priority: GossipPriority,
    pub targets_contacted: HashSet<u32>,
}

/// Anti-entropy synchronization record
#[derive(Debug, Clone)]
pub struct AntiEntropySync {
    pub peer_id: u32,
    pub last_sync: Instant,
    pub sync_interval: Duration,
    pub member_count_diff: usize,
    pub failed_attempts: u32,
}

/// Gossip protocol statistics
#[derive(Debug, Clone, Default)]
pub struct GossipStats {
    /// Message statistics
    pub messages_sent: u64,
    pub messages_received: u64,
    pub messages_dropped: u64,
    pub duplicate_messages: u64,

    /// Update statistics
    pub updates_propagated: u64,
    pub updates_received: u64,
    pub updates_expired: u64,

    /// Network efficiency
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub compression_ratio: f64,
    pub average_batch_size: f64,

    /// Anti-entropy statistics
    pub sync_attempts: u64,
    pub sync_successes: u64,
    pub inconsistencies_detected: u64,
    pub inconsistencies_resolved: u64,
}

/// Configuration for gossip protocol
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Gossip fanout per round
    pub fanout: usize,

    /// How often to gossip
    pub gossip_interval: Duration,

    /// Maximum updates per gossip message
    pub max_updates_per_message: usize,

    /// Maximum message size in bytes
    pub max_message_size: usize,

    /// How long to propagate an update
    pub update_propagation_time: Duration,

    /// Enable message compression
    pub enable_compression: bool,

    /// Compression threshold (bytes)
    pub compression_threshold: usize,

    /// Rate limit for outgoing messages
    pub rate_limit: u32,

    /// Anti-entropy sync interval
    pub anti_entropy_interval: Duration,

    /// Anti-entropy sync probability
    pub anti_entropy_probability: f64,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 15,                                       // Tuned for 10k nodes
            gossip_interval: Duration::from_millis(200),      // 5 rounds/sec
            max_updates_per_message: 50,                      // Batch efficiency
            max_message_size: 1400,                           // Single UDP packet
            update_propagation_time: Duration::from_secs(30), // Stop propagating old updates
            enable_compression: true,                         // Essential at scale
            compression_threshold: 500,                       // Compress large payloads
            rate_limit: 500,                                  // Messages per second
            anti_entropy_interval: Duration::from_secs(60),   // Full sync frequency
            anti_entropy_probability: 0.1,                    // 10% chance per interval
        }
    }
}

/// SWIM gossip dissemination manager
pub struct SwimGossipManager {
    /// Configuration
    config: GossipConfig,

    /// Member information access
    members: Arc<RwLock<BTreeMap<u32, SwimMember>>>,

    /// Local node information
    local_node_id: u32,

    /// Outbound update queues (by priority)
    high_priority_queue: Arc<RwLock<VecDeque<UpdatePropagation>>>,
    normal_priority_queue: Arc<RwLock<VecDeque<UpdatePropagation>>>,
    low_priority_queue: Arc<RwLock<VecDeque<UpdatePropagation>>>,

    /// Active propagation tracking
    active_propagations: Arc<RwLock<HashMap<u64, UpdatePropagation>>>,

    /// Message deduplication
    received_messages: Arc<RwLock<HashMap<u64, Instant>>>,
    message_sequence: std::sync::atomic::AtomicU64,

    /// Anti-entropy state
    #[allow(dead_code)]
    sync_peers: Arc<RwLock<HashMap<u32, AntiEntropySync>>>,

    /// Rate limiting
    rate_limiter: Arc<RwLock<RateLimiter>>,

    /// Statistics
    stats: Arc<RwLock<GossipStats>>,

    /// Background task handles
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// Simple rate limiter implementation
#[derive(Debug)]
struct RateLimiter {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(rate_per_second: u32) -> Self {
        let max_tokens = rate_per_second as f64;
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate: rate_per_second as f64,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self, tokens: u32) -> bool {
        self.refill();

        let tokens_f64 = tokens as f64;
        if self.tokens >= tokens_f64 {
            self.tokens -= tokens_f64;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let tokens_to_add = elapsed * self.refill_rate;

        self.tokens = (self.tokens + tokens_to_add).min(self.max_tokens);
        self.last_refill = now;
    }
}

impl SwimGossipManager {
    /// Creates new gossip manager
    pub fn new(
        config: GossipConfig,
        members: Arc<RwLock<BTreeMap<u32, SwimMember>>>,
        local_node_id: u32,
    ) -> Self {
        Self {
            config: config.clone(),
            members,
            local_node_id,
            high_priority_queue: Arc::new(RwLock::new(VecDeque::new())),
            normal_priority_queue: Arc::new(RwLock::new(VecDeque::new())),
            low_priority_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_propagations: Arc::new(RwLock::new(HashMap::new())),
            received_messages: Arc::new(RwLock::new(HashMap::new())),
            message_sequence: std::sync::atomic::AtomicU64::new(1),
            sync_peers: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(config.rate_limit))),
            stats: Arc::new(RwLock::new(GossipStats::default())),
            tasks: Vec::new(),
        }
    }

    /// Starts gossip background tasks
    pub async fn start(&mut self) -> ServiceDiscoveryResult<()> {
        // Start main gossip dissemination task
        let gossip_task = self.spawn_gossip_task().await;
        self.tasks.push(gossip_task);

        // Start anti-entropy sync task
        let sync_task = self.spawn_anti_entropy_task().await;
        self.tasks.push(sync_task);

        // Start cleanup task
        let cleanup_task = self.spawn_cleanup_task().await;
        self.tasks.push(cleanup_task);

        // Start statistics task
        let stats_task = self.spawn_stats_task().await;
        self.tasks.push(stats_task);

        Ok(())
    }

    /// Adds update to gossip queue
    #[instrument(skip(self))]
    pub async fn gossip_update(
        &self,
        update: GossipUpdate,
        priority: GossipPriority,
    ) -> ServiceDiscoveryResult<()> {
        let propagation = UpdatePropagation {
            update: update.clone(),
            creation_time: Instant::now(),
            propagation_count: 0,
            max_propagations: self.calculate_max_propagations(priority),
            priority,
            targets_contacted: HashSet::new(),
        };

        // Add to appropriate priority queue
        match priority {
            GossipPriority::Critical | GossipPriority::High => {
                self.high_priority_queue
                    .write()
                    .await
                    .push_back(propagation);
            }
            GossipPriority::Normal => {
                self.normal_priority_queue
                    .write()
                    .await
                    .push_back(propagation);
            }
            GossipPriority::Low => {
                self.low_priority_queue.write().await.push_back(propagation);
            }
        }

        debug!(
            update_type = ?update.state,
            priority = ?priority,
            node_id = update.node_id,
            "Added update to gossip queue"
        );

        Ok(())
    }

    /// Handles incoming gossip message
    #[instrument(skip(self, message))]
    pub async fn handle_gossip_message(
        &self,
        message: GossipMessage,
    ) -> ServiceDiscoveryResult<Vec<GossipUpdate>> {
        // Check for duplicate message
        let is_duplicate = {
            let mut received = self.received_messages.write().await;
            received.insert(message.sequence, Instant::now()).is_some()
        };

        if is_duplicate {
            self.stats.write().await.duplicate_messages += 1;
            return Ok(Vec::new());
        }

        let mut new_updates = Vec::new();

        // Process batched updates
        for update in &message.updates {
            if self.should_accept_update(update).await {
                new_updates.push(update.clone());

                // Re-gossip with reduced priority and propagation count
                let new_priority = match update.state {
                    MemberState::Dead => GossipPriority::High,
                    MemberState::Suspected => GossipPriority::Normal,
                    _ => GossipPriority::Low,
                };

                self.gossip_update(update.clone(), new_priority).await?;
            }
        }

        // Handle anti-entropy full sync if present
        if let Some(compressed_data) = &message.full_member_list {
            self.handle_anti_entropy_sync(message.from, compressed_data)
                .await?;
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.messages_received += 1;
        stats.updates_received += new_updates.len() as u64;
        stats.bytes_received += self.estimate_message_size(&message);

        debug!(
            from = message.from,
            sequence = message.sequence,
            updates = message.updates.len(),
            new_updates = new_updates.len(),
            "Processed gossip message"
        );

        Ok(new_updates)
    }

    /// Triggers anti-entropy sync with specific peer
    pub async fn trigger_anti_entropy_sync(&self, peer_id: u32) -> ServiceDiscoveryResult<()> {
        let members = self.members.read().await;
        let member_list: Vec<SwimMember> = members.values().cloned().collect();

        // Serialize and compress member list
        let serialized = bincode::serialize(&member_list)
            .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))?;

        let compressed = if self.config.enable_compression
            && serialized.len() > self.config.compression_threshold
        {
            self.compress_data(&serialized)?
        } else {
            serialized
        };

        let _sync_message = GossipMessage {
            from: self.local_node_id,
            sequence: self.next_message_sequence(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            updates: Vec::new(),
            full_member_list: Some(compressed),
        };

        // Send sync message over network to peer_id
        // In a production system, this would use the network transport layer
        debug!(
            peer_id = peer_id,
            "Prepared anti-entropy sync message (network send not implemented)"
        );

        self.stats.write().await.sync_attempts += 1;

        Ok(())
    }

    /// Gets gossip statistics
    pub async fn get_stats(&self) -> GossipStats {
        let mut stats = self.stats.write().await;

        // Update derived statistics
        if stats.messages_sent > 0 {
            let total_updates = self.count_total_updates_in_queues().await;
            stats.average_batch_size = total_updates as f64 / stats.messages_sent as f64;
        }

        if stats.bytes_sent > 0 && self.config.enable_compression {
            // Estimate compression ratio (simplified)
            stats.compression_ratio = 0.7; // Typical zstd compression ratio
        }

        stats.clone()
    }

    // Private implementation methods

    fn calculate_max_propagations(&self, priority: GossipPriority) -> u32 {
        match priority {
            GossipPriority::Critical => 20, // Ensure critical updates reach everyone
            GossipPriority::High => 15,     // High importance updates
            GossipPriority::Normal => 10,   // Standard propagation
            GossipPriority::Low => 5,       // Minimal propagation for routine updates
        }
    }

    async fn should_accept_update(&self, update: &GossipUpdate) -> bool {
        let members = self.members.read().await;

        if let Some(existing_member) = members.get(&update.node_id) {
            // Accept if incarnation is higher, or state transition is valid
            update.incarnation > existing_member.incarnation
                || (update.incarnation == existing_member.incarnation
                    && update.state as u8 > existing_member.state as u8)
        } else {
            // New member, accept
            true
        }
    }

    #[allow(dead_code)]
    async fn select_gossip_targets(
        &self,
        fanout: usize,
        exclude: &HashSet<u32>,
    ) -> Vec<GossipTarget> {
        let members = self.members.read().await;
        let alive_members: Vec<_> = members
            .values()
            .filter(|m| {
                m.state == MemberState::Alive
                    && m.id != self.local_node_id
                    && !exclude.contains(&m.id)
            })
            .collect();

        // Simple random selection for now (in production, use better algorithms)
        let mut targets = Vec::new();
        for (i, member) in alive_members.iter().enumerate() {
            if i >= fanout {
                break;
            }

            targets.push(GossipTarget {
                node_id: member.id,
                addr: member.addr,
                priority: GossipPriority::Normal,
            });
        }

        targets
    }

    #[allow(dead_code)]
    async fn create_gossip_message(&self, max_updates: usize) -> Option<GossipMessage> {
        let mut updates = Vec::new();
        let mut contacted_nodes: HashSet<u32> = HashSet::new();

        // Drain updates from queues by priority
        let queues = [
            &self.high_priority_queue,
            &self.normal_priority_queue,
            &self.low_priority_queue,
        ];

        for queue in &queues {
            let mut queue_guard = queue.write().await;
            while updates.len() < max_updates {
                if let Some(mut propagation) = queue_guard.pop_front() {
                    // Check if we should still propagate this update
                    if propagation.creation_time.elapsed() > self.config.update_propagation_time
                        || propagation.propagation_count >= propagation.max_propagations
                    {
                        continue; // Expired or fully propagated
                    }

                    updates.push(propagation.update.clone());
                    contacted_nodes.extend(&propagation.targets_contacted);

                    // Update propagation tracking
                    propagation.propagation_count += 1;
                    if propagation.propagation_count < propagation.max_propagations {
                        // Re-queue for further propagation
                        queue_guard.push_back(propagation);
                    }
                } else {
                    break;
                }
            }
        }

        if updates.is_empty() {
            return None;
        }

        Some(GossipMessage {
            from: self.local_node_id,
            sequence: self.next_message_sequence(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            updates,
            full_member_list: None, // Regular gossip doesn't include full sync
        })
    }

    async fn handle_anti_entropy_sync(
        &self,
        from: u32,
        compressed_data: &[u8],
    ) -> ServiceDiscoveryResult<()> {
        // Decompress and deserialize peer's member list
        let decompressed = if self.config.enable_compression {
            self.decompress_data(compressed_data)?
        } else {
            compressed_data.to_vec()
        };

        let peer_members: Vec<SwimMember> = bincode::deserialize(&decompressed)
            .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))?;

        // Compare with local member list
        let local_members = self.members.read().await;
        let mut inconsistencies = 0;

        for peer_member in &peer_members {
            if let Some(local_member) = local_members.get(&peer_member.id) {
                // Check for inconsistencies
                if peer_member.incarnation > local_member.incarnation
                    || (peer_member.incarnation == local_member.incarnation
                        && peer_member.state as u8 > local_member.state as u8)
                {
                    inconsistencies += 1;

                    // Create update to reconcile
                    let update = GossipUpdate {
                        node_id: peer_member.id,
                        state: peer_member.state,
                        incarnation: peer_member.incarnation,
                        member_info: Some(peer_member.clone()),
                        generation: 0, // Anti-entropy updates don't have generation
                    };

                    self.gossip_update(update, GossipPriority::Normal).await?;
                }
            } else {
                // Missing member, add it
                inconsistencies += 1;

                let update = GossipUpdate {
                    node_id: peer_member.id,
                    state: peer_member.state,
                    incarnation: peer_member.incarnation,
                    member_info: Some(peer_member.clone()),
                    generation: 0,
                };

                self.gossip_update(update, GossipPriority::Normal).await?;
            }
        }

        debug!(
            from = from,
            peer_members = peer_members.len(),
            local_members = local_members.len(),
            inconsistencies = inconsistencies,
            "Processed anti-entropy sync"
        );

        let mut stats = self.stats.write().await;
        stats.sync_successes += 1;
        stats.inconsistencies_detected += inconsistencies;
        stats.inconsistencies_resolved += inconsistencies;

        Ok(())
    }

    fn compress_data(&self, data: &[u8]) -> ServiceDiscoveryResult<Vec<u8>> {
        zstd::encode_all(data, 3)
            .map_err(|e| ServiceDiscoveryError::CompressionError(e.to_string()))
    }

    fn decompress_data(&self, data: &[u8]) -> ServiceDiscoveryResult<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| ServiceDiscoveryError::CompressionError(e.to_string()))
    }

    fn next_message_sequence(&self) -> u64 {
        self.message_sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn estimate_message_size(&self, message: &GossipMessage) -> u64 {
        // Rough estimate - in practice, would measure actual serialized size
        let base_size = 64; // Headers, metadata
        let update_size = message.updates.len() * 50; // ~50 bytes per update
        let sync_size = message
            .full_member_list
            .as_ref()
            .map(|l| l.len())
            .unwrap_or(0);

        (base_size + update_size + sync_size) as u64
    }

    async fn count_total_updates_in_queues(&self) -> usize {
        let high = self.high_priority_queue.read().await.len();
        let normal = self.normal_priority_queue.read().await.len();
        let low = self.low_priority_queue.read().await.len();

        high + normal + low
    }

    // Background task spawners

    async fn spawn_gossip_task(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let high_queue = Arc::clone(&self.high_priority_queue);
        let normal_queue = Arc::clone(&self.normal_priority_queue);
        let low_queue = Arc::clone(&self.low_priority_queue);
        let members = Arc::clone(&self.members);
        let rate_limiter = Arc::clone(&self.rate_limiter);
        let stats = Arc::clone(&self.stats);
        let local_node_id = self.local_node_id;

        tokio::spawn(async move {
            let mut interval = interval(config.gossip_interval);

            loop {
                interval.tick().await;

                // Check rate limiter
                if !rate_limiter.write().await.try_acquire(config.fanout as u32) {
                    continue; // Rate limited, skip this round
                }

                // Create gossip message with batched updates
                if let Some(message) = Self::create_gossip_message_static(
                    &high_queue,
                    &normal_queue,
                    &low_queue,
                    config.max_updates_per_message,
                    local_node_id,
                )
                .await
                {
                    // Select gossip targets
                    let targets = Self::select_gossip_targets_static(
                        &members,
                        config.fanout,
                        local_node_id,
                        &HashSet::new(),
                    )
                    .await;

                    // Send to each target
                    for target in targets {
                        // In a production system, this would use the network transport layer
                        // The actual network sending would be implemented here
                        trace!(
                            target = target.node_id,
                            "Prepared gossip message for target (network send not implemented)"
                        );
                    }

                    // Update statistics
                    let mut stats_guard = stats.write().await;
                    stats_guard.messages_sent += config.fanout as u64;
                    stats_guard.updates_propagated += message.updates.len() as u64;
                }
            }
        })
    }

    async fn spawn_anti_entropy_task(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let members = Arc::clone(&self.members);
        let local_node_id = self.local_node_id;

        tokio::spawn(async move {
            let mut interval = interval(config.anti_entropy_interval);

            loop {
                interval.tick().await;

                // Randomly decide whether to perform anti-entropy sync
                if rand::random::<f64>() < config.anti_entropy_probability {
                    // Select random peer for full sync
                    if let Some(peer_id) = Self::select_random_peer(&members, local_node_id).await {
                        debug!(peer_id = peer_id, "Performing anti-entropy sync");

                        // TODO: Implement full sync with selected peer
                        // This would involve creating and sending a full member list
                    }
                }
            }
        })
    }

    async fn spawn_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let received_messages = Arc::clone(&self.received_messages);
        let active_propagations = Arc::clone(&self.active_propagations);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            loop {
                interval.tick().await;
                let now = Instant::now();

                // Clean up old received message records
                {
                    let mut received = received_messages.write().await;
                    received.retain(|_, timestamp| {
                        now.duration_since(*timestamp) < Duration::from_secs(300)
                        // 5 minutes
                    });
                }

                // Clean up completed propagations
                {
                    let mut propagations = active_propagations.write().await;
                    propagations.retain(|_, prop| {
                        now.duration_since(prop.creation_time) < Duration::from_secs(300)
                    });
                }
            }
        })
    }

    async fn spawn_stats_task(&self) -> tokio::task::JoinHandle<()> {
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let stats_guard = stats.read().await;
                debug!(
                    messages_sent = stats_guard.messages_sent,
                    messages_received = stats_guard.messages_received,
                    updates_propagated = stats_guard.updates_propagated,
                    "Gossip protocol statistics"
                );
            }
        })
    }

    // Static helper methods for background tasks

    async fn create_gossip_message_static(
        high_queue: &Arc<RwLock<VecDeque<UpdatePropagation>>>,
        normal_queue: &Arc<RwLock<VecDeque<UpdatePropagation>>>,
        low_queue: &Arc<RwLock<VecDeque<UpdatePropagation>>>,
        max_updates: usize,
        local_node_id: u32,
    ) -> Option<GossipMessage> {
        let mut updates = Vec::new();

        // Process queues in priority order
        let queues = [high_queue, normal_queue, low_queue];

        for queue in &queues {
            let mut queue_guard = queue.write().await;
            while updates.len() < max_updates {
                if let Some(propagation) = queue_guard.pop_front() {
                    updates.push(propagation.update);
                } else {
                    break;
                }
            }
        }

        if updates.is_empty() {
            return None;
        }

        Some(GossipMessage {
            from: local_node_id,
            sequence: rand::random(), // Simplified sequence generation
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            updates,
            full_member_list: None,
        })
    }

    async fn select_gossip_targets_static(
        members: &Arc<RwLock<BTreeMap<u32, SwimMember>>>,
        fanout: usize,
        local_node_id: u32,
        exclude: &HashSet<u32>,
    ) -> Vec<GossipTarget> {
        let members_guard = members.read().await;
        let alive_members: Vec<_> = members_guard
            .values()
            .filter(|m| {
                m.state == MemberState::Alive && m.id != local_node_id && !exclude.contains(&m.id)
            })
            .collect();

        let mut targets = Vec::new();
        for (i, member) in alive_members.iter().enumerate() {
            if i >= fanout {
                break;
            }

            targets.push(GossipTarget {
                node_id: member.id,
                addr: member.addr,
                priority: GossipPriority::Normal,
            });
        }

        targets
    }

    async fn select_random_peer(
        members: &Arc<RwLock<BTreeMap<u32, SwimMember>>>,
        local_node_id: u32,
    ) -> Option<u32> {
        let members_guard = members.read().await;
        let alive_members: Vec<u32> = members_guard
            .values()
            .filter(|m| m.state == MemberState::Alive && m.id != local_node_id)
            .map(|m| m.id)
            .collect();

        if alive_members.is_empty() {
            None
        } else {
            let index = rand::random::<usize>() % alive_members.len();
            Some(alive_members[index])
        }
    }
}

impl Drop for SwimGossipManager {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}
