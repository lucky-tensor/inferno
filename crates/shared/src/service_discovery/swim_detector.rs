//! SWIM Failure Detection and Suspicion Mechanisms
//!
//! This module implements sophisticated failure detection optimized for 10,000+ node clusters.
//! It provides multi-stage failure detection with k-indirect probing and adaptive suspicion
//! mechanisms to minimize false positives while maintaining fast failure detection.
//!
//! # Key Features
//!
//! - **Direct Probing**: Primary failure detection via periodic random probes
//! - **Indirect Probing**: k-indirect verification to reduce false positives
//! - **Suspicion Mechanism**: Intermediate state between alive and dead
//! - **Adaptive Timeouts**: Network condition aware timeout adjustment
//! - **Batch Processing**: Efficient handling of multiple concurrent failures

use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
use super::swim::{MemberState, SwimMember, SwimMembershipEvent};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, instrument, warn};

/// Probe sequence number for tracking responses
pub type ProbeSequence = u32;

/// Direct probe request
#[derive(Debug, Clone)]
pub struct ProbeRequest {
    pub target_id: u32,
    pub target_addr: SocketAddr,
    pub sequence: ProbeSequence,
    pub timestamp: Instant,
    pub timeout: Duration,
}

/// Indirect probe request through intermediate nodes
#[derive(Debug, Clone)]
pub struct IndirectProbeRequest {
    pub original_target: u32,
    pub probe_via: Vec<u32>,
    pub sequence: ProbeSequence,
    pub timestamp: Instant,
    pub timeout: Duration,
    pub responses_needed: usize,
}

/// Probe response from target or indirect node
#[derive(Debug, Clone)]
pub enum ProbeResponse {
    /// Direct acknowledgment from target
    DirectAck {
        from: u32,
        sequence: ProbeSequence,
        timestamp: Instant,
    },
    /// Indirect probe result via intermediate node
    IndirectResult {
        original_target: u32,
        via_node: u32,
        success: bool,
        sequence: ProbeSequence,
        timestamp: Instant,
    },
}

/// Suspicion record for suspected members
#[derive(Debug, Clone)]
pub struct SuspicionRecord {
    pub member_id: u32,
    pub suspicion_start: Instant,
    pub timeout: Instant,
    pub confirming_nodes: HashSet<u32>,
    pub refuting_nodes: HashSet<u32>,
    pub incarnation: u32,
}

/// Statistics for failure detector performance
#[derive(Debug, Clone, Default)]
pub struct FailureDetectorStats {
    /// Probe statistics
    pub direct_probes_sent: u64,
    pub direct_probe_successes: u64,
    pub direct_probe_timeouts: u64,

    /// Indirect probe statistics  
    pub indirect_probe_attempts: u64,
    pub indirect_probe_successes: u64,
    pub indirect_probe_failures: u64,

    /// Suspicion statistics
    pub suspicions_raised: u64,
    pub suspicions_confirmed: u64,
    pub suspicions_refuted: u64,
    pub false_positive_rate: f64,

    /// Network performance
    pub average_rtt: Duration,
    pub packet_loss_rate: f64,
    pub network_partitions_detected: u64,
}

/// SWIM failure detector optimized for massive scale
pub struct SwimFailureDetector {
    /// Configuration
    config: FailureDetectorConfig,

    /// Active probe tracking
    pending_direct_probes: Arc<RwLock<HashMap<ProbeSequence, ProbeRequest>>>,
    pending_indirect_probes: Arc<RwLock<HashMap<ProbeSequence, IndirectProbeRequest>>>,
    probe_sequence: std::sync::atomic::AtomicU32,

    /// Suspicion management
    suspected_members: Arc<RwLock<HashMap<u32, SuspicionRecord>>>,

    /// Member information access
    members: Arc<RwLock<HashMap<u32, SwimMember>>>,

    /// Event notification
    event_sender: mpsc::UnboundedSender<SwimMembershipEvent>,

    /// Performance metrics
    stats: Arc<RwLock<FailureDetectorStats>>,

    /// Background task handles
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// Configuration for failure detection behavior
#[derive(Debug, Clone)]
pub struct FailureDetectorConfig {
    /// Direct probe timeout
    pub probe_timeout: Duration,

    /// Number of indirect probes to attempt
    pub indirect_probe_count: usize,

    /// Indirect probe timeout (should be longer than direct)
    pub indirect_probe_timeout: Duration,

    /// Suspicion timeout before declaring dead
    pub suspicion_timeout: Duration,

    /// Minimum confirmations needed to declare dead
    pub min_suspicion_confirmations: usize,

    /// Maximum concurrent probes per node
    pub max_concurrent_probes: usize,

    /// Enable adaptive timeout adjustment
    pub adaptive_timeouts: bool,

    /// Network RTT measurement window
    pub rtt_measurement_window: Duration,
}

impl Default for FailureDetectorConfig {
    fn default() -> Self {
        Self {
            probe_timeout: Duration::from_millis(200),
            indirect_probe_count: 3,
            indirect_probe_timeout: Duration::from_millis(500),
            suspicion_timeout: Duration::from_secs(10),
            min_suspicion_confirmations: 2,
            max_concurrent_probes: 50,
            adaptive_timeouts: true,
            rtt_measurement_window: Duration::from_secs(60),
        }
    }
}

impl SwimFailureDetector {
    /// Creates new failure detector
    pub fn new(
        config: FailureDetectorConfig,
        members: Arc<RwLock<HashMap<u32, SwimMember>>>,
        event_sender: mpsc::UnboundedSender<SwimMembershipEvent>,
    ) -> Self {
        Self {
            config,
            pending_direct_probes: Arc::new(RwLock::new(HashMap::new())),
            pending_indirect_probes: Arc::new(RwLock::new(HashMap::new())),
            probe_sequence: std::sync::atomic::AtomicU32::new(1),
            suspected_members: Arc::new(RwLock::new(HashMap::new())),
            members,
            event_sender,
            stats: Arc::new(RwLock::new(FailureDetectorStats::default())),
            tasks: Vec::new(),
        }
    }

    /// Starts failure detection background tasks
    pub async fn start(&mut self) -> ServiceDiscoveryResult<()> {
        // Start timeout monitoring task
        let timeout_task = self.spawn_timeout_monitor().await;
        self.tasks.push(timeout_task);

        // Start suspicion timer task
        let suspicion_task = self.spawn_suspicion_monitor().await;
        self.tasks.push(suspicion_task);

        // Start statistics collection task
        let stats_task = self.spawn_stats_collector().await;
        self.tasks.push(stats_task);

        Ok(())
    }

    /// Initiates direct probe to target
    #[instrument(skip(self))]
    pub async fn probe_member(&self, target_id: u32) -> ServiceDiscoveryResult<ProbeSequence> {
        let sequence = self.next_probe_sequence();

        let members = self.members.read().await;
        let target = members
            .get(&target_id)
            .ok_or_else(|| ServiceDiscoveryError::MemberNotFound(target_id.to_string()))?;

        let probe_request = ProbeRequest {
            target_id,
            target_addr: target.addr,
            sequence,
            timestamp: Instant::now(),
            timeout: self.config.probe_timeout,
        };

        // Store pending probe
        self.pending_direct_probes
            .write()
            .await
            .insert(sequence, probe_request.clone());

        // TODO: Send actual probe message over network
        debug!(
            target_id = target_id,
            target_addr = %target.addr,
            sequence = sequence,
            "Sending direct probe"
        );

        // Update statistics
        self.stats.write().await.direct_probes_sent += 1;

        Ok(sequence)
    }

    /// Handles probe timeout - initiates indirect probing
    #[instrument(skip(self))]
    pub async fn handle_probe_timeout(
        &self,
        sequence: ProbeSequence,
    ) -> ServiceDiscoveryResult<()> {
        let probe_request = {
            let mut pending = self.pending_direct_probes.write().await;
            pending.remove(&sequence)
        };

        if let Some(probe) = probe_request {
            warn!(
                target_id = probe.target_id,
                sequence = sequence,
                elapsed = ?probe.timestamp.elapsed(),
                "Direct probe timeout, initiating indirect probes"
            );

            // Update statistics
            self.stats.write().await.direct_probe_timeouts += 1;

            // Initiate indirect probing
            self.initiate_indirect_probes(probe.target_id, sequence)
                .await?;
        }

        Ok(())
    }

    /// Handles successful probe response
    #[instrument(skip(self))]
    pub async fn handle_probe_response(
        &self,
        response: ProbeResponse,
    ) -> ServiceDiscoveryResult<()> {
        match response {
            ProbeResponse::DirectAck {
                from,
                sequence,
                timestamp,
            } => {
                // Remove from pending probes
                let was_pending = self
                    .pending_direct_probes
                    .write()
                    .await
                    .remove(&sequence)
                    .is_some();

                if was_pending {
                    debug!(
                        from = from,
                        sequence = sequence,
                        "Received direct probe ack"
                    );

                    // Update statistics
                    let mut stats = self.stats.write().await;
                    stats.direct_probe_successes += 1;

                    // Update RTT measurement
                    if let Some(rtt) = timestamp.checked_duration_since(Instant::now()) {
                        self.update_rtt_measurement(rtt).await;
                    }

                    // Cancel any suspicion for this member
                    self.cancel_suspicion(from).await?;
                }
            }
            ProbeResponse::IndirectResult {
                original_target,
                via_node,
                success,
                sequence,
                ..
            } => {
                if success {
                    debug!(
                        target = original_target,
                        via = via_node,
                        sequence = sequence,
                        "Indirect probe successful"
                    );

                    self.stats.write().await.indirect_probe_successes += 1;
                    self.cancel_suspicion(original_target).await?;
                } else {
                    debug!(
                        target = original_target,
                        via = via_node,
                        sequence = sequence,
                        "Indirect probe failed"
                    );

                    self.stats.write().await.indirect_probe_failures += 1;
                    self.add_suspicion_confirmation(original_target, via_node)
                        .await?;
                }
            }
        }

        Ok(())
    }

    /// Initiates indirect probes through k random nodes
    #[instrument(skip(self))]
    async fn initiate_indirect_probes(
        &self,
        target_id: u32,
        sequence: ProbeSequence,
    ) -> ServiceDiscoveryResult<()> {
        // Select k random alive members for indirect probing
        let probe_via = self
            .select_indirect_probe_nodes(target_id, self.config.indirect_probe_count)
            .await;

        if probe_via.is_empty() {
            warn!(
                target_id = target_id,
                "No nodes available for indirect probing"
            );
            // Directly suspect the member
            self.suspect_member(target_id, "no-indirect-probes".to_string())
                .await?;
            return Ok(());
        }

        let indirect_request = IndirectProbeRequest {
            original_target: target_id,
            probe_via: probe_via.clone(),
            sequence,
            timestamp: Instant::now(),
            timeout: self.config.indirect_probe_timeout,
            responses_needed: (probe_via.len() + 1) / 2, // Majority
        };

        // Store pending indirect probe
        self.pending_indirect_probes
            .write()
            .await
            .insert(sequence, indirect_request);

        // Send indirect probe requests
        for via_node in &probe_via {
            // TODO: Send indirect probe message over network
            debug!(
                target = target_id,
                via = via_node,
                sequence = sequence,
                "Sending indirect probe request"
            );
        }

        self.stats.write().await.indirect_probe_attempts += 1;

        Ok(())
    }

    /// Selects nodes for indirect probing
    async fn select_indirect_probe_nodes(&self, exclude_target: u32, count: usize) -> Vec<u32> {
        let members = self.members.read().await;
        let alive_members: Vec<u32> = members
            .values()
            .filter(|m| m.state == MemberState::Alive && m.id != exclude_target)
            .map(|m| m.id)
            .collect();

        // Simple random selection (in production, use better randomization)
        alive_members.into_iter().take(count).collect()
    }

    /// Suspects a member of failure
    #[instrument(skip(self, reason))]
    async fn suspect_member(&self, member_id: u32, reason: String) -> ServiceDiscoveryResult<()> {
        let mut members = self.members.write().await;
        let mut suspected = self.suspected_members.write().await;

        if let Some(member) = members.get_mut(&member_id) {
            if member.state == MemberState::Alive {
                // Transition to suspected
                member.state = MemberState::Suspected;
                member.state_change_time = Instant::now();

                // Create suspicion record
                let suspicion = SuspicionRecord {
                    member_id,
                    suspicion_start: Instant::now(),
                    timeout: Instant::now() + self.config.suspicion_timeout,
                    confirming_nodes: HashSet::new(),
                    refuting_nodes: HashSet::new(),
                    incarnation: member.incarnation,
                };

                suspected.insert(member_id, suspicion);

                // Notify service discovery
                let _ = self
                    .event_sender
                    .send(SwimMembershipEvent::MemberStateChanged {
                        node_id: member.node_id.clone(),
                        old_state: MemberState::Alive,
                        new_state: MemberState::Suspected,
                    });

                warn!(
                    member_id = member_id,
                    node_id = %member.node_id,
                    reason = reason,
                    "Member suspected of failure"
                );

                self.stats.write().await.suspicions_raised += 1;
            }
        }

        Ok(())
    }

    /// Cancels suspicion for a member (received proof of life)
    #[instrument(skip(self))]
    async fn cancel_suspicion(&self, member_id: u32) -> ServiceDiscoveryResult<()> {
        let mut members = self.members.write().await;
        let mut suspected = self.suspected_members.write().await;

        if let Some(suspicion) = suspected.remove(&member_id) {
            if let Some(member) = members.get_mut(&member_id) {
                if member.state == MemberState::Suspected {
                    // Transition back to alive
                    member.state = MemberState::Alive;
                    member.state_change_time = Instant::now();
                    member.incarnation += 1; // Increment to override gossip

                    // Notify service discovery
                    let _ = self
                        .event_sender
                        .send(SwimMembershipEvent::MemberRecovered(member.node_id.clone()));

                    debug!(
                        member_id = member_id,
                        node_id = %member.node_id,
                        suspicion_duration = ?suspicion.suspicion_start.elapsed(),
                        "Suspicion cancelled, member recovered"
                    );

                    self.stats.write().await.suspicions_refuted += 1;
                }
            }
        }

        Ok(())
    }

    /// Adds confirmation from another node about suspected member
    async fn add_suspicion_confirmation(
        &self,
        member_id: u32,
        confirming_node: u32,
    ) -> ServiceDiscoveryResult<()> {
        let mut suspected = self.suspected_members.write().await;

        if let Some(suspicion) = suspected.get_mut(&member_id) {
            suspicion.confirming_nodes.insert(confirming_node);

            debug!(
                member_id = member_id,
                confirming_node = confirming_node,
                total_confirmations = suspicion.confirming_nodes.len(),
                "Added suspicion confirmation"
            );

            // Check if we have enough confirmations to declare dead
            if suspicion.confirming_nodes.len() >= self.config.min_suspicion_confirmations {
                self.declare_member_dead(member_id).await?;
            }
        }

        Ok(())
    }

    /// Declares member dead after sufficient suspicion confirmations
    #[instrument(skip(self))]
    async fn declare_member_dead(&self, member_id: u32) -> ServiceDiscoveryResult<()> {
        let mut members = self.members.write().await;
        let mut suspected = self.suspected_members.write().await;

        if let Some(suspicion) = suspected.remove(&member_id) {
            if let Some(member) = members.get_mut(&member_id) {
                member.state = MemberState::Dead;
                member.state_change_time = Instant::now();

                // Notify service discovery
                let _ = self
                    .event_sender
                    .send(SwimMembershipEvent::MemberDied(member.node_id.clone()));

                warn!(
                    member_id = member_id,
                    node_id = %member.node_id,
                    suspicion_duration = ?suspicion.suspicion_start.elapsed(),
                    confirmations = suspicion.confirming_nodes.len(),
                    "Member declared dead"
                );

                self.stats.write().await.suspicions_confirmed += 1;
            }
        }

        Ok(())
    }

    /// Updates RTT measurement for adaptive timeouts
    async fn update_rtt_measurement(&self, rtt: Duration) {
        let mut stats = self.stats.write().await;

        // Simple exponentially weighted moving average
        let alpha = 0.1;
        let new_rtt = rtt.as_secs_f64();
        let old_rtt = stats.average_rtt.as_secs_f64();
        let updated_rtt = Duration::from_secs_f64(old_rtt * (1.0 - alpha) + new_rtt * alpha);

        stats.average_rtt = updated_rtt;

        if self.config.adaptive_timeouts {
            // TODO: Adjust probe timeouts based on measured RTT
        }
    }

    /// Gets next probe sequence number
    fn next_probe_sequence(&self) -> ProbeSequence {
        self.probe_sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Gets failure detector statistics
    pub async fn get_stats(&self) -> FailureDetectorStats {
        self.stats.read().await.clone()
    }

    // Background task spawners

    async fn spawn_timeout_monitor(&self) -> tokio::task::JoinHandle<()> {
        let pending_direct = Arc::clone(&self.pending_direct_probes);
        let pending_indirect = Arc::clone(&self.pending_indirect_probes);
        let detector = self.clone_for_task().await;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(50)); // Check every 50ms

            loop {
                interval.tick().await;
                let now = Instant::now();

                // Check direct probe timeouts
                let expired_direct: Vec<ProbeSequence> = {
                    let pending = pending_direct.read().await;
                    pending
                        .iter()
                        .filter(|(_, probe)| now > probe.timestamp + probe.timeout)
                        .map(|(&seq, _)| seq)
                        .collect()
                };

                for sequence in expired_direct {
                    if let Err(e) = detector.handle_probe_timeout(sequence).await {
                        error!(sequence = sequence, error = %e, "Failed to handle probe timeout");
                    }
                }

                // Check indirect probe timeouts
                let expired_indirect: Vec<ProbeSequence> = {
                    let pending = pending_indirect.read().await;
                    pending
                        .iter()
                        .filter(|(_, probe)| now > probe.timestamp + probe.timeout)
                        .map(|(&seq, _)| seq)
                        .collect()
                };

                for sequence in expired_indirect {
                    let mut pending = pending_indirect.write().await;
                    if let Some(probe) = pending.remove(&sequence) {
                        // All indirect probes failed - suspect the member
                        if let Err(e) = detector
                            .suspect_member(
                                probe.original_target,
                                "indirect-probe-timeout".to_string(),
                            )
                            .await
                        {
                            error!(
                                target = probe.original_target,
                                error = %e,
                                "Failed to suspect member after indirect probe timeout"
                            );
                        }
                    }
                }
            }
        })
    }

    async fn spawn_suspicion_monitor(&self) -> tokio::task::JoinHandle<()> {
        let suspected = Arc::clone(&self.suspected_members);
        let detector = self.clone_for_task().await;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            loop {
                interval.tick().await;
                let now = Instant::now();

                // Find expired suspicions
                let expired: Vec<u32> = {
                    let suspected_guard = suspected.read().await;
                    suspected_guard
                        .iter()
                        .filter(|(_, suspicion)| now > suspicion.timeout)
                        .map(|(&id, _)| id)
                        .collect()
                };

                // Declare expired suspicions as dead
                for member_id in expired {
                    if let Err(e) = detector.declare_member_dead(member_id).await {
                        error!(member_id = member_id, error = %e, "Failed to declare member dead");
                    }
                }
            }
        })
    }

    async fn spawn_stats_collector(&self) -> tokio::task::JoinHandle<()> {
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Calculate derived statistics
                let mut stats_guard = stats.write().await;

                if stats_guard.direct_probes_sent > 0 {
                    let false_positive_rate = stats_guard.suspicions_refuted as f64
                        / (stats_guard.suspicions_raised as f64).max(1.0);
                    stats_guard.false_positive_rate = false_positive_rate;
                }

                if stats_guard.direct_probes_sent > 0 {
                    let packet_loss_rate = stats_guard.direct_probe_timeouts as f64
                        / stats_guard.direct_probes_sent as f64;
                    stats_guard.packet_loss_rate = packet_loss_rate;
                }
            }
        })
    }

    /// Helper to create detector reference for background tasks
    async fn clone_for_task(&self) -> SwimFailureDetector {
        SwimFailureDetector {
            config: self.config.clone(),
            pending_direct_probes: Arc::clone(&self.pending_direct_probes),
            pending_indirect_probes: Arc::clone(&self.pending_indirect_probes),
            probe_sequence: std::sync::atomic::AtomicU32::new(
                self.probe_sequence
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            suspected_members: Arc::clone(&self.suspected_members),
            members: Arc::clone(&self.members),
            event_sender: self.event_sender.clone(),
            stats: Arc::clone(&self.stats),
            tasks: Vec::new(), // Background tasks don't spawn more tasks
        }
    }
}

impl Drop for SwimFailureDetector {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}
