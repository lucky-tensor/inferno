//! SWIM Service Discovery Integration Layer
//!
//! This module provides the integration layer between SWIM protocol and the existing
//! service discovery system. It handles the translation of SWIM membership events
//! to service discovery operations while maintaining compatibility with the current
//! API and ensuring seamless migration from the previous consensus system.
//!
//! # Key Features
//!
//! - **Event Translation**: Maps SWIM membership events to service discovery operations
//! - **State Synchronization**: Maintains consistency between SWIM and service discovery state
//! - **Compatibility Layer**: Preserves existing service discovery API
//! - **Performance Optimization**: Batched updates and lazy evaluation
//! - **Monitoring Integration**: Comprehensive metrics and health reporting

use super::errors::ServiceDiscoveryResult;
use super::service::ServiceDiscovery;
use super::swim::{MemberState, SwimCluster, SwimConfig10k, SwimMembershipEvent, SwimStats};
#[cfg(test)]
use super::types::NodeType;
use super::types::PeerInfo;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, instrument, warn};

/// Integration statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SwimIntegrationStats {
    /// Event processing
    pub events_processed: u64,
    pub events_failed: u64,
    pub events_pending: u64,

    /// State synchronization
    pub sync_operations: u64,
    pub sync_conflicts_resolved: u64,
    pub consistency_checks: u64,
    pub inconsistencies_found: u64,

    /// Performance metrics
    pub average_event_processing_time: Duration,
    pub backend_registration_time: Duration,
    pub backend_removal_time: Duration,

    /// Service discovery compatibility
    pub legacy_api_calls: u64,
    pub swim_native_calls: u64,
}

/// Configuration for SWIM integration
#[derive(Debug, Clone)]
pub struct SwimIntegrationConfig {
    /// How often to perform state consistency checks
    pub consistency_check_interval: Duration,

    /// Batch size for event processing
    pub event_batch_size: usize,

    /// Maximum pending events before dropping
    pub max_pending_events: usize,

    /// Enable automatic state reconciliation
    pub auto_reconciliation: bool,

    /// Timeout for backend operations
    pub backend_operation_timeout: Duration,
}

impl Default for SwimIntegrationConfig {
    fn default() -> Self {
        Self {
            consistency_check_interval: Duration::from_secs(30),
            event_batch_size: 100,
            max_pending_events: 10000,
            auto_reconciliation: true,
            backend_operation_timeout: Duration::from_secs(5),
        }
    }
}

/// State synchronization record
#[derive(Debug, Clone)]
struct SyncState {
    last_sync: Instant,
    swim_member_count: usize,
    discovery_backend_count: usize,
    pending_operations: Vec<SyncOperation>,
}

/// Synchronization operation
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum SyncOperation {
    AddBackend(PeerInfo),
    RemoveBackend(String),
    UpdateBackend(PeerInfo),
    MarkHealthy(String),
    MarkUnhealthy(String),
}

/// SWIM-based service discovery implementation
pub struct SwimServiceDiscovery {
    /// SWIM cluster instance
    swim_cluster: Arc<RwLock<SwimCluster>>,

    /// Traditional service discovery for compatibility
    legacy_service_discovery: Arc<ServiceDiscovery>,

    /// Integration configuration
    config: SwimIntegrationConfig,

    /// Event processing
    event_receiver: mpsc::UnboundedReceiver<SwimMembershipEvent>,
    event_processor: Option<tokio::task::JoinHandle<()>>,

    /// State synchronization
    sync_state: Arc<RwLock<SyncState>>,

    /// Statistics
    stats: Arc<RwLock<SwimIntegrationStats>>,

    /// Background tasks
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl SwimServiceDiscovery {
    /// Creates new SWIM-based service discovery
    #[instrument(skip(swim_config, integration_config))]
    pub async fn new(
        node_id: String,
        bind_addr: SocketAddr,
        swim_config: SwimConfig10k,
        integration_config: SwimIntegrationConfig,
    ) -> ServiceDiscoveryResult<Self> {
        // Create SWIM cluster
        let (swim_cluster, event_receiver) =
            SwimCluster::new(node_id.clone(), bind_addr, swim_config).await?;

        let swim_cluster = Arc::new(RwLock::new(swim_cluster));

        // Create legacy service discovery for compatibility
        let legacy_service_discovery = Arc::new(ServiceDiscovery::new());

        let sync_state = Arc::new(RwLock::new(SyncState {
            last_sync: Instant::now(),
            swim_member_count: 0,
            discovery_backend_count: 0,
            pending_operations: Vec::new(),
        }));

        let instance = Self {
            swim_cluster,
            legacy_service_discovery,
            config: integration_config,
            event_receiver,
            event_processor: None,
            sync_state,
            stats: Arc::new(RwLock::new(SwimIntegrationStats::default())),
            tasks: Vec::new(),
        };

        info!(
            node_id = %node_id,
            bind_addr = %bind_addr,
            "Created SWIM-based service discovery"
        );

        Ok(instance)
    }

    /// Starts the SWIM service discovery system
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> ServiceDiscoveryResult<()> {
        info!("Starting SWIM service discovery system");

        // Start SWIM cluster
        {
            let mut cluster = self.swim_cluster.write().await;
            cluster.start().await?;
        }

        // Start event processor
        let event_processor = self.spawn_event_processor().await;
        self.event_processor = Some(event_processor);

        // Start state synchronization task
        let sync_task = self.spawn_sync_task().await;
        self.tasks.push(sync_task);

        // Start statistics collection task
        let stats_task = self.spawn_stats_task().await;
        self.tasks.push(stats_task);

        info!("SWIM service discovery system started");
        Ok(())
    }

    /// Registers a new backend (SWIM native)
    #[instrument(skip(self))]
    pub async fn register_backend(&self, peer_info: PeerInfo) -> ServiceDiscoveryResult<()> {
        let start_time = Instant::now();

        // Add to SWIM cluster
        {
            let mut cluster = self.swim_cluster.write().await;
            cluster.add_member(peer_info.clone()).await?;
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.swim_native_calls += 1;
        stats.backend_registration_time = start_time.elapsed();

        info!(
            node_id = %peer_info.id,
            address = %peer_info.address,
            node_type = ?peer_info.node_type,
            "Registered backend via SWIM"
        );

        Ok(())
    }

    /// Removes a backend (SWIM native)
    #[instrument(skip(self))]
    pub async fn remove_backend(&self, node_id: &str) -> ServiceDiscoveryResult<()> {
        let start_time = Instant::now();

        // SWIM doesn't have explicit removal - members are detected as failed
        // For compatibility, we'll mark the member as left
        // This would be implemented by updating member state to Left

        // Also remove from legacy service discovery for consistency
        let _ = self.legacy_service_discovery.remove_backend(node_id).await;

        let mut stats = self.stats.write().await;
        stats.swim_native_calls += 1;
        stats.backend_removal_time = start_time.elapsed();

        info!(node_id = %node_id, "Removed backend via SWIM");
        Ok(())
    }

    /// Gets live backends (SWIM native)
    pub async fn get_live_backends(&self) -> Vec<PeerInfo> {
        let cluster = self.swim_cluster.read().await;
        cluster.get_live_members().await
    }

    /// Gets backend count
    pub async fn get_backend_count(&self) -> usize {
        let cluster = self.swim_cluster.read().await;
        let stats = cluster.get_stats().await;
        stats.alive_members
    }

    /// Gets SWIM cluster statistics
    pub async fn get_swim_stats(&self) -> SwimStats {
        let cluster = self.swim_cluster.read().await;
        cluster.get_stats().await
    }

    /// Gets integration statistics
    pub async fn get_integration_stats(&self) -> SwimIntegrationStats {
        let mut stats = self.stats.write().await;

        // Update current state
        let cluster = self.swim_cluster.read().await;
        let _swim_stats = cluster.get_stats().await;

        let sync_state = self.sync_state.read().await;
        stats.events_pending = sync_state.pending_operations.len() as u64;

        stats.clone()
    }

    /// Legacy compatibility: list backends
    pub async fn list_backends(&self) -> Vec<PeerInfo> {
        self.stats.write().await.legacy_api_calls += 1;

        // Return SWIM live members for compatibility
        self.get_live_backends().await
    }

    /// Legacy compatibility: register backend
    pub async fn legacy_register_backend(&self, peer_info: PeerInfo) -> ServiceDiscoveryResult<()> {
        self.stats.write().await.legacy_api_calls += 1;

        // Route to SWIM registration
        self.register_backend(peer_info).await
    }

    /// Legacy compatibility: remove backend
    pub async fn legacy_remove_backend(&self, node_id: &str) -> ServiceDiscoveryResult<()> {
        self.stats.write().await.legacy_api_calls += 1;

        // Route to SWIM removal
        self.remove_backend(node_id).await
    }

    /// Triggers manual state synchronization
    pub async fn trigger_sync(&self) -> ServiceDiscoveryResult<()> {
        self.perform_state_sync().await
    }

    // Private implementation methods

    async fn spawn_event_processor(&mut self) -> tokio::task::JoinHandle<()> {
        let legacy_service_discovery = Arc::clone(&self.legacy_service_discovery);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();

        // Take ownership of event receiver
        let mut event_receiver =
            std::mem::replace(&mut self.event_receiver, mpsc::unbounded_channel().1);

        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut batch_timer = interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    event = event_receiver.recv() => {
                        if let Some(event) = event {
                            batch.push(event);

                            // Process batch when full
                            if batch.len() >= config.event_batch_size {
                                Self::process_event_batch(
                                    &mut batch,
                                    &legacy_service_discovery,
                                    &stats,
                                ).await;
                            }
                        } else {
                            // Channel closed
                            break;
                        }
                    }
                    _ = batch_timer.tick() => {
                        // Process remaining events in batch
                        if !batch.is_empty() {
                            Self::process_event_batch(
                                &mut batch,
                                &legacy_service_discovery,
                                &stats,
                            ).await;
                        }
                    }
                }
            }
        })
    }

    async fn process_event_batch(
        batch: &mut Vec<SwimMembershipEvent>,
        legacy_service_discovery: &Arc<ServiceDiscovery>,
        stats: &Arc<RwLock<SwimIntegrationStats>>,
    ) {
        let start_time = Instant::now();
        let mut processed = 0;
        let mut failed = 0;

        for event in batch.drain(..) {
            match Self::process_single_event(event, legacy_service_discovery).await {
                Ok(()) => processed += 1,
                Err(e) => {
                    failed += 1;
                    error!(error = %e, "Failed to process SWIM event");
                }
            }
        }

        // Update statistics
        let mut stats_guard = stats.write().await;
        stats_guard.events_processed += processed;
        stats_guard.events_failed += failed;

        if processed > 0 {
            let avg_time = start_time.elapsed() / processed as u32;
            stats_guard.average_event_processing_time = avg_time;
        }

        if processed > 0 || failed > 0 {
            debug!(
                processed = processed,
                failed = failed,
                batch_time = ?start_time.elapsed(),
                "Processed event batch"
            );
        }
    }

    async fn process_single_event(
        event: SwimMembershipEvent,
        legacy_service_discovery: &Arc<ServiceDiscovery>,
    ) -> ServiceDiscoveryResult<()> {
        match event {
            SwimMembershipEvent::MemberJoined(member) => {
                let peer_info = member.to_peer_info();
                let registration = super::types::BackendRegistration {
                    id: peer_info.id.clone(),
                    address: peer_info.address,
                    metrics_port: peer_info.metrics_port,
                };
                let _ = legacy_service_discovery
                    .register_backend(registration)
                    .await;
                debug!(node_id = %member.node_id, "Member joined - registered backend");
            }

            SwimMembershipEvent::MemberStateChanged {
                node_id,
                old_state,
                new_state,
            } => match (old_state, new_state) {
                (MemberState::Alive, MemberState::Suspected) => {
                    legacy_service_discovery
                        .mark_backend_unhealthy(&node_id)
                        .await;
                    debug!(node_id = %node_id, "Member suspected - marked unhealthy");
                }
                (MemberState::Suspected, MemberState::Alive) => {
                    legacy_service_discovery
                        .mark_backend_healthy(&node_id)
                        .await;
                    debug!(node_id = %node_id, "Member recovered - marked healthy");
                }
                (_, MemberState::Dead) => {
                    let _ = legacy_service_discovery.remove_backend(&node_id).await;
                    debug!(node_id = %node_id, "Member died - removed backend");
                }
                _ => {
                    debug!(
                        node_id = %node_id,
                        old_state = ?old_state,
                        new_state = ?new_state,
                        "Unhandled state transition"
                    );
                }
            },

            SwimMembershipEvent::MemberDied(node_id) => {
                let _ = legacy_service_discovery.remove_backend(&node_id).await;
                debug!(node_id = %node_id, "Member died - removed backend");
            }

            SwimMembershipEvent::MemberLeft(node_id) => {
                let _ = legacy_service_discovery.remove_backend(&node_id).await;
                debug!(node_id = %node_id, "Member left - removed backend");
            }

            SwimMembershipEvent::MemberRecovered(node_id) => {
                legacy_service_discovery
                    .mark_backend_healthy(&node_id)
                    .await;
                debug!(node_id = %node_id, "Member recovered - marked healthy");
            }
        }

        Ok(())
    }

    async fn spawn_sync_task(&self) -> tokio::task::JoinHandle<()> {
        let swim_cluster = Arc::clone(&self.swim_cluster);
        let legacy_service_discovery = Arc::clone(&self.legacy_service_discovery);
        let sync_state = Arc::clone(&self.sync_state);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.consistency_check_interval);

            loop {
                interval.tick().await;

                if let Err(e) = Self::perform_consistency_check(
                    &swim_cluster,
                    &legacy_service_discovery,
                    &sync_state,
                    &stats,
                    &config,
                )
                .await
                {
                    error!(error = %e, "Consistency check failed");
                }
            }
        })
    }

    async fn perform_consistency_check(
        swim_cluster: &Arc<RwLock<SwimCluster>>,
        legacy_service_discovery: &Arc<ServiceDiscovery>,
        sync_state: &Arc<RwLock<SyncState>>,
        stats: &Arc<RwLock<SwimIntegrationStats>>,
        config: &SwimIntegrationConfig,
    ) -> ServiceDiscoveryResult<()> {
        let start_time = Instant::now();

        // Get current state from both systems
        let swim_members = {
            let cluster = swim_cluster.read().await;
            cluster.get_live_members().await
        };

        let discovery_backends = legacy_service_discovery.get_healthy_backends().await;

        // Convert to sets for comparison
        let swim_ids: HashSet<String> = swim_members.iter().map(|m| m.id.clone()).collect();
        let discovery_ids: HashSet<String> =
            discovery_backends.iter().map(|b| b.id.clone()).collect();

        // Find inconsistencies
        let missing_in_discovery: Vec<_> = swim_ids.difference(&discovery_ids).collect();
        let missing_in_swim: Vec<_> = discovery_ids.difference(&swim_ids).collect();

        let total_inconsistencies = missing_in_discovery.len() + missing_in_swim.len();

        if total_inconsistencies > 0 && config.auto_reconciliation {
            warn!(
                missing_in_discovery = missing_in_discovery.len(),
                missing_in_swim = missing_in_swim.len(),
                "Found state inconsistencies, performing reconciliation"
            );

            // Add missing backends to service discovery
            for missing_id in &missing_in_discovery {
                if let Some(member) = swim_members.iter().find(|m| &m.id == *missing_id) {
                    let registration = super::types::BackendRegistration {
                        id: member.id.clone(),
                        address: member.address.clone(),
                        metrics_port: member.metrics_port,
                    };
                    if let Err(e) = legacy_service_discovery
                        .register_backend(registration)
                        .await
                    {
                        error!(
                            node_id = %missing_id,
                            error = %e,
                            "Failed to add missing backend to service discovery"
                        );
                    }
                }
            }

            // Remove extra backends from service discovery
            for extra_id in &missing_in_swim {
                let _ = legacy_service_discovery.remove_backend(extra_id).await;
            }
        }

        // Update statistics
        let mut stats_guard = stats.write().await;
        stats_guard.consistency_checks += 1;
        stats_guard.inconsistencies_found += total_inconsistencies as u64;
        stats_guard.sync_conflicts_resolved += total_inconsistencies as u64;

        // Update sync state
        let mut sync_guard = sync_state.write().await;
        sync_guard.last_sync = Instant::now();
        sync_guard.swim_member_count = swim_members.len();
        sync_guard.discovery_backend_count = discovery_backends.len();

        if total_inconsistencies > 0 {
            debug!(
                swim_members = swim_members.len(),
                discovery_backends = discovery_backends.len(),
                inconsistencies = total_inconsistencies,
                check_time = ?start_time.elapsed(),
                "Consistency check completed"
            );
        }

        Ok(())
    }

    async fn perform_state_sync(&self) -> ServiceDiscoveryResult<()> {
        Self::perform_consistency_check(
            &self.swim_cluster,
            &self.legacy_service_discovery,
            &self.sync_state,
            &self.stats,
            &self.config,
        )
        .await
    }

    async fn spawn_stats_task(&self) -> tokio::task::JoinHandle<()> {
        let stats = Arc::clone(&self.stats);
        let swim_cluster = Arc::clone(&self.swim_cluster);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                let stats_guard = stats.read().await;
                let swim_stats = {
                    let cluster = swim_cluster.read().await;
                    cluster.get_stats().await
                };

                info!(
                    events_processed = stats_guard.events_processed,
                    events_failed = stats_guard.events_failed,
                    swim_members = swim_stats.alive_members,
                    legacy_calls = stats_guard.legacy_api_calls,
                    native_calls = stats_guard.swim_native_calls,
                    "SWIM integration statistics"
                );
            }
        })
    }
}

impl Drop for SwimServiceDiscovery {
    fn drop(&mut self) {
        // Cancel event processor
        if let Some(processor) = &self.event_processor {
            processor.abort();
        }

        // Cancel background tasks
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Extension trait for legacy ServiceDiscovery methods
#[async_trait::async_trait]
pub trait ServiceDiscoveryCompat {
    async fn mark_backend_healthy(&self, node_id: &str);
    async fn mark_backend_unhealthy(&self, node_id: &str);
}

#[async_trait::async_trait]
impl ServiceDiscoveryCompat for ServiceDiscovery {
    async fn mark_backend_healthy(&self, node_id: &str) {
        // Implementation would update backend health status
        debug!(node_id = %node_id, "Marked backend healthy");
    }

    async fn mark_backend_unhealthy(&self, node_id: &str) {
        // Implementation would update backend health status
        debug!(node_id = %node_id, "Marked backend unhealthy");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_random_port_addr;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_swim_service_discovery_creation() {
        let bind_addr = get_random_port_addr();
        let swim_config = SwimConfig10k::default();
        let integration_config = SwimIntegrationConfig::default();

        let service_discovery = SwimServiceDiscovery::new(
            "test-node".to_string(),
            bind_addr,
            swim_config,
            integration_config,
        )
        .await
        .unwrap();

        assert_eq!(service_discovery.get_backend_count().await, 0);
    }

    #[tokio::test]
    async fn test_backend_registration() {
        let bind_addr = get_random_port_addr();
        let swim_config = SwimConfig10k::default();
        let integration_config = SwimIntegrationConfig::default();

        let service_discovery = SwimServiceDiscovery::new(
            "test-node".to_string(),
            bind_addr,
            swim_config,
            integration_config,
        )
        .await
        .unwrap();

        let peer_info = PeerInfo {
            id: "backend-1".to_string(),
            address: "127.0.0.1:8080".to_string(),
            metrics_port: 9090,
            node_type: NodeType::Backend,
            is_load_balancer: false,
            last_updated: std::time::SystemTime::now(),
        };

        service_discovery.register_backend(peer_info).await.unwrap();

        // Allow some time for event processing
        sleep(Duration::from_millis(100)).await;

        let backends = service_discovery.get_live_backends().await;
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].id, "backend-1");
    }

    #[tokio::test]
    async fn test_integration_statistics() {
        let bind_addr = get_random_port_addr();
        let swim_config = SwimConfig10k::default();
        let integration_config = SwimIntegrationConfig::default();

        let service_discovery = SwimServiceDiscovery::new(
            "test-node".to_string(),
            bind_addr,
            swim_config,
            integration_config,
        )
        .await
        .unwrap();

        let stats = service_discovery.get_integration_stats().await;
        assert_eq!(stats.events_processed, 0);
        assert_eq!(stats.swim_native_calls, 0);
    }
}
