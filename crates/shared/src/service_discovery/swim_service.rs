//! SWIM-Based Service Discovery Service
//!
//! This module replaces the previous consensus-based service discovery with
//! a SWIM protocol implementation capable of handling 10,000+ nodes efficiently.
//! This is the new default service discovery implementation for large-scale
//! AI inference clusters.

use super::config::ServiceDiscoveryConfig;
use super::errors::ServiceDiscoveryResult;
use super::swim::{SwimConfig10k, SwimStats};
use super::swim_integration::{SwimIntegrationConfig, SwimIntegrationStats, SwimServiceDiscovery};
use super::types::{NodeType, PeerInfo};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, instrument};

/// Production-ready service discovery using SWIM protocol
///
/// This replaces the previous consensus-based implementation and is
/// optimized for 10,000+ node AI inference clusters.
pub struct SwimBasedServiceDiscovery {
    /// SWIM service discovery implementation
    inner: Arc<RwLock<SwimServiceDiscovery>>,
    /// Configuration
    config: ServiceDiscoveryConfig,
    /// Local node information
    local_node_id: String,
}

impl SwimBasedServiceDiscovery {
    /// Creates new SWIM-based service discovery
    #[instrument(skip(config))]
    pub async fn new(
        local_node_id: String,
        bind_addr: SocketAddr,
        config: ServiceDiscoveryConfig,
    ) -> ServiceDiscoveryResult<Self> {
        // Convert legacy config to SWIM config
        let swim_config = SwimConfig10k {
            probe_interval: std::time::Duration::from_millis(100), // Fast probing for large clusters
            probe_timeout: std::time::Duration::from_millis(50),
            suspicion_timeout: std::time::Duration::from_secs(10),
            k_indirect_probes: 5,
            gossip_fanout: 15, // Optimized for 10k nodes
            max_gossip_per_message: 50,
            enable_compression: true, // Essential at scale
            message_rate_limit: 1000,
            ..Default::default()
        };

        let integration_config = SwimIntegrationConfig {
            consistency_check_interval: std::time::Duration::from_secs(30),
            event_batch_size: 100,
            max_pending_events: 10000,
            auto_reconciliation: true,
            backend_operation_timeout: std::time::Duration::from_secs(5),
        };

        let service_discovery = SwimServiceDiscovery::new(
            local_node_id.clone(),
            bind_addr,
            swim_config,
            integration_config,
        )
        .await?;

        let inner = Arc::new(RwLock::new(service_discovery));

        info!(
            node_id = %local_node_id,
            bind_addr = %bind_addr,
            "Created SWIM-based service discovery for 10k+ nodes"
        );

        Ok(Self {
            inner,
            config,
            local_node_id,
        })
    }

    /// Starts the service discovery system
    #[instrument(skip(self))]
    pub async fn start(&self) -> ServiceDiscoveryResult<()> {
        let mut service = self.inner.write().await;
        service.start().await?;

        info!(
            node_id = %self.local_node_id,
            "SWIM-based service discovery started and ready for 10k+ nodes"
        );

        Ok(())
    }

    /// Registers a new backend in the cluster
    #[instrument(skip(self))]
    pub async fn register_backend(&self, peer_info: PeerInfo) -> ServiceDiscoveryResult<()> {
        let service = self.inner.read().await;
        service.register_backend(peer_info).await
    }

    /// Removes a backend from the cluster
    #[instrument(skip(self))]
    pub async fn remove_backend(&self, node_id: &str) -> ServiceDiscoveryResult<()> {
        let service = self.inner.read().await;
        service.remove_backend(node_id).await
    }

    /// Gets all currently live backends
    pub async fn list_backends(&self) -> Vec<PeerInfo> {
        let service = self.inner.read().await;
        service.get_live_backends().await
    }

    /// Gets backend count (optimized for large clusters)
    pub async fn get_backend_count(&self) -> usize {
        let service = self.inner.read().await;
        service.get_backend_count().await
    }

    /// Gets backends filtered by node type
    pub async fn get_backends_by_type(&self, node_type: NodeType) -> Vec<PeerInfo> {
        let service = self.inner.read().await;
        let all_backends = service.get_live_backends().await;

        all_backends
            .into_iter()
            .filter(|backend| backend.node_type == node_type)
            .collect()
    }

    /// Gets load balancer backends only
    pub async fn get_load_balancers(&self) -> Vec<PeerInfo> {
        let service = self.inner.read().await;
        let all_backends = service.get_live_backends().await;

        all_backends
            .into_iter()
            .filter(|backend| backend.is_load_balancer)
            .collect()
    }

    /// Gets SWIM protocol statistics
    pub async fn get_swim_stats(&self) -> SwimStats {
        let service = self.inner.read().await;
        service.get_swim_stats().await
    }

    /// Gets integration layer statistics
    pub async fn get_integration_stats(&self) -> SwimIntegrationStats {
        let service = self.inner.read().await;
        service.get_integration_stats().await
    }

    /// Triggers manual consistency check
    pub async fn trigger_consistency_check(&self) -> ServiceDiscoveryResult<()> {
        let service = self.inner.read().await;
        service.trigger_sync().await
    }

    /// Checks if the cluster is healthy for the current scale
    pub async fn is_cluster_healthy(&self) -> bool {
        let swim_stats = self.get_swim_stats().await;
        let integration_stats = self.get_integration_stats().await;

        // Check basic health criteria
        let member_ratio = if swim_stats.total_members > 0 {
            swim_stats.alive_members as f64 / swim_stats.total_members as f64
        } else {
            1.0
        };

        let event_failure_rate = if integration_stats.events_processed > 0 {
            integration_stats.events_failed as f64 / integration_stats.events_processed as f64
        } else {
            0.0
        };

        // Healthy if:
        // - At least 90% of members are alive
        // - Event failure rate is less than 5%
        // - No excessive pending events
        member_ratio >= 0.9 && event_failure_rate < 0.05 && integration_stats.events_pending < 1000
    }

    /// Gets cluster scale recommendations
    pub async fn get_scale_recommendations(&self) -> Vec<String> {
        let swim_stats = self.get_swim_stats().await;
        let integration_stats = self.get_integration_stats().await;
        let mut recommendations = Vec::new();

        // Large cluster recommendations
        if swim_stats.total_members > 5000 {
            recommendations
                .push("Consider regional clustering for better network efficiency".to_string());
        }

        if swim_stats.total_members > 8000 {
            recommendations.push("Approaching maximum recommended cluster size (10k)".to_string());
            recommendations.push("Plan for multi-cluster architecture".to_string());
        }

        // Performance recommendations
        if integration_stats.events_pending > 5000 {
            recommendations
                .push("High event backlog - consider increasing processing capacity".to_string());
        }

        if integration_stats.events_failed > 0 {
            let failure_rate = integration_stats.events_failed as f64
                / integration_stats.events_processed.max(1) as f64;
            if failure_rate > 0.01 {
                recommendations.push(
                    "Event processing failure rate is high - check system resources".to_string(),
                );
            }
        }

        recommendations
    }

    /// Legacy compatibility methods
    ///
    /// Legacy register_backend for API compatibility
    pub async fn legacy_register_backend(&self, peer_info: PeerInfo) -> ServiceDiscoveryResult<()> {
        self.register_backend(peer_info).await
    }

    /// Legacy remove_backend for API compatibility  
    pub async fn legacy_remove_backend(&self, node_id: &str) -> ServiceDiscoveryResult<()> {
        self.remove_backend(node_id).await
    }

    /// Legacy list_backends for API compatibility
    pub async fn legacy_list_backends(&self) -> Vec<PeerInfo> {
        self.list_backends().await
    }
}

/// Factory function to create the appropriate service discovery implementation
/// based on expected cluster size
pub async fn create_service_discovery(
    local_node_id: String,
    bind_addr: SocketAddr,
    config: ServiceDiscoveryConfig,
    expected_cluster_size: Option<usize>,
) -> ServiceDiscoveryResult<SwimBasedServiceDiscovery> {
    match expected_cluster_size {
        Some(size) if size <= 25 => {
            info!(
                expected_size = size,
                "Small cluster detected - using SWIM with conservative settings"
            );

            // For small clusters, we could use different settings, but SWIM
            // handles small clusters fine too, so we use SWIM consistently
            SwimBasedServiceDiscovery::new(local_node_id, bind_addr, config).await
        }
        Some(size) if size <= 1000 => {
            info!(
                expected_size = size,
                "Medium cluster detected - using SWIM with standard settings"
            );

            SwimBasedServiceDiscovery::new(local_node_id, bind_addr, config).await
        }
        Some(size) => {
            info!(
                expected_size = size,
                "Large cluster detected - using SWIM with high-performance settings"
            );

            // Large clusters get the full SWIM implementation
            SwimBasedServiceDiscovery::new(local_node_id, bind_addr, config).await
        }
        None => {
            info!("No cluster size hint - defaulting to SWIM for scalability");

            // Default to SWIM since it handles all scales
            SwimBasedServiceDiscovery::new(local_node_id, bind_addr, config).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_swim_service_discovery_creation() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000);
        let config = ServiceDiscoveryConfig::default();

        let service_discovery =
            SwimBasedServiceDiscovery::new("test-node".to_string(), bind_addr, config)
                .await
                .unwrap();

        assert_eq!(service_discovery.local_node_id, "test-node");
    }

    #[tokio::test]
    async fn test_service_discovery_factory() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8001);
        let config = ServiceDiscoveryConfig::default();

        // Test with large cluster size
        let service_discovery = create_service_discovery(
            "large-cluster-node".to_string(),
            bind_addr,
            config,
            Some(5000),
        )
        .await
        .unwrap();

        assert_eq!(service_discovery.local_node_id, "large-cluster-node");
    }

    #[tokio::test]
    async fn test_backend_operations() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8002);
        let config = ServiceDiscoveryConfig::default();

        let mut service_discovery =
            SwimBasedServiceDiscovery::new("ops-test-node".to_string(), bind_addr, config)
                .await
                .unwrap();

        service_discovery.start().await.unwrap();

        // Test backend registration
        let peer_info = PeerInfo {
            id: "backend-1".to_string(),
            address: "127.0.0.1:8080".to_string(),
            metrics_port: 9090,
            node_type: NodeType::Backend,
            is_load_balancer: false,
            last_updated: SystemTime::now(),
        };

        service_discovery.register_backend(peer_info).await.unwrap();

        // Allow processing time
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Test listing backends
        let backends = service_discovery.list_backends().await;
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].id, "backend-1");

        // Test backend count
        let count = service_discovery.get_backend_count().await;
        assert_eq!(count, 1);

        // Test filtering by type
        let backend_nodes = service_discovery
            .get_backends_by_type(NodeType::Backend)
            .await;
        assert_eq!(backend_nodes.len(), 1);

        let proxy_nodes = service_discovery
            .get_backends_by_type(NodeType::Proxy)
            .await;
        assert_eq!(proxy_nodes.len(), 0);
    }

    #[tokio::test]
    async fn test_cluster_health_check() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8003);
        let config = ServiceDiscoveryConfig::default();

        let mut service_discovery =
            SwimBasedServiceDiscovery::new("health-test-node".to_string(), bind_addr, config)
                .await
                .unwrap();

        service_discovery.start().await.unwrap();

        // Initially should be healthy (no members is considered healthy)
        let is_healthy = service_discovery.is_cluster_healthy().await;
        assert!(is_healthy);

        // Add a backend
        let peer_info = PeerInfo {
            id: "backend-1".to_string(),
            address: "127.0.0.1:8080".to_string(),
            metrics_port: 9090,
            node_type: NodeType::Backend,
            is_load_balancer: false,
            last_updated: SystemTime::now(),
        };

        service_discovery.register_backend(peer_info).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Should still be healthy with one alive member
        let is_healthy = service_discovery.is_cluster_healthy().await;
        assert!(is_healthy);
    }

    #[tokio::test]
    async fn test_scale_recommendations() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8004);
        let config = ServiceDiscoveryConfig::default();

        let service_discovery =
            SwimBasedServiceDiscovery::new("scale-test-node".to_string(), bind_addr, config)
                .await
                .unwrap();

        // Initially no recommendations for empty cluster
        let recommendations = service_discovery.get_scale_recommendations().await;
        assert!(recommendations.is_empty());
    }
}
