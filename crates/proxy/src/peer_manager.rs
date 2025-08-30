//! # Peer Management Module
//!
//! This module handles backend peer selection, scoring, and load balancing
//! for the Inferno Proxy. It integrates with service discovery to get the
//! list of available backends and implements various load balancing algorithms.
//!
//! ## Design Principles
//!
//! - **Algorithm Pluggability**: Support for multiple load balancing strategies
//! - **Performance-Based Scoring**: Use backend vitals for intelligent routing
//! - **Failover Support**: Graceful handling when backends become unavailable
//! - **Metrics Integration**: Track selection performance and backend health

use inferno_shared::service_discovery::{NodeVitals, ServiceDiscovery};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, instrument, warn};

/// Peer selection algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    Weighted,
}

impl From<&str> for LoadBalancingAlgorithm {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "round_robin" => Self::RoundRobin,
            "least_connections" => Self::LeastConnections,
            "weighted" => Self::Weighted,
            _ => Self::RoundRobin,
        }
    }
}

/// Backend peer with performance metrics
#[derive(Debug, Clone)]
pub struct BackendPeer {
    pub id: String,
    pub address: String,
    pub is_healthy: bool,
    pub vitals: Option<NodeVitals>,
    pub performance_score: Option<f64>,
}

impl BackendPeer {
    /// Creates a new backend peer
    pub fn new(id: String, address: String, is_healthy: bool, vitals: Option<NodeVitals>) -> Self {
        let performance_score = vitals.as_ref().map(Self::calculate_score);

        Self {
            id,
            address,
            is_healthy,
            vitals,
            performance_score,
        }
    }

    /// Calculates performance score for this backend (lower is better)
    fn calculate_score(vitals: &NodeVitals) -> f64 {
        // Performance scoring algorithm:
        // - More weight on active requests (directly affects latency)
        // - Medium weight on resource usage (affects future capacity)
        // - Low weight on failed responses (historical indicator)

        let requests_weight = 2.0;
        let resource_weight = 1.0;
        let failure_weight = 0.1;

        (vitals.active_requests.unwrap_or(0) as f64 * requests_weight)
            + ((vitals.cpu_usage.unwrap_or(0.0) / 100.0) * resource_weight)
            + ((vitals.memory_usage.unwrap_or(0.0) / 100.0) * resource_weight)
            + ((vitals.error_rate.unwrap_or(0.0) / 100.0) * failure_weight)
    }

    /// Returns true if this peer is available for traffic
    pub fn is_available(&self) -> bool {
        self.is_healthy && self.vitals.as_ref().is_some_and(|v| v.ready)
    }

    /// Gets the current load on this backend
    pub fn current_load(&self) -> u32 {
        self.vitals
            .as_ref()
            .map_or(0, |v| v.active_requests.unwrap_or(0) as u32)
    }
}

/// Manages backend peer selection and load balancing
pub struct PeerManager {
    service_discovery: Arc<ServiceDiscovery>,
    algorithm: LoadBalancingAlgorithm,
    round_robin_counter: AtomicUsize,
}

impl PeerManager {
    /// Creates a new peer manager
    pub fn new(
        service_discovery: Arc<ServiceDiscovery>,
        algorithm: LoadBalancingAlgorithm,
    ) -> Self {
        Self {
            service_discovery,
            algorithm,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Selects the best backend peer for the next request
    #[instrument(skip(self))]
    pub async fn select_peer(&self) -> Option<String> {
        let peers = self.get_available_peers().await;

        if peers.is_empty() {
            warn!("No available peers for load balancing");
            return None;
        }

        debug!(
            peer_count = peers.len(),
            algorithm = ?self.algorithm,
            "Selecting peer for request"
        );

        let selected = match self.algorithm {
            LoadBalancingAlgorithm::RoundRobin => self.select_round_robin(&peers),
            LoadBalancingAlgorithm::LeastConnections => self.select_least_connections(&peers),
            LoadBalancingAlgorithm::Weighted => self.select_weighted(&peers),
        };

        if let Some(ref address) = selected {
            debug!(
                selected_peer = %address,
                algorithm = ?self.algorithm,
                "Peer selected successfully"
            );
        }

        selected
    }

    /// Gets all available (healthy and ready) backend peers
    #[instrument(skip(self))]
    async fn get_available_peers(&self) -> Vec<BackendPeer> {
        let all_backends = self.service_discovery.get_all_backends().await;

        all_backends
            .into_iter()
            .map(|(id, address, is_healthy, vitals)| {
                BackendPeer::new(id, address, is_healthy, vitals)
            })
            .filter(|peer| peer.is_available())
            .collect()
    }

    /// Implements round-robin peer selection
    #[instrument(skip(self, peers))]
    fn select_round_robin(&self, peers: &[BackendPeer]) -> Option<String> {
        if peers.is_empty() {
            return None;
        }

        let index = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % peers.len();
        Some(peers[index].address.clone())
    }

    /// Implements least-connections peer selection
    #[instrument(skip(self, peers))]
    fn select_least_connections(&self, peers: &[BackendPeer]) -> Option<String> {
        if peers.is_empty() {
            return None;
        }

        let best_peer = peers.iter().min_by_key(|peer| peer.current_load());

        best_peer.map(|peer| {
            debug!(
                selected_peer = %peer.address,
                current_load = peer.current_load(),
                "Selected peer with least connections"
            );
            peer.address.clone()
        })
    }

    /// Implements weighted peer selection based on performance scores
    #[instrument(skip(self, peers))]
    fn select_weighted(&self, peers: &[BackendPeer]) -> Option<String> {
        if peers.is_empty() {
            return None;
        }

        // Find the peer with the best (lowest) performance score
        let best_peer = peers
            .iter()
            .filter(|peer| peer.performance_score.is_some())
            .min_by(|a, b| {
                a.performance_score
                    .partial_cmp(&b.performance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        best_peer.map(|peer| {
            debug!(
                selected_peer = %peer.address,
                performance_score = ?peer.performance_score,
                requests_in_progress = peer.vitals.as_ref().and_then(|v| v.active_requests).map(|r| r as u32),
                cpu_usage = peer.vitals.as_ref().map(|v| v.cpu_usage),
                "Selected peer with best performance score"
            );
            peer.address.clone()
        })
    }

    /// Gets statistics about peer selection and load balancing
    pub async fn get_peer_stats(&self) -> PeerManagerStats {
        let peers = self.get_available_peers().await;
        let total_peers = self.service_discovery.backend_count().await;

        let avg_score = if !peers.is_empty() {
            peers
                .iter()
                .filter_map(|p| p.performance_score)
                .sum::<f64>()
                / peers.len() as f64
        } else {
            0.0
        };

        let total_load = peers.iter().map(|p| p.current_load()).sum::<u32>();

        PeerManagerStats {
            total_peers,
            available_peers: peers.len(),
            algorithm: self.algorithm.clone(),
            average_performance_score: avg_score,
            total_load,
            round_robin_position: self.round_robin_counter.load(Ordering::Relaxed),
        }
    }
}

/// Statistics about peer manager performance
#[derive(Debug)]
pub struct PeerManagerStats {
    pub total_peers: usize,
    pub available_peers: usize,
    pub algorithm: LoadBalancingAlgorithm,
    pub average_performance_score: f64,
    pub total_load: u32,
    pub round_robin_position: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_shared::service_discovery::NodeVitals;

    fn create_test_vitals(requests: u32, cpu: f64, memory: f64, error_rate: f64) -> NodeVitals {
        NodeVitals {
            ready: true,
            cpu_usage: Some(cpu),
            memory_usage: Some(memory),
            active_requests: Some(requests as u64),
            avg_response_time_ms: Some(100.0),
            error_rate: Some(error_rate),
            status_message: Some("healthy".to_string()),
        }
    }

    #[test]
    fn test_backend_peer_score_calculation() {
        let vitals = create_test_vitals(5, 50.0, 60.0, 2.0);
        let score = BackendPeer::calculate_score(&vitals);

        // Expected: (5 * 2.0) + (0.5 * 1.0) + (0.6 * 1.0) + (0.02 * 0.1)
        // = 10.0 + 0.5 + 0.6 + 0.002 = 11.102
        assert!((score - 11.102).abs() < 0.01);
    }

    #[test]
    fn test_backend_peer_availability() {
        // Healthy and ready
        let peer1 = BackendPeer::new(
            "test1".to_string(),
            "127.0.0.1:3000".to_string(),
            true,
            Some(create_test_vitals(5, 50.0, 60.0, 2.0)),
        );
        assert!(peer1.is_available());

        // Healthy but not ready
        let mut vitals = create_test_vitals(5, 50.0, 60.0, 2.0);
        vitals.ready = false;
        let peer2 = BackendPeer::new(
            "test2".to_string(),
            "127.0.0.1:3001".to_string(),
            true,
            Some(vitals),
        );
        assert!(!peer2.is_available());

        // Not healthy
        let peer3 = BackendPeer::new(
            "test3".to_string(),
            "127.0.0.1:3002".to_string(),
            false,
            Some(create_test_vitals(5, 50.0, 60.0, 2.0)),
        );
        assert!(!peer3.is_available());
    }

    #[test]
    fn test_load_balancing_algorithm_from_string() {
        assert_eq!(
            LoadBalancingAlgorithm::from("round_robin"),
            LoadBalancingAlgorithm::RoundRobin
        );
        assert_eq!(
            LoadBalancingAlgorithm::from("least_connections"),
            LoadBalancingAlgorithm::LeastConnections
        );
        assert_eq!(
            LoadBalancingAlgorithm::from("weighted"),
            LoadBalancingAlgorithm::Weighted
        );
        assert_eq!(
            LoadBalancingAlgorithm::from("unknown"),
            LoadBalancingAlgorithm::RoundRobin
        );
    }
}
