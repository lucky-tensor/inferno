//! Comprehensive tests for consensus algorithms including edge cases
//!
//! This module contains unit tests and integration tests for the consensus
//! resolution algorithms, covering majority rule logic, timestamp tie-breaking,
//! and various edge cases for distributed service discovery.

use crate::service_discovery::consensus::ConsensusResolver;
use crate::service_discovery::types::{NodeType, PeerInfo};
use std::time::{Duration, SystemTime};

/// Helper function to create a test PeerInfo with specified parameters
fn create_peer_info(
    id: &str,
    address: &str,
    metrics_port: u16,
    node_type: NodeType,
    is_load_balancer: bool,
    timestamp_offset_secs: i64,
) -> PeerInfo {
    let timestamp = if timestamp_offset_secs >= 0 {
        SystemTime::now() + Duration::from_secs(timestamp_offset_secs as u64)
    } else {
        SystemTime::now() - Duration::from_secs((-timestamp_offset_secs) as u64)
    };

    PeerInfo {
        id: id.to_string(),
        address: address.to_string(),
        metrics_port,
        node_type,
        is_load_balancer,
        last_updated: timestamp,
    }
}

#[tokio::test]
async fn test_consensus_single_peer_response() {
    let resolver = ConsensusResolver::new();

    let peer_responses = vec![vec![create_peer_info(
        "backend-1",
        "10.0.1.5:3000",
        9090,
        NodeType::Backend,
        false,
        0,
    )]];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with single peer");

    assert_eq!(consensus.len(), 1);
    assert_eq!(consensus[0].id, "backend-1");
    assert_eq!(metrics.conflicts_detected, 0);
    assert_eq!(metrics.tie_breaks, 0);
    assert_eq!(metrics.peer_count, 1);
    assert_eq!(metrics.majority_nodes, 1);
}

#[tokio::test]
async fn test_consensus_no_conflicts() {
    let resolver = ConsensusResolver::new();

    let peer1 = create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, 0);
    let peer2 = create_peer_info("proxy-1", "10.0.1.1:8080", 6100, NodeType::Proxy, true, 0);

    let peer_responses = vec![
        vec![peer1.clone(), peer2.clone()],
        vec![peer1.clone(), peer2.clone()],
        vec![peer1.clone(), peer2.clone()],
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with no conflicts");

    assert_eq!(consensus.len(), 2);
    assert_eq!(metrics.conflicts_detected, 0);
    assert_eq!(metrics.tie_breaks, 0);
    assert_eq!(metrics.peer_count, 3);
    assert_eq!(metrics.majority_nodes, 2);
}

#[tokio::test]
async fn test_consensus_majority_rule() {
    let resolver = ConsensusResolver::new();

    // Create different versions of the same peer
    let peer_v1 = create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, -10);
    let peer_v2 = create_peer_info("backend-1", "10.0.1.6:3000", 9090, NodeType::Backend, false, 0);

    let peer_responses = vec![
        vec![peer_v1.clone()], // Peer 1 sees old version
        vec![peer_v2.clone()], // Peer 2 sees new version  
        vec![peer_v2.clone()], // Peer 3 sees new version (majority)
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with majority rule");

    assert_eq!(consensus.len(), 1);
    assert_eq!(consensus[0].address, "10.0.1.6:3000"); // Majority version wins
    assert_eq!(metrics.conflicts_detected, 1);
    assert_eq!(metrics.tie_breaks, 0);
    assert_eq!(metrics.peer_count, 3);
    assert_eq!(metrics.majority_nodes, 1);
}

#[tokio::test]
async fn test_consensus_timestamp_tie_breaking() {
    let resolver = ConsensusResolver::new();

    // Create two versions with equal votes but different timestamps
    let peer_v1 = create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, -10);
    let peer_v2 = create_peer_info("backend-1", "10.0.1.6:3000", 9090, NodeType::Backend, false, 0);

    let peer_responses = vec![
        vec![peer_v1.clone()], // Peer 1 sees old version
        vec![peer_v2.clone()], // Peer 2 sees new version
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with timestamp tie-breaking");

    assert_eq!(consensus.len(), 1);
    assert_eq!(consensus[0].address, "10.0.1.6:3000"); // Newer timestamp wins
    assert_eq!(metrics.conflicts_detected, 1);
    assert_eq!(metrics.tie_breaks, 1);
    assert_eq!(metrics.peer_count, 2);
    assert_eq!(metrics.majority_nodes, 1);
}

#[tokio::test]
async fn test_consensus_multiple_nodes_with_conflicts() {
    let resolver = ConsensusResolver::new();

    let backend1_v1 = create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, -10);
    let backend1_v2 = create_peer_info("backend-1", "10.0.1.6:3000", 9090, NodeType::Backend, false, 0);
    let proxy1 = create_peer_info("proxy-1", "10.0.1.1:8080", 6100, NodeType::Proxy, true, 0);

    let peer_responses = vec![
        vec![backend1_v1.clone(), proxy1.clone()],
        vec![backend1_v2.clone(), proxy1.clone()],
        vec![backend1_v2.clone(), proxy1.clone()],
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with multiple nodes");

    assert_eq!(consensus.len(), 2);
    assert_eq!(metrics.conflicts_detected, 1); // Only backend-1 has conflict
    assert_eq!(metrics.tie_breaks, 0);
    assert_eq!(metrics.peer_count, 3);
    assert_eq!(metrics.majority_nodes, 2);

    // Find the backend-1 in consensus
    let backend_consensus = consensus
        .iter()
        .find(|p| p.id == "backend-1")
        .expect("backend-1 should be in consensus");
    assert_eq!(backend_consensus.address, "10.0.1.6:3000"); // Majority wins
}

#[tokio::test]
async fn test_consensus_insufficient_peers() {
    let resolver = ConsensusResolver::new();

    // Test with empty peer responses
    let peer_responses = vec![];

    let result = resolver.resolve_consensus(peer_responses).await;
    
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("No peer responses provided"));
    }
}

#[tokio::test]
async fn test_consensus_empty_peer_responses() {
    let resolver = ConsensusResolver::new();

    let peer_responses = vec![vec![], vec![]];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with empty responses");

    assert_eq!(consensus.len(), 0);
    assert_eq!(metrics.conflicts_detected, 0);
    assert_eq!(metrics.tie_breaks, 0);
    assert_eq!(metrics.peer_count, 2);
    assert_eq!(metrics.majority_nodes, 0);
    assert_eq!(metrics.total_nodes, 0);
}

#[tokio::test]
async fn test_consensus_complex_scenario() {
    let resolver = ConsensusResolver::new();

    // Complex scenario with multiple nodes, different conflicts
    let backend1_v1 = create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, -20);
    let backend1_v2 = create_peer_info("backend-1", "10.0.1.6:3000", 9090, NodeType::Backend, false, -10);
    let backend1_v3 = create_peer_info("backend-1", "10.0.1.7:3000", 9090, NodeType::Backend, false, 0);

    let backend2_v1 = create_peer_info("backend-2", "10.0.2.5:3000", 9091, NodeType::Backend, false, -5);
    let backend2_v2 = create_peer_info("backend-2", "10.0.2.6:3000", 9091, NodeType::Backend, false, 5);

    let proxy1 = create_peer_info("proxy-1", "10.0.1.1:8080", 6100, NodeType::Proxy, true, 0);

    let peer_responses = vec![
        vec![backend1_v1.clone(), backend2_v1.clone(), proxy1.clone()],
        vec![backend1_v2.clone(), backend2_v2.clone(), proxy1.clone()],
        vec![backend1_v3.clone(), backend2_v2.clone(), proxy1.clone()],
        vec![backend1_v3.clone(), backend2_v1.clone(), proxy1.clone()],
        vec![backend1_v3.clone(), backend2_v2.clone(), proxy1.clone()],
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed in complex scenario");

    assert_eq!(consensus.len(), 3);
    assert_eq!(metrics.conflicts_detected, 2); // backend-1 and backend-2 have conflicts
    assert_eq!(metrics.peer_count, 5);
    assert_eq!(metrics.majority_nodes, 3);

    // backend-1: v3 appears in 3 responses (majority)
    let backend1_consensus = consensus
        .iter()
        .find(|p| p.id == "backend-1")
        .expect("backend-1 should be in consensus");
    assert_eq!(backend1_consensus.address, "10.0.1.7:3000");

    // backend-2: v2 appears in 3 responses (majority)
    let backend2_consensus = consensus
        .iter()
        .find(|p| p.id == "backend-2")
        .expect("backend-2 should be in consensus");
    assert_eq!(backend2_consensus.address, "10.0.2.6:3000");

    // proxy-1: no conflicts
    let proxy_consensus = consensus
        .iter()
        .find(|p| p.id == "proxy-1")
        .expect("proxy-1 should be in consensus");
    assert_eq!(proxy_consensus.address, "10.0.1.1:8080");
}

#[tokio::test]
async fn test_consensus_performance_metrics() {
    let resolver = ConsensusResolver::new();

    let peer_responses = vec![
        vec![create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, 0)],
        vec![create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, 0)],
    ];

    let start = std::time::Instant::now();
    let (_, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed");
    let duration = start.elapsed();

    // Verify performance metrics are collected
    assert!(metrics.consensus_duration_micros > 0);
    assert!(metrics.consensus_duration_micros < duration.as_micros() as u64 * 2); // Reasonable upper bound
    assert_eq!(metrics.peer_count, 2);
    assert_eq!(metrics.total_nodes, 2);
    assert_eq!(metrics.majority_nodes, 1);
}

#[tokio::test]
async fn test_consensus_different_node_types() {
    let resolver = ConsensusResolver::new();

    let backend = create_peer_info("backend-1", "10.0.1.5:3000", 9090, NodeType::Backend, false, 0);
    let proxy = create_peer_info("proxy-1", "10.0.1.1:8080", 6100, NodeType::Proxy, true, 0);
    let governator = create_peer_info("gov-1", "10.0.1.10:7000", 9095, NodeType::Governator, false, 0);

    let peer_responses = vec![
        vec![backend.clone(), proxy.clone()],
        vec![backend.clone(), governator.clone()],
        vec![proxy.clone(), governator.clone()],
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with different node types");

    assert_eq!(consensus.len(), 3);
    assert_eq!(metrics.conflicts_detected, 0);
    assert_eq!(metrics.peer_count, 3);

    // Verify all node types are present
    let node_types: std::collections::HashSet<NodeType> = consensus
        .iter()
        .map(|p| p.node_type)
        .collect();
    assert_eq!(node_types.len(), 3);
    assert!(node_types.contains(&NodeType::Backend));
    assert!(node_types.contains(&NodeType::Proxy));
    assert!(node_types.contains(&NodeType::Governator));
}

#[tokio::test]
async fn test_consensus_identical_timestamps() {
    let resolver = ConsensusResolver::new();

    let now = SystemTime::now();
    let peer_v1 = PeerInfo {
        id: "backend-1".to_string(),
        address: "10.0.1.5:3000".to_string(),
        metrics_port: 9090,
        node_type: NodeType::Backend,
        is_load_balancer: false,
        last_updated: now,
    };

    let peer_v2 = PeerInfo {
        id: "backend-1".to_string(),
        address: "10.0.1.6:3000".to_string(), // Different address
        metrics_port: 9090,
        node_type: NodeType::Backend,
        is_load_balancer: false,
        last_updated: now, // Same timestamp
    };

    let peer_responses = vec![
        vec![peer_v1.clone()],
        vec![peer_v2.clone()],
    ];

    let (consensus, metrics) = resolver
        .resolve_consensus(peer_responses)
        .await
        .expect("Consensus should succeed with identical timestamps");

    assert_eq!(consensus.len(), 1);
    assert_eq!(metrics.conflicts_detected, 1);
    assert_eq!(metrics.tie_breaks, 1);
    
    // With identical timestamps, the algorithm should still pick one consistently
    assert!(consensus[0].address == "10.0.1.5:3000" || consensus[0].address == "10.0.1.6:3000");
}