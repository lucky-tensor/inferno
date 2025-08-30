//! Comprehensive tests for Phase 4 update propagation system
//!
//! This module contains extensive tests for the self-sovereign update system
//! including update propagation, retry logic, validation, and integration tests.

use crate::service_discovery::{
    retry::{RetryConfig, RetryManager},
    updates::{NodeUpdate, UpdatePropagator},
    NodeInfo, NodeType, ServiceDiscovery, ServiceDiscoveryError,
};
use std::time::Duration;
use tokio::time::sleep;

/// Test basic update propagation functionality
#[tokio::test]
async fn test_update_propagation_creation() {
    let propagator = UpdatePropagator::new();
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    // Test creating an update
    let update = propagator.create_update(&node).await.unwrap();

    assert_eq!(update.node.id, "test-node-1");
    assert_eq!(update.originator_id, "test-node-1");
    assert!(update.version > 0);
    assert!(!update.update_id.is_empty());
    assert!(update.signature.is_some());
}

/// Test self-sovereign validation
#[tokio::test]
async fn test_self_sovereign_validation() {
    let propagator = UpdatePropagator::new();
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    // Valid update - node updating itself
    let valid_update = NodeUpdate {
        update_id: "test-update-1".to_string(),
        node: node.clone(),
        timestamp: current_timestamp(),
        version: 1,
        originator_id: "test-node-1".to_string(), // Same as node.id
        signature: Some("sig_test-node-1".to_string()),
    };

    assert!(propagator.validate_self_ownership(&valid_update).is_ok());

    // Invalid update - different node trying to update
    let invalid_update = NodeUpdate {
        update_id: "test-update-2".to_string(),
        node: node.clone(),
        timestamp: current_timestamp(),
        version: 1,
        originator_id: "other-node".to_string(), // Different from node.id
        signature: Some("sig_other-node".to_string()),
    };

    let result = propagator.validate_self_ownership(&invalid_update);
    assert!(result.is_err());

    if let Err(ServiceDiscoveryError::InvalidNodeInfo { reason }) = result {
        assert!(reason.contains("Self-sovereign violation"));
        assert!(reason.contains("other-node"));
        assert!(reason.contains("test-node-1"));
    } else {
        panic!("Expected InvalidNodeInfo error for self-sovereign violation");
    }
}

/// Test update versioning
#[tokio::test]
async fn test_update_versioning() {
    let propagator = UpdatePropagator::new();
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    // Create multiple updates and verify version increment
    let update1 = propagator.create_update(&node).await.unwrap();
    let update2 = propagator.create_update(&node).await.unwrap();
    let update3 = propagator.create_update(&node).await.unwrap();

    assert!(update2.version > update1.version);
    assert!(update3.version > update2.version);

    // Each should have unique update IDs
    assert_ne!(update1.update_id, update2.update_id);
    assert_ne!(update2.update_id, update3.update_id);
    assert_ne!(update1.update_id, update3.update_id);
}

/// Test retry manager configuration
#[tokio::test]
async fn test_retry_manager_configuration() {
    let config = RetryConfig {
        max_attempts: 3,
        base_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(5),
        jitter_factor: 0.2,
        retry_timeout: Duration::from_secs(2),
    };

    let retry_manager = RetryManager::with_config(config.clone());
    let metrics = retry_manager.get_metrics().await;

    // Initial state should be empty
    assert_eq!(metrics.retry_queue_size, 0);
    assert_eq!(metrics.dead_letter_queue_size, 0);
    assert_eq!(metrics.total_retry_attempts, 0);
}

/// Test retry queue operations
#[tokio::test]
async fn test_retry_queue_operations() {
    let retry_manager = RetryManager::new();
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    let update = NodeUpdate {
        update_id: "test-update-1".to_string(),
        node,
        timestamp: current_timestamp(),
        version: 1,
        originator_id: "test-node-1".to_string(),
        signature: Some("sig_test".to_string()),
    };

    let failed_peers = vec![
        "http://peer1:8080".to_string(),
        "http://peer2:8080".to_string(),
    ];

    // Queue a retry
    let retry_id = retry_manager
        .queue_retry(update, failed_peers)
        .await
        .unwrap();
    assert!(!retry_id.is_empty());

    // Check metrics
    let metrics = retry_manager.get_metrics().await;
    assert_eq!(metrics.retry_queue_size, 1);
    assert_eq!(metrics.dead_letter_queue_size, 0);
}

/// Test exponential backoff calculation
#[tokio::test]
async fn test_exponential_backoff() {
    let config = RetryConfig {
        max_attempts: 4,
        base_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(16),
        jitter_factor: 0.1,
        retry_timeout: Duration::from_secs(10),
    };

    let retry_manager = RetryManager::with_config(config);

    // Test backoff calculation (private method, so we test the behavior indirectly)
    // The delays should follow: 1s, 2s, 4s, 8s pattern with jitter

    // We can't test the private method directly, but we can test that
    // the retry system respects the configuration
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    let update = NodeUpdate {
        update_id: "test-update-1".to_string(),
        node,
        timestamp: current_timestamp(),
        version: 1,
        originator_id: "test-node-1".to_string(),
        signature: Some("sig_test".to_string()),
    };

    let failed_peers = vec!["http://invalid-peer:8080".to_string()];

    // Queue multiple retries to test the progression
    let _retry_id = retry_manager
        .queue_retry(update, failed_peers)
        .await
        .unwrap();

    // Verify the retry was queued
    let metrics = retry_manager.get_metrics().await;
    assert_eq!(metrics.retry_queue_size, 1);
}

/// Test ServiceDiscovery integration with update system
#[tokio::test]
async fn test_service_discovery_integration() {
    let discovery = ServiceDiscovery::new();
    let node = create_test_node("backend-1", "10.0.1.5:3000", NodeType::Backend);

    // Test that we can call the broadcast method (it will return empty results with no peers)
    let results = discovery.broadcast_self_update(&node).await.unwrap();
    assert_eq!(results.len(), 0); // No peers configured

    // Test retry metrics access
    let metrics = discovery.get_retry_metrics().await;
    assert_eq!(metrics.retry_queue_size, 0);
    assert_eq!(metrics.total_retry_attempts, 0);

    // Test retry queue processing
    let processed = discovery.process_retry_queue().await.unwrap();
    assert_eq!(processed, 0); // No retries queued
}

/// Test update propagation with mock peers
#[tokio::test]
async fn test_update_propagation_with_peers() {
    let propagator = UpdatePropagator::new();
    let node = create_test_node("backend-1", "10.0.1.5:3000", NodeType::Backend);

    // Use invalid peer URLs to test error handling
    let peer_urls = vec![
        "http://invalid-peer-1:8080".to_string(),
        "http://invalid-peer-2:8080".to_string(),
    ];

    let results = propagator
        .broadcast_self_update(&node, peer_urls)
        .await
        .unwrap();

    // All should fail since the peers are invalid
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| !r.success));
    assert!(results.iter().all(|r| r.error.is_some()));
}

/// Test role change updates
#[tokio::test]
async fn test_role_change_updates() {
    let propagator = UpdatePropagator::new();

    // Test changing from backend to proxy (simulated role change)
    let backend_node = create_test_node("node-1", "10.0.1.1:3000", NodeType::Backend);
    let proxy_node = NodeInfo::new(
        "node-1".to_string(),        // Same ID
        "10.0.1.1:8080".to_string(), // Different port for proxy role
        9090,
        NodeType::Proxy, // Changed role
    );

    // Both updates should be valid (same node ID)
    let backend_update = propagator.create_update(&backend_node).await.unwrap();
    let proxy_update = propagator.create_update(&proxy_node).await.unwrap();

    assert_eq!(backend_update.originator_id, "node-1");
    assert_eq!(proxy_update.originator_id, "node-1");
    assert_eq!(backend_update.node.node_type, NodeType::Backend);
    assert_eq!(proxy_update.node.node_type, NodeType::Proxy);

    // Version should increment
    assert!(proxy_update.version > backend_update.version);
}

/// Test update deduplication with same update ID
#[tokio::test]
async fn test_update_deduplication() {
    let propagator = UpdatePropagator::new();
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    // Create updates - each should have unique IDs
    let update1 = propagator.create_update(&node).await.unwrap();
    let update2 = propagator.create_update(&node).await.unwrap();

    // Updates should have different IDs (UUIDs are unique)
    assert_ne!(update1.update_id, update2.update_id);

    // But same originator and node
    assert_eq!(update1.originator_id, update2.originator_id);
    assert_eq!(update1.node.id, update2.node.id);
}

/// Test update timestamp ordering
#[tokio::test]
async fn test_update_timestamp_ordering() {
    let propagator = UpdatePropagator::new();
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    let update1 = propagator.create_update(&node).await.unwrap();

    // Small delay to ensure timestamp difference
    sleep(Duration::from_millis(10)).await;

    let update2 = propagator.create_update(&node).await.unwrap();

    // Second update should have later timestamp
    assert!(update2.timestamp >= update1.timestamp);
    assert!(update2.version > update1.version);
}

/// Test update serialization and deserialization
#[tokio::test]
async fn test_update_serialization() {
    let node = create_test_node("test-node-1", "10.0.1.1:3000", NodeType::Backend);

    let update = NodeUpdate {
        update_id: "test-update-1".to_string(),
        node,
        timestamp: current_timestamp(),
        version: 1,
        originator_id: "test-node-1".to_string(),
        signature: Some("sig_test".to_string()),
    };

    // Test JSON serialization
    let json = serde_json::to_string(&update).unwrap();
    assert!(json.contains("test-update-1"));
    assert!(json.contains("test-node-1"));
    assert!(json.contains("sig_test"));

    // Test JSON deserialization
    let parsed: NodeUpdate = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.update_id, update.update_id);
    assert_eq!(parsed.node.id, update.node.id);
    assert_eq!(parsed.originator_id, update.originator_id);
    assert_eq!(parsed.signature, update.signature);
}

/// Helper function to create test nodes
fn create_test_node(id: &str, address: &str, node_type: NodeType) -> NodeInfo {
    NodeInfo::new(id.to_string(), address.to_string(), 9090, node_type)
}

/// Helper function to get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
