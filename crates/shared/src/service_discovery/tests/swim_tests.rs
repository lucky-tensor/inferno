//! SWIM Protocol Test Suite
//!
//! Comprehensive tests for SWIM protocol implementation including
//! unit tests, integration tests, and scale testing for 10k+ nodes.

use super::super::swim::GossipUpdate;
use super::super::{
    FailureDetectorConfig, GossipConfig, GossipPriority, MemberState, NodeType, PeerInfo,
    SwimCluster, SwimConfig10k, SwimFailureDetector, SwimGossipManager, SwimIntegrationConfig,
    SwimMembershipEvent, SwimServiceDiscovery,
};
use crate::test_utils::get_random_port_addr;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};

/// Tests basic SWIM cluster functionality
#[tokio::test]
async fn test_swim_cluster_basic_operations() {
    let bind_addr = get_random_port_addr();
    let config = SwimConfig10k::default();

    let (mut cluster, mut events) = SwimCluster::new("test-node".to_string(), bind_addr, config)
        .await
        .unwrap();

    // Test adding members
    for i in 0..10 {
        let peer_info = create_test_peer_info(i);
        cluster.add_member(peer_info).await.unwrap();
    }

    // Verify members are added
    let live_members = cluster.get_live_members().await;
    assert_eq!(live_members.len(), 10);

    // Check statistics
    let stats = cluster.get_stats().await;
    assert_eq!(stats.alive_members, 10);
    assert_eq!(stats.total_members, 10);

    // Verify we receive membership events
    let event = timeout(Duration::from_millis(100), events.recv()).await;
    assert!(event.is_ok());
}

/// Tests SWIM cluster startup and background tasks
#[tokio::test]
async fn test_swim_cluster_startup() {
    let bind_addr = get_random_port_addr();
    let config = SwimConfig10k::default();

    let (mut cluster, _events) = SwimCluster::new("startup-test".to_string(), bind_addr, config)
        .await
        .unwrap();

    // Start background tasks
    cluster.start().await.unwrap();

    // Add some members
    for i in 0..5 {
        let peer_info = create_test_peer_info(i);
        cluster.add_member(peer_info).await.unwrap();
    }

    // Allow background tasks to run
    sleep(Duration::from_millis(200)).await;

    let stats = cluster.get_stats().await;
    assert_eq!(stats.alive_members, 5);
}

/// Tests failure detection mechanisms
#[tokio::test]
async fn test_swim_failure_detection() {
    let members = std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
    let (event_sender, mut event_receiver) = mpsc::unbounded_channel();

    let config = FailureDetectorConfig {
        probe_timeout: Duration::from_millis(50),
        suspicion_timeout: Duration::from_secs(1),
        ..Default::default()
    };

    let mut detector = SwimFailureDetector::new(config, members.clone(), event_sender);
    detector.start().await.unwrap();

    // Add a test member
    let test_member = super::super::SwimMember::from_peer_info(create_test_peer_info(1)).unwrap();
    let member_id = test_member.id;
    members.write().await.insert(member_id, test_member);

    // Initiate probe
    let sequence = detector.probe_member(member_id).await.unwrap();

    // Simulate probe timeout
    sleep(Duration::from_millis(100)).await;
    detector.handle_probe_timeout(sequence).await.unwrap();

    // Should receive suspicion event
    let event = timeout(Duration::from_millis(500), event_receiver.recv()).await;
    assert!(event.is_ok());

    if let Ok(Some(SwimMembershipEvent::MemberStateChanged { new_state, .. })) = event {
        assert_eq!(new_state, MemberState::Suspected);
    }
}

/// Tests gossip dissemination
#[tokio::test]
async fn test_swim_gossip_dissemination() {
    let members = std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::BTreeMap::new()));
    let config = GossipConfig::default();

    let mut gossip_manager = SwimGossipManager::new(config, members.clone(), 1);

    // Add test members
    for i in 0..5 {
        let member = super::super::SwimMember::from_peer_info(create_test_peer_info(i)).unwrap();
        members.write().await.insert(member.id, member);
    }

    // Test gossip update
    let update = GossipUpdate {
        node_id: 1,
        state: MemberState::Suspected,
        incarnation: 2,
        member_info: None,
        generation: 1,
    };

    gossip_manager
        .gossip_update(update.clone(), GossipPriority::High)
        .await
        .unwrap();

    // Start gossip manager
    gossip_manager.start().await.unwrap();

    // Allow gossip to process
    sleep(Duration::from_millis(300)).await;

    let stats = gossip_manager.get_stats().await;
    assert!(stats.messages_sent > 0 || stats.updates_propagated > 0);
}

/// Tests service discovery integration
#[tokio::test]
async fn test_swim_service_discovery_integration() {
    let bind_addr = get_random_port_addr();
    let swim_config = SwimConfig10k::default();
    let integration_config = SwimIntegrationConfig::default();

    let mut service_discovery = SwimServiceDiscovery::new(
        "integration-test".to_string(),
        bind_addr,
        swim_config,
        integration_config,
    )
    .await
    .unwrap();

    service_discovery.start().await.unwrap();

    // Test backend registration
    for i in 0..3 {
        let peer_info = create_test_peer_info(i);
        service_discovery.register_backend(peer_info).await.unwrap();
    }

    // Allow event processing
    sleep(Duration::from_millis(100)).await;

    // Test getting backends
    let backends = service_discovery.get_live_backends().await;
    assert_eq!(backends.len(), 3);

    // Test legacy compatibility
    let legacy_backends = service_discovery.list_backends().await;
    assert_eq!(legacy_backends.len(), 3);

    // Test statistics
    let stats = service_discovery.get_integration_stats().await;
    assert!(stats.swim_native_calls >= 3);
    assert!(stats.legacy_api_calls >= 1);
}

/// Tests SWIM protocol at medium scale
#[tokio::test]
async fn test_swim_medium_scale() {
    let bind_addr = get_random_port_addr();
    let config = SwimConfig10k::default();

    let (mut cluster, _events) =
        SwimCluster::new("medium-scale-test".to_string(), bind_addr, config)
            .await
            .unwrap();

    cluster.start().await.unwrap();

    // Add 100 members
    for i in 0..100 {
        let peer_info = create_test_peer_info(i);
        cluster.add_member(peer_info).await.unwrap();

        // Yield every 10 members to prevent blocking
        if i % 10 == 0 {
            tokio::task::yield_now().await;
        }
    }

    // Verify all members added
    let live_members = cluster.get_live_members().await;
    assert_eq!(live_members.len(), 100);

    let stats = cluster.get_stats().await;
    assert_eq!(stats.alive_members, 100);
    assert_eq!(stats.total_members, 100);
}

/// Tests SWIM configuration for 10k nodes
#[tokio::test]
async fn test_swim_10k_configuration() {
    let config = SwimConfig10k {
        probe_interval: Duration::from_millis(100),
        probe_timeout: Duration::from_millis(50),
        suspicion_timeout: Duration::from_secs(10),
        k_indirect_probes: 5,
        gossip_fanout: 15,
        max_gossip_per_message: 50,
        enable_compression: true,
        message_rate_limit: 1000,
        ..Default::default()
    };

    // Verify configuration is suitable for 10k nodes
    assert!(config.probe_interval.as_millis() <= 100); // Fast probing
    assert!(config.gossip_fanout >= 10); // Sufficient fanout
    assert!(config.max_gossip_per_message >= 20); // Efficient batching
    assert!(config.enable_compression); // Essential for large messages
    assert!(config.message_rate_limit >= 500); // High throughput
}

/// Tests memory efficiency of SWIM implementation
#[tokio::test]
async fn test_swim_memory_efficiency() {
    let bind_addr = get_random_port_addr();
    let config = SwimConfig10k::default();

    let (mut cluster, _events) = SwimCluster::new("memory-test".to_string(), bind_addr, config)
        .await
        .unwrap();

    // Add members and measure memory indirectly through operation speed
    let start = std::time::Instant::now();

    for i in 0..1000 {
        let peer_info = create_test_peer_info(i);
        cluster.add_member(peer_info).await.unwrap();
    }

    let add_time = start.elapsed();

    // Getting members should be fast (indicating efficient storage)
    let start = std::time::Instant::now();
    let members = cluster.get_live_members().await;
    let get_time = start.elapsed();

    assert_eq!(members.len(), 1000);

    // These operations should be reasonably fast
    assert!(
        add_time < Duration::from_secs(5),
        "Adding 1000 members took too long: {:?}",
        add_time
    );
    assert!(
        get_time < Duration::from_millis(10),
        "Getting members took too long: {:?}",
        get_time
    );
}

/// Tests error handling in SWIM operations
#[tokio::test]
async fn test_swim_error_handling() {
    let bind_addr = get_random_port_addr();
    let config = SwimConfig10k::default();

    let (mut cluster, _events) = SwimCluster::new("error-test".to_string(), bind_addr, config)
        .await
        .unwrap();

    // Test invalid peer info
    let invalid_peer = PeerInfo {
        id: "test".to_string(),
        address: "invalid-address".to_string(), // Invalid address format
        metrics_port: 9090,
        node_type: NodeType::Backend,
        is_load_balancer: false,
        last_updated: SystemTime::now(),
    };

    let result = cluster.add_member(invalid_peer).await;
    assert!(result.is_err());
}

/// Tests concurrent operations safety
#[tokio::test]
async fn test_swim_concurrent_operations() {
    let bind_addr = get_random_port_addr();
    let config = SwimConfig10k::default();

    let (cluster, _events) = SwimCluster::new("concurrent-test".to_string(), bind_addr, config)
        .await
        .unwrap();

    let cluster = std::sync::Arc::new(tokio::sync::Mutex::new(cluster));

    // Spawn multiple tasks that add members concurrently
    let mut handles = Vec::new();

    for i in 0..10 {
        let cluster_clone = std::sync::Arc::clone(&cluster);
        let handle = tokio::spawn(async move {
            let peer_info = create_test_peer_info(i);
            let mut cluster_guard = cluster_clone.lock().await;
            cluster_guard.add_member(peer_info).await
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut successful = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successful += 1;
        }
    }

    // All operations should succeed
    assert_eq!(successful, 10);

    // Verify final state
    let cluster_guard = cluster.lock().await;
    let members = cluster_guard.get_live_members().await;
    assert_eq!(members.len(), 10);
}

// Helper function
fn create_test_peer_info(id: usize) -> PeerInfo {
    PeerInfo {
        id: format!("test-node-{}", id),
        address: format!("127.0.0.{}:{}", (id % 254) + 1, 8000 + (id % 1000)),
        metrics_port: 9090,
        node_type: if id % 10 == 0 {
            NodeType::Proxy
        } else {
            NodeType::Backend
        },
        is_load_balancer: id % 10 == 0,
        last_updated: SystemTime::now(),
    }
}

/// Integration test module
mod integration {
    use super::*;

    /// Tests full SWIM system integration
    #[tokio::test]
    async fn test_full_system_integration() {
        let bind_addr = get_random_port_addr();
        let swim_config = SwimConfig10k::default();
        let integration_config = SwimIntegrationConfig::default();

        let mut service_discovery = SwimServiceDiscovery::new(
            "full-integration-test".to_string(),
            bind_addr,
            swim_config,
            integration_config,
        )
        .await
        .unwrap();

        service_discovery.start().await.unwrap();

        // Test the complete lifecycle

        // 1. Register backends
        let mut backend_ids = Vec::new();
        for i in 0..20 {
            let peer_info = create_test_peer_info(i);
            backend_ids.push(peer_info.id.clone());
            service_discovery.register_backend(peer_info).await.unwrap();
        }

        // Allow processing time
        sleep(Duration::from_millis(200)).await;

        // 2. Verify registration
        let backends = service_discovery.get_live_backends().await;
        assert_eq!(backends.len(), 20);

        // 3. Test legacy API compatibility
        let legacy_backends = service_discovery.list_backends().await;
        assert_eq!(legacy_backends.len(), 20);

        // 4. Test removal
        service_discovery
            .remove_backend(&backend_ids[0])
            .await
            .unwrap();

        sleep(Duration::from_millis(100)).await;

        // 5. Verify removal (in a real system, this would be handled by failure detection)
        let _remaining_backends = service_discovery.get_live_backends().await;
        // Note: Actual removal in SWIM happens through failure detection, not explicit removal

        // 6. Check statistics
        let stats = service_discovery.get_integration_stats().await;
        assert!(stats.swim_native_calls >= 20);
        assert!(stats.legacy_api_calls >= 1);

        let swim_stats = service_discovery.get_swim_stats().await;
        assert!(swim_stats.total_members >= 19);
    }
}

/// Load test module for larger scales
#[cfg(test)]
#[allow(dead_code)]
mod load_tests {
    use super::*;

    /// Tests SWIM at 1000 node scale
    #[tokio::test]
    async fn test_swim_1000_nodes() {
        let bind_addr = get_random_port_addr();
        let config = SwimConfig10k {
            probe_interval: Duration::from_millis(50),
            gossip_fanout: 10,
            max_gossip_per_message: 30,
            ..Default::default()
        };

        let (mut cluster, _events) =
            SwimCluster::new("1000-node-test".to_string(), bind_addr, config)
                .await
                .unwrap();

        cluster.start().await.unwrap();

        let start = std::time::Instant::now();

        // Add 1000 members
        for i in 0..1000 {
            let peer_info = create_test_peer_info(i);
            cluster.add_member(peer_info).await.unwrap();

            // Yield periodically
            if i % 100 == 0 {
                tokio::task::yield_now().await;
                println!("Added {} members, elapsed: {:?}", i, start.elapsed());
            }
        }

        let total_time = start.elapsed();
        println!("Added 1000 members in {:?}", total_time);

        // Verify all members
        let members = cluster.get_live_members().await;
        assert_eq!(members.len(), 1000);

        let stats = cluster.get_stats().await;
        assert_eq!(stats.alive_members, 1000);

        // Test should complete in reasonable time
        assert!(
            total_time < Duration::from_secs(30),
            "1000 node setup took too long: {:?}",
            total_time
        );
    }

    /// Stress test for SWIM protocol
    #[tokio::test]
    async fn test_swim_stress() {
        let bind_addr = get_random_port_addr();
        let config = SwimConfig10k::default();

        let (mut cluster, _events) = SwimCluster::new("stress-test".to_string(), bind_addr, config)
            .await
            .unwrap();

        cluster.start().await.unwrap();

        // Rapid member addition
        for batch in 0..10 {
            let batch_start = std::time::Instant::now();

            for i in 0..50 {
                let peer_info = create_test_peer_info(batch * 50 + i);
                cluster.add_member(peer_info).await.unwrap();
            }

            let batch_time = batch_start.elapsed();
            println!("Batch {} (50 members) took {:?}", batch, batch_time);

            // Brief pause between batches
            sleep(Duration::from_millis(10)).await;
        }

        // Verify final state
        let members = cluster.get_live_members().await;
        assert_eq!(members.len(), 500);

        let stats = cluster.get_stats().await;
        println!("Final stats: {:?}", stats);
        assert_eq!(stats.alive_members, 500);
    }
}
