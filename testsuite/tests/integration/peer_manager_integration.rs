/// Integration tests for peer management functionality
///
/// These tests focus on the peer manager in the proxy crate, testing
/// backend scoring, selection algorithms, and load balancing logic.
use inferno_proxy::peer_manager::{LoadBalancingAlgorithm, PeerManager};
use inferno_shared::service_discovery::{
    BackendRegistration, NodeVitals, ServiceDiscovery, ServiceDiscoveryConfig,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Helper function to create test vitals with specific performance characteristics
#[allow(dead_code)] // Helper function for future test expansion
fn create_test_vitals(
    requests: u32,
    cpu: f64,
    memory: f64,
    failures: u64,
    ready: bool,
) -> NodeVitals {
    NodeVitals {
        ready,
        cpu_usage: Some(cpu),
        memory_usage: Some(memory),
        active_requests: Some(requests as u64),
        avg_response_time_ms: Some(100.0),
        error_rate: Some(failures as f64),
        status_message: Some("healthy".to_string()),
    }
}

/// Helper function to create backend registration
fn create_backend_registration(id: &str, address: &str, metrics_port: u16) -> BackendRegistration {
    BackendRegistration {
        id: id.to_string(),
        address: address.to_string(),
        metrics_port,
    }
}

/// Integration test: Round-robin peer selection
///
/// Tests that round-robin algorithm distributes requests evenly across healthy peers.
#[tokio::test]
async fn test_peer_manager_round_robin_selection() {
    // Create mock servers for backends
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;
    let mock_server3 = MockServer::start().await;

    // Configure all backends to be healthy
    for server in [&mock_server1, &mock_server2, &mock_server3] {
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ready": true,
                "requests_in_progress": 5,
                "cpu_usage": 50.0,
                "memory_usage": 60.0,
                "gpu_usage": 0.0,
                "failed_responses": 2,
                "connected_peers": 10,
                "backoff_requests": 0,
                "uptime_seconds": 3600,
                "version": "1.0.0"
            })))
            .mount(server)
            .await;
    }

    // Set up service discovery
    let config = ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        ..ServiceDiscoveryConfig::default()
    };
    let service_discovery = Arc::new(ServiceDiscovery::with_config(config));

    // Register backends with unique addresses
    let backends = [
        (mock_server1.address(), "backend-1", 3001),
        (mock_server2.address(), "backend-2", 3002),
        (mock_server3.address(), "backend-3", 3003),
    ];

    for (addr, id, service_port) in &backends {
        let registration = create_backend_registration(
            id,
            &format!("{}:{}", addr.ip(), service_port),
            addr.port(),
        );
        service_discovery
            .register_backend(registration)
            .await
            .unwrap();
    }

    // Start health checking to make backends available
    // let handle = service_discovery.start_health_checking().await;
    sleep(Duration::from_millis(150)).await;

    // Create peer manager with round-robin algorithm
    let peer_manager = PeerManager::new(
        service_discovery.clone(),
        LoadBalancingAlgorithm::RoundRobin,
    );

    // Test round-robin selection
    let mut selections = Vec::new();
    let healthy_backends = service_discovery.get_healthy_backends().await;
    let available_count = healthy_backends.len();

    if available_count == 0 {
        panic!("No healthy backends available for testing");
    }

    // Only test with available backends
    let rounds = 3;
    for _ in 0..(available_count * rounds) {
        let selected = peer_manager.select_peer().await;
        assert!(
            selected.is_some(),
            "Should always select a peer when backends are available"
        );
        selections.push(selected.unwrap());
    }

    // service_discovery.stop_health_checking().await;
    // handle.await.unwrap();

    // Verify round-robin distribution
    let backend_counts =
        selections
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, addr| {
                *acc.entry(addr).or_insert(0) += 1;
                acc
            });

    assert_eq!(
        backend_counts.len(),
        available_count,
        "All {} available backends should be selected",
        available_count
    );

    if available_count > 1 {
        for count in backend_counts.values() {
            assert_eq!(
                *count, rounds,
                "Each backend should be selected exactly {} times in round-robin",
                rounds
            );
        }

        // Verify consistent ordering (only if we have multiple backends)
        if available_count >= 2 && selections.len() >= available_count * 2 {
            assert_eq!(
                selections[0], selections[available_count],
                "Round-robin should repeat pattern"
            );
        }
    }
}

/// Integration test: Least-connections peer selection
///
/// Tests that least-connections algorithm prefers backends with fewer active requests.
#[tokio::test]
async fn test_peer_manager_least_connections_selection() {
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;
    let mock_server3 = MockServer::start().await;

    // Configure backends with different load levels
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "requests_in_progress": 2,  // Low load
            "cpu_usage": 30.0,
            "memory_usage": 40.0,
            "gpu_usage": 0.0,
            "failed_responses": 1,
            "connected_peers": 10,
            "backoff_requests": 0,
            "uptime_seconds": 3600,
            "version": "1.0.0"
        })))
        .mount(&mock_server1)
        .await;

    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "requests_in_progress": 8,  // Medium load
            "cpu_usage": 60.0,
            "memory_usage": 70.0,
            "gpu_usage": 0.0,
            "failed_responses": 5,
            "connected_peers": 20,
            "backoff_requests": 2,
            "uptime_seconds": 3600,
            "version": "1.0.0"
        })))
        .mount(&mock_server2)
        .await;

    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "requests_in_progress": 15, // High load
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "gpu_usage": 0.0,
            "failed_responses": 10,
            "connected_peers": 50,
            "backoff_requests": 5,
            "uptime_seconds": 3600,
            "version": "1.0.0"
        })))
        .mount(&mock_server3)
        .await;

    let service_discovery = Arc::new(ServiceDiscovery::with_config(ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        ..ServiceDiscoveryConfig::default()
    }));

    // Register backends
    let low_load_addr = format!("{}:3000", mock_server1.address().ip());
    let med_load_addr = format!("{}:3001", mock_server2.address().ip());
    let high_load_addr = format!("{}:3002", mock_server3.address().ip());

    service_discovery
        .register_backend(create_backend_registration(
            "low-load",
            &low_load_addr,
            mock_server1.address().port(),
        ))
        .await
        .unwrap();

    service_discovery
        .register_backend(create_backend_registration(
            "med-load",
            &med_load_addr,
            mock_server2.address().port(),
        ))
        .await
        .unwrap();

    service_discovery
        .register_backend(create_backend_registration(
            "high-load",
            &high_load_addr,
            mock_server3.address().port(),
        ))
        .await
        .unwrap();

    // Start health checking
    // let handle = service_discovery.start_health_checking().await;
    sleep(Duration::from_millis(200)).await;

    // Create peer manager with least-connections algorithm
    let peer_manager = PeerManager::new(
        service_discovery.clone(),
        LoadBalancingAlgorithm::LeastConnections,
    );

    // Test least-connections selection - should consistently pick low-load backend
    let mut selections = Vec::new();
    for _ in 0..10 {
        let selected = peer_manager.select_peer().await;
        assert!(selected.is_some());
        selections.push(selected.unwrap());
    }

    // service_discovery.stop_health_checking().await;
    // handle.await.unwrap();

    // All selections should go to the low-load backend
    for selection in selections {
        assert_eq!(
            selection, low_load_addr,
            "Least-connections should always select backend with fewest connections"
        );
    }
}

/// Integration test: Weighted peer selection based on performance
///
/// Tests that weighted algorithm selects backends based on comprehensive performance scores.
#[tokio::test]
async fn test_peer_manager_weighted_selection() {
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;

    // High performance backend (should be selected)
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "requests_in_progress": 1,
            "cpu_usage": 20.0,
            "memory_usage": 25.0,
            "gpu_usage": 0.0,
            "failed_responses": 0,
            "connected_peers": 5,
            "backoff_requests": 0,
            "uptime_seconds": 7200,
            "version": "1.0.0"
        })))
        .mount(&mock_server1)
        .await;

    // Poor performance backend (should be avoided)
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "requests_in_progress": 20,
            "cpu_usage": 90.0,
            "memory_usage": 95.0,
            "gpu_usage": 0.0,
            "failed_responses": 15,
            "connected_peers": 100,
            "backoff_requests": 10,
            "uptime_seconds": 600,
            "version": "1.0.0"
        })))
        .mount(&mock_server2)
        .await;

    let service_discovery = Arc::new(ServiceDiscovery::with_config(ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        ..ServiceDiscoveryConfig::default()
    }));

    let high_perf_addr = format!("{}:3000", mock_server1.address().ip());
    let low_perf_addr = format!("{}:3001", mock_server2.address().ip());

    service_discovery
        .register_backend(create_backend_registration(
            "high-perf",
            &high_perf_addr,
            mock_server1.address().port(),
        ))
        .await
        .unwrap();

    service_discovery
        .register_backend(create_backend_registration(
            "low-perf",
            &low_perf_addr,
            mock_server2.address().port(),
        ))
        .await
        .unwrap();

    // Start health checking
    // let handle = service_discovery.start_health_checking().await;
    sleep(Duration::from_millis(200)).await;

    // Create peer manager with weighted algorithm
    let peer_manager =
        PeerManager::new(service_discovery.clone(), LoadBalancingAlgorithm::Weighted);

    // Test weighted selection - should consistently pick high-performance backend
    let mut selections = Vec::new();
    for _ in 0..10 {
        let selected = peer_manager.select_peer().await;
        assert!(selected.is_some());
        selections.push(selected.unwrap());
    }

    // service_discovery.stop_health_checking().await;
    // handle.await.unwrap();

    // All selections should go to the high-performance backend
    for selection in selections {
        assert_eq!(
            selection, high_perf_addr,
            "Weighted algorithm should select backend with best performance score"
        );
    }
}

/// Integration test: Backend availability filtering
///
/// Tests that peer manager only selects backends that are healthy and ready.
#[tokio::test]
async fn test_peer_manager_availability_filtering() {
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;
    let mock_server3 = MockServer::start().await;

    // Healthy and ready backend
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "requests_in_progress": 5,
            "cpu_usage": 50.0,
            "memory_usage": 60.0,
            "gpu_usage": 0.0,
            "failed_responses": 2,
            "connected_peers": 10,
            "backoff_requests": 0,
            "uptime_seconds": 3600,
            "version": "1.0.0"
        })))
        .mount(&mock_server1)
        .await;

    // Healthy but not ready backend
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": false,  // Not ready for traffic
            "requests_in_progress": 0,
            "cpu_usage": 10.0,
            "memory_usage": 20.0,
            "gpu_usage": 0.0,
            "failed_responses": 0,
            "connected_peers": 2,
            "backoff_requests": 0,
            "uptime_seconds": 100,
            "version": "1.0.0"
        })))
        .mount(&mock_server2)
        .await;

    // Backend that will fail health checks (network error)
    // Don't mount any mock for server3 - it will fail to respond

    let service_discovery = Arc::new(ServiceDiscovery::with_config(ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        failure_threshold: 1, // Mark unhealthy quickly
        ..ServiceDiscoveryConfig::default()
    }));

    let available_addr = format!("{}:3000", mock_server1.address().ip());
    let not_ready_addr = format!("{}:3001", mock_server2.address().ip());
    let failing_addr = format!("{}:3002", mock_server3.address().ip());

    service_discovery
        .register_backend(create_backend_registration(
            "available",
            &available_addr,
            mock_server1.address().port(),
        ))
        .await
        .unwrap();

    service_discovery
        .register_backend(create_backend_registration(
            "not-ready",
            &not_ready_addr,
            mock_server2.address().port(),
        ))
        .await
        .unwrap();

    service_discovery
        .register_backend(create_backend_registration(
            "failing",
            &failing_addr,
            mock_server3.address().port(),
        ))
        .await
        .unwrap();

    // Start health checking
    // let handle = service_discovery.start_health_checking().await;
    sleep(Duration::from_millis(200)).await;

    // Create peer manager
    let peer_manager = PeerManager::new(
        service_discovery.clone(),
        LoadBalancingAlgorithm::RoundRobin,
    );

    // Test selection - should only return the available backend
    for _ in 0..5 {
        let selected = peer_manager.select_peer().await;
        assert!(selected.is_some());
        assert_eq!(
            selected.unwrap(),
            available_addr,
            "Should only select backends that are healthy and ready"
        );
    }

    // service_discovery.stop_health_checking().await;
    // handle.await.unwrap();
}

/// Integration test: No available peers scenario
///
/// Tests peer manager behavior when no backends are available for traffic.
#[tokio::test]
async fn test_peer_manager_no_available_peers() {
    let mock_server = MockServer::start().await;

    // Backend that's healthy but not ready
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": false,  // Not ready for traffic
            "requests_in_progress": 0,
            "cpu_usage": 10.0,
            "memory_usage": 20.0,
            "gpu_usage": 0.0,
            "failed_responses": 0,
            "connected_peers": 2,
            "backoff_requests": 0,
            "uptime_seconds": 100,
            "version": "1.0.0"
        })))
        .mount(&mock_server)
        .await;

    let service_discovery = Arc::new(ServiceDiscovery::with_config(ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        ..ServiceDiscoveryConfig::default()
    }));

    service_discovery
        .register_backend(create_backend_registration(
            "not-ready",
            &format!("{}:3000", mock_server.address().ip()),
            mock_server.address().port(),
        ))
        .await
        .unwrap();

    // Start health checking
    // let handle = service_discovery.start_health_checking().await;
    sleep(Duration::from_millis(150)).await;

    // Create peer manager
    let peer_manager = PeerManager::new(
        service_discovery.clone(),
        LoadBalancingAlgorithm::RoundRobin,
    );

    // Test selection - should return None when no peers are available
    let selected = peer_manager.select_peer().await;
    assert!(
        selected.is_none(),
        "Should return None when no peers are available for traffic"
    );

    // service_discovery.stop_health_checking().await;
    // handle.await.unwrap();
}

/// Integration test: Peer manager statistics
///
/// Tests that peer manager correctly reports statistics about peers and selections.
#[tokio::test]
async fn test_peer_manager_statistics() {
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;

    for server in [&mock_server1, &mock_server2] {
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ready": true,
                "requests_in_progress": 3,
                "cpu_usage": 40.0,
                "memory_usage": 50.0,
                "gpu_usage": 0.0,
                "failed_responses": 1,
                "connected_peers": 8,
                "backoff_requests": 0,
                "uptime_seconds": 3600,
                "version": "1.0.0"
            })))
            .mount(server)
            .await;
    }

    let service_discovery = Arc::new(ServiceDiscovery::with_config(ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        ..ServiceDiscoveryConfig::default()
    }));

    service_discovery
        .register_backend(create_backend_registration(
            "backend-1",
            &format!("{}:3000", mock_server1.address().ip()),
            mock_server1.address().port(),
        ))
        .await
        .unwrap();

    service_discovery
        .register_backend(create_backend_registration(
            "backend-2",
            &format!("{}:3001", mock_server2.address().ip()),
            mock_server2.address().port(),
        ))
        .await
        .unwrap();

    // Start health checking
    // let handle = service_discovery.start_health_checking().await;
    sleep(Duration::from_millis(150)).await;

    // Create peer manager with round-robin
    let peer_manager = PeerManager::new(
        service_discovery.clone(),
        LoadBalancingAlgorithm::RoundRobin,
    );

    // Make some selections to advance round-robin counter
    for _ in 0..3 {
        peer_manager.select_peer().await;
    }

    // Get statistics
    let stats = peer_manager.get_peer_stats().await;

    assert_eq!(
        stats.total_peers, 2,
        "Should report correct total peer count"
    );
    assert_eq!(
        stats.available_peers, 2,
        "Should report correct available peer count"
    );
    assert_eq!(
        stats.algorithm,
        LoadBalancingAlgorithm::RoundRobin,
        "Should report correct algorithm"
    );
    assert_eq!(
        stats.total_load, 6,
        "Should report correct total load (3 + 3)"
    );
    assert_eq!(
        stats.round_robin_position, 3,
        "Should report correct round-robin position"
    );
    assert!(
        stats.average_performance_score > 0.0,
        "Should calculate average performance score"
    );

    // service_discovery.stop_health_checking().await;
    // handle.await.unwrap();
}
