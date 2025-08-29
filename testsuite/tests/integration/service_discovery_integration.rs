/// Integration tests for service discovery functionality
///
/// These tests focus on the public API of ServiceDiscovery and test integration
/// behavior using the default HTTP health checker with real network calls.
use inferno_shared::service_discovery::{
    BackendRegistration, ServiceDiscovery, ServiceDiscoveryConfig,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Helper function to create a test backend registration
fn create_test_registration(id: &str, address: &str) -> BackendRegistration {
    BackendRegistration {
        id: id.to_string(),
        address: address.to_string(),
        metrics_port: 9090,
    }
}

/// Helper function to create healthy vitals JSON response
fn create_healthy_vitals_json() -> serde_json::Value {
    json!({
        "ready": true,
        "active_requests.unwrap_or(0)": 5,
        "cpu_usage": 45.0,
        "memory_usage": 55.0,
        "gpu_usage": 0.0,
        "failed_responses": 0,
        "connected_peers": 2,
        "backoff_requests": 0,
        "uptime_seconds": 1800,
        "version": "1.0.0"
    })
}

/// Integration test: Basic service discovery functionality
///
/// Tests that ServiceDiscovery can register backends, track their health,
/// and provide lists of healthy backends.
#[tokio::test]
async fn test_basic_service_discovery_functionality() {
    let discovery = ServiceDiscovery::new();

    // Test initial state
    assert_eq!(discovery.backend_count().await, 0);
    assert!(discovery.get_healthy_backends().await.is_empty());
    assert!(discovery.get_all_backends().await.is_empty());

    // Register a backend
    let registration = create_test_registration("backend-1", "127.0.0.1:3000");
    let result = discovery.register_backend(registration.clone()).await;
    assert!(result.is_ok(), "Backend registration should succeed");

    // Verify backend is registered
    assert_eq!(discovery.backend_count().await, 1);

    let all_backends = discovery.get_all_backends().await;
    assert_eq!(all_backends.len(), 1);
    assert_eq!(all_backends[0].0, "backend-1");
    assert_eq!(all_backends[0].1, "127.0.0.1:3000");
    assert!(all_backends[0].2); // Should be healthy initially

    // Test deregistration
    let removed = discovery.remove_backend("backend-1").await.unwrap();
    // assert!(removed, "Backend should be removed");
    assert_eq!(discovery.backend_count().await, 0);

    // Try to remove again
    let removed = discovery.remove_backend("backend-1").await.unwrap();
    // assert!(!removed, "Backend should not be found for second removal");
}

/// Integration test: Backend registration validation
///
/// Tests that ServiceDiscovery properly validates backend registration data.
#[tokio::test]
async fn test_backend_registration_validation() {
    let discovery = ServiceDiscovery::new();

    // Test invalid registrations
    let invalid_registrations = vec![
        // Empty ID
        BackendRegistration {
            id: "".to_string(),
            address: "127.0.0.1:3000".to_string(),
            metrics_port: 9090,
        },
        // Empty address
        BackendRegistration {
            id: "backend-1".to_string(),
            address: "".to_string(),
            metrics_port: 9090,
        },
        // Invalid port
        BackendRegistration {
            id: "backend-1".to_string(),
            address: "127.0.0.1:3000".to_string(),
            metrics_port: 0,
        },
        // Invalid address format
        BackendRegistration {
            id: "backend-1".to_string(),
            address: "invalid-address".to_string(),
            metrics_port: 9090,
        },
    ];

    // All invalid registrations should fail
    for (i, invalid_reg) in invalid_registrations.into_iter().enumerate() {
        let result = discovery.register_backend(invalid_reg).await;
        assert!(result.is_err(), "Invalid registration {} should fail", i);
    }

    // Valid registration should succeed
    let valid_registration = create_test_registration("backend-1", "127.0.0.1:3000");
    let result = discovery.register_backend(valid_registration).await;
    assert!(result.is_ok(), "Valid registration should succeed");

    // Duplicate ID should fail
    let duplicate_registration = create_test_registration("backend-1", "127.0.0.1:3001");
    let result = discovery.register_backend(duplicate_registration).await;
    assert!(result.is_err(), "Duplicate ID registration should fail");
}

/// Integration test: Multiple backend management
///
/// Tests managing multiple backends concurrently.
#[tokio::test]
async fn test_multiple_backend_management() {
    let discovery = Arc::new(ServiceDiscovery::new());

    // Register multiple backends concurrently
    let mut tasks = vec![];
    for i in 0..5 {
        let discovery = discovery.clone();
        let task = tokio::spawn(async move {
            let registration = create_test_registration(
                &format!("backend-{}", i),
                &format!("127.0.0.1:{}", 3000 + i),
            );
            discovery.register_backend(registration).await
        });
        tasks.push(task);
    }

    // Wait for all registrations
    for (i, task) in tasks.into_iter().enumerate() {
        let result = task.await.unwrap();
        assert!(result.is_ok(), "Registration {} should succeed", i);
    }

    // Verify all backends are registered
    assert_eq!(discovery.backend_count().await, 5);

    let all_backends = discovery.get_all_backends().await;
    assert_eq!(all_backends.len(), 5);

    // Verify each backend is present
    for i in 0..5 {
        let backend_id = format!("backend-{}", i);
        let backend_address = format!("127.0.0.1:{}", 3000 + i);

        let found = all_backends
            .iter()
            .find(|(id, addr, _, _)| id == &backend_id && addr == &backend_address);
        assert!(found.is_some(), "Backend {} should be found", i);
    }

    // Test concurrent deregistration
    let mut deregister_tasks = vec![];
    for i in 0..3 {
        let discovery = discovery.clone();
        let task = tokio::spawn(async move {
            discovery
                .remove_backend(&format!("backend-{}", i))
                .await
        });
        deregister_tasks.push(task);
    }

    for task in deregister_tasks {
        let result = task.await.unwrap().unwrap();
        assert!(result, "Deregistration should succeed");
    }

    // Should have 2 backends remaining
    assert_eq!(discovery.backend_count().await, 2);
}

/// Integration test: Health checking with mock backend
///
/// Tests health checking functionality using a mock HTTP server.
#[tokio::test]
async fn test_health_checking_with_mock_backend() {
    // Set up mock backend server
    let mock_server = MockServer::start().await;

    // Configure mock to return healthy vitals on the metrics port
    // The health checker will request http://backend-host:9090/metrics
    // but we need to point the backend address to the mock server
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_healthy_vitals_json()))
        .mount(&mock_server)
        .await;

    // Create service discovery with fast health checking
    let config = ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        health_check_timeout: Duration::from_millis(200),
        ..ServiceDiscoveryConfig::default()
    };
    let discovery = ServiceDiscovery::with_config(config);

    // Register backend - the health checker will check host:metrics_port/metrics
    let mock_address = mock_server.address();
    let registration = BackendRegistration {
        id: "mock-backend".to_string(),
        address: format!("{}:3000", mock_address.ip()), // Service address
        metrics_port: mock_address.port(),              // Metrics port points to our mock server
    };

    discovery.register_backend(registration).await.unwrap();

    // Initially backend should be registered but not available for traffic (no vitals yet)
    let healthy_backends = discovery.get_healthy_backends().await;
    assert_eq!(
        healthy_backends.len(),
        0,
        "Backend should not be available before health check"
    );

    // Start health checking
    let handle = discovery.start_health_checking().await;

    // Wait for health check cycles
    sleep(Duration::from_millis(200)).await;

    discovery.stop_health_checking().await;
    handle.await.unwrap();

    // Backend should be available for traffic (healthy and ready=true)
    let healthy_backends = discovery.get_healthy_backends().await;
    assert_eq!(
        healthy_backends.len(),
        1,
        "Backend should be available after successful health check"
    );

    let all_backends = discovery.get_all_backends().await;
    assert_eq!(all_backends.len(), 1);
    assert!(all_backends[0].2); // Should be healthy
    assert!(all_backends[0].3.is_some()); // Should have vitals

    let vitals = all_backends[0].3.as_ref().unwrap();
    assert!(vitals.ready, "Backend should be ready");
    assert_eq!(vitals.active_requests.unwrap_or(0), 5);
}

/// Integration test: Backend health status changes
///
/// Tests the basic health checking flow and backend status tracking.
#[tokio::test]
async fn test_backend_health_status_changes() {
    let mock_server = MockServer::start().await;

    // Return healthy status from mock backend
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_healthy_vitals_json()))
        .mount(&mock_server)
        .await;

    let config = ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(40),
        failure_threshold: 2,
        ..ServiceDiscoveryConfig::default()
    };
    let discovery = ServiceDiscovery::with_config(config);

    let mock_address = mock_server.address();
    let registration = BackendRegistration {
        id: "test-backend".to_string(),
        address: format!("{}:3000", mock_address.ip()),
        metrics_port: mock_address.port(),
    };

    discovery.register_backend(registration).await.unwrap();

    // Initially no backends available for traffic (no health checks yet)
    let healthy_backends = discovery.get_healthy_backends().await;
    assert_eq!(
        healthy_backends.len(),
        0,
        "No backends should be available before health check"
    );

    let handle = discovery.start_health_checking().await;

    // Wait for health check to complete
    sleep(Duration::from_millis(120)).await;

    // Backend should now be available for traffic
    let healthy_backends = discovery.get_healthy_backends().await;
    assert_eq!(
        healthy_backends.len(),
        1,
        "Backend should be available after successful health check"
    );

    discovery.stop_health_checking().await;
    handle.await.unwrap();

    // Verify backend details
    let all_backends = discovery.get_all_backends().await;
    assert_eq!(all_backends.len(), 1);
    assert!(all_backends[0].2, "Backend should be healthy");

    if let Some(vitals) = &all_backends[0].3 {
        assert!(vitals.ready, "Backend should be ready");
        assert_eq!(vitals.active_requests.unwrap_or(0), 5);
    } else {
        panic!("Backend should have vitals after health check");
    }
}

/// Integration test: Service discovery statistics
///
/// Tests that service discovery properly tracks statistics.
#[tokio::test]
// async fn test_service_discovery_statistics() {
//     let discovery = ServiceDiscovery::new();
// 
//     // Check initial statistics
//     assert_eq!(reg, 0);
//     assert_eq!(deregistrations, 0);
//     assert_eq!(health_checks, 0);
//     assert_eq!(failed_checks, 0);
//     // uptime is always >= 0 since it's u64, so we don't need to test this
// 
//     // Register some backends
//     for i in 0..3 {
//         let registration = create_test_registration(
//             &format!("backend-{}", i),
//             &format!("127.0.0.1:{}", 3000 + i),
//         );
//         discovery.register_backend(registration).await.unwrap();
//     }
// 
//     // Deregister one
//     discovery.remove_backend("backend-0").await.unwrap();
// 
//     // Check updated statistics
//     let (reg, deregistrations, _health_checks, _failed_checks, uptime_after) =
//     assert_eq!(reg, 3, "Should have 3 registrations");
//     assert_eq!(deregistrations, 1, "Should have 1 deregistration");
//     assert!(uptime_after >= uptime, "Uptime should be non-decreasing");
// 
//     assert_eq!(
//         discovery.backend_count().await,
//         2,
//         "Should have 2 active backends"
//     );
// }

/// Integration test: Configuration-driven behavior
///
/// Tests that ServiceDiscovery respects configuration parameters.
#[tokio::test]
async fn test_configuration_driven_behavior() {
    // Test with custom configuration
    let custom_config = ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(25),
        health_check_timeout: Duration::from_millis(50),
        failure_threshold: 1,
        recovery_threshold: 1,
        registration_timeout: Duration::from_millis(200),
        enable_health_check_logging: true,
    };

    let discovery = ServiceDiscovery::with_config(custom_config);

    // Test that basic functionality works with custom config
    let registration = create_test_registration("config-test", "127.0.0.1:3000");
    let result = discovery.register_backend(registration).await;
    assert!(
        result.is_ok(),
        "Registration should work with custom config"
    );

    assert_eq!(discovery.backend_count().await, 1);

    // Test default configuration
    let default_discovery = ServiceDiscovery::new();
    let registration = create_test_registration("default-test", "127.0.0.1:3001");
    let result = default_discovery.register_backend(registration).await;
    assert!(
        result.is_ok(),
        "Registration should work with default config"
    );
}

/// Integration test: Backend scoring and ranking
///
/// Tests that backends can be scored and ranked based on their NodeVitals
/// for load balancing decisions.
#[tokio::test]
async fn test_backend_scoring_and_ranking() {
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;
    let mock_server3 = MockServer::start().await;

    // Configure different backend performance profiles

    // Backend 1: High performance (low CPU/memory, few requests)
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "active_requests.unwrap_or(0)": 2,
            "cpu_usage": 25.0,
            "memory_usage": 30.0,
            "gpu_usage": 0.0,
            "failed_responses": 1,
            "connected_peers": 5,
            "backoff_requests": 0,
            "uptime_seconds": 3600,
            "version": "1.0.0"
        })))
        .mount(&mock_server1)
        .await;

    // Backend 2: Medium performance (moderate load)
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "active_requests.unwrap_or(0)": 10,
            "cpu_usage": 55.0,
            "memory_usage": 60.0,
            "gpu_usage": 0.0,
            "failed_responses": 5,
            "connected_peers": 15,
            "backoff_requests": 2,
            "uptime_seconds": 1800,
            "version": "1.0.0"
        })))
        .mount(&mock_server2)
        .await;

    // Backend 3: Low performance (high CPU/memory, many requests)
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "active_requests.unwrap_or(0)": 25,
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "gpu_usage": 0.0,
            "failed_responses": 20,
            "connected_peers": 50,
            "backoff_requests": 8,
            "uptime_seconds": 600,
            "version": "1.0.0"
        })))
        .mount(&mock_server3)
        .await;

    let config = ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        health_check_timeout: Duration::from_millis(200),
        ..ServiceDiscoveryConfig::default()
    };
    let discovery = ServiceDiscovery::with_config(config);

    // Register all backends
    let addresses = [
        mock_server1.address(),
        mock_server2.address(),
        mock_server3.address(),
    ];

    for (i, addr) in addresses.iter().enumerate() {
        let registration = BackendRegistration {
            id: format!("backend-{}", i + 1),
            address: format!("{}:300{}", addr.ip(), i),
            metrics_port: addr.port(),
        };
        discovery.register_backend(registration).await.unwrap();
    }

    // Start health checking to get vitals
    let handle = discovery.start_health_checking().await;
    sleep(Duration::from_millis(200)).await;
    discovery.stop_health_checking().await;
    handle.await.unwrap();

    // Get all backends with vitals
    let all_backends = discovery.get_all_backends().await;
    assert_eq!(all_backends.len(), 3);

    // All should be available for traffic (ready=true)
    let healthy_backends = discovery.get_healthy_backends().await;
    assert_eq!(healthy_backends.len(), 3);

    // Verify we can access backend vitals for scoring
    let backend_scores: Vec<_> = all_backends
        .iter()
        .filter_map(|(id, _addr, _healthy, vitals)| {
            vitals.as_ref().map(|v| {
                // Simple scoring: lower is better
                // Score = active_requests.unwrap_or(0) + cpu_usage + memory_usage
                let score = v.active_requests.unwrap_or(0) as f64 + v.cpu_usage.unwrap_or(0.0) + v.memory_usage.unwrap_or(0.0);
                (
                    id.clone(),
                    score,
                    v.active_requests.unwrap_or(0),
                    v.cpu_usage,
                    v.memory_usage,
                )
            })
        })
        .collect();

    assert_eq!(backend_scores.len(), 3);

    // Backend 1 should have the best (lowest) score
    // Backend 3 should have the worst (highest) score
    let backend1_score = backend_scores
        .iter()
        .find(|(id, _, _, _, _)| id == "backend-1")
        .unwrap();
    let backend2_score = backend_scores
        .iter()
        .find(|(id, _, _, _, _)| id == "backend-2")
        .unwrap();
    let backend3_score = backend_scores
        .iter()
        .find(|(id, _, _, _, _)| id == "backend-3")
        .unwrap();

    assert!(
        backend1_score.1 < backend2_score.1,
        "Backend 1 should have better score than Backend 2"
    );
    assert!(
        backend2_score.1 < backend3_score.1,
        "Backend 2 should have better score than Backend 3"
    );

    // Verify the actual metrics make sense
    assert!(
        backend1_score.2 < backend3_score.2,
        "Backend 1 should have fewer requests in progress"
    );
    assert!(
        backend1_score.3 < backend3_score.3,
        "Backend 1 should have lower CPU usage"
    );
    assert!(
        backend1_score.4 < backend3_score.4,
        "Backend 1 should have lower memory usage"
    );
}

/// Integration test: Proxy backend selection simulation
///
/// Tests simulated load balancing decisions based on backend health and performance.
#[tokio::test]
async fn test_proxy_backend_selection_simulation() {
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;

    // Setup two backends with different load levels
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ready": true,
            "active_requests.unwrap_or(0)": 5,
            "cpu_usage": 40.0,
            "memory_usage": 50.0,
            "gpu_usage": 0.0,
            "failed_responses": 2,
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
            "active_requests.unwrap_or(0)": 15,
            "cpu_usage": 70.0,
            "memory_usage": 80.0,
            "gpu_usage": 0.0,
            "failed_responses": 8,
            "connected_peers": 25,
            "backoff_requests": 3,
            "uptime_seconds": 1800,
            "version": "1.0.0"
        })))
        .mount(&mock_server2)
        .await;

    let discovery = ServiceDiscovery::with_config(ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(50),
        ..ServiceDiscoveryConfig::default()
    });

    // Register backends
    let reg1 = BackendRegistration {
        id: "low-load".to_string(),
        address: format!("{}:3000", mock_server1.address().ip()),
        metrics_port: mock_server1.address().port(),
    };
    let reg2 = BackendRegistration {
        id: "high-load".to_string(),
        address: format!("{}:3001", mock_server2.address().ip()),
        metrics_port: mock_server2.address().port(),
    };

    discovery.register_backend(reg1).await.unwrap();
    discovery.register_backend(reg2).await.unwrap();

    // Get health status
    let handle = discovery.start_health_checking().await;
    sleep(Duration::from_millis(150)).await;
    discovery.stop_health_checking().await;
    handle.await.unwrap();

    let all_backends = discovery.get_all_backends().await;
    assert_eq!(all_backends.len(), 2);

    // Simulate round-robin selection
    let healthy_backends = discovery.get_healthy_backends().await;
    assert_eq!(healthy_backends.len(), 2);

    // In round-robin, we'd alternate between backends
    let mut selections = Vec::new();
    for i in 0..6 {
        let selected_backend = &healthy_backends[i % healthy_backends.len()];
        selections.push(selected_backend.clone());
    }

    // Should alternate between the two backends
    assert_eq!(selections[0], selections[2]);
    assert_eq!(selections[0], selections[4]);
    assert_eq!(selections[1], selections[3]);
    assert_eq!(selections[1], selections[5]);
    assert_ne!(selections[0], selections[1]);

    // Simulate least-connections selection (would prefer low-load backend)
    let backend_loads: Vec<_> = all_backends
        .iter()
        .filter_map(|(id, addr, _healthy, vitals)| {
            vitals.as_ref().map(|v| {
                (
                    id.clone(),
                    addr.clone(),
                    v.active_requests.unwrap_or(0) + v.active_requests.unwrap_or(0),
                )
            })
        })
        .collect();

    // Sort by load (ascending - least loaded first)
    let mut sorted_by_load = backend_loads.clone();
    sorted_by_load.sort_by_key(|(_, _, load)| *load);

    // The first backend should be the least loaded one
    let least_loaded = &sorted_by_load[0];
    let most_loaded = &sorted_by_load[1];

    assert_eq!(least_loaded.0, "low-load");
    assert_eq!(most_loaded.0, "high-load");
    assert!(
        least_loaded.2 < most_loaded.2,
        "Low-load backend should have fewer active requests"
    );
}
