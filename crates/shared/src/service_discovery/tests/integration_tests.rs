//! Integration tests for service discovery system
//!
//! These tests validate end-to-end functionality of the service discovery system
//! including health checking, peer registration, consensus, and configuration loading.

use crate::service_discovery::{
    AuthMode, BackendRegistration, HealthChecker, HttpHealthChecker, NodeInfo, NodeType,
    ServiceDiscovery, ServiceDiscoveryConfig,
};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_complete_service_discovery_workflow() {
    // Test the complete workflow: registration -> health checking -> consensus -> updates
    let config = ServiceDiscoveryConfig::default();
    let discovery = ServiceDiscovery::with_config(config);

    // 1. Test backend registration
    let backend1 = BackendRegistration {
        id: "backend-1".to_string(),
        address: "10.0.1.5:3000".to_string(),
        metrics_port: 9090,
    };

    let backend2 = BackendRegistration {
        id: "backend-2".to_string(),
        address: "10.0.1.6:3000".to_string(),
        metrics_port: 9090,
    };

    // Register backends
    discovery.register_backend(backend1).await.unwrap();
    discovery.register_backend(backend2).await.unwrap();

    // Verify registration
    assert_eq!(discovery.backend_count().await, 2);

    // 2. Test health checking startup
    assert!(!discovery.is_health_check_running().await);
    discovery.start_health_check_loop().await.unwrap();
    assert!(discovery.is_health_check_running().await);

    // 3. Test health check doesn't crash (run for short time)
    sleep(Duration::from_millis(100)).await;

    // 4. Test graceful shutdown
    discovery.stop_health_check_loop().await.unwrap();
    assert!(!discovery.is_health_check_running().await);

    // 5. Test that backends are still registered (health checks would remove unhealthy ones)
    let backends = discovery.get_healthy_backends().await;
    assert_eq!(backends.len(), 2);
}

#[tokio::test]
#[ignore] // Disabled - consensus removed in favor of SWIM
#[allow(clippy::useless_vec)]
async fn test_multi_peer_registration_with_consensus() {
    let discovery = ServiceDiscovery::new();

    let node = NodeInfo::new(
        "test-node".to_string(),
        "10.0.1.7:3000".to_string(),
        9090,
        NodeType::Backend,
    );

    // Test with empty peer URLs (should succeed with empty results)
    let peer_urls = vec![];
    let result = discovery.register_with_peers(&node, peer_urls).await;
    assert!(result.is_ok());
    let (responses, failed) = result.unwrap();
    assert_eq!(responses.len(), 0);
    assert_eq!(failed.len(), 0);

    // Test consensus resolution with mock data (instead of network calls)
    // This tests the consensus algorithm without network timeouts
    use crate::service_discovery::{PeerInfo, RegistrationResponse};
    use std::time::SystemTime;

    let peer_info = PeerInfo {
        id: "consensus-test-node".to_string(),
        address: "10.0.1.100:3000".to_string(),
        metrics_port: 9090,
        node_type: NodeType::Backend,
        is_load_balancer: false,
        last_updated: SystemTime::now(),
    };

    let _mock_responses = vec![RegistrationResponse {
        status: "registered".to_string(),
        message: Some("Success".to_string()),
        peers: vec![peer_info.clone()],
    }];

    // Disabled - consensus functionality removed
}

#[tokio::test]
async fn test_self_sovereign_updates() {
    let discovery = ServiceDiscovery::new();

    let node = NodeInfo::new(
        "updating-node".to_string(),
        "10.0.1.8:3000".to_string(),
        9090,
        NodeType::Backend,
    );

    // Test self-sovereign update with empty peer list
    let result = discovery.broadcast_self_update(&node).await;
    assert!(result.is_ok());
    let updates = result.unwrap();
    assert_eq!(updates.len(), 0); // No peers to update

    // Test retry metrics
    let metrics = discovery.get_retry_metrics().await;
    assert_eq!(metrics.retry_queue_size, 0);
    assert_eq!(metrics.successful_retries, 0);
}

#[tokio::test]
async fn test_configuration_from_environment() {
    use std::env;

    // Clean up any leftover environment variables from other tests
    env::remove_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_INTERVAL");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_TIMEOUT");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_FAILURE_THRESHOLD");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_SHARED_SECRET");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_ENABLE_LOGGING");

    // Set test environment variables
    env::set_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_INTERVAL", "10");
    env::set_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_TIMEOUT", "3");
    env::set_var("INFERNO_SERVICE_DISCOVERY_FAILURE_THRESHOLD", "5");
    env::set_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE", "shared_secret");
    env::set_var("INFERNO_SERVICE_DISCOVERY_SHARED_SECRET", "test-secret-123");

    let config = ServiceDiscoveryConfig::from_env().unwrap();

    assert_eq!(config.health_check_interval, Duration::from_secs(10));
    assert_eq!(config.health_check_timeout, Duration::from_secs(3));
    assert_eq!(config.failure_threshold, 5);
    assert_eq!(config.auth_mode, AuthMode::SharedSecret);
    assert_eq!(config.shared_secret, Some("test-secret-123".to_string()));

    // Test authentication validation
    assert!(config.validate_auth(Some("Bearer test-secret-123")));
    assert!(!config.validate_auth(Some("Bearer wrong-secret")));
    assert!(!config.validate_auth(None));

    // Clean up environment
    env::remove_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_INTERVAL");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_TIMEOUT");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_FAILURE_THRESHOLD");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE");
    env::remove_var("INFERNO_SERVICE_DISCOVERY_SHARED_SECRET");
}

#[tokio::test]
async fn test_configuration_validation_errors() {
    use std::env;

    // Clean up any environment variables that might interfere with our tests
    let vars_to_clean = [
        "INFERNO_SERVICE_DISCOVERY_AUTH_MODE",
        "INFERNO_SERVICE_DISCOVERY_SHARED_SECRET",
        "INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_INTERVAL",
        "INFERNO_SERVICE_DISCOVERY_ENABLE_LOGGING",
    ];
    for var in &vars_to_clean {
        env::remove_var(var);
    }

    // Test invalid auth mode
    env::set_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE", "invalid_mode");
    let result = ServiceDiscoveryConfig::from_env();
    assert!(result.is_err());
    env::remove_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE");

    // Test shared secret mode without secret - ensure secret is NOT set
    env::remove_var("INFERNO_SERVICE_DISCOVERY_SHARED_SECRET");
    env::set_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE", "shared_secret");
    let result = ServiceDiscoveryConfig::from_env();
    assert!(
        result.is_err(),
        "Should fail when shared_secret auth mode is used without a secret"
    );
    env::remove_var("INFERNO_SERVICE_DISCOVERY_AUTH_MODE");

    // Test invalid numeric values
    env::set_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_INTERVAL", "invalid");
    let result = ServiceDiscoveryConfig::from_env();
    assert!(result.is_err());
    env::remove_var("INFERNO_SERVICE_DISCOVERY_HEALTH_CHECK_INTERVAL");

    // Test invalid boolean values
    env::set_var("INFERNO_SERVICE_DISCOVERY_ENABLE_LOGGING", "maybe");
    let result = ServiceDiscoveryConfig::from_env();
    assert!(result.is_err());
    env::remove_var("INFERNO_SERVICE_DISCOVERY_ENABLE_LOGGING");
}

#[tokio::test]
async fn test_health_checker_functionality() {
    let health_checker = HttpHealthChecker::new(Duration::from_secs(1));

    let node = NodeInfo::new(
        "test-node".to_string(),
        "127.0.0.1:9999".to_string(), // Use a port that's likely not running
        9090,
        NodeType::Backend,
    );

    // Test health check against non-existent endpoint (should fail)
    let result = health_checker.check_health(&node).await;
    assert!(result.is_ok()); // The method should return Ok(HealthCheckResult)

    let health_result = result.unwrap();
    assert!(!health_result.is_healthy()); // But the result should indicate unhealthy
    assert!(health_result.error_message().is_some());
}

#[tokio::test]
#[ignore] // Disabled - consensus removed in favor of SWIM
#[allow(clippy::useless_vec)]
async fn test_peer_discovery_and_consensus() {
    let _discovery = ServiceDiscovery::new();

    // Create mock peer responses for consensus testing
    use crate::service_discovery::{PeerInfo, RegistrationResponse};
    use std::time::SystemTime;

    let peer1 = PeerInfo {
        id: "backend-1".to_string(),
        address: "10.0.1.5:3000".to_string(),
        metrics_port: 9090,
        node_type: NodeType::Backend,
        is_load_balancer: false,
        last_updated: SystemTime::now(),
    };

    let peer2 = PeerInfo {
        id: "backend-2".to_string(),
        address: "10.0.1.6:3000".to_string(),
        metrics_port: 9090,
        node_type: NodeType::Backend,
        is_load_balancer: false,
        last_updated: SystemTime::now(),
    };

    let _responses = vec![
        RegistrationResponse {
            status: "registered".to_string(),
            message: Some("Successfully registered".to_string()),
            peers: vec![peer1.clone(), peer2.clone()],
        },
        RegistrationResponse {
            status: "registered".to_string(),
            message: Some("Successfully registered".to_string()),
            peers: vec![peer1.clone(), peer2.clone()], // Same peers = consensus
        },
    ];

    // Disabled - consensus functionality removed
}

#[tokio::test]
async fn test_backend_removal_and_recovery() {
    let discovery = ServiceDiscovery::new();

    let backend = BackendRegistration {
        id: "test-backend".to_string(),
        address: "10.0.1.10:3000".to_string(),
        metrics_port: 9090,
    };

    // Register backend
    discovery.register_backend(backend).await.unwrap();
    assert_eq!(discovery.backend_count().await, 1);

    // Remove backend
    discovery.remove_backend("test-backend").await.unwrap();
    assert_eq!(discovery.backend_count().await, 0);

    // Test removing non-existent backend (should not error)
    discovery.remove_backend("non-existent").await.unwrap();
}

#[test]
fn test_configuration_default_values() {
    let config = ServiceDiscoveryConfig::default();

    assert_eq!(config.health_check_interval, Duration::from_secs(5));
    assert_eq!(config.health_check_timeout, Duration::from_secs(2));
    assert_eq!(config.failure_threshold, 3);
    assert_eq!(config.recovery_threshold, 2);
    assert_eq!(config.registration_timeout, Duration::from_secs(30));
    assert!(!config.enable_health_check_logging);
    assert_eq!(config.auth_mode, AuthMode::Open);
    assert_eq!(config.shared_secret, None);
}

#[test]
fn test_configuration_validation() {
    // Valid default configuration
    let config = ServiceDiscoveryConfig::default();
    assert!(config.validate().is_ok());

    // Valid shared secret configuration
    let config = ServiceDiscoveryConfig::with_shared_secret("test-secret".to_string());
    assert!(config.validate().is_ok());

    // Invalid configuration - shared secret mode without secret
    let config = ServiceDiscoveryConfig {
        auth_mode: AuthMode::SharedSecret,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    // Invalid configuration - empty shared secret
    let config_empty_secret = ServiceDiscoveryConfig {
        auth_mode: AuthMode::SharedSecret,
        shared_secret: Some("".to_string()),
        ..Default::default()
    };
    assert!(config_empty_secret.validate().is_err());
}
