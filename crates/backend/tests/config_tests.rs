//! Configuration Tests
//!
//! Tests for backend configuration loading and validation.

use inferno_backend::BackendConfig;

#[test]
fn test_configuration_from_env_defaults() {
    // Falls back to defaults when no env vars are set
    let config = BackendConfig::from_env().unwrap();
    assert_eq!(config.service_name, "inferno-backend");
}

#[test]
fn test_default_configuration() {
    let config = BackendConfig::default();
    assert_eq!(config.listen_addr.port(), 3000);
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.service_name, "inferno-backend");
    assert!(config.enable_metrics);
}
