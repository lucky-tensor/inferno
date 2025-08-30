use inferno_governator::{GovernatorCliOptions, GovernatorConfig};
use inferno_shared::{HealthCheckOptions, LoggingOptions, MetricsOptions};
use std::net::SocketAddr;

#[test]
fn test_config_default_values() {
    let config = GovernatorConfig::default();

    assert_eq!(
        config.listen_addr,
        "127.0.0.1:4000".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(config.database_url, "sqlite://governator.db");
    assert_eq!(config.providers, vec!["aws", "gcp"]);
    assert_eq!(config.cost_alert_threshold, 1000.0);
    assert_eq!(config.optimization_interval, 300);
    assert_eq!(config.max_instances_per_region, 100);
    assert!(config.enable_autoscaling);
    assert_eq!(config.min_instances, 1);
    assert_eq!(config.max_instances, 100);
    assert_eq!(config.scale_up_threshold, 80.0);
    assert_eq!(config.scale_down_threshold, 20.0);
    assert!(config.enable_metrics);
    assert_eq!(
        config.metrics_addr,
        "127.0.0.1:9092".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(config.health_check_path, "/health");
    assert_eq!(config.budget_mode, "warn");
    assert_eq!(config.monthly_budget, 10000.0);
}

#[test]
fn test_config_from_env() {
    let config = GovernatorConfig::from_env().unwrap();

    // Should return default values since environment variables aren't set
    assert_eq!(config, GovernatorConfig::default());
}

#[test]
fn test_cli_to_config_conversion() {
    let cli = GovernatorCliOptions {
        listen_addr: "127.0.0.1:5000".parse().unwrap(),
        database_url: "postgresql://test:test@localhost/test".to_string(),
        providers: "aws,gcp,azure".to_string(),
        cost_alert_threshold: 2000.0,
        optimization_interval: 600,
        max_instances_per_region: 50,
        enable_autoscaling: false,
        min_instances: 2,
        max_instances: 20,
        scale_up_threshold: 70.0,
        scale_down_threshold: 30.0,
        metrics_endpoint: "prometheus:9090".to_string(),
        alert_webhook: Some("http://localhost:8080/webhook".to_string()),
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: None,
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        budget_mode: "enforce".to_string(),
        monthly_budget: 5000.0,
    };

    let config = cli.to_config().unwrap();

    assert_eq!(
        config.listen_addr,
        "127.0.0.1:5000".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(config.database_url, "postgresql://test:test@localhost/test");
    assert_eq!(config.providers, vec!["aws", "gcp", "azure"]);
    assert_eq!(config.cost_alert_threshold, 2000.0);
    assert_eq!(config.optimization_interval, 600);
    assert_eq!(config.max_instances_per_region, 50);
    assert!(!config.enable_autoscaling);
    assert_eq!(config.min_instances, 2);
    assert_eq!(config.max_instances, 20);
    assert_eq!(config.scale_up_threshold, 70.0);
    assert_eq!(config.scale_down_threshold, 30.0);
    assert_eq!(config.budget_mode, "enforce");
    assert_eq!(config.monthly_budget, 5000.0);
}

#[test]
fn test_providers_parsing_with_whitespace() {
    let cli = GovernatorCliOptions {
        listen_addr: "127.0.0.1:4000".parse().unwrap(),
        database_url: "sqlite://test.db".to_string(),
        providers: " aws , gcp , azure ".to_string(),
        cost_alert_threshold: 1000.0,
        optimization_interval: 300,
        max_instances_per_region: 100,
        enable_autoscaling: true,
        min_instances: 1,
        max_instances: 100,
        scale_up_threshold: 80.0,
        scale_down_threshold: 20.0,
        metrics_endpoint: "prometheus:9090".to_string(),
        alert_webhook: None,
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: None,
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        budget_mode: "warn".to_string(),
        monthly_budget: 10000.0,
    };

    let config = cli.to_config().unwrap();
    assert_eq!(config.providers, vec!["aws", "gcp", "azure"]);
}

#[test]
fn test_invalid_socket_addr_handling() {
    let cli = GovernatorCliOptions {
        listen_addr: "127.0.0.1:4000".parse().unwrap(),
        database_url: "sqlite://test.db".to_string(),
        providers: "aws".to_string(),
        cost_alert_threshold: 1000.0,
        optimization_interval: 300,
        max_instances_per_region: 100,
        enable_autoscaling: true,
        min_instances: 1,
        max_instances: 100,
        scale_up_threshold: 80.0,
        scale_down_threshold: 20.0,
        metrics_endpoint: "invalid-endpoint".to_string(),
        alert_webhook: Some("invalid-webhook".to_string()),
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: None,
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        budget_mode: "warn".to_string(),
        monthly_budget: 10000.0,
    };

    let config = cli.to_config().unwrap();

    // Invalid endpoints should result in None values
    assert!(config.metrics_endpoint.is_none());
    assert!(config.alert_webhook.is_none());
}

#[tokio::test]
async fn test_cli_run_method() {
    let cli = GovernatorCliOptions {
        listen_addr: "127.0.0.1:4000".parse().unwrap(),
        database_url: "sqlite://test.db".to_string(),
        providers: "aws,gcp".to_string(),
        cost_alert_threshold: 1000.0,
        optimization_interval: 300,
        max_instances_per_region: 100,
        enable_autoscaling: true,
        min_instances: 1,
        max_instances: 100,
        scale_up_threshold: 80.0,
        scale_down_threshold: 20.0,
        metrics_endpoint: "prometheus:9090".to_string(),
        alert_webhook: None,
        logging: LoggingOptions {
            log_level: "info".to_string(),
        },
        metrics: MetricsOptions {
            enable_metrics: true,
            operations_addr: None,
            metrics_addr: None,
        },
        health_check: HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        budget_mode: "warn".to_string(),
        monthly_budget: 10000.0,
    };

    // Test that run method executes successfully (placeholder implementation)
    let result = cli.run().await;
    assert!(result.is_ok());
}
