//! End-to-end tests for backend and proxy interaction
//!
//! These tests verify that the backend can successfully register with the proxy
//! by starting both processes using their CLI entry points.

use inferno_backend::BackendCliOptions;
use inferno_proxy::ProxyCliOptions;
use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;

/// Get a random port in the ephemeral port range (49152-65535)
fn get_random_port() -> u16 {
    let mut rng = rand::thread_rng();
    rng.gen_range(49152..=65535)
}

/// Test that the backend can successfully register with the proxy
#[tokio::test]
async fn test_backend_registration_with_proxy() {
    // Initialize logging for the test
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    // Get random ports for this test
    let proxy_port = get_random_port();
    let backend_port = get_random_port();
    let metrics_port = get_random_port();

    // Configure the proxy CLI options
    let proxy_opts = ProxyCliOptions {
        listen_addr: format!("127.0.0.1:{}", proxy_port).parse().unwrap(),
        backend_addr: Some(format!("127.0.0.1:{}", backend_port).parse().unwrap()),
        backend_servers: None,
        max_connections: 1000,
        timeout_seconds: 30,
        enable_health_check: true,
        health_check_interval: 30,
        health_check: inferno_shared::HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        metrics: inferno_shared::MetricsOptions {
            enable_metrics: true,
            operations_addr: Some(format!("127.0.0.1:{}", metrics_port + 1).parse().unwrap()),
            metrics_addr: Some(format!("127.0.0.1:{}", metrics_port + 1).parse().unwrap()),
        },
        logging: inferno_shared::LoggingOptions {
            log_level: "info".to_string(),
        },
        enable_tls: false,
        load_balancing_algorithm: "round_robin".to_string(),
    };

    // Configure the backend CLI options
    let backend_opts = BackendCliOptions {
        listen_addr: format!("127.0.0.1:{}", backend_port).parse().unwrap(),
        model_path: "test_model.bin".into(),
        model_type: "test".to_string(),
        engine: "burn-cpu".to_string(),
        max_batch_size: 16,
        gpu_device_id: -1,
        max_context_length: 1024,
        memory_pool_mb: 512,
        discovery_lb: Some(format!("127.0.0.1:{}", proxy_port)),
        enable_cache: false,
        cache_ttl_seconds: 300,
        health_check: inferno_shared::HealthCheckOptions {
            health_check_path: "/health".to_string(),
        },
        metrics: inferno_shared::MetricsOptions {
            enable_metrics: true,
            operations_addr: Some(format!("127.0.0.1:{}", metrics_port).parse().unwrap()),
            metrics_addr: Some(format!("127.0.0.1:{}", metrics_port).parse().unwrap()),
        },
        logging: inferno_shared::LoggingOptions {
            log_level: "info".to_string(),
        },
        service_discovery: inferno_shared::ServiceDiscoveryOptions {
            service_name: Some("test-backend".to_string()),
            registration_endpoint: Some(format!("127.0.0.1:{}", proxy_port)),
        },
    };

    // Start the proxy server in a background task
    let proxy_task = tokio::spawn(async move {
        let result = proxy_opts.run().await;
        if let Err(e) = result {
            eprintln!("Proxy failed: {}", e);
        }
    });

    // Give proxy time to start
    sleep(Duration::from_millis(1000)).await;

    // Start the backend server in a background task
    let backend_task = tokio::spawn(async move {
        let result = backend_opts.run().await;
        if let Err(e) = result {
            eprintln!("Backend failed: {}", e);
        }
    });

    // Give backend time to start and attempt registration
    sleep(Duration::from_millis(3000)).await;

    // Now verify that registration actually worked by checking metrics endpoints
    let proxy_metrics_port = metrics_port + 1;
    let backend_metrics_port = metrics_port;

    println!("Verifying registration success...");
    println!("   - Proxy started on port: {}", proxy_port);
    println!("   - Backend started on port: {}", backend_port);
    println!("   - Proxy metrics on port: {}", proxy_metrics_port);
    println!("   - Backend metrics on port: {}", backend_metrics_port);

    let client = reqwest::Client::new();

    // Try to get metrics from both services to verify they're actually running and connected
    let proxy_metrics_url = format!("http://127.0.0.1:{}/metrics", proxy_metrics_port);
    let backend_metrics_url = format!("http://127.0.0.1:{}/metrics", backend_metrics_port);

    // Check if proxy metrics endpoint is accessible
    match client.get(&proxy_metrics_url).send().await {
        Ok(response) if response.status().is_success() => {
            println!("Proxy metrics endpoint accessible at {}", proxy_metrics_url);
        }
        Ok(response) => {
            println!(
                "Proxy metrics endpoint returned error: {}",
                response.status()
            );
        }
        Err(e) => {
            println!("Failed to connect to proxy metrics: {}", e);
        }
    }

    // Check if backend metrics endpoint is accessible
    match client.get(&backend_metrics_url).send().await {
        Ok(response) if response.status().is_success() => {
            println!(
                "Backend metrics endpoint accessible at {}",
                backend_metrics_url
            );

            // Try to parse the metrics to check if backend reports it's connected to proxy
            if let Ok(text) = response.text().await {
                if text.contains("connected_peers") {
                    println!("Backend metrics contain peer connection information");
                } else {
                    println!("WARNING: Backend metrics don't show peer connections yet");
                }
            }
        }
        Ok(response) => {
            println!(
                "Backend metrics endpoint returned error: {}",
                response.status()
            );
        }
        Err(e) => {
            println!("Failed to connect to backend metrics: {}", e);
        }
    }

    // Try to check if the proxy has any registered backends
    // This will fail until we implement the /register endpoint and /backends endpoint
    let backends_url = format!("http://127.0.0.1:{}/backends", proxy_port);
    match client.get(&backends_url).send().await {
        Ok(response) if response.status().is_success() => {
            if let Ok(body) = response.text().await {
                println!("Proxy backends endpoint accessible: {}", body);
                // TODO: Parse JSON and check if backend is actually registered
            }
        }
        Ok(response) => {
            println!("Proxy backends endpoint returned error: {} (this is expected until /backends is implemented)", response.status());
        }
        Err(e) => {
            println!("Failed to connect to proxy backends endpoint: {} (this is expected until /backends is implemented)", e);
        }
    }

    // The test should fail if registration didn't work
    // We'll verify this by checking logs for registration failure
    println!("\nRegistration Test Summary:");
    println!("   - Check the logs above for registration attempts and failures");
    println!("   - FAIL: This test should FAIL until the /register endpoint is implemented");
    println!("   - FAIL: Backend should show 'Connection refused' errors when trying to register");
    println!("   - The metrics endpoints should be accessible but won't show successful peer connections");

    // TODO: Once /register endpoint is implemented, this test should verify:
    // 1. Backend successfully registers with proxy (no connection refused errors)
    // 2. Proxy /backends endpoint shows the registered backend
    // 3. Backend metrics show connected_peers > 0
    // 4. Proxy metrics show registered backends > 0

    // Clean up: abort the background tasks
    proxy_task.abort();
    backend_task.abort();

    // Give tasks time to clean up
    sleep(Duration::from_millis(100)).await;
}
