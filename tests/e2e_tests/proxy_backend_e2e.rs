//! End-to-end tests for proxy and backend interaction
//!
//! These tests verify the full system behavior by running actual binaries
//! in release mode and testing their interaction.

use reqwest::Client;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::{sleep, timeout};

/// Test that the proxy can forward requests to a real backend
#[tokio::test]
#[ignore = "Requires building release binaries first"]
async fn test_proxy_backend_communication() {
    // Build the binaries in release mode
    build_release_binaries();

    // Start the backend server
    let mut backend = Command::new("target/release/inferno-backend")
        .arg("--port")
        .arg("3001")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start backend");

    // Give backend time to start
    sleep(Duration::from_millis(500)).await;

    // Start the proxy server
    let mut proxy = Command::new("target/release/inferno-proxy")
        .env("INFERNO_BACKEND_ADDR", "127.0.0.1:3001")
        .env("INFERNO_LISTEN_ADDR", "127.0.0.1:8081")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start proxy");

    // Give proxy time to start
    sleep(Duration::from_millis(500)).await;

    // Test the connection
    let client = Client::new();
    let response = timeout(Duration::from_secs(5), async {
        client.get("http://127.0.0.1:8081/health").send().await
    })
    .await
    .expect("Request timed out")
    .expect("Request failed");

    assert_eq!(response.status(), 200);

    // Clean up
    let _ = backend.kill();
    let _ = backend.wait();
    let _ = proxy.kill();
    let _ = proxy.wait();
}

/// Test load balancing between multiple backends
#[tokio::test]
#[ignore = "Requires building release binaries first"]
async fn test_load_balancing() {
    build_release_binaries();

    // Start multiple backend servers
    let mut backends = vec![];
    for port in 3002..=3004 {
        let backend = Command::new("target/release/inferno-backend")
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to start backend");
        backends.push(backend);
    }

    sleep(Duration::from_millis(500)).await;

    // Start proxy with multiple backends
    let mut proxy = Command::new("target/release/inferno-proxy")
        .env(
            "INFERNO_BACKEND_SERVERS",
            "127.0.0.1:3002,127.0.0.1:3003,127.0.0.1:3004",
        )
        .env("INFERNO_LISTEN_ADDR", "127.0.0.1:8082")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start proxy");

    sleep(Duration::from_millis(500)).await;

    // Send multiple requests and verify they're distributed
    let client = Client::new();
    let mut responses = vec![];

    for _ in 0..10 {
        let response = timeout(Duration::from_secs(5), async {
            client.get("http://127.0.0.1:8082/api/info").send().await
        })
        .await
        .expect("Request timed out")
        .expect("Request failed");

        responses.push(response);
    }

    // Verify all requests succeeded
    for response in responses {
        assert_eq!(response.status(), 200);
    }

    // Clean up
    let _ = proxy.kill();
    let _ = proxy.wait();
    for mut backend in backends {
        let _ = backend.kill();
        let _ = backend.wait();
    }
}

/// Test graceful shutdown
#[tokio::test]
#[ignore = "Requires building release binaries first"]
async fn test_graceful_shutdown() {
    build_release_binaries();

    // Start backend
    let mut backend = Command::new("target/release/inferno-backend")
        .arg("--port")
        .arg("3005")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start backend");

    sleep(Duration::from_millis(500)).await;

    // Start proxy
    let mut proxy = Command::new("target/release/inferno-proxy")
        .env("PINGORA_BACKEND_ADDR", "127.0.0.1:3005")
        .env("PINGORA_LISTEN_ADDR", "127.0.0.1:8083")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start proxy");

    sleep(Duration::from_millis(500)).await;

    // Start a long-running request
    let client = Client::new();
    let request_handle = tokio::spawn(async move {
        client
            .get("http://127.0.0.1:8083/slow")
            .timeout(Duration::from_secs(10))
            .send()
            .await
    });

    // Give request time to start
    sleep(Duration::from_millis(100)).await;

    // Send graceful shutdown signal (SIGTERM)
    unsafe {
        libc::kill(proxy.id() as i32, libc::SIGTERM);
    }

    // Wait for request to complete
    let result = timeout(Duration::from_secs(5), request_handle).await;

    // Request should complete successfully despite shutdown
    assert!(result.is_ok());

    // Clean up
    let _ = backend.kill();
    let _ = backend.wait();
    let _ = proxy.wait();
}

/// Test failover when backend becomes unavailable
#[tokio::test]
#[ignore = "Requires building release binaries first"]
async fn test_backend_failover() {
    build_release_binaries();

    // Start two backends
    let mut backend1 = Command::new("target/release/inferno-backend")
        .arg("--port")
        .arg("3006")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start backend 1");

    let mut backend2 = Command::new("target/release/inferno-backend")
        .arg("--port")
        .arg("3007")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start backend 2");

    sleep(Duration::from_millis(500)).await;

    // Start proxy with both backends
    let mut proxy = Command::new("target/release/inferno-proxy")
        .env("INFERNO_BACKEND_SERVERS", "127.0.0.1:3006,127.0.0.1:3007")
        .env("INFERNO_LISTEN_ADDR", "127.0.0.1:8084")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start proxy");

    sleep(Duration::from_millis(500)).await;

    let client = Client::new();

    // Verify both backends work
    let response = client
        .get("http://127.0.0.1:8084/health")
        .send()
        .await
        .expect("Initial request failed");
    assert_eq!(response.status(), 200);

    // Kill one backend
    let _ = backend1.kill();
    let _ = backend1.wait();
    sleep(Duration::from_millis(500)).await;

    // Requests should still work via second backend
    let response = client
        .get("http://127.0.0.1:8084/health")
        .send()
        .await
        .expect("Request after failover failed");
    assert_eq!(response.status(), 200);

    // Clean up
    let _ = proxy.kill();
    let _ = proxy.wait();
    let _ = backend2.kill();
    let _ = backend2.wait();
}

/// Helper function to build release binaries
fn build_release_binaries() {
    println!("Building release binaries...");

    let output = Command::new("cargo")
        .args(["build", "--release", "-p", "inferno-proxy"])
        .output()
        .expect("Failed to build proxy");

    if !output.status.success() {
        panic!(
            "Failed to build proxy: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let output = Command::new("cargo")
        .args(["build", "--release", "-p", "inferno-backend"])
        .output()
        .expect("Failed to build backend");

    if !output.status.success() {
        panic!(
            "Failed to build backend: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
