# Test Harness Guide

## Testing Strategy Overview

We implement a comprehensive testing pyramid with four levels of testing, from fast unit tests to complete end-to-end scenarios:

1. **Doc Tests** - Documentation examples with private type testing
2. **Module Tests** - Standalone test files within each crate 
3. **Integration Tests** - Cross-component testing with controlled environments
4. **End-to-End Tests** - Full binary execution with real network interfaces

## Test Classification Decision Matrix

### When to use Doc Tests
✅ **Use Doc Tests when:**
- Testing private types or functions (keep encapsulation)
- Providing usage examples in documentation
- Testing individual functions in isolation
- Validating public API behavior with simple examples
- The test doubles as documentation

❌ **Don't use Doc Tests when:**
- Testing complex multi-step workflows
- Requiring extensive setup or teardown
- Testing interactions between multiple components
- Needing async test frameworks or serial execution

### When to use Module Tests (in crate/tests/)
✅ **Use Module Tests when:**
- Testing public functions across module boundaries
- Complex test scenarios with multiple assertions
- Tests require substantial setup/teardown logic
- Testing error conditions and edge cases
- Need test utilities and helper functions
- Tests are specific to one crate's functionality

❌ **Don't use Module Tests when:**
- Testing interactions between different crates
- Requiring network communication or real services
- Testing the full application binary behavior

### When to use Integration Tests (test-suite/)
✅ **Use Integration Tests when:**
- Testing communication between different crates
- Verifying service endpoints work correctly
- Testing with controlled mock environments
- Validating configuration-driven behavior
- Testing load balancing, service discovery protocols
- Need to start/stop services but not full binaries

❌ **Don't use Integration Tests when:**
- Testing private implementation details
- Requiring full binary execution and real network interfaces
- Simple function validation that doesn't cross boundaries

### When to use E2E Tests (test-suite/e2e/)
✅ **Use E2E Tests when:**
- Testing complete user workflows
- Validating real binary execution
- Testing with actual network interfaces and ports
- Verifying system behavior under real conditions
- Testing graceful shutdown, process management
- Validating metrics collection in live environment

❌ **Don't use E2E Tests when:**
- Testing implementation details
- Simple function validation
- Fast feedback is required (E2E tests are slow)

## Testing Pyramid

```
    /\
   /  \    E2E Tests (Slow, High Value)
  /____\
 /      \   Integration Tests (Medium Speed)
/________\
\        /  Module Tests (Fast)
 \______/
  \    /    Unit Tests (Very Fast)
   \__/
```

## Practical Examples from Our Codebase

### Current Test Organization
```
crates/
├── shared/
│   ├── src/
│   │   ├── service_discovery.rs    # Contains doc tests for private HealthCheckResult
│   │   ├── metrics.rs              # Contains doc tests for MetricsCollector
│   │   └── error.rs                # Contains doc tests for error types
│   └── tests/
│       ├── cli_tests.rs            # 5 module tests for CLI parsing
│       ├── error_tests.rs          # 5 module tests for error handling
│       └── metrics_tests.rs        # 10 module tests for metrics collection
├── proxy/
│   └── tests/
│       ├── config_tests.rs         # 14 module tests for configuration
│       ├── error_tests.rs          # 7 module tests for proxy errors
│       ├── integration_tests.rs    # 4 integration tests for components
│       ├── metrics_tests.rs        # 11 module tests for metrics
│       ├── performance_tests.rs    # 6 performance regression tests
│       ├── server_tests.rs         # 7 module tests for server lifecycle
│       └── service_tests.rs        # 3 module tests for proxy service
└── test-suite/
    ├── tests/integration/           # Cross-component integration tests
    │   ├── service_discovery_integration.rs  # 9 service discovery tests
    │   └── peer_manager_integration.rs       # 6 peer management tests
    └── tests/e2e/                   # End-to-end with real processes
        ├── proxy_backend_e2e.rs     # Real binary execution tests
        └── service_discovery_e2e.rs # Process communication tests
```

### Decision Examples

#### ✅ Doc Test Example (Private Type Testing)
```rust
// In src/service_discovery.rs - testing private HealthCheckResult
/// Health check result from monitoring a backend
///
/// # Examples
/// ```
/// use inferno_shared::service_discovery::NodeVitals;
/// 
/// let vitals = NodeVitals {
///     ready: true,
///     requests_in_progress: 5,
///     // ... other fields
/// };
/// assert!(vitals.ready);
/// ```
pub(crate) enum HealthCheckResult {
    Healthy(NodeVitals),
    // ...
}
```

#### ✅ Module Test Example (Complex Setup)
```rust
// In crates/shared/tests/metrics_tests.rs
#[test]
fn test_concurrent_metrics_updates() {
    let collector = Arc::new(MetricsCollector::new());
    let mut handles = vec![];

    // Complex setup with multiple threads
    for _ in 0..10 {
        let collector_clone = Arc::clone(&collector);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                collector_clone.record_request();
                collector_clone.record_response(200);
            }
        });
        handles.push(handle);
    }
    
    // Wait and validate results...
}
```

#### ✅ Integration Test Example (Cross-Component)
```rust
// In test-suite/tests/integration/service_discovery_integration.rs
#[tokio::test]
async fn test_proxy_backend_selection_simulation() {
    let discovery = ServiceDiscovery::new();
    
    // Register multiple backends
    register_backend(&discovery, "backend-1", 3000).await;
    register_backend(&discovery, "backend-2", 3001).await;
    
    // Test selection logic across components
    let selected = select_best_backend(&discovery).await;
    assert!(selected.is_some());
}
```

#### ✅ E2E Test Example (Real Processes) 
```rust
// In test-suite/tests/e2e/proxy_backend_e2e.rs  
#[tokio::test]
async fn test_proxy_backend_communication() {
    // Start real proxy binary
    let proxy_process = Command::new("cargo")
        .args(["run", "--bin", "inferno-proxy"])
        .spawn()
        .expect("Failed to start proxy");
        
    // Start real backend binary
    let backend_process = Command::new("cargo")
        .args(["run", "--bin", "inferno-backend"])
        .spawn()
        .expect("Failed to start backend");
        
    // Test real HTTP communication
    let response = reqwest::get("http://127.0.0.1:8080/test").await?;
    assert_eq!(response.status(), 200);
    
    // Cleanup processes...
}
```

## 1. Doc Tests

### Purpose
Test individual functions and methods in isolation with documentation examples.

### Location
Within module definitions using `///` comments with code blocks.

### Example
```rust
/// Validates proxy configuration settings
///
/// # Example
/// ```
/// use inferno_proxy::ProxyConfig;
///
/// let config = ProxyConfig::default();
/// assert!(config.validate().is_ok());
/// ```
pub fn validate(&self) -> Result<(), ConfigError> {
    // implementation
}
```

### Running Unit Tests
```bash
# Run all doc tests
cargo test --doc

# Run specific module doc tests
cargo test --doc config
```

## 2. Module Tests

### Purpose
Test exported public functions and struct behavior within a single crate using standalone test files.

### Location
`crates/{crate-name}/tests/{module-name}_tests.rs` files

### Current Structure
Based on our implementation, each crate contains focused test files:

```
crates/shared/tests/
├── cli_tests.rs            # CLI parsing and validation (5 tests)
├── error_tests.rs          # Error creation and classification (5 tests)
└── metrics_tests.rs        # Metrics collection and calculations (10 tests)

crates/proxy/tests/
├── config_tests.rs         # Configuration validation (14 tests)
├── error_tests.rs          # Proxy-specific error handling (7 tests)
├── integration_tests.rs    # Component integration (4 tests)
├── metrics_tests.rs        # Proxy metrics functionality (11 tests)
├── performance_tests.rs    # Performance regression tests (6 tests)
├── server_tests.rs         # Server lifecycle management (7 tests)
└── service_tests.rs        # Proxy service functionality (3 tests)
```

### Key Principles
- **One test file per logical module/component**
- **Focus on public API testing within the crate**
- **Include complex scenarios requiring setup/teardown**
- **Test error conditions and edge cases**
- **Keep tests isolated to single crate functionality**

### Example Module Test
```rust
// tests/config_test.rs
use inferno_proxy::{ProxyConfig, ProxyError};
use std::time::Duration;

#[test]
fn test_config_validation_success() {
    let config = ProxyConfig {
        service_port: 8080,
        metrics_port: 9090,
        max_connections: 1000,
        ..Default::default()
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_port_conflict() {
    let config = ProxyConfig {
        service_port: 8080,
        metrics_port: 8080,  // Same port - should fail
        ..Default::default()
    };

    assert!(config.validate().is_err());
}
```

### Running Module Tests
```bash
# Run all module tests
cargo test --tests

# Run specific module test
cargo test --test config_test
```

## 3. Integration Tests

### Purpose
Verify that client-server connections work correctly for different endpoints without running full binaries.

### Test Scenarios
- HTTP client connections to proxy endpoints
- Metrics endpoint responses (`/metrics`, `/telemetry`)
- Service discovery registration endpoints
- Load balancing algorithm behavior
- Configuration loading and validation

### Example Integration Test
```rust
// tests/integration_tests.rs
use tokio::net::TcpListener;
use reqwest::Client;
use pingora_proxy_demo::{ProxyServer, ProxyConfig};

#[tokio::test]
async fn test_metrics_endpoint() {
    // Start server on random port
    let config = ProxyConfig {
        service_port: 0,  // Random port
        metrics_port: 0,  // Random port
        ..Default::default()
    };

    let server = ProxyServer::new(config).await.unwrap();
    let metrics_addr = server.metrics_addr();

    // Start server in background
    tokio::spawn(async move {
        server.run().await.unwrap();
    });

    // Wait for server startup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test /metrics endpoint
    let client = Client::new();
    let response = client
        .get(&format!("http://{}/metrics", metrics_addr))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body.get("ready").is_some());
}
```

### Running Integration Tests
```bash
# Run integration tests
cargo test --test integration_tests

# Run with logging
RUST_LOG=debug cargo test --test integration_tests -- --nocapture
```

## 4. End-to-End (E2E) Tests

### Purpose
Test complete system behavior with actual binary execution and real network interfaces.

### Key Requirements
- **Random Ports**: All services must use random available ports to avoid conflicts
- **Process Management**: Each server runs as subprocess with proper cleanup
- **No Zombie Processes**: All processes terminated when tests complete
- **Real Network**: Full TCP/HTTP communication between components

### E2E Test Scenarios

#### 1. Basic Client-Server Communication
```rust
#[tokio::test]
async fn test_client_load_balancer_communication() {
    let mut test_env = TestEnvironment::new().await;

    // Start load balancer
    let lb = test_env.start_load_balancer().await;

    // Send request to load balancer
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("http://{}/test", lb.service_addr()))
        .send()
        .await
        .unwrap();

    // Should get 503 (no backends yet)
    assert_eq!(response.status(), 503);

    test_env.cleanup().await;
}
```

#### 2. Service Discovery Announcement
```rust
#[tokio::test]
async fn test_backend_service_discovery_announcement() {
    let mut test_env = TestEnvironment::new().await;

    // Start load balancer
    let lb = test_env.start_load_balancer().await;

    // Start backend - should auto-register
    let backend = test_env.start_backend(&[lb.service_addr()]).await;

    // Wait for registration
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check load balancer knows about backend
    let backends = lb.get_backends().await;
    assert_eq!(backends.len(), 1);
    assert_eq!(backends[0], backend.service_addr().to_string());

    test_env.cleanup().await;
}
```

#### 3. Graceful Backend Shutdown
```rust
#[tokio::test]
async fn test_backend_graceful_shutdown() {
    let mut test_env = TestEnvironment::new().await;

    let lb = test_env.start_load_balancer().await;
    let backend = test_env.start_backend(&[lb.service_addr()]).await;

    // Wait for registration
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check backend is registered
    let backends = lb.get_backends().await;
    assert_eq!(backends.len(), 1);

    // Graceful shutdown backend
    backend.graceful_shutdown().await;

    // Backend should set ready=false
    let metrics = backend.get_metrics().await;
    assert!(!metrics.ready);

    // Wait for deregistration
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Load balancer should remove backend
    let backends = lb.get_backends().await;
    assert_eq!(backends.len(), 0);

    test_env.cleanup().await;
}
```

#### 4. Request Routing Through System
```rust
#[tokio::test]
async fn test_request_routing_end_to_end() {
    let mut test_env = TestEnvironment::new().await;

    let lb = test_env.start_load_balancer().await;
    let backend = test_env.start_backend(&[lb.service_addr()]).await;

    // Wait for service discovery
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Send request through load balancer to backend
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("http://{}/api/test", lb.service_addr()))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    // Check backend received request
    let metrics = backend.get_metrics().await;
    assert!(metrics.requests_in_progress >= 0);

    test_env.cleanup().await;
}
```

#### 5. Streaming Response Handling
```rust
#[tokio::test]
async fn test_streaming_response() {
    let mut test_env = TestEnvironment::new().await;

    let lb = test_env.start_load_balancer().await;
    let backend = test_env.start_backend(&[lb.service_addr()]).await;

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Request streaming endpoint
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("http://{}/stream", lb.service_addr()))
        .send()
        .await
        .unwrap();

    // Verify streaming response
    let mut stream = response.bytes_stream();
    let mut chunks = 0;

    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
        chunks += 1;
        if chunks >= 3 { break; } // Test first few chunks
    }

    assert!(chunks >= 3);
    test_env.cleanup().await;
}
```

#### 6. Multiple Backend Load Distribution
```rust
#[tokio::test]
async fn test_multiple_backend_load_distribution() {
    let mut test_env = TestEnvironment::new().await;

    let lb = test_env.start_load_balancer().await;

    // Start 3 backends
    let backend1 = test_env.start_backend(&[lb.service_addr()]).await;
    let backend2 = test_env.start_backend(&[lb.service_addr()]).await;
    let backend3 = test_env.start_backend(&[lb.service_addr()]).await;

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Send 10 requests
    let client = reqwest::Client::new();
    for i in 0..10 {
        let response = client
            .get(&format!("http://{}/test/{}", lb.service_addr(), i))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    // Check that requests were distributed
    let metrics1 = backend1.get_metrics().await;
    let metrics2 = backend2.get_metrics().await;
    let metrics3 = backend3.get_metrics().await;

    let total_requests = metrics1.requests_processed +
                        metrics2.requests_processed +
                        metrics3.requests_processed;
    assert_eq!(total_requests, 10);

    // Each backend should have received some requests (round-robin)
    assert!(metrics1.requests_processed > 0);
    assert!(metrics2.requests_processed > 0);
    assert!(metrics3.requests_processed > 0);

    test_env.cleanup().await;
}
```

#### 7. Health Check Integration
```rust
#[tokio::test]
async fn test_health_check_backend_removal() {
    let mut test_env = TestEnvironment::new().await;

    let lb = test_env.start_load_balancer().await;
    let backend = test_env.start_backend(&[lb.service_addr()]).await;

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify backend is registered
    let backends = lb.get_backends().await;
    assert_eq!(backends.len(), 1);

    // Set backend to not ready
    backend.set_ready(false).await;

    // Wait for health check cycle
    tokio::time::sleep(Duration::from_secs(6)).await;

    // Load balancer should remove unhealthy backend
    let backends = lb.get_backends().await;
    assert_eq!(backends.len(), 0);

    test_env.cleanup().await;
}
```

#### 8. Metrics Collection Validation
```rust
#[tokio::test]
async fn test_metrics_collection() {
    let mut test_env = TestEnvironment::new().await;

    let lb = test_env.start_load_balancer().await;
    let backend = test_env.start_backend(&[lb.service_addr()]).await;

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Send requests to generate metrics
    let client = reqwest::Client::new();
    for _ in 0..5 {
        let _ = client
            .get(&format!("http://{}/test", lb.service_addr()))
            .send()
            .await;
    }

    // Check backend metrics via /metrics endpoint
    let metrics_response = client
        .get(&format!("http://{}/metrics", backend.metrics_addr()))
        .send()
        .await
        .unwrap();

    let metrics: serde_json::Value = metrics_response.json().await.unwrap();

    assert_eq!(metrics["ready"], true);
    assert!(metrics["requests_in_progress"].as_u64().unwrap() >= 0);

    // Check Prometheus format
    let prometheus_response = client
        .get(&format!("http://{}/telemetry", backend.metrics_addr()))
        .send()
        .await
        .unwrap();

    let prometheus_text = prometheus_response.text().await.unwrap();
    assert!(prometheus_text.contains("node_ready"));

    test_env.cleanup().await;
}
```

## Test Environment Infrastructure

### TestEnvironment Struct
```rust
pub struct TestEnvironment {
    processes: Vec<Child>,
    temp_dirs: Vec<TempDir>,
    port_pool: PortPool,
}

impl TestEnvironment {
    pub async fn new() -> Self {
        Self {
            processes: Vec::new(),
            temp_dirs: Vec::new(),
            port_pool: PortPool::new(),
        }
    }

    pub async fn start_load_balancer(&mut self) -> LoadBalancerHandle {
        let service_port = self.port_pool.get_port().await;
        let metrics_port = self.port_pool.get_port().await;

        // Create config file
        let config = format!(r#"
node:
  type: "load_balancer"
network:
  service_port: {}
  metrics_port: {}
"#, service_port, metrics_port);

        let config_file = self.write_temp_config(&config).await;

        // Start process
        let process = Command::new("cargo")
            .args(&["run", "--", "--config", config_file.path()])
            .spawn()
            .expect("Failed to start load balancer");

        self.processes.push(process);

        LoadBalancerHandle::new(service_port, metrics_port)
    }

    pub async fn cleanup(&mut self) {
        // Terminate all processes
        for mut process in self.processes.drain(..) {
            let _ = process.kill();
            let _ = process.wait();
        }

        // Clean up temp files
        self.temp_dirs.clear();
    }
}
```

### Port Management
```rust
pub struct PortPool {
    used_ports: HashSet<u16>,
}

impl PortPool {
    pub async fn get_port(&mut self) -> u16 {
        loop {
            let port = self.random_port();
            if self.is_port_available(port).await && !self.used_ports.contains(&port) {
                self.used_ports.insert(port);
                return port;
            }
        }
    }

    async fn is_port_available(&self, port: u16) -> bool {
        TcpListener::bind(format!("127.0.0.1:{}", port))
            .await
            .is_ok()
    }
}
```

## Running Tests

### Full Test Suite
```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test types
cargo test --doc           # Unit tests
cargo test --tests         # Module tests
cargo test --test e2e_test # E2E tests
```

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      # Unit tests (fast)
      - name: Run unit tests
        run: cargo test --doc

      # Module tests (medium)
      - name: Run module tests
        run: cargo test --tests

      # E2E tests (slow)
      - name: Run E2E tests
        run: cargo test --test e2e_tests
        timeout-minutes: 10
```

### Performance Test Integration
```bash
# Run benchmarks with tests
cargo test --benches

# Generate performance baseline
cargo bench --bench proxy_benchmarks
```

## Test Data and Fixtures

### Mock Backends
```rust
// tests/common/mock_backend.rs
pub struct MockBackend {
    server: HttpServer,
    responses: HashMap<String, (StatusCode, String)>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            server: HttpServer::new(),
            responses: HashMap::new(),
        }
    }

    pub fn with_response<S: Into<String>>(mut self, path: S, status: StatusCode, body: S) -> Self {
        self.responses.insert(path.into(), (status, body.into()));
        self
    }
}
```

### Test Configuration Templates
```rust
// tests/common/config_templates.rs
pub fn minimal_load_balancer_config(service_port: u16, metrics_port: u16) -> String {
    format!(r#"
node:
  type: "load_balancer"
network:
  service_port: {}
  metrics_port: {}
load_balancer:
  algorithm: "round_robin"
"#, service_port, metrics_port)
}

pub fn backend_config(service_port: u16, metrics_port: u16, lb_addrs: &[String]) -> String {
    let lb_list = lb_addrs.join("\"\n    - \"");
    format!(r#"
node:
  type: "backend"
network:
  service_port: {}
  metrics_port: {}
discovery:
  load_balancers:
    - "{}"
"#, service_port, metrics_port, lb_list)
}
```

This comprehensive test harness ensures robust validation of the Inferno Proxy system at all levels, from individual function correctness to complete end-to-end system behavior.
