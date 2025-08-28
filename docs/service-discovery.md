# Minimalist Service Discovery

## Overview

Dead simple service discovery for Pingora proxy. Backends register themselves, load balancers check their metrics port for health and status.

## How It Works

1. **Backend starts** → Announces itself to load balancers
2. **Load balancer** → Adds backend to pool, starts checking metrics port
3. **Backend fails** → Load balancer removes it from pool (metrics port unreachable or ready=false)
4. **Backend recovers** → Re-announces itself

That's it.

## Configuration

```yaml
# For backends
discovery:
  load_balancers: ["lb1:8080", "lb2:8080"]
  announce_interval: "30s"
  metrics_port: 9090

# For load balancers  
health_check:
  interval: "5s"
  timeout: "2s"
  path: "/metrics"
```

## Protocol

### Backend Registration
```
POST /register
{
  "id": "backend-1",
  "address": "10.0.1.5:3000",
  "metrics_port": 9090
}
```

### Metrics Port (replaces health check)
```
GET /metrics → JSON vitals (human-readable)
GET /telemetry → Prometheus metrics
```

### Backend List (internal)
```
GET /backends → ["10.0.1.5:3000", "10.0.1.6:3000"]
```

## Implementation

```rust
use std::collections::HashSet;
use tokio::sync::RwLock;

pub struct ServiceDiscovery {
    backends: RwLock<HashSet<String>>,
}

impl ServiceDiscovery {
    pub async fn register(&self, addr: String) {
        self.backends.write().await.insert(addr);
    }
    
    pub async fn remove(&self, addr: &str) {
        self.backends.write().await.remove(addr);
    }
    
    pub async fn list(&self) -> Vec<String> {
        self.backends.read().await.iter().cloned().collect()
    }
}
```

## Metrics-Based Health Checking

```rust
#[derive(Deserialize)]
struct NodeVitals {
    ready: bool,
    requests_in_progress: u32,
    cpu_usage: f64,
    memory_usage: f64,
    failed_responses: u64,
    connected_peers: u32,
    backoff_requests: u32,
}

async fn health_check_loop(discovery: Arc<ServiceDiscovery>) {
    let client = reqwest::Client::new();
    
    loop {
        let backends = discovery.list().await;
        
        for backend in backends {
            let url = format!("http://{}:9090/metrics", backend.split(':').next().unwrap());
            
            match client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    // Parse vitals from metrics response
                    if let Ok(vitals) = response.json::<NodeVitals>().await {
                        if !vitals.ready {
                            discovery.remove(&backend).await;
                        }
                    }
                }
                _ => {
                    discovery.remove(&backend).await;
                }
            }
        }
        
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
```

## Load Balancing

```rust
impl ProxyService {
    async fn select_backend(&self) -> Option<String> {
        let backends = self.discovery.list().await;
        if backends.is_empty() { return None; }
        
        // Round robin
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % backends.len();
        Some(backends[index].clone())
    }
}
```

## Metrics Port Specification

### GET /metrics (JSON vitals - human readable)
```json
{
  "ready": true,
  "requests_in_progress": 42,
  "cpu_usage": 35.2,
  "memory_usage": 67.8,
  "gpu_usage": 0.0,
  "failed_responses": 15,
  "connected_peers": 3,
  "backoff_requests": 0,
  "uptime_seconds": 86400,
  "version": "1.0.0"
}
```

### GET /telemetry (Prometheus format)
```
# HELP node_ready Whether the node is ready to receive requests
# TYPE node_ready gauge
node_ready 1

# HELP requests_in_progress Current number of requests being processed  
# TYPE requests_in_progress gauge
requests_in_progress 42

# HELP cpu_usage_percent CPU usage percentage
# TYPE cpu_usage_percent gauge
cpu_usage_percent 35.2

# HELP memory_usage_percent Memory usage percentage
# TYPE memory_usage_percent gauge
memory_usage_percent 67.8

# HELP failed_responses_total Total number of failed responses
# TYPE failed_responses_total counter
failed_responses_total 15
```

## Operations

### Starting a backend
```bash
# Backend auto-registers on startup with metrics port
./backend --discovery-lb=lb1:8080,lb2:8080 --metrics-port=9090
```

### Checking status
```bash
# Check backend vitals (human-readable JSON)
curl http://backend1:9090/metrics

# Check load balancer's view
curl http://lb1:8080/backends

# Prometheus metrics for monitoring tools
curl http://backend1:9090/telemetry
```

### Graceful shutdown
```bash
# Backend sets ready=false then de-registers
kill -TERM $backend_pid
```

## That's All

Two simple endpoints: `/metrics` (JSON vitals) and `/telemetry` (Prometheus). Single port to monitor everything.

Works for thousands of backends. Easy to understand. Easy to debug.