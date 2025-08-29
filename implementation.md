# Inferno Implementation Details

## Architecture Overview

Inferno uses a **dual-server architecture** that separates application concerns from operational concerns. This design provides clean separation of responsibilities and allows for independent scaling and monitoring.

## Component Architecture

### Proxy Server

The proxy server runs **two independent HTTP servers**:

1. **Primary Proxy Service (Port 8080)**
   - **Framework**: Cloudflare Pingora 
   - **Purpose**: High-performance HTTP proxy for request forwarding
   - **Responsibilities**:
     - Request routing and load balancing
     - Backend connection pooling
     - Request/response transformation
     - Traffic forwarding only
   - **No operational endpoints** - all operations handled by operations server

2. **Operations Server (Port 6100)**
   - **Framework**: Hyper (shared component)
   - **Purpose**: Monitoring, health checks, and service discovery
   - **Endpoints**:
     - `GET /metrics` - Prometheus metrics
     - `GET /health` - Health check
     - `POST /registration` - Backend registration

### Backend Server

The backend server also runs **two independent HTTP servers**:

1. **Inference Service (Port 8080)**
   - **Framework**: Hyper
   - **Purpose**: AI model inference requests
   - **Responsibilities**:
     - Model loading and inference
     - Request processing
     - Response generation
     - AI-specific logic

2. **Operations Server (Port 6100)**
   - **Framework**: Hyper (shared component) 
   - **Purpose**: Same as proxy operations server
   - **Endpoints**: Same as proxy operations server

## Operations Server Implementation

### Shared Component Design

The operations server is implemented as a **shared Rust component** in `crates/shared/src/operations_server.rs`:

```rust
pub struct OperationsServer {
    metrics: Arc<MetricsCollector>,
    bind_addr: SocketAddr,
    service_name: String,
    version: String,
    connected_peers: Arc<AtomicU32>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}
```

### Key Features

- **Framework**: Built on Hyper for HTTP server functionality
- **Port Standardization**: All operations servers use port 6100
- **Shared Codebase**: Same implementation used by proxy and backend
- **Backward Compatibility**: Type alias `MetricsServer = OperationsServer`
- **Performance**: < 1ms latency, > 10,000 RPS capability

### Endpoints

#### `GET /metrics`
- **Format**: Prometheus-compatible JSON
- **Content**: NodeVitals specification
- **Data**: Service metrics, uptime, request counts, performance stats
- **Usage**: Monitoring systems, service discovery

#### `GET /health`
- **Format**: Plain text
- **Response**: `healthy` (200 OK) or error status
- **Purpose**: Load balancer health checks, monitoring
- **Timeout**: < 10μs response time

#### `POST /registration`
- **Format**: JSON payload
- **Purpose**: Backend service registration with proxy
- **Schema**:
  ```json
  {
    "id": "backend-id",
    "address": "127.0.0.1:8080",
    "metrics_port": 6100
  }
  ```
- **Response**: `{"status": "registered"}` or error

## Service Discovery Flow

### Registration Process

1. **Backend Startup**: Backend starts both inference server (8080) and operations server (6100)
2. **Proxy Discovery**: Backend discovers proxy address from configuration
3. **Registration Request**: Backend sends POST to `http://proxy:6100/registration`
4. **Proxy Response**: Proxy operations server acknowledges registration
5. **Health Monitoring**: Proxy begins health checking backend operations server

### Communication Pattern

```
Backend :8080 (Inference) ←─────── Client Requests ←─────── Proxy :8080 (Pingora)
   │                                                            │
   │                                                            │
Backend :6100 (Operations) ──── Registration ──────────→ Proxy :6100 (Operations)
   │                                                            │
   │                              Health Checks                │
   └──────────── /health ←──────────────────────────────────────┘
```

## Technical Benefits

### Separation of Concerns
- **Application Logic**: Focused on core functionality (proxy/inference)
- **Operational Logic**: Isolated to operations server
- **Clean Interfaces**: Well-defined API boundaries

### Performance Optimization
- **Specialized Servers**: Each server optimized for its purpose
- **Reduced Overhead**: No operational endpoints on high-traffic paths
- **Independent Scaling**: Can scale operations vs application servers differently

### Operational Excellence
- **Standardized Monitoring**: Same operations endpoints across all components
- **Centralized Health Checks**: Consistent health checking protocol
- **Service Discovery**: Built-in registration and discovery mechanism

## Configuration

### Environment Variables

Both proxy and backend support operations server configuration:

```bash
# Operations server address (default: 127.0.0.1:6100)
export INFERNO_OPERATIONS_ADDR="0.0.0.0:6100"

# Enable metrics collection (default: true)
export INFERNO_ENABLE_METRICS="true"
```

### Code Configuration

```rust
use inferno_shared::OperationsServer;

// Create operations server
let operations_server = OperationsServer::new(
    metrics,
    "127.0.0.1:6100".parse().unwrap()
);

// Start server (non-blocking)
operations_server.start().await?;
```

## Testing

### Unit Tests
- **Operations Server**: 7 comprehensive unit tests
- **HTTP Endpoints**: All endpoints tested with mock data
- **Error Handling**: Invalid requests, network failures
- **Performance**: Response time and throughput validation

### Integration Tests
- **End-to-End**: Full service discovery workflow
- **Multi-Backend**: Multiple backend registration
- **Health Checking**: Backend failure and recovery scenarios
- **Load Testing**: High-throughput request handling

## Migration Notes

### From Previous Architecture
- **Renamed**: `MetricsServer` → `OperationsServer`
- **Port Change**: 9090/9091 → 6100 (standardized)
- **Endpoint Addition**: Added `/registration` endpoint
- **Proxy Simplification**: Removed operational endpoints from main proxy service

### Backward Compatibility
- **Type Alias**: `pub type MetricsServer = OperationsServer;`
- **Method Compatibility**: Same public API surface
- **Configuration**: Supports both old and new environment variables

This architecture provides a robust foundation for distributed AI inference with proper separation of concerns, standardized operations, and high-performance characteristics.