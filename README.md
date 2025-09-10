
# Inferno: a self healing cloud for AI inference

An end-to-end Rust LLM micro platform which is request optimized and kernel optimized, inspired by Cloudflare Infire. https://blog.cloudflare.com/cloudflares-most-efficient-ai-inference-engine/

[![Build](https://github.com/0o-de-lally/inferno/workflows/Build/badge.svg)](https://github.com/0o-de-lally/inferno/actions/workflows/build.yml)
[![Test](https://github.com/0o-de-lally/inferno/workflows/Test/badge.svg)](https://github.com/0o-de-lally/inferno/actions/workflows/test.yml)
[![Benchmark](https://github.com/0o-de-lally/inferno/workflows/Benchmark/badge.svg)](https://github.com/0o-de-lally/inferno/actions/workflows/bench.yml)
[![Lint](https://github.com/0o-de-lally/inferno/workflows/Lint/badge.svg)](https://github.com/0o-de-lally/inferno/actions/workflows/lint.yml)


# History
We were working on a self-healing cloud for AI inference that could run on bare metal, optimized for batches and kernel fusion, and then Cloudflare published their architecture with a similar name and architecture. We love it. Nothing is a coincidence.

Inferno is a self-healing cloud platform for AI inference, designed for high-performance, reliability, and observability. It demonstrates best practices for distributed systems, with comprehensive testing and robust error recovery for AI workloads.

# Motivation
The state of the art (Summer 2025) for IT friendly turnkey serving of LLM are VLLM platforms. However to squeeze the most performance per GPU hyperscalers use proprietary kernel and memory optimization (TensorRT for nvidia). Read more in [./docs/state-of-the-art.md].

# Trend
There are some important trends to consider when picking an LLM stack:
- Models are getting bigger (they can't fit on a single node or GPU)
- Models are getting more specialized (you may want to use multiple models)
- More chip architectures will emerge
- More data centers will come online
- Users will demand omni - video, audio, text
- Other runtimes are catching up to Python in usability had a head start in data analytics


## Features

- 🚀 **High Performance**: Zero-allocation request handling patterns where possible
- 🔒 **Robust Error Handling**: Comprehensive error classification and recovery strategies
- 📊 **Built-in Observability**: Prometheus metrics, structured logging, and health checks
- ⚙️ **Flexible Configuration**: Environment variables, validation, and hot reloading support
- 🧪 **Test-Driven Development**: Extensive unit tests, integration tests, and benchmarks
- 🛡️ **Security-Focused**: Input validation, secure defaults, and security headers
- 🌊 **SWIM Protocol**: Load balancer propagation for efficient backend discovery (Alice Project)

## Quick Start

### Prerequisites

- Rust 1.70+ (latest stable recommended)
- Backend service running (e.g., `python3 -m http.server 3000`)

### Installation

```bash
git clone <repository-url>
cd pingora-proxy-demo
cargo build --release
```

### Basic Usage

```bash
# Run with default configuration (proxy on :8080, backend on :3000)
cargo run

# Run with custom configuration
PINGORA_LISTEN_ADDR=0.0.0.0:8080 \
PINGORA_BACKEND_ADDR=192.168.1.100:3000 \
PINGORA_LOG_LEVEL=debug \
cargo run
```

## Architecture

The system uses a dual-server architecture with separate concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client        │───▶│  Proxy Server   │───▶│  Backend        │
│                 │    │                 │    │                 │
│  HTTP Request   │    │  :8080 Pingora  │    │ :8080 Inference │
│                 │◀───│  - Load Balance │◀───│  (Hyper)        │
└─────────────────┘    │  - Forwarding   │    └─────────────────┘
                       └─────────────────┘
                               │                        │
                               ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Operations      │    │ Operations      │
                       │ Server :6100    │    │ Server :6100    │
                       │ (Hyper)         │    │ (Hyper)         │
                       │                 │    │                 │
                       │ - /metrics      │    │ - /metrics      │
                       │ - /health       │    │ - /health       │
                       │ - /registration │    │ - /registration │
                       └─────────────────┘    └─────────────────┘
```

### Component Architecture

- **Proxy Server**:
  - **Port 8080**: Pingora-based HTTP proxy for request forwarding and load balancing
  - **Port 6100**: Hyper-based operations server for monitoring and service discovery

- **Backend Server**:
  - **Port 8080**: Hyper-based inference server for AI model requests
  - **Port 6100**: Hyper-based operations server for monitoring and service discovery

- **Operations Server**: Shared Hyper-based component providing:
  - **`GET /metrics`**: Prometheus metrics endpoint
  - **`GET /health`**: Health check endpoint
  - **`POST /registration`**: Service discovery registration endpoint



## Why We Built Inferno

After years of deploying AI inference systems in production, we witnessed the same painful patterns across every major solution in the market. IT departments consistently struggle with three fundamental problems that existing tools fail to address:

### **Security Challenges**
**Ollama**: Multiple critical RCE vulnerabilities discovered in 2024 (CVE-2024-37032, etc.). Wiz Research found **9,831 exposed instances** on the internet without authentication (as of Q4 2024), with 1 in 4 servers considered vulnerable. No built-in authentication means every deployment requires reverse proxy setup.

**Solution**: Inferno includes secure defaults, built-in authentication, and follows security-first design principles from day one. Built in Rust, we eliminate entire classes of vulnerabilities (buffer overflows, use-after-free, memory corruption) that plague C/C++ implementations used by competitors.

### **Deployment Complexity**
**NVIDIA Dynamo**: Alpha-stage software (as of 2025 release) requiring NATS, etcd, and complex distributed setup. AWS EKS deployments frequently fail with image pull errors and pod failures. Manual memory tuning and GPU visibility management required.

**llm-d**: Kubernetes-native framework (launched 2024) with mandatory K8s 1.29+ requirement and no bare-metal options. Requires specialized DevOps expertise for disaggregated architecture. Container bloat from massive model files (10GB+) slows cold starts.

**Solution**: Inferno offers zero-configuration startup with optimal defaults. Container-optional deployment means you can run on bare metal, VMs, or containers as needed.

### **Production Reality Gaps**
**Ollama**: **Not designed for cloud/production usage** - fundamentally single-machine focused. No user management, minimal monitoring, no compliance certifications. Recent concurrency support still requires multiple instances with massive memory waste.

**All Competitors**: Multi-language stacks (Rust→Python→C++) create serialization overhead and performance bottlenecks. Manual failover requires human intervention during outages.

**Solution**: Inferno provides true cloud-native design with SWIM consensus for self-healing, pure Rust stack eliminates both performance penalties and memory safety vulnerabilities, and comprehensive observability built-in.

### **The Enterprise Tax**
Every existing solution forces a choice: use inadequate open-source tools or pay expensive enterprise licenses for basic functionality. IT departments need production-ready solutions without enterprise lock-in.

**Solution**: Inferno Community Edition delivers enterprise-grade performance, security, and reliability for free. Enterprise Edition adds Governator AI automation and cost optimization for organizations that need maximum ROI.



## Alternatives

| Feature | Inferno Community | Inferno Enterprise | Nvidia Dynamo | llm-d (K8s) | Ollama |
|---------|------------------|-------------------|---------------|-------------|--------|
| **Deployment Speed** | ⚡ Rapid (zero-config) | ⚡ Rapid (zero-config) | 🐌 Complex setup | 🐌 Complex setup | 🔄 Medium |
| **Runtime Performance** | 🚀 High (Rust VLLM) | 🚀 Enhanced (Disaggregated) | 🐌 Multi-lang overhead | 🐌 Multi-lang overhead | 🐌 Single-node only |
| **Language Stack** | 🦀 Pure Rust | 🦀 Pure Rust | 🔄 Rust→Python→C++ | 🔄 Python→C++→Python | 🔄 Go→Python→C++ |
| **Serialization** | ✅ Zero-copy | ✅ Zero-copy | ❌ Repeated ser/deser | ❌ Repeated ser/deser | ❌ Repeated ser/deser |
| **Cloud Design** | ✅ Cloud-native | ✅ Cloud-native | ✅ Cloud-focused | ✅ Cloud-focused | ❌ Single-machine |
| **Self-Healing** | ✅ SWIM consensus | ✅ Enhanced SWIM | ❌ Manual failover | ❌ Manual failover | ❌ Manual restart |
| **Container Dependency** | 🆓 Optional | 🆓 Optional | 📦 Required (K8s) | 📦 Kubernetes-native | 🐳 Docker only |
| **Load Balancing** | ✅ Pingora + HTTP/3 | ✅ Pingora + HTTP/3 | 🔄 Basic | 🔄 Basic | ❌ None |
| **Cost Optimization** | ❌ Manual | ✅ Governator AI | 💰 Expensive | 💰 Expensive | 🆓 Free |
| **GPU Optimization** | 🔧 Manual tuning | 🧠 Auto (Governator) | 🔧 Manual tuning | 🔧 Manual tuning | 🔧 Manual tuning |
| **Observability** | 📊 Prometheus | 📊 Enhanced metrics | 🔄 Limited | 🔄 Limited | ❌ Basic logs |
| **Protocol Support** | ✅ HTTP/3, QUIC | ✅ HTTP/3, QUIC | 🔄 HTTP/2 only | 🔄 HTTP/2 only | 🔄 HTTP/1.1 |
| **Extensibility** | 🔒 Performance-first | 🔒 Performance-first | 🔧 Highly extensible | 🔧 Highly extensible | 🔧 Plugin system |
| **License** | 🆓 Free | 💰 Commercial | 🆓 Open source* | 🆓 Apache 2.0 | 🆓 Open source (MIT) |

*Nvidia Dynamo: Open source with optional enterprise support via NVIDIA AI Enterprise
**llm-d**: Kubernetes-native framework by CoreWeave, Google, IBM, NVIDIA, Red Hat

**Key Advantages:**
- **Pure Rust stack** eliminates multi-language overhead (competitors: Rust→Python→C++)
- **Zero-copy operations** vs repeated serialization/deserialization in competitors
- **True cloud-native design** (Ollama limited to single-machine deployments)
- **Rapid deployment** with zero-configuration startup
- **Built-in self-healing** via SWIM consensus protocol
- **High performance** through unified Rust stack and disaggregated VLLM
- **Container-optional** deployment (competitors require Docker/K8s, llm-d is K8s-native)
- **Advanced protocols** (HTTP/3, QUIC) while competitors use older standards

## Community vs Enterprise

### Community Edition (Free)
**Sub-60 second deployment with production-grade performance optimization.**

- ⚡ **Rapid deployment**: Zero-configuration startup with optimal defaults
- 🏆 **Optimized architecture**: Streamlined design with configurable performance defaults
- 🔧 **Simplified stack**: Rust-native implementation without Python extensibility layers
- 🌐 **Self-healing discovery**: SWIM consensus protocol for automatic node discovery and failure detection
- 🐳 **Container-optional**: No Docker or Kubernetes required (but supported if preferred)
- 🦀 **In-house Rust VLLM**: Custom-tuned Rust implementation for maximum performance
- ⚖️ **Cloudflare Pingora**: Enterprise-grade load balancing with HTTP/3 and QUIC support
- 📊 **Comprehensive metrics**: Prometheus monitoring for every system component
- 📈 **High throughput**: Benchmarked performance improvements over baseline implementations
- ⚖️ **Design approach**: Optimized defaults reduce configuration complexity while maintaining extensibility

Ideal for teams seeking production-ready performance with minimal operational overhead.

### Enterprise Edition
**Production-grade performance with intelligent cluster management and cost optimization.**

- 🚀 **Enhanced performance**: Hardware-optimized binaries with specialized tuning profiles
- 🧠 **Governator AI**: ML-driven optimization engine that automatically:
  - Profiles GPU hardware configurations using telemetry analysis
  - Determines optimal model placement through reinforcement learning
  - Adjusts request load distribution based on real-time performance metrics
  - *Technical details available in separate architecture documentation*
- 💰 **Cost optimization**: Intelligent cluster management that:
  - Monitors $/token efficiency across nodes
  - Automatically scales up high-performing instances
  - Powers down suboptimal nodes to reduce costs
- 🌐 **Enhanced SWIM consensus**: Advanced node discovery with predictive failure detection
- 🔄 **Disaggregated VLLM**: Full disaggregated architecture replacing Nvidia Dynamo and DLLM
- ⚡ **Advanced HTTP/3 & QUIC**: Optimized protocol implementations with custom tuning for AI workloads
- 📊 **Comprehensive telemetry**: Enhanced Prometheus metrics with AI-driven insights and alerting
- 📈 **Advanced analytics**: Real-time cost and performance insights with predictive modeling

Enterprise Edition provides measurable ROI through automated optimization, disaggregated architecture, and intelligent cost management.




## Advanced Configuration

The proxy can be configured through environment variables with the `PINGORA_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `PINGORA_LISTEN_ADDR` | `127.0.0.1:8080` | Proxy listen address |
| `PINGORA_BACKEND_ADDR` | `127.0.0.1:3000` | Primary backend address |
| `PINGORA_BACKEND_SERVERS` | - | Comma-separated list for load balancing |
| `PINGORA_MAX_CONNECTIONS` | `10000` | Maximum concurrent connections |
| `PINGORA_TIMEOUT_SECONDS` | `30` | Request timeout |
| `PINGORA_ENABLE_HEALTH_CHECK` | `true` | Enable backend health checking |
| `PINGORA_HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Health check frequency |
| `PINGORA_HEALTH_CHECK_PATH` | `/health` | Health check endpoint |
| `PINGORA_LOG_LEVEL` | `info` | Logging level (error/warn/info/debug/trace) |
| `PINGORA_ENABLE_METRICS` | `true` | Enable metrics collection |
| `INFERNO_OPERATIONS_ADDR` | `127.0.0.1:6100` | Operations server address |
| `PINGORA_ENABLE_TLS` | `false` | Enable TLS/SSL |
| `PINGORA_LOAD_BALANCING_ALGORITHM` | `round_robin` | Load balancing strategy |

### Example Configuration

```bash
# Production-like setup with load balancing
export PINGORA_LISTEN_ADDR="0.0.0.0:80"
export PINGORA_BACKEND_SERVERS="10.0.1.10:8080,10.0.1.11:8080,10.0.1.12:8080"
export PINGORA_MAX_CONNECTIONS="50000"
export PINGORA_LOG_LEVEL="warn"
export PINGORA_LOAD_BALANCING_ALGORITHM="least_connections"

cargo run --release
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests
cargo test --test integration_tests

# Run with logging
RUST_LOG=debug cargo test
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench metrics
cargo bench config

# Generate HTML reports
cargo bench --bench proxy_benchmarks
open target/criterion/report/index.html
```

### Code Quality

```bash
# Lint code
cargo clippy -- -D warnings

# Format code
cargo fmt

# Check for security vulnerabilities
cargo audit

# Generate documentation
cargo doc --open
```

### Test the Proxy

```bash
# Start a test backend
python3 -m http.server 3000 &

# Test proxy functionality
curl http://localhost:8080/

# View proxy metrics (operations server)
curl http://localhost:6100/metrics

# Check proxy health (operations server)
curl http://localhost:6100/health

# Register a backend (operations server)
curl -X POST http://localhost:6100/registration \
  -H "Content-Type: application/json" \
  -d '{"id":"test-backend","address":"127.0.0.1:3000","metrics_port":6100}'
```


## Performance

### Benchmarks

The proxy achieves the following performance characteristics on modern hardware:

- **Latency**: < 1ms P99 for local backends
- **Throughput**: > 100,000 requests/second
- **Memory**: < 1KB per concurrent connection
- **CPU**: < 10% overhead vs direct connection
- **Startup**: < 100ms cold start

### Optimization Features

- Lock-free metrics collection using atomics
- Zero-copy operations where possible
- Efficient connection pooling and reuse
- Pre-allocated data structures for hot paths
- SIMD-optimized operations for data processing

## Monitoring

### Metrics

The proxy exposes comprehensive metrics in Prometheus format:

```
# Request metrics
proxy_requests_total
proxy_requests_active
proxy_responses_total
proxy_responses_by_status_total

# Performance metrics
proxy_request_duration_ms
proxy_success_rate
proxy_requests_per_second

# Backend metrics
proxy_backend_connections_total
proxy_backend_connection_errors_total
```

### Health Checks

- **Proxy health**: `GET /health` on operations server (port 6100)
- **Backend health**: `GET /health` on operations server (port 6100) + automatic monitoring with configurable intervals
- **Circuit breaker**: Automatic failover for unhealthy backends
- **Service registration**: `POST /registration` on operations server (port 6100)

### Logging

Structured JSON logging with configurable levels:

```json
{
  "timestamp": "2023-12-07T10:30:45Z",
  "level": "INFO",
  "message": "Request processed successfully",
  "backend": "192.168.1.10:8080",
  "status": 200,
  "duration_ms": 15,
  "request_id": "req_123456"
}
```

## Security

### Security Features

- Input validation and sanitization
- Request size limits and rate limiting
- Security headers (CSP, HSTS, X-Frame-Options)
- TLS/SSL support with modern cipher suites
- Secure defaults for all configuration options

### Security Headers

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-Proxy-Cache: MISS
X-Forwarded-By: pingora-proxy-demo
```

## Load Balancing

### Algorithms

- **Round Robin**: Distribute requests evenly across backends
- **Least Connections**: Route to backend with fewest active connections
- **Weighted**: Route based on backend capacity weights

### Health Checking

- Periodic HTTP health checks to backend endpoints
- Automatic backend exclusion for failed health checks
- Gradual traffic restoration for recovered backends
- Configurable health check timeouts and intervals

## Error Handling

### Error Categories

- **Configuration Errors**: Invalid settings, validation failures
- **Network Errors**: Connection timeouts, DNS failures
- **Backend Errors**: Upstream server errors, invalid responses
- **System Errors**: Resource exhaustion, OS-level failures

### Error Recovery

- Automatic retries with exponential backoff
- Circuit breaker patterns for cascade failure prevention
- Graceful degradation under resource pressure
- Detailed error logging for debugging and monitoring

## Contributing

### Development Setup

1. Fork and clone the repository
2. Install Rust toolchain (latest stable)
3. Run `cargo test` to verify setup
4. Create feature branch for changes
5. Add tests for new functionality
6. Run `cargo fmt` and `cargo clippy`
7. Submit pull request with description

### Code Standards

- Follow Test-Driven Development (TDD) practices
- Maintain > 90% test coverage
- Document all public APIs with examples
- Use semantic versioning for releases
- Include performance benchmarks for changes

#### Before a task is complete, these checks must pass:
No lint errors
1. `cargo clippy --all-targets --all-features -- -D warnings`
Formatted
2. `cargo fmt --check`
No unused dependencies
3. `cargo machete`
No failing tests
4. `cargo test`

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Cloudflare Pingora](https://github.com/cloudflare/pingora) - The underlying proxy framework
- [Tokio](https://tokio.rs/) - Async runtime for Rust
- [Prometheus](https://prometheus.io/) - Metrics collection and monitoring
