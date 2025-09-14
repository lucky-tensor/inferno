# Project 3: Docker End-to-End Test Harness

## Overview
Create an end-to-end test harness using Docker and Docker Swarm with one node in load balancer mode and two nodes in inference mode. Validate inference requests can be sent to the load balancer and return useful responses, with Docker logs verification.

## Requirements
- Set up Docker Swarm cluster with multiple nodes
- Deploy one load balancer node and two inference backend nodes
- Send inference requests to load balancer and validate responses
- Monitor and verify Docker logs for job execution across all nodes
- Test load balancing, service discovery, and fault tolerance

## Task Checklist

### ✅ Completed Tasks
- [x] **Analyze existing proxy/load balancer and backend components** - Examined architecture and APIs

### 🔄 In Progress Tasks
- [ ] Currently no tasks in progress

### 📋 Pending Tasks

#### Architecture and Design
- [ ] **Design Docker Swarm architecture for end-to-end testing**
  - Plan network topology and service communication
  - Design node roles and responsibilities
  - Plan service discovery and registration flow
  - Define test scenarios and validation criteria

#### Docker Infrastructure
- [ ] **Create Dockerfile for the Inferno application**
  - Multi-stage build for efficient image size
  - Include all necessary dependencies and models
  - Configure proper entrypoints for different service modes
  - Optimize for both development and production use

- [ ] **Create Docker Compose configuration for multi-node setup**
  - Define services for proxy and backend nodes
  - Configure networking between services
  - Set up environment variables and volumes
  - Configure logging for test validation

- [ ] **Implement Docker Swarm stack configuration** 
  - Create swarm stack definition file
  - Configure service placement and constraints
  - Set up overlay networking for multi-host deployment
  - Define scaling and update policies

#### Test Implementation  
- [ ] **Create test harness script for end-to-end validation**
  - Implement cluster setup and teardown automation
  - Add service health checking and readiness waits
  - Create test orchestration framework
  - Add detailed logging and reporting

- [ ] **Add inference request testing and response validation**
  - Implement HTTP client for inference requests  
  - Create test cases with various prompts and parameters
  - Validate response format and content quality
  - Test load balancing across multiple backends

- [ ] **Implement Docker logs monitoring and verification**
  - Create log aggregation and analysis tools
  - Verify job execution in backend node logs
  - Check load balancer routing decisions
  - Validate service discovery registration events

#### Integration Testing
- [ ] **Test complete end-to-end workflow with load balancing**
  - Test normal operation with all nodes healthy
  - Test fault tolerance when backend nodes fail
  - Test service discovery and dynamic registration
  - Test scaling up and down of backend nodes

## Implementation Notes

### Architecture Analysis

#### Proxy/Load Balancer Component
- **Framework**: Pingora-based high-performance reverse proxy
- **Main Port**: 8080 (configurable via INFERNO_LISTEN_ADDR)
- **Operations Port**: 6100 (metrics/health/registration)
- **Load Balancing**: Round robin, least connections, weighted
- **Service Discovery**: Dynamic backend registration with health checks

#### Backend Component  
- **Main Port**: 3000 (configurable via INFERNO_BACKEND_LISTEN_ADDR)
- **Operations Port**: 6100 (metrics and health)
- **AI Integration**: Burn framework with TinyLlama model
- **Registration**: Auto-registers with proxy operations servers

#### API Endpoints
```bash
# Main inference endpoint
POST /inference
{
  "prompt": "Hello, world!",
  "max_tokens": 10,
  "temperature": 0.7
}

# Health checks
GET /health                    # Service health
GET /metrics                   # NodeVitals JSON
POST /registration             # Backend registration
```

### Docker Configuration

#### Network Architecture
```yaml
version: '3.8'

networks:
  inferno-net:
    driver: overlay
    attachable: true

services:
  proxy-lb:
    image: inferno:latest
    command: ["inferno", "proxy", 
              "--listen-addr", "0.0.0.0:8080",
              "--backend-servers", "backend-1:3000,backend-2:3000"]
    ports:
      - "8080:8080"    # Main proxy traffic
      - "6100:6100"    # Operations server
    networks:
      - inferno-net

  backend-1:
    image: inferno:latest
    command: ["inferno", "backend",
              "--listen-addr", "0.0.0.0:3000",
              "--discovery-lb", "proxy-lb:6100"]
    ports:
      - "3001:3000"
      - "6201:6100"
    networks:
      - inferno-net

  backend-2:
    image: inferno:latest  
    command: ["inferno", "backend",
              "--listen-addr", "0.0.0.0:3000", 
              "--discovery-lb", "proxy-lb:6100"]
    ports:
      - "3002:3000"
      - "6202:6100"
    networks:
      - inferno-net
```

#### Port Mapping Strategy
- **8080**: Main proxy endpoint for client requests
- **6100-6103**: Operations servers (metrics/health/registration)
- **3001-3002**: Backend inference services
- **Container Internal**: Services communicate via overlay network

### Test Scenarios

#### 1. Basic Functionality
- Start all services and verify health endpoints
- Send inference requests and validate responses
- Check service discovery registration in logs
- Verify load balancing distribution

#### 2. Load Balancing Validation  
- Send multiple requests and verify round-robin distribution
- Check backend selection in proxy logs
- Validate response consistency across backends
- Test with different load balancing algorithms

#### 3. Fault Tolerance
- Stop one backend and verify traffic redirects
- Restart backend and verify re-registration
- Test proxy resilience to backend failures  
- Validate graceful degradation scenarios

#### 4. Service Discovery
- Start backends after proxy and verify registration
- Stop and restart services to test re-discovery
- Validate health check behavior in logs
- Test authentication and security features

### Log Verification Strategy

#### Expected Log Patterns
```bash
# Proxy logs
[INFO] Starting proxy server on 0.0.0.0:8080
[INFO] Backend registered: backend-1:3000
[INFO] Routing request to backend-1:3000
[INFO] Health check passed: backend-1:3000

# Backend logs  
[INFO] Starting backend server on 0.0.0.0:3000
[INFO] Registering with proxy: proxy-lb:6100
[INFO] Processing inference request: "Hello world"
[INFO] Inference completed in 45ms
```

#### Validation Commands
```bash
# Check service startup
docker service logs inferno_proxy-lb | grep "Starting proxy"

# Verify backend registration
docker service logs inferno_proxy-lb | grep "Backend registered"  

# Check inference processing
docker service logs inferno_backend-1 | grep "Processing inference"

# Validate load balancing  
docker service logs inferno_proxy-lb | grep "Routing request"
```

### Success Criteria

#### Functional Requirements
- ✅ All services start successfully and pass health checks
- ✅ Backends register with proxy via service discovery  
- ✅ Inference requests return valid responses
- ✅ Load balancing distributes requests across backends
- ✅ Service discovery handles backend failures gracefully

#### Performance Requirements
- ✅ Inference requests complete within 5 seconds
- ✅ Load balancer handles 100+ requests per second
- ✅ Service discovery updates within 30 seconds
- ✅ No memory leaks or resource exhaustion over 1 hour test

#### Observability Requirements  
- ✅ All critical events appear in Docker logs
- ✅ Metrics endpoints provide system health data
- ✅ Log analysis can verify request routing decisions
- ✅ Service discovery events are properly logged

## Dependencies

### Docker Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- Docker Swarm mode enabled
- Sufficient resources (4GB+ RAM recommended)

### Inferno Requirements
- Compiled Inferno binary
- TinyLlama model files (auto-downloaded)
- Network connectivity for model downloads
- Sufficient disk space for container images

## Files to Create
- `/home/jeef/inferno/Dockerfile` - Multi-stage application build
- `/home/jeef/inferno/docker-compose.yml` - Local testing setup
- `/home/jeef/inferno/docker-stack.yml` - Production swarm stack  
- `/home/jeef/inferno/scripts/e2e-test.sh` - Test harness script
- `/home/jeef/inferno/scripts/validate-logs.sh` - Log validation script