# Service Configuration Guide

## Overview

Complete configuration guide for Pingora proxy services. Supports environment variables, CLI arguments, and YAML files for maximum deployment flexibility.

## Quick Start

### Minimal Load Balancer
```bash
# Start with defaults
cargo run

# Or with basic config
cargo run -- --service-port=8080 --metrics-port=9090
```

### Minimal Backend
```bash
# Backend with service discovery
cargo run -- --type=backend --discovery-lb=lb1:8080,lb2:8080
```

## Configuration Sources

Configuration is loaded in priority order:
1. **CLI Arguments** (highest priority)
2. **Environment Variables**  
3. **YAML Configuration File**
4. **Built-in Defaults** (lowest priority)

## YAML Configuration

### Complete Configuration Example
```yaml
# pingora.yaml - Complete configuration example

# Node identification
node:
  type: "load_balancer"  # load_balancer | backend
  id: "${HOSTNAME}"      # Auto-generate from hostname
  region: "us-west-2"
  zone: "us-west-2a"
  environment: "production"

# Network settings  
network:
  service_port: 8080     # Main proxy traffic
  metrics_port: 9090     # Metrics and health vitals
  bind_address: "0.0.0.0"
  
# Service discovery
discovery:
  # Load balancer endpoints for backend registration
  load_balancers:
    - "lb1.example.com:8080"
    - "lb2.example.com:8080"  
    - "lb3.example.com:8080"
  
  # Announcement settings
  announce_interval: "30s"
  retry_attempts: 3
  timeout: "5s"

# Load balancer specific settings
load_balancer:
  algorithm: "round_robin"  # round_robin | least_connections | ip_hash
  
  # Backend health checking via metrics port
  health_check:
    interval: "5s"
    timeout: "2s" 
    failure_threshold: 3
    recovery_threshold: 2
    
  # Connection settings
  max_connections: 10000
  connection_timeout: "30s"
  
# Backend specific settings  
backend:
  # Resource limits
  max_connections: 1000
  weight: 1.0
  
  # Graceful shutdown
  shutdown:
    drain_timeout: "30s"
    force_timeout: "10s"
    
# Logging configuration
logging:
  level: "info"           # error | warn | info | debug | trace
  format: "json"          # json | pretty
  
# Optional TLS settings
tls:
  enabled: false
  cert_path: "/etc/ssl/certs/server.pem"
  key_path: "/etc/ssl/private/server.key"
```

### Load Balancer Configuration
```yaml
# lb.yaml - Load balancer specific config
node:
  type: "load_balancer"
  
network:
  service_port: 8080
  metrics_port: 9090
  
load_balancer:
  algorithm: "round_robin"
  max_connections: 10000
  
  health_check:
    interval: "5s"
    timeout: "2s"
```

### Backend Configuration  
```yaml
# backend.yaml - Backend specific config
node:
  type: "backend"
  
network:
  service_port: 3000
  metrics_port: 9090
  
discovery:
  load_balancers: ["lb1:8080", "lb2:8080"]
  announce_interval: "30s"
  
backend:
  max_connections: 1000
  weight: 1.0
```

### Multi-Environment Configuration
```yaml
# Environment-specific overrides
development:
  logging:
    level: "debug"
    format: "pretty"
  network:
    service_port: 8080
    
production:
  logging:
    level: "warn" 
    format: "json"
  network:
    service_port: 80
  tls:
    enabled: true
```

## Environment Variables

All YAML settings can be overridden with environment variables using `PINGORA_` prefix:

```bash
# Node settings
export PINGORA_NODE_TYPE="backend"
export PINGORA_NODE_ID="backend-01" 
export PINGORA_NODE_REGION="us-west-2"

# Network settings
export PINGORA_NETWORK_SERVICE_PORT="3000"
export PINGORA_NETWORK_METRICS_PORT="9090"
export PINGORA_NETWORK_BIND_ADDRESS="0.0.0.0"

# Discovery settings
export PINGORA_DISCOVERY_LOAD_BALANCERS="lb1:8080,lb2:8080,lb3:8080"
export PINGORA_DISCOVERY_ANNOUNCE_INTERVAL="30s"

# Load balancer settings
export PINGORA_LOAD_BALANCER_ALGORITHM="round_robin"
export PINGORA_LOAD_BALANCER_MAX_CONNECTIONS="10000"

# Backend settings  
export PINGORA_BACKEND_MAX_CONNECTIONS="1000"
export PINGORA_BACKEND_WEIGHT="1.5"

# Logging
export PINGORA_LOGGING_LEVEL="info"
export PINGORA_LOGGING_FORMAT="json"

# TLS
export PINGORA_TLS_ENABLED="true"
export PINGORA_TLS_CERT_PATH="/etc/ssl/certs/server.pem"
export PINGORA_TLS_KEY_PATH="/etc/ssl/private/server.key"
```

## CLI Arguments

```bash
# Basic node settings
--type=backend                    # Node type: load_balancer | backend
--node-id=backend-01              # Unique node identifier  
--region=us-west-2                # Deployment region
--environment=production          # Environment name

# Network settings
--service-port=3000               # Main service port
--metrics-port=9090               # Metrics and health port
--bind-address=0.0.0.0           # Bind address

# Discovery settings  
--discovery-lb=lb1:8080,lb2:8080  # Load balancer addresses
--announce-interval=30s           # How often to announce

# Load balancer settings
--algorithm=round_robin           # Load balancing algorithm
--max-connections=10000           # Maximum connections

# Backend settings
--weight=1.5                      # Backend weight for load balancing
--max-backend-connections=1000    # Max connections for backend

# Logging
--log-level=info                  # Logging level
--log-format=json                 # Log output format

# Configuration file
--config=/path/to/config.yaml     # YAML config file path
```

## Configuration Validation

### Required Settings
- **service_port**: Must be available port (1-65535)
- **metrics_port**: Must be different from service_port
- **node.type**: Must be "load_balancer" or "backend"

### Backend Requirements
- **discovery.load_balancers**: At least one load balancer address
- **node.id**: Must be unique across all backends

### Load Balancer Requirements  
- **load_balancer.algorithm**: Valid algorithm name
- **load_balancer.max_connections**: Must be > 0

### Validation Examples
```yaml
# ❌ Invalid - same ports
network:
  service_port: 8080
  metrics_port: 8080  # Error: ports must be different

# ❌ Invalid - backend without discovery
node:
  type: "backend"
# Missing discovery.load_balancers

# ✅ Valid - minimal backend
node:
  type: "backend"
discovery:
  load_balancers: ["lb1:8080"]
```

## Deployment Patterns

### Docker Deployment
```bash
# Load balancer
docker run -p 8080:8080 -p 9090:9090 \
  -e PINGORA_NODE_TYPE=load_balancer \
  pingora-proxy

# Backend with discovery
docker run -p 3000:3000 -p 9090:9090 \
  -e PINGORA_NODE_TYPE=backend \
  -e PINGORA_DISCOVERY_LOAD_BALANCERS=lb1:8080,lb2:8080 \
  pingora-proxy
```

### Kubernetes Deployment
```yaml
# ConfigMap for common settings
apiVersion: v1
kind: ConfigMap
metadata:
  name: pingora-config
data:
  pingora.yaml: |
    logging:
      level: "info"
      format: "json"
    network:
      metrics_port: 9090
---
# Load balancer deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pingora-lb
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: pingora
        image: pingora-proxy:latest
        env:
        - name: PINGORA_NODE_TYPE
          value: "load_balancer"
        ports:
        - containerPort: 8080
        - containerPort: 9090
```

### Systemd Service
```ini
# /etc/systemd/system/pingora.service
[Unit]
Description=Pingora Proxy
After=network.target

[Service]
Type=exec
User=pingora
ExecStart=/usr/local/bin/pingora --config=/etc/pingora/config.yaml
Restart=always
RestartSec=5

# Environment file
EnvironmentFile=/etc/pingora/environment

[Install]
WantedBy=multi-user.target
```

## Monitoring Integration

### Prometheus Configuration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'pingora-backends'
    static_configs:
      - targets: ['backend1:9090', 'backend2:9090']
    metrics_path: '/telemetry'
    scrape_interval: 15s
    
  - job_name: 'pingora-load-balancers'  
    static_configs:
      - targets: ['lb1:9090', 'lb2:9090']
    metrics_path: '/telemetry'
    scrape_interval: 5s
```

### Health Check Integration
```bash
# Health check script for load balancer
#!/bin/bash
response=$(curl -s http://localhost:9090/metrics)
ready=$(echo "$response" | jq -r '.ready')

if [ "$ready" = "true" ]; then
  exit 0  # Healthy
else
  exit 1  # Unhealthy
fi
```

## Troubleshooting

### Common Configuration Issues

1. **Port Already in Use**
```bash
# Check what's using the port
lsof -i :8080

# Use different port
--service-port=8081
```

2. **Backend Can't Reach Load Balancer**
```bash
# Test connectivity
telnet lb1 8080

# Check DNS resolution
nslookup lb1
```

3. **Invalid Configuration**
```bash
# Validate config file
pingora --config=/path/to/config.yaml --validate-config

# Check logs for validation errors
journalctl -u pingora -f
```

### Debug Configuration
```yaml
# Enable debug logging and validation
logging:
  level: "debug"
  
# Add validation mode (exits after config check)
debug:
  validate_and_exit: true
  dump_config: true
```

## Security Considerations

### TLS Configuration
```yaml
tls:
  enabled: true
  cert_path: "/etc/ssl/certs/pingora.pem"
  key_path: "/etc/ssl/private/pingora.key"
  
  # Optional: Client certificate verification
  client_ca_path: "/etc/ssl/ca/client-ca.pem"
  verify_client: true
```

### Network Security
```yaml
network:
  bind_address: "127.0.0.1"  # Only local connections
  
  # Access control
  allowed_ips:
    - "10.0.0.0/8"
    - "192.168.0.0/16"
```

This configuration guide provides comprehensive coverage of all configuration options while maintaining the minimalist philosophy established in the service discovery specification.