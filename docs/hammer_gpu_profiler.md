# Hammer GPU Profiler Specification

## Overview

The Hammer GPU Profiler is a comprehensive stress testing and benchmarking tool designed to measure the inference capacity and performance characteristics of backend nodes in the Inferno network. It extracts critical metrics including latency, GPU utilization, memory consumption, and CPU usage under various load conditions to establish baseline health scores and cost-per-token calculations.

## Objectives

1. **Health Assessment**: Establish a starting health score for newly joined nodes
2. **Performance Profiling**: Measure inference capacity under different load patterns
3. **Cost Analysis**: Calculate cost-per-token for rental and energy consumption
4. **Capacity Planning**: Determine optimal workload distribution across nodes

## Core Metrics

### Primary Metrics
- **Latency**: Request processing time (average and P99)
- **GPU Utilization**: Percentage of GPU compute resources used
- **Memory Usage**: GPU memory consumption patterns
- **CPU Usage**: Host CPU utilization during inference

### Secondary Metrics
- **Throughput**: Requests per second at saturation
- **Error Rate**: Failed requests under load
- **Temperature**: Thermal characteristics under stress
- **Power Consumption**: Energy usage for cost calculations

## Test Scenarios

### 1. Parallel Load Testing
- Execute multiple independent requests simultaneously
- Measure performance degradation as parallel requests increase
- Test scenarios: 1, 2, 4, 8, 16, 32, 64 parallel requests

### 2. Concurrent Batch Processing
- Process batched requests with varying batch sizes
- Measure throughput optimization through batching
- Test batch sizes: 1, 4, 8, 16, 32, 64

### 3. GPU Saturation Testing
- Gradually increase load until GPU utilization reaches 100%
- Identify the saturation point and performance cliff
- Measure graceful degradation characteristics

### 4. Sustained Load Testing
- Run continuous load for extended periods (10-30 minutes)
- Monitor thermal throttling and performance stability
- Detect memory leaks or performance degradation over time

## Implementation Architecture

### Governator Integration (Recommended)
```
[Governator] → [Local Inference Engine] → [Metrics Collection]
     ↓
[Health Scoring & Cost Analysis]
     ↓
[Network Registration & Routing Decisions]
```

**Architecture Benefits:**
- **Lean Binary Strategy**: No additional deployment overhead - hammer functionality integrated into existing governator binary
- **Direct System Access**: Full access to system metrics (GPU, CPU, memory) without network overhead
- **Integrated Governance**: Health scores directly feed into node admission and routing decisions
- **Unified Configuration**: Single configuration point for both governance and performance testing
- **Resource Efficiency**: Shared memory and process space with governator reduces system overhead

**Implementation within Governator:**
- Hammer module as a governator subcommand: `governator hammer --profile <profile_name>`
- Automatic health assessment during node registration
- Periodic background health checks during operation
- Integration with existing governator metrics and logging systems

### External Coordination (Optional)
For multi-node testing campaigns, external coordination can be achieved through:
```
[Network Controller] → [Governator Hammer API] → [Local Testing]
                    ↓
            [Aggregated Results]
```

This allows centralized orchestration while maintaining the lean binary approach.

## Technical Specifications

### Governator Hammer Module Components

#### 1. Load Generator
```rust
struct LoadGenerator {
    request_patterns: Vec<RequestPattern>,
    concurrency_levels: Vec<u32>,
    batch_sizes: Vec<u32>,
}

enum RequestPattern {
    Parallel(u32),      // N concurrent requests
    Sequential(u32),    // N sequential requests  
    Batch(u32),         // Batch size N
    Sustained(Duration), // Continuous load for duration
}
```

#### 2. Metrics Collector
```rust
struct MetricsCollector {
    gpu_monitor: GpuMonitor,
    cpu_monitor: CpuMonitor,
    memory_monitor: MemoryMonitor,
    thermal_monitor: ThermalMonitor,
}

struct PerformanceMetrics {
    latency_avg: Duration,
    latency_p99: Duration,
    gpu_utilization: f32,
    memory_usage: u64,
    cpu_usage: f32,
    power_consumption: Option<f32>,
    error_rate: f32,
}
```

#### 3. Health Score Calculator
```rust
struct HealthScore {
    overall_score: f32,     // 0.0 - 1.0
    latency_score: f32,
    throughput_score: f32,
    stability_score: f32,
    efficiency_score: f32,
}

impl HealthScore {
    fn calculate(metrics: &PerformanceMetrics, baseline: &Baseline) -> Self;
}
```

### Test Configuration

#### 1. Workload Profiles
```yaml
workload_profiles:
  - name: "small_model"
    model_size: "7B"
    context_length: 2048
    expected_tokens: 256
    
  - name: "medium_model"  
    model_size: "13B"
    context_length: 4096
    expected_tokens: 512
    
  - name: "large_model"
    model_size: "70B" 
    context_length: 8192
    expected_tokens: 1024
```

#### 2. Test Scenarios
```yaml
test_scenarios:
  - name: "baseline_capacity"
    duration: 300s
    ramp_up: 30s
    pattern: "parallel"
    max_concurrency: 64
    
  - name: "sustained_load"
    duration: 1800s
    pattern: "sustained"
    target_utilization: 0.8
    
  - name: "batch_optimization"
    duration: 600s
    pattern: "batch"
    batch_sizes: [1, 4, 8, 16, 32]
```

## Cost Analysis Framework

### Energy Cost Calculation
```rust
struct EnergyCost {
    power_consumption: f32,    // Watts
    duration: Duration,        // Test duration
    energy_rate: f32,         // Cost per kWh
    tokens_generated: u64,
}

impl EnergyCost {
    fn cost_per_token(&self) -> f32 {
        let kwh_used = (self.power_consumption * self.duration.as_secs_f32()) / 3600.0 / 1000.0;
        let total_cost = kwh_used * self.energy_rate;
        total_cost / self.tokens_generated as f32
    }
}
```

### Rental Cost Integration
```rust
struct RentalCost {
    hourly_rate: f32,         // Cost per hour
    utilization: f32,         // 0.0 - 1.0
    tokens_per_hour: f32,     // Throughput capacity
}

impl RentalCost {
    fn cost_per_token(&self) -> f32 {
        (self.hourly_rate * self.utilization) / self.tokens_per_hour
    }
}
```

## Output Format

### Health Report
```json
{
  "node_id": "node_123",
  "timestamp": "2024-08-30T12:00:00Z",
  "hardware_info": {
    "gpu_model": "RTX 4090",
    "gpu_memory": "24GB",
    "cpu_model": "AMD Ryzen 9 7950X",
    "ram": "64GB"
  },
  "health_score": {
    "overall": 0.87,
    "latency": 0.92,
    "throughput": 0.85,
    "stability": 0.89,
    "efficiency": 0.83
  },
  "performance_metrics": {
    "max_throughput": 45.2,
    "avg_latency_ms": 67.3,
    "p99_latency_ms": 142.7,
    "max_gpu_utilization": 0.94,
    "peak_memory_usage": "22.1GB"
  },
  "cost_analysis": {
    "energy_cost_per_token": 0.0012,
    "rental_cost_per_token": 0.0089,
    "total_cost_per_token": 0.0101
  }
}
```

## Integration with Inferno Network

### Governator Integration Points

#### Service Discovery Integration
- Leverage existing governator service discovery mechanisms
- Health scores automatically published to network registry
- Coordinate multi-node testing through existing governance channels

#### Metrics Export
- Utilize governator's existing metrics infrastructure
- Export hammer results through same Prometheus endpoints
- Integrate with governator's observability stack

#### Governance Decision Making
- Health scores directly influence node admission policies
- Performance metrics feed into routing algorithms
- Cost analysis informs workload placement decisions
- Automatic node quarantine for failing health checks

## Implementation Phases

### Phase 1: Governator Integration
- Add hammer subcommand to governator CLI
- Basic load generation and metrics collection within governator process
- Health score calculation and local storage

### Phase 2: Governance Integration  
- Integrate health scores into node registration process
- Automatic health assessment during governator startup
- Performance-based routing decision integration

### Phase 3: Advanced Features
- Cost analysis and optimization algorithms
- Periodic background health monitoring
- Predictive capacity planning

### Phase 4: Network Coordination
- Multi-node testing campaign coordination
- Historical trend analysis across network
- Automated anomaly detection and response

## Security Considerations

- Authenticated access to hammer endpoints
- Rate limiting to prevent abuse
- Resource isolation during testing
- Audit logging of all test activities

## Future Enhancements

- Machine learning models for performance prediction
- Automated anomaly detection
- Dynamic test scenario generation
- Integration with hardware monitoring systems
- Support for specialized inference hardware (TPUs, etc.)