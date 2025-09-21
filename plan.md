# Inferno Codebase Deep Clean & Refactor Plan

## Current State Analysis

The codebase has become messy due to experimentation with multiple inference engines (Candle, Burn, etc.) and has accumulated technical debt. The recent aggressive minimization to GPU-only has improved things, but architectural concerns remain.

## Core Problems Identified

1. **Mixed Concerns**: Inference, memory management, health checks, and service discovery are tightly coupled
2. **Monolithic Structure**: Too many responsibilities in single crates
3. **Inconsistent Abstractions**: Some modules have clean interfaces, others are implementation-heavy
4. **Experimental Residue**: Dead code and unused abstractions from engine experiments
5. **Testing Gaps**: Integration tests are limited, unit tests are scattered

## Refactoring Strategy: Extract Standalone Crates

### Phase 1: Core Infrastructure Extraction

#### 1.1 `inferno-memory` Crate
**Purpose**: GPU memory management and allocation
**Extract From**: `crates/inference/src/memory/`
**Dependencies**: CUDA runtime, minimal
**API**:
```rust
pub struct CudaMemoryPool;
pub struct MemoryTracker;
pub trait GpuAllocator;
```

#### 1.2 `inferno-models` Crate
**Purpose**: Model loading and configuration
**Extract From**: `crates/inference/src/models/`
**Dependencies**: safetensors, serde, burn
**API**:
```rust
pub trait ModelLoader<T>;
pub struct LlamaLoader;
pub struct ModelConfig;
```

#### 1.3 `inferno-health` Crate
**Purpose**: Health monitoring and metrics
**Extract From**: `crates/inference/src/health/`
**Dependencies**: tokio, serde
**API**:
```rust
pub struct HealthChecker;
pub enum HealthStatus;
pub trait HealthProvider;
```

#### 1.4 `inferno-discovery` Crate
**Purpose**: Service discovery and registration
**Extract From**: `crates/inference/src/service/`
**Dependencies**: tokio, reqwest, serde
**API**:
```rust
pub struct ServiceRegistration;
pub trait DiscoveryProvider;
```

### Phase 2: Core Inference Reorganization

#### 2.1 `inferno-inference` Crate (Simplified)
**Purpose**: Pure inference engine abstractions
**Keep**: Engine traits, request/response types
**Remove**: Memory management, health checks, service discovery
**Dependencies**: Only the extracted crates above

#### 2.2 `inferno-backend` Crate (Orchestrator)
**Purpose**: Orchestrate all components together
**Dependencies**: All extracted crates + inferno-inference
**Responsibilities**: Configuration, startup, coordination

### Phase 3: Clean Interfaces

#### 3.1 Dependency Injection
Replace direct instantiation with trait-based injection:
```rust
pub struct InferenceEngine<M, H, D>
where
    M: MemoryManager,
    H: HealthProvider,
    D: DiscoveryProvider
{
    memory: M,
    health: H,
    discovery: D,
}
```

#### 3.2 Event-Driven Architecture
Implement pub/sub for component communication:
```rust
pub enum InfernoEvent {
    MemoryAlert(MemoryStats),
    HealthStatusChanged(HealthStatus),
    ServiceRegistered(ServiceInfo),
}
```

## Implementation Plan

### Week 1: Memory & Models Extraction
- [ ] Create `inferno-memory` crate with clean API
- [ ] Extract memory management from inference crate
- [ ] Create `inferno-models` crate
- [ ] Move model loading logic to dedicated crate
- [ ] Update all references and imports

### Week 2: Health & Discovery Extraction
- [ ] Create `inferno-health` crate with monitoring traits
- [ ] Extract health checking from inference crate
- [ ] Create `inferno-discovery` crate for service registration
- [ ] Implement clean service discovery abstractions

### Week 3: Interface Cleanup
- [ ] Simplify `inferno-inference` to pure inference concerns
- [ ] Implement dependency injection patterns
- [ ] Create event bus for component communication
- [ ] Update configuration to support modular architecture

### Week 4: Integration & Testing
- [ ] Create comprehensive integration tests
- [ ] Update `inferno-backend` to orchestrate all components
- [ ] Verify all functionality works with new architecture
- [ ] Performance benchmarking to ensure no regressions

## Benefits

1. **Separation of Concerns**: Each crate has a single, clear responsibility
2. **Testability**: Easier to unit test individual components
3. **Reusability**: Other projects can use specific crates (e.g., just memory management)
4. **Maintainability**: Changes in one area don't cascade through entire codebase
5. **Team Development**: Different teams can work on different crates independently

## Migration Strategy

1. **Backwards Compatibility**: Keep existing public APIs during transition
2. **Incremental Migration**: Extract one crate at a time
3. **Feature Flags**: Use cargo features to enable new vs old implementations
4. **Integration Testing**: Ensure functionality remains intact throughout process

## Success Metrics

- [ ] Compilation time reduction (fewer dependencies per crate)
- [ ] Test coverage improvement (easier to test isolated components)
- [ ] Documentation clarity (each crate has focused docs)
- [ ] Development velocity increase (fewer merge conflicts, clearer ownership)