# Test Coverage TODO List

This checklist tracks all uncovered functions that need tests to achieve 100% function coverage.

## Status Summary
- **Total Uncovered Functions**: 130+
- **Priority**: Focus on High Priority functions first
- **Coverage Goal**: 100% function coverage

---

## High Priority Functions (Core Public API)

### CLI Options - Main Entry Points
- [x] `crates/governator/src/cli_options.rs:102` - `async fn run(self) -> Result<()>` - Runs governator server
- [x] `crates/governator/src/cli_options.rs:135` - `fn to_config(&self) -> Result<GovernatorConfig>` - Converts CLI to config
- [x] `crates/backend/src/cli_options.rs:82` - `async fn run(self) -> Result<()>` - Runs backend server
- [x] `crates/backend/src/cli_options.rs:181` - `fn to_config(&self) -> Result<BackendConfig>` - Converts CLI to config
- [x] `crates/proxy/src/cli_options.rs:78` - `async fn run(self) -> Result<()>` - Runs proxy server
- [x] `crates/proxy/src/cli_options.rs:115` - `fn to_config(&self) -> Result<ProxyConfig>` - Converts CLI to config

### Operations Server Core Functions
- [x] `shared/src/operations_server.rs:134` - `fn new(metrics: Arc<MetricsCollector>, bind_addr: SocketAddr) -> Self`
- [x] `shared/src/operations_server.rs:170` - `fn with_service_info(...) -> Self`
- [x] `shared/src/operations_server.rs:188` - `fn with_service_discovery(...) -> Self`
- [x] `shared/src/operations_server.rs:271` - `async fn start(mut self) -> Result<()>`
- [x] `shared/src/operations_server.rs:383` - `async fn shutdown(&mut self) -> Result<()>`
- [x] `shared/src/operations_server.rs:401` - `fn bind_addr(&self) -> SocketAddr`
- [x] `shared/src/operations_server.rs:228` - `fn connected_peers_handle(&self) -> Arc<AtomicU32>`

### Service Discovery Client Core Functions
- [ ] `shared/src/service_discovery/client.rs:118` - `fn new(request_timeout: Duration) -> Self` (ClientConfig)
- [ ] `shared/src/service_discovery/client.rs:145` - `fn high_throughput() -> Self` (ClientConfig)
- [ ] `shared/src/service_discovery/client.rs:228` - `fn new(request_timeout: Duration) -> Self` (Client)
- [ ] `shared/src/service_discovery/client.rs:251` - `fn with_config(config: ClientConfig) -> Self`
- [ ] `shared/src/service_discovery/client.rs:315` - `async fn register_with_peer(...) -> Result<()>`
- [ ] `shared/src/service_discovery/client.rs:368` - `async fn update_with_peer(...) -> Result<()>`
- [ ] `shared/src/service_discovery/client.rs:521` - `async fn discover_peers(&self, peer_url: &str) -> Result<Vec<PeerInfo>>`
- [ ] `shared/src/service_discovery/client.rs:624` - `async fn check_peer_health(&self, peer_url: &str) -> Result<bool>`
- [ ] `shared/src/service_discovery/client.rs:682` - `fn config(&self) -> &ClientConfig`

### Service Discovery Service Core Functions
- [ ] `shared/src/service_discovery/service.rs:80` - `fn new() -> Self`
- [ ] `shared/src/service_discovery/service.rs:117` - `fn with_config(config: ServiceDiscoveryConfig) -> Self`
- [ ] `shared/src/service_discovery/service.rs:162` - `async fn register_backend(&self, registration: BackendRegistration) -> Result<()>`
- [ ] `shared/src/service_discovery/service.rs:246` - `async fn get_healthy_backends(&self) -> Vec<NodeInfo>`
- [ ] `shared/src/service_discovery/service.rs:315` - `async fn get_all_peers(&self) -> Vec<PeerInfo>`
- [ ] `shared/src/service_discovery/service.rs:348` - `async fn get_backend(&self, backend_id: &str) -> Option<NodeInfo>`
- [ ] `shared/src/service_discovery/service.rs:378` - `async fn remove_backend(&self, backend_id: &str) -> bool`
- [ ] `shared/src/service_discovery/service.rs:859` - `async fn get_all_backends(&self) -> Vec<NodeInfo>`
- [ ] `shared/src/service_discovery/service.rs:1109` - `async fn backend_count(&self) -> usize`

### Retry Manager Core Functions
- [ ] `shared/src/service_discovery/retry.rs:165` - `fn new() -> Self`
- [ ] `shared/src/service_discovery/retry.rs:178` - `fn with_config(config: RetryConfig) -> Self`
- [ ] `shared/src/service_discovery/retry.rs:204` - `async fn queue_retry(...) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:258` - `async fn process_retry_queue(&self) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:472` - `async fn get_metrics(&self) -> RetryMetrics`

---

## Medium Priority Functions (Supporting API)

### Node Type Utilities
- [ ] `shared/src/service_discovery/types.rs:110` - `fn can_serve_inference(&self) -> bool`
- [ ] `shared/src/service_discovery/types.rs:130` - `fn can_load_balance(&self) -> bool`
- [ ] `shared/src/service_discovery/types.rs:153` - `fn default_capabilities(&self) -> Vec<String>`
- [ ] `shared/src/service_discovery/types.rs:191` - `fn as_str(&self) -> &'static str`
- [ ] `shared/src/service_discovery/types.rs:218` - `fn parse(s: &str) -> Option<Self>`

### Node Info / Peer Info Functions
- [ ] `shared/src/service_discovery/types.rs:350` - `fn new(...) -> Self` (NodeInfo)
- [ ] `shared/src/service_discovery/types.rs:385` - `fn update_timestamp(&mut self)`
- [ ] `shared/src/service_discovery/types.rs:409` - `fn metrics_url(&self) -> String`
- [ ] `shared/src/service_discovery/types.rs:434` - `fn telemetry_url(&self) -> String`
- [ ] `shared/src/service_discovery/types.rs:464` - `fn has_capability(&self, capability: &str) -> bool`
- [ ] `shared/src/service_discovery/types.rs:556` - `fn from_node_info(node_info: &NodeInfo) -> Self` (PeerInfo)
- [ ] `shared/src/service_discovery/types.rs:592` - `fn to_node_info(&self) -> NodeInfo` (PeerInfo)
- [ ] `shared/src/service_discovery/types.rs:647` - `fn to_node_info(&self) -> NodeInfo` (BackendRegistration)
- [ ] `shared/src/service_discovery/types.rs:681` - `fn from_node_info(node_info: &NodeInfo) -> Self` (BackendRegistration)

---

## Lower Priority Functions (Internal/Private)

### Operations Server Internal Handlers
- [ ] `shared/src/operations_server.rs:440` - `async fn handle_request(...) -> Response<Body>`
- [ ] `shared/src/operations_server.rs:511` - `async fn handle_metrics_request(...) -> Response<Body>`
- [ ] `shared/src/operations_server.rs:590` - `async fn handle_health_request() -> Response<Body>`
- [ ] `shared/src/operations_server.rs:608` - `async fn handle_registration_request(...) -> Response<Body>`

### Service Discovery Internal Functions
- [ ] `shared/src/service_discovery/service.rs:216` - `async fn is_health_checked(&self, backend_id: &str) -> bool`
- [ ] `shared/src/service_discovery/service.rs:433` - `async fn register_with_peers(...) -> Result<()>`
- [ ] `shared/src/service_discovery/service.rs:531` - `async fn register_with_peer_with_backoff(...) -> Result<()>`
- [ ] `shared/src/service_discovery/service.rs:647` - `async fn resolve_consensus(...) -> NodeInfo`
- [ ] `shared/src/service_discovery/service.rs:721` - `async fn broadcast_self_update(...) -> Result<()>`
- [ ] `shared/src/service_discovery/service.rs:793` - `async fn get_retry_metrics(&self) -> RetryMetrics`
- [ ] `shared/src/service_discovery/service.rs:833` - `async fn process_retry_queue(&self) -> Result<()>`

### Service Discovery Client Internal Functions  
- [ ] `shared/src/service_discovery/client.rs:100` - `fn new(request_timeout: Duration) -> Self` (ClientConfig)
- [ ] `shared/src/service_discovery/client.rs:154` - `fn default() -> Self` (ClientConfig)
- [ ] `shared/src/service_discovery/client.rs:393` - `async fn register_with_peer_action(...) -> Result<()>`

### Retry Manager Internal Functions
- [ ] `shared/src/service_discovery/retry.rs:331` - `async fn process_retry_entry(&self, entry: RetryEntry) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:381` - `async fn send_update_to_peer(...) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:395` - `async fn reschedule_retry(&self, entry: RetryEntry) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:416` - `fn calculate_backoff_delay(&self, attempt: usize) -> u64`
- [ ] `shared/src/service_discovery/retry.rs:428` - `async fn move_to_dead_letter_queue(...) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:459` - `async fn remove_retry_entry(&self, retry_id: &str) -> Result<()>`
- [ ] `shared/src/service_discovery/retry.rs:478` - `fn default() -> Self`
- [ ] `shared/src/service_discovery/retry.rs:485` - `fn current_timestamp() -> u64`

---

## Files with Additional Uncovered Functions

### Other Files Requiring Analysis
- [ ] `backend/src/health.rs` - 4 missed functions (need individual analysis)
- [ ] `backend/src/inference.rs` - 2 missed functions (need individual analysis)
- [ ] `backend/src/main.rs` - 2 missed functions (need individual analysis)
- [ ] `backend/src/registration.rs` - 3 missed functions (need individual analysis)
- [ ] `cli/src/main.rs` - 2 missed functions (need individual analysis)
- [ ] `cli/src/cli_options.rs` - 2 missed functions (need individual analysis)
- [ ] `governator/src/main.rs` - 2 missed functions (need individual analysis)
- [ ] `proxy/src/main.rs` - 2 missed functions (need individual analysis)
- [ ] `proxy/src/lib.rs` - 5 missed functions (need individual analysis)
- [ ] `shared/src/cli.rs` - 5 missed functions (need individual analysis)
- [ ] `shared/src/metrics.rs` - 2 missed functions (need individual analysis)
- [ ] `shared/src/service_discovery/auth.rs` - 11 missed functions (all uncovered)
- [ ] `shared/src/service_discovery/config.rs` - 4 missed functions (need individual analysis)
- [ ] `shared/src/service_discovery/consensus.rs` - 1 missed function (need individual analysis)  
- [ ] `shared/src/service_discovery/errors.rs` - 8 missed functions (need individual analysis)
- [ ] `shared/src/service_discovery/health.rs` - 9 missed functions (need individual analysis)

---

## Implementation Strategy

1. **Start with High Priority functions** - These are the core public APIs
2. **Create focused test files** for each module being tested
3. **Use existing test patterns** - Follow the testing conventions already established in the codebase
4. **Mock external dependencies** where necessary
5. **Update this checklist** as functions get test coverage
6. **Run `cargo coverage`** after implementing tests to verify coverage improvements

## Progress Tracking

**Completed**: 0/130+ functions  
**In Progress**: 0 functions  
**Current Focus**: Starting with CLI options run() and to_config() methods

---

*Generated on 2025-08-30 for 100% function coverage goal*