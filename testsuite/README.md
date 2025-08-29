# Inferno Test Organization

The test suite is organized into three distinct categories:

## 1. Integration Tests (`./integration/`)

Integration tests verify the complete functionality of individual components and their APIs. These tests:
- Test public APIs and entry points
- Use mock dependencies (like mock servers)
- Run in test/debug mode
- Focus on functional correctness

**Examples:**
- `proxy_integration.rs` - Tests proxy server API and request handling

**Run with:** `cargo test --test integration`

## 2. Module Tests (`./module_tests/`)

Module tests verify the public functions and interfaces of library modules. These tests:
- Test public module functions
- Test module behavior in isolation
- Focus on edge cases and error conditions
- Use test doubles and mocks

**Examples:**
- `proxy_module.rs` - Tests proxy configuration, metrics, and error handling

**Run with:** `cargo test --test module_tests`

## 3. End-to-End Tests (`./e2e_tests/`)

End-to-end tests verify the complete system behavior with real binaries. These tests:
- Build and run actual release binaries
- Test real interaction between proxy and backend servers
- Verify system behavior under realistic conditions
- Test failover, load balancing, and graceful shutdown

**Examples:**
- `proxy_backend_e2e.rs` - Tests proxy and backend servers working together

**Run with:** `cargo test --test e2e_tests -- --ignored`

## Running Tests

```bash
# Run all unit tests (in source files)
cargo test --lib

# Run all integration tests
cargo test --test integration

# Run all module tests  
cargo test --test module_tests

# Run e2e tests (requires building release binaries)
cargo test --test e2e_tests -- --ignored

# Run all tests
cargo test --all
```

## Test Guidelines

1. **Unit tests** (in source files with `#[cfg(test)]`) should be fast and focused on single functions
2. **Integration tests** should test component APIs with mocked dependencies
3. **Module tests** should test public module interfaces thoroughly
4. **E2E tests** should verify real-world scenarios with actual binaries

## Performance Requirements

- Unit tests: < 100ms each
- Integration tests: < 1s each
- Module tests: < 500ms each
- E2E tests: < 10s each (includes binary compilation)