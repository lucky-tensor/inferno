//! # Proxy Performance Benchmarks
//!
//! Comprehensive benchmarking suite for the Pingora proxy demo measuring:
//! - Request processing latency and throughput
//! - Memory allocation patterns and efficiency
//! - CPU utilization under various load patterns
//! - Concurrent connection handling performance
//! - Metrics collection overhead
//! - Configuration parsing and validation speed
//!
//! ## Benchmark Categories
//!
//! 1. **Hot Path Benchmarks**: Critical request processing paths
//! 2. **Concurrency Benchmarks**: Multi-threaded performance characteristics
//! 3. **Memory Benchmarks**: Allocation patterns and memory efficiency
//! 4. **Scalability Benchmarks**: Performance under increasing load
//! 5. **Cold Start Benchmarks**: Initialization and startup performance
//!
//! ## Performance Targets
//!
//! - Request latency: < 1ms P99 for local backends
//! - Throughput: > 100,000 RPS on modern hardware
//! - Memory efficiency: < 1KB per concurrent connection
//! - CPU efficiency: < 10% overhead vs direct connection
//! - Startup time: < 100ms
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench
//!
//! # Run specific benchmark group
//! cargo bench metrics
//! cargo bench config
//!
//! # Generate HTML reports
//! cargo bench --bench proxy_benchmarks
//! open target/criterion/report/index.html
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pingora_proxy_demo::error::ProxyError;
use pingora_proxy_demo::metrics::MetricsCollector;
use pingora_proxy_demo::{ProxyConfig, ProxyServer, ProxyService};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmarks for metrics collection performance
///
/// These benchmarks measure the performance impact of metrics collection
/// on request processing, ensuring that observability doesn't significantly
/// impact proxy performance.
///
/// ## Performance Requirements
///
/// - Metric update latency: < 10ns per operation
/// - Metric snapshot creation: < 1ms for full collection
/// - Memory overhead: < 100 bytes per metric
/// - Thread contention: Minimal impact on concurrent updates
fn bench_metrics_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");
    group.throughput(Throughput::Elements(1));

    let metrics = Arc::new(MetricsCollector::new());

    // Benchmark individual metric operations (hot path)
    group.bench_function("record_request", |b| {
        b.iter(|| {
            metrics.record_request();
        })
    });

    group.bench_function("record_response", |b| {
        b.iter(|| {
            metrics.record_response(200);
        })
    });

    group.bench_function("record_error", |b| {
        b.iter(|| {
            metrics.record_error();
        })
    });

    group.bench_function("record_request_duration", |b| {
        let duration = Duration::from_millis(15);
        b.iter(|| {
            metrics.record_request_duration(duration);
        })
    });

    group.bench_function("record_upstream_selection_time", |b| {
        let duration = Duration::from_micros(50);
        b.iter(|| {
            metrics.record_upstream_selection_time(duration);
        })
    });

    // Benchmark metric snapshot creation (cold path)
    group.bench_function("metrics_snapshot", |b| {
        // Pre-populate with some data
        for _ in 0..1000 {
            metrics.record_request();
            metrics.record_response(200);
        }

        b.iter(|| {
            let _snapshot = metrics.snapshot();
        })
    });

    // Benchmark Prometheus format generation
    group.bench_function("prometheus_format", |b| {
        // Pre-populate with data
        for _ in 0..1000 {
            metrics.record_request();
            metrics.record_response(200);
        }
        let snapshot = metrics.snapshot();

        b.iter(|| {
            let _prometheus = snapshot.to_prometheus_format();
        })
    });

    group.finish();
}

/// Benchmarks for configuration parsing and validation
///
/// These benchmarks measure the performance of configuration operations
/// which occur during startup and potentially during hot reloading.
///
/// ## Performance Requirements
///
/// - Configuration parsing: < 10ms for typical configs
/// - Validation time: < 1ms per configuration section
/// - Environment variable access: < 100μs total
/// - Memory allocation: Minimal during validation
fn bench_config_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("config");

    // Benchmark default configuration creation
    group.bench_function("config_default", |b| {
        b.iter(|| {
            let _config = ProxyConfig::default();
        })
    });

    // Benchmark configuration validation
    group.bench_function("config_validation", |b| {
        let config = ProxyConfig::default();
        b.iter(|| {
            let _validated = ProxyConfig::new(config.clone()).unwrap();
        })
    });

    // Benchmark environment variable loading
    group.bench_function("config_from_env", |b| {
        // Set up some environment variables
        std::env::set_var("PINGORA_LISTEN_ADDR", "127.0.0.1:8080");
        std::env::set_var("PINGORA_BACKEND_ADDR", "127.0.0.1:3000");
        std::env::set_var("PINGORA_MAX_CONNECTIONS", "10000");

        b.iter(|| {
            let _config = ProxyConfig::from_env().unwrap();
        });

        // Clean up
        std::env::remove_var("PINGORA_LISTEN_ADDR");
        std::env::remove_var("PINGORA_BACKEND_ADDR");
        std::env::remove_var("PINGORA_MAX_CONNECTIONS");
    });

    // Benchmark effective backends calculation
    group.bench_function("effective_backends", |b| {
        let mut config = ProxyConfig::default();
        config.backend_servers = vec![
            "192.168.1.1:8080".parse().unwrap(),
            "192.168.1.2:8080".parse().unwrap(),
            "192.168.1.3:8080".parse().unwrap(),
            "192.168.1.4:8080".parse().unwrap(),
            "192.168.1.5:8080".parse().unwrap(),
        ];

        b.iter(|| {
            let _backends = config.effective_backends();
        })
    });

    group.finish();
}

/// Benchmarks for error handling performance
///
/// These benchmarks measure the performance impact of error creation,
/// propagation, and handling in both success and failure scenarios.
///
/// ## Performance Requirements
///
/// - Error creation: < 100ns for common error types
/// - Error propagation: Zero allocation where possible
/// - Error conversion: < 50ns for From implementations
/// - HTTP status mapping: < 10ns constant time
fn bench_error_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling");

    // Benchmark error creation (various types)
    group.bench_function("error_network", |b| {
        b.iter(|| {
            let _error = ProxyError::network("127.0.0.1:8080", "Connection refused", None);
        })
    });

    group.bench_function("error_backend", |b| {
        b.iter(|| {
            let _error = ProxyError::backend("api.example.com", 500, "Internal Server Error");
        })
    });

    group.bench_function("error_timeout", |b| {
        let timeout = Duration::from_secs(30);
        b.iter(|| {
            let _error = ProxyError::timeout(timeout, "backend connection");
        })
    });

    // Benchmark error conversion
    group.bench_function("error_io_conversion", |b| {
        b.iter(|| {
            let io_error = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
            let _proxy_error: ProxyError = io_error.into();
        })
    });

    // Benchmark HTTP status mapping
    group.bench_function("error_http_status", |b| {
        let error = ProxyError::backend("backend", 404, "Not Found");
        b.iter(|| {
            let _status = error.to_http_status();
        })
    });

    // Benchmark temporary classification
    group.bench_function("error_is_temporary", |b| {
        let error = ProxyError::timeout(Duration::from_secs(10), "request");
        b.iter(|| {
            let _temporary = error.is_temporary();
        })
    });

    group.finish();
}

/// Benchmarks for proxy service creation and operations
///
/// These benchmarks measure the performance of core proxy operations
/// including service creation, request routing logic, and cleanup.
///
/// ## Performance Requirements
///
/// - Service creation: < 1μs
/// - Request routing decision: < 10μs
/// - Connection establishment: < 1ms to local backends
/// - Memory footprint: < 1KB per service instance
fn bench_proxy_service(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("proxy_service");

    // Benchmark proxy service creation
    group.bench_function("service_creation", |b| {
        let config = Arc::new(ProxyConfig::default());
        let metrics = Arc::new(MetricsCollector::new());

        b.iter(|| {
            let _service = ProxyService::new(Arc::clone(&config), Arc::clone(&metrics));
        })
    });

    // Benchmark server creation (async operation)
    group.bench_function("server_creation", |b| {
        let config = ProxyConfig::default();

        b.to_async(&rt).iter(|| async {
            let _server = ProxyServer::new(config.clone()).await.unwrap();
        })
    });

    group.finish();
}

/// Benchmarks for concurrent operations
///
/// These benchmarks measure performance under concurrent load,
/// simulating real-world usage patterns with multiple threads
/// updating metrics and handling requests simultaneously.
///
/// ## Performance Requirements
///
/// - Concurrent metric updates: Linear scaling to 16+ threads
/// - Lock contention: Minimal (lock-free operations preferred)
/// - Memory ordering overhead: < 5% performance impact
/// - Thread safety: No data races or inconsistencies
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency");

    // Benchmark concurrent metrics updates
    for thread_count in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_metrics", thread_count),
            thread_count,
            |b, &thread_count| {
                let metrics = Arc::new(MetricsCollector::new());

                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let metrics = Arc::clone(&metrics);
                            std::thread::spawn(move || {
                                for _ in 0..1000 {
                                    metrics.record_request();
                                    metrics.record_response(200);
                                    metrics.record_request_duration(Duration::from_millis(10));
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks for memory allocation patterns
///
/// These benchmarks measure memory allocation efficiency and
/// identify opportunities for zero-allocation optimizations.
///
/// ## Performance Requirements
///
/// - Hot path operations: Zero allocations preferred
/// - Configuration operations: Minimal allocations
/// - Error handling: Avoid allocations in common cases
/// - Metrics collection: Pre-allocated data structures
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Benchmark allocation-heavy operations
    group.bench_function("string_allocations", |b| {
        b.iter(|| {
            // Simulate operations that might allocate strings
            let backend = format!("192.168.1.{}:8080", 1);
            let error = ProxyError::network(backend, "Connection failed", None);
            let _status = error.to_http_status();
        })
    });

    // Benchmark pre-allocated vs dynamic structures
    group.bench_function("pre_allocated_metrics", |b| {
        let metrics = MetricsCollector::new();
        b.iter(|| {
            // These operations should be allocation-free
            metrics.record_request();
            metrics.record_response(200);
            metrics.record_request_duration(Duration::from_millis(5));
        })
    });

    group.finish();
}

/// Benchmarks for scalability under increasing load
///
/// These benchmarks measure how performance degrades (or scales)
/// as the number of concurrent operations increases.
///
/// ## Performance Requirements
///
/// - Linear scaling: Performance should scale linearly with cores
/// - Graceful degradation: No cliff-edge performance drops
/// - Resource bounds: Memory usage should remain bounded
/// - Latency stability: P99 latency should remain stable under load
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    // Benchmark metrics collection under increasing load
    for ops_per_second in [1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(
            BenchmarkId::new("metrics_throughput", ops_per_second),
            ops_per_second,
            |b, &ops_per_second| {
                let metrics = Arc::new(MetricsCollector::new());
                let ops_per_iteration = ops_per_second / 1000; // Scale down for benchmark

                b.iter(|| {
                    for _ in 0..ops_per_iteration {
                        metrics.record_request();
                        metrics.record_response(200);
                    }
                });
            },
        );
    }

    // Benchmark configuration validation with increasing complexity
    for backend_count in [1, 10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("config_validation_backends", backend_count),
            backend_count,
            |b, &backend_count| {
                let mut config = ProxyConfig::default();
                config.backend_servers = (0..*backend_count)
                    .map(|i| format!("192.168.1.{}:8080", i % 255).parse().unwrap())
                    .collect();

                b.iter(|| {
                    let _validated = ProxyConfig::new(config.clone()).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks for startup and initialization performance
///
/// These benchmarks measure cold start performance including
/// configuration loading, service initialization, and readiness.
///
/// ## Performance Requirements
///
/// - Cold start time: < 100ms total
/// - Configuration loading: < 10ms
/// - Service initialization: < 50ms
/// - Memory pre-allocation: < 10MB initial footprint
fn bench_startup_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("startup");

    // Benchmark complete startup sequence
    group.bench_function("full_startup", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ProxyConfig::default();
            let _server = ProxyServer::new(config).await.unwrap();
        })
    });

    // Benchmark configuration loading from environment
    group.bench_function("config_loading", |b| {
        b.iter(|| {
            let _config = ProxyConfig::from_env().unwrap();
        })
    });

    // Benchmark metrics collector initialization
    group.bench_function("metrics_initialization", |b| {
        b.iter(|| {
            let _metrics = MetricsCollector::new();
        })
    });

    group.finish();
}

/// Integration benchmark that simulates realistic proxy workload
///
/// This benchmark simulates a more realistic workload pattern
/// with mixed operations, realistic timing, and concurrent access.
///
/// ## Workload Characteristics
///
/// - Request patterns: Bursty traffic with realistic inter-arrival times
/// - Response patterns: Mix of success/error responses  
/// - Configuration changes: Periodic validation and updates
/// - Metrics collection: Continuous background collection
fn bench_realistic_workload(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("realistic_workload");
    group.sample_size(10); // Fewer samples for integration benchmarks
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time

    group.bench_function("mixed_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let config = Arc::new(ProxyConfig::default());
            let metrics = Arc::new(MetricsCollector::new());
            let _service = ProxyService::new(config, Arc::clone(&metrics));

            // Simulate 1000 requests with realistic patterns
            for i in 0..1000 {
                metrics.record_request();

                // Simulate request processing delay
                if i % 100 == 0 {
                    tokio::time::sleep(Duration::from_micros(10)).await;
                }

                // Mix of success and error responses
                match i % 20 {
                    0 => metrics.record_response(404), // 5% 404s
                    1 => metrics.record_response(500), // 5% 500s
                    _ => metrics.record_response(200), // 90% success
                }

                // Record realistic response times
                let response_time = match i % 10 {
                    0..=6 => Duration::from_millis(1 + (i % 10)), // Fast responses
                    7..=8 => Duration::from_millis(50 + (i % 50)), // Medium responses
                    _ => Duration::from_millis(200),              // Slow responses
                };
                metrics.record_request_duration(response_time);
            }

            // Collect final metrics (simulates monitoring)
            let _snapshot = metrics.snapshot();
        })
    });

    group.finish();
}

// Configure criterion benchmark groups
criterion_group!(
    benches,
    bench_metrics_performance,
    bench_config_performance,
    bench_error_performance,
    bench_proxy_service,
    bench_concurrent_operations,
    bench_memory_efficiency,
    bench_scalability,
    bench_startup_performance,
    bench_realistic_workload
);

criterion_main!(benches);

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    /// Test that benchmark functions can run without panicking
    ///
    /// These tests ensure that all benchmark code is valid and
    /// can execute without errors. They don't measure performance,
    /// just correctness.
    #[test]
    fn test_metrics_operations() {
        let metrics = MetricsCollector::new();

        // Ensure all operations work
        metrics.record_request();
        metrics.record_response(200);
        metrics.record_error();
        metrics.record_request_duration(Duration::from_millis(10));
        metrics.record_upstream_selection_time(Duration::from_micros(50));

        let snapshot = metrics.snapshot();
        assert!(snapshot.total_requests > 0);

        let _prometheus = snapshot.to_prometheus_format();
    }

    #[test]
    fn test_config_operations() {
        let config = ProxyConfig::default();
        let _validated = ProxyConfig::new(config.clone()).unwrap();
        let _backends = config.effective_backends();
    }

    #[test]
    fn test_error_operations() {
        let error = ProxyError::network("127.0.0.1:8080", "Connection refused", None);
        let _status = error.to_http_status();
        let _temporary = error.is_temporary();

        let timeout_error = ProxyError::timeout(Duration::from_secs(30), "backend connection");
        assert_eq!(timeout_error.to_http_status(), 504);
    }

    #[tokio::test]
    async fn test_proxy_operations() {
        let config = ProxyConfig::default();
        let _server = ProxyServer::new(config).await.unwrap();
    }
}
