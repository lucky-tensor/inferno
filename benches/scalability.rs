use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use inferno_proxy::ProxyConfig;
use inferno_shared::MetricsCollector;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmarks for scalability under increasing load
///
/// These benchmarks measure how performance degrades (or scales)
/// as the number of concurrent operations increases.
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
                let config = ProxyConfig {
                    backend_servers: (0..backend_count)
                        .map(|i| format!("192.168.1.{}:8080", i % 255).parse().unwrap())
                        .collect(),
                    ..Default::default()
                };

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
fn bench_startup_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("startup");

    // Benchmark complete startup sequence
    group.bench_function("full_startup", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = ProxyConfig::default();
                let _server = inferno_proxy::ProxyServer::new(config).await.unwrap();
            })
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
fn bench_realistic_workload(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("realistic_workload");
    group.sample_size(10); // Fewer samples for integration benchmarks
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time

    group.bench_function("mixed_operations", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = Arc::new(ProxyConfig::default());
                let metrics = Arc::new(MetricsCollector::new());
                let _service = inferno_proxy::ProxyService::new(config, Arc::clone(&metrics));

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
        })
    });

    group.finish();
}

criterion_group!(benches, bench_scalability, bench_startup_performance, bench_realistic_workload);
criterion_main!(benches);