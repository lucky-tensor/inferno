use criterion::{criterion_group, criterion_main, Criterion};
use inferno_shared::{InfernoError, MetricsCollector};
use std::time::Duration;

/// Benchmarks for memory allocation patterns
///
/// These benchmarks measure memory allocation efficiency and
/// identify opportunities for zero-allocation optimizations.
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Benchmark allocation-heavy operations
    group.bench_function("string_allocations", |b| {
        b.iter(|| {
            // Simulate operations that might allocate strings
            let backend = format!("192.168.1.{}:8080", 1);
            let error = InfernoError::network(backend.clone(), "Connection failed", None);
            let _status = match &error {
                InfernoError::Network { .. } => 502,
                InfernoError::Timeout { .. } => 504,
                _ => 500,
            };
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

criterion_group!(benches, bench_memory_efficiency);
criterion_main!(benches);