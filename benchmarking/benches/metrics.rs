use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use inferno_shared::MetricsCollector;
use std::sync::Arc;
use std::time::Duration;

/// Benchmarks for metrics collection performance
///
/// These benchmarks measure the performance impact of metrics collection
/// on request processing, ensuring that observability doesn't significantly
/// impact proxy performance.
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

criterion_group!(benches, bench_metrics_performance);
criterion_main!(benches);
