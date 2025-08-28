use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use inferno_shared::MetricsCollector;
use std::sync::Arc;
use std::time::Duration;

/// Benchmarks for concurrent operations
///
/// These benchmarks measure performance under concurrent load,
/// simulating real-world usage patterns with multiple threads
/// updating metrics and handling requests simultaneously.
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
                            let metrics: Arc<MetricsCollector> = Arc::clone(&metrics);
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

criterion_group!(benches, bench_concurrent_operations);
criterion_main!(benches);