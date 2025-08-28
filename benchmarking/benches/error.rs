use criterion::{criterion_group, criterion_main, Criterion};
use inferno_shared::InfernoError;
use std::time::Duration;

/// Benchmarks for error handling performance
///
/// These benchmarks measure the performance impact of error creation,
/// propagation, and handling in both success and failure scenarios.
fn bench_error_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling");

    // Benchmark error creation (various types)
    group.bench_function("error_network", |b| {
        b.iter(|| {
            let _error = InfernoError::network("127.0.0.1:8080", "Connection refused", None);
        })
    });

    group.bench_function("error_backend", |b| {
        b.iter(|| {
            let _error = InfernoError::backend("api.example.com", 500, "Internal Server Error");
        })
    });

    group.bench_function("error_timeout", |b| {
        let timeout = Duration::from_secs(30);
        b.iter(|| {
            let _error = InfernoError::timeout(timeout, "backend connection");
        })
    });

    // Benchmark error conversion
    group.bench_function("error_io_conversion", |b| {
        b.iter(|| {
            let io_error = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
            let _proxy_error: InfernoError = io_error.into();
        })
    });

    // Benchmark HTTP status mapping
    group.bench_function("error_http_status", |b| {
        let error = InfernoError::backend("backend", 404, "Not Found");
        b.iter(|| {
            let _status = match &error {
                InfernoError::Network { .. } => 502,
                InfernoError::Timeout { .. } => 504,
                _ => 500,
            };
        })
    });

    // Benchmark temporary classification
    group.bench_function("error_is_temporary", |b| {
        let error = InfernoError::timeout(Duration::from_secs(10), "request");
        b.iter(|| {
            let _temporary = matches!(error, InfernoError::Timeout { .. });
        })
    });

    group.finish();
}

criterion_group!(benches, bench_error_performance);
criterion_main!(benches);
