use criterion::{criterion_group, criterion_main, Criterion};
use inferno_proxy::{ProxyConfig, ProxyServer, ProxyService};
use inferno_shared::MetricsCollector;
use std::sync::Arc;

/// Benchmarks for proxy service creation and operations
///
/// These benchmarks measure the performance of core proxy operations
/// including service creation, request routing logic, and cleanup.
fn bench_proxy_service(c: &mut Criterion) {
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
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.iter(|| {
            let config = config.clone();
            rt.block_on(async {
                let _server = ProxyServer::new(config).await.unwrap();
            })
        })
    });

    group.finish();
}

criterion_group!(benches, bench_proxy_service);
criterion_main!(benches);
