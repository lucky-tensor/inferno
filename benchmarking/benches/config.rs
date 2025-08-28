use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use inferno_proxy::ProxyConfig;

/// Benchmarks for configuration parsing and validation
///
/// These benchmarks measure the performance of configuration operations
/// which occur during startup and potentially during hot reloading.
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
        let config = ProxyConfig {
            backend_servers: vec![
                "192.168.1.1:8080".parse().unwrap(),
                "192.168.1.2:8080".parse().unwrap(),
                "192.168.1.3:8080".parse().unwrap(),
                "192.168.1.4:8080".parse().unwrap(),
                "192.168.1.5:8080".parse().unwrap(),
            ],
            ..Default::default()
        };

        b.iter(|| {
            let _backends = config.effective_backends();
        })
    });

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

criterion_group!(benches, bench_config_performance);
criterion_main!(benches);
