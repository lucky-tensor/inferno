//! Performance benchmarks for HTTP/3 service discovery client
//!
//! These benchmarks measure the performance characteristics of the HTTP/3
//! client for service discovery operations, including client creation,
//! configuration, and metrics operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use inferno_shared::service_discovery::http3_client::{
    Http3ClientConfig, Http3Metrics, Http3ServiceDiscoveryClient,
};
use inferno_shared::service_discovery::{NodeInfo, NodeType};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark HTTP/3 client creation performance
fn bench_client_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("http3_client_creation");

    group.bench_function("default_config", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _client = Http3ServiceDiscoveryClient::new(black_box(Duration::from_secs(5)))
                    .await
                    .unwrap();
            })
        })
    });

    group.bench_function("custom_config", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = Http3ClientConfig::new(black_box(Duration::from_secs(5)));
                let _client = Http3ServiceDiscoveryClient::with_config(config)
                    .await
                    .unwrap();
            })
        })
    });

    group.bench_function("high_throughput_config", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = Http3ClientConfig::high_throughput(black_box(Duration::from_secs(10)));
                let _client = Http3ServiceDiscoveryClient::with_config(config)
                    .await
                    .unwrap();
            })
        })
    });

    group.bench_function("testing_config", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = Http3ClientConfig::for_testing();
                let _client = Http3ServiceDiscoveryClient::with_config(config)
                    .await
                    .unwrap();
            })
        })
    });

    group.finish();
}

/// Benchmark HTTP/3 configuration operations
fn bench_config_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("http3_config");

    group.bench_function("config_creation", |b| {
        b.iter(|| {
            let _config = Http3ClientConfig::new(black_box(Duration::from_secs(5)));
        })
    });

    group.bench_function("config_cloning", |b| {
        let config = Http3ClientConfig::new(Duration::from_secs(5));
        b.iter(|| {
            let _cloned = black_box(&config).clone();
        })
    });

    group.bench_function("high_throughput_creation", |b| {
        b.iter(|| {
            let _config = Http3ClientConfig::high_throughput(black_box(Duration::from_secs(10)));
        })
    });

    group.bench_function("testing_config_creation", |b| {
        b.iter(|| {
            let _config = Http3ClientConfig::for_testing();
        })
    });

    group.finish();
}

/// Benchmark HTTP/3 metrics operations
fn bench_metrics_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("http3_metrics");

    group.bench_function("metrics_creation", |b| {
        b.iter(|| {
            let _metrics = Http3Metrics::default();
        })
    });

    group.bench_function("metrics_cloning", |b| {
        let metrics = Http3Metrics::default();
        b.iter(|| {
            let _cloned = black_box(&metrics).clone();
        })
    });

    group.bench_function("metrics_update", |b| {
        b.iter(|| {
            let metrics = Http3Metrics {
                zero_rtt_connections: black_box(1),
                bytes_sent: black_box(1024),
                bytes_received: black_box(2048),
                ..Default::default()
            };
            black_box(metrics);
        })
    });

    group.bench_function("metrics_batch_update", |b| {
        b.iter(|| {
            let mut metrics = Http3Metrics {
                zero_rtt_connections: 0,
                connection_migrations: 0,
                stream_resets: 0,
                avg_rtt_us: 0,
                retransmitted_packets: 0,
                active_connections: 0,
                bytes_sent: 0,
                bytes_received: 0,
            };
            for i in 0..10 {
                metrics.zero_rtt_connections = black_box(i as u64);
                metrics.connection_migrations = black_box(i as u64);
                metrics.stream_resets = black_box(i as u64);
                metrics.avg_rtt_us = black_box(500);
                metrics.retransmitted_packets = black_box(i as u64);
                metrics.active_connections = black_box(5);
                metrics.bytes_sent = black_box(i as u64 * 1024);
                metrics.bytes_received = black_box(i as u64 * 2048);
            }
            black_box(metrics);
        })
    });

    group.finish();
}

/// Benchmark multiple client operations
fn bench_multiple_clients(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("http3_multiple_clients");

    for count in [1, 5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| {
                rt.block_on(async {
                    let mut clients = Vec::with_capacity(count);
                    for _ in 0..count {
                        let client = Http3ServiceDiscoveryClient::new(Duration::from_secs(5))
                            .await
                            .unwrap();
                        clients.push(client);
                    }
                    black_box(clients);
                })
            })
        });
    }

    group.finish();
}

/// Benchmark node info creation for HTTP/3 operations
fn bench_node_info_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("http3_node_info");

    group.bench_function("single_node", |b| {
        b.iter(|| {
            let _node = NodeInfo::new(
                black_box("backend-1".to_string()),
                black_box("10.0.1.5:3000".to_string()),
                black_box(9090),
                black_box(NodeType::Backend),
            );
        })
    });

    group.bench_function("batch_10_nodes", |b| {
        b.iter(|| {
            let mut nodes = Vec::with_capacity(10);
            for i in 0..10 {
                let node = NodeInfo::new(
                    format!("backend-{}", i),
                    format!("10.0.1.{}:3000", i + 1),
                    9090,
                    NodeType::Backend,
                );
                nodes.push(node);
            }
            black_box(nodes);
        })
    });

    group.bench_function("batch_100_nodes", |b| {
        b.iter(|| {
            let mut nodes = Vec::with_capacity(100);
            for i in 0..100 {
                let node = NodeInfo::new(
                    format!("backend-{}", i),
                    format!("10.0.1.{}:3000", i + 1),
                    9090,
                    NodeType::Backend,
                );
                nodes.push(node);
            }
            black_box(nodes);
        })
    });

    group.finish();
}

/// Benchmark configuration comparison
fn bench_config_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("http3_config_comparison");

    let default_config = Http3ClientConfig::new(Duration::from_secs(5));
    let high_throughput = Http3ClientConfig::high_throughput(Duration::from_secs(5));
    let testing_config = Http3ClientConfig::for_testing();

    group.bench_function("default_vs_high_throughput", |b| {
        b.iter(|| {
            let _default_streams = black_box(default_config.max_concurrent_streams);
            let _high_streams = black_box(high_throughput.max_concurrent_streams);
            let _comparison = _default_streams < _high_streams;
        })
    });

    group.bench_function("all_configs_comparison", |b| {
        b.iter(|| {
            let configs = vec![&default_config, &high_throughput, &testing_config];
            let mut max_streams = 0u64;
            for config in configs {
                if config.max_concurrent_streams > max_streams {
                    max_streams = config.max_concurrent_streams;
                }
            }
            black_box(max_streams);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_client_creation,
    bench_config_operations,
    bench_metrics_operations,
    bench_multiple_clients,
    bench_node_info_creation,
    bench_config_comparison
);
criterion_main!(benches);
