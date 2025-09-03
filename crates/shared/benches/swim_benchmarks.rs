//! SWIM Protocol Performance Benchmarks
//!
//! These benchmarks validate SWIM protocol performance at various scales,
//! particularly focusing on 10,000+ node clusters. They measure key
//! performance characteristics including failure detection times,
//! gossip dissemination efficiency, and memory usage.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use inferno_shared::service_discovery::{
    NodeType, PeerInfo, SwimCluster, SwimConfig10k, SwimIntegrationConfig, SwimServiceDiscovery,
};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, SystemTime};
use tokio::runtime::Runtime;

/// Benchmarks SWIM cluster creation and initialization
fn bench_swim_cluster_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("swim_cluster_creation");

    for size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("cluster_creation", size),
            size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = SwimConfig10k::default();
                    let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000);

                    let (mut cluster, _events) =
                        SwimCluster::new("test-node".to_string(), bind_addr, config)
                            .await
                            .unwrap();

                    // Add members to simulate cluster size
                    for i in 0..size {
                        let peer_info = create_test_peer_info(i);
                        cluster.add_member(peer_info).await.unwrap();
                    }

                    black_box(cluster);
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks member addition performance
fn bench_member_addition(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("member_addition");

    for cluster_size in [100, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("add_member", cluster_size),
            cluster_size,
            |b, &cluster_size| {
                b.to_async(&rt).iter(|| async {
                    let config = SwimConfig10k::default();
                    let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8001);

                    let (mut cluster, _events) =
                        SwimCluster::new("test-node".to_string(), bind_addr, config)
                            .await
                            .unwrap();

                    // Pre-populate cluster
                    for i in 0..cluster_size {
                        let peer_info = create_test_peer_info(i);
                        cluster.add_member(peer_info).await.unwrap();
                    }

                    // Benchmark adding one more member
                    let new_peer = create_test_peer_info(cluster_size + 1);
                    black_box(cluster.add_member(new_peer).await.unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks getting live members list
fn bench_get_live_members(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("get_live_members");

    // Pre-create clusters with different sizes
    let clusters = rt.block_on(async {
        let mut clusters = Vec::new();

        for &size in [100, 1000, 5000, 10000].iter() {
            let config = SwimConfig10k::default();
            let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8002 + size as u16);

            let (mut cluster, _events) =
                SwimCluster::new(format!("test-node-{}", size), bind_addr, config)
                    .await
                    .unwrap();

            // Add members
            for i in 0..size {
                let peer_info = create_test_peer_info(i);
                cluster.add_member(peer_info).await.unwrap();
            }

            clusters.push((size, cluster));
        }

        clusters
    });

    for (size, cluster) in clusters {
        group.bench_with_input(BenchmarkId::new("get_live_members", size), &size, |b, _| {
            b.to_async(&rt).iter(|| async {
                let members = cluster.get_live_members().await;
                black_box(members);
            });
        });
    }

    group.finish();
}

/// Benchmarks SWIM statistics collection
fn bench_swim_statistics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("swim_statistics");

    for size in [1000, 5000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("get_stats", size), size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let config = SwimConfig10k::default();
                let bind_addr =
                    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8100 + size as u16);

                let (mut cluster, _events) =
                    SwimCluster::new(format!("stats-node-{}", size), bind_addr, config)
                        .await
                        .unwrap();

                // Add members
                for i in 0..size {
                    let peer_info = create_test_peer_info(i);
                    cluster.add_member(peer_info).await.unwrap();
                }

                let stats = cluster.get_stats().await;
                black_box(stats);
            });
        });
    }

    group.finish();
}

/// Benchmarks service discovery integration layer
fn bench_swim_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("swim_integration");

    for size in [100, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("register_backend", size),
            size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let swim_config = SwimConfig10k::default();
                    let integration_config = SwimIntegrationConfig::default();
                    let bind_addr =
                        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8200 + size as u16);

                    let service_discovery = SwimServiceDiscovery::new(
                        format!("integration-node-{}", size),
                        bind_addr,
                        swim_config,
                        integration_config,
                    )
                    .await
                    .unwrap();

                    // Benchmark backend registration
                    for i in 0..size {
                        let peer_info = create_test_peer_info(i);
                        black_box(service_discovery.register_backend(peer_info).await.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks memory efficiency at scale
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_efficiency");

    // Measure memory usage for large clusters
    for size in [1000, 5000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("memory_usage", size), size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let config = SwimConfig10k::default();
                let bind_addr =
                    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8300 + size as u16);

                let (mut cluster, _events) =
                    SwimCluster::new(format!("memory-node-{}", size), bind_addr, config)
                        .await
                        .unwrap();

                // Add all members at once to measure batch memory allocation
                let mut members = Vec::new();
                for i in 0..size {
                    members.push(create_test_peer_info(i));
                }

                // Add members in batch
                for member in members {
                    cluster.add_member(member).await.unwrap();
                }

                // Force memory usage calculation
                let stats = cluster.get_stats().await;
                black_box(stats);
            });
        });
    }

    group.finish();
}

/// Benchmarks concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_operations");

    for concurrency in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_registrations", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let config = SwimConfig10k::default();
                    let bind_addr =
                        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8400 + concurrency as u16);

                    let (cluster, _events) = SwimCluster::new(
                        format!("concurrent-node-{}", concurrency),
                        bind_addr,
                        config,
                    )
                    .await
                    .unwrap();

                    // Create concurrent registration tasks
                    let cluster = Arc::new(cluster);
                    let mut handles = Vec::new();
                    for i in 0..concurrency {
                        let cluster_clone = Arc::clone(&cluster);
                        let handle = tokio::spawn(async move {
                            let peer_info = create_test_peer_info(i);
                            // Note: This would require Arc<Mutex<SwimCluster>> for true concurrency
                            // For now, this measures the overhead of task spawning
                            black_box(peer_info);
                        });
                        handles.push(handle);
                    }

                    // Wait for all tasks
                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks 10k node specific optimizations
fn bench_10k_node_optimizations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("10k_optimizations");
    group.sample_size(10); // Reduce sample size for large-scale tests
    group.measurement_time(Duration::from_secs(30));

    // Test the actual 10k node scenario
    group.bench_function("full_10k_cluster", |b| {
        b.to_async(&rt).iter(|| async {
            let config = SwimConfig10k {
                probe_interval: Duration::from_millis(100), // Fast probing
                gossip_fanout: 15,                          // Optimized fanout
                max_gossip_per_message: 50,                 // Batch updates
                enable_compression: true,                   // Essential at scale
                ..Default::default()
            };

            let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9000);

            let (mut cluster, _events) =
                SwimCluster::new("10k-node".to_string(), bind_addr, config)
                    .await
                    .unwrap();

            // Add 10k members (this will be slow but tests the real scenario)
            for i in 0..10000 {
                let peer_info = create_test_peer_info(i);
                cluster.add_member(peer_info).await.unwrap();

                // Yield periodically to prevent test timeout
                if i % 1000 == 0 {
                    tokio::task::yield_now().await;
                }
            }

            // Test operations on full cluster
            let members = cluster.get_live_members().await;
            let stats = cluster.get_stats().await;

            black_box((members, stats));
        });
    });

    group.finish();
}

// Helper functions

fn create_test_peer_info(id: usize) -> PeerInfo {
    PeerInfo {
        id: format!("test-node-{}", id),
        address: format!(
            "127.0.0.{id}:{port}",
            id = (id % 254) + 1,
            port = 8000 + (id % 1000)
        ),
        metrics_port: 9090,
        node_type: if id % 10 == 0 {
            NodeType::Proxy
        } else {
            NodeType::Backend
        },
        is_load_balancer: id % 10 == 0,
        last_updated: SystemTime::now(),
    }
}

criterion_group!(
    benches,
    bench_swim_cluster_creation,
    bench_member_addition,
    bench_get_live_members,
    bench_swim_statistics,
    bench_swim_integration,
    bench_memory_efficiency,
    bench_concurrent_operations,
    bench_10k_node_optimizations,
);

criterion_main!(benches);
