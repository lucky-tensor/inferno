//! Performance benchmarks for Phase 4 update propagation system
//!
//! These benchmarks measure the performance characteristics of the
//! self-sovereign update system including update creation, validation,
//! retry logic, and propagation operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use inferno_shared::service_discovery::{
    retry::{RetryConfig, RetryManager},
    updates::{NodeUpdate, UpdatePropagator},
    NodeInfo, NodeType,
};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark update creation performance
fn bench_update_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let propagator = UpdatePropagator::new();

    let nodes: Vec<NodeInfo> = (0..100)
        .map(|i| {
            NodeInfo::new(
                format!("backend-{}", i),
                format!("10.0.1.{}:3000", i + 1),
                9090,
                NodeType::Backend,
            )
        })
        .collect();

    let mut group = c.benchmark_group("update_creation");

    group.bench_function("single_update", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _update = propagator
                    .create_update(black_box(&nodes[0]))
                    .await
                    .unwrap();
            })
        })
    });

    group.bench_function("batch_10_updates", |b| {
        b.iter(|| {
            rt.block_on(async {
                for node in &nodes[0..10] {
                    let _update = propagator.create_update(black_box(node)).await.unwrap();
                }
            })
        })
    });

    group.bench_function("batch_100_updates", |b| {
        b.iter(|| {
            rt.block_on(async {
                for node in &nodes {
                    let _update = propagator.create_update(black_box(node)).await.unwrap();
                }
            })
        })
    });

    group.finish();
}

/// Benchmark update validation performance
fn bench_update_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let propagator = UpdatePropagator::new();
    let node = NodeInfo::new(
        "test-backend".to_string(),
        "10.0.1.1:3000".to_string(),
        9090,
        NodeType::Backend,
    );

    // Pre-create updates for validation benchmarks
    let updates: Vec<NodeUpdate> = rt.block_on(async {
        let mut updates = Vec::new();
        for _ in 0..1000 {
            updates.push(propagator.create_update(&node).await.unwrap());
        }
        updates
    });

    let mut group = c.benchmark_group("update_validation");

    group.bench_function("single_validation", |b| {
        b.iter(|| {
            let _ = propagator.validate_self_ownership(black_box(&updates[0]));
        })
    });

    group.bench_function("batch_100_validations", |b| {
        b.iter(|| {
            for update in &updates[0..100] {
                let _ = propagator.validate_self_ownership(black_box(update));
            }
        })
    });

    group.bench_function("batch_1000_validations", |b| {
        b.iter(|| {
            for update in &updates {
                let _ = propagator.validate_self_ownership(black_box(update));
            }
        })
    });

    group.finish();
}

/// Benchmark retry manager operations
fn bench_retry_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = RetryConfig {
        max_attempts: 4,
        base_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(10),
        jitter_factor: 0.1,
        retry_timeout: Duration::from_secs(5),
    };

    let retry_manager = RetryManager::with_config(config);
    let node = NodeInfo::new(
        "test-backend".to_string(),
        "10.0.1.1:3000".to_string(),
        9090,
        NodeType::Backend,
    );

    // Create test update
    let update = NodeUpdate {
        update_id: "test-update".to_string(),
        node: node.clone(),
        timestamp: current_timestamp(),
        version: 1,
        originator_id: "test-backend".to_string(),
        signature: Some("sig_test".to_string()),
    };

    let failed_peers = vec![
        "http://peer1:8080".to_string(),
        "http://peer2:8080".to_string(),
        "http://peer3:8080".to_string(),
    ];

    let mut group = c.benchmark_group("retry_operations");

    group.bench_function("queue_single_retry", |b| {
        b.iter(|| {
            rt.block_on(async {
                let retry_manager = RetryManager::with_config(RetryConfig::default());
                let _id = retry_manager
                    .queue_retry(black_box(update.clone()), black_box(failed_peers.clone()))
                    .await
                    .unwrap();
            })
        })
    });

    group.bench_function("queue_10_retries", |b| {
        b.iter(|| {
            rt.block_on(async {
                let retry_manager = RetryManager::with_config(RetryConfig::default());
                for i in 0..10 {
                    let mut update_clone = update.clone();
                    update_clone.update_id = format!("test-update-{}", i);
                    let _id = retry_manager
                        .queue_retry(black_box(update_clone), black_box(failed_peers.clone()))
                        .await
                        .unwrap();
                }
            })
        })
    });

    group.bench_function("get_metrics", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _metrics = retry_manager.get_metrics().await;
            })
        })
    });

    group.finish();
}

/// Benchmark update propagation with varying numbers of peers
fn bench_update_propagation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let propagator = UpdatePropagator::new();
    let node = NodeInfo::new(
        "test-backend".to_string(),
        "10.0.1.1:3000".to_string(),
        9090,
        NodeType::Backend,
    );

    let mut group = c.benchmark_group("update_propagation");

    // Test with different numbers of peers (all will fail since they're invalid URLs)
    for peer_count in [1, 5, 10, 25, 50].iter() {
        let peers: Vec<String> = (0..*peer_count)
            .map(|i| format!("http://invalid-peer-{}:8080", i))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("broadcast_to_peers", peer_count),
            &peers,
            |b, peers| {
                b.iter(|| {
                    rt.block_on(async {
                        let _results = propagator
                            .broadcast_self_update(black_box(&node), black_box(peers.clone()))
                            .await
                            .unwrap();
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark serialization performance
fn bench_serialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let propagator = UpdatePropagator::new();
    let node = NodeInfo::new(
        "test-backend".to_string(),
        "10.0.1.1:3000".to_string(),
        9090,
        NodeType::Backend,
    );

    let update = rt.block_on(async { propagator.create_update(&node).await.unwrap() });

    let mut group = c.benchmark_group("serialization");

    group.bench_function("serialize_to_json", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&update)).unwrap();
        })
    });

    group.bench_function("serialize_to_vec", |b| {
        b.iter(|| {
            let _bytes = serde_json::to_vec(black_box(&update)).unwrap();
        })
    });

    let json = serde_json::to_string(&update).unwrap();

    group.bench_function("deserialize_from_json", |b| {
        b.iter(|| {
            let _update: NodeUpdate = serde_json::from_str(black_box(&json)).unwrap();
        })
    });

    let bytes = serde_json::to_vec(&update).unwrap();

    group.bench_function("deserialize_from_bytes", |b| {
        b.iter(|| {
            let _update: NodeUpdate = serde_json::from_slice(black_box(&bytes)).unwrap();
        })
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let propagator = UpdatePropagator::new();

    let mut group = c.benchmark_group("memory_usage");

    group.bench_function("memory_intensive_updates", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut updates = Vec::new();

                // Create 1000 updates in memory
                for i in 0..1000 {
                    let node = NodeInfo::new(
                        format!("backend-{}", i),
                        format!("10.0.1.{}:3000", i % 255 + 1),
                        9090,
                        NodeType::Backend,
                    );

                    let update = propagator.create_update(&node).await.unwrap();
                    updates.push(black_box(update));
                }

                // Validate all updates
                for update in &updates {
                    let _ = propagator.validate_self_ownership(update);
                }
            })
        })
    });

    group.finish();
}

/// Helper function to get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

criterion_group!(
    benches,
    bench_update_creation,
    bench_update_validation,
    bench_retry_operations,
    bench_update_propagation,
    bench_serialization,
    bench_memory_usage
);
criterion_main!(benches);
