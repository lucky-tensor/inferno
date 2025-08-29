//! Performance benchmarks for consensus algorithms
//!
//! This benchmark suite measures the performance characteristics of the
//! consensus resolution algorithms under various conditions including
//! different peer counts, node counts, and conflict scenarios.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inferno_shared::service_discovery::consensus::ConsensusResolver;
use inferno_shared::service_discovery::types::{NodeType, PeerInfo};
use std::time::{Duration, SystemTime};
use tokio::runtime::Runtime;

/// Helper function to create test peer information
fn create_test_peer(
    id: &str,
    address: &str,
    metrics_port: u16,
    node_type: NodeType,
    timestamp_offset_secs: i64,
) -> PeerInfo {
    let timestamp = if timestamp_offset_secs >= 0 {
        SystemTime::now() + Duration::from_secs(timestamp_offset_secs as u64)
    } else {
        SystemTime::now() - Duration::from_secs((-timestamp_offset_secs) as u64)
    };

    PeerInfo {
        id: id.to_string(),
        address: address.to_string(),
        metrics_port,
        node_type,
        is_load_balancer: node_type == NodeType::Proxy,
        last_updated: timestamp,
    }
}

/// Generate peer responses with specified parameters for benchmarking
fn generate_peer_responses(
    num_peers: usize,
    nodes_per_peer: usize,
    conflict_rate: f64,
) -> Vec<Vec<PeerInfo>> {
    let mut responses = Vec::new();

    for peer_idx in 0..num_peers {
        let mut peer_response = Vec::new();

        for node_idx in 0..nodes_per_peer {
            let node_id = format!("node-{}", node_idx);
            
            // Create conflicts based on conflict_rate
            let has_conflict = (peer_idx as f64 / num_peers as f64) < conflict_rate;
            let address_variant = if has_conflict {
                peer_idx % 3  // Create up to 3 different versions
            } else {
                0  // All peers agree on version 0
            };

            let address = format!("10.0.{}.{}:3000", node_idx + 1, address_variant + 1);
            let timestamp_offset = if has_conflict { 
                (peer_idx as i64 - 2) * 5  // Different timestamps for conflicts
            } else { 
                0  // Same timestamp for no conflicts
            };

            let node_type = match node_idx % 3 {
                0 => NodeType::Backend,
                1 => NodeType::Proxy,
                _ => NodeType::Governator,
            };

            let peer = create_test_peer(
                &node_id,
                &address,
                9090 + node_idx as u16,
                node_type,
                timestamp_offset,
            );

            peer_response.push(peer);
        }

        responses.push(peer_response);
    }

    responses
}

/// Benchmark consensus resolution with varying peer counts
fn bench_consensus_peer_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let resolver = ConsensusResolver::new();

    let mut group = c.benchmark_group("consensus_peer_scaling");
    
    // Test with different numbers of peers (no conflicts for baseline)
    for num_peers in [2, 5, 10, 20, 50, 100].iter() {
        let peer_responses = generate_peer_responses(*num_peers, 10, 0.0);
        
        group.throughput(Throughput::Elements(*num_peers as u64));
        group.bench_with_input(
            BenchmarkId::new("no_conflicts", num_peers),
            &peer_responses,
            |b, responses| {
                b.to_async(&rt).iter(|| async {
                    let _ = resolver.resolve_consensus(responses.clone()).await.unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark consensus resolution with varying node counts per peer
fn bench_consensus_node_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let resolver = ConsensusResolver::new();

    let mut group = c.benchmark_group("consensus_node_scaling");
    
    // Test with different numbers of nodes per peer
    for nodes_per_peer in [1, 5, 10, 25, 50, 100, 250, 500].iter() {
        let peer_responses = generate_peer_responses(5, *nodes_per_peer, 0.0);
        
        group.throughput(Throughput::Elements(*nodes_per_peer as u64 * 5)); // total nodes
        group.bench_with_input(
            BenchmarkId::new("no_conflicts", nodes_per_peer),
            &peer_responses,
            |b, responses| {
                b.to_async(&rt).iter(|| async {
                    let _ = resolver.resolve_consensus(responses.clone()).await.unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark consensus resolution with varying conflict rates
fn bench_consensus_conflict_handling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let resolver = ConsensusResolver::new();

    let mut group = c.benchmark_group("consensus_conflict_handling");
    
    // Test with different conflict rates
    for conflict_rate in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0].iter() {
        let peer_responses = generate_peer_responses(10, 20, *conflict_rate);
        
        group.bench_with_input(
            BenchmarkId::new("conflict_rate", (conflict_rate * 100.0) as u32),
            &peer_responses,
            |b, responses| {
                b.to_async(&rt).iter(|| async {
                    let _ = resolver.resolve_consensus(responses.clone()).await.unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark consensus resolution memory usage patterns
fn bench_consensus_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let resolver = ConsensusResolver::new();

    let mut group = c.benchmark_group("consensus_memory_usage");
    
    // Test scenarios designed to stress memory allocation
    let scenarios = [
        ("small_uniform", generate_peer_responses(3, 5, 0.0)),
        ("medium_uniform", generate_peer_responses(10, 20, 0.0)),
        ("large_uniform", generate_peer_responses(25, 50, 0.0)),
        ("small_conflicts", generate_peer_responses(3, 5, 0.8)),
        ("medium_conflicts", generate_peer_responses(10, 20, 0.8)),
        ("large_conflicts", generate_peer_responses(25, 50, 0.8)),
    ];

    for (scenario_name, peer_responses) in scenarios.iter() {
        let total_nodes = peer_responses.len() * peer_responses[0].len();
        group.throughput(Throughput::Elements(total_nodes as u64));
        
        group.bench_with_input(
            BenchmarkId::new("scenario", scenario_name),
            peer_responses,
            |b, responses| {
                b.to_async(&rt).iter(|| async {
                    let _ = resolver.resolve_consensus(responses.clone()).await.unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark worst-case scenarios for consensus resolution
fn bench_consensus_worst_case(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let resolver = ConsensusResolver::new();

    let mut group = c.benchmark_group("consensus_worst_case");
    
    // Worst case: Every peer has different information for every node
    let worst_case_responses = {
        let num_peers = 10;
        let nodes_per_peer = 50;
        let mut responses = Vec::new();

        for peer_idx in 0..num_peers {
            let mut peer_response = Vec::new();
            
            for node_idx in 0..nodes_per_peer {
                // Each peer sees different version of each node
                let peer = create_test_peer(
                    &format!("node-{}", node_idx),
                    &format!("10.0.{}.{}:3000", node_idx + 1, peer_idx + 1),
                    9090 + node_idx as u16,
                    NodeType::Backend,
                    peer_idx as i64 * 10, // Different timestamps
                );
                peer_response.push(peer);
            }
            
            responses.push(peer_response);
        }
        
        responses
    };

    group.bench_function("maximum_conflicts", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = resolver.resolve_consensus(worst_case_responses.clone()).await.unwrap();
        });
    });

    // Best case: All peers agree on everything
    let best_case_responses = generate_peer_responses(10, 50, 0.0);
    
    group.bench_function("zero_conflicts", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = resolver.resolve_consensus(best_case_responses.clone()).await.unwrap();
        });
    });

    group.finish();
}

/// Benchmark consensus resolver creation and setup
fn bench_consensus_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus_setup");

    group.bench_function("resolver_creation", |b| {
        b.iter(|| {
            let _ = ConsensusResolver::new();
        });
    });

    group.finish();
}

/// Benchmark realistic distributed system scenarios
fn bench_consensus_realistic_scenarios(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let resolver = ConsensusResolver::new();

    let mut group = c.benchmark_group("consensus_realistic");
    
    // Scenario 1: Typical small cluster (3 proxies, 10 backends)
    let small_cluster = generate_peer_responses(3, 13, 0.1);
    
    group.bench_function("small_cluster", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = resolver.resolve_consensus(small_cluster.clone()).await.unwrap();
        });
    });

    // Scenario 2: Medium cluster with occasional conflicts (5 proxies, 25 backends)
    let medium_cluster = generate_peer_responses(5, 30, 0.15);
    
    group.bench_function("medium_cluster", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = resolver.resolve_consensus(medium_cluster.clone()).await.unwrap();
        });
    });

    // Scenario 3: Large cluster during network partition (10 proxies, 100+ nodes, high conflicts)
    let partition_scenario = generate_peer_responses(10, 120, 0.4);
    
    group.bench_function("network_partition", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = resolver.resolve_consensus(partition_scenario.clone()).await.unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_consensus_setup,
    bench_consensus_peer_scaling,
    bench_consensus_node_scaling,
    bench_consensus_conflict_handling,
    bench_consensus_memory_usage,
    bench_consensus_worst_case,
    bench_consensus_realistic_scenarios,
);

criterion_main!(benches);