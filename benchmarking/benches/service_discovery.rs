//! Service Discovery Performance Benchmarks
//!
//! Comprehensive benchmarks for the service discovery protocol measuring:
//! - Backend registration latency
//! - Health check performance
//! - Concurrent backend access
//! - Memory usage patterns
//! - Scalability under load

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inferno_shared::service_discovery::{
    BackendRegistration, NodeVitals, ServiceDiscovery, ServiceDiscoveryConfig,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Creates a test backend registration
fn create_test_registration(id: &str, port: u16) -> BackendRegistration {
    BackendRegistration {
        id: id.to_string(),
        address: format!("127.0.0.1:{}", port),
        metrics_port: 9090,
    }
}

/// Creates test node vitals
fn create_test_vitals() -> NodeVitals {
    NodeVitals {
        ready: true,
        cpu_usage: Some(45.0),
        memory_usage: Some(55.0),
        active_requests: Some(5),
        avg_response_time_ms: Some(100.0),
        error_rate: Some(1.0),
        status_message: Some("healthy".to_string()),
    }
}

/// Benchmark backend registration performance
fn bench_backend_registration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backend_registration");

    // Test registration latency for different numbers of backends
    for backend_count in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*backend_count as u64));
        group.bench_with_input(
            BenchmarkId::new("register_backends", backend_count),
            backend_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let discovery = ServiceDiscovery::new();

                        for i in 0..count {
                            let registration = create_test_registration(
                                &format!("backend-{}", i),
                                3000 + i as u16,
                            );
                            discovery.register_backend(registration).await.unwrap();
                            black_box(());
                        }

                        black_box(discovery.backend_count().await)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark single backend registration latency
fn bench_single_registration_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("single_backend_registration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let discovery = ServiceDiscovery::new();
                let registration = create_test_registration("test-backend", 3000);
                discovery.register_backend(registration).await.unwrap();
                black_box(());
            });
        });
    });
}

/// Benchmark backend deregistration performance
fn bench_backend_deregistration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("backend_deregistration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let discovery = ServiceDiscovery::new();
                let registration = create_test_registration("test-backend", 3000);
                discovery.register_backend(registration).await.unwrap();

                discovery.remove_backend("test-backend").await.unwrap();
                black_box(());
            })
        });
    });
}

/// Benchmark concurrent backend access patterns
fn bench_concurrent_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_access");

    // Test concurrent reads with different numbers of threads
    for thread_count in [1, 4, 16, 64].iter() {
        group.throughput(Throughput::Elements(*thread_count as u64));
        group.bench_with_input(
            BenchmarkId::new("get_healthy_backends", thread_count),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    rt.block_on(async move {
                        let discovery = Arc::new(ServiceDiscovery::new());

                        // Register some backends first
                        for i in 0..10 {
                            let registration =
                                create_test_registration(&format!("backend-{}", i), 3000 + i);
                            discovery.register_backend(registration).await.unwrap();
                        }

                        // Spawn concurrent readers
                        let mut handles = Vec::new();
                        for _ in 0..threads {
                            let discovery = Arc::clone(&discovery);
                            let handle = tokio::spawn(async move {
                                black_box(discovery.get_healthy_backends().await)
                            });
                            handles.push(handle);
                        }

                        // Wait for all readers to complete
                        for handle in handles {
                            black_box(handle.await.unwrap());
                        }
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark service discovery scalability under load
fn bench_scalability_under_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("scalability");

    // Test performance with increasing numbers of backends
    for backend_count in [10, 100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*backend_count as u64));
        group.bench_with_input(
            BenchmarkId::new("mixed_operations", backend_count),
            backend_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let discovery = ServiceDiscovery::new();

                        // Register backends
                        for i in 0..count {
                            let registration = create_test_registration(
                                &format!("backend-{}", i),
                                3000 + (i % 10000) as u16,
                            );
                            discovery.register_backend(registration).await.unwrap();
                        }

                        // Perform mixed read operations
                        black_box(discovery.get_healthy_backends().await);
                        black_box(discovery.get_all_backends().await);
                        black_box(discovery.backend_count().await);
                        black_box(discovery.backend_count().await);

                        // Deregister some backends
                        for i in 0..(count / 4) {
                            discovery
                                .remove_backend(&format!("backend-{}", i))
                                .await
                                .unwrap();
                            black_box(());
                        }
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark health check configuration performance
fn bench_health_check_config(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("service_discovery_creation_with_config", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = ServiceDiscoveryConfig {
                    health_check_interval: Duration::from_secs(5),
                    health_check_timeout: Duration::from_secs(2),
                    failure_threshold: 3,
                    recovery_threshold: 2,
                    registration_timeout: Duration::from_secs(30),
                    enable_health_check_logging: false,
                    auth_mode: inferno_shared::service_discovery::AuthMode::Open,
                    shared_secret: None,
                };

                let discovery = ServiceDiscovery::with_config(config);

                // Register a backend to ensure the system is functional
                let registration = create_test_registration("test-backend", 3000);
                discovery.register_backend(registration).await.unwrap();
                black_box(());

                black_box(discovery.backend_count().await);
            });
        });
    });
}

/// Benchmark data structure serialization/deserialization
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let registration = create_test_registration("benchmark-backend", 3000);
    let vitals = create_test_vitals();

    group.bench_function("backend_registration_serialize", |b| {
        b.iter(|| black_box(serde_json::to_string(&registration).unwrap()));
    });

    group.bench_function("backend_registration_deserialize", |b| {
        let json = serde_json::to_string(&registration).unwrap();
        b.iter(|| {
            black_box(serde_json::from_str::<BackendRegistration>(&json).unwrap());
        });
    });

    group.bench_function("node_vitals_serialize", |b| {
        b.iter(|| black_box(serde_json::to_string(&vitals).unwrap()));
    });

    group.bench_function("node_vitals_deserialize", |b| {
        let json = serde_json::to_string(&vitals).unwrap();
        b.iter(|| {
            black_box(serde_json::from_str::<NodeVitals>(&json).unwrap());
        });
    });

    group.finish();
}

/// Benchmark memory efficiency patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_patterns");

    // Test memory usage with different backend counts
    for backend_count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*backend_count as u64));
        group.bench_with_input(
            BenchmarkId::new("memory_usage_scaling", backend_count),
            backend_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let discovery = ServiceDiscovery::new();

                        // Register many backends to test memory scaling
                        for i in 0..count {
                            let registration = create_test_registration(
                                &format!("backend-{}", i),
                                3000 + (i % 10000) as u16,
                            );
                            discovery.register_backend(registration).await.unwrap();
                        }

                        // Force some operations to ensure all data is in memory
                        black_box(discovery.get_all_backends().await);

                        // Return the discovery to prevent early deallocation
                        black_box(discovery)
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark error handling performance
fn bench_error_handling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("registration_validation_errors", |b| {
        b.iter(|| {
            rt.block_on(async {
                let discovery = ServiceDiscovery::new();

                // Test various validation errors
                let invalid_registrations = vec![
                    BackendRegistration {
                        id: "".to_string(), // Empty ID
                        address: "127.0.0.1:3000".to_string(),
                        metrics_port: 9090,
                    },
                    BackendRegistration {
                        id: "test".to_string(),
                        address: "".to_string(), // Empty address
                        metrics_port: 9090,
                    },
                    BackendRegistration {
                        id: "test".to_string(),
                        address: "127.0.0.1:3000".to_string(),
                        metrics_port: 0, // Invalid port
                    },
                    BackendRegistration {
                        id: "test".to_string(),
                        address: "invalid-address".to_string(), // Invalid format
                        metrics_port: 9090,
                    },
                ];

                for registration in invalid_registrations {
                    black_box(discovery.register_backend(registration).await.unwrap_err());
                }
            })
        });
    });
}

criterion_group!(
    service_discovery_benchmarks,
    bench_backend_registration,
    bench_single_registration_latency,
    bench_backend_deregistration,
    bench_concurrent_access,
    bench_scalability_under_load,
    bench_health_check_config,
    bench_serialization,
    bench_memory_patterns,
    bench_error_handling,
);

criterion_main!(service_discovery_benchmarks);
