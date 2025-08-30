//! Performance benchmarks for input validation and sanitization
//!
//! This benchmark suite measures the performance characteristics of the
//! input validation and sanitization system under various conditions including
//! different input sizes, character sets, and validation complexity.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inferno_shared::service_discovery::validation::{
    validate_address, validate_and_sanitize_node_info, validate_capabilities, validate_node_id,
    validate_port,
};
use inferno_shared::service_discovery::{NodeInfo, NodeType};

/// Generate test node IDs of various lengths and character sets
fn generate_node_ids() -> Vec<(String, String)> {
    vec![
        ("short".to_string(), "node1".to_string()),
        ("medium".to_string(), "backend-service-001".to_string()),
        ("long".to_string(), format!("{}-{}", "a".repeat(50), "backend")),
        ("max_length".to_string(), "b".repeat(128)),
        (
            "with_whitespace".to_string(),
            "  node-with-spaces  ".to_string(),
        ),
        ("unicode".to_string(), "node-测试-123".to_string()),
        ("special_chars".to_string(), "node_test.123-final".to_string()),
    ]
}

/// Generate test addresses of various formats and complexity
fn generate_addresses() -> Vec<(String, String)> {
    vec![
        ("ipv4_simple".to_string(), "127.0.0.1:8080".to_string()),
        ("ipv4_standard".to_string(), "192.168.1.100:3000".to_string()),
        ("ipv6_simple".to_string(), "[::1]:8080".to_string()),
        (
            "ipv6_full".to_string(),
            "[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:9090".to_string(),
        ),
        ("hostname_simple".to_string(), "localhost:5000".to_string()),
        (
            "hostname_fqdn".to_string(),
            "backend.service.example.com:8080".to_string(),
        ),
        (
            "hostname_long".to_string(),
            format!("{}.example.com:3000", "sub".repeat(20)),
        ),
        (
            "with_whitespace".to_string(),
            "  192.168.1.1:8080  ".to_string(),
        ),
    ]
}

/// Generate test capability lists of various sizes and content
fn generate_capability_lists() -> Vec<(String, Vec<String>)> {
    vec![
        ("empty".to_string(), vec![]),
        ("single".to_string(), vec!["inference".to_string()]),
        (
            "typical".to_string(),
            vec![
                "inference".to_string(),
                "gpu_support".to_string(),
                "model_serving".to_string(),
            ],
        ),
        (
            "large".to_string(),
            (0..20)
                .map(|i| format!("capability_{}", i))
                .collect::<Vec<_>>(),
        ),
        (
            "max_size".to_string(),
            (0..32).map(|i| format!("cap{}", i)).collect::<Vec<_>>(),
        ),
        (
            "with_whitespace".to_string(),
            vec![
                "  inference  ".to_string(),
                " gpu_support ".to_string(),
                "model_serving".to_string(),
            ],
        ),
        (
            "long_names".to_string(),
            vec![
                "a".repeat(60),
                "very_long_capability_name_that_approaches_limit".to_string(),
            ],
        ),
    ]
}

/// Generate test NodeInfo structures with various characteristics
fn generate_node_infos() -> Vec<(String, NodeInfo)> {
    vec![
        (
            "minimal".to_string(),
            NodeInfo::new(
                "node1".to_string(),
                "127.0.0.1:3000".to_string(),
                9090,
                NodeType::Backend,
            ),
        ),
        (
            "typical_backend".to_string(),
            NodeInfo {
                id: "backend-service-001".to_string(),
                address: "192.168.1.100:8080".to_string(),
                metrics_port: 9090,
                node_type: NodeType::Backend,
                is_load_balancer: false,
                capabilities: vec![
                    "inference".to_string(),
                    "gpu_support".to_string(),
                    "model_serving".to_string(),
                ],
                last_updated: std::time::SystemTime::now(),
            },
        ),
        (
            "complex_proxy".to_string(),
            NodeInfo {
                id: "proxy-lb-east-001".to_string(),
                address: "proxy.service.example.com:8080".to_string(),
                metrics_port: 6100,
                node_type: NodeType::Proxy,
                is_load_balancer: true,
                capabilities: vec![
                    "load_balancing".to_string(),
                    "health_checking".to_string(),
                    "request_routing".to_string(),
                    "service_discovery".to_string(),
                    "metrics_collection".to_string(),
                    "circuit_breaking".to_string(),
                ],
                last_updated: std::time::SystemTime::now(),
            },
        ),
        (
            "with_whitespace".to_string(),
            NodeInfo {
                id: "  node-with-spaces  ".to_string(),
                address: " 10.0.1.5:3000 ".to_string(),
                metrics_port: 9090,
                node_type: NodeType::Backend,
                is_load_balancer: false,
                capabilities: vec![
                    " inference ".to_string(),
                    "  gpu_support  ".to_string(),
                ],
                last_updated: std::time::SystemTime::now(),
            },
        ),
        (
            "large_capability_list".to_string(),
            NodeInfo {
                id: "governator-main".to_string(),
                address: "governator.internal:5000".to_string(),
                metrics_port: 9090,
                node_type: NodeType::Governator,
                is_load_balancer: false,
                capabilities: (0..25).map(|i| format!("capability_{}", i)).collect(),
                last_updated: std::time::SystemTime::now(),
            },
        ),
    ]
}

/// Benchmark node ID validation performance
fn bench_node_id_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_node_id");

    let test_ids = generate_node_ids();

    for (name, node_id) in test_ids.iter() {
        group.throughput(Throughput::Bytes(node_id.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("validate", name),
            node_id,
            |b, id| {
                b.iter(|| {
                    let _ = validate_node_id(id);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark address validation performance
fn bench_address_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_address");

    let test_addresses = generate_addresses();

    for (name, address) in test_addresses.iter() {
        group.throughput(Throughput::Bytes(address.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("validate", name),
            address,
            |b, addr| {
                b.iter(|| {
                    let _ = validate_address(addr);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark port validation performance
fn bench_port_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_port");

    let test_ports = vec![
        1024, 3000, 8080, 9090, 6100, 5432, 27017, 65535,
    ];

    for port in test_ports.iter() {
        group.bench_with_input(
            BenchmarkId::new("validate", port),
            port,
            |b, &port| {
                b.iter(|| {
                    let _ = validate_port(port);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark capability validation performance
fn bench_capability_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_capabilities");

    let test_capability_lists = generate_capability_lists();

    for (name, capabilities) in test_capability_lists.iter() {
        let total_chars: usize = capabilities.iter().map(|c| c.len()).sum();
        group.throughput(Throughput::Bytes(total_chars as u64));
        group.bench_with_input(
            BenchmarkId::new("validate", name),
            capabilities,
            |b, caps| {
                b.iter(|| {
                    let _ = validate_capabilities(caps);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complete NodeInfo validation and sanitization
fn bench_node_info_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_node_info");

    let test_node_infos = generate_node_infos();

    for (name, node_info) in test_node_infos.iter() {
        // Estimate size for throughput calculation
        let size = node_info.id.len()
            + node_info.address.len()
            + node_info.capabilities.iter().map(|c| c.len()).sum::<usize>();
        
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("validate_and_sanitize", name),
            node_info,
            |b, node| {
                b.iter(|| {
                    let _ = validate_and_sanitize_node_info(node.clone());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark validation error handling paths
fn bench_validation_error_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_error_cases");

    // Test various error conditions to ensure error paths are performant
    let too_long_node_id = "x".repeat(200);
    let too_long_address = format!("{}:8080", "x".repeat(300));
    
    let error_cases = vec![
        ("empty_node_id", ""),
        ("too_long_node_id", too_long_node_id.as_str()),
        ("invalid_chars_node_id", "node@invalid#chars"),
        ("invalid_address", "not-an-address"),
        ("empty_address", ""),
        ("invalid_port_in_address", "host:99999"),
        ("too_long_address", too_long_address.as_str()),
    ];

    for (name, test_input) in error_cases.iter() {
        group.bench_with_input(
            BenchmarkId::new("error_case", name),
            test_input,
            |b, input| {
                b.iter(|| {
                    if name.contains("node_id") {
                        let _ = validate_node_id(input);
                    } else if name.contains("address") {
                        let _ = validate_address(input);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch validation scenarios
fn bench_batch_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_batch");

    // Simulate batch validation scenarios common in real applications
    let batch_sizes = [1, 10, 50, 100, 500];

    for &batch_size in batch_sizes.iter() {
        let node_infos: Vec<NodeInfo> = (0..batch_size)
            .map(|i| NodeInfo {
                id: format!("batch-node-{}", i),
                address: format!("192.168.1.{}:8080", (i % 254) + 1),
                metrics_port: 9090,
                node_type: match i % 3 {
                    0 => NodeType::Backend,
                    1 => NodeType::Proxy,
                    _ => NodeType::Governator,
                },
                is_load_balancer: i % 3 == 1,
                capabilities: vec![
                    "capability_a".to_string(),
                    format!("capability_{}", i),
                ],
                last_updated: std::time::SystemTime::now(),
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_validate", batch_size),
            &node_infos,
            |b, nodes| {
                b.iter(|| {
                    for node in nodes {
                        let _ = validate_and_sanitize_node_info(node.clone());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns in validation
fn bench_validation_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_memory");

    // Test cases designed to stress different memory allocation patterns
    let test_cases = vec![
        (
            "no_sanitization_needed",
            NodeInfo::new(
                "clean-node-id".to_string(),
                "192.168.1.1:8080".to_string(),
                9090,
                NodeType::Backend,
            ),
        ),
        (
            "minimal_sanitization",
            NodeInfo {
                id: "clean-node-id".to_string(),
                address: " 192.168.1.1:8080 ".to_string(), // Only address needs trimming
                metrics_port: 9090,
                node_type: NodeType::Backend,
                is_load_balancer: false,
                capabilities: vec!["inference".to_string()],
                last_updated: std::time::SystemTime::now(),
            },
        ),
        (
            "full_sanitization",
            NodeInfo {
                id: "  node-with-spaces  ".to_string(),
                address: " 192.168.1.1:8080 ".to_string(),
                metrics_port: 9090,
                node_type: NodeType::Backend,
                is_load_balancer: false,
                capabilities: vec![
                    "  inference  ".to_string(),
                    " gpu_support ".to_string(),
                    "model_serving".to_string(),
                ],
                last_updated: std::time::SystemTime::now(),
            },
        ),
    ];

    for (name, node_info) in test_cases.iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_pattern", name),
            node_info,
            |b, node| {
                b.iter(|| {
                    let _ = validate_and_sanitize_node_info(node.clone());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_node_id_validation,
    bench_address_validation,
    bench_port_validation,
    bench_capability_validation,
    bench_node_info_validation,
    bench_validation_error_cases,
    bench_batch_validation,
    bench_validation_memory_patterns,
);

criterion_main!(benches);