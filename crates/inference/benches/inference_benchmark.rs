use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inferno_inference::inference::EngineType;
use inferno_inference::{create_engine, create_math_test_request, InfernoConfig};
use std::path::PathBuf;
use tokio::runtime::Runtime;

struct BenchmarkSetup {
    runtime: Runtime,
    config: InfernoConfig,
}

impl BenchmarkSetup {
    fn new() -> Self {
        let runtime = Runtime::new().unwrap();

        // Try to use model path from environment or use a default
        let model_path =
            std::env::var("BENCH_MODEL_PATH").unwrap_or_else(|_| "/tmp/test-model".to_string());

        let config = InfernoConfig {
            model_path,
            model_name: "benchmark-model".to_string(),
            ..Default::default()
        };

        Self { runtime, config }
    }
}

fn bench_inference_latency(c: &mut Criterion) {
    let setup = BenchmarkSetup::new();

    let mut group = c.benchmark_group("inference_latency");

    // Test different prompt lengths
    let prompts = vec![
        ("short", "What is 2+2?"),
        ("medium", "Explain the concept of machine learning in simple terms."),
        ("long", "Write a detailed explanation of how neural networks work, including forward propagation, backpropagation, and gradient descent. Make sure to cover the mathematical foundations and practical applications."),
    ];

    for (name, prompt) in prompts {
        group.bench_with_input(BenchmarkId::new("cpu", name), &prompt, |b, prompt| {
            b.to_async(&setup.runtime).iter(|| async {
                let mut engine = create_engine(EngineType::CandleCpu);
                engine.initialize(setup.config.clone()).await.unwrap();

                let mut request = create_math_test_request();
                request.prompt = prompt.to_string();

                let response = engine.process(black_box(request)).await.unwrap();
                black_box(response)
            });
        });

        #[cfg(feature = "candle-cuda")]
        group.bench_with_input(BenchmarkId::new("cuda", name), &prompt, |b, prompt| {
            b.to_async(&setup.runtime).iter(|| async {
                let mut engine = create_engine(EngineType::CandleCuda);
                engine.initialize(setup.config.clone()).await.unwrap();

                let mut request = create_math_test_request();
                request.prompt = prompt.to_string();

                let response = engine.process(black_box(request)).await.unwrap();
                black_box(response)
            });
        });
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let setup = BenchmarkSetup::new();

    let mut group = c.benchmark_group("throughput");

    // Set throughput measurement
    group.throughput(Throughput::Elements(1));

    let batch_sizes = vec![1, 5, 10];

    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("cpu_batch", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&setup.runtime).iter(|| async {
                    let mut engine = create_engine(EngineType::CandleCpu);
                    engine.initialize(setup.config.clone()).await.unwrap();

                    let mut results = Vec::new();
                    for i in 0..batch_size {
                        let mut request = create_math_test_request();
                        request.prompt = format!("Calculate {i} + {i}");

                        let response = engine.process(black_box(request)).await.unwrap();
                        results.push(response);
                    }
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn bench_model_loading(c: &mut Criterion) {
    let setup = BenchmarkSetup::new();

    let mut group = c.benchmark_group("model_loading");

    group.bench_function("cpu_load", |b| {
        b.to_async(&setup.runtime).iter(|| async {
            let mut engine = create_engine(EngineType::CandleCpu);
            engine
                .initialize(black_box(setup.config.clone()))
                .await
                .unwrap();
            black_box(engine)
        });
    });

    #[cfg(feature = "candle-cuda")]
    group.bench_function("cuda_load", |b| {
        b.to_async(&setup.runtime).iter(|| async {
            let mut engine = create_engine(EngineType::CandleCuda);
            engine
                .initialize(black_box(setup.config.clone()))
                .await
                .unwrap();
            black_box(engine)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_inference_latency,
    bench_throughput,
    bench_model_loading
);
criterion_main!(benches);
