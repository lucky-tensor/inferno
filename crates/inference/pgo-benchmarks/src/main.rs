// Simple configuration validator for PGO benchmarks

use std::path::PathBuf;

struct PGOBenchmarkConfig {
    model_path: PathBuf,
    original_binary: PathBuf,
    pgo_binary: PathBuf,
}

impl PGOBenchmarkConfig {
    fn new() -> Option<Self> {
        let model_path = std::env::var("BENCH_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                PathBuf::from(format!(
                    "{}/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0",
                    home
                ))
            });

        // Find workspace root using CARGO_MANIFEST_DIR
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| std::env::current_dir().unwrap().to_string_lossy().to_string());

        // pgo-benchmarks is at: workspace/crates/inference/pgo-benchmarks
        // So workspace root is: manifest_dir + "../../.."
        let workspace_root = PathBuf::from(&manifest_dir)
            .parent()  // crates/inference/
            .and_then(|p| p.parent())  // crates/
            .and_then(|p| p.parent())  // workspace root
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("../../.."));

        // Expected binary locations (built by PGO script)
        let original_binary = workspace_root.join("target/release/examples/concurrent_inference.original");
        let pgo_binary = workspace_root.join("target/release/examples/concurrent_inference.pgo");

        // Check if required files exist
        if !model_path.exists() {
            eprintln!("❌ Model not found at {:?}", model_path);
            eprintln!("   Set BENCH_MODEL_PATH environment variable or download the model:");
            eprintln!("   inferno download TinyLlama_TinyLlama-1.1B-Chat-v1.0");
            return None;
        }

        if !original_binary.exists() {
            eprintln!(
                "❌ Original binary not found at {:?}",
                original_binary
            );
            eprintln!("   Run the PGO script first from workspace root:");
            eprintln!("   ./crates/inference/benches/build-pgo-concurrent.sh");
            return None;
        }

        if !pgo_binary.exists() {
            eprintln!(
                "❌ PGO-optimized binary not found at {:?}",
                pgo_binary
            );
            eprintln!("   Run the PGO script first from workspace root:");
            eprintln!("   ./crates/inference/benches/build-pgo-concurrent.sh");
            return None;
        }

        Some(Self {
            model_path,
            original_binary,
            pgo_binary,
        })
    }
}

fn main() {
    println!("🔍 Validating PGO benchmark configuration...");

    match PGOBenchmarkConfig::new() {
        Some(config) => {
            println!("✅ All required files found:");
            println!("   Model: {:?}", config.model_path);
            println!("   Original binary: {:?}", config.original_binary);
            println!("   PGO binary: {:?}", config.pgo_binary);
            println!("🎯 Configuration is valid - ready to run benchmarks!");
        }
        None => {
            eprintln!("❌ Configuration validation failed");
            std::process::exit(1);
        }
    }
}