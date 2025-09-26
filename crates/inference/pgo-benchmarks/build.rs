use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../../scripts/build-pgo.sh");
    println!("cargo:rerun-if-changed=../../../scripts/build-pgo-examples.sh");
    println!("cargo:rerun-if-env-changed=BENCH_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=SKIP_PGO_BUILD");

    // Allow skipping PGO build for faster iteration
    if env::var("SKIP_PGO_BUILD").is_ok() {
        println!("cargo:warning=⚡ SKIP_PGO_BUILD set - skipping PGO binary preparation");
        println!("cargo:warning=   Benchmarks will check for existing binaries at runtime");
        return;
    }

    println!("cargo:warning=🚀 Preparing PGO benchmark environment...");
    println!("cargo:warning=   This may take several minutes on first run");

    let workspace_root = find_workspace_root();

    // Check for model path
    let model_path = env::var("BENCH_MODEL_PATH").unwrap_or_else(|_| {
        let home = env::var("HOME").unwrap_or_default();
        format!(
            "{}/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors",
            home
        )
    });

    if !Path::new(&model_path).exists() {
        println!("cargo:warning=⚠️  Model not found at: {}", model_path);
        println!("cargo:warning=   Set BENCH_MODEL_PATH environment variable or download a model");
        println!("cargo:warning=   PGO benchmarks will be skipped during execution");
        return;
    }

    // Build baseline binaries if they don't exist
    let baseline_binary = workspace_root.join("target/release/inferno-baseline");
    let baseline_concurrent =
        workspace_root.join("target/release/examples/concurrent_inference-baseline");

    if !baseline_binary.exists() || !baseline_concurrent.exists() {
        println!("cargo:warning=📦 Building baseline binaries...");
        println!("cargo:warning=   This step builds inferno + concurrent_inference examples");
        if let Err(e) = build_baseline_binaries(&workspace_root) {
            println!("cargo:warning=❌ Failed to build baseline binaries: {}", e);
            println!("cargo:warning=   Benchmarks will skip missing binaries");
            return;
        }
    } else {
        println!("cargo:warning=✅ Baseline binaries already exist");
    }

    // Build PGO binaries if they don't exist
    let pgo_binary = workspace_root.join("target/release/inferno-pgo");
    let pgo_concurrent = workspace_root.join("target/release/examples/concurrent_inference-pgo");

    if !pgo_binary.exists() || !pgo_concurrent.exists() {
        println!("cargo:warning=🔥 Building PGO-optimized binaries...");
        println!(
            "cargo:warning=   This step runs profiling workloads and rebuilds with optimization"
        );
        println!("cargo:warning=   Expected time: 3-10 minutes depending on model size");
        if let Err(e) = build_pgo_binaries(&workspace_root, &model_path) {
            println!("cargo:warning=❌ Failed to build PGO binaries: {}", e);
            println!("cargo:warning=   Will run benchmarks with available binaries only");
            return;
        }
    } else {
        println!("cargo:warning=✅ PGO binaries already exist");
    }

    println!("cargo:warning=✅ PGO benchmark environment ready!");
    println!("cargo:warning=   Model: {}", model_path);
    println!(
        "cargo:warning=   Baseline CLI: {}",
        baseline_binary.display()
    );
    println!("cargo:warning=   PGO CLI: {}", pgo_binary.display());
    println!(
        "cargo:warning=   Baseline Concurrent: {}",
        baseline_concurrent.display()
    );
    println!(
        "cargo:warning=   PGO Concurrent: {}",
        pgo_concurrent.display()
    );
}

fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        if current.join("Cargo.toml").exists() {
            if let Ok(content) = fs::read_to_string(current.join("Cargo.toml")) {
                if content.contains("[workspace]") {
                    return current;
                }
            }
        }

        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => break,
        }
    }

    // Fallback: assume we're in crates/inference/pgo-benchmarks/ subdirectory
    env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn build_baseline_binaries(workspace_root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Build main inferno binary
    let output = Command::new("cargo")
        .args(&["build", "--release", "--bin", "inferno"])
        .env("RUSTFLAGS", "-C target-cpu=native")
        .current_dir(workspace_root)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "Failed to build baseline inferno binary: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    // Build examples
    let output = Command::new("cargo")
        .args(&[
            "build",
            "--release",
            "--package",
            "inferno-inference",
            "--examples",
            "--features",
            "examples",
        ])
        .env("RUSTFLAGS", "-C target-cpu=native")
        .current_dir(workspace_root)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "Failed to build baseline examples: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    // Copy to baseline locations
    let inferno_from = workspace_root.join("target/release/inferno");
    let inferno_to = workspace_root.join("target/release/inferno-baseline");
    fs::copy(&inferno_from, &inferno_to)?;

    let concurrent_from = workspace_root.join("target/release/examples/concurrent_inference");
    let concurrent_to =
        workspace_root.join("target/release/examples/concurrent_inference-baseline");
    fs::copy(&concurrent_from, &concurrent_to)?;

    println!("cargo:warning=✅ Baseline binaries built successfully");
    Ok(())
}

fn build_pgo_binaries(
    workspace_root: &Path,
    model_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let pgo_script = workspace_root.join("scripts/build-pgo.sh");
    let pgo_examples_script = workspace_root.join("scripts/build-pgo-examples.sh");

    // Build main PGO binary (best effort)
    if pgo_script.exists() {
        let output = Command::new("bash")
            .arg(&pgo_script)
            .arg("--fast")
            .env("BENCH_MODEL_PATH", model_path)
            .current_dir(workspace_root)
            .output()?;

        if !output.status.success() {
            println!("cargo:warning=⚠️  Main PGO build failed, continuing with examples only");
        }
    }

    // Build PGO examples binary
    if pgo_examples_script.exists() {
        let output = Command::new("bash")
            .arg(&pgo_examples_script)
            .env("BENCH_MODEL_PATH", model_path)
            .current_dir(workspace_root)
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "PGO examples build failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
    }

    println!("cargo:warning=✅ PGO binaries built successfully");
    Ok(())
}
