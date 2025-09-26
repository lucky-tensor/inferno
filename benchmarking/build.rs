use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../scripts/build-pgo.sh");
    println!("cargo:rerun-if-changed=../scripts/build-pgo-examples.sh");
    println!("cargo:rerun-if-changed=../target/release/inferno");
    println!("cargo:rerun-if-env-changed=BENCH_MODEL_PATH");

    // Check if we're running the pgo_benches benchmark specifically
    let is_pgo_bench =
        env::args().any(|arg| arg.contains("pgo_benches") || arg.contains("pgo-benches"));
    let is_bench = env::var("CARGO_CFG_TEST").is_err()
        && env::args().any(|arg| arg == "bench" || arg.contains("bench"));

    if !is_bench {
        println!("cargo:warning=Skipping PGO binary preparation (not running benchmarks)");
        return;
    }

    // For pgo_benches, we need to build everything
    if is_pgo_bench {
        println!("cargo:warning=🚀 Detected pgo_benches - building complete PGO benchmark environment...");
    }

    println!("cargo:warning=Preparing PGO benchmark environment...");

    let workspace_root = find_workspace_root();
    let baseline_binary = workspace_root.join("target/release/inferno-baseline");
    let pgo_binary = workspace_root.join("target/release/inferno-pgo");

    // Check if model path is available
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

    // Build baseline binary if it doesn't exist
    if !baseline_binary.exists() {
        println!("cargo:warning=Building baseline binary for PGO comparison...");
        if let Err(e) = build_baseline_binary(&workspace_root) {
            println!("cargo:warning=Failed to build baseline binary: {}", e);
            return;
        }
    } else {
        println!(
            "cargo:warning=✅ Baseline binary found at: {}",
            baseline_binary.display()
        );
    }

    // Build PGO binary if it doesn't exist
    if !pgo_binary.exists() {
        println!("cargo:warning=Building PGO-optimized binary (this may take several minutes)...");
        if let Err(e) = build_pgo_binary(&workspace_root, &model_path) {
            println!("cargo:warning=Failed to build PGO binary: {}", e);
            println!("cargo:warning=PGO benchmarks will compare against baseline only");
            return;
        }
    } else {
        println!(
            "cargo:warning=✅ PGO binary found at: {}",
            pgo_binary.display()
        );
    }

    println!("cargo:warning=🎉 PGO benchmark environment ready!");
}

fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        if current.join("Cargo.toml").exists() {
            // Check if it's a workspace root by looking for [workspace] section
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

    // Fallback: assume we're in benchmarking/ subdirectory
    env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn build_baseline_binary(workspace_root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Build main inferno binary
    let output = Command::new("cargo")
        .args(["build", "--release", "--bin", "inferno"])
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
        .args([
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
            "Failed to build baseline examples binary: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    // Copy to baseline locations
    let inferno_from = workspace_root.join("target/release/inferno");
    let inferno_to = workspace_root.join("target/release/inferno-baseline");
    fs::copy(&inferno_from, &inferno_to)?;

    let concurrent_inference_from =
        workspace_root.join("target/release/examples/concurrent_inference");
    let concurrent_inference_to =
        workspace_root.join("target/release/examples/concurrent_inference-baseline");
    fs::copy(&concurrent_inference_from, &concurrent_inference_to)?;

    println!("cargo:warning=✅ Baseline binaries created:");
    println!("cargo:warning=   {}", inferno_to.display());
    println!("cargo:warning=   {}", concurrent_inference_to.display());
    Ok(())
}

fn build_pgo_binary(
    workspace_root: &Path,
    model_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let pgo_script = workspace_root.join("scripts/build-pgo.sh");
    let pgo_examples_script = workspace_root.join("scripts/build-pgo-examples.sh");

    if !pgo_script.exists() {
        return Err("PGO build script not found at scripts/build-pgo.sh".into());
    }

    if !pgo_examples_script.exists() {
        return Err("PGO examples build script not found at scripts/build-pgo-examples.sh".into());
    }

    // Build main PGO binary
    let output = Command::new("bash")
        .arg(&pgo_script)
        .arg("--fast") // Use fast mode for build.rs to avoid excessive build times
        .env("BENCH_MODEL_PATH", model_path)
        .current_dir(workspace_root)
        .output()?;

    if !output.status.success() {
        println!("cargo:warning=⚠️  Main PGO build failed, continuing with examples only");
    }

    // Build PGO examples binary
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

    let pgo_examples_binary =
        workspace_root.join("target/release/examples/concurrent_inference-pgo");
    if !pgo_examples_binary.exists() {
        return Err("PGO concurrent_inference example was not created successfully".into());
    }

    println!(
        "cargo:warning=✅ PGO concurrent_inference example created at: {}",
        pgo_examples_binary.display()
    );

    // Check for main PGO binary (optional)
    let pgo_binary = workspace_root.join("target/release/inferno-pgo");
    if pgo_binary.exists() {
        println!(
            "cargo:warning=✅ PGO main binary also available at: {}",
            pgo_binary.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_workspace_root() {
        let root = find_workspace_root();
        assert!(
            root.join("Cargo.toml").exists(),
            "Workspace root should contain Cargo.toml"
        );
    }
}
