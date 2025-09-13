use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use git2::{Cred, FetchOptions, RemoteCallbacks, Repository};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest;
use sha2::{Digest, Sha256};
use std::env;
use std::io::Read;
use std::path::Path;
use std::process::Command;
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Downloads a model from Hugging Face Hub
///
/// # Arguments
/// * `model_id` - The Hugging Face model ID (e.g., "microsoft/DialoGPT-medium")
/// * `output_dir` - Directory where the model will be downloaded
/// * `hf_token` - Optional Hugging Face token for private/gated models
/// * `resume` - Whether to resume interrupted downloads
/// * `use_xet` - Whether to use xet via Python huggingface_hub instead of Git LFS
///
/// # Example
/// ```
/// use inferno_cli::model_downloader::download_model;
///
/// # tokio_test::block_on(async {
/// download_model("microsoft/DialoGPT-medium", "./models", None, false, false).await.unwrap();
/// # });
/// ```
pub async fn download_model(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&String>,
    resume: bool,
    use_xet: bool,
) -> Result<()> {
    println!("🔄 Starting model download...");

    // Get HF token from parameter, environment, or prompt user
    let token = get_hf_token(hf_token).await?;

    // Create output directory for model files
    let model_dir = format!("{}/{}", output_dir, model_id.replace("/", "_"));
    println!("📁 Creating model directory: {}", model_dir);
    fs::create_dir_all(&model_dir).await?;

    // Download model from Hugging Face with resume capability
    println!("⬇️  Downloading model from Hugging Face: {}", model_id);

    if use_xet {
        println!("🚀 Using xet backend via Python huggingface_hub");
        download_model_with_xet(model_id, &model_dir, token.as_deref()).await?;
    } else {
        println!("📦 Using Git LFS backend (default)");
        download_huggingface_model(model_id, &model_dir, token.as_deref(), resume).await?;
    }

    Ok(())
}

async fn get_hf_token(provided_token: Option<&String>) -> Result<Option<String>> {
    // 1. Use provided token if available
    if let Some(token) = provided_token {
        return Ok(Some(token.clone()));
    }

    // 2. Check environment variables
    if let Ok(token) = env::var("HUGGINGFACE_HUB_TOKEN") {
        println!("🔑 Using HF token from HUGGINGFACE_HUB_TOKEN environment variable");
        return Ok(Some(token));
    }

    if let Ok(token) = env::var("HF_TOKEN") {
        println!("🔑 Using HF token from HF_TOKEN environment variable");
        return Ok(Some(token));
    }

    // 3. Check for token in HF cache directory
    if let Ok(home) = env::var("HOME") {
        let token_file = format!("{}/.cache/huggingface/token", home);
        if let Ok(token) = tokio::fs::read_to_string(&token_file).await {
            let token = token.trim().to_string();
            if !token.is_empty() {
                println!("🔑 Using HF token from cache file: {}", token_file);
                return Ok(Some(token));
            }
        }
    }

    // 4. Try to prompt user for token (only for gated models)
    println!("ℹ️  No HF token found. This may be needed for gated/private models.");
    println!("💡 You can set HF_TOKEN environment variable or use --hf-token parameter");

    Ok(None)
}

async fn download_huggingface_model(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
    resume: bool,
) -> Result<()> {
    // Method 1: Try using git2 (Rust git library) with LFS support
    println!("📦 Using git2 to clone repository...");
    match clone_repo_with_lfs(model_id, output_dir, hf_token, resume).await {
        Ok(_) => {
            println!("✅ Successfully cloned repository with LFS files");
            return Ok(());
        }
        Err(e) => {
            println!("⚠️  Git2 clone failed: {}, trying alternative method...", e);
        }
    }

    // Method 2: Fallback to wget for individual files
    println!("🌐 Using wget to download model files...");
    download_model_files_with_wget(model_id, output_dir, hf_token).await?;

    Ok(())
}

async fn clone_repo_with_lfs(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
    resume: bool,
) -> Result<()> {
    let clone_url = if let Some(token) = hf_token {
        format!("https://{}@huggingface.co/{}", token, model_id)
    } else {
        format!("https://huggingface.co/{}", model_id)
    };

    // Check if we should resume (repository already exists)
    let repo = if resume
        && Path::new(output_dir).exists()
        && Path::new(&format!("{}/.git", output_dir)).exists()
    {
        println!("  🔄 Resuming from existing repository...");
        Repository::open(output_dir)?
    } else {
        // Clone the repository fresh with progress tracking
        let progress_bar = ProgressBar::new_spinner();
        progress_bar.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")
                .unwrap()
                .tick_strings(&["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]),
        );
        progress_bar.set_message("Cloning repository...");

        let mut callbacks = RemoteCallbacks::new();
        callbacks.credentials(|_url, username_from_url, _allowed_types| {
            if let Some(token) = hf_token {
                Cred::userpass_plaintext(username_from_url.unwrap_or("oauth2"), token)
            } else {
                Cred::default()
            }
        });

        // Add progress callback for Git operations
        callbacks.pack_progress(|stage, current, total| {
            let pct = if total > 0 {
                (100 * current) / total
            } else {
                0
            };
            progress_bar.set_message(format!(
                "Cloning: {:?} {}% ({}/{})",
                stage, pct, current, total
            ));
        });

        let mut fetch_options = FetchOptions::new();
        fetch_options.remote_callbacks(callbacks);

        let mut builder = git2::build::RepoBuilder::new();
        builder.fetch_options(fetch_options);

        println!("  📦 Cloning repository from Hugging Face...");
        let repo = builder.clone(&clone_url, Path::new(output_dir))?;
        progress_bar.finish_with_message("✅ Repository cloned");
        repo
    };

    // Now handle LFS files
    println!("  📥 Downloading LFS files...");
    download_lfs_files(&repo, output_dir, hf_token).await?;

    Ok(())
}

async fn download_lfs_files(
    repo: &Repository,
    repo_dir: &str,
    hf_token: Option<&str>,
) -> Result<()> {
    // Create a multi-progress bar for all downloads
    let multi_progress = MultiProgress::new();
    let main_progress = multi_progress.add(ProgressBar::new_spinner());
    main_progress.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]),
    );
    main_progress.set_message("Scanning for LFS files using Git...");

    // Use Git to find LFS files
    let mut lfs_files = Vec::new();
    find_lfs_files_with_git(repo, repo_dir, &mut lfs_files).await?;

    if lfs_files.is_empty() {
        main_progress.finish_with_message("ℹ️ No LFS files found");
        return Ok(());
    }

    main_progress.finish_with_message(format!("📄 Found {} LFS files", lfs_files.len()));

    // Extract model ID from repo directory
    let model_id = extract_model_id_from_repo_path(repo_dir)?;

    // Download each LFS file with individual progress bars
    let mut download_tasks = Vec::new();

    for lfs_file in &lfs_files {
        let file_path = lfs_file.path.clone();
        let file_name = lfs_file
            .path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let expected_size = lfs_file.size;

        // Check if file is already downloaded and verified
        if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
            if metadata.len() == expected_size {
                // Verify hash of existing file
                println!("  🔍 Verifying existing file: {}", file_name);
                if verify_file_hash(&file_path, &lfs_file.oid)
                    .await
                    .unwrap_or(false)
                {
                    println!(
                        "  ✅ LFS file already complete and verified: {} ({} bytes)",
                        file_name, expected_size
                    );
                    continue;
                } else {
                    println!("  ⚠️  Hash mismatch for {}, re-downloading...", file_name);
                }
            }
        }

        // Create progress bar for this file
        let progress_bar = multi_progress.add(ProgressBar::new(expected_size));
        progress_bar.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})"
            )
            .unwrap()
            .progress_chars("#>-")
        );
        progress_bar.set_message(format!("📥 {}", file_name));

        let model_id_clone = model_id.clone();
        let hf_token_clone = hf_token.map(|s| s.to_string());

        download_tasks.push(async move {
            download_lfs_file_with_progress(
                lfs_file.clone(),
                &model_id_clone,
                hf_token_clone.as_deref(),
                progress_bar,
            )
            .await
        });
    }

    // Execute all downloads concurrently
    let results = futures_util::future::join_all(download_tasks).await;

    // Check if all downloads succeeded
    for result in results {
        result?;
    }

    // Verify all downloaded files
    println!("🔍 Verifying integrity of downloaded files...");
    let mut all_verified = true;
    for lfs_file in &lfs_files {
        let file_name = lfs_file.path.file_name().unwrap().to_string_lossy();
        print!("  🔍 Verifying {}: ", file_name);

        if verify_file_hash(&lfs_file.path, &lfs_file.oid).await? {
            println!("✅ Hash verified");
        } else {
            println!("❌ Hash mismatch!");
            all_verified = false;
        }
    }

    if all_verified {
        println!("✅ All LFS files downloaded and verified successfully!");
    } else {
        return Err(anyhow!("Some files failed hash verification"));
    }

    Ok(())
}

#[derive(Clone)]
struct LfsFile {
    path: std::path::PathBuf,
    oid: String,
    size: u64,
}

async fn find_lfs_files_with_git(
    repo: &Repository,
    repo_dir: &str,
    lfs_files: &mut Vec<LfsFile>,
) -> Result<()> {
    // Use git2 to iterate through the repository index and find LFS files
    let index = repo.index()?;

    for entry in index.iter() {
        let file_path = std::path::Path::new(repo_dir).join(std::str::from_utf8(&entry.path)?);

        // Check if this is a text file that could be an LFS pointer
        if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
            if content.starts_with("version https://git-lfs.github.com/spec/v1") {
                // This is an LFS pointer file, parse it using Git's LFS info
                if let Ok(lfs_file) = parse_lfs_pointer(&content, &file_path) {
                    println!(
                        "    📄 Found LFS file: {} ({} bytes)",
                        file_path.file_name().unwrap().to_string_lossy(),
                        lfs_file.size
                    );
                    lfs_files.push(lfs_file);
                }
            }
        }
    }

    Ok(())
}

fn parse_lfs_pointer(content: &str, path: &Path) -> Result<LfsFile> {
    let mut oid = None;
    let mut size = None;

    for line in content.lines() {
        if line.starts_with("oid sha256:") {
            oid = Some(line.strip_prefix("oid sha256:").unwrap().to_string());
        } else if line.starts_with("size ") {
            size = Some(line.strip_prefix("size ").unwrap().parse::<u64>()?);
        }
    }

    let oid = oid.ok_or_else(|| anyhow!("No OID found in LFS pointer"))?;
    let size = size.ok_or_else(|| anyhow!("No size found in LFS pointer"))?;

    Ok(LfsFile {
        path: path.to_path_buf(),
        oid,
        size,
    })
}

fn extract_model_id_from_repo_path(repo_path: &str) -> Result<String> {
    let path_parts: Vec<&str> = repo_path.split('/').collect();
    if let Some(last_part) = path_parts.last() {
        if last_part.contains('_') {
            // Convert underscore back to slash (e.g., "openai_gpt-oss-20b" -> "openai/gpt-oss-20b")
            let model_id = last_part.replace('_', "/");
            return Ok(model_id);
        }
    }
    Err(anyhow!(
        "Could not extract model ID from repo path: {}",
        repo_path
    ))
}

async fn download_lfs_file_with_progress(
    lfs_file: LfsFile,
    model_id: &str,
    hf_token: Option<&str>,
    progress_bar: ProgressBar,
) -> Result<()> {
    let file_name = lfs_file.path.file_name().unwrap().to_string_lossy();

    // Build Hugging Face URL for the file
    let relative_path = lfs_file
        .path
        .strip_prefix(
            lfs_file
                .path
                .ancestors()
                .find(|p| p.join(".git").exists())
                .unwrap(),
        )
        .unwrap();
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model_id,
        relative_path.display()
    );

    // Create HTTP client with optional authentication
    let client = reqwest::Client::new();
    let mut request = client.get(&url);

    if let Some(token) = hf_token {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    // Send request and get response
    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download {}: HTTP {}",
            file_name,
            response.status()
        ));
    }

    // Get content length and create stream
    let total_size = response.content_length().unwrap_or(lfs_file.size);
    progress_bar.set_length(total_size);

    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(&lfs_file.path).await?;
    let mut downloaded = 0u64;

    // Download with progress tracking
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_bar.set_position(downloaded);
    }

    // Verify the downloaded file hash
    progress_bar.set_message(format!("🔍 Verifying {}", file_name));
    let hash_valid = verify_file_hash(&lfs_file.path, &lfs_file.oid).await?;

    if hash_valid {
        progress_bar.finish_with_message(format!("✅ {} (verified)", file_name));
    } else {
        progress_bar.finish_with_message(format!("❌ {} (hash mismatch)", file_name));
        return Err(anyhow!("Hash verification failed for {}", file_name));
    }

    Ok(())
}

async fn download_model_files_with_wget(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
) -> Result<()> {
    let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);

    // Common model files to try downloading (prioritizing safetensors)
    let files_to_try = vec![
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ];

    let mut downloaded_any = false;

    for file in files_to_try {
        let url = format!("{}/{}", base_url, file);
        let output_file = format!("{}/{}", output_dir, file);

        println!("  📄 Trying to download: {}", file);

        let mut wget_args = vec![&url, "-O", &output_file, "--timeout=30", "--tries=2", "-q"];

        // Add authorization header if token is provided
        let auth_header;
        if let Some(token) = hf_token {
            auth_header = format!("Authorization: Bearer {}", token);
            wget_args.extend_from_slice(&["--header", &auth_header]);
        }

        let status = Command::new("wget").args(&wget_args).status();

        match status {
            Ok(status) if status.success() => {
                println!("  ✅ Downloaded: {}", file);
                downloaded_any = true;
            }
            _ => {
                println!("  ❌ Failed to download: {}", file);
                // Remove failed download file if it exists
                let _ = fs::remove_file(&output_file).await;
            }
        }
    }

    if !downloaded_any {
        return Err(anyhow!(
            "Failed to download any model files from {}",
            model_id
        ));
    }

    Ok(())
}

async fn verify_file_hash(file_path: &Path, expected_oid: &str) -> Result<bool> {
    println!("    Expected: {}", expected_oid);

    // Read file and compute SHA256 hash
    let mut file = std::fs::File::open(file_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let computed_hash = hex::encode(hasher.finalize());
    println!("    Computed: {}", computed_hash);

    let hash_match = computed_hash == expected_oid;
    println!("    Match: {}", if hash_match { "✅" } else { "❌" });

    Ok(hash_match)
}

/// Download model using xet via Python huggingface_hub
async fn download_model_with_xet(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
) -> Result<()> {
    println!("🐍 Checking Python and huggingface_hub availability...");

    // Check if Python is available
    let python_check = Command::new("python3")
        .arg("-c")
        .arg("import huggingface_hub; print(huggingface_hub.__version__)")
        .output();

    match python_check {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("✅ Found huggingface_hub version: {}", version);

            // Check if version is >= 0.32.0 (when xet support was added)
            if is_version_compatible(&version, "0.32.0") {
                println!("✅ Version supports xet backend");
            } else {
                println!("⚠️  Version {} may not support xet (requires >= 0.32.0)", version);
                println!("💡 Consider upgrading: pip install --upgrade huggingface_hub");
            }
        },
        Ok(output) => {
            println!("❌ huggingface_hub import failed:");
            println!("{}", String::from_utf8_lossy(&output.stderr));
            return Err(anyhow!("huggingface_hub not available. Install with: pip install huggingface_hub"));
        },
        Err(e) => {
            println!("❌ Python3 not found: {}", e);
            // Try python as fallback
            let python_fallback = Command::new("python")
                .arg("-c")
                .arg("import huggingface_hub; print(huggingface_hub.__version__)")
                .output();

            match python_fallback {
                Ok(output) if output.status.success() => {
                    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    println!("✅ Found huggingface_hub via 'python': {}", version);
                },
                _ => {
                    return Err(anyhow!("Neither python3 nor python found. Please install Python and huggingface_hub"));
                }
            }
        }
    }

    println!("📥 Downloading model using Python huggingface_hub with xet...");

    // Create Python script to download the model
    let python_script = create_xet_download_script(model_id, output_dir, hf_token)?;

    // Write script to temporary file
    let temp_script_path = format!("{}/download_script.py", output_dir);
    tokio::fs::write(&temp_script_path, python_script).await?;

    // Execute the Python script
    let mut cmd = Command::new("python3");
    cmd.arg(&temp_script_path);

    println!("🏃 Executing Python download script...");
    let output = cmd.output()?;

    // Clean up temporary script
    let _ = tokio::fs::remove_file(&temp_script_path).await;

    if output.status.success() {
        println!("✅ Xet download completed successfully!");
        println!("{}", String::from_utf8_lossy(&output.stdout));
    } else {
        println!("❌ Xet download failed:");
        println!("{}", String::from_utf8_lossy(&output.stderr));

        // Fallback to Git LFS
        println!("🔄 Falling back to Git LFS...");
        return download_huggingface_model(model_id, output_dir, hf_token, false).await;
    }

    Ok(())
}

/// Create Python script for xet-based download
fn create_xet_download_script(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
) -> Result<String> {
    let token_setup = if let Some(token) = hf_token {
        format!(r#"
import os
os.environ["HF_TOKEN"] = "{}"
from huggingface_hub import login
login(token="{}")
"#, token, token)
    } else {
        String::new()
    };

    let script = format!(r#"#!/usr/bin/env python3
"""
Xet-enabled model download script for Inferno
This script uses huggingface_hub to download models with xet backend when available.
"""

import os
import sys
from pathlib import Path

# Ensure proper home directory setup for xet/huggingface_hub
home_dir = Path.home()
hf_cache_dir = home_dir / ".cache" / "huggingface"
xet_cache_dir = home_dir / ".cache" / "xet"

# Create cache directories if they don't exist
hf_cache_dir.mkdir(parents=True, exist_ok=True)
xet_cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables for huggingface_hub and xet
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["HF_CACHE"] = str(hf_cache_dir)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir)

# Set xet-specific environment variables if not already set
if "XET_HOME" not in os.environ:
    os.environ["XET_HOME"] = str(xet_cache_dir)

print(f"🏠 HF cache directory: {{hf_cache_dir}}")
print(f"🏠 Xet cache directory: {{xet_cache_dir}}")

try:
    from huggingface_hub import snapshot_download
    print("📦 huggingface_hub imported successfully")
except ImportError as e:
    print(f"❌ Failed to import huggingface_hub: {{e}}")
    print("💡 Install with: pip install huggingface_hub")
    sys.exit(1)

{token_setup}

def main():
    model_id = "{}"
    output_dir = "{}"

    print(f"🚀 Starting xet-enabled download of {{model_id}}")
    print(f"📁 Output directory: {{output_dir}}")

    try:
        # Use snapshot_download which automatically uses xet when available
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # Force actual file downloads
            resume_download=True,  # Enable resume capability
            # xet will be used automatically if available (huggingface_hub >= 0.32.0)
        )

        print(f"✅ Model downloaded successfully to: {{downloaded_path}}")

        # List downloaded files
        downloaded_files = list(Path(output_dir).rglob("*"))
        print(f"📄 Downloaded {{len(downloaded_files)}} files:")
        for file_path in sorted(downloaded_files)[:10]:  # Show first 10 files
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {{file_path.name}} ({{size_mb:.1f}} MB)")

        if len(downloaded_files) > 10:
            print(f"  ... and {{len(downloaded_files) - 10}} more files")

    except Exception as e:
        print(f"❌ Download failed: {{e}}")
        print("💡 This may be due to network issues, authentication, or missing dependencies")
        sys.exit(1)

if __name__ == "__main__":
    main()
"#, model_id, output_dir);

    Ok(script)
}

/// Check if version string is compatible (simplified semver check)
fn is_version_compatible(version: &str, min_version: &str) -> bool {
    // Simple version comparison for major.minor.patch
    let parse_version = |v: &str| -> Vec<u32> {
        v.split('.')
            .take(3)
            .filter_map(|n| n.parse().ok())
            .collect()
    };

    let current = parse_version(version);
    let minimum = parse_version(min_version);

    if current.len() < 2 || minimum.len() < 2 {
        return false; // Invalid version format
    }

    // Compare major.minor.patch
    for i in 0..std::cmp::min(current.len(), minimum.len()) {
        match current[i].cmp(&minimum[i]) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => continue,
        }
    }

    // If all compared parts are equal, versions are compatible
    true
}
