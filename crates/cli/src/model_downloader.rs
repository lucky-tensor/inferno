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
///
/// # Example
/// ```
/// use inferno_cli::model_downloader::download_model;
///
/// # tokio_test::block_on(async {
/// download_model("microsoft/DialoGPT-medium", "./models", None, false).await.unwrap();
/// # });
/// ```
pub async fn download_model(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&String>,
    resume: bool,
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
    download_huggingface_model(model_id, &model_dir, token.as_deref(), resume).await?;

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
