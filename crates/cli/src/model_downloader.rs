use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use git2::{Cred, FetchOptions, RemoteCallbacks, Repository};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest;
use sha2::{Digest, Sha256};
use std::env;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Helper function to format model directory name consistently
fn format_model_dir_name(model_id: &str) -> String {
    model_id.replace("/", "_")
}

/// Check if essential model files already exist in the output directory
async fn check_existing_model_files(
    output_dir: &str,
    model_id: &str,
) -> Result<Option<Vec<String>>> {
    let model_dir_name = format_model_dir_name(model_id);
    let full_output_dir = Path::new(output_dir).join(&model_dir_name);

    if !full_output_dir.exists() {
        return Ok(None);
    }

    // Essential files we expect to find
    let essential_files = vec![
        "model.safetensors",     // Primary model weights (safetensors format only)
        "tokenizer.json",        // Tokenizer configuration
        "tokenizer_config.json", // Tokenizer metadata
    ];

    let mut existing_files = Vec::new();
    let mut has_model_file = false;

    for filename in &essential_files {
        let file_path = full_output_dir.join(filename);
        if file_path.exists() {
            existing_files.push(filename.to_string());
            if filename.contains(".safetensors") {
                has_model_file = true;
            }
        }
    }

    // Check for any .safetensors file (not just model.safetensors)
    if !has_model_file {
        if let Ok(entries) = tokio::fs::read_dir(&full_output_dir).await {
            let mut entries = entries;
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(file_name) = entry.file_name().into_string() {
                    if file_name.ends_with(".safetensors") {
                        existing_files.push(file_name);
                        has_model_file = true;
                        break;
                    }
                }
            }
        }
    }

    if has_model_file {
        Ok(Some(existing_files))
    } else {
        Ok(None)
    }
}

/// Downloads a model from Hugging Face Hub
///
/// # Arguments
/// * `model_id` - The Hugging Face model ID (e.g., "microsoft/DialoGPT-medium")
/// * `output_dir` - Directory where the model will be downloaded
/// * `hf_token` - Optional Hugging Face token for private/gated models (or use HF_TOKEN env var)
/// * `resume` - Whether to resume interrupted downloads
/// * `use_xet` - Whether to use native Rust hf-hub with automatic xet backend instead of Git LFS
///
/// # Example
/// ```no_run
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
    println!("Starting model download...");

    // Check if model files already exist
    if let Some(existing_files) = check_existing_model_files(output_dir, model_id).await? {
        println!(
            "Model already downloaded! Found {} existing files:",
            existing_files.len()
        );

        let model_dir_name = format_model_dir_name(model_id);
        let full_output_dir = Path::new(output_dir).join(&model_dir_name);

        let mut total_size = 0u64;
        for filename in &existing_files {
            let file_path = full_output_dir.join(filename);
            if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                let size = metadata.len();
                total_size += size;
                let size_mb = size as f64 / (1024.0 * 1024.0);
                println!("  - {} ({:.1} MB)", filename, size_mb);
            } else {
                println!("  - {}", filename);
            }
        }

        let total_mb = total_size as f64 / (1024.0 * 1024.0);
        println!("Total: {:.1} MB already available", total_mb);
        println!("Skipping download - model is ready to use!");
        return Ok(());
    }

    // Setup inferno-specific cache directory structure
    let hf_cache_dir = setup_inferno_cache_directory(output_dir).await?;

    // Get HF token from parameter, environment, or prompt user
    let token = get_hf_token(hf_token).await?;

    // Create output directory for model files
    let model_dir = format!("{}/{}", output_dir, model_id.replace("/", "_"));
    println!("Creating model directory: {}", model_dir);
    fs::create_dir_all(&model_dir).await?;

    // Download model from Hugging Face with resume capability
    println!("Downloading model from Hugging Face: {}", model_id);

    if use_xet {
        download_model_with_xet(model_id, &model_dir, token.as_deref(), &hf_cache_dir).await?;
    } else {
        println!("Using Git LFS backend");
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
    println!("INFO: No HF token found. This may be needed for gated/private models.");
    println!("INFO: You can set HF_TOKEN environment variable or use --hf-token parameter");

    Ok(None)
}

/// Setup inferno-specific cache directory structure within the models directory
async fn setup_inferno_cache_directory(models_dir: &str) -> Result<PathBuf> {
    let cache_dir = PathBuf::from(models_dir).join("cache");
    let hf_cache = cache_dir.join("huggingface");

    // Create cache directory structure
    fs::create_dir_all(&cache_dir).await?;
    fs::create_dir_all(&hf_cache).await?;

    println!("Using inferno cache directory: {}", cache_dir.display());
    println!("  HuggingFace cache: {}", hf_cache.display());

    Ok(hf_cache) // Return the HuggingFace cache directory for use in API builders
}

async fn download_huggingface_model(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
    resume: bool,
) -> Result<()> {
    // Method 1: Try using git2 (Rust git library) with LFS support
    println!("Using git2 to clone repository...");
    match clone_repo_with_lfs(model_id, output_dir, hf_token, resume).await {
        Ok(_) => {
            println!("Successfully cloned repository with LFS files");
            return Ok(());
        }
        Err(e) => {
            println!(
                "WARNING: Git2 clone failed: {}, trying alternative method...",
                e
            );
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
        println!("  Resuming from existing repository...");
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

        println!("  Cloning repository from Hugging Face...");
        let repo = builder.clone(&clone_url, Path::new(output_dir))?;
        progress_bar.finish_with_message("Repository cloned");
        repo
    };

    // Now handle LFS files
    println!("  Downloading LFS files...");
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
        main_progress.finish_with_message("No LFS files found");
        return Ok(());
    }

    main_progress.finish_with_message(format!("Found {} LFS files", lfs_files.len()));

    // Extract model ID from repo directory
    let model_id = extract_model_id_from_repo_path(repo_dir)?;

    // Track cached vs downloaded files
    let mut cached_files = Vec::new();
    let mut download_tasks = Vec::new();

    let start_time = std::time::Instant::now();

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
                if verify_file_hash(&file_path, &lfs_file.oid)
                    .await
                    .unwrap_or(false)
                {
                    cached_files.push((file_name.clone(), expected_size));
                    continue;
                } else {
                    println!(
                        "  WARNING: Hash mismatch for {}, re-downloading...",
                        file_name
                    );
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
        progress_bar.set_message(format!("Downloading {}", file_name));

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
    println!("Verifying integrity of downloaded files...");
    let mut all_verified = true;
    for lfs_file in &lfs_files {
        let file_name = lfs_file.path.file_name().unwrap().to_string_lossy();
        print!("  Verifying {}: ", file_name);

        if verify_file_hash(&lfs_file.path, &lfs_file.oid).await? {
            println!("Hash verified");
        } else {
            println!("Hash mismatch!");
            all_verified = false;
        }
    }

    if all_verified {
        let total_duration = start_time.elapsed();
        let downloaded_count = lfs_files.len() - cached_files.len();

        // Show cache/download summary
        if !cached_files.is_empty() {
            println!("Restored {} files from Git LFS cache:", cached_files.len());
            let mut cached_size = 0u64;
            for (filename, size) in &cached_files {
                cached_size += size;
                let size_mb = *size as f64 / (1024.0 * 1024.0);
                println!("  - {} ({:.1} MB)", filename, size_mb);
            }
        }

        if downloaded_count > 0 {
            println!("Downloaded {} files via Git LFS", downloaded_count);
        }

        if !cached_files.is_empty() && downloaded_count > 0 {
            println!(
                "All LFS files completed successfully! ({} downloaded, {} from cache) in {:.1}s",
                downloaded_count,
                cached_files.len(),
                total_duration.as_secs_f64()
            );
        } else if !cached_files.is_empty() {
            println!(
                "All LFS files restored from cache in {:.3}s",
                total_duration.as_secs_f64()
            );
        } else {
            println!(
                "All LFS files downloaded and verified successfully! in {:.1}s",
                total_duration.as_secs_f64()
            );
        }
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
                        "  Found LFS file: {} ({} bytes)",
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
    progress_bar.set_message(format!("Verifying {}", file_name));
    let hash_valid = verify_file_hash(&lfs_file.path, &lfs_file.oid).await?;

    if hash_valid {
        progress_bar.finish_with_message(format!("{} (verified)", file_name));
    } else {
        progress_bar.finish_with_message(format!("{} (hash mismatch)", file_name));
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

    // Essential files: only .safetensors and tokenizer files
    let files_to_try = vec![
        "model.safetensors",     // Primary model weights (safetensors format only)
        "tokenizer.json",        // Tokenizer configuration
        "tokenizer_config.json", // Tokenizer metadata
    ];

    let mut downloaded_any = false;

    for file in files_to_try {
        let url = format!("{}/{}", base_url, file);
        let output_file = format!("{}/{}", output_dir, file);

        println!("  Trying to download: {}", file);

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
                println!("  Downloaded: {}", file);
                downloaded_any = true;
            }
            _ => {
                println!("  Failed to download: {}", file);
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
    println!("    Match: {}", if hash_match { "YES" } else { "NO" });

    Ok(hash_match)
}

/// Download model using native Rust hf-hub crate (includes xet support)
pub(crate) async fn download_model_with_xet(
    model_id: &str,
    output_dir: &str,
    hf_token: Option<&str>,
    cache_dir: &Path,
) -> Result<()> {
    println!("Using HuggingFace Hub's native API with xet backend for optimal performance");

    // Initialize the API client with custom cache directory
    let api_result = if let Some(token) = hf_token {
        hf_hub::api::tokio::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .with_token(Some(token.to_string()))
            .build()
    } else {
        hf_hub::api::tokio::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .build()
    };

    let api = match api_result {
        Ok(api) => api,
        Err(e) => {
            println!("ERROR: Failed to initialize HuggingFace Hub API: {}", e);
            println!("Falling back to Git LFS...");
            return download_huggingface_model(model_id, output_dir, hf_token, false).await;
        }
    };

    let repo = api.model(model_id.to_string());

    // Try to get the repository info first to see what files are available
    println!("Discovering available files...");

    // Essential files: only .safetensors and tokenizer files
    let essential_files = vec![
        "model.safetensors",     // Primary model weights (safetensors format only)
        "tokenizer.json",        // Tokenizer configuration
        "tokenizer_config.json", // Tokenizer metadata
    ];

    // Additional tokenizer files that some models might have
    let optional_tokenizer_files = vec![
        "vocab.txt",               // Vocabulary file
        "merges.txt",              // BPE merges file
        "special_tokens_map.json", // Special tokens configuration
    ];

    let mut downloaded_files = Vec::new();
    let mut cached_files = Vec::new();
    let mut failed_files = Vec::new();

    let start_time = std::time::Instant::now();

    // Download essential files first
    for filename in &essential_files {
        let file_start = std::time::Instant::now();

        match repo.get(filename).await {
            Ok(file_path) => {
                let target_path = Path::new(output_dir).join(filename);
                let file_duration = file_start.elapsed();

                // Quick downloads (< 100ms) are likely cache hits for reasonably sized files
                let is_cache_hit = file_duration.as_millis() < 100;

                match tokio::fs::copy(&file_path, &target_path).await {
                    Ok(_) => {
                        if is_cache_hit {
                            cached_files.push(filename.to_string());
                        } else {
                            downloaded_files.push(filename.to_string());
                        }
                    }
                    Err(e) => {
                        println!("WARNING: Failed to copy {}: {}", filename, e);
                        failed_files.push(filename.to_string());
                    }
                }
            }
            Err(_) => {
                // Silently skip files that don't exist - this is normal
                failed_files.push(filename.to_string());
            }
        }
    }

    // Try optional tokenizer files if we got the essential model file
    let has_model_file = downloaded_files.iter().any(|f| f.contains(".safetensors"))
        || cached_files.iter().any(|f| f.contains(".safetensors"));
    if has_model_file {
        for filename in &optional_tokenizer_files {
            let file_start = std::time::Instant::now();

            match repo.get(filename).await {
                Ok(file_path) => {
                    let target_path = Path::new(output_dir).join(filename);
                    let file_duration = file_start.elapsed();
                    let is_cache_hit = file_duration.as_millis() < 100;

                    if tokio::fs::copy(&file_path, &target_path).await.is_ok() {
                        if is_cache_hit {
                            cached_files.push(filename.to_string());
                        } else {
                            downloaded_files.push(filename.to_string());
                        }
                    }
                }
                Err(_) => {
                    // Silently skip optional tokenizer files that don't exist
                }
            }
        }
    }

    let all_files = [&downloaded_files[..], &cached_files[..]].concat();

    if all_files.is_empty() {
        println!("ERROR: No .safetensors model files found");
        println!("Falling back to Git LFS...");
        return download_huggingface_model(model_id, output_dir, hf_token, false).await;
    }

    // Ensure we have the essential model file
    if !has_model_file {
        println!("ERROR: Required model.safetensors file not found");
        println!("Falling back to Git LFS...");
        return download_huggingface_model(model_id, output_dir, hf_token, false).await;
    }

    let total_duration = start_time.elapsed();
    let mut total_size = 0u64;

    // Show downloaded files (new network downloads)
    if !downloaded_files.is_empty() {
        println!("Downloaded {} files:", downloaded_files.len());
        for filename in &downloaded_files {
            let file_path = Path::new(output_dir).join(filename);
            if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                let size = metadata.len();
                total_size += size;
                let size_mb = size as f64 / (1024.0 * 1024.0);
                println!("  - {} ({:.1} MB)", filename, size_mb);
            }
        }
    }

    // Show cached files (restored from local cache)
    if !cached_files.is_empty() {
        println!("Restored {} files from cache:", cached_files.len());
        for filename in &cached_files {
            let file_path = Path::new(output_dir).join(filename);
            if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                let size = metadata.len();
                total_size += size;
                let size_mb = size as f64 / (1024.0 * 1024.0);
                println!("  - {} ({:.1} MB)", filename, size_mb);
            }
        }
    }

    let total_mb = total_size as f64 / (1024.0 * 1024.0);

    if !cached_files.is_empty() && !downloaded_files.is_empty() {
        println!(
            "Total: {:.1} MB ({} downloaded, {} from cache) in {:.1}s",
            total_mb,
            downloaded_files.len(),
            cached_files.len(),
            total_duration.as_secs_f64()
        );
    } else if !cached_files.is_empty() {
        println!(
            "Total: {:.1} MB restored from cache in {:.3}s",
            total_mb,
            total_duration.as_secs_f64()
        );
    } else {
        println!(
            "Total: {:.1} MB downloaded successfully in {:.1}s",
            total_mb,
            total_duration.as_secs_f64()
        );
    }

    Ok(())
}
