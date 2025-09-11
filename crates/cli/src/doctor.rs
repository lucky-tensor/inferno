//! System diagnostics and hardware compatibility checking module
//!
//! This module implements the "doctor" command that checks system capabilities
//! for running inference engines, including GPU detection, driver versions,
//! model compatibility, and overall system readiness.

use crate::cli_options::DoctorCliOptions;
use anyhow::Result as AnyhowResult;
use inferno_shared::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::{Command, Stdio};
use sysinfo::System;
use walkdir::WalkDir;

/// GPU information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub memory_mb: Option<u64>,
    pub compute_capability: Option<String>,
    pub is_compatible: bool,
    pub issues: Vec<String>,
}

/// GPU vendor enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

/// CPU information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub name: String,
    pub cores: usize,
    pub threads: usize,
    pub frequency_mhz: u64,
    pub supports_avx: bool,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub is_compatible: bool,
    pub issues: Vec<String>,
}

/// Model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub format: ModelFormat,
    pub size_mb: u64,
    pub is_optimized: bool,
    pub compatible_backends: Vec<Backend>,
    pub issues: Vec<String>,
}

/// Model format enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    SafeTensors,
    Pytorch,
    Gguf,
    Onnx,
    Unknown,
}

/// Available backends enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Backend {
    Cpu,
    Cuda,
    Rocm,
}

/// Overall system diagnostics result
#[derive(Debug, Serialize, Deserialize)]
pub struct DiagnosticsResult {
    pub gpus: Vec<GpuInfo>,
    pub cpu: CpuInfo,
    pub models: Vec<ModelInfo>,
    pub compatibility_matrix: HashMap<String, HashMap<String, CompatibilityStatus>>,
    pub overall_score: u8,
    pub max_score: u8,
    pub recommendations: Vec<String>,
    pub system_ready: bool,
}

/// Compatibility status for model/backend combinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityStatus {
    Compatible,
    Warning(String),
    Incompatible(String),
}

impl std::fmt::Display for CompatibilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompatibilityStatus::Compatible => write!(f, "✅"),
            CompatibilityStatus::Warning(_) => write!(f, "⚠️"),
            CompatibilityStatus::Incompatible(_) => write!(f, "❌"),
        }
    }
}

/// Main entry point for the doctor command
pub async fn run_diagnostics(opts: DoctorCliOptions) -> Result<()> {
    println!("🔍 Inferno System Diagnostics");
    println!("=============================\n");

    if opts.verbose {
        println!("Scanning system configuration...");
    }

    // Collect system information
    let mut diagnostics = DiagnosticsResult {
        gpus: Vec::new(),
        cpu: detect_cpu_info()?,
        models: Vec::new(),
        compatibility_matrix: HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: Vec::new(),
        system_ready: false,
    };

    // Detect GPUs
    if opts.verbose {
        println!("Detecting GPUs...");
    }
    diagnostics.gpus = detect_gpus().await.map_err(|e| {
        inferno_shared::InfernoError::internal(format!("GPU detection failed: {}", e), None)
    })?;

    // Scan models
    if opts.verbose {
        println!("Scanning models in {}...", opts.model_dir);
    }
    diagnostics.models = scan_models(&opts.model_dir).map_err(|e| {
        inferno_shared::InfernoError::internal(format!("Model scanning failed: {}", e), None)
    })?;

    // Calculate compatibility matrix
    diagnostics.compatibility_matrix = calculate_compatibility_matrix(&diagnostics);

    // Calculate overall score
    calculate_overall_score(&mut diagnostics);

    // Generate recommendations
    generate_recommendations(&mut diagnostics);

    // Display results
    match opts.format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&diagnostics).unwrap());
        }
        "yaml" => {
            println!("{}", serde_yaml::to_string(&diagnostics).unwrap());
        }
        _ => {
            display_results_table(&diagnostics, opts.verbose);
        }
    }

    Ok(())
}

/// Detect NVIDIA and AMD GPUs
pub async fn detect_gpus() -> AnyhowResult<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Detect NVIDIA GPUs
    if let Ok(nvidia_gpus) = detect_nvidia_gpus().await {
        gpus.extend(nvidia_gpus);
    }

    // Detect AMD GPUs
    if let Ok(amd_gpus) = detect_amd_gpus().await {
        gpus.extend(amd_gpus);
    }

    Ok(gpus)
}

/// Detect NVIDIA GPUs using nvidia-smi
async fn detect_nvidia_gpus() -> AnyhowResult<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Try to run nvidia-smi
    let output = Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 4 {
                    let name = parts[0].to_string();
                    let driver_version = Some(parts[1].to_string());
                    let memory_mb = parts[2].parse::<u64>().ok();
                    let compute_capability = Some(parts[3].to_string());

                    // Get CUDA version
                    let cuda_version = get_cuda_version().await;

                    let mut issues = Vec::new();
                    let mut is_compatible = true;

                    // Check compatibility
                    if let Some(ref cc) = compute_capability {
                        if let Ok(cc_float) = cc.parse::<f32>() {
                            if cc_float < 7.0 {
                                issues.push("Compute capability < 7.0 may have limited support".to_string());
                                is_compatible = false;
                            }
                        }
                    }

                    // Check driver version
                    if let Some(ref driver) = driver_version {
                        if let Ok(version) = parse_driver_version(driver) {
                            if version < 525.0 {
                                issues.push("Driver version may be too old for CUDA 12.x".to_string());
                            }
                        }
                    }

                    gpus.push(GpuInfo {
                        vendor: GpuVendor::Nvidia,
                        name,
                        driver_version,
                        cuda_version,
                        memory_mb,
                        compute_capability,
                        is_compatible,
                        issues,
                    });
                }
            }
        }
        Ok(_) => {
            // nvidia-smi exists but failed
            // This might indicate driver issues
        }
        Err(_) => {
            // nvidia-smi not found, no NVIDIA drivers installed
        }
    }

    Ok(gpus)
}

/// Detect AMD GPUs using rocm-smi
async fn detect_amd_gpus() -> AnyhowResult<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Try to run rocm-smi
    let output = Command::new("rocm-smi")
        .args(&["--showproductname", "--showmeminfo", "vram", "--csv"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.starts_with("card") && line.contains(',') {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        let name = parts[1].trim().to_string();
                        
                        // Try to get memory info
                        let memory_mb = extract_memory_from_rocm_output(&stdout);
                        
                        let mut issues = Vec::new();
                        let mut is_compatible = false;

                        // Check ROCm installation
                        if !is_rocm_installed() {
                            issues.push("ROCm not properly installed".to_string());
                        } else {
                            is_compatible = true;
                        }

                        gpus.push(GpuInfo {
                            vendor: GpuVendor::Amd,
                            name,
                            driver_version: get_rocm_version(),
                            cuda_version: None,
                            memory_mb,
                            compute_capability: None,
                            is_compatible,
                            issues,
                        });
                    }
                }
            }
        }
        Ok(_) => {
            // rocm-smi exists but failed
        }
        Err(_) => {
            // rocm-smi not found
            // Check for AMD GPUs in lspci
            if let Ok(amd_detected) = check_amd_gpus_lspci() {
                if amd_detected {
                    gpus.push(GpuInfo {
                        vendor: GpuVendor::Amd,
                        name: "AMD GPU (ROCm not installed)".to_string(),
                        driver_version: None,
                        cuda_version: None,
                        memory_mb: None,
                        compute_capability: None,
                        is_compatible: false,
                        issues: vec!["ROCm drivers not installed".to_string()],
                    });
                }
            }
        }
    }

    Ok(gpus)
}

/// Get CUDA version from nvcc
async fn get_cuda_version() -> Option<String> {
    let output = Command::new("nvcc")
        .args(&["--version"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            // Extract version using regex
            let re = Regex::new(r"release (\d+\.\d+)").ok()?;
            if let Some(captures) = re.captures(&stdout) {
                return Some(captures[1].to_string());
            }
        }
    }
    None
}

/// Parse driver version string to float for comparison
pub fn parse_driver_version(version: &str) -> AnyhowResult<f32> {
    let re = Regex::new(r"(\d+)\.(\d+)")?;
    if let Some(captures) = re.captures(version) {
        let major: u32 = captures[1].parse()?;
        let minor: u32 = captures[2].parse()?;
        Ok(major as f32 + (minor as f32 / 100.0))
    } else {
        Err(anyhow::anyhow!("Invalid version format"))
    }
}

/// Extract memory information from rocm-smi output
pub fn extract_memory_from_rocm_output(output: &str) -> Option<u64> {
    for line in output.lines() {
        if line.contains("Total VRAM") {
            let re = Regex::new(r"(\d+)\s*MB").ok()?;
            if let Some(captures) = re.captures(line) {
                return captures[1].parse().ok();
            }
        }
    }
    None
}

/// Check if ROCm is properly installed
pub fn is_rocm_installed() -> bool {
    std::path::Path::new("/opt/rocm").exists() || 
    Command::new("hipcc").arg("--version").output().is_ok()
}

/// Get ROCm version
fn get_rocm_version() -> Option<String> {
    let output = Command::new("hipcc")
        .args(&["--version"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let re = Regex::new(r"HIP version: (\d+\.\d+\.\d+)").ok()?;
            if let Some(captures) = re.captures(&stdout) {
                return Some(captures[1].to_string());
            }
        }
    }
    None
}

/// Check for AMD GPUs using lspci
pub fn check_amd_gpus_lspci() -> AnyhowResult<bool> {
    let output = Command::new("lspci")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.to_lowercase().contains("amd") || stdout.to_lowercase().contains("radeon"))
    } else {
        Ok(false)
    }
}

/// Detect CPU information and capabilities
pub fn detect_cpu_info() -> Result<CpuInfo> {
    let mut system = System::new_all();
    system.refresh_cpu();

    let cpus = system.cpus();
    let cpu_name = if !cpus.is_empty() {
        cpus[0].brand().to_string()
    } else {
        "Unknown CPU".to_string()
    };

    let cores = system.physical_core_count().unwrap_or(1);
    let threads = cpus.len();
    let frequency_mhz = if !cpus.is_empty() {
        cpus[0].frequency()
    } else {
        0
    };

    // Detect instruction set support
    let (supports_avx, supports_avx2, supports_avx512) = detect_cpu_features();

    let mut issues = Vec::new();
    let mut is_compatible = true;

    if cores < 2 {
        issues.push("Low core count may impact performance".to_string());
    }

    if !supports_avx2 {
        issues.push("AVX2 not supported, CPU inference may be slow".to_string());
        is_compatible = false;
    }

    Ok(CpuInfo {
        name: cpu_name,
        cores,
        threads,
        frequency_mhz,
        supports_avx,
        supports_avx2,
        supports_avx512,
        is_compatible,
        issues,
    })
}

/// Detect CPU instruction set features
pub fn detect_cpu_features() -> (bool, bool, bool) {
    // Try to detect features using cpuid on x86/x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avx512f") {
            return (true, true, true);
        } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            return (true, true, false);
        } else if is_x86_feature_detected!("avx") {
            return (true, false, false);
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            return (true, true, false);
        } else if is_x86_feature_detected!("avx") {
            return (true, false, false);
        }
    }

    // Default fallback
    (false, false, false)
}

/// Scan for models in the specified directory
pub fn scan_models(model_dir: &str) -> AnyhowResult<Vec<ModelInfo>> {
    let mut models = Vec::new();

    if !std::path::Path::new(model_dir).exists() {
        return Ok(models);
    }

    for entry in WalkDir::new(model_dir).max_depth(3) {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                let format = match extension.to_str() {
                    Some("safetensors") => ModelFormat::SafeTensors,
                    Some("pt") | Some("pth") | Some("bin") => ModelFormat::Pytorch,
                    Some("gguf") => ModelFormat::Gguf,
                    Some("onnx") => ModelFormat::Onnx,
                    _ => continue,
                };

                let metadata = entry.metadata()?;
                let size_mb = metadata.len() / 1024 / 1024;

                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                let is_optimized = check_model_optimization(path, &format);
                let compatible_backends = determine_compatible_backends(&format);

                let mut issues = Vec::new();
                if format == ModelFormat::Gguf {
                    issues.push("GGUF format not yet supported by Burn framework".to_string());
                }
                if size_mb > 4096 {
                    issues.push("Large model may require significant GPU memory".to_string());
                }

                models.push(ModelInfo {
                    name,
                    path: path.to_string_lossy().to_string(),
                    format,
                    size_mb,
                    is_optimized,
                    compatible_backends,
                    issues,
                });
            }
        }
    }

    Ok(models)
}

/// Check if a model has been optimized
pub fn check_model_optimization(path: &std::path::Path, format: &ModelFormat) -> bool {
    match format {
        ModelFormat::SafeTensors => {
            // Check for quantization markers or specific naming patterns
            let path_str = path.to_string_lossy().to_lowercase();
            path_str.contains("quantized") || 
            path_str.contains("int8") || 
            path_str.contains("fp16") ||
            path_str.contains("optimized")
        }
        _ => false,
    }
}

/// Determine which backends are compatible with a model format
pub fn determine_compatible_backends(format: &ModelFormat) -> Vec<Backend> {
    match format {
        ModelFormat::SafeTensors => vec![Backend::Cpu, Backend::Cuda, Backend::Rocm],
        ModelFormat::Pytorch => vec![Backend::Cpu, Backend::Cuda],
        ModelFormat::Onnx => vec![Backend::Cpu, Backend::Cuda],
        ModelFormat::Gguf => vec![], // Not yet supported
        ModelFormat::Unknown => vec![], // Unknown format not supported
    }
}

/// Calculate compatibility matrix for models and backends
pub fn calculate_compatibility_matrix(
    diagnostics: &DiagnosticsResult,
) -> HashMap<String, HashMap<String, CompatibilityStatus>> {
    let mut matrix = HashMap::new();

    for model in &diagnostics.models {
        let mut model_compatibility = HashMap::new();

        // CPU compatibility
        if model.compatible_backends.contains(&Backend::Cpu) {
            let status = if diagnostics.cpu.is_compatible {
                if model.size_mb > 2048 {
                    CompatibilityStatus::Warning("Large model may be slow on CPU".to_string())
                } else {
                    CompatibilityStatus::Compatible
                }
            } else {
                CompatibilityStatus::Incompatible("CPU not suitable for inference".to_string())
            };
            model_compatibility.insert("CPU".to_string(), status);
        } else {
            model_compatibility.insert(
                "CPU".to_string(),
                CompatibilityStatus::Incompatible("Model format not supported".to_string()),
            );
        }

        // CUDA compatibility
        let has_nvidia_gpu = diagnostics
            .gpus
            .iter()
            .any(|gpu| gpu.vendor == GpuVendor::Nvidia && gpu.is_compatible);

        if model.compatible_backends.contains(&Backend::Cuda) {
            let status = if has_nvidia_gpu {
                CompatibilityStatus::Compatible
            } else {
                CompatibilityStatus::Incompatible("No compatible NVIDIA GPU found".to_string())
            };
            model_compatibility.insert("CUDA".to_string(), status);
        } else {
            model_compatibility.insert(
                "CUDA".to_string(),
                CompatibilityStatus::Incompatible("Model format not supported".to_string()),
            );
        }

        // ROCm compatibility
        let has_amd_gpu = diagnostics
            .gpus
            .iter()
            .any(|gpu| gpu.vendor == GpuVendor::Amd && gpu.is_compatible);

        if model.compatible_backends.contains(&Backend::Rocm) {
            let status = if has_amd_gpu {
                CompatibilityStatus::Compatible
            } else {
                CompatibilityStatus::Incompatible("No compatible AMD GPU found".to_string())
            };
            model_compatibility.insert("ROCm".to_string(), status);
        } else {
            model_compatibility.insert(
                "ROCm".to_string(),
                CompatibilityStatus::Incompatible("Model format not supported".to_string()),
            );
        }

        matrix.insert(model.name.clone(), model_compatibility);
    }

    matrix
}

/// Calculate overall system score
pub fn calculate_overall_score(diagnostics: &mut DiagnosticsResult) {
    let mut score = 0;
    let mut max_score = 0;

    // CPU score
    max_score += 2;
    if diagnostics.cpu.is_compatible {
        score += 2;
    } else if diagnostics.cpu.cores >= 2 {
        score += 1;
    }

    // GPU score
    max_score += 3;
    let nvidia_count = diagnostics
        .gpus
        .iter()
        .filter(|g| g.vendor == GpuVendor::Nvidia && g.is_compatible)
        .count();
    let amd_count = diagnostics
        .gpus
        .iter()
        .filter(|g| g.vendor == GpuVendor::Amd && g.is_compatible)
        .count();

    if nvidia_count > 0 {
        score += 3;
    } else if amd_count > 0 {
        score += 2;
    } else if !diagnostics.gpus.is_empty() {
        score += 1;
    }

    // Models score
    max_score += 2;
    let compatible_models = diagnostics
        .models
        .iter()
        .filter(|m| m.format == ModelFormat::SafeTensors)
        .count();

    if compatible_models > 0 {
        score += 2;
    } else if !diagnostics.models.is_empty() {
        score += 1;
    }

    // Driver/Software score
    max_score += 3;
    let has_cuda = diagnostics
        .gpus
        .iter()
        .any(|g| g.vendor == GpuVendor::Nvidia && g.cuda_version.is_some());
    let has_rocm = diagnostics
        .gpus
        .iter()
        .any(|g| g.vendor == GpuVendor::Amd && g.driver_version.is_some());

    if has_cuda && has_rocm {
        score += 3;
    } else if has_cuda || has_rocm {
        score += 2;
    }

    diagnostics.overall_score = score;
    diagnostics.max_score = max_score;
    diagnostics.system_ready = score >= (max_score / 2);
}

/// Generate recommendations based on diagnostics
pub fn generate_recommendations(diagnostics: &mut DiagnosticsResult) {
    // CPU recommendations
    if !diagnostics.cpu.supports_avx2 {
        diagnostics
            .recommendations
            .push("Consider upgrading CPU to support AVX2 for better performance".to_string());
    }

    // GPU recommendations
    let nvidia_gpus = diagnostics
        .gpus
        .iter()
        .filter(|g| g.vendor == GpuVendor::Nvidia)
        .count();
    let amd_gpus = diagnostics
        .gpus
        .iter()
        .filter(|g| g.vendor == GpuVendor::Amd)
        .count();

    if nvidia_gpus == 0 && amd_gpus == 0 {
        diagnostics
            .recommendations
            .push("Consider adding a GPU for accelerated inference".to_string());
    }

    // Driver recommendations
    for gpu in &diagnostics.gpus {
        if !gpu.issues.is_empty() {
            if gpu.vendor == GpuVendor::Nvidia && gpu.cuda_version.is_none() {
                diagnostics
                    .recommendations
                    .push("Install CUDA toolkit for NVIDIA GPU acceleration".to_string());
            }
            if gpu.vendor == GpuVendor::Amd && !is_rocm_installed() {
                diagnostics
                    .recommendations
                    .push("Install ROCm drivers for AMD GPU acceleration".to_string());
            }
        }
    }

    // Model recommendations
    let safetensors_count = diagnostics
        .models
        .iter()
        .filter(|m| m.format == ModelFormat::SafeTensors)
        .count();

    if safetensors_count == 0 && !diagnostics.models.is_empty() {
        diagnostics
            .recommendations
            .push("Convert models to SafeTensors format for best compatibility".to_string());
    }

    if diagnostics.models.is_empty() {
        diagnostics
            .recommendations
            .push("Download models using 'inferno download' command".to_string());
    }
}

/// Display results in table format
pub fn display_results_table(diagnostics: &DiagnosticsResult, verbose: bool) {
    // Hardware Detection
    println!("Hardware Detection:");
    println!("==================");

    // CPU Info
    println!(
        "{} CPU: {} ({} cores, {} threads)",
        if diagnostics.cpu.is_compatible {
            "✅"
        } else {
            "⚠️"
        },
        diagnostics.cpu.name,
        diagnostics.cpu.cores,
        diagnostics.cpu.threads
    );

    if verbose || !diagnostics.cpu.is_compatible {
        println!("   Features: AVX: {}, AVX2: {}, AVX512: {}", 
            if diagnostics.cpu.supports_avx { "✅" } else { "❌" },
            if diagnostics.cpu.supports_avx2 { "✅" } else { "❌" },
            if diagnostics.cpu.supports_avx512 { "✅" } else { "❌" }
        );
        for issue in &diagnostics.cpu.issues {
            println!("   ⚠️  {}", issue);
        }
    }

    // GPU Info
    if diagnostics.gpus.is_empty() {
        println!("❌ No GPUs detected");
    } else {
        for gpu in &diagnostics.gpus {
            let vendor_str = match gpu.vendor {
                GpuVendor::Nvidia => "NVIDIA",
                GpuVendor::Amd => "AMD",
                GpuVendor::Intel => "Intel",
                GpuVendor::Unknown => "Unknown",
            };

            println!(
                "{} GPU: {} {} {}",
                if gpu.is_compatible { "✅" } else { "⚠️" },
                vendor_str,
                gpu.name,
                gpu.memory_mb
                    .map(|m| format!("({} MB)", m))
                    .unwrap_or_default()
            );

            if verbose || !gpu.is_compatible {
                if let Some(ref driver) = gpu.driver_version {
                    println!("   Driver: {}", driver);
                }
                if let Some(ref cuda) = gpu.cuda_version {
                    println!("   CUDA: {}", cuda);
                }
                if let Some(ref cc) = gpu.compute_capability {
                    println!("   Compute Capability: {}", cc);
                }
                for issue in &gpu.issues {
                    println!("   ⚠️  {}", issue);
                }
            }
        }
    }

    println!();

    // Model Compatibility
    println!("Model Compatibility:");
    println!("===================");

    if diagnostics.models.is_empty() {
        println!("No models found in model directory");
    } else {
        println!("{:<30} {:<8} {:<8} {:<8}", "Model", "CPU", "CUDA", "ROCm");
        println!("{:-<54}", "");

        for model in &diagnostics.models {
            if let Some(model_compat) = diagnostics.compatibility_matrix.get(&model.name) {
                println!(
                    "{:<30} {:<8} {:<8} {:<8}",
                    if model.name.len() > 29 {
                        format!("{}...", &model.name[..26])
                    } else {
                        model.name.clone()
                    },
                    model_compat.get("CPU").unwrap_or(&CompatibilityStatus::Incompatible("N/A".to_string())),
                    model_compat.get("CUDA").unwrap_or(&CompatibilityStatus::Incompatible("N/A".to_string())),
                    model_compat.get("ROCm").unwrap_or(&CompatibilityStatus::Incompatible("N/A".to_string()))
                );
            }
        }
    }

    println!();

    // Overall Score
    println!("System Readiness:");
    println!("================");
    println!(
        "Score: {}/{} ({:.1}%)",
        diagnostics.overall_score,
        diagnostics.max_score,
        (diagnostics.overall_score as f32 / diagnostics.max_score as f32) * 100.0
    );

    println!(
        "Status: {}",
        if diagnostics.system_ready {
            "✅ System ready for inference"
        } else {
            "⚠️  System needs configuration"
        }
    );

    if !diagnostics.recommendations.is_empty() {
        println!();
        println!("Recommendations:");
        for (i, rec) in diagnostics.recommendations.iter().enumerate() {
            println!("{}. {}", i + 1, rec);
        }
    }
}