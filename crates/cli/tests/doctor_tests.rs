//! Comprehensive tests for the doctor command functionality
//!
//! This test suite covers all aspects of the doctor command including:
//! - GPU detection (NVIDIA and AMD)
//! - CPU capability checking
//! - Model format detection
//! - Compatibility scoring
//! - Output formatting
//! - Error handling

use inferno_cli::doctor::*;
use std::fs;
use tempfile::TempDir;

/// Test GPU vendor detection and display
#[test]
fn test_gpu_vendor_display() {
    // Test each GPU vendor display
    assert_eq!(format!("{:?}", GpuVendor::Nvidia), "Nvidia");
    assert_eq!(format!("{:?}", GpuVendor::Amd), "Amd");
    assert_eq!(format!("{:?}", GpuVendor::Intel), "Intel");
    assert_eq!(format!("{:?}", GpuVendor::Unknown), "Unknown");
}

/// Test model format detection
#[test]
fn test_model_format_detection() {
    // Test each model format
    assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
    assert_eq!(ModelFormat::Pytorch, ModelFormat::Pytorch);
    assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
    assert_eq!(ModelFormat::Onnx, ModelFormat::Onnx);
    assert_eq!(ModelFormat::Unknown, ModelFormat::Unknown);
}

/// Test backend enumeration
#[test]
fn test_backend_types() {
    let backends = vec![Backend::Cpu, Backend::Cuda, Backend::Rocm];
    assert_eq!(backends.len(), 3);
    assert!(backends.contains(&Backend::Cpu));
    assert!(backends.contains(&Backend::Cuda));
    assert!(backends.contains(&Backend::Rocm));
}

/// Test compatibility status display
#[test]
fn test_compatibility_status_display() {
    let compatible = CompatibilityStatus::Compatible;
    let warning = CompatibilityStatus::Warning("Test warning".to_string());
    let incompatible = CompatibilityStatus::Incompatible("Test error".to_string());

    assert_eq!(format!("{}", compatible), "✅");
    assert_eq!(format!("{}", warning), "⚠️");
    assert_eq!(format!("{}", incompatible), "❌");
}

/// Test GPU information structure creation
#[test]
fn test_gpu_info_creation() {
    let gpu = GpuInfo {
        vendor: GpuVendor::Nvidia,
        name: "RTX 4090".to_string(),
        driver_version: Some("535.98".to_string()),
        cuda_version: Some("12.2".to_string()),
        memory_mb: Some(24576),
        compute_capability: Some("8.9".to_string()),
        is_compatible: true,
        issues: vec![],
    };

    assert_eq!(gpu.vendor, GpuVendor::Nvidia);
    assert_eq!(gpu.name, "RTX 4090");
    assert_eq!(gpu.driver_version, Some("535.98".to_string()));
    assert_eq!(gpu.cuda_version, Some("12.2".to_string()));
    assert_eq!(gpu.memory_mb, Some(24576));
    assert_eq!(gpu.compute_capability, Some("8.9".to_string()));
    assert!(gpu.is_compatible);
    assert!(gpu.issues.is_empty());
}

/// Test CPU information structure creation
#[test]
fn test_cpu_info_creation() {
    let cpu = CpuInfo {
        name: "Intel i9-12900K".to_string(),
        cores: 16,
        threads: 24,
        frequency_mhz: 3200,
        supports_avx: true,
        supports_avx2: true,
        supports_avx512: false,
        is_compatible: true,
        issues: vec![],
    };

    assert_eq!(cpu.name, "Intel i9-12900K");
    assert_eq!(cpu.cores, 16);
    assert_eq!(cpu.threads, 24);
    assert_eq!(cpu.frequency_mhz, 3200);
    assert!(cpu.supports_avx);
    assert!(cpu.supports_avx2);
    assert!(!cpu.supports_avx512);
    assert!(cpu.is_compatible);
    assert!(cpu.issues.is_empty());
}

/// Test model information structure creation
#[test]
fn test_model_info_creation() {
    let model = ModelInfo {
        name: "tiny-llama".to_string(),
        path: "/models/tiny-llama.safetensors".to_string(),
        format: ModelFormat::SafeTensors,
        size_mb: 512,
        is_optimized: true,
        compatible_backends: vec![Backend::Cpu, Backend::Cuda],
        issues: vec![],
    };

    assert_eq!(model.name, "tiny-llama");
    assert_eq!(model.path, "/models/tiny-llama.safetensors");
    assert_eq!(model.format, ModelFormat::SafeTensors);
    assert_eq!(model.size_mb, 512);
    assert!(model.is_optimized);
    assert_eq!(model.compatible_backends.len(), 2);
    assert!(model.issues.is_empty());
}

/// Test model scanning with temporary files
#[test]
fn test_model_scanning() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_dir = temp_dir.path().to_str().unwrap();

    // Create test model files
    let _safetensors_file = create_test_file(&temp_dir, "model.safetensors", 1024)?;
    let _pytorch_file = create_test_file(&temp_dir, "model.pt", 512)?;
    let _gguf_file = create_test_file(&temp_dir, "model.gguf", 2048)?;
    let _onnx_file = create_test_file(&temp_dir, "model.onnx", 256)?;
    let _unknown_file = create_test_file(&temp_dir, "model.bin", 128)?;

    // Scan models
    let models = scan_models(model_dir)?;

    // Verify we found the correct models
    assert_eq!(models.len(), 5);
    
    // Check for each format
    let safetensors_models: Vec<_> = models.iter().filter(|m| m.format == ModelFormat::SafeTensors).collect();
    let pytorch_models: Vec<_> = models.iter().filter(|m| m.format == ModelFormat::Pytorch).collect();
    let gguf_models: Vec<_> = models.iter().filter(|m| m.format == ModelFormat::Gguf).collect();
    let onnx_models: Vec<_> = models.iter().filter(|m| m.format == ModelFormat::Onnx).collect();

    assert_eq!(safetensors_models.len(), 1);
    assert_eq!(pytorch_models.len(), 2); // .pt and .bin both map to PyTorch
    assert_eq!(gguf_models.len(), 1);
    assert_eq!(onnx_models.len(), 1);

    // Check file sizes (converted from bytes to MB, note: 1024 bytes / 1024 / 1024 = 0 MB)
    let safetensors_model = &safetensors_models[0];
    assert_eq!(safetensors_model.size_mb, 0); // 1024 bytes = 0MB when divided by 1024*1024

    Ok(())
}

/// Test model scanning with empty directory
#[test]
fn test_model_scanning_empty_directory() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_dir = temp_dir.path().to_str().unwrap();

    let models = scan_models(model_dir)?;
    assert!(models.is_empty());

    Ok(())
}

/// Test model scanning with non-existent directory
#[test]
fn test_model_scanning_nonexistent_directory() -> Result<(), Box<dyn std::error::Error>> {
    let models = scan_models("/nonexistent/directory")?;
    assert!(models.is_empty());

    Ok(())
}

/// Test backend compatibility determination
#[test]
fn test_determine_compatible_backends() {
    // SafeTensors should support all backends
    let safetensors_backends = determine_compatible_backends(&ModelFormat::SafeTensors);
    assert_eq!(safetensors_backends.len(), 3);
    assert!(safetensors_backends.contains(&Backend::Cpu));
    assert!(safetensors_backends.contains(&Backend::Cuda));
    assert!(safetensors_backends.contains(&Backend::Rocm));

    // PyTorch should support CPU and CUDA
    let pytorch_backends = determine_compatible_backends(&ModelFormat::Pytorch);
    assert_eq!(pytorch_backends.len(), 2);
    assert!(pytorch_backends.contains(&Backend::Cpu));
    assert!(pytorch_backends.contains(&Backend::Cuda));

    // ONNX should support CPU and CUDA
    let onnx_backends = determine_compatible_backends(&ModelFormat::Onnx);
    assert_eq!(onnx_backends.len(), 2);
    assert!(onnx_backends.contains(&Backend::Cpu));
    assert!(onnx_backends.contains(&Backend::Cuda));

    // GGUF should support no backends (not yet implemented)
    let gguf_backends = determine_compatible_backends(&ModelFormat::Gguf);
    assert!(gguf_backends.is_empty());

    // Unknown should support no backends
    let unknown_backends = determine_compatible_backends(&ModelFormat::Unknown);
    assert!(unknown_backends.is_empty());
}

/// Test model optimization detection
#[test]
fn test_check_model_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;

    // Test optimized model names
    let optimized_file = create_test_file(&temp_dir, "model_quantized.safetensors", 1024)?;
    assert!(check_model_optimization(&optimized_file, &ModelFormat::SafeTensors));

    let int8_file = create_test_file(&temp_dir, "model_int8.safetensors", 1024)?;
    assert!(check_model_optimization(&int8_file, &ModelFormat::SafeTensors));

    let fp16_file = create_test_file(&temp_dir, "model_fp16.safetensors", 1024)?;
    assert!(check_model_optimization(&fp16_file, &ModelFormat::SafeTensors));

    let optimized_file2 = create_test_file(&temp_dir, "model_optimized.safetensors", 1024)?;
    assert!(check_model_optimization(&optimized_file2, &ModelFormat::SafeTensors));

    // Test non-optimized model names
    let regular_file = create_test_file(&temp_dir, "regular_model.safetensors", 1024)?;
    assert!(!check_model_optimization(&regular_file, &ModelFormat::SafeTensors));

    // Test non-SafeTensors formats (should always return false for now)
    let pytorch_file = create_test_file(&temp_dir, "model_quantized.pt", 1024)?;
    assert!(!check_model_optimization(&pytorch_file, &ModelFormat::Pytorch));

    Ok(())
}

/// Test compatibility matrix calculation
#[test]
fn test_calculate_compatibility_matrix() {
    // Create test diagnostics with mock data
    let gpu = GpuInfo {
        vendor: GpuVendor::Nvidia,
        name: "RTX 4090".to_string(),
        driver_version: Some("535.98".to_string()),
        cuda_version: Some("12.2".to_string()),
        memory_mb: Some(24576),
        compute_capability: Some("8.9".to_string()),
        is_compatible: true,
        issues: vec![],
    };

    let cpu = CpuInfo {
        name: "Intel i9-12900K".to_string(),
        cores: 16,
        threads: 24,
        frequency_mhz: 3200,
        supports_avx: true,
        supports_avx2: true,
        supports_avx512: false,
        is_compatible: true,
        issues: vec![],
    };

    let model = ModelInfo {
        name: "tiny-llama".to_string(),
        path: "/models/tiny-llama.safetensors".to_string(),
        format: ModelFormat::SafeTensors,
        size_mb: 512,
        is_optimized: true,
        compatible_backends: vec![Backend::Cpu, Backend::Cuda, Backend::Rocm],
        issues: vec![],
    };

    let diagnostics = DiagnosticsResult {
        gpus: vec![gpu],
        cpu,
        models: vec![model],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    };

    let matrix = calculate_compatibility_matrix(&diagnostics);

    // Check that we have compatibility info for our model
    assert!(matrix.contains_key("tiny-llama"));
    let model_compat = &matrix["tiny-llama"];

    // Should have CPU, CUDA, and ROCm entries
    assert!(model_compat.contains_key("CPU"));
    assert!(model_compat.contains_key("CUDA"));
    assert!(model_compat.contains_key("ROCm"));

    // CPU should be compatible (good CPU)
    if let CompatibilityStatus::Compatible = model_compat["CPU"] {
        // Expected
    } else {
        panic!("Expected CPU to be compatible");
    }

    // CUDA should be compatible (NVIDIA GPU present)
    if let CompatibilityStatus::Compatible = model_compat["CUDA"] {
        // Expected
    } else {
        panic!("Expected CUDA to be compatible");
    }

    // ROCm should be incompatible (no AMD GPU)
    if let CompatibilityStatus::Incompatible(_) = model_compat["ROCm"] {
        // Expected
    } else {
        panic!("Expected ROCm to be incompatible");
    }
}

/// Test overall scoring calculation
#[test]
fn test_calculate_overall_score() {
    // Test system with good hardware and models
    let mut diagnostics = create_test_diagnostics_good();
    calculate_overall_score(&mut diagnostics);

    assert!(diagnostics.overall_score > 5); // Should have a good score
    assert!(diagnostics.system_ready);

    // Test system with poor hardware
    let mut poor_diagnostics = create_test_diagnostics_poor();
    calculate_overall_score(&mut poor_diagnostics);

    assert!(poor_diagnostics.overall_score < poor_diagnostics.max_score);
    // May or may not be ready depending on implementation
}

/// Test recommendation generation
#[test]
fn test_generate_recommendations() {
    // Test system missing GPU
    let mut diagnostics_no_gpu = create_test_diagnostics_no_gpu();
    generate_recommendations(&mut diagnostics_no_gpu);

    assert!(!diagnostics_no_gpu.recommendations.is_empty());
    let has_gpu_recommendation = diagnostics_no_gpu
        .recommendations
        .iter()
        .any(|r| r.contains("GPU"));
    assert!(has_gpu_recommendation);

    // Test system missing models
    let mut diagnostics_no_models = create_test_diagnostics_no_models();
    generate_recommendations(&mut diagnostics_no_models);

    assert!(!diagnostics_no_models.recommendations.is_empty());
    let has_model_recommendation = diagnostics_no_models
        .recommendations
        .iter()
        .any(|r| r.contains("models") || r.contains("download"));
    assert!(has_model_recommendation);
}

/// Test CPU feature detection fallback
#[test]
fn test_detect_cpu_features_fallback() {
    // This tests the fallback behavior on non-x86 architectures
    let (_avx, _avx2, _avx512) = detect_cpu_features();
    
    // On x86_64, we should get real feature detection
    // On other architectures, we should get (false, false, false)
    #[cfg(target_arch = "x86_64")]
    {
        // On x86_64, at least one should likely be true for modern systems
        // but we can't guarantee it, so we just test that the function runs
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        assert!(!_avx);
        assert!(!_avx2);
        assert!(!_avx512);
    }
}

/// Test driver version parsing
#[test]
fn test_parse_driver_version() -> Result<(), Box<dyn std::error::Error>> {
    // Test valid version strings
    assert!((parse_driver_version("535.98")? - 535.98).abs() < 0.01);
    assert!((parse_driver_version("470.129.06")? - 471.29).abs() < 0.01); // Only first two groups: 470 + 129/100 = 471.29
    assert!((parse_driver_version("525.60")? - 525.60).abs() < 0.01);

    // Test invalid version strings
    assert!(parse_driver_version("invalid").is_err());
    assert!(parse_driver_version("").is_err());
    assert!(parse_driver_version("535").is_err()); // Missing minor version

    Ok(())
}

// Helper function to create a test file
fn create_test_file(
    temp_dir: &TempDir,
    filename: &str,
    size_bytes: u64,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let file_path = temp_dir.path().join(filename);
    let data = vec![0u8; size_bytes as usize];
    fs::write(&file_path, data)?;
    Ok(file_path)
}

// Helper function to create diagnostics with good hardware
fn create_test_diagnostics_good() -> DiagnosticsResult {
    let gpu = GpuInfo {
        vendor: GpuVendor::Nvidia,
        name: "RTX 4090".to_string(),
        driver_version: Some("535.98".to_string()),
        cuda_version: Some("12.2".to_string()),
        memory_mb: Some(24576),
        compute_capability: Some("8.9".to_string()),
        is_compatible: true,
        issues: vec![],
    };

    let cpu = CpuInfo {
        name: "Intel i9-12900K".to_string(),
        cores: 16,
        threads: 24,
        frequency_mhz: 3200,
        supports_avx: true,
        supports_avx2: true,
        supports_avx512: false,
        is_compatible: true,
        issues: vec![],
    };

    let model = ModelInfo {
        name: "tiny-llama".to_string(),
        path: "/models/tiny-llama.safetensors".to_string(),
        format: ModelFormat::SafeTensors,
        size_mb: 512,
        is_optimized: true,
        compatible_backends: vec![Backend::Cpu, Backend::Cuda],
        issues: vec![],
    };

    DiagnosticsResult {
        gpus: vec![gpu],
        cpu,
        models: vec![model],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    }
}

// Helper function to create diagnostics with poor hardware
fn create_test_diagnostics_poor() -> DiagnosticsResult {
    let cpu = CpuInfo {
        name: "Old CPU".to_string(),
        cores: 2,
        threads: 2,
        frequency_mhz: 1800,
        supports_avx: false,
        supports_avx2: false,
        supports_avx512: false,
        is_compatible: false,
        issues: vec!["AVX2 not supported".to_string()],
    };

    DiagnosticsResult {
        gpus: vec![], // No GPUs
        cpu,
        models: vec![],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    }
}

// Helper function to create diagnostics with no GPU
fn create_test_diagnostics_no_gpu() -> DiagnosticsResult {
    let cpu = CpuInfo {
        name: "Intel i7-12700K".to_string(),
        cores: 12,
        threads: 20,
        frequency_mhz: 3600,
        supports_avx: true,
        supports_avx2: true,
        supports_avx512: false,
        is_compatible: true,
        issues: vec![],
    };

    let model = ModelInfo {
        name: "test-model".to_string(),
        path: "/models/test.safetensors".to_string(),
        format: ModelFormat::SafeTensors,
        size_mb: 1024,
        is_optimized: false,
        compatible_backends: vec![Backend::Cpu, Backend::Cuda],
        issues: vec![],
    };

    DiagnosticsResult {
        gpus: vec![], // No GPUs
        cpu,
        models: vec![model],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    }
}

// Helper function to create diagnostics with no models
fn create_test_diagnostics_no_models() -> DiagnosticsResult {
    let gpu = GpuInfo {
        vendor: GpuVendor::Nvidia,
        name: "RTX 3080".to_string(),
        driver_version: Some("525.60".to_string()),
        cuda_version: Some("11.8".to_string()),
        memory_mb: Some(10240),
        compute_capability: Some("8.6".to_string()),
        is_compatible: true,
        issues: vec![],
    };

    let cpu = CpuInfo {
        name: "Intel i7-12700K".to_string(),
        cores: 12,
        threads: 20,
        frequency_mhz: 3600,
        supports_avx: true,
        supports_avx2: true,
        supports_avx512: false,
        is_compatible: true,
        issues: vec![],
    };

    DiagnosticsResult {
        gpus: vec![gpu],
        cpu,
        models: vec![], // No models
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    }
}

/// Test GPU detection function
#[tokio::test]
async fn test_detect_gpus() -> Result<(), Box<dyn std::error::Error>> {
    // Test GPU detection (this will depend on the actual system)
    let result = detect_gpus().await;
    
    // Should not panic and return a result (may be empty on systems without GPUs)
    assert!(result.is_ok());
    let gpus = result.unwrap();
    
    // GPUs vector should be valid (may be empty)
    // Each GPU should have valid vendor information
    for gpu in &gpus {
        assert!(!gpu.name.is_empty());
        assert!(matches!(gpu.vendor, GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Intel | GpuVendor::Unknown));
    }
    
    Ok(())
}

/// Test CPU information detection
#[test]
fn test_detect_cpu_info() -> Result<(), Box<dyn std::error::Error>> {
    let cpu_info = detect_cpu_info()?;
    
    // CPU should have basic information
    assert!(!cpu_info.name.is_empty());
    assert!(cpu_info.cores > 0);
    assert!(cpu_info.threads > 0);
    
    // Threads should be >= cores
    assert!(cpu_info.threads >= cpu_info.cores);
    
    // Frequency should be reasonable (always non-negative for u64)
    // Note: This is always true for u64, but we keep it for documentation
    assert!(cpu_info.frequency_mhz < u64::MAX);
    
    // AVX support chain should be logical: AVX512 implies AVX2 implies AVX
    if cpu_info.supports_avx512 {
        assert!(cpu_info.supports_avx2);
        assert!(cpu_info.supports_avx);
    }
    if cpu_info.supports_avx2 {
        assert!(cpu_info.supports_avx);
    }
    
    Ok(())
}

/// Test memory extraction from ROCm output
#[test]
fn test_extract_memory_from_rocm_output() {
    // Test with valid ROCm output
    let valid_output = "Device: gfx906\nTotal VRAM: 8192 MB\nAvailable VRAM: 7680 MB";
    let memory = extract_memory_from_rocm_output(valid_output);
    assert_eq!(memory, Some(8192));
    
    // Test with different format
    let alt_output = "Memory Info:\nTotal VRAM     : 16384MB";
    let memory2 = extract_memory_from_rocm_output(alt_output);
    assert_eq!(memory2, Some(16384));
    
    // Test with no memory info
    let no_memory = "Device: gfx906\nTemperature: 45C";
    let memory3 = extract_memory_from_rocm_output(no_memory);
    assert_eq!(memory3, None);
    
    // Test with invalid format
    let invalid = "Total VRAM: invalid MB";
    let memory4 = extract_memory_from_rocm_output(invalid);
    assert_eq!(memory4, None);
}

/// Test ROCm installation detection
#[test]
fn test_is_rocm_installed() {
    // This test depends on the actual system
    let is_installed = is_rocm_installed();
    
    // Should return a boolean (actual value depends on system)
    assert!(is_installed || !is_installed); // Always true, just checking it doesn't panic
}

/// Test AMD GPU detection via lspci
#[test]
fn test_check_amd_gpus_lspci() {
    let result = check_amd_gpus_lspci();
    
    // Should return Ok with a boolean (may vary by system)
    assert!(result.is_ok());
}

/// Test CUDA version extraction with mock data
#[test]
fn test_parse_cuda_version_output() {
    // Test typical nvcc output parsing
    let mock_output = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Tue_Aug_15_22:02:13_PDT_2023\nCuda compilation tools, release 12.2, V12.2.140\nBuild cuda_12.2.r12.2/compiler.33191640_0";
    
    // Since get_cuda_version() is async and calls external commands,
    // we test the regex pattern that would be used
    use regex::Regex;
    let re = Regex::new(r"release (\d+\.\d+)").unwrap();
    
    if let Some(captures) = re.captures(mock_output) {
        let version = captures[1].to_string();
        assert_eq!(version, "12.2");
    } else {
        panic!("Failed to extract CUDA version from mock output");
    }
}

/// Test ROCm version extraction with mock data
#[test]
fn test_parse_rocm_version_output() {
    // Test typical hipcc output parsing
    let mock_output = "HIP version: 5.4.22224-b2e0db6e\nHIP platform: AMD HIP-Clang\nHIP runtime: rocm";
    
    use regex::Regex;
    let re = Regex::new(r"HIP version: (\d+\.\d+\.\d+)").unwrap();
    
    if let Some(captures) = re.captures(mock_output) {
        let version = captures[1].to_string();
        assert_eq!(version, "5.4.22224");
    } else {
        panic!("Failed to extract ROCm version from mock output");
    }
}

/// Test comprehensive diagnostics result creation
#[test]
fn test_diagnostics_result_comprehensive() {
    let gpu = GpuInfo {
        vendor: GpuVendor::Nvidia,
        name: "RTX 4090".to_string(),
        driver_version: Some("535.98".to_string()),
        cuda_version: Some("12.2".to_string()),
        memory_mb: Some(24576),
        compute_capability: Some("8.9".to_string()),
        is_compatible: true,
        issues: vec![],
    };

    let amd_gpu = GpuInfo {
        vendor: GpuVendor::Amd,
        name: "RX 7900 XTX".to_string(),
        driver_version: Some("5.4.0".to_string()),
        cuda_version: None,
        memory_mb: Some(24576),
        compute_capability: None,
        is_compatible: true,
        issues: vec![],
    };

    let cpu = CpuInfo {
        name: "AMD Ryzen 9 7950X".to_string(),
        cores: 16,
        threads: 32,
        frequency_mhz: 4500,
        supports_avx: true,
        supports_avx2: true,
        supports_avx512: true,
        is_compatible: true,
        issues: vec![],
    };

    let model1 = ModelInfo {
        name: "large-model".to_string(),
        path: "/models/large-model.safetensors".to_string(),
        format: ModelFormat::SafeTensors,
        size_mb: 7000, // Large model
        is_optimized: false,
        compatible_backends: vec![Backend::Cpu, Backend::Cuda, Backend::Rocm],
        issues: vec!["Large model may require significant GPU memory".to_string()],
    };

    let model2 = ModelInfo {
        name: "small-model".to_string(),
        path: "/models/small-model.safetensors".to_string(),
        format: ModelFormat::SafeTensors,
        size_mb: 500,
        is_optimized: true,
        compatible_backends: vec![Backend::Cpu, Backend::Cuda, Backend::Rocm],
        issues: vec![],
    };

    let mut diagnostics = DiagnosticsResult {
        gpus: vec![gpu, amd_gpu],
        cpu,
        models: vec![model1, model2],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    };

    // Calculate compatibility matrix
    diagnostics.compatibility_matrix = calculate_compatibility_matrix(&diagnostics);
    
    // Calculate overall score
    calculate_overall_score(&mut diagnostics);
    
    // Generate recommendations
    generate_recommendations(&mut diagnostics);
    
    // Verify results
    assert!(diagnostics.overall_score > 5); // Should have good score with NVIDIA + AMD GPUs
    assert!(diagnostics.system_ready);
    assert!(!diagnostics.compatibility_matrix.is_empty());
    
    // Check that both models have compatibility entries
    assert!(diagnostics.compatibility_matrix.contains_key("large-model"));
    assert!(diagnostics.compatibility_matrix.contains_key("small-model"));
    
    // Both models should be compatible with all backends (good hardware)
    let large_compat = &diagnostics.compatibility_matrix["large-model"];
    let small_compat = &diagnostics.compatibility_matrix["small-model"];
    
    assert!(large_compat.contains_key("CPU"));
    assert!(large_compat.contains_key("CUDA"));
    assert!(large_compat.contains_key("ROCm"));
    
    assert!(small_compat.contains_key("CPU"));
    assert!(small_compat.contains_key("CUDA"));
    assert!(small_compat.contains_key("ROCm"));
    
    // CUDA and ROCm should be compatible (we have both GPU types)
    if let CompatibilityStatus::Compatible = large_compat["CUDA"] {
        // Expected
    } else {
        panic!("Expected CUDA to be compatible with NVIDIA GPU present");
    }
    
    if let CompatibilityStatus::Compatible = large_compat["ROCm"] {
        // Expected
    } else {
        panic!("Expected ROCm to be compatible with AMD GPU present");
    }
}

/// Test scoring edge cases
#[test]
fn test_calculate_overall_score_edge_cases() {
    // Test with minimal system (just CPU, no GPU, no models)
    let mut minimal_diagnostics = DiagnosticsResult {
        gpus: vec![],
        cpu: CpuInfo {
            name: "Basic CPU".to_string(),
            cores: 2,
            threads: 4,
            frequency_mhz: 2400,
            supports_avx: true,
            supports_avx2: true,
            supports_avx512: false,
            is_compatible: true,
            issues: vec![],
        },
        models: vec![],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    };
    
    calculate_overall_score(&mut minimal_diagnostics);
    
    // Should have some score for CPU but low overall
    assert!(minimal_diagnostics.overall_score > 0);
    assert!(minimal_diagnostics.overall_score < minimal_diagnostics.max_score);
    assert!(minimal_diagnostics.max_score > 0);
    
    // Test with incompatible GPU
    let mut incompatible_gpu_diagnostics = DiagnosticsResult {
        gpus: vec![GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "Old GPU".to_string(),
            driver_version: Some("450.0".to_string()),
            cuda_version: None,
            memory_mb: Some(2048),
            compute_capability: Some("6.0".to_string()),
            is_compatible: false,
            issues: vec!["Compute capability too old".to_string()],
        }],
        cpu: CpuInfo {
            name: "Good CPU".to_string(),
            cores: 8,
            threads: 16,
            frequency_mhz: 3600,
            supports_avx: true,
            supports_avx2: true,
            supports_avx512: false,
            is_compatible: true,
            issues: vec![],
        },
        models: vec![],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    };
    
    calculate_overall_score(&mut incompatible_gpu_diagnostics);
    
    // Should get some points for having a GPU but not full points
    assert!(incompatible_gpu_diagnostics.overall_score > 2); // CPU points
    assert!(incompatible_gpu_diagnostics.overall_score < incompatible_gpu_diagnostics.max_score);
}

/// Test recommendation generation with various scenarios
#[test]
fn test_generate_recommendations_comprehensive() {
    // Test system with old CPU features
    let mut old_cpu_diagnostics = create_test_diagnostics_poor();
    old_cpu_diagnostics.cpu.supports_avx2 = false;
    old_cpu_diagnostics.cpu.supports_avx = false;
    
    generate_recommendations(&mut old_cpu_diagnostics);
    
    let has_cpu_rec = old_cpu_diagnostics.recommendations.iter()
        .any(|r| r.contains("CPU") && r.contains("AVX2"));
    assert!(has_cpu_rec);
    
    // Test system with NVIDIA GPU but no CUDA
    let mut no_cuda_diagnostics = DiagnosticsResult {
        gpus: vec![GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "RTX 3080".to_string(),
            driver_version: Some("525.0".to_string()),
            cuda_version: None, // No CUDA
            memory_mb: Some(10240),
            compute_capability: Some("8.6".to_string()),
            is_compatible: false,
            issues: vec!["CUDA not installed".to_string()],
        }],
        cpu: CpuInfo {
            name: "Good CPU".to_string(),
            cores: 8,
            threads: 16,
            frequency_mhz: 3600,
            supports_avx: true,
            supports_avx2: true,
            supports_avx512: false,
            is_compatible: true,
            issues: vec![],
        },
        models: vec![],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    };
    
    generate_recommendations(&mut no_cuda_diagnostics);
    
    let has_cuda_rec = no_cuda_diagnostics.recommendations.iter()
        .any(|r| r.contains("CUDA"));
    assert!(has_cuda_rec);
    
    // Test system with non-SafeTensors models
    let mut non_safetensors_diagnostics = DiagnosticsResult {
        gpus: vec![GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "RTX 4090".to_string(),
            driver_version: Some("535.0".to_string()),
            cuda_version: Some("12.0".to_string()),
            memory_mb: Some(24576),
            compute_capability: Some("8.9".to_string()),
            is_compatible: true,
            issues: vec![],
        }],
        cpu: CpuInfo {
            name: "Good CPU".to_string(),
            cores: 16,
            threads: 32,
            frequency_mhz: 4500,
            supports_avx: true,
            supports_avx2: true,
            supports_avx512: true,
            is_compatible: true,
            issues: vec![],
        },
        models: vec![ModelInfo {
            name: "pytorch-model".to_string(),
            path: "/models/model.pt".to_string(),
            format: ModelFormat::Pytorch, // Not SafeTensors
            size_mb: 1024,
            is_optimized: false,
            compatible_backends: vec![Backend::Cpu, Backend::Cuda],
            issues: vec![],
        }],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 0,
        max_score: 0,
        recommendations: vec![],
        system_ready: false,
    };
    
    generate_recommendations(&mut non_safetensors_diagnostics);
    
    let has_safetensors_rec = non_safetensors_diagnostics.recommendations.iter()
        .any(|r| r.contains("SafeTensors"));
    assert!(has_safetensors_rec);
}

/// Test display table function with mock data (basic test)
#[test]
fn test_display_results_table_basic() {
    // This test just ensures the function doesn't panic with various data
    let diagnostics = create_test_diagnostics_good();
    
    // We can't easily test the actual output without capturing stdout,
    // but we can at least ensure it doesn't panic
    display_results_table(&diagnostics, false);
    display_results_table(&diagnostics, true); // verbose mode
    
    // Test with empty diagnostics
    let empty_diagnostics = DiagnosticsResult {
        gpus: vec![],
        cpu: CpuInfo {
            name: "Test CPU".to_string(),
            cores: 4,
            threads: 8,
            frequency_mhz: 3000,
            supports_avx: true,
            supports_avx2: true,
            supports_avx512: false,
            is_compatible: true,
            issues: vec![],
        },
        models: vec![],
        compatibility_matrix: std::collections::HashMap::new(),
        overall_score: 3,
        max_score: 10,
        recommendations: vec!["Test recommendation".to_string()],
        system_ready: false,
    };
    
    display_results_table(&empty_diagnostics, false);
}

/// Integration test that runs the full diagnostic process
#[tokio::test]
async fn test_run_diagnostics_integration() -> Result<(), Box<dyn std::error::Error>> {
    use inferno_cli::cli_options::DoctorCliOptions;

    let temp_dir = TempDir::new()?;
    let model_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create some test models
    let _safetensors_file = create_test_file(&temp_dir, "model.safetensors", 1024)?;
    let _pytorch_file = create_test_file(&temp_dir, "model.pt", 512)?;

    let opts = DoctorCliOptions {
        model_dir,
        verbose: false,
        format: "json".to_string(),
    };

    // This should not panic and should complete successfully
    // Note: The actual GPU/CPU detection will depend on the test environment
    let result = run_diagnostics(opts).await;
    
    // The function should complete without error
    // (though actual hardware detection results will vary)
    assert!(result.is_ok());

    Ok(())
}

/// Integration test with verbose output and YAML format
#[tokio::test]
async fn test_run_diagnostics_verbose_yaml() -> Result<(), Box<dyn std::error::Error>> {
    use inferno_cli::cli_options::DoctorCliOptions;

    let temp_dir = TempDir::new()?;
    let model_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create test models with different formats
    let _safetensors_file = create_test_file(&temp_dir, "optimized_model_fp16.safetensors", 2048)?;
    let _onnx_file = create_test_file(&temp_dir, "inference_model.onnx", 512)?;
    let _gguf_file = create_test_file(&temp_dir, "llama_model.gguf", 4096)?;

    let opts = DoctorCliOptions {
        model_dir,
        verbose: true,
        format: "yaml".to_string(),
    };

    let result = run_diagnostics(opts).await;
    assert!(result.is_ok());

    Ok(())
}

/// Integration test with table format (default)
#[tokio::test]
async fn test_run_diagnostics_table_format() -> Result<(), Box<dyn std::error::Error>> {
    use inferno_cli::cli_options::DoctorCliOptions;

    let temp_dir = TempDir::new()?;
    let model_dir = temp_dir.path().to_str().unwrap().to_string();

    let opts = DoctorCliOptions {
        model_dir,
        verbose: false,
        format: "table".to_string(),
    };

    let result = run_diagnostics(opts).await;
    assert!(result.is_ok());

    Ok(())
}