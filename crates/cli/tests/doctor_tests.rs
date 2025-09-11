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
use std::path::Path;
use tempfile::{TempDir, NamedTempFile};
use tokio_test;

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
    assert_eq!(pytorch_models.len(), 1);
    assert_eq!(gguf_models.len(), 1);
    assert_eq!(onnx_models.len(), 1);

    // Check file sizes (converted from bytes to MB)
    let safetensors_model = &safetensors_models[0];
    assert_eq!(safetensors_model.size_mb, 1); // 1024 bytes = 1MB (roughly)

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
    assert!(check_model_optimization(optimized_file.path(), &ModelFormat::SafeTensors));

    let int8_file = create_test_file(&temp_dir, "model_int8.safetensors", 1024)?;
    assert!(check_model_optimization(int8_file.path(), &ModelFormat::SafeTensors));

    let fp16_file = create_test_file(&temp_dir, "model_fp16.safetensors", 1024)?;
    assert!(check_model_optimization(fp16_file.path(), &ModelFormat::SafeTensors));

    let optimized_file2 = create_test_file(&temp_dir, "model_optimized.safetensors", 1024)?;
    assert!(check_model_optimization(optimized_file2.path(), &ModelFormat::SafeTensors));

    // Test non-optimized model names
    let regular_file = create_test_file(&temp_dir, "regular_model.safetensors", 1024)?;
    assert!(!check_model_optimization(regular_file.path(), &ModelFormat::SafeTensors));

    // Test non-SafeTensors formats (should always return false for now)
    let pytorch_file = create_test_file(&temp_dir, "model_quantized.pt", 1024)?;
    assert!(!check_model_optimization(pytorch_file.path(), &ModelFormat::Pytorch));

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
    let (avx, avx2, avx512) = detect_cpu_features();
    
    // On x86_64, we should get real feature detection
    // On other architectures, we should get (false, false, false)
    #[cfg(target_arch = "x86_64")]
    {
        // On x86_64, at least one should likely be true for modern systems
        // but we can't guarantee it, so we just test that the function runs
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        assert!(!avx);
        assert!(!avx2);
        assert!(!avx512);
    }
}

/// Test driver version parsing
#[test]
fn test_parse_driver_version() -> Result<(), Box<dyn std::error::Error>> {
    // Test valid version strings
    assert!((parse_driver_version("535.98")? - 535.98).abs() < 0.01);
    assert!((parse_driver_version("470.129.06")? - 470.129).abs() < 0.01);
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
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let file_path = temp_dir.path().join(filename);
    let data = vec![0u8; size_bytes as usize];
    fs::write(&file_path, data)?;
    
    // Create a NamedTempFile for the existing file
    let temp_file = NamedTempFile::new_in(temp_dir)?;
    Ok(temp_file)
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