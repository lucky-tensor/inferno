//! Test-Driven Development tests for quantization metadata types and operations
//!
//! These tests define the expected behavior for detecting and handling
//! different quantization schemes (w8a8, compressed-tensors, etc.)

use candle_core::DType;
use inferno_llama::diagnostic::{
    LayerQuantizationConfig, QuantizationConfig as DiagnosticQuantizationConfig,
    QuantizationParams, QuantizationScheme, WeightAnalyzer,
};
use std::collections::HashMap;

/// Test W8A8 quantization scheme detection and properties
#[test]
fn test_w8a8_quantization_scheme() {
    let scheme = QuantizationScheme::W8A8;

    assert!(scheme.is_supported(), "W8A8 should be supported");
    assert_eq!(
        scheme.memory_reduction_factor(),
        2.0,
        "W8A8 should provide ~2x memory reduction"
    );

    // Test serialization/deserialization
    let json = serde_json::to_string(&scheme).unwrap();
    let deserialized: QuantizationScheme = serde_json::from_str(&json).unwrap();
    assert_eq!(scheme, deserialized);
}

/// Test W4A16 quantization scheme detection and properties
#[test]
fn test_w4a16_quantization_scheme() {
    let scheme = QuantizationScheme::W4A16;

    // W4A16 is not yet implemented, so should not be supported
    assert!(!scheme.is_supported(), "W4A16 should not be supported yet");
    assert_eq!(
        scheme.memory_reduction_factor(),
        4.0,
        "W4A16 should provide ~4x memory reduction"
    );
}

/// Test compressed tensors quantization scheme
#[test]
fn test_compressed_tensors_scheme() {
    let scheme = QuantizationScheme::CompressedTensors("gptq".to_string());

    assert!(
        !scheme.is_supported(),
        "Compressed tensors should not be supported yet"
    );
    assert_eq!(
        scheme.memory_reduction_factor(),
        2.0,
        "Should provide conservative 2x estimate"
    );

    // Test with different compressed tensor formats
    let schemes = vec![
        QuantizationScheme::CompressedTensors("gptq".to_string()),
        QuantizationScheme::CompressedTensors("awq".to_string()),
        QuantizationScheme::CompressedTensors("squant".to_string()),
    ];

    for scheme in schemes {
        assert!(
            !scheme.is_supported(),
            "Compressed tensor schemes should not be supported yet"
        );
    }
}

/// Test quantization config creation and validation
#[test]
fn test_quantization_config_creation() {
    // Test default (no quantization)
    let default_config = DiagnosticQuantizationConfig::default();
    assert_eq!(default_config.scheme, QuantizationScheme::None);
    assert!(default_config.symmetric);
    assert!(default_config.per_layer_config.is_none());

    // Test W8A8 config
    let w8a8_config = DiagnosticQuantizationConfig {
        scheme: QuantizationScheme::W8A8,
        symmetric: true,
        zero_point: Some(128),
        scale: Some(0.1),
        global_params: Some(QuantizationParams {
            bits: 8,
            group_size: None,
            block_size: None,
            custom_params: None,
        }),
        ..Default::default()
    };

    assert_eq!(w8a8_config.scheme, QuantizationScheme::W8A8);
    assert_eq!(w8a8_config.zero_point, Some(128));
    assert_eq!(w8a8_config.scale, Some(0.1));
}

/// Test per-layer quantization configuration
#[test]
fn test_per_layer_quantization_config() {
    let mut per_layer_config = HashMap::new();

    // Different layers might use different quantization schemes
    per_layer_config.insert(
        "layers.0.self_attn.q_proj".to_string(),
        LayerQuantizationConfig {
            dtype: DType::U8,
            scheme: QuantizationScheme::W8A8,
            params: Some(QuantizationParams {
                bits: 8,
                group_size: Some(128),
                block_size: None,
                custom_params: None,
            }),
        },
    );

    per_layer_config.insert(
        "layers.0.mlp.gate_proj".to_string(),
        LayerQuantizationConfig {
            dtype: DType::U8,
            scheme: QuantizationScheme::W8A8,
            params: Some(QuantizationParams {
                bits: 8,
                group_size: Some(64),
                block_size: None,
                custom_params: None,
            }),
        },
    );

    let config = DiagnosticQuantizationConfig {
        scheme: QuantizationScheme::W8A8,
        per_layer_config: Some(per_layer_config.clone()),
        symmetric: true,
        ..Default::default()
    };

    assert!(config.per_layer_config.is_some());
    assert_eq!(config.per_layer_config.as_ref().unwrap().len(), 2);

    // Test layer-specific configuration retrieval
    let q_proj_config = config
        .per_layer_config
        .as_ref()
        .unwrap()
        .get("layers.0.self_attn.q_proj")
        .unwrap();
    assert_eq!(q_proj_config.dtype, DType::U8);
    assert_eq!(q_proj_config.scheme, QuantizationScheme::W8A8);
    assert_eq!(q_proj_config.params.as_ref().unwrap().group_size, Some(128));
}

/// Test quantization detection from model path patterns
#[tokio::test]
async fn test_quantization_detection_from_path() {
    // Test W8A8 detection
    let w8a8_paths = vec![
        "/models/model-w8a8",
        "/models/model-quantized.w8a8",
        "/models/model-int8",
        "/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8",
    ];

    for path in w8a8_paths {
        let result = WeightAnalyzer::detect_quantization(path).await;
        assert!(
            result.is_ok(),
            "Should detect quantization for path: {}",
            path
        );

        let config = result.unwrap();
        assert_eq!(
            config.scheme,
            QuantizationScheme::W8A8,
            "Should detect W8A8 for path: {}",
            path
        );
    }

    // Test W4A16 detection
    let w4a16_paths = vec![
        "/models/model-w4a16",
        "/models/model-int4",
        "/models/model-4bit",
    ];

    for path in w4a16_paths {
        let result = WeightAnalyzer::detect_quantization(path).await;
        assert!(
            result.is_ok(),
            "Should detect quantization for path: {}",
            path
        );

        let config = result.unwrap();
        assert_eq!(
            config.scheme,
            QuantizationScheme::W4A16,
            "Should detect W4A16 for path: {}",
            path
        );
    }

    // Test no quantization
    let unquantized_paths = vec![
        "/models/standard-model",
        "/models/meta-llama_Llama-3.1-8B-Instruct",
        "/models/tinyllama-1.1b",
    ];

    for path in unquantized_paths {
        let result = WeightAnalyzer::detect_quantization(path).await;
        assert!(
            result.is_ok(),
            "Should not error for unquantized path: {}",
            path
        );

        let config = result.unwrap();
        assert_eq!(
            config.scheme,
            QuantizationScheme::None,
            "Should detect no quantization for path: {}",
            path
        );
    }
}

/// Test quantization parameters validation
#[test]
fn test_quantization_params_validation() {
    // Valid 8-bit parameters
    let valid_8bit = QuantizationParams {
        bits: 8,
        group_size: Some(128),
        block_size: None,
        custom_params: None,
    };

    assert_eq!(valid_8bit.bits, 8);
    assert_eq!(valid_8bit.group_size, Some(128));

    // Valid 4-bit parameters
    let valid_4bit = QuantizationParams {
        bits: 4,
        group_size: Some(64),
        block_size: Some(256),
        custom_params: None,
    };

    assert_eq!(valid_4bit.bits, 4);
    assert_eq!(valid_4bit.group_size, Some(64));
    assert_eq!(valid_4bit.block_size, Some(256));

    // Parameters with custom fields
    let mut custom_params = HashMap::new();
    custom_params.insert(
        "calibration_dataset".to_string(),
        serde_json::Value::String("c4".to_string()),
    );
    custom_params.insert(
        "percdamp".to_string(),
        serde_json::Value::Number(serde_json::Number::from_f64(0.01).unwrap()),
    );

    let custom = QuantizationParams {
        bits: 4,
        group_size: Some(128),
        block_size: None,
        custom_params: Some(custom_params.clone()),
    };

    assert!(custom.custom_params.is_some());
    assert_eq!(custom.custom_params.as_ref().unwrap().len(), 2);
}

/// Test quantization configuration validation
#[test]
fn test_quantization_config_validation() {
    let config = DiagnosticQuantizationConfig {
        scheme: QuantizationScheme::W8A8,
        symmetric: false,
        zero_point: Some(127),
        scale: Some(0.05),
        global_params: Some(QuantizationParams {
            bits: 8,
            group_size: Some(256),
            block_size: None,
            custom_params: None,
        }),
        per_layer_config: None,
    };

    // Test configuration properties
    assert_eq!(config.scheme, QuantizationScheme::W8A8);
    assert!(!config.symmetric);
    assert_eq!(config.zero_point, Some(127));
    assert_eq!(config.scale, Some(0.05));
    assert!(config.global_params.is_some());
}

/// Test quantization scheme compatibility checks
#[test]
fn test_quantization_compatibility() {
    // Test supported schemes
    assert!(QuantizationScheme::None.is_supported());
    assert!(QuantizationScheme::W8A8.is_supported());

    // Test unsupported schemes
    assert!(!QuantizationScheme::W4A16.is_supported());
    assert!(!QuantizationScheme::CompressedTensors("gptq".to_string()).is_supported());
    assert!(!QuantizationScheme::Custom("proprietary".to_string()).is_supported());
}

/// Test memory reduction calculations
#[test]
fn test_memory_reduction_calculations() {
    let test_cases = vec![
        (QuantizationScheme::None, 1.0),
        (QuantizationScheme::W8A8, 2.0),
        (QuantizationScheme::W4A16, 4.0),
        (
            QuantizationScheme::CompressedTensors("gptq".to_string()),
            2.0,
        ),
        (QuantizationScheme::Custom("unknown".to_string()), 1.0),
    ];

    for (scheme, expected_factor) in test_cases {
        assert_eq!(
            scheme.memory_reduction_factor(),
            expected_factor,
            "Memory reduction factor mismatch for scheme: {:?}",
            scheme
        );
    }
}

/// Test edge cases and error conditions
#[test]
fn test_quantization_edge_cases() {
    // Test empty custom parameters
    let empty_custom = QuantizationParams {
        bits: 8,
        group_size: None,
        block_size: None,
        custom_params: Some(HashMap::new()),
    };

    assert!(empty_custom.custom_params.is_some());
    assert_eq!(empty_custom.custom_params.as_ref().unwrap().len(), 0);

    // Test very large group sizes
    let large_group = QuantizationParams {
        bits: 4,
        group_size: Some(8192),
        block_size: Some(16384),
        custom_params: None,
    };

    assert_eq!(large_group.group_size, Some(8192));
    assert_eq!(large_group.block_size, Some(16384));
}

/// Test real model quantization detection
#[tokio::test]
async fn test_real_model_quantization_detection() {
    // This should detect W8A8 quantization
    let quantized_model = "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8";

    let result = WeightAnalyzer::detect_quantization(quantized_model).await;
    assert!(
        result.is_ok(),
        "Should successfully analyze quantized model"
    );

    let config = result.unwrap();
    assert_eq!(config.scheme, QuantizationScheme::W8A8);
    assert!(config.symmetric); // Should use symmetric quantization by default

    // This should detect no quantization
    let unquantized_model = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";

    let result = WeightAnalyzer::detect_quantization(unquantized_model).await;
    assert!(
        result.is_ok(),
        "Should successfully analyze unquantized model"
    );

    let config = result.unwrap();
    assert_eq!(config.scheme, QuantizationScheme::None);
}
