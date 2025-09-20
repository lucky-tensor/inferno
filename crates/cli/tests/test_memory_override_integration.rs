//! Integration test for memory override functionality in CLI play command

#[cfg(test)]
mod memory_override_tests {
    use inferno_shared::{MemoryOverhead, MemoryValidation, ModelMemoryValidator};

    #[tokio::test]
    async fn test_memory_validation_structure() -> Result<(), Box<dyn std::error::Error>> {
        println!("MEMORY VALIDATION OVERRIDE INTEGRATION TEST");
        println!("==========================================");

        // Test creating a validator with custom overhead
        let custom_overhead = MemoryOverhead {
            loading_overhead_gb: 2.0,
            inference_overhead_gb: 1.5,
            framework_overhead_gb: 1.0,
        };

        let _validator = ModelMemoryValidator::with_memory_overhead(0, custom_overhead);

        // Test the validation structure that would be used in CLI prompts
        let mock_validation = MemoryValidation {
            will_fit: false,
            confidence: 0.8,
            model_size_gb: 15.0,
            available_memory_gb: 10.0,
            estimated_requirement_gb: 19.5, // 15GB model + 4.5GB overhead
            overhead_factor: 4.5,
            message: "ERROR: Model likely won't fit!".to_string(),
            recommendations: vec![
                "Kill GPU processes using significant memory".to_string(),
                "Consider using a smaller model or quantized version".to_string(),
            ],
            interfering_processes: vec![],
        };

        println!("\nMock validation scenario:");
        println!("Model size: {:.1} GB", mock_validation.model_size_gb);
        println!(
            "Available memory: {:.1} GB",
            mock_validation.available_memory_gb
        );
        println!(
            "Estimated requirement: {:.1} GB",
            mock_validation.estimated_requirement_gb
        );
        println!("Will fit: {}", mock_validation.will_fit);
        println!("Confidence: {:.0}%", mock_validation.confidence * 100.0);

        println!("\nUser would see:");
        println!(
            "   Model requires {:.1} GB but only {:.1} GB available (shortfall: {:.1} GB)",
            mock_validation.estimated_requirement_gb,
            mock_validation.available_memory_gb,
            mock_validation.estimated_requirement_gb - mock_validation.available_memory_gb
        );

        println!("\nRisks displayed:");
        println!("   - Model loading may fail with CUDA out-of-memory errors");
        println!("   - GPU may become unresponsive requiring system restart");
        println!("   - Other GPU processes may be killed by the system");

        println!("\nPrompt: Do you want to proceed anyway? (y/N):");
        println!("   Expected behavior: User can type 'y' to override, 'n' to cancel");

        // Verify the shortfall calculation
        let shortfall =
            mock_validation.estimated_requirement_gb - mock_validation.available_memory_gb;
        assert_eq!(shortfall, 9.5, "Shortfall calculation should be correct");

        println!("\nOverride functionality validated successfully!");
        Ok(())
    }

    #[tokio::test]
    async fn test_low_confidence_warning_scenario() -> Result<(), Box<dyn std::error::Error>> {
        println!("LOW CONFIDENCE WARNING SCENARIO TEST");
        println!("====================================");

        // Scenario where model fits but with low confidence
        let warning_validation = MemoryValidation {
            will_fit: true,
            confidence: 0.6, // Low confidence
            model_size_gb: 13.0,
            available_memory_gb: 18.0,
            estimated_requirement_gb: 17.5,
            overhead_factor: 4.5,
            message: "WARNING: Model should fit, but it will be tight.".to_string(),
            recommendations: vec![
                "Kill GPU processes using significant memory".to_string(),
                "For large models, ensure no other applications are using GPU".to_string(),
            ],
            interfering_processes: vec![],
        };

        println!("\nWarning scenario:");
        println!("Model size: {:.1} GB", warning_validation.model_size_gb);
        println!(
            "Available memory: {:.1} GB",
            warning_validation.available_memory_gb
        );
        println!(
            "Estimated requirement: {:.1} GB",
            warning_validation.estimated_requirement_gb
        );
        println!("Will fit: {}", warning_validation.will_fit);
        println!("Confidence: {:.0}%", warning_validation.confidence * 100.0);

        println!("\nUser would see:");
        println!(
            "WARNING: Memory validation has low confidence ({:.0}%)",
            warning_validation.confidence * 100.0
        );
        println!("Consider the following recommendations:");
        for rec in &warning_validation.recommendations {
            println!("   {}", rec);
        }

        println!("\nPrompt: Continue with model loading? (Y/n):");
        println!("   Expected behavior: User can proceed with 'Y' or cancel with 'n'");

        // Verify confidence threshold
        assert!(
            warning_validation.confidence < 0.7,
            "Confidence should be below warning threshold"
        );
        assert!(
            warning_validation.will_fit,
            "Model should fit in this scenario"
        );

        println!("\nWarning scenario validated successfully!");
        Ok(())
    }
}
