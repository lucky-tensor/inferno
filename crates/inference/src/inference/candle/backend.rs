//! Candle backend types and device management

use crate::inference::InferenceError;
use serde::{Deserialize, Serialize};

#[cfg(any(
))]
use candle_core::Device;

/// Candle backend types for different hardware acceleration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandleBackendType {
    /// CPU backend using optimized CPU kernels
    Cpu,
    /// CUDA backend with GPU acceleration and custom kernels
    Cuda,
    /// Metal backend for Apple Silicon acceleration
    Metal,
}

impl std::fmt::Display for CandleBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "Candle-CPU"),
            Self::Cuda => write!(f, "Candle-CUDA"),
            Self::Metal => write!(f, "Candle-Metal"),
        }
    }
}

impl CandleBackendType {
    /// Create device for the specified backend type
    #[cfg(any(
    ))]
    pub fn create_device(&self) -> Result<Device, InferenceError> {
        match self {
            Self::Cpu => {
                tracing::info!("Initializing CPU device for Candle inference");
                Ok(Device::Cpu)
            }
            Self::Cuda => {
                tracing::info!("Initializing CUDA device for Candle inference");
                match Device::new_cuda(0) {
                    Ok(device) => {
                        tracing::info!("Created CUDA device successfully");
                        Ok(device)
                    }
                    Err(e) => {
                        tracing::error!("Failed to create CUDA device: {}", e);
                        Err(InferenceError::InitializationError(format!(
                            "CUDA device initialization failed: {}",
                            e
                        )))
                    }
                }
            }
            Self::Metal => {
                tracing::info!("Initializing Metal device for Candle inference");
                match Device::new_metal(0) {
                    Ok(device) => {
                        tracing::info!("Created Metal device successfully");
                        Ok(device)
                    }
                    Err(e) => {
                        tracing::error!("Failed to create Metal device: {}", e);
                        Err(InferenceError::InitializationError(format!(
                            "Metal device initialization failed: {}",
                            e
                        )))
                    }
                }
            }
        }
    }
}
