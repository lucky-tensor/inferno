//! Candle backend types and device management

use crate::inference::InferenceError;
use serde::{Deserialize, Serialize};

#[cfg(any(
    feature = "candle-cpu",
    feature = "candle-cuda",
    feature = "candle-metal"
))]
use candle_core::Device;

/// Candle backend types for different hardware acceleration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandleBackendType {
    /// CPU backend using optimized CPU kernels
    Cpu,
    /// CUDA backend with GPU acceleration and custom kernels
    #[cfg(feature = "candle-cuda")]
    Cuda,
    /// Metal backend for Apple Silicon acceleration
    #[cfg(feature = "candle-metal")]
    Metal,
}

impl std::fmt::Display for CandleBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "Candle-CPU"),
            #[cfg(feature = "candle-cuda")]
            Self::Cuda => write!(f, "Candle-CUDA"),
            #[cfg(feature = "candle-metal")]
            Self::Metal => write!(f, "Candle-Metal"),
        }
    }
}

impl CandleBackendType {
    /// Create device for the specified backend type
    #[cfg(any(
        feature = "candle-cpu",
        feature = "candle-cuda",
        feature = "candle-metal"
    ))]
    pub fn create_device(&self) -> Result<Device, InferenceError> {
        match self {
            Self::Cpu => {
                tracing::info!("Initializing CPU device for Candle inference");
                Ok(Device::Cpu)
            }
            #[cfg(feature = "candle-cuda")]
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
            #[cfg(feature = "candle-metal")]
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
