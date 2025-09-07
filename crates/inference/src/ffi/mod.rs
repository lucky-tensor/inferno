//! Foreign Function Interface (FFI) bindings for VLLM C++/CUDA backend
//!
//! This module provides safe Rust wrappers around the C++/CUDA implementation
//! of VLLM inference engine. It handles memory safety, error conversion, and
//! provides idiomatic Rust interfaces.

#![allow(dead_code, missing_docs, clippy::all)]

#[cfg(feature = "cuda")]
pub mod bindings;
#[cfg(feature = "cuda")]
pub mod safety;
#[cfg(feature = "cuda")]
pub mod types;

#[cfg(feature = "cuda")]
pub use bindings::*;
#[cfg(feature = "cuda")]
pub use safety::*;
#[cfg(feature = "cuda")]
pub use types::*;

use crate::error::{VLLMError, VLLMResult};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Main VLLM wrapper that provides safe access to the C++ implementation
#[cfg(feature = "cuda")]
pub struct VLLMWrapper {
    handle: VLLMEngineHandle,
    config: VLLMConfig,
    is_initialized: bool,
}

#[cfg(feature = "cuda")]
impl VLLMWrapper {
    /// Create a new VLLM wrapper with the given configuration
    pub fn new(config: crate::config::VLLMConfig) -> VLLMResult<Self> {
        let c_config = VLLMConfig::from_rust_config(&config)?;
        let mut handle = std::ptr::null_mut();

        let result = unsafe { vllm_create_engine(&c_config, &mut handle) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::InitializationFailed(get_error_string(result)));
        }

        Ok(Self {
            handle,
            config: c_config,
            is_initialized: true,
        })
    }

    /// Load a model from the specified path
    pub fn load_model(&mut self, model_path: &str) -> VLLMResult<()> {
        if !self.is_initialized {
            return Err(VLLMError::EngineNotInitialized);
        }

        let c_path = CString::new(model_path)
            .map_err(|_| VLLMError::InvalidArgument("Invalid model path".to_string()))?;

        let result = unsafe { vllm_load_model(self.handle, c_path.as_ptr()) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::ModelLoadFailed(format!(
                "Failed to load model from {}: {}",
                model_path,
                get_error_string(result)
            )));
        }

        Ok(())
    }

    /// Submit an inference request
    pub fn submit_request(&self, request: &InferenceRequest) -> VLLMResult<RequestHandle> {
        if !self.is_initialized {
            return Err(VLLMError::EngineNotInitialized);
        }

        let c_request = VLLMInferenceRequest::from_rust_request(request)?;
        let mut request_handle = std::ptr::null_mut();

        let result = unsafe { vllm_submit_request(self.handle, &c_request, &mut request_handle) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::InferenceFailed(get_error_string(result)));
        }

        Ok(RequestHandle::new(request_handle))
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> VLLMResult<MemoryStats> {
        if !self.is_initialized {
            return Err(VLLMError::EngineNotInitialized);
        }

        let mut stats = VLLMMemoryStats::default();
        let result = unsafe { vllm_get_memory_stats(self.handle, &mut stats) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::OperationFailed(format!(
                "Failed to get memory stats: {}",
                get_error_string(result)
            )));
        }

        Ok(MemoryStats::from_c_stats(&stats))
    }

    /// Perform health check
    pub fn health_check(&self) -> VLLMResult<()> {
        if !self.is_initialized {
            return Err(VLLMError::EngineNotInitialized);
        }

        let result = unsafe { vllm_health_check(self.handle) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::HealthCheckFailed(get_error_string(result)));
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for VLLMWrapper {
    fn drop(&mut self) {
        if self.is_initialized && !self.handle.is_null() {
            unsafe {
                let _ = vllm_destroy_engine(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for VLLMWrapper {}
#[cfg(feature = "cuda")]
unsafe impl Sync for VLLMWrapper {}

/// Safe wrapper for VLLM request handles
#[cfg(feature = "cuda")]
pub struct RequestHandle {
    handle: VLLMRequestHandle,
}

#[cfg(feature = "cuda")]
impl RequestHandle {
    fn new(handle: VLLMRequestHandle) -> Self {
        Self { handle }
    }

    /// Get the response for this request
    pub fn get_response(&self) -> VLLMResult<InferenceResponse> {
        let mut response = VLLMInferenceResponse::default();
        let result = unsafe { vllm_get_response(self.handle, &mut response) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::InferenceFailed(get_error_string(result)));
        }

        InferenceResponse::from_c_response(&response)
    }

    /// Cancel this request
    pub fn cancel(&self) -> VLLMResult<()> {
        let result = unsafe { vllm_cancel_request(self.handle) };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::OperationFailed(format!(
                "Failed to cancel request: {}",
                get_error_string(result)
            )));
        }

        Ok(())
    }
}

/// Handle for CUDA operations
#[cfg(feature = "cuda")]
pub struct CudaHandle {
    device_id: i32,
    is_initialized: bool,
}

#[cfg(feature = "cuda")]
impl CudaHandle {
    pub fn new(device_id: i32) -> VLLMResult<Self> {
        // Validate device exists
        let mut device_count = 0;
        let result = unsafe { vllm_get_cuda_device_count(&mut device_count) };

        if result != VLLMErrorCode::VLLM_SUCCESS || device_id >= device_count {
            return Err(VLLMError::InvalidArgument(format!(
                "Invalid device ID: {}",
                device_id
            )));
        }

        Ok(Self {
            device_id,
            is_initialized: true,
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn get_device_info(&self) -> VLLMResult<String> {
        let mut buffer = vec![0u8; 1024];
        let result = unsafe {
            vllm_get_cuda_device_info(
                self.device_id,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len(),
            )
        };

        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::OperationFailed(format!(
                "Failed to get device info: {}",
                get_error_string(result)
            )));
        }

        let cstr = unsafe { CStr::from_ptr(buffer.as_ptr() as *const c_char) };
        Ok(cstr.to_string_lossy().to_string())
    }
}

/// Main VLLM handle that combines engine and CUDA operations
#[cfg(feature = "cuda")]
pub struct VLLMHandle {
    wrapper: VLLMWrapper,
    cuda_handle: CudaHandle,
}

#[cfg(feature = "cuda")]
impl VLLMHandle {
    pub fn new(config: crate::config::VLLMConfig) -> VLLMResult<Self> {
        let cuda_handle = CudaHandle::new(config.device_id)?;
        let wrapper = VLLMWrapper::new(config)?;

        Ok(Self {
            wrapper,
            cuda_handle,
        })
    }

    pub fn wrapper(&self) -> &VLLMWrapper {
        &self.wrapper
    }

    pub fn wrapper_mut(&mut self) -> &mut VLLMWrapper {
        &mut self.wrapper
    }

    pub fn cuda_handle(&self) -> &CudaHandle {
        &self.cuda_handle
    }
}

/// Convert C error code to human-readable string
#[cfg(feature = "cuda")]
fn get_error_string(error_code: VLLMErrorCode) -> String {
    let c_str = unsafe { vllm_get_error_string(error_code) };

    if c_str.is_null() {
        return format!("Unknown error code: {:?}", error_code);
    }

    unsafe { CStr::from_ptr(c_str).to_string_lossy().to_string() }
}

// CPU-only fallback implementations
#[cfg(not(feature = "cuda"))]
pub struct VLLMWrapper;

#[cfg(not(feature = "cuda"))]
impl VLLMWrapper {
    pub fn new(_config: crate::config::VLLMConfig) -> VLLMResult<Self> {
        Err(VLLMError::CudaNotAvailable)
    }
}

#[cfg(not(feature = "cuda"))]
pub struct CudaHandle;

#[cfg(not(feature = "cuda"))]
pub struct VLLMHandle;

#[cfg(not(feature = "cuda"))]
impl VLLMHandle {
    pub fn new(_config: crate::config::VLLMConfig) -> VLLMResult<Self> {
        Err(VLLMError::CudaNotAvailable)
    }
}
