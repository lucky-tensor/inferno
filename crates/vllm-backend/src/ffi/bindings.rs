//! FFI bindings to the C++/CUDA VLLM implementation
//!
//! This module contains the raw FFI declarations that correspond to the
//! C++ header files. In a real implementation, these would be generated
//! by bindgen during the build process.

#![allow(
    non_camel_case_types,
    missing_docs,
    clippy::wildcard_imports,
    non_upper_case_globals
)]

use super::types::*;
use std::os::raw::{c_char, c_int};

#[cfg(feature = "cuda")]
extern "C" {
    // Engine lifecycle functions
    pub fn vllm_create_engine(
        config: *const VLLMConfig,
        engine: *mut VLLMEngineHandle,
    ) -> VLLMErrorCode;

    pub fn vllm_destroy_engine(engine: VLLMEngineHandle) -> VLLMErrorCode;

    pub fn vllm_load_model(engine: VLLMEngineHandle, model_path: *const c_char) -> VLLMErrorCode;

    pub fn vllm_unload_model(engine: VLLMEngineHandle) -> VLLMErrorCode;

    // Inference functions
    pub fn vllm_submit_request(
        engine: VLLMEngineHandle,
        request: *const VLLMInferenceRequest,
        request_handle: *mut VLLMRequestHandle,
    ) -> VLLMErrorCode;

    pub fn vllm_get_response(
        request_handle: VLLMRequestHandle,
        response: *mut VLLMInferenceResponse,
    ) -> VLLMErrorCode;

    pub fn vllm_cancel_request(request_handle: VLLMRequestHandle) -> VLLMErrorCode;

    pub fn vllm_free_response(response: *mut VLLMInferenceResponse) -> VLLMErrorCode;

    // Batch processing functions
    pub fn vllm_submit_batch(
        engine: VLLMEngineHandle,
        requests: *const VLLMInferenceRequest,
        num_requests: usize,
        request_handles: *mut VLLMRequestHandle,
    ) -> VLLMErrorCode;

    pub fn vllm_get_batch_responses(
        request_handles: *const VLLMRequestHandle,
        num_requests: usize,
        responses: *mut VLLMInferenceResponse,
    ) -> VLLMErrorCode;

    // Memory management functions
    pub fn vllm_get_memory_stats(
        engine: VLLMEngineHandle,
        stats: *mut VLLMMemoryStats,
    ) -> VLLMErrorCode;

    pub fn vllm_clear_cache(engine: VLLMEngineHandle) -> VLLMErrorCode;

    pub fn vllm_set_memory_pool_size(engine: VLLMEngineHandle, size_mb: usize) -> VLLMErrorCode;

    // Health and monitoring functions
    pub fn vllm_health_check(engine: VLLMEngineHandle) -> VLLMErrorCode;

    pub fn vllm_get_engine_info(
        engine: VLLMEngineHandle,
        info_buffer: *mut c_char,
        buffer_size: usize,
    ) -> VLLMErrorCode;

    // Utility functions
    pub fn vllm_get_error_string(error_code: VLLMErrorCode) -> *const c_char;

    pub fn vllm_set_log_level(level: c_int) -> VLLMErrorCode;

    pub fn vllm_get_cuda_device_count(device_count: *mut c_int) -> VLLMErrorCode;

    pub fn vllm_get_cuda_device_info(
        device_id: c_int,
        info_buffer: *mut c_char,
        buffer_size: usize,
    ) -> VLLMErrorCode;

    // Stream processing functions
    pub fn vllm_set_stream_callback(
        engine: VLLMEngineHandle,
        callback: VLLMStreamCallback,
        user_data: *mut std::ffi::c_void,
    ) -> VLLMErrorCode;

    pub fn vllm_enable_streaming(engine: VLLMEngineHandle, enable: bool) -> VLLMErrorCode;
}

/// Callback function type for streaming responses
#[cfg(feature = "cuda")]
pub type VLLMStreamCallback = Option<
    unsafe extern "C" fn(
        request_id: u64,
        token: *const c_char,
        is_finished: bool,
        user_data: *mut std::ffi::c_void,
    ),
>;

// CUDA kernel bindings
#[cfg(feature = "cuda")]
extern "C" {
    // Memory operations
    pub fn cuda_memcpy_async(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cuda_memset_async(
        ptr: *mut std::ffi::c_void,
        value: c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cuda_memory_prefetch(
        ptr: *mut std::ffi::c_void,
        count: usize,
        device_id: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // Tensor operations
    pub fn cuda_tensor_copy(
        dst: *mut f32,
        src: *const f32,
        num_elements: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cuda_tensor_add(
        result: *mut f32,
        a: *const f32,
        b: *const f32,
        num_elements: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cuda_tensor_multiply(
        result: *mut f32,
        a: *const f32,
        b: *const f32,
        num_elements: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // Attention mechanisms
    pub fn cuda_scaled_dot_product_attention(
        output: *mut f32,
        query: *const f32,
        key: *const f32,
        value: *const f32,
        mask: *const f32,
        batch_size: c_int,
        seq_len: c_int,
        head_dim: c_int,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // Sampling operations
    pub fn cuda_top_k_sampling(
        output_tokens: *mut c_int,
        output_probs: *mut f32,
        logits: *const f32,
        batch_size: c_int,
        vocab_size: c_int,
        k: c_int,
        temperature: f32,
        seed: u64,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cuda_top_p_sampling(
        output_tokens: *mut c_int,
        output_probs: *mut f32,
        logits: *const f32,
        batch_size: c_int,
        vocab_size: c_int,
        p: f32,
        temperature: f32,
        seed: u64,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // Performance utilities
    pub fn cuda_warmup_kernels(device_id: c_int) -> cudaError_t;

    pub fn cuda_synchronize_stream(stream: cudaStream_t) -> cudaError_t;

    pub fn cuda_get_memory_info(
        free_bytes: *mut usize,
        total_bytes: *mut usize,
        device_id: c_int,
    ) -> cudaError_t;
}

// CUDA memory management bindings
#[cfg(feature = "cuda")]
extern "C" {
    pub fn cuda_memory_pool_create(
        config: *const CudaMemoryPoolConfig,
        pool: *mut CudaMemoryPoolHandle,
    ) -> CudaMemoryError;

    pub fn cuda_memory_pool_destroy(pool: CudaMemoryPoolHandle) -> CudaMemoryError;

    pub fn cuda_memory_allocate(
        pool: CudaMemoryPoolHandle,
        size: usize,
        alignment: usize,
        allocation: *mut CudaAllocation,
    ) -> CudaMemoryError;

    pub fn cuda_memory_deallocate(
        pool: CudaMemoryPoolHandle,
        allocation: *const CudaAllocation,
    ) -> CudaMemoryError;

    pub fn cuda_memory_pool_get_stats(
        pool: CudaMemoryPoolHandle,
        stats: *mut CudaMemoryPoolStats,
    ) -> CudaMemoryError;
}

// CUDA type definitions
#[cfg(feature = "cuda")]
pub type cudaError_t = u32;
#[cfg(feature = "cuda")]
pub type cudaStream_t = *mut std::ffi::c_void;
#[cfg(feature = "cuda")]
pub type cudaMemcpyKind = u32;

// Memory management types
#[cfg(feature = "cuda")]
pub type CudaMemoryPoolHandle = *mut std::ffi::c_void;

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaMemoryPoolConfig {
    pub device_id: c_int,
    pub initial_pool_size_bytes: usize,
    pub max_pool_size_bytes: usize,
    pub min_block_size_bytes: usize,
    pub max_block_size_bytes: usize,
    pub alignment_bytes: usize,
    pub enable_memory_mapping: bool,
    pub enable_peer_access: bool,
    pub growth_factor: f64,
    pub max_cached_blocks: usize,
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaAllocation {
    pub ptr: *mut std::ffi::c_void,
    pub size: usize,
    pub alignment: usize,
    pub device_id: c_int,
    pub is_pinned: bool,
    pub allocation_id: u64,
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaMemoryPoolStats {
    pub total_allocated_bytes: usize,
    pub total_freed_bytes: usize,
    pub current_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub num_allocations: usize,
    pub num_deallocations: usize,
    pub fragmentation_bytes: usize,
    pub fragmentation_percentage: f64,
    pub largest_free_block: usize,
    pub smallest_free_block: usize,
}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CudaMemoryError {
    CUDA_MEMORY_SUCCESS = 0,
    CUDA_MEMORY_ERROR_OUT_OF_MEMORY = 1,
    CUDA_MEMORY_ERROR_INVALID_ARGUMENT = 2,
    CUDA_MEMORY_ERROR_DEVICE_ERROR = 3,
    CUDA_MEMORY_ERROR_ALREADY_EXISTS = 4,
    CUDA_MEMORY_ERROR_NOT_FOUND = 5,
    CUDA_MEMORY_ERROR_FRAGMENTATION = 6,
}

// CUDA constants
#[cfg(feature = "cuda")]
pub const cudaSuccess: cudaError_t = 0;
#[cfg(feature = "cuda")]
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
#[cfg(feature = "cuda")]
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
#[cfg(feature = "cuda")]
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
