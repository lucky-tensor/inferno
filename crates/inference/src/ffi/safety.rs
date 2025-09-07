//! Memory safety wrappers and utilities for FFI operations
//!
//! This module provides safe abstractions around potentially unsafe FFI operations,
//! including automatic cleanup, RAII patterns, and error handling.

#![allow(clippy::all, dead_code, unused_imports)]

use super::bindings::*;
use crate::error::{VLLMError, VLLMResult};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// RAII wrapper for CUDA streams
#[cfg(feature = "cuda")]
pub struct SafeCudaStream {
    stream: cudaStream_t,
    device_id: i32,
    is_valid: AtomicBool,
}

#[cfg(feature = "cuda")]
impl SafeCudaStream {
    pub fn new(device_id: i32) -> VLLMResult<Self> {
        // In a real implementation, we would create a CUDA stream here
        let stream = std::ptr::null_mut(); // Placeholder

        Ok(Self {
            stream,
            device_id,
            is_valid: AtomicBool::new(true),
        })
    }

    pub fn stream(&self) -> cudaStream_t {
        if self.is_valid.load(Ordering::Acquire) {
            self.stream
        } else {
            std::ptr::null_mut()
        }
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn synchronize(&self) -> VLLMResult<()> {
        if !self.is_valid.load(Ordering::Acquire) {
            return Err(VLLMError::OperationFailed(
                "Stream is not valid".to_string(),
            ));
        }

        let result = unsafe { cuda_synchronize_stream(self.stream) };
        if result != cudaSuccess {
            return Err(VLLMError::CudaError(format!(
                "Stream synchronization failed: {}",
                result
            )));
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for SafeCudaStream {
    fn drop(&mut self) {
        self.is_valid.store(false, Ordering::Release);
        // In a real implementation, we would destroy the CUDA stream here
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for SafeCudaStream {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SafeCudaStream {}

/// RAII wrapper for CUDA memory allocations
#[cfg(feature = "cuda")]
pub struct SafeCudaAllocation {
    allocation: CudaAllocation,
    pool: CudaMemoryPoolHandle,
    is_valid: AtomicBool,
}

#[cfg(feature = "cuda")]
impl SafeCudaAllocation {
    pub fn new(pool: CudaMemoryPoolHandle, size: usize, alignment: usize) -> VLLMResult<Self> {
        let mut allocation = CudaAllocation {
            ptr: std::ptr::null_mut(),
            size: 0,
            alignment: 0,
            device_id: 0,
            is_pinned: false,
            allocation_id: 0,
        };

        let result = unsafe { cuda_memory_allocate(pool, size, alignment, &mut allocation) };

        if result != CudaMemoryError::CUDA_MEMORY_SUCCESS {
            return Err(VLLMError::OutOfMemory(format!(
                "Failed to allocate {} bytes",
                size
            )));
        }

        Ok(Self {
            allocation,
            pool,
            is_valid: AtomicBool::new(true),
        })
    }

    pub fn ptr(&self) -> *mut std::ffi::c_void {
        if self.is_valid.load(Ordering::Acquire) {
            self.allocation.ptr
        } else {
            std::ptr::null_mut()
        }
    }

    pub fn size(&self) -> usize {
        self.allocation.size
    }

    pub fn device_id(&self) -> i32 {
        self.allocation.device_id
    }

    pub fn allocation_id(&self) -> u64 {
        self.allocation.allocation_id
    }

    /// Cast the allocation to a typed pointer
    pub unsafe fn as_ptr<T>(&self) -> *mut T {
        self.ptr() as *mut T
    }

    /// Cast the allocation to a slice (requires careful size checking)
    pub unsafe fn as_slice<T>(&self, len: usize) -> &[T] {
        if self.size() < len * std::mem::size_of::<T>() {
            panic!("Slice length exceeds allocation size");
        }
        std::slice::from_raw_parts(self.as_ptr::<T>(), len)
    }

    /// Cast the allocation to a mutable slice
    pub unsafe fn as_slice_mut<T>(&mut self, len: usize) -> &mut [T] {
        if self.size() < len * std::mem::size_of::<T>() {
            panic!("Slice length exceeds allocation size");
        }
        std::slice::from_raw_parts_mut(self.as_ptr::<T>(), len)
    }
}

#[cfg(feature = "cuda")]
impl Drop for SafeCudaAllocation {
    fn drop(&mut self) {
        if self.is_valid.swap(false, Ordering::AcqRel) && !self.allocation.ptr.is_null() {
            unsafe {
                let _ = cuda_memory_deallocate(self.pool, &self.allocation);
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for SafeCudaAllocation {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SafeCudaAllocation {}

/// Safe wrapper for memory pool operations
#[cfg(feature = "cuda")]
pub struct SafeCudaMemoryPool {
    handle: CudaMemoryPoolHandle,
    config: CudaMemoryPoolConfig,
    is_valid: AtomicBool,
    allocations: Arc<Mutex<HashMap<u64, Arc<SafeCudaAllocation>>>>,
}

#[cfg(feature = "cuda")]
impl SafeCudaMemoryPool {
    pub fn new(config: CudaMemoryPoolConfig) -> VLLMResult<Self> {
        let mut handle = std::ptr::null_mut();
        let result = unsafe { cuda_memory_pool_create(&config, &mut handle) };

        if result != CudaMemoryError::CUDA_MEMORY_SUCCESS {
            return Err(VLLMError::InitializationFailed(
                "Failed to create CUDA memory pool".to_string(),
            ));
        }

        Ok(Self {
            handle,
            config,
            is_valid: AtomicBool::new(true),
            allocations: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn allocate(&self, size: usize, alignment: usize) -> VLLMResult<Arc<SafeCudaAllocation>> {
        if !self.is_valid.load(Ordering::Acquire) {
            return Err(VLLMError::OperationFailed(
                "Memory pool is not valid".to_string(),
            ));
        }

        let allocation = Arc::new(SafeCudaAllocation::new(self.handle, size, alignment)?);
        let allocation_id = allocation.allocation_id();

        // Track the allocation
        self.allocations
            .lock()
            .insert(allocation_id, allocation.clone());

        Ok(allocation)
    }

    pub fn get_stats(&self) -> VLLMResult<CudaMemoryPoolStats> {
        if !self.is_valid.load(Ordering::Acquire) {
            return Err(VLLMError::OperationFailed(
                "Memory pool is not valid".to_string(),
            ));
        }

        let mut stats = CudaMemoryPoolStats {
            total_allocated_bytes: 0,
            total_freed_bytes: 0,
            current_allocated_bytes: 0,
            peak_allocated_bytes: 0,
            num_allocations: 0,
            num_deallocations: 0,
            fragmentation_bytes: 0,
            fragmentation_percentage: 0.0,
            largest_free_block: 0,
            smallest_free_block: 0,
        };

        let result = unsafe { cuda_memory_pool_get_stats(self.handle, &mut stats) };

        if result != CudaMemoryError::CUDA_MEMORY_SUCCESS {
            return Err(VLLMError::OperationFailed(
                "Failed to get memory pool stats".to_string(),
            ));
        }

        Ok(stats)
    }

    pub fn device_id(&self) -> i32 {
        self.config.device_id
    }

    pub fn trim_unused(&self) -> VLLMResult<()> {
        if !self.is_valid.load(Ordering::Acquire) {
            return Err(VLLMError::OperationFailed(
                "Memory pool is not valid".to_string(),
            ));
        }

        // Remove any deallocated entries from our tracking
        self.allocations
            .lock()
            .retain(|_, allocation| Arc::strong_count(allocation) > 1);

        Ok(())
    }

    pub fn active_allocations(&self) -> usize {
        self.allocations.lock().len()
    }
}

#[cfg(feature = "cuda")]
impl Drop for SafeCudaMemoryPool {
    fn drop(&mut self) {
        if self.is_valid.swap(false, Ordering::AcqRel) && !self.handle.is_null() {
            // Clear all tracked allocations first
            self.allocations.lock().clear();

            // Destroy the pool
            unsafe {
                let _ = cuda_memory_pool_destroy(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for SafeCudaMemoryPool {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SafeCudaMemoryPool {}

/// Thread-safe resource manager for CUDA resources
#[cfg(feature = "cuda")]
pub struct CudaResourceManager {
    streams: Arc<Mutex<HashMap<i32, Vec<Arc<SafeCudaStream>>>>>,
    memory_pools: Arc<Mutex<HashMap<i32, Arc<SafeCudaMemoryPool>>>>,
}

#[cfg(feature = "cuda")]
impl CudaResourceManager {
    pub fn new() -> Self {
        Self {
            streams: Arc::new(Mutex::new(HashMap::new())),
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_or_create_memory_pool(
        &self,
        device_id: i32,
        config: CudaMemoryPoolConfig,
    ) -> VLLMResult<Arc<SafeCudaMemoryPool>> {
        let mut pools = self.memory_pools.lock();

        if let Some(pool) = pools.get(&device_id) {
            if pool.is_valid.load(Ordering::Acquire) {
                return Ok(pool.clone());
            }
        }

        let pool = Arc::new(SafeCudaMemoryPool::new(config)?);
        pools.insert(device_id, pool.clone());
        Ok(pool)
    }

    pub fn get_or_create_stream(&self, device_id: i32) -> VLLMResult<Arc<SafeCudaStream>> {
        let mut streams = self.streams.lock();

        let device_streams = streams.entry(device_id).or_insert_with(Vec::new);

        // Try to find a valid stream
        for stream in device_streams.iter() {
            if stream.is_valid.load(Ordering::Acquire) {
                return Ok(stream.clone());
            }
        }

        // Create a new stream
        let stream = Arc::new(SafeCudaStream::new(device_id)?);
        device_streams.push(stream.clone());
        Ok(stream)
    }

    pub fn cleanup_device(&self, device_id: i32) {
        self.streams.lock().remove(&device_id);
        self.memory_pools.lock().remove(&device_id);
    }

    pub fn cleanup_all(&self) {
        self.streams.lock().clear();
        self.memory_pools.lock().clear();
    }
}

#[cfg(feature = "cuda")]
impl Default for CudaResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global resource manager instance
#[cfg(feature = "cuda")]
static GLOBAL_RESOURCE_MANAGER: once_cell::sync::Lazy<CudaResourceManager> =
    once_cell::sync::Lazy::new(CudaResourceManager::new);

/// Get the global CUDA resource manager
#[cfg(feature = "cuda")]
pub fn get_cuda_resource_manager() -> &'static CudaResourceManager {
    &GLOBAL_RESOURCE_MANAGER
}

/// Safe wrapper for string handling in FFI
pub struct SafeCString {
    c_string: std::ffi::CString,
}

impl SafeCString {
    pub fn new(s: &str) -> VLLMResult<Self> {
        let c_string = std::ffi::CString::new(s)
            .map_err(|_| VLLMError::InvalidArgument("String contains null byte".to_string()))?;

        Ok(Self { c_string })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_char {
        self.c_string.as_ptr()
    }
}

/// Utility macro for safe FFI calls with error handling
#[macro_export]
macro_rules! safe_ffi_call {
    ($call:expr) => {{
        let result = unsafe { $call };
        if result != VLLMErrorCode::VLLM_SUCCESS {
            return Err(VLLMError::OperationFailed($crate::ffi::get_error_string(
                result,
            )));
        }
    }};
}

/// Utility macro for safe CUDA calls with error handling  
#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! safe_cuda_call {
    ($call:expr) => {{
        let result = unsafe { $call };
        if result != $crate::ffi::bindings::cudaSuccess {
            return Err(VLLMError::CudaError(format!("CUDA error: {}", result)));
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_cstring() {
        let safe_str = SafeCString::new("test string").unwrap();
        assert!(!safe_str.as_ptr().is_null());

        // Test null byte handling
        let result = SafeCString::new("test\0string");
        assert!(result.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_resource_manager() {
        let manager = CudaResourceManager::new();
        assert_eq!(manager.streams.lock().len(), 0);
        assert_eq!(manager.memory_pools.lock().len(), 0);
    }
}
