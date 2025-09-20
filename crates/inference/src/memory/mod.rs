//! GPU memory management

use crate::error::{AllocationError, InfernoResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// GPU memory allocator trait
#[async_trait]
pub trait GpuAllocator: Send + Sync {
    /// Allocate memory on the GPU
    async fn allocate(&self, size: usize, alignment: usize) -> InfernoResult<DeviceMemory>;

    /// Deallocate memory on the GPU
    async fn deallocate(&self, memory: DeviceMemory) -> InfernoResult<()>;

    /// Get memory statistics
    async fn get_stats(&self) -> InfernoResult<MemoryStats>;
}

/// CUDA memory pool implementation
pub struct CudaMemoryPool {
    device_id: i32,
    allocation_counter: AtomicU64,
}

impl CudaMemoryPool {
    /// Create a new CUDA memory pool
    pub fn new(device_id: i32) -> InfernoResult<Self> {
        // Validate device ID
        if device_id < -1 {
            return Err(crate::error::AllocationError::DeviceMemory(format!(
                "Invalid device ID: {device_id}"
            ))
            .into());
        }

        Ok(Self {
            device_id,
            allocation_counter: AtomicU64::new(0),
        })
    }

    /// Get the device ID
    pub const fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Check if CUDA is available for this pool
    pub const fn is_cuda_available(&self) -> bool {
        self.device_id >= 0
    }

    /// Get pool configuration info
    pub fn info(&self) -> MemoryPoolInfo {
        MemoryPoolInfo {
            device_id: self.device_id,
            pool_type: if self.device_id >= 0 { "cuda" } else { "cpu" }.to_string(),
            is_available: self.is_cuda_available(),
        }
    }
}

#[async_trait]
impl GpuAllocator for CudaMemoryPool {
    async fn allocate(&self, size: usize, alignment: usize) -> InfernoResult<DeviceMemory> {
        // Validate input parameters
        if size == 0 {
            return Err(AllocationError::InvalidAlignment(size).into());
        }

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(AllocationError::InvalidAlignment(alignment).into());
        }

        // CPU-only allocation for now (CUDA implementation would go here)
        if self.device_id < 0 {
            // CPU allocation using aligned memory
            let layout = std::alloc::Layout::from_size_align(size, alignment)
                .map_err(|_| AllocationError::InvalidAlignment(alignment))?;

            let ptr = unsafe { std::alloc::alloc(layout) }.cast::<std::ffi::c_void>();

            if ptr.is_null() {
                return Err(AllocationError::OutOfMemory {
                    requested: size,
                    available: 0, // Could query system memory here
                }
                .into());
            }

            let allocation_id = self.allocation_counter.fetch_add(1, Ordering::SeqCst);

            Ok(DeviceMemory {
                ptr,
                size,
                device_id: self.device_id,
                allocation_id,
            })
        } else {
            // GPU allocation would be implemented here with CUDA
            {
                Err(crate::error::InfernoError::CudaNotAvailable)
            }

            {
                // TODO: Implement actual CUDA allocation using cudarc
                let allocation_id = self.allocation_counter.fetch_add(1, Ordering::SeqCst);

                Ok(DeviceMemory {
                    ptr: std::ptr::null_mut(), // Would be actual CUDA ptr
                    size,
                    device_id: self.device_id,
                    allocation_id,
                })
            }
        }
    }

    async fn deallocate(&self, memory: DeviceMemory) -> InfernoResult<()> {
        if memory.device_id != self.device_id {
            return Err(AllocationError::DeviceMemory(format!(
                "Device ID mismatch: expected {}, got {}",
                self.device_id, memory.device_id
            ))
            .into());
        }

        if memory.ptr.is_null() {
            return Err(
                AllocationError::DeviceMemory("Null pointer in deallocation".to_string()).into(),
            );
        }

        if memory.device_id < 0 {
            // CPU deallocation
            let layout = std::alloc::Layout::from_size_align(memory.size, 8) // Assume 8-byte alignment
                .map_err(|_| {
                    AllocationError::DeviceMemory("Invalid layout for deallocation".to_string())
                })?;

            unsafe {
                std::alloc::dealloc(memory.ptr.cast::<u8>(), layout);
            }
        } else {
            // GPU deallocation would be implemented here
            {
                return Err(crate::error::InfernoError::CudaNotAvailable);
            }

            {
                // TODO: Implement actual CUDA deallocation
                tracing::debug!(
                    "Deallocating GPU memory: allocation_id={}",
                    memory.allocation_id
                );
            }
        }

        Ok(())
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    async fn get_stats(&self) -> InfernoResult<MemoryStats> {
        let mut stats = MemoryStats {
            device_id: self.device_id,
            ..Default::default()
        };

        if self.device_id < 0 {
            // CPU memory stats (simplified)
            stats.total_memory_bytes = 8 * 1024 * 1024 * 1024; // Assume 8GB RAM
            stats.utilization_percentage = 0.5; // Placeholder
            stats.num_allocations = self.allocation_counter.load(Ordering::SeqCst) as usize;
        } else {
            {
                return Err(crate::error::InfernoError::CudaNotAvailable);
            }

            {
                // TODO: Get actual CUDA memory stats using cudarc
                stats.total_memory_bytes = 12 * 1024 * 1024 * 1024; // Placeholder: 12GB GPU
                stats.utilization_percentage = 0.3; // Placeholder
                stats.num_allocations = self.allocation_counter.load(Ordering::SeqCst) as usize;
            }
        }

        Ok(stats)
    }
}

/// Memory tracker for resource monitoring
pub struct MemoryTracker {
    /// GPU device ID being tracked
    device_id: i32,
    /// Number of active allocations
    active_allocations: AtomicU64,
    /// Total bytes allocated
    total_allocated_bytes: AtomicU64,
    /// Peak memory usage
    peak_memory_bytes: AtomicU64,
}

impl MemoryTracker {
    /// Create a new memory tracker
    #[must_use]
    pub const fn new(device_id: i32) -> Self {
        Self {
            device_id,
            active_allocations: AtomicU64::new(0),
            total_allocated_bytes: AtomicU64::new(0),
            peak_memory_bytes: AtomicU64::new(0),
        }
    }

    /// Track an allocation
    pub fn track_allocation(&self, memory: &DeviceMemory) {
        self.active_allocations.fetch_add(1, Ordering::SeqCst);
        let new_total = self
            .total_allocated_bytes
            .fetch_add(memory.size as u64, Ordering::SeqCst)
            + memory.size as u64;

        // Update peak memory usage
        let mut current_peak = self.peak_memory_bytes.load(Ordering::SeqCst);
        while new_total > current_peak {
            match self.peak_memory_bytes.compare_exchange_weak(
                current_peak,
                new_total,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual,
            }
        }

        tracing::debug!(
            "Tracked allocation: device={}, size={}, total={}, active={}",
            memory.device_id,
            memory.size,
            new_total,
            self.active_allocations.load(Ordering::SeqCst)
        );
    }

    /// Track a deallocation
    pub fn track_deallocation(&self, memory: &DeviceMemory) {
        let prev_active = self.active_allocations.fetch_sub(1, Ordering::SeqCst);
        let prev_total = self
            .total_allocated_bytes
            .fetch_sub(memory.size as u64, Ordering::SeqCst);

        tracing::debug!(
            "Tracked deallocation: device={}, size={}, remaining_total={}, active={}",
            memory.device_id,
            memory.size,
            prev_total.saturating_sub(memory.size as u64),
            prev_active.saturating_sub(1)
        );
    }

    /// Get tracking statistics
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn get_stats(&self) -> MemoryStats {
        let active_allocations = self.active_allocations.load(Ordering::SeqCst) as usize;
        let total_allocated = self.total_allocated_bytes.load(Ordering::SeqCst) as usize;
        let peak_memory = self.peak_memory_bytes.load(Ordering::SeqCst) as usize;

        MemoryStats {
            device_id: self.device_id,
            used_memory_bytes: total_allocated,
            total_memory_bytes: if self.device_id < 0 {
                8 * 1024 * 1024 * 1024 // 8GB RAM assumption
            } else {
                12 * 1024 * 1024 * 1024 // 12GB GPU assumption
            },
            free_memory_bytes: if self.device_id < 0 {
                (8_usize * 1024 * 1024 * 1024).saturating_sub(total_allocated)
            } else {
                (12_usize * 1024 * 1024 * 1024).saturating_sub(total_allocated)
            },
            utilization_percentage: if self.device_id < 0 {
                total_allocated as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)
            } else {
                total_allocated as f64 / (12.0 * 1024.0 * 1024.0 * 1024.0)
            },
            num_allocations: active_allocations,
            fragmentation_bytes: peak_memory.saturating_sub(total_allocated), // Simple fragmentation estimate
            cached_memory_bytes: 0, // No caching implemented yet
            num_deallocations: 0,   // Would need separate counter
        }
    }

    /// Get the device ID being tracked
    pub const fn device_id(&self) -> i32 {
        self.device_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = CudaMemoryPool::new(-1);
        assert!(pool.is_ok());
        let pool = pool.unwrap();
        assert_eq!(pool.device_id(), -1);
        assert!(!pool.is_cuda_available());

        let pool = CudaMemoryPool::new(0);
        assert!(pool.is_ok());
        let pool = pool.unwrap();
        assert_eq!(pool.device_id(), 0);
        assert!(pool.is_cuda_available());

        // Invalid device ID
        let pool = CudaMemoryPool::new(-2);
        assert!(pool.is_err());
    }

    #[test]
    fn test_memory_pool_info() {
        let pool = CudaMemoryPool::new(-1).unwrap();
        let info = pool.info();
        assert_eq!(info.device_id, -1);
        assert_eq!(info.pool_type, "cpu");
        assert!(!info.is_available);

        let pool = CudaMemoryPool::new(0).unwrap();
        let info = pool.info();
        assert_eq!(info.device_id, 0);
        assert_eq!(info.pool_type, "cuda");
        assert!(info.is_available);
    }

    #[tokio::test]
    async fn test_cpu_allocation_deallocation() {
        let pool = CudaMemoryPool::new(-1).unwrap();

        // Test successful allocation
        let memory = pool.allocate(1024, 8).await;
        assert!(memory.is_ok());
        let memory = memory.unwrap();
        assert_eq!(memory.size, 1024);
        assert_eq!(memory.device_id, -1);
        assert!(!memory.ptr.is_null());

        // Test deallocation
        let result = pool.deallocate(memory).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_allocation_validation() {
        let pool = CudaMemoryPool::new(-1).unwrap();

        // Zero size should fail
        let result = pool.allocate(0, 8).await;
        assert!(result.is_err());

        // Invalid alignment should fail
        let result = pool.allocate(1024, 0).await;
        assert!(result.is_err());

        let result = pool.allocate(1024, 3).await; // Not power of 2
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new(-1);
        assert_eq!(tracker.device_id(), -1);

        let memory = DeviceMemory {
            ptr: std::ptr::null_mut(),
            size: 1024,
            device_id: -1,
            allocation_id: 1,
        };

        tracker.track_allocation(&memory);
        let stats = tracker.get_stats();
        assert_eq!(stats.used_memory_bytes, 1024);
        assert_eq!(stats.num_allocations, 1);

        tracker.track_deallocation(&memory);
        let stats = tracker.get_stats();
        assert_eq!(stats.used_memory_bytes, 0);
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let pool = CudaMemoryPool::new(-1).unwrap();
        let stats = pool.get_stats().await;
        assert!(stats.is_ok());
        let stats = stats.unwrap();
        assert_eq!(stats.device_id, -1);
        assert!(stats.total_memory_bytes > 0);
    }
}

/// Device memory handle
#[derive(Debug)]
pub struct DeviceMemory {
    /// Raw pointer to allocated memory
    pub ptr: *mut std::ffi::c_void,
    /// Size of allocation in bytes
    pub size: usize,
    /// GPU device ID where memory is allocated
    pub device_id: i32,
    /// Unique identifier for this allocation
    pub allocation_id: u64,
}

unsafe impl Send for DeviceMemory {}
unsafe impl Sync for DeviceMemory {}

/// Memory usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total device memory in bytes
    pub total_memory_bytes: usize,
    /// Currently allocated memory in bytes
    pub used_memory_bytes: usize,
    /// Available memory in bytes
    pub free_memory_bytes: usize,
    /// Memory held in cache in bytes
    pub cached_memory_bytes: usize,
    /// Memory lost to fragmentation in bytes
    pub fragmentation_bytes: usize,
    /// Memory utilization as percentage (0.0-1.0)
    pub utilization_percentage: f64,
    /// GPU device ID
    pub device_id: i32,
    /// Total number of allocations performed
    pub num_allocations: usize,
    /// Total number of deallocations performed
    pub num_deallocations: usize,
}

/// Memory pool configuration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolInfo {
    /// GPU device ID
    pub device_id: i32,
    /// Pool type (cpu, cuda)
    pub pool_type: String,
    /// Whether the pool backend is available
    pub is_available: bool,
}
