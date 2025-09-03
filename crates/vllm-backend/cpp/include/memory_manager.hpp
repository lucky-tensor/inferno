#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Memory pool handle
typedef struct CudaMemoryPool* CudaMemoryPoolHandle;

// Memory allocation info
typedef struct {
    void* ptr;
    size_t size;
    size_t alignment;
    int device_id;
    bool is_pinned;
    uint64_t allocation_id;
} CudaAllocation;

// Memory statistics
typedef struct {
    size_t total_allocated_bytes;
    size_t total_freed_bytes;
    size_t current_allocated_bytes;
    size_t peak_allocated_bytes;
    size_t num_allocations;
    size_t num_deallocations;
    size_t fragmentation_bytes;
    double fragmentation_percentage;
    size_t largest_free_block;
    size_t smallest_free_block;
} CudaMemoryPoolStats;

// Error codes for memory operations
typedef enum {
    CUDA_MEMORY_SUCCESS = 0,
    CUDA_MEMORY_ERROR_OUT_OF_MEMORY = 1,
    CUDA_MEMORY_ERROR_INVALID_ARGUMENT = 2,
    CUDA_MEMORY_ERROR_DEVICE_ERROR = 3,
    CUDA_MEMORY_ERROR_ALREADY_EXISTS = 4,
    CUDA_MEMORY_ERROR_NOT_FOUND = 5,
    CUDA_MEMORY_ERROR_FRAGMENTATION = 6,
} CudaMemoryError;

// Memory pool configuration
typedef struct {
    int device_id;
    size_t initial_pool_size_bytes;
    size_t max_pool_size_bytes;
    size_t min_block_size_bytes;
    size_t max_block_size_bytes;
    size_t alignment_bytes;
    bool enable_memory_mapping;
    bool enable_peer_access;
    double growth_factor;
    size_t max_cached_blocks;
} CudaMemoryPoolConfig;

// Pool lifecycle functions
CudaMemoryError cuda_memory_pool_create(const CudaMemoryPoolConfig* config, 
                                        CudaMemoryPoolHandle* pool);
CudaMemoryError cuda_memory_pool_destroy(CudaMemoryPoolHandle pool);
CudaMemoryError cuda_memory_pool_reset(CudaMemoryPoolHandle pool);

// Memory allocation functions
CudaMemoryError cuda_memory_allocate(CudaMemoryPoolHandle pool, 
                                     size_t size,
                                     size_t alignment,
                                     CudaAllocation* allocation);
CudaMemoryError cuda_memory_allocate_pinned(CudaMemoryPoolHandle pool,
                                            size_t size,
                                            size_t alignment,
                                            CudaAllocation* allocation);
CudaMemoryError cuda_memory_deallocate(CudaMemoryPoolHandle pool, 
                                       const CudaAllocation* allocation);

// Bulk allocation functions
CudaMemoryError cuda_memory_allocate_batch(CudaMemoryPoolHandle pool,
                                           const size_t* sizes,
                                           const size_t* alignments,
                                           CudaAllocation* allocations,
                                           size_t num_allocations);
CudaMemoryError cuda_memory_deallocate_batch(CudaMemoryPoolHandle pool,
                                             const CudaAllocation* allocations,
                                             size_t num_allocations);

// Memory pool management
CudaMemoryError cuda_memory_pool_get_stats(CudaMemoryPoolHandle pool, 
                                           CudaMemoryPoolStats* stats);
CudaMemoryError cuda_memory_pool_trim(CudaMemoryPoolHandle pool);
CudaMemoryError cuda_memory_pool_compact(CudaMemoryPoolHandle pool);
CudaMemoryError cuda_memory_pool_set_max_size(CudaMemoryPoolHandle pool, 
                                              size_t max_size_bytes);

// Device memory utilities
CudaMemoryError cuda_get_device_memory_info(int device_id,
                                            size_t* free_bytes,
                                            size_t* total_bytes);
CudaMemoryError cuda_set_memory_fraction(int device_id, double fraction);
CudaMemoryError cuda_enable_peer_access(int src_device, int dst_device);
CudaMemoryError cuda_disable_peer_access(int src_device, int dst_device);

// Memory mapping and prefetching
CudaMemoryError cuda_memory_prefetch_async(void* ptr, 
                                           size_t size,
                                           int target_device,
                                           cudaStream_t stream);
CudaMemoryError cuda_memory_advise(void* ptr,
                                   size_t size,
                                   cudaMemoryAdvise advice,
                                   int device_id);

// Cache management
CudaMemoryError cuda_memory_cache_flush(CudaMemoryPoolHandle pool);
CudaMemoryError cuda_memory_cache_get_info(CudaMemoryPoolHandle pool,
                                           size_t* cached_bytes,
                                           size_t* num_cached_blocks);

// Memory debugging and validation
CudaMemoryError cuda_memory_validate_pointer(const void* ptr, size_t size);
CudaMemoryError cuda_memory_leak_check(CudaMemoryPoolHandle pool);
CudaMemoryError cuda_memory_dump_stats(CudaMemoryPoolHandle pool, 
                                       char* buffer,
                                       size_t buffer_size);

// Multi-device support
CudaMemoryError cuda_memory_pool_create_multi_device(const CudaMemoryPoolConfig* configs,
                                                     int num_devices,
                                                     CudaMemoryPoolHandle* pools);
CudaMemoryError cuda_memory_cross_device_copy(void* dst_ptr,
                                              int dst_device,
                                              const void* src_ptr,
                                              int src_device,
                                              size_t size,
                                              cudaStream_t stream);

// Advanced features
CudaMemoryError cuda_memory_pool_set_attribute(CudaMemoryPoolHandle pool,
                                               int attribute,
                                               const void* value,
                                               size_t value_size);
CudaMemoryError cuda_memory_pool_get_attribute(CudaMemoryPoolHandle pool,
                                               int attribute,
                                               void* value,
                                               size_t* value_size);

// Memory pool attributes
enum CudaMemoryPoolAttribute {
    CUDA_MEMORY_POOL_ATTR_GROWTH_FACTOR = 0,
    CUDA_MEMORY_POOL_ATTR_MAX_CACHED_BLOCKS = 1,
    CUDA_MEMORY_POOL_ATTR_DEFRAG_THRESHOLD = 2,
    CUDA_MEMORY_POOL_ATTR_ENABLE_LOGGING = 3,
};

// Memory advice options
enum CudaMemoryAdvice {
    CUDA_MEMORY_ADVISE_SET_READ_MOSTLY = 1,
    CUDA_MEMORY_ADVISE_UNSET_READ_MOSTLY = 2,
    CUDA_MEMORY_ADVISE_SET_PREFERRED_LOCATION = 3,
    CUDA_MEMORY_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    CUDA_MEMORY_ADVISE_SET_ACCESSED_BY = 5,
    CUDA_MEMORY_ADVISE_UNSET_ACCESSED_BY = 6,
};

#ifdef __cplusplus
}
#endif