cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(inferno_vllm_cpp LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# CUDA architecture detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# CUDA compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler=-fPIC")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(CPP_SOURCES
    src/vllm_wrapper.cpp
    src/memory_manager.cpp
)

set(CUDA_SOURCES
    src/cuda_kernels.cu
)

# Create the shared library
add_library(inferno_vllm_cpp SHARED
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Link CUDA libraries
target_link_libraries(inferno_vllm_cpp
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cusparse
)

# Set library properties
set_target_properties(inferno_vllm_cpp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install rules
install(TARGETS inferno_vllm_cpp
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()