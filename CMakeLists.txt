cmake_minimum_required(VERSION 3.20)

# DGX Spark: Blackwell GeForce SM121a (compute capability 12.1)
set(CMAKE_CUDA_ARCHITECTURES "121a")

project(dgx_spark_gemms CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA flags
# --expt-relaxed-constexpr is needed for 87_* blockwise/groupwise kernels (sm120_mma_tma_blockwise_scaling.hpp)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -Xcompiler=-fno-gnu-unique --expt-relaxed-constexpr")
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third-party)
include_directories(${CMAKE_SOURCE_DIR}/third-party/cutlass/examples/common)

set(CUTLASS_AVAILABLE FALSE)
if(EXISTS ${CMAKE_SOURCE_DIR}/third-party/cutlass/include)
    set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/cutlass/include)
    set(CUTLASS_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/cutlass/tools/util/include)
    set(CUTLASS_AVAILABLE TRUE)
endif()

if(CUTLASS_AVAILABLE)
    include_directories(${CUTLASS_INCLUDE_DIR})
    include_directories(${CUTLASS_UTILS_INCLUDE_DIR})
endif()

# Blackwell GeForce kernels for DGX Spark (SM121a)
if(CUTLASS_AVAILABLE)
    message(STATUS "Building Blackwell GeForce kernels for SM121a (DGX Spark)")

    # 79_blackwell_geforce_gemm examples
    add_executable(79a_blackwell_geforce_nvfp4_bf16_gemm
        ${CMAKE_SOURCE_DIR}/sample_kernels/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
    target_link_libraries(79a_blackwell_geforce_nvfp4_bf16_gemm CUDA::cudart)
    set_property(TARGET 79a_blackwell_geforce_nvfp4_bf16_gemm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(79b_blackwell_geforce_nvfp4_nvfp4_gemm
        ${CMAKE_SOURCE_DIR}/sample_kernels/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu)
    target_link_libraries(79b_blackwell_geforce_nvfp4_nvfp4_gemm CUDA::cudart)
    set_property(TARGET 79b_blackwell_geforce_nvfp4_nvfp4_gemm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm
        ${CMAKE_SOURCE_DIR}/sample_kernels/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu)
    target_link_libraries(79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm CUDA::cudart)
    set_property(TARGET 79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(79d_blackwell_geforce_nvfp4_grouped_gemm
        ${CMAKE_SOURCE_DIR}/sample_kernels/79d_blackwell_geforce_nvfp4_grouped_gemm.cu)
    target_link_libraries(79d_blackwell_geforce_nvfp4_grouped_gemm CUDA::cudart)
    set_property(TARGET 79d_blackwell_geforce_nvfp4_grouped_gemm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    # 80_blackwell_geforce_sparse_gemm examples
    add_executable(80a_blackwell_geforce_mxfp8_bf16_sparse_gemm
        ${CMAKE_SOURCE_DIR}/sample_kernels/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu)
    target_link_libraries(80a_blackwell_geforce_mxfp8_bf16_sparse_gemm CUDA::cudart)
    set_property(TARGET 80a_blackwell_geforce_mxfp8_bf16_sparse_gemm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm
        ${CMAKE_SOURCE_DIR}/sample_kernels/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu)
    target_link_libraries(80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm CUDA::cudart)
    set_property(TARGET 80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    # 87_blackwell_geforce_gemm_blockwise examples
    add_executable(87a_blackwell_geforce_fp8_bf16_gemm_blockwise
        ${CMAKE_SOURCE_DIR}/sample_kernels/87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu)
    target_link_libraries(87a_blackwell_geforce_fp8_bf16_gemm_blockwise CUDA::cudart)
    set_property(TARGET 87a_blackwell_geforce_fp8_bf16_gemm_blockwise PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(87b_blackwell_geforce_fp8_bf16_gemm_groupwise
        ${CMAKE_SOURCE_DIR}/sample_kernels/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu)
    target_link_libraries(87b_blackwell_geforce_fp8_bf16_gemm_groupwise CUDA::cudart)
    set_property(TARGET 87b_blackwell_geforce_fp8_bf16_gemm_groupwise PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

    add_executable(87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise
        ${CMAKE_SOURCE_DIR}/sample_kernels/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu)
    target_link_libraries(87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise CUDA::cudart)
    set_property(TARGET 87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

else()
    message(STATUS "Skipping Blackwell kernels (CUTLASS not found)")
endif()

enable_testing()

if(CUTLASS_AVAILABLE)
    add_test(NAME 79a_blackwell_geforce_nvfp4_bf16_gemm COMMAND 79a_blackwell_geforce_nvfp4_bf16_gemm)
    add_test(NAME 79b_blackwell_geforce_nvfp4_nvfp4_gemm COMMAND 79b_blackwell_geforce_nvfp4_nvfp4_gemm)
    add_test(NAME 79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm COMMAND 79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm)
    add_test(NAME 79d_blackwell_geforce_nvfp4_grouped_gemm COMMAND 79d_blackwell_geforce_nvfp4_grouped_gemm)
    add_test(NAME 80a_blackwell_geforce_mxfp8_bf16_sparse_gemm COMMAND 80a_blackwell_geforce_mxfp8_bf16_sparse_gemm)
    add_test(NAME 80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm COMMAND 80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm)
    add_test(NAME 87a_blackwell_geforce_fp8_bf16_gemm_blockwise COMMAND 87a_blackwell_geforce_fp8_bf16_gemm_blockwise)
    add_test(NAME 87b_blackwell_geforce_fp8_bf16_gemm_groupwise COMMAND 87b_blackwell_geforce_fp8_bf16_gemm_groupwise)
    add_test(NAME 87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise COMMAND 87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise)
endif()

# Build:  cmake -B build && cmake --build build
# Test:   ctest --test-dir build
