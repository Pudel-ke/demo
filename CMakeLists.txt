cmake_minimum_required(VERSION 3.18)

project(demo LANGUAGES CXX CUDA)

# RTX 5060 -> Compute Capability 12.0 -> sm_120
set(CMAKE_CUDA_ARCHITECTURES 120)

# 使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 使用 C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成可执行文件
add_executable(gemm gemm_cuda.cu)
