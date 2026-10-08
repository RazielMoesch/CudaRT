

cmake_minimum_required(VERSION 3.24)

project(
  CudaRT VERSION 0.1.0 LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(CUDAToolkit REQUIRED)

add_library(
  cudart_runtime
  src/tensor.cpp
  src/model.cpp
  src/utils.cpp
  src/loader.cpp
  src/layers/linear.cu
  src/layers/relu.cu
  src/layers/flatten.cpp
)

target_include_directories(
  cudart_runtime
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

add_executable(
  cudart_example
  examples/mnist.cpp
)

target_compile_definitions(cudart_example PRIVATE CUDART_BUILD_TYPE="$<CONFIG>")

target_link_libraries(cudart_runtime PUBLIC CUDA::cudart)

target_link_libraries(
  cudart_example
  PRIVATE
  cudart_runtime
)
