

#include "cudart/layers/relu.hpp"
#include "cudart/utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace cudart {

namespace {
__global__
void relu_kernel(
  float* data,
  size_t count
) {

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i >= count ) {
    return;
  }

  if ( data[i] < 0.0f ) {

    data[i] = 0.0f;

  }
}
}

Tensor ReLU::forward(Tensor input) {

  if ( input.numel() == 0 ) {
    return input;
  }

  constexpr size_t THREADS_PER_BLOCK = 256;

  const size_t block_count = 
    ( input.numel() + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;

    relu_kernel<<<block_count, THREADS_PER_BLOCK>>>( input.data(), input.numel() );

    check_cuda(
      cudaGetLastError()
    );

    return input;

}

}
