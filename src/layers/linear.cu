

#include "cudart/layers/linear.hpp"
#include "cudart/utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudart {

namespace {

__global__
void linear_kernel(

  const float* input,
  const float* weight,
  const float* bias,
  float* output,

  size_t batch_size,
  size_t in_features,
  size_t out_features

) {

  const size_t out_feature = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t batch = blockIdx.y * blockDim.y + threadIdx.y;

  if ( batch >= batch_size || out_feature >= out_features ) {
    return;
  }

  float sum = 0.0f;

  for ( size_t i = 0; i < in_features; ++i ) {
  
    const float x  = input[ batch * in_features + i ];

    const float w = weight[ out_feature * in_features + i ];

    sum += x * w;

  }

  if ( bias != nullptr ) {
    sum += bias[out_feature];
  }

  output[ batch * out_features + out_feature ] = sum;

}
}

Linear::Linear (
  size_t in_features,
  size_t out_features,
  bool use_bias
) 
: in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
  weight_({ out_features, in_features }), bias_( use_bias ? std::vector<size_t>{out_features} : std::vector<size_t>{} )
{

  if (in_features_ == 0) {

    throw std::invalid_argument("Linear: in_features cannot be zero");

  }

  if ( out_features == 0 ) {

    throw std::invalid_argument("Linear: out_features cannot be zero");

  }

}

Tensor Linear::forward(Tensor input) {

  if ( input.ndim() == 0 ) {

    throw std::runtime_error(
      "Linear: input tensor has no dimensions"
    );

  }

  const auto& input_shape = input.shape();

  if ( input_shape.back() != in_features_ ) {

    throw std::runtime_error(
      "Linear: final input dimension does not match in_features"
    );

  }

  const size_t batch_size = input.numel() / in_features_;

  std::vector<size_t> output_shape = input_shape;
  output_shape.back() = out_features_;

  Tensor output(output_shape);

  if (batch_size == 0) {
    return output;
  }

  constexpr int TILE_X = 16;
  constexpr int TILE_Y = 16;

  dim3 block( TILE_X, TILE_Y );

  dim3 grid(
    ( out_features_ + TILE_X - 1 ) / TILE_X,
    ( batch_size + TILE_Y - 1 ) / TILE_Y
  );

  linear_kernel<<<grid, block>>>(
    input.data(),
    weight_.data(),
    use_bias_ ? bias_.data() : nullptr,
    output.data(),
    batch_size,
    in_features_,
    out_features_
  );

  check_cuda(cudaGetLastError());

  return output;

}

size_t Linear::in_features() const { return in_features_; }
size_t Linear::out_features() const { return out_features_; }
bool Linear::has_bias() const { return use_bias_; }
const Tensor& Linear::weights() const { return weight_; }
const Tensor& Linear::bias() const { return bias_; }

void Linear::load_parameters(const float* weight_data, const float* bias_data) {
  if (weight_data == nullptr || (use_bias_ && bias_data == nullptr)) {
    throw std::invalid_argument("Linear: missing parameter data");
  }
  if (!use_bias_ && bias_data != nullptr) {
    throw std::invalid_argument("Linear: bias supplied to a layer without bias");
  }
  weight_.copy_from_host(weight_data);
  if (use_bias_) {
    bias_.copy_from_host(bias_data);
  }
}


}
