


#include "cudart/utils.hpp"
#include "cudart/tensor.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>


namespace cudart {
Tensor::Tensor( const std::vector<size_t>& shape )
: shape_(shape),
  numel_(calculate_numel(shape))
{

  allocate();

}

Tensor::~Tensor() {
  release();
}


Tensor::Tensor( Tensor&& other ) noexcept 
: shape_(std::move(other.shape_)),
  numel_(other.numel_),
  data_(other.data_)
{

  other.numel_ = 0;
  other.data_ = nullptr;

}

Tensor& Tensor::operator=( Tensor&& other ) noexcept {

  if ( this == &other ) {
    return *this;
  }

  release();

  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;

  other.numel_ = 0;
  other.data_ = nullptr;

  return *this;

}

const std::vector<size_t>& Tensor::shape() const {

  return shape_;

}


size_t Tensor::ndim() const {

  return shape_.size();

}

size_t Tensor::numel() const {

  return numel_;

}

size_t Tensor::bytes() const {

  return numel_ * sizeof(float);

}


float* Tensor::data() {

  return data_;

}

const float* Tensor::data() const {

  return data_;

}

void Tensor::copy_from_host( const float* host_data ) {

  check_cuda(
    cudaMemcpy(
      data_,
      host_data,
      bytes(),
      cudaMemcpyHostToDevice
    )
  );

}

void Tensor::copy_to_host( float* host_data ) const {

  check_cuda(
    cudaMemcpy(
      host_data,
      data_,
      bytes(),
      cudaMemcpyDeviceToHost
    )
  );

}

void Tensor::reshape(
  const std::vector<size_t>& new_shape
) {
  size_t new_numel = calculate_numel(new_shape);

  if ( new_numel != numel_ ) {
    throw std::runtime_error(
      "Tensor::reshape cannot change the number of elements"
    );
  }

  shape_ = new_shape;

}

size_t Tensor::calculate_numel(
  const std::vector<size_t>& shape
) {

  if ( shape.empty() ) {
    return 0;
  }

  size_t count = 1;

  for ( size_t dim : shape ) {
    count *= dim;
  }

  return count;

}


void Tensor::allocate() {

  if ( numel_ == 0 ) {
    return;
  }

  check_cuda(
    cudaMalloc(
      reinterpret_cast<void**>(&data_),
      bytes()
    )
  );

}

void Tensor::release() {

  if ( data_ != nullptr ) {
    cudaFree(data_);
    data_ = nullptr;
  }

  numel_ = 0;

}


}


