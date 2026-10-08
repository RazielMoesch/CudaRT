
#include "cudart/layers/flatten.hpp"

#include <stdexcept>
#include <vector>


namespace cudart {

Flatten::Flatten( size_t start_dim ) : start_dim_(start_dim) {

}

Tensor Flatten::forward( Tensor input ) {

  const auto& shape = input.shape();

  if ( shape.empty() ) {
    throw std::runtime_error(
      "Flatten: input tensor has no dimensions"
    );
  }

  if ( start_dim_ >= shape.size() ) {

    throw std::runtime_error(
      "Flatten: start_dim is outside input dimensions"
    );

  }

  std::vector<size_t> new_shape;

  for ( size_t i = 0; i < start_dim_; ++i ) {

    new_shape.push_back(shape[i]);

  }

  size_t flattened_dim = 1;
  for ( size_t i = start_dim_; i < shape.size(); ++i ) {
    flattened_dim *= shape[i];
  }
  new_shape.push_back(flattened_dim);

  input.reshape(new_shape);

  return input;

}

size_t Flatten::start_dim() const {
  return start_dim_;
}


}
