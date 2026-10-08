

#include "cudart/model.hpp"

#include <utility>

namespace cudart {

Tensor Model::forward(Tensor input) {

  Tensor current = std::move(input);

  for ( auto& layer: layers_ ) {

    current = layer->forward(
      std::move(current)
    );

  }

  return current;

}

Layer& Model::layer(size_t index) {
    return *layers_.at(index);
}

const Layer& Model::layer(size_t index) const {
    return *layers_.at(index);
}

size_t Model::layer_count() const {
  return layers_.size();
}

bool Model::empty() const {
  return layers_.empty();
}

}

