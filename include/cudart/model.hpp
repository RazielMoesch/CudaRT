

#pragma once

#include "cudart/tensor.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace cudart {


class Layer {
public:
  virtual ~Layer() = default;

  virtual Tensor forward(Tensor input) = 0;


};

class Model {
public: 

  Model() = default;
  
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(Model&&) noexcept = default;
  Model& operator=(Model&&) noexcept = default;

  template<typename LayerType, typename... Args>
  LayerType& add( Args&&... args ) {
    auto layer = std::make_unique<LayerType>(
      std::forward<Args>(args)...
    );

    LayerType& reference = *layer;

    layers_.push_back(std::move(layer));

    return reference;

  }

  Layer& layer(size_t index);
  const Layer& layer(size_t index) const;

  Tensor forward( Tensor input );

  size_t layer_count() const;

  bool empty() const;

private:
  std::vector<std::unique_ptr<Layer>> layers_;

};



}

