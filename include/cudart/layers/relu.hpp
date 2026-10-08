


#pragma once

#include "cudart/model.hpp"

namespace cudart {

class ReLU : public Layer {
public:
  ReLU() = default;
  Tensor forward(Tensor input) override;


};

}


