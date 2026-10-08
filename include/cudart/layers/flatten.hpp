


#pragma once

#include "cudart/model.hpp"
#include <cstddef>

namespace cudart {

class Flatten : public Layer {
public:

  explicit Flatten( size_t start_dim = 1 );

  Tensor forward( Tensor input ) override;

  size_t start_dim() const;

private:
  size_t start_dim_;



};

}



