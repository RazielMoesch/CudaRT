


#pragma once

#include "cudart/model.hpp"
#include <cstddef>


namespace cudart {


class Linear : public Layer {
public:
  Linear(
    size_t in_features,
    size_t out_features,
    bool use_bias = true
  );

  Tensor forward(Tensor input) override;

  size_t in_features() const;
  size_t out_features() const;
  bool has_bias() const;

  const Tensor& weights() const;
  const Tensor& bias() const;

  void load_parameters(
    const float* weigth_data,
    const float* bias_data = nullptr
  );

private:
  size_t in_features_;
  size_t out_features_;
  bool use_bias_;

  Tensor weight_;
  Tensor bias_;




};


}

