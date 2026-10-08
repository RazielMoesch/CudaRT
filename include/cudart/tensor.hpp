

#pragma once
#include <cstddef>
#include <vector>

namespace cudart {

class Tensor {
public:

  // Constructor initializes default values 
  Tensor() = default;

  // Init with shape
  explicit Tensor( const std::vector<size_t>& shape );

  ~Tensor();

  // dont allow passing in a living tensor
  Tensor( const Tensor& ) = delete;
  Tensor& operator=( Tensor& ) = delete;
  
  // allow passing in dying tensor
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  const std::vector<size_t>& shape() const;

  size_t ndim() const;
  size_t numel() const;
  size_t bytes() const;

  float* data();
  const float* data() const;

  void copy_from_host( const float* host_data );
  void copy_to_host( float* host_data ) const;

  void reshape( const std::vector<size_t>& new_shape );



private:
  std::vector<size_t> shape_;
  size_t numel_ = 0;
  float* data_ = nullptr;
  
  static size_t calculate_numel(
    const std::vector<size_t>& shape
  );

  void allocate();
  void release();
  

};


}

