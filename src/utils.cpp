
#include "cudart/utils.hpp"
#include <stdexcept>

namespace cudart {

void check_cuda( cudaError_t error ) {

  if ( error != cudaSuccess ) {
    
    throw std::runtime_error(
      cudaGetErrorString(error)
    );

  }

}



}
