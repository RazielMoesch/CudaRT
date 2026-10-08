#pragma once

#include "cudart/model.hpp"

#include <string>

namespace cudart {

class Loader {
public:
  static void load(Model& model, const std::string& path);
};

}
