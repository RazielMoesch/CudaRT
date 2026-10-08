#include "cudart/layers/flatten.hpp"
#include "cudart/layers/linear.hpp"
#include "cudart/layers/relu.hpp"
#include "cudart/loader.hpp"
#include "cudart/model.hpp"
#include "cudart/tensor.hpp"
#include "cudart/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

size_t positive_size(const char* value) {
  const std::string text(value);
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a positive integer");
  }
  const auto number = std::stoull(text);
  if (number == 0 || number > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("Integer is out of range");
  }
  return static_cast<size_t>(number);
}

std::vector<float> load_images(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Could not open images: " + path);
  const auto length = file.tellg();
  if (length <= 0 || length % 784 != 0) {
    throw std::runtime_error("Images must contain N * 784 raw uint8 pixels");
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(length));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(bytes.data()), length)) {
    throw std::runtime_error("Could not read images");
  }
  std::vector<float> images(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) images[i] = float(bytes[i]) / 255.0f;
  return images;
}

struct Events {
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  Events() {
    cudart::check_cuda(cudaEventCreate(&start));
    const auto error = cudaEventCreate(&end);
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudart::check_cuda(error);
    }
  }
  ~Events() { cudaEventDestroy(start); cudaEventDestroy(end); }
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
};

std::vector<float> infer(cudart::Model& model, const float* pixels, size_t count,
                         Events* events = nullptr, float* forward_ms = nullptr) {
  cudart::Tensor input({count, 28, 28});
  input.copy_from_host(pixels);
  if (events) cudart::check_cuda(cudaEventRecord(events->start));
  auto output = model.forward(std::move(input));
  if (output.shape() != std::vector<size_t>{count, 10}) {
    throw std::runtime_error("Unexpected output shape");
  }
  if (events) {
    cudart::check_cuda(cudaEventRecord(events->end));
    cudart::check_cuda(cudaEventSynchronize(events->end));
    cudart::check_cuda(cudaEventElapsedTime(forward_ms, events->start, events->end));
  }
  std::vector<float> logits(output.numel());
  output.copy_to_host(logits.data());
  return logits;
}

void write_array(std::ostream& out, const std::vector<float>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}
}

int main(int argc, char** argv) {
  try {
    const bool benchmark = argc > 1 && std::string(argv[1]) == "--benchmark";
    if ((!benchmark && argc != 3) || (benchmark && argc != 9)) {
      std::cerr << "Usage:\n  cudart_example <model.crt> <image.bin>\n"
                << "  cudart_example --benchmark <model.crt> <images.bin> <result.json> "
                   "<batch-size> <warmup> <iterations> <device-index>\n";
      return 1;
    }
    const int offset = benchmark ? 2 : 1;
    const std::string weights = argv[offset];
    const auto pixels = load_images(argv[offset + 1]);
    const size_t image_count = pixels.size() / 784;
    const size_t batch = benchmark ? positive_size(argv[5]) : 1;
    const size_t warmup = benchmark ? positive_size(argv[6]) : 1;
    const size_t iterations = benchmark ? positive_size(argv[7]) : 1;
    if (batch > image_count || (!benchmark && image_count != 1)) {
      throw std::invalid_argument("Batch exceeds image count, or single-image input is invalid");
    }
    int device = 0;
    if (benchmark) {
      const std::string device_text(argv[8]);
      if (device_text.empty() || device_text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Invalid CUDA device index");
      }
      device = std::stoi(device_text);
    }
    cudart::check_cuda(cudaSetDevice(device));
    cudart::Model model;
    model.add<cudart::Flatten>();
    model.add<cudart::Linear>(784, 128);
    model.add<cudart::ReLU>();
    model.add<cudart::Linear>(128, 10);
    cudart::Loader::load(model, weights);

    if (!benchmark) {
      const auto logits = infer(model, pixels.data(), 1);
      std::cout << std::setprecision(9) << "CudaRT MNIST\nLogits:\n";
      for (size_t i = 0; i < logits.size(); ++i) std::cout << "  " << i << ": " << logits[i] << '\n';
      std::cout << "Prediction: " << std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())) << '\n';
      return 0;
    }

    // Evaluate every supplied image; timing repeatedly uses the first full batch.
    std::vector<float> all_logits;
    all_logits.reserve(image_count * 10);
    for (size_t i = 0; i < image_count; i += batch) {
      const auto output = infer(model, pixels.data() + i * 784, std::min(batch, image_count - i));
      all_logits.insert(all_logits.end(), output.begin(), output.end());
    }
    for (size_t i = 0; i < warmup; ++i) infer(model, pixels.data(), batch);
    cudart::check_cuda(cudaDeviceSynchronize());
    Events events;
    cudart::check_cuda(cudaEventRecord(events.start));
    cudart::check_cuda(cudaEventRecord(events.end));
    cudart::check_cuda(cudaEventSynchronize(events.end));
    std::vector<float> forward_times, request_times;
    for (size_t i = 0; i < iterations; ++i) {
      float forward_ms = 0;
      infer(model, pixels.data(), batch, &events, &forward_ms);
      forward_times.push_back(forward_ms);
      // Separate timing run: no event synchronization inside the request.
      const auto start = Clock::now();
      infer(model, pixels.data(), batch);
      request_times.push_back(std::chrono::duration<float, std::milli>(Clock::now() - start).count());
    }
    cudaDeviceProp properties{};
    cudart::check_cuda(cudaGetDeviceProperties(&properties, device));
    int runtime_version = 0, driver_version = 0;
    cudart::check_cuda(cudaRuntimeGetVersion(&runtime_version));
    cudart::check_cuda(cudaDriverGetVersion(&driver_version));
    std::ofstream result(argv[4]);
    if (!result) throw std::runtime_error("Could not open result JSON");
    result << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "{\"schema_version\":1,\"build_type\":\"" << CUDART_BUILD_TYPE
           << "\",\"image_count\":" << image_count
           << ",\"batch_size\":" << batch << ",\"warmup\":" << warmup
           << ",\"iterations\":" << iterations << ",\"device_index\":" << device
           << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
           << "\",\"cuda_runtime_version\":" << runtime_version
           << ",\"cuda_driver_version\":" << driver_version << ",\"logits\":";
    write_array(result, all_logits);
    result << ",\"forward_ms\":";
    write_array(result, forward_times);
    result << ",\"request_ms\":";
    write_array(result, request_times);
    result << "}\n";
    result.close();
    if (!result) throw std::runtime_error("Failed writing result JSON");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
