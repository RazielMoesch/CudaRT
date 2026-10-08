#include "cudart/loader.hpp"
#include "cudart/layers/linear.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudart {

namespace {


struct Parameter {
    std::vector<size_t> shape;
    std::vector<float> data;
};

using ParameterMap =
    std::unordered_map<std::string, Parameter>;



void read_bytes(
    std::ifstream& file,
    void* destination,
    size_t count
) {
    if (count == 0) {
        return;
    }

    if (count >
        static_cast<size_t>(
            std::numeric_limits<std::streamsize>::max()
        )) {
        throw std::runtime_error(
            "Loader: read size is too large"
        );
    }

    if (!file.read(
            static_cast<char*>(destination),
            static_cast<std::streamsize>(count)
        )) {
        throw std::runtime_error(
            "Loader: unexpected end of file"
        );
    }
}



uint32_t read_u32(std::ifstream& file) {
    unsigned char bytes[4];

    read_bytes(file, bytes, 4);

    return
        uint32_t(bytes[0]) |
        (uint32_t(bytes[1]) << 8) |
        (uint32_t(bytes[2]) << 16) |
        (uint32_t(bytes[3]) << 24);
}

uint64_t read_u64(std::ifstream& file) {
    uint64_t result = 0;

    for (int i = 0; i < 8; ++i) {
        unsigned char byte;

        read_bytes(file, &byte, 1);

        result |= uint64_t(byte) << (i * 8);
    }

    return result;
}


std::pair<std::string, Parameter>
read_parameter(std::ifstream& file) {

    const uint32_t name_length = read_u32(file);

    if (name_length == 0 || name_length > 4096) {
        throw std::runtime_error(
            "Loader: invalid parameter name length"
        );
    }

    std::string name(name_length, '\0');

    read_bytes(
        file,
        name.data(),
        name_length
    );



    const uint32_t ndim = read_u32(file);

    if (ndim == 0 || ndim > 8) {
        throw std::runtime_error(
            "Loader: invalid tensor dimension count"
        );
    }

    Parameter parameter;

    parameter.shape.reserve(ndim);

    size_t numel = 1;

    for (uint32_t i = 0; i < ndim; ++i) {

        const uint64_t dimension = read_u64(file);

        if (
            dimension == 0 ||
            dimension >
                std::numeric_limits<size_t>::max()
        ) {
            throw std::runtime_error(
                "Loader: invalid tensor dimension"
            );
        }

        const size_t dim =
            static_cast<size_t>(dimension);

        if (
            numel >
            std::numeric_limits<size_t>::max() / dim
        ) {
            throw std::runtime_error(
                "Loader: tensor size overflow"
            );
        }

        numel *= dim;

        parameter.shape.push_back(dim);
    }


    if (
        numel >
        std::numeric_limits<size_t>::max() /
            sizeof(float)
    ) {
        throw std::runtime_error(
            "Loader: tensor byte size overflow"
        );
    }

    parameter.data.resize(numel);

    read_bytes(
        file,
        parameter.data.data(),
        numel * sizeof(float)
    );

    return {
        std::move(name),
        std::move(parameter)
    };
}



void validate_shape(
    const Parameter& parameter,
    const std::vector<size_t>& expected,
    const std::string& name
) {
    if (parameter.shape != expected) {
        throw std::runtime_error(
            "Loader: shape mismatch for " + name
        );
    }
}

} 

void Loader::load(
    Model& model,
    const std::string& path
) {

    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Loader: could not open " + path
        );
    }



    char magic[4];

    read_bytes(file, magic, 4);

    if (std::memcmp(magic, "CRT1", 4) != 0) {
        throw std::runtime_error(
            "Loader: invalid CRT1 file header"
        );
    }

    const uint32_t tensor_count = read_u32(file);

    if (tensor_count > 100000) {
        throw std::runtime_error(
            "Loader: unreasonable tensor count"
        );
    }


    ParameterMap parameters;

    for (uint32_t i = 0; i < tensor_count; ++i) {

        auto [name, parameter] =
            read_parameter(file);

        auto result = parameters.emplace(
            std::move(name),
            std::move(parameter)
        );

        if (!result.second) {
            throw std::runtime_error(
                "Loader: duplicate parameter name"
            );
        }
    }


    size_t linear_index = 0;
    std::unordered_set<std::string> used_names;

    for (
        size_t i = 0;
        i < model.layer_count();
        ++i
    ) {
        auto* linear =
            dynamic_cast<Linear*>(&model.layer(i));

        if (linear == nullptr) {
            continue;
        }

        const std::string prefix =
            "linear" + std::to_string(linear_index);

        const std::string weight_name =
            prefix + ".weight";

        const std::string bias_name =
            prefix + ".bias";

        const auto weight_it =
            parameters.find(weight_name);

        if (weight_it == parameters.end()) {
            throw std::runtime_error(
                "Loader: missing " + weight_name
            );
        }

        validate_shape(
            weight_it->second,
            {
                linear->out_features(),
                linear->in_features()
            },
            weight_name
        );

        used_names.insert(weight_name);

        if (linear->has_bias()) {

            const auto bias_it =
                parameters.find(bias_name);

            if (bias_it == parameters.end()) {
                throw std::runtime_error(
                    "Loader: missing " + bias_name
                );
            }

            validate_shape(
                bias_it->second,
                {linear->out_features()},
                bias_name
            );

            used_names.insert(bias_name);
        }

        ++linear_index;
    }

    if (used_names.size() != parameters.size()) {
        throw std::runtime_error(
            "Loader: unexpected parameter tensors in file"
        );
    }



    linear_index = 0;

    for (
        size_t i = 0;
        i < model.layer_count();
        ++i
    ) {
        auto* linear =
            dynamic_cast<Linear*>(&model.layer(i));

        if (linear == nullptr) {
            continue;
        }

        const std::string prefix =
            "linear" + std::to_string(linear_index);

        const auto& weights =
            parameters.at(prefix + ".weight").data;

        const float* bias = nullptr;

        if (linear->has_bias()) {
            bias = parameters.at(
                prefix + ".bias"
            ).data.data();
        }

        linear->load_parameters(
            weights.data(),
            bias
        );

        ++linear_index;
    }
}

} 
