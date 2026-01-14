#pragma once

#include <memory>
#include <vector>

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Tensor;

inline std::vector<std::shared_ptr<TensorType>> GetTensorTypes(
    std::vector<std::shared_ptr<Tensor>> ir_tensors) {
    std::vector<std::shared_ptr<TensorType>> ir_types;
    for (auto ir_tensor : ir_tensors) {
        ir_types.push_back(ir_tensor->type);
    }
    return ir_types;
}

}  // namespace galois::ir
