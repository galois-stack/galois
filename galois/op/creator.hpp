#pragma once

#include "galois/ir/ir.hpp"

namespace galois::ir {
class Builder;
}

namespace galois::op {

class Creator : public Named {
   public:
    virtual std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) = 0;
    virtual void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                         std::shared_ptr<ir::Builder> ir_builder) = 0;

    ~Creator() {}
};

}  // namespace galois::op
