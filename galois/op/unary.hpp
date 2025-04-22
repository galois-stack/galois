
#pragma once

#include "galois/ir/builder.hpp"

namespace galois::op {

class UnaryCreator : public Creator {
   public:
    virtual std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) = 0;

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return this->InferTypeImpl(ir_input_types.front());
    }

    virtual void ExpressInline(std::shared_ptr<ir::Tensor> ir_input,
                               std::shared_ptr<ir::Tensor> ir_output,
                               std::shared_ptr<ir::Builder> ir_builder) = 0;

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<ir::Alloca>(this->InferTypeImpl(ir_inputs[0]->type));
        this->ExpressInline(ir_inputs[0], ir_output, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }
};

}  // namespace galois::op
