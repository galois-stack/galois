#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

class FullCreator : public op::Creator {
   public:
    static std::shared_ptr<FullCreator> Create(std::shared_ptr<ir::TensorType> ir_tensor_type) {
        auto self = std::make_shared<FullCreator>();
        self->ir_tensor_type = ir_tensor_type;
        self->name = "Full";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>>) override {
        return ir_tensor_type;
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_value = ir_inputs.front();
        auto ir_output = ir_builder->Create<ir::Alloca>(this->InferType({}));
        ir_builder->ExpressCreator<op::FillCreator>({ir_output, ir_value});
        ir_builder->Create<ir::Return>(ir_output);
    }

    std::shared_ptr<ir::TensorType> ir_tensor_type = nullptr;
};

}  // namespace galois::op
