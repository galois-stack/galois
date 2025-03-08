#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/fill.hpp"
#include "galois/op/operator_creator.hpp"

namespace galois::op {

class FullCreator : public op::OperatorCreator {
   public:
    static std::shared_ptr<FullCreator> Create(std::shared_ptr<ir::TensorType> ir_tensor_type,
                                               double value) {
        auto self = std::make_shared<FullCreator>();
        self->ir_tensor_type = ir_tensor_type;
        self->value = value;
        self->fill_creator = FillCreator::Create(ir_tensor_type, value);
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>>) override {
        return ir_tensor_type;
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<ir::Alloca>(this->InferType({}));
        this->fill_creator->AffineExpress({ir_output}, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }

    std::shared_ptr<ir::TensorType> ir_tensor_type = nullptr;
    std::shared_ptr<FillCreator> fill_creator = nullptr;
    double value = 0;
};

}  // namespace galois::op
