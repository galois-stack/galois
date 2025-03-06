#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/operator_creator.hpp"

namespace galois::op {

class FullCreator : public op::OperatorCreator {
   public:
    static std::shared_ptr<FullCreator> Create(std::shared_ptr<ir::TensorType> ir_tensor_type) {
        auto self = std::make_shared<FullCreator>();
        self->ir_tensor_type = ir_tensor_type;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>>) override {
        return ir_tensor_type;
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<ir::Alloca>(this->InferType({}));
        this->AffineExpressImpl(ir_output, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_ts,
                           std::shared_ptr<ir::Builder> ir_builder) {
        if (ir_ts->type->IsScalar()) {
            auto ir_zero = ir_builder->Create<ir::ConstantFloat>(ir::FloatType::Create(32), 0.0);
            ir_builder->Create<ir::Write>(ir_zero, ir_ts);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_ts->type->shape);
            auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_ts);
            this->AffineExpressImpl({ir_accessor}, ir_builder);
        }
    }

    std::shared_ptr<ir::TensorType> ir_tensor_type = nullptr;
};

}  // namespace galois::op
