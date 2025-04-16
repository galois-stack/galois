#pragma once

#include "galois/ir/builder.hpp"

namespace galois::op {

class FillCreator : public op::Creator {
   public:
    static std::shared_ptr<FillCreator> Create() {
        auto self = std::make_shared<FillCreator>();
        self->name = "Fill";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>>) override {
        return ir::void_;
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        this->AffineExpressImpl(ir_inputs.front(), ir_inputs.back(), ir_builder);
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_ts, std::shared_ptr<ir::Tensor> ir_value,
                           std::shared_ptr<ir::Builder> ir_builder) {
        if (ir_ts->type->IsScalar()) {
            ir_builder->Create<ir::Write>(ir_value, ir_ts);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_ts->type->shape);
            auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_ts);
            this->AffineExpressImpl(ir_accessor, ir_value, ir_builder);
        }
    }
};

}  // namespace galois::op
