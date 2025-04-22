#pragma once

#include "galois/ir/builder.hpp"

namespace galois::op {

class CopyCreator : public op::Creator {
   public:
    static std::shared_ptr<CopyCreator> Create() {
        auto self = std::make_shared<CopyCreator>();
        self->name = "Copy";
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

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_src, std::shared_ptr<ir::Tensor> ir_dst,
                           std::shared_ptr<ir::Builder> ir_builder) {
        if (ir_src->type->IsScalar()) {
            ir_builder->Create<ir::Write>(ir_src, ir_dst);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_src->type->shape);
            auto ir_accessor_src = ir_builder->CreateIdentityAccessor(ir_src);
            auto ir_accessor_dst = ir_builder->CreateIdentityAccessor(ir_dst);
            this->AffineExpressImpl(ir_accessor_src, ir_accessor_dst, ir_builder);
        }
    }
};

}  // namespace galois::op
