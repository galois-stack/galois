#pragma once

#include "galois/op/unary.hpp"

namespace galois::op {

class UnaryInstrinsicCreator : public UnaryCreator {
   public:
    static std::shared_ptr<UnaryInstrinsicCreator> Create(std::string intrinsic_name) {
        auto self = std::make_shared<UnaryInstrinsicCreator>();
        self->intrinsic_name = intrinsic_name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) override {
        return ir_input_type;
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input,
                           std::shared_ptr<ir::Tensor> ir_output,
                           std::shared_ptr<ir::Builder> ir_builder) override {
        if (ir_input->type->IsScalar()) {
            auto ir_value = ir_builder->Create<ir::UnaryIntrinsic>(this->intrinsic_name, ir_input);
            ir_builder->Create<ir::Write>(ir_value, ir_output);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
            auto ir_input_accessor = ir_builder->CreateIdentityAccessor(ir_input);
            auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_output);
            this->AffineExpressImpl(ir_input_accessor, ir_output_accessor, ir_builder);
        }
    }

   private:
    std::string intrinsic_name;
};

}  // namespace galois::op
