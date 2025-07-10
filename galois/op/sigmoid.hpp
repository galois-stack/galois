#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/unary.hpp"

namespace galois::op {

class SigmoidCreator : public UnaryCreator {
   public:
    static std::shared_ptr<SigmoidCreator> Create() {
        auto self = std::make_shared<SigmoidCreator>();
        self->name = "Sigmoid";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) override {
        return ir_input_type;
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input, std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        if (ir_input->type->IsScalar()) {
            // sigmoid(x) = 1 / (1 + exp(-x))
            auto one = ir_builder->GetConstant(ir_input->type, 1.0);
            auto neg_x = ir_builder->Sub(ir_builder->GetConstant(ir_input->type, 0.0), 
                                        ir_input);
            auto exp_neg_x = ir_builder->Create<ir::UnaryIntrinsic>("exp", neg_x);
            auto denom = ir_builder->Add(one, exp_neg_x);
            auto result = ir_builder->Div(one, denom);

            ir_builder->Write(result, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_output_block = ir_builder->CreateIdentityAccessor(ir_output);
        auto ir_input_block = ir_builder->CreateIdentityAccessor(ir_input);
        this->ExpressInline(ir_input_block, ir_output_block, ir_builder);
    }
};

}  // namespace galois::op
