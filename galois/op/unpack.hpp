#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/unary.hpp"

namespace galois::op {

class UnpackCreator : public UnaryCreator {
   public:
    static std::shared_ptr<UnpackCreator> Create() {
        auto self = std::make_shared<UnpackCreator>();
        self->name = "Unpack";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) override {
        auto ir_scalar_type = ir_input_type->DataType();
        auto shape = ir_input_type->NormalizeShape();
        return ir::TensorType::Create(ir_scalar_type, shape);
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input,
                           std::shared_ptr<ir::Tensor> ir_output,
                           std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_input_normalize_shape = ir_input->type->NormalizeShape();
        GALOIS_ASSERT(ir_output->type->shape == ir_input_normalize_shape);

        if (ir_input->type->IsScalar()) {
            ir_builder->Create<ir::Write>(ir_input, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_input_block = ir_builder->CreateIdentityAccessor(ir_input);
        if (ir_input_block->type->IsScalar()) {
            auto ir_output_block = ir_builder->CreateIdentityAccessor(ir_output);
            this->AffineExpressImpl(ir_input_block, ir_output_block, ir_builder);
        } else {
            auto input_block_normalize_shape = ir_input_block->type->NormalizeShape();
            auto ir_output_block_origin = ir_builder->CreateIdentityAccessor(ir_output);
            ir_output_block_origin->transform_matrix.diagonal().array() *=
                input_block_normalize_shape.array();
            auto ir_output_block = ir_builder->Create<ir::SliceView>(ir_output_block_origin,
                                                                     input_block_normalize_shape);
            this->AffineExpressImpl(ir_input_block, ir_output_block, ir_builder);
        }
    }
};

}  // namespace galois::op
