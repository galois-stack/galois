#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

class UnpackCreator : public OperatorCreator {
   public:
    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        auto ir_scalar_type = ir_input_types.front()->ScalarType();
        auto shape = ir_input_types.front()->NormalizeShape();
        return TensorType::Create(ir_scalar_type, shape);
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto ir_output = ir_outputs.front();
        auto ir_input_normalize_shape = ir_input->type->NormalizeShape();
        GALOIS_ASSERT(ir_output->type->shape == ir_input_normalize_shape);

        if (ir_input->type->IsScalar()) {
            ir_builder->Create<Write>(ir_input, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_input_block = ir_builder->CreateIdentityAccessor(ir_input);
        if (ir_input_block->type->IsScalar()) {
            auto ir_output_block = ir_builder->CreateIdentityAccessor(ir_output);
            this->AffineExpress({ir_input_block}, {ir_output_block}, ir_builder);
        } else {
            auto input_block_normalize_shape = ir_input_block->type->NormalizeShape();
            auto ir_output_block_origin = ir_builder->CreateIdentityAccessor(ir_output);
            ir_output_block_origin->transform_matrix.diagonal().array() *=
                input_block_normalize_shape.array();
            auto ir_output_block =
                ir_builder->Create<Slice>(ir_output_block_origin, input_block_normalize_shape);
            this->AffineExpress({ir_input_block}, {ir_output_block}, ir_builder);
        }
    }
};

}  // namespace galois::op
