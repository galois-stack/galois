#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/unary.hpp"

namespace galois::op {

class PackCreator : public UnaryCreator {
   public:
    PackCreator() = default;

    static std::shared_ptr<PackCreator> Create(std::shared_ptr<ir::TensorType> ir_pack_type) {
        std::shared_ptr<PackCreator> self(new PackCreator);
        self->pack_type = ir_pack_type;
        self->name = "Pack";
        self->fullname = self->name;
        return self;
    }

    PackCreator(std::shared_ptr<ir::TensorType> ir_pack_type) { this->pack_type = ir_pack_type; }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) override {
        return pack_type;
    };

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input, std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_normalize_shape = ir_output->type->NormalizeShape();
        GALOIS_ASSERT(ir_input->type->shape == ir_output_normalize_shape);

        if (ir_input->type->IsScalar()) {
            ir_builder->Create<ir::Write>(ir_input, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_output_block = ir_builder->CreateIdentityAccessor(ir_output);
        if (ir_output_block->type->IsScalar()) {
            auto ir_input_block = ir_builder->CreateIdentityAccessor(ir_input);
            this->ExpressInline(ir_input_block, ir_output_block, ir_builder);
        } else {
            auto output_block_normalize_shape = ir_output_block->type->NormalizeShape();
            auto ir_input_block_origin = ir_builder->CreateIdentityAccessor(ir_input);
            ir_input_block_origin->transform_matrix.diagonal().array() *=
                output_block_normalize_shape.array();
            auto ir_input_block = ir_builder->Create<ir::SliceView>(ir_input_block_origin,
                                                                    output_block_normalize_shape);
            this->ExpressInline(ir_input_block, ir_output_block, ir_builder);
        }
    }

    std::shared_ptr<ir::TensorType> pack_type;
};

}  // namespace galois::op
