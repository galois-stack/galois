#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

class PackKernel : public Kernel {
   public:
    bool Match(std::vector<std::shared_ptr<Tensor>> ir_inputs,
               std::vector<std::shared_ptr<Tensor>> ir_outputs,
               std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs[0];
        auto ir_output = ir_outputs[0];
        if (ir_output->type->value_type == FloatType::Create(32) &&
            ir_input->type->value_type == FloatType::Create(32)) {
            return true;
        }
        return false;
    }

    void Build(std::vector<std::shared_ptr<Tensor>> ir_inputs,
               std::vector<std::shared_ptr<Tensor>> ir_outputs,
               std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs[0];
        auto ir_output = ir_outputs[0];

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_accessor_input = ir_builder->CreateIdentityAccessor(ir_input);
        auto ir_accessor_output = ir_builder->CreateIdentityAccessor(ir_output);
        ir_builder->Create<Write>(ir_accessor_input, ir_accessor_output);
    }
};

class PackCreator : public OperatorCreator {
   public:
    static std::shared_ptr<PackCreator> Create(std::shared_ptr<TensorType> ir_pack_type) {
        std::shared_ptr<PackCreator> self(new PackCreator);
        self->pack_type = ir_pack_type;
        return self;
    }

    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        return pack_type;
    };

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto ir_output = ir_outputs.front();
        auto ir_output_normalize_shape = ir_output->type->NormalizeShape();
        GALOIS_ASSERT(ir_input->type->shape == ir_output_normalize_shape);

        if (ir_input->type->IsScalar()) {
            ir_builder->Create<Write>(ir_input, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_output_block = ir_builder->CreateIdentityAccessor(ir_output);
        if (ir_output_block->type->IsScalar()) {
            auto ir_input_block = ir_builder->CreateIdentityAccessor(ir_input);
            this->AffineExpress({ir_input_block}, {ir_output_block}, ir_builder);
        } else {
            auto output_block_normalize_shape = ir_output_block->type->NormalizeShape();
            auto ir_input_block_origin = ir_builder->CreateIdentityAccessor(ir_input);
            ir_input_block_origin->transform_matrix.diagonal().array() *=
                output_block_normalize_shape.array();
            auto ir_input_block =
                ir_builder->Create<Slice>(ir_input_block_origin, output_block_normalize_shape);
            this->AffineExpress({ir_input_block}, {ir_output_block}, ir_builder);
        }
    }

    std::shared_ptr<TensorType> pack_type;
};

}  // namespace galois::op
