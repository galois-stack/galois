#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

using namespace ir;

class AddCreator : public OperatorCreator {
   public:
    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 2);
        return {ir_input_types[0]};
    };

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        if (ir_inputs[0]->type->IsScalar()) {
            auto ir_add = ir_builder->Create<ir::Add>(ir_inputs[0], ir_inputs[1]);
            ir_builder->Create<Write>(ir_add, ir_outputs[0]);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_inputs[0]->type->shape);
        auto ir_input_accessor0 = ir_builder->CreateIdentityAccessor(ir_inputs[0]);
        auto ir_input_accessor1 = ir_builder->CreateIdentityAccessor(ir_inputs[1]);
        auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_outputs[0]);
        this->AffineExpress({ir_input_accessor0, ir_input_accessor1}, {ir_output_accessor},
                            ir_builder);
    }
};

class SetZeroCreator : public OperatorCreator {
   public:
    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return ir_input_types.front();
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
        if (ir_accessor->type->IsScalar()) {
            auto ir_zero = ir_builder->Create<ConstantFloat>(FloatType::Create(32), 0.0);
            ir_builder->Create<Write>(ir_zero, ir_accessor);
        } else {
            this->AffineExpress({ir_accessor}, {}, ir_builder);
        }
    }
};

}  // namespace galois::op
