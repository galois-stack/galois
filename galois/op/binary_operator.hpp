#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

using namespace ir;

class SetZeroCreator : public OperatorCreator {
   public:
    static std::shared_ptr<SetZeroCreator> Create() { return std::make_shared<SetZeroCreator>(); }

    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return ir_input_types.front();
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
        if (ir_accessor->type->IsScalar()) {
            auto ir_zero = ir_builder->Create<ConstantFloat>(FloatType::Create(32), 0.0);
            ir_builder->Create<Write>(ir_zero, ir_accessor);
        } else {
            this->AffineExpress({ir_accessor}, ir_builder);
        }
    }
};

}  // namespace galois::op
