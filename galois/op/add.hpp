#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/operator_creator.hpp"

namespace galois::op {

using namespace ir;

class AddCreator : public OperatorCreator {
   public:
    static std::shared_ptr<AddCreator> Create() {
        auto self = std::make_shared<AddCreator>();
        return self;
    }

    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        return ir_input_types.front();
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_out =
            ir_builder->Create<Alloca>(this->InferType({ir_inputs[0]->type, ir_inputs[1]->type}));
        {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_out->type->shape);
            auto ir_accessor_out = ir_builder->CreateIdentityAccessor(ir_out);
            auto ir_accessor_in0 = ir_builder->CreateIdentityAccessor(ir_inputs[0]);
            auto ir_accessor_in1 = ir_builder->CreateIdentityAccessor(ir_inputs[1]);
            auto ir_add = ir_builder->Create<Add>(ir_accessor_in0, ir_accessor_in1);
            ir_builder->Create<Write>(ir_add, ir_accessor_out);
        }
        ir_builder->Create<Return>(ir_out);
    }

    std::shared_ptr<TensorType> ir_tensor_type = nullptr;
};

}  // namespace galois::op
