#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/operator_creator.hpp"

namespace galois::op {

using namespace ir;

class FullCreator : public OperatorCreator {
   public:
    static std::shared_ptr<FullCreator> Create(std::shared_ptr<TensorType> ir_tensor_type) {
        auto self = std::make_shared<FullCreator>();
        self->ir_tensor_type = ir_tensor_type;
        return self;
    }

    std::shared_ptr<TensorType> InferType(std::vector<std::shared_ptr<TensorType>>) override {
        return ir_tensor_type;
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<Alloca>(this->InferType({}));
        {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
            auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_output);
            auto ir_value = ir_builder->Create<ConstantFloat>(FloatType::Create(32), 1.0);
            ir_builder->Create<Write>(ir_value, ir_accessor);
        }
        ir_builder->Create<Return>(ir_output);
    }

    std::shared_ptr<TensorType> ir_tensor_type = nullptr;
};

}  // namespace galois::op
