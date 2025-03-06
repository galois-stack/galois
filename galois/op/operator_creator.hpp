#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

class OperatorCreator : public Named {
   public:
    virtual std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) = 0;
    virtual void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                               std::shared_ptr<ir::Builder> ir_builder) = 0;

    ~OperatorCreator() {}
};

class SetZeroCreator : public OperatorCreator {
   public:
    static std::shared_ptr<SetZeroCreator> Create() { return std::make_shared<SetZeroCreator>(); }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return ir_input_types.front();
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
        if (ir_accessor->type->IsScalar()) {
            auto ir_zero = ir_builder->GetZero(ir_accessor->type);
            ir_builder->Create<ir::Write>(ir_zero, ir_accessor);
        } else {
            this->AffineExpress({ir_accessor}, ir_builder);
        }
    }
};

class UnaryOperatorCreator : public OperatorCreator {
   public:
    virtual std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) = 0;

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return this->InferTypeImpl(ir_input_types.front());
    }

    virtual void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input,
                                   std::shared_ptr<ir::Tensor> ir_output,
                                   std::shared_ptr<ir::Builder> ir_builder) = 0;

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<ir::Alloca>(this->InferTypeImpl(ir_inputs[0]->type));
        this->AffineExpressImpl(ir_inputs[0], ir_output, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }
};

}  // namespace galois::op
