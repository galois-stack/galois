#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/full.hpp"
#include "galois/op/operator_creator.hpp"

namespace galois::op {

class BinaryOperatorCreator : public OperatorCreator {
   public:
    virtual std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type0,
        std::shared_ptr<ir::TensorType> ir_input_type1) = 0;

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return this->InferTypeImpl(ir_input_types.front(), ir_input_types.back());
    }

    virtual void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input0,
                                   std::shared_ptr<ir::Tensor> ir_input1,
                                   std::shared_ptr<ir::Tensor> ir_output,
                                   std::shared_ptr<ir::Builder> ir_builder) = 0;

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        // auto ir_output = ir_builder->Create<ir::Alloca>(
        //     this->InferTypeImpl(ir_inputs[0]->type, ir_inputs[1]->type));
        // set_zero_creator->AffineExpress({ir_output}, ir_builder);
        auto ir_output_type = this->InferTypeImpl(ir_inputs[0]->type, ir_inputs[1]->type);
        auto ir_output = ir_builder->Express<op::FullCreator>({}, ir_output_type);
        this->AffineExpressImpl(ir_inputs[0], ir_inputs[1], ir_output, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }

   private:
    std::shared_ptr<SetZeroCreator> set_zero_creator = SetZeroCreator::Create();
};

}  // namespace galois::op
