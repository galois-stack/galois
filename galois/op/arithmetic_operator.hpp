#pragma once

#include "galois/op/binary_operator.hpp"

namespace galois::op {

template <typename Instruction>
class ArithemticCreator : public BinaryOperatorCreator {
   public:
    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type0,
        std::shared_ptr<ir::TensorType> ir_input_type1) override {
        GALOIS_ASSERT(ir_input_type0 == ir_input_type1);
        return ir_input_type0;
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input0,
                           std::shared_ptr<ir::Tensor> ir_input1,
                           std::shared_ptr<ir::Tensor> ir_output,
                           std::shared_ptr<ir::Builder> ir_builder) override {
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_accessor_out = ir_builder->CreateIdentityAccessor(ir_output);
        auto ir_accessor_in0 = ir_builder->CreateIdentityAccessor(ir_input0);
        auto ir_accessor_in1 = ir_builder->CreateIdentityAccessor(ir_input1);
        auto ir_add = ir_builder->Create<Instruction>(ir_accessor_in0, ir_accessor_in1);
        ir_builder->Create<ir::Write>(ir_add, ir_accessor_out);
    }
};

using AddCreator = ArithemticCreator<ir::Add>;
using SubCreator = ArithemticCreator<ir::Sub>;
using MulCreator = ArithemticCreator<ir::Mul>;
using DivCreator = ArithemticCreator<ir::Div>;

}  // namespace galois::op
