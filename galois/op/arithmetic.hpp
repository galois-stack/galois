#pragma once

#include "galois/op/binary.hpp"

namespace galois::op {

template <ir::ArithmeticInstruction::Operation Operation>
class ArithmeticCreator : public BinaryCreator {
   public:
    static std::shared_ptr<ArithmeticCreator> Create() {
        auto self = std::make_shared<ArithmeticCreator>();
        self->name = "ToName ";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type0,
        std::shared_ptr<ir::TensorType> ir_input_type1) override {
        GALOIS_ASSERT(ir_input_type0 == ir_input_type1);
        return ir_input_type0;
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input0, std::shared_ptr<ir::Tensor> ir_input1,
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        if (ir_input0->type->IsScalar() && ir_input1->type->IsScalar()) {
            auto ir_re =
                ir_builder->Create<ir::ArithmeticInstruction>(Operation, ir_input0, ir_input1);
            ir_builder->Create<ir::Write>(ir_re, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_accessor_out = ir_builder->CreateIdentityAccessor(ir_output);
        auto ir_accessor_in0 = ir_builder->CreateIdentityAccessor(ir_input0);
        auto ir_accessor_in1 = ir_builder->CreateIdentityAccessor(ir_input1);

        this->ExpressInline(ir_accessor_in0, ir_accessor_in1, ir_accessor_out, ir_builder);
    }
};

using AddCreator = ArithmeticCreator<ir::ArithmeticInstruction::Add>;
using SubCreator = ArithmeticCreator<ir::ArithmeticInstruction::Sub>;
using MulCreator = ArithmeticCreator<ir::ArithmeticInstruction::Mul>;
using DivCreator = ArithmeticCreator<ir::ArithmeticInstruction::Div>;

}  // namespace galois::op
