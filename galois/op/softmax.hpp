#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/view.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/sum.hpp"
#include "galois/op/unary_intrinsic.hpp"

namespace galois::op {

class SoftmaxCreator : public op::Creator {
   public:
    static std::shared_ptr<SoftmaxCreator> Create() {
        auto self = std::make_shared<SoftmaxCreator>();
        self->name = "Softmax";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return ir_input_types.front();
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto ir_exp = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_input}, "exp");
        auto ir_exp_sum = ir_builder->ExpressCreator<op::SumCreator>({ir_exp});
        auto ir_exp_sum_broadcast_view =
            ir_builder->Create<ir::view::BroadCast>(ir_exp_sum, ir_exp->type->shape);
        auto ir_softmax =
            ir_builder->ExpressCreator<op::DivCreator>({ir_exp, ir_exp_sum_broadcast_view});
        ir_builder->Return(ir_softmax);
    }
};

}  // namespace galois::op
