#pragma once

#include "galois/ir/builder.hpp"
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

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto ir_exp = ir_builder->Express<op::UnaryInstrinsicCreator>({ir_input}, "exp");
        auto ir_exp_sum = ir_builder->Express<op::SumCreator>({ir_exp});
        auto ir_exp_sum_broadcast = ir_builder->Create<ir::Alloca>(ir_exp->type);
        ir_builder->Express<op::FillCreator>({ir_exp_sum_broadcast, ir_exp_sum});
        auto ir_softmax = ir_builder->Express<op::DivCreator>({ir_exp, ir_exp_sum_broadcast});
        ir_builder->Create<ir::Return>(ir_softmax);
    }
};

}  // namespace galois::op
