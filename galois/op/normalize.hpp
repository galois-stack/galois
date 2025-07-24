#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/shape.hpp"
#include "galois/op/reduce_prod.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/sum.hpp"
#include "galois/op/unary_intrinsic.hpp"

namespace galois::op {
class NormalizeCreator : public op::Creator {
   public:
    static std::shared_ptr<NormalizeCreator> Create() {
        auto self = std::make_shared<NormalizeCreator>();
        self->name = "Normalize";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());
        auto base_type = Cast<ir::TensorType>(ir_input_types.front());
        auto out_shape = base_type->shape;
        return ir::TensorType::Create(base_type->value_type, out_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);
        this->ExpressInline(ir_inputs, ir_output, ir_builder);
        ir_builder->Return(ir_output);
    }

    void ExpressInline(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                  std::shared_ptr<ir::Tensor> ir_output, std::shared_ptr<ir::Builder> ir_builder) {
        auto total_shape = ir_builder->ExpressCreator<op::ShapeCreator>({ir_inputs[0]});
        auto total_size = ir_builder->ExpressCreator<op::ReduceProdCreator>({total_shape});

        auto ir_total_size = ir_builder->Alloca(ir_inputs[0]->type);
        ir_builder->ExpressCreator<op::FillCreator>({ir_total_size, total_size});

        auto ir_sum = ir_builder->ExpressCreator<op::SumCreator>({ir_inputs[0]});
        auto ir_sum_broadcast =
            ir_builder->Create<ir::view::Broadcast>(ir_sum, ir_inputs[0]->type->shape);

        auto ir_mean =
            ir_builder->ExpressCreator<op::DivCreator>({ir_sum_broadcast, ir_total_size});
        auto ir_sub = ir_builder->ExpressCreator<op::SubCreator>({ir_inputs[0], ir_mean});
        auto ir_pow = ir_builder->ExpressCreator<op::MulCreator>({ir_sub, ir_sub});
        ir_pow = ir_builder->ExpressCreator<op::DivCreator>({ir_pow, ir_total_size});

        auto ir_pow_sum = ir_builder->ExpressCreator<op::SumCreator>({ir_pow});
        auto ir_pow_sum_broadcast =
            ir_builder->Create<ir::view::Broadcast>(ir_pow_sum, ir_inputs[0]->type->shape);

        auto ir_epsilon = ir_builder->Alloca(ir_inputs[0]->type);
        ir_builder->ExpressCreator<op::FillCreator>(
            {ir_epsilon, ir_builder->GetConstant(ir_inputs[0]->type->DataType(), 1e-5f)});
        auto ir_pow_sum_epsilon =
            ir_builder->ExpressCreator<op::AddCreator>({ir_pow_sum_broadcast, ir_epsilon});
        auto ir_std =
            ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_pow_sum_epsilon}, "sqrt");
        auto ir_standard = ir_builder->ExpressCreator<op::DivCreator>({ir_sub, ir_std});

        auto ir_gama_broadcast =
            ir_builder->Create<ir::view::Broadcast>(ir_inputs[1], ir_inputs[0]->type->shape);
        auto ir_beta_broadcast =
            ir_builder->Create<ir::view::Broadcast>(ir_inputs[2], ir_inputs[0]->type->shape);

        auto result_1 =
            ir_builder->ExpressCreator<op::MulCreator>({ir_standard, ir_gama_broadcast});
        auto result_2 = ir_builder->ExpressCreator<op::AddCreator>({result_1, ir_beta_broadcast});
        ir_builder->Write(result_2, ir_output);
    }
};
}  // namespace galois::op