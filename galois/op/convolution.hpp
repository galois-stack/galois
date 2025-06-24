#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/sum.hpp"

namespace galois::op {
class ConvolutionCreator : public op::Creator {
   public:
    static std::shared_ptr<ConvolutionCreator> Create() {
        auto self = std::make_shared<ConvolutionCreator>();
        self->name = "Convolution";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());
        auto base_type = Cast<ir::TensorType>(ir_input_types.front());
        auto out_shape = base_type->shape;
        int64_t total = 0;
        for (size_t i = 0; i < out_shape.size(); ++i) {
            // out_w = (in_w + 2*padding - kernel_size) / stride + 1
            out_shape[i] = (ir_input_types[0]->shape[i] - ir_input_types[1]->shape[i]) + 1;
            GALOIS_ASSERT(out_shape[i] <= ir_input_types[0]->shape[i]);
        }
        GALOIS_ASSERT(out_shape.size() == 2);

        return ir::TensorType::Create(base_type->value_type, out_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);
        auto zero = ir_builder->GetZero(ir_output->type->DataType());
        ir_builder->ExpressCreator<op::FillCreator>({ir_output, zero});
        this->ExpressInline(ir_inputs, ir_output, ir_builder);
        ir_builder->Return(ir_output);
    }

    void ExpressInline(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) {
        auto ir_act = ir_inputs[0];
        auto ir_weight = ir_inputs[1];

        auto input_shape = ir_act->type->shape;
        auto kernel_shape = ir_weight->type->shape;
        auto output_shape = ir_output->type->shape;

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(
            Eigen::Vector4i64(output_shape[0], output_shape[1], kernel_shape[0], kernel_shape[1]));

        auto output_accessor = ir_builder->CreateAccessor(ir_output);
        output_accessor->transform_matrix(0, 0) = 1;
        output_accessor->transform_matrix(1, 1) = 1;

        auto input_accessor = ir_builder->CreateAccessor(ir_act);
        input_accessor->transform_matrix(0, 0) = 1;
        input_accessor->transform_matrix(0, 2) = 1;
        input_accessor->transform_matrix(1, 1) = 1;
        input_accessor->transform_matrix(1, 3) = 1;

        auto kernel_accessor = ir_builder->CreateAccessor(ir_weight);
        kernel_accessor->transform_matrix(0, 2) = 1;
        kernel_accessor->transform_matrix(1, 3) = 1;

        if (output_accessor->type->IsScalar()) {
            auto mul_result = ir_builder->Mul(input_accessor, kernel_accessor);
            auto add_result = ir_builder->Add(output_accessor, mul_result);
            ir_builder->Write(add_result, output_accessor);
            return;
        }

        this->ExpressInline({input_accessor, kernel_accessor}, output_accessor, ir_builder);
    }
};
}  // namespace galois::op