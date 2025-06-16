#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/sum.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/unary_intrinsic.hpp"

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
            out_shape[i] = (ir_input_types[0]->shape[i] + 2 * 0 - ir_input_types[1]->shape[i]) / 1 + 1;
            GALOIS_ASSERT( out_shape[i] <= ir_input_types[0]->shape[i] );
        }
        
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
        auto ir_act = ir_inputs[0];
        auto ir_weight = ir_inputs[1];

        auto input_shape = ir_act->type->shape;
        auto kernel_shape = ir_weight->type->shape;
        auto output_shape = ir_output->type->shape;
        auto zero = ir_builder->GetZero(ir_output->type->DataType());
        ir_builder->ExpressCreator<op::FillCreator>({ir_output, zero});

        for (int i = 0; i < output_shape[0]; ++i) {
            for (int j = 0; j < output_shape[1]; ++j) {
                
                auto output_accessor = ir_builder->CreateAccessor(ir_output);
                output_accessor->transform_matrix.resize(0, 0);
                output_accessor->shift_vector[0] = i;
                output_accessor->shift_vector[1] = j;

                for (int ki = 0; ki < kernel_shape[0]; ++ki) {
                    for (int kj = 0; kj < kernel_shape[1]; ++kj) {
                        int input_i = i + ki;
                        int input_j = j + kj;

                        auto input_accessor = ir_builder->CreateAccessor(ir_act);
                        input_accessor->transform_matrix.resize(0, 0);
                        input_accessor->shift_vector[0] = input_i;
                        input_accessor->shift_vector[1] = input_j;
                        GALOIS_ASSERT(input_accessor->type->IsScalar());

                        auto kernel_accessor = ir_builder->CreateAccessor(ir_weight);
                        kernel_accessor->transform_matrix.resize(0, 0);
                        kernel_accessor->shift_vector[0] = ki;
                        kernel_accessor->shift_vector[1] = kj;
                        GALOIS_ASSERT(kernel_accessor->type->IsScalar());

                        auto mul_result = ir_builder->Mul(input_accessor, kernel_accessor);

                        this->SumInline(output_accessor, mul_result, ir_builder);
                    }
                }
            }
        }
    }

    void SumInline(std::shared_ptr<ir::Tensor> ir_output, std::shared_ptr<ir::Tensor> ir_input,
                       std::shared_ptr<ir::Builder> ir_builder) {
        if (ir_input->type->IsScalar()) {
            auto ir_add = ir_builder->Add(ir_input, ir_output);
            ir_builder->Write(ir_add, ir_output);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
            auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
            this->SumInline(ir_accessor, ir_output, ir_builder);
        }
    }

};
}  // namespace galois::op