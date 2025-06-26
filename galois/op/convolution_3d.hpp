#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/sum.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

class Convolution3DCreator : public op::Creator {
private:
    int stride_h = 1;
    int stride_w = 1;

public:
    static std::shared_ptr<Convolution3DCreator> Create(int stride_h = 1, int stride_w = 1) {
        auto self = std::make_shared<Convolution3DCreator>();
        self->name = "Convolution3D";
        self->fullname = self->name;
        self->stride_h = stride_h;
        self->stride_w = stride_w;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 2);
        auto input_type = Cast<ir::TensorType>(ir_input_types[0]);
        auto weight_type = Cast<ir::TensorType>(ir_input_types[1]);

        GALOIS_ASSERT(input_type->shape.size() == 3);
        GALOIS_ASSERT(weight_type->shape.size() == 4);
        
        GALOIS_ASSERT(input_type->shape[0] == weight_type->shape[1]);

        auto input_h = input_type->shape[1];
        auto input_w = input_type->shape[2];
        auto kernel_h = weight_type->shape[2];
        auto kernel_w = weight_type->shape[3];
        auto out_channels = weight_type->shape[0];
        
        auto out_h = (input_h - kernel_h) / stride_h + 1;
        auto out_w = (input_w - kernel_w) / stride_w + 1;

        GALOIS_ASSERT(out_h > 0 && out_w > 0);
        
        Eigen::VectorXi64 out_shape(3);
        out_shape[0] = out_channels;
        out_shape[1] = out_h;
        out_shape[2] = out_w;
        
        return ir::TensorType::Create(input_type->value_type, out_shape);
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
                std::shared_ptr<ir::Tensor> ir_output, std::shared_ptr<ir::Builder> ir_builder) {
        auto ir_input = ir_inputs[0];   // [C_in, H, W]
        auto ir_weight = ir_inputs[1];  // [C_out, C_in, KH, KW]

        auto input_shape = ir_input->type->shape;   // [C_in, H, W]
        auto weight_shape = ir_weight->type->shape; // [C_out, C_in, KH, KW]
        auto output_shape = ir_output->type->shape; // [C_out, H_out, W_out]

        // Create 6D grid: [out_c, out_h, out_w, kernel_h, kernel_w, in_c]
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(Eigen::Vector<int64_t, 6>(
            output_shape[0], output_shape[1], output_shape[2], 
            weight_shape[2], weight_shape[3], weight_shape[1]));

        // Output accessor: maps (out_c, out_h, out_w, *, *, *) -> (out_c, out_h, out_w)
        auto output_accessor = ir_builder->CreateAccessor(ir_output);
        output_accessor->transform_matrix(0, 0) = 1; // out_c
        output_accessor->transform_matrix(1, 1) = 1; // out_h
        output_accessor->transform_matrix(2, 2) = 1; // out_w

        // Input accessor with stride support: maps grid -> input[C, H, W]
        auto input_accessor = ir_builder->CreateAccessor(ir_input);
        input_accessor->transform_matrix(0, 5) = 1;        // in_c -> C
        input_accessor->transform_matrix(1, 1) = stride_h; // out_h * stride_h -> H
        input_accessor->transform_matrix(1, 3) = 1;        // kernel_h -> H
        input_accessor->transform_matrix(2, 2) = stride_w; // out_w * stride_w -> W  
        input_accessor->transform_matrix(2, 4) = 1;        // kernel_w -> W

        // Weight accessor: maps (out_c, *, *, kernel_h, kernel_w, in_c) -> (out_c, in_c, kernel_h, kernel_w)
        auto weight_accessor = ir_builder->CreateAccessor(ir_weight);
        weight_accessor->transform_matrix(0, 0) = 1; // out_c
        weight_accessor->transform_matrix(1, 5) = 1; // in_c
        weight_accessor->transform_matrix(2, 3) = 1; // kernel_h
        weight_accessor->transform_matrix(3, 4) = 1; // kernel_w

        if(output_accessor->type->IsScalar()){
            auto mul_result = ir_builder->Mul(input_accessor, weight_accessor);
            auto add_result = ir_builder->Add(output_accessor, mul_result);
            ir_builder->Write(add_result, output_accessor);
            return;
        }

        this->ExpressInline({input_accessor, weight_accessor}, output_accessor, ir_builder);
    }
};

}  // namespace galois::op
