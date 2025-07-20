#pragma once

#include "galois/assert.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

template <int N>
class ConvolutionNDCreator : public op::Creator {
   private:
    std::array<int, N> strides_;

   public:
    static std::shared_ptr<ConvolutionNDCreator<N>> Create(const std::array<int, N>& strides = {}) {
        auto self = std::make_shared<ConvolutionNDCreator<N>>();
        self->name = "Convolution" + std::to_string(N) + "D";
        self->fullname = self->name;

        self->strides_ = strides;

        // Set defaults if not provided
        for (int i = 0; i < N; ++i) {
            if (self->strides_[i] == 0) self->strides_[i] = 1;
        }

        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 2);
        auto input_type = Cast<ir::TensorType>(ir_input_types[0]);
        auto weight_type = Cast<ir::TensorType>(ir_input_types[1]);

        // Input format: [C_in, D1, D2, ..., DN]
        // Weight format: [C_out, C_in, K1, K2, ..., KN]
        GALOIS_ASSERT(input_type->shape.size() == N + 1);
        GALOIS_ASSERT(weight_type->shape.size() == N + 2);
        GALOIS_ASSERT(input_type->shape[0] ==
                      weight_type->shape[1]);  // input_channels == weight_input_channels

        auto output_channels = weight_type->shape[0];
        Eigen::VectorXi64 output_shape(N + 1);
        output_shape[0] = output_channels;

        // Calculate output spatial dimensions
        for (int i = 0; i < N; ++i) {
            auto input_dim = input_type->shape[i + 1];
            auto kernel_dim = weight_type->shape[i + 2];
            auto output_dim = (input_dim - kernel_dim) / strides_[i] + 1;
            GALOIS_ASSERT(output_dim > 0);
            output_shape[i + 1] = output_dim;
        }

        return ir::TensorType::Create(input_type->value_type, output_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);
        ir_builder->ExpressCreator<op::FillCreator>(
            {ir_output, ir_builder->GetZero(ir_output->type->DataType())});
        this->ExpressInline(ir_inputs, ir_output, ir_builder);
        ir_builder->Return(ir_output);
    }

    void ExpressInline(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) {
        auto ir_input = ir_inputs[0];   // [C_in, D1, D2, ..., DN]
        auto ir_weight = ir_inputs[1];  // [C_out, C_in, K1, K2, ..., KN]

        auto input_shape = ir_input->type->shape;
        auto weight_shape = ir_weight->type->shape;
        auto output_shape = ir_output->type->shape;

        // Create (2N+2)-dimensional grid: [out_c, out_d1, out_d2, ..., out_dn, k1, k2, ..., kn,
        // in_c]
        Eigen::VectorXi64 grid_shape(2 * N + 2);

        // Output spatial dimensions
        grid_shape[0] = output_shape[0];  // output_channels
        for (int i = 0; i < N; ++i) {
            grid_shape[i + 1] = output_shape[i + 1];  // output spatial dims
        }

        // Kernel dimensions
        for (int i = 0; i < N; ++i) {
            grid_shape[N + 1 + i] = weight_shape[i + 2];  // kernel spatial dims
        }
        grid_shape[2 * N + 1] = weight_shape[1];  // input_channels

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(grid_shape);

        // Output accessor
        auto output_accessor = ir_builder->CreateAccessor(ir_output);
        output_accessor->transform_matrix(0, 0) = 1;  // out_c
        for (int i = 0; i < N; ++i) {
            output_accessor->transform_matrix(i + 1, i + 1) = 1;  // output spatial dims
        }

        auto input_accessor = ir_builder->CreateAccessor(ir_input);
        input_accessor->transform_matrix(0, 2 * N + 1) = 1;  // in_c mapping
        for (int i = 0; i < N; ++i) {
            // Map output spatial position to input spatial position: out_d * stride + k
            input_accessor->transform_matrix(i + 1, i + 1) = strides_[i];  // out_d * stride
            input_accessor->transform_matrix(i + 1, N + 1 + i) = 1;  // k (no dilation for now)
        }

        // Weight accessor
        auto weight_accessor = ir_builder->CreateAccessor(ir_weight);
        weight_accessor->transform_matrix(0, 0) = 1;          // out_c
        weight_accessor->transform_matrix(1, 2 * N + 1) = 1;  // in_c
        for (int i = 0; i < N; ++i) {
            weight_accessor->transform_matrix(i + 2, N + 1 + i) = 1;  // kernel spatial dims
        }

        if (output_accessor->type->IsScalar()) {
            auto mul_result = ir_builder->Mul(input_accessor, weight_accessor);
            auto add_result = ir_builder->Add(output_accessor, mul_result);
            ir_builder->Write(add_result, output_accessor);
            return;
        }

        this->ExpressInline({input_accessor, weight_accessor}, output_accessor, ir_builder);
    }
};

// Factory function to create a ConvolutionND creator with specified strides
template <int N>
std::shared_ptr<ConvolutionNDCreator<N>> CreateConvolutionND(
    const std::array<int, N>& strides = {}) {
    return ConvolutionNDCreator<N>::Create(strides);
}

}  // namespace galois::op