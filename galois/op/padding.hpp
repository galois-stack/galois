#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

class PaddingCreator : public OperatorCreator {
   public:
    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return TensorType::Create(Cast<TensorType>(ir_input_types.front())->value_type,
                                  padding_shape);
    };

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto ir_output = ir_outputs.front();
        auto input_shape = ir_input->type->shape;
        auto output_shape = ir_output->type->shape;

        if (input_shape.size() == 0) {
            ir_builder->Create<Write>(ir_input, ir_output);
            return;
        }

        GALOIS_ASSERT(output_shape[0] >= input_shape[0]);
        {
            Eigen::VectorXi64 input_shape_outer(1);
            input_shape_outer[0] = input_shape[0];
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(input_shape_outer);

            auto ir_input_left_origin = ir_builder->CreateAccessor(ir_input);
            ir_input_left_origin->transform_matrix(0) = 1;
            Eigen::VectorXi64 input_slice_shape(input_shape.size() - 1);
            for (int64_t i = 0; i < input_slice_shape.size(); ++i) {
                input_slice_shape[i] = input_shape[i + 1];
            }
            auto ir_input_slice =
                ir_builder->Create<Slice>(ir_input_left_origin, input_slice_shape);

            auto ir_output_left_origin = ir_builder->CreateAccessor(ir_output);
            ir_output_left_origin->transform_matrix(0) = 1;
            Eigen::VectorXi64 output_slice_shape(output_shape.size() - 1);
            for (int64_t i = 0; i < output_slice_shape.size(); ++i) {
                output_slice_shape[i] = output_shape[i + 1];
            }
            auto ir_output_slice =
                ir_builder->Create<Slice>(ir_output_left_origin, output_slice_shape);
            AffineExpress({ir_input_slice}, {ir_output_slice}, ir_builder);
        }
        {
            Eigen::VectorXi64 remainder_shape = output_shape;
            remainder_shape[0] = output_shape[0] - input_shape[0];
            auto ir_output_left_origin = ir_builder->CreateAccessor(ir_output);
            ir_output_left_origin->shift_vector[0] = input_shape[0];
            auto ir_output_slice =
                ir_builder->Create<Slice>(ir_output_left_origin, remainder_shape);
            set_zero_creator.AffineExpress({ir_output_slice}, {}, ir_builder);
        }
    }

    Eigen::VectorXi64 padding_shape;
    SetZeroCreator set_zero_creator;
};

}  // namespace galois::op
