#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/fill.hpp"
#include "galois/op/unary.hpp"

namespace galois::op {

class PaddingCreator : public UnaryCreator {
   public:
    static std::shared_ptr<PaddingCreator> Create(Eigen::VectorXi64 padding_shape) {
        std::shared_ptr<PaddingCreator> self(new PaddingCreator);
        self->padding_shape = padding_shape;
        self->name = "Padding";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) override {
        return ir::TensorType::Create(Cast<ir::TensorType>(ir_input_type)->value_type,
                                      padding_shape);
    };

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input,
                           std::shared_ptr<ir::Tensor> ir_output,
                           std::shared_ptr<ir::Builder> ir_builder) override {
        auto input_shape = ir_input->type->shape;
        auto output_shape = ir_output->type->shape;

        if (input_shape.size() == 0) {
            ir_builder->Create<ir::Write>(ir_input, ir_output);
            return;
        }

        GALOIS_ASSERT(output_shape[0] >= input_shape[0]);
        {
            Eigen::VectorXi64 input_shape_outer(1);
            input_shape_outer[0] = input_shape[0];
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(input_shape_outer);

            auto ir_input_left_origin = ir_builder->CreateAccessor(ir_input);
            ir_input_left_origin->transform_matrix(0) = 1;
            Eigen::VectorXi64 input_slice_shape(input_shape.size());
            input_slice_shape[0] = 1;
            for (int64_t i = 1; i < input_slice_shape.size(); ++i) {
                input_slice_shape[i] = input_shape[i];
            }
            auto ir_input_slice =
                ir_builder->Create<ir::SliceView>(ir_input_left_origin, input_slice_shape);

            auto ir_output_left_origin = ir_builder->CreateAccessor(ir_output);
            ir_output_left_origin->transform_matrix(0) = 1;
            Eigen::VectorXi64 output_slice_shape(output_shape.size());
            output_slice_shape[0] = 1;
            for (int64_t i = 1; i < output_slice_shape.size(); ++i) {
                output_slice_shape[i] = output_shape[i];
            }
            auto ir_output_slice =
                ir_builder->Create<ir::SliceView>(ir_output_left_origin, output_slice_shape);

            auto ir_input_slice_squeeze = ir_builder->Create<ir::SqueezeDimView>(ir_input_slice, 0);
            auto ir_output_slice_squeeze =
                ir_builder->Create<ir::SqueezeDimView>(ir_output_slice, 0);
            this->AffineExpressImpl(ir_input_slice_squeeze, ir_output_slice_squeeze, ir_builder);
        }
        {
            Eigen::VectorXi64 remainder_shape = output_shape;
            remainder_shape[0] = output_shape[0] - input_shape[0];
            auto ir_output_left_origin = ir_builder->CreateAccessor(ir_output);
            ir_output_left_origin->shift_vector[0] = input_shape[0];
            auto ir_output_slice =
                ir_builder->Create<ir::SliceView>(ir_output_left_origin, remainder_shape);
            ir_builder->ExpressCreator<op::FillCreator>(
                {ir_output_slice, ir_builder->GetZero(ir_output_slice->type->DataType())});
        }
    }

    Eigen::VectorXi64 padding_shape;
};

}  // namespace galois::op
