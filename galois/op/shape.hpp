#pragma once

#include "galois/ir/builder.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

class ShapeCreator : public op::Creator {
   public:
    static std::shared_ptr<ShapeCreator> Create() {
        auto self = std::make_shared<ShapeCreator>();
        self->name = "Shape";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());
        Eigen::VectorXi64 output_shape(1);
        output_shape[0] = ir_input_types.front()->shape.size();
        auto new_value_type = Cast<ir::TensorType>(ir_input_types.front())->DataType();
        return ir::TensorType::Create(new_value_type, output_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);
        auto ir_input_shape = ir_inputs.front()->type->shape;
        ir_builder->ExpressCreator<op::FillCreator>(
            {ir_output, ir_builder->GetZero(ir_output_type->DataType())});

        for (int64_t i = 0; i < ir_input_shape.size(); ++i) {
            auto ir_accessor_out = ir_builder->CreateAccessor(ir_output);
            ir_accessor_out->transform_matrix.resize(0, 0);
            ir_accessor_out->shift_vector[0] = i;
            auto ir_shape = ir_builder->GetConstant(ir_inputs.front()->type->DataType(), ir_input_shape[i]);
            auto ir_write = ir_builder->Write(ir_shape, ir_accessor_out);
        }

        ir_builder->Return(ir_output);
    }
};
}  // namespace galois::op
