#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/tensor.hpp"
#include "galois/op/concatenate.hpp"

namespace galois::op {

class StackCreator : public op::Creator {
   public:
    static std::shared_ptr<StackCreator> Create(int64_t dim) {
        auto self = std::make_shared<StackCreator>();
        self->dim = dim;
        self->name = "Stack";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());
        auto base_type = ir_input_types[0];
        auto shape = base_type->shape;
        Eigen::VectorXi64 new_shape(shape.size() + 1);
        new_shape.head(dim) = shape.head(dim);
        new_shape(dim) = ir_input_types.size();
        new_shape.tail(shape.size() - dim) = shape.tail(shape.size() - dim);
        return ir::TensorType::Create(base_type->value_type, new_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        std::vector<std::shared_ptr<ir::Tensor>> expanded_inputs;
        for (auto& input : ir_inputs) {
            auto shape = input->type->shape;
            auto stride = input->type->stride;
            Eigen::VectorXi64 new_shape(shape.size() + 1);
            new_shape.head(dim) = shape.head(dim);
            new_shape(dim) = 1;
            new_shape.tail(shape.size() - dim) = shape.tail(shape.size() - dim);
            Eigen::VectorXi64 new_stride(stride.size() + 1);

            new_stride.head(dim) = stride.head(dim);
            new_stride(dim) = (dim < stride.size() ? stride(dim) : 1);
            new_stride.tail(stride.size() - dim) = stride.tail(stride.size() - dim);

            auto type = ir::TensorType::Create(input->type->value_type, new_shape, new_stride);
            auto bitcast = ir_builder->Create<ir::BitCastView>(input, type);
            expanded_inputs.push_back(bitcast);
        }

        auto concat = op::ConcatenateCreator::Create(dim);
        concat->Express(expanded_inputs, ir_builder);
    }

    int64_t dim;
};

}  // namespace galois::op
