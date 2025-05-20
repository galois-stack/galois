#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/tensor.hpp"
#include "galois/ir/view.hpp"

namespace galois::op {

class FlattenCreator : public op::Creator {
   public:
    static std::shared_ptr<FlattenCreator> Create() {
        auto self = std::make_shared<FlattenCreator>();
        self->name = "Flatten";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        auto input_type = ir_input_types[0];
        int64_t total_size = 1;
        for (int64_t i = 0; i < input_type->shape.size(); ++i) {
            total_size *= input_type->shape[i];
        }
        Eigen::VectorXi64 new_shape(1);
        new_shape(0) = total_size;
        Eigen::VectorXi64 new_stride(1);
        new_stride(0) = 1;
        return ir::TensorType::Create(input_type->value_type, new_shape, new_stride);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        GALOIS_ASSERT(ir_inputs.size() == 1);
        auto input = ir_inputs[0];
        auto ir_flatten_view = ir_builder->Create<ir::FlattenView>(input);
        ir_builder->Return(ir_flatten_view);
    }
};

}  // namespace galois::op
