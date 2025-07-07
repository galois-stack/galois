#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/copy.hpp"
#include "galois/op/creator.hpp"

namespace galois::op {
class BroadCastCreator : public op::Creator {
   public:
    static std::shared_ptr<BroadCastCreator> Create(Eigen::VectorXi64 broadcast_shape) {
        auto self = std::make_shared<BroadCastCreator>();
        self->broadcast_shape = broadcast_shape;
        self->name = "Broadcast";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());

        auto new_value_type = Cast<ir::TensorType>(ir_input_types.front())->DataType();
        return ir::TensorType::Create(new_value_type, broadcast_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);

        auto input_broadcast =
            ir_builder->Create<ir::view::Broadcast>(ir_inputs[0], ir_output_type->shape);
        auto ir_copy = op::CopyCreator::Create();
        ir_copy->ExpressInline(input_broadcast, ir_output, ir_builder);

        ir_builder->Return(ir_output);
    }

    Eigen::VectorXi64 broadcast_shape;
};

}  // namespace galois::op
