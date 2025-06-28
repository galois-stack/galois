#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/op/creator.hpp"
#include "galois/op/sum.hpp"

namespace galois::op {
class BroadCastCreator : public op::Creator {
   public:
    static std::shared_ptr<BroadCastCreator> Create(Eigen::VectorXi64 broadcast_shape) {
        auto self = std::make_shared<BroadCastCreator>();
        self->broadcast_shape = broadcast_shape;
        self->name = "BroadCast";
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
        auto zero = ir_builder->GetZero(ir_output->type->DataType());
        ir_builder->ExpressCreator<op::FillCreator>({ir_output, zero});

        auto input_broadcast =
            ir_builder->Create<ir::view::BroadCast>(ir_inputs[0], ir_output_type->shape);
        this->ExpressInline(input_broadcast, ir_output, ir_builder);

        ir_builder->Return(ir_output);
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_act, std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) {
        auto input_shape = ir_act->type->shape;
        auto output_shape = ir_output->type->shape;

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(output_shape);

        auto output_accessor = ir_builder->CreateAccessor(ir_output);
        output_accessor->transform_matrix(0, 0) = 1;
        output_accessor->transform_matrix(1, 1) = 1;

        auto input_accessor = ir_builder->CreateAccessor(ir_act);
        input_accessor->transform_matrix(0, 0) = 1;
        input_accessor->transform_matrix(1, 1) = 1;

        if (input_accessor->type->IsScalar()) {
            ir_builder->Write(input_accessor, output_accessor);
            return;
        }

        this->ExpressInline(input_accessor, output_accessor, ir_builder);
    }

    Eigen::VectorXi64 broadcast_shape;
};

}  // namespace galois::op