#pragma once
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/creator.hpp"

namespace galois::op {

class SliceCreator : public op::Creator {
   public:
    static std::shared_ptr<SliceCreator> Create(Eigen::VectorXi64 slice_shape) {
        std::shared_ptr<SliceCreator> self(new SliceCreator);
        self->slice_shape = slice_shape;
        self->name = "Slice";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return ir::TensorType::Create(Cast<ir::TensorType>(ir_input_types.front())->value_type,
                                  slice_shape);
    };

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Create<ir::Alloca>(ir_output_type);
        this->_Express({ir_inputs[0], ir_output}, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }

    void _Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                  std::shared_ptr<ir::Builder> ir_builder) {
        auto ir_input = ir_inputs[0];
        auto ir_output = ir_inputs[1];
        auto input_shape = ir_input->type->shape;
        auto output_shape = ir_output->type->shape;

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(output_shape);
        auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_output);
        auto ir_input_accessor = ir_builder->CreateIdentityAccessor(ir_input);
        ir_builder->Create<ir::Write>(ir_input_accessor, ir_output_accessor);
    }

    Eigen::VectorXi64 slice_shape;
};

}  // namespace galois::op
