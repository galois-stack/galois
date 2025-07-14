#pragma once

#include "galois/op/creator.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

class SelectCreator : public Creator {
   public:
    static std::shared_ptr<SelectCreator> Create() {
        auto self = std::make_shared<SelectCreator>();
        self->name = "Select";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_condition_type,
        std::shared_ptr<ir::TensorType> ir_true_type,
        std::shared_ptr<ir::TensorType> ir_false_type) {
        
        GALOIS_ASSERT(ir_condition_type->DataType() == ir::bool_);
        
        GALOIS_ASSERT(ir_true_type->IsMatch(ir_false_type));
        
        // Handle scalar case
        if (ir_condition_type->IsScalar() && ir_true_type->IsScalar() && ir_false_type->IsScalar()) {
            return ir_true_type;
        }
        
        if (!ir_condition_type->IsScalar()) {
            GALOIS_ASSERT(ir_condition_type->shape == ir_true_type->shape);
        }
        
        return ir::TensorType::Create(ir_true_type->value_type, ir_true_type->shape);
    }

    // Helper function for inline expression with 3 inputs
    void ExpressInline(std::shared_ptr<ir::Tensor> ir_condition,
                       std::shared_ptr<ir::Tensor> ir_true_value,
                       std::shared_ptr<ir::Tensor> ir_false_value,
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) {
        
        // Handle scalar case
        if (ir_condition->type->IsScalar() && ir_true_value->type->IsScalar() && 
            ir_false_value->type->IsScalar()) {
            auto ir_select = ir_builder->Select(ir_condition, ir_true_value, ir_false_value);
            ir_builder->Write(ir_select, ir_output);
            return;
        }

        // Handle tensor case with grid-based parallel execution
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_accessor_out = ir_builder->CreateIdentityAccessor(ir_output);
        auto ir_accessor_condition = ir_builder->CreateIdentityAccessor(ir_condition);
        auto ir_accessor_true = ir_builder->CreateIdentityAccessor(ir_true_value);
        auto ir_accessor_false = ir_builder->CreateIdentityAccessor(ir_false_value);
        
        auto ir_select = ir_builder->Select(ir_accessor_condition, ir_accessor_true, ir_accessor_false);
        ir_builder->Write(ir_select, ir_accessor_out);
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 3);
        return InferTypeImpl(ir_input_types[0], ir_input_types[1], ir_input_types[2]);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        GALOIS_ASSERT(ir_inputs.size() == 3);
        auto ir_output_type = this->InferTypeImpl(ir_inputs[0]->type, ir_inputs[1]->type, ir_inputs[2]->type);
        auto ir_output = ir_builder->Alloca(ir_output_type);
        ir_builder->ExpressCreator<op::FillCreator>(
            {ir_output, ir_builder->GetZero(ir_output_type->DataType())});
        this->ExpressInline(ir_inputs[0], ir_inputs[1], ir_inputs[2], ir_output, ir_builder);
        ir_builder->Return(ir_output);
    }
};

}  // namespace galois::op
