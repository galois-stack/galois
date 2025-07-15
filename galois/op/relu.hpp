#pragma once

#include "galois/op/creator.hpp"
#include "galois/op/compare.hpp"
#include "galois/op/select.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

class ReluCreator : public Creator {
   public:
    static std::shared_ptr<ReluCreator> Create() {
        auto self = std::make_shared<ReluCreator>();
        self->name = "Relu";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return ir_input_types[0];
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        GALOIS_ASSERT(ir_inputs.size() == 1);
        auto ir_input = ir_inputs[0];
        auto ir_output_type = this->InferType({ir_input->type});
        auto ir_output = ir_builder->Alloca(ir_output_type);
        
        ir_builder->ExpressCreator<op::FillCreator>(
            {ir_output, ir_builder->GetZero(ir_output_type->DataType())});
        
        this->ExpressInline(ir_input, ir_output, ir_builder);
        ir_builder->Return(ir_output);
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input, 
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) {
        
        auto ir_zero = ir_builder->GetZero(ir_input->type->DataType());
        
        if (ir_input->type->IsScalar()) {
            auto ir_condition = ir_builder->Greater(ir_input, ir_zero);
            auto ir_select = ir_builder->Select(ir_condition, ir_input, ir_zero);
            ir_builder->Write(ir_select, ir_output);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
            
            auto ir_accessor_input = ir_builder->CreateIdentityAccessor(ir_input);
            auto ir_accessor_output = ir_builder->CreateIdentityAccessor(ir_output);
            
            auto ir_condition = ir_builder->Greater(ir_accessor_input, ir_zero);
            auto ir_select = ir_builder->Select(ir_condition, ir_accessor_input, ir_zero);
            ir_builder->Write(ir_select, ir_accessor_output);
        }
    }
};

}  // namespace galois::op
