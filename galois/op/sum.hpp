#pragma once

#include "galois/ir/builder.hpp"
#include "galois/op/fill.hpp"

namespace galois::op {

class SumCreator : public op::Creator {
   public:
    static std::shared_ptr<SumCreator> Create() {
        auto self = std::make_shared<SumCreator>();
        self->name = "Sum";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        auto input_type = ir_input_types.front();
        return input_type->DataType();
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_re_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_re = ir_builder->Alloca(ir_re_type);
        ir_builder->ExpressCreator<op::FillCreator>(
            {ir_re, ir_builder->GetZero(ir_re_type->DataType())});
        this->ExpressInline(ir_inputs.front(), ir_re, ir_builder);
        ir_builder->Return(ir_re);
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input, std::shared_ptr<ir::Tensor> ir_re,
                       std::shared_ptr<ir::Builder> ir_builder) {
        if (ir_input->type->IsScalar()) {
            auto ir_add = ir_builder->Add(ir_input, ir_re);
            ir_builder->Write(ir_add, ir_re);
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
            auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
            this->ExpressInline(ir_accessor, ir_re, ir_builder);
        }
    }
};

}  // namespace galois::op
