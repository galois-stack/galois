#pragma once

#include <string>
#include <vector>

#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/arithmetic.hpp"

namespace galois::op {

class OperatorFusionNoOptCreator : public op::Creator {
   public:
    static std::shared_ptr<OperatorFusionNoOptCreator> Create() {
        auto self = std::make_shared<OperatorFusionNoOptCreator>();
        self->name = "OperatorFusionNoOpt";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return ir_input_types.front();
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);


        auto ir_add = ir_builder->ExpressCreator<op::AddCreator>({ir_inputs[0], ir_inputs[1]});
        auto ir_sub = ir_builder->ExpressCreator<op::SubCreator>({ir_add, ir_inputs[2]});

        ir_builder->Write(ir_sub, ir_output);

        ir_builder->Return(ir_output);
    }
};

}  // namespace galois::op