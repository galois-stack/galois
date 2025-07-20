#pragma once

#include <string>
#include <vector>

#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

class OperatorFusionCreator : public op::Creator {
   public:
    static std::shared_ptr<OperatorFusionCreator> Create(
        const std::vector<std::string>& operations) {
        auto self = std::make_shared<OperatorFusionCreator>();
        self->operations = operations;
        self->name = "OperatorFusion";
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

        this->ExpressInline(ir_inputs, ir_output, ir_builder);

        ir_builder->Return(ir_output);
    }

    void ExpressInline(std::vector<std::shared_ptr<ir::Tensor>> ir_src,
                       std::shared_ptr<ir::Tensor> ir_dst,
                       std::shared_ptr<ir::Builder> ir_builder) {
        if (ir_src[0]->type->IsScalar() && ir_dst->type->IsScalar()) {
            std::shared_ptr<ir::Tensor> result;
            for (size_t i = 1; i < ir_src.size(); ++i) {
                if (i == 1) {
                    if (operations[i - 1] == "add") {
                        result = ir_builder->Add(ir_src[i - 1], ir_src[i]);
                    } else if (operations[i - 1] == "sub") {
                        result = ir_builder->Sub(ir_src[i - 1], ir_src[i]);
                    } else if (operations[i - 1] == "mul") {
                        result = ir_builder->Mul(ir_src[i - 1], ir_src[i]);
                    } else if (operations[i - 1] == "div") {
                        result = ir_builder->Div(ir_src[i - 1], ir_src[i]);
                    }
                } else {
                    if (operations[i - 1] == "add") {
                        result = ir_builder->Add(result, ir_src[i]);
                    } else if (operations[i - 1] == "sub") {
                        result = ir_builder->Sub(result, ir_src[i]);
                    } else if (operations[i - 1] == "mul") {
                        result = ir_builder->Mul(result, ir_src[i]);
                    } else if (operations[i - 1] == "div") {
                        result = ir_builder->Div(result, ir_src[i]);
                    }
                }
            }
            ir_builder->Write(result, ir_dst);
            return;
        } else {
            auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_src[0]->type->shape);

            std::vector<std::shared_ptr<ir::Tensor>> ir_accessor_src_vec;
            for (size_t i = 0; i < ir_src.size(); ++i) {
                auto ir_accessor_src_x = ir_builder->CreateIdentityAccessor(ir_src[i]);
                ir_accessor_src_vec.push_back(ir_accessor_src_x);
            }

            auto ir_accessor_dst = ir_builder->CreateIdentityAccessor(ir_dst);

            this->ExpressInline(ir_accessor_src_vec, ir_accessor_dst, ir_builder);
        }
    }

    std::vector<std::string> operations;
};

}  // namespace galois::op