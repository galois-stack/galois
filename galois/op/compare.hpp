#pragma once

#include "galois/op/binary.hpp"

namespace galois::op {

template <ir::CompareInstruction::Operation Operation>
class CompareCreator : public BinaryCreator {
   public:
    static std::shared_ptr<CompareCreator> Create() {
        auto self = std::make_shared<CompareCreator>();
        self->name = GetOperationName();
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type0,
        std::shared_ptr<ir::TensorType> ir_input_type1) override {
        GALOIS_ASSERT(ir_input_type0->IsMatch(ir_input_type1));
        if (ir_input_type0->IsScalar()) {
            return ir::bool_;
        } else {
            // For tensors, return tensor of booleans with same shape
            return ir::TensorType::Create(ir::bool_, ir_input_type0->shape);
        }
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input0, std::shared_ptr<ir::Tensor> ir_input1,
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        if (ir_input0->type->IsScalar() && ir_input1->type->IsScalar()) {
            auto ir_re =
                ir_builder->Create<ir::CompareInstruction>(Operation, ir_input0, ir_input1);
            ir_builder->Write(ir_re, ir_output);
            return;
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
        auto ir_accessor_out = ir_builder->CreateIdentityAccessor(ir_output);
        auto ir_accessor_in0 = ir_builder->CreateIdentityAccessor(ir_input0);
        auto ir_accessor_in1 = ir_builder->CreateIdentityAccessor(ir_input1);

        this->ExpressInline(ir_accessor_in0, ir_accessor_in1, ir_accessor_out, ir_builder);
    }

   private:
    static std::string GetOperationName() {
        if (Operation == ir::CompareInstruction::Equal) {
            return "Equal";
        } else if (Operation == ir::CompareInstruction::NotEqual) {
            return "NotEqual";
        } else if (Operation == ir::CompareInstruction::Less) {
            return "Less";
        } else if (Operation == ir::CompareInstruction::LessEqual) {
            return "LessEqual";
        } else if (Operation == ir::CompareInstruction::Greater) {
            return "Greater";
        } else if (Operation == ir::CompareInstruction::GreaterEqual) {
            return "GreaterEqual";
        }
    }
};

using EqualCreator = CompareCreator<ir::CompareInstruction::Equal>;
using NotEqualCreator = CompareCreator<ir::CompareInstruction::NotEqual>;
using LessCreator = CompareCreator<ir::CompareInstruction::Less>;
using LessEqualCreator = CompareCreator<ir::CompareInstruction::LessEqual>;
using GreaterCreator = CompareCreator<ir::CompareInstruction::Greater>;
using GreaterEqualCreator = CompareCreator<ir::CompareInstruction::GreaterEqual>;

}  // namespace galois::op
