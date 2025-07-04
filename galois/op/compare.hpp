#pragma once

#include "galois/op/binary.hpp"
#include "galois/ir/global_context.h"  // 用于获取 bool_ 类型

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
        GALOIS_ASSERT(ir_input_type0 == ir_input_type1);
        // 返回 shape 一致，数据类型为 bool 的 TensorType
        return ir::TensorType::Create(ir::bool_, ir_input_type0->NormalizeShape());
    }

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_input0,
                       std::shared_ptr<ir::Tensor> ir_input1,
                       std::shared_ptr<ir::Tensor> ir_output,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        if (ir_input0->type->IsScalar() && ir_input1->type->IsScalar()) {
            auto ir_re = ir_builder->Create<ir::CompareInstruction>(Operation, ir_input0, ir_input1);
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
        using Op = ir::CompareInstruction::Operation;
        if (Operation == Op::EQ) return "Equal";
        if (Operation == Op::NE) return "NotEqual";
        if (Operation == Op::LT) return "LessThan";
        if (Operation == Op::LE) return "LessEqual";
        if (Operation == Op::GT) return "GreaterThan";
        if (Operation == Op::GE) return "GreaterEqual";
        return "UnknownCompareOp";
    }
};

// 类型别名，像使用 AddCreator 一样用
using EqualCreator        = CompareCreator<ir::CompareInstruction::EQ>;
using NotEqualCreator     = CompareCreator<ir::CompareInstruction::NE>;
using LessThanCreator     = CompareCreator<ir::CompareInstruction::LT>;
using LessEqualCreator    = CompareCreator<ir::CompareInstruction::LE>;
using GreaterThanCreator  = CompareCreator<ir::CompareInstruction::GT>;
using GreaterEqualCreator = CompareCreator<ir::CompareInstruction::GE>;

}  // namespace galois::op
