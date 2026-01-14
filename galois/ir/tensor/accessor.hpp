#pragma once

#include "galois/ir/tensor/base.hpp"
#include "galois/ir/tensor/write.hpp"

namespace galois::ir {

class Accessor : public Instruction {
   public:
    static std::shared_ptr<Accessor> Create(std::shared_ptr<Tensor> ir_tensor,
                                            Eigen::MatrixXi64 transform_matrix,
                                            Eigen::VectorXi64 shift_vector) {
        std::shared_ptr<Accessor> self(new Accessor);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;
        self->transform_matrix = transform_matrix;
        self->shift_vector = shift_vector;
        self->type = ir_tensor->type->value_type;
        self->tag = "Accessor";
        return self;
    }

    bool IsReaded() {
        for (auto inst_with_index : this->instruction_with_index_list) {
            if (!Is<Write>(inst_with_index.instruction) || inst_with_index.operand_index == 0) {
                return true;
            }
        }

        return false;
    }

    bool IsWritten() {
        for (auto inst_with_index : this->instruction_with_index_list) {
            if (Is<Write>(inst_with_index.instruction) && inst_with_index.operand_index == 1) {
                return true;
            }
        }

        return false;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Accessor>(this->shared_from_this()));
    }

   public:
    Eigen::MatrixXi64 transform_matrix;
    Eigen::VectorXi64 shift_vector;
    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir
