#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Indexing : public Instruction {
   public:
    static std::shared_ptr<Indexing> Create(std::shared_ptr<Tensor> ir_tensor,
                                            std::vector<std::shared_ptr<Tensor>> ir_indices) {
        std::shared_ptr<Indexing> self(new Indexing);
        self->OperandResize(1 + ir_indices.size());
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;
        for (int64_t i = 0; i < ir_indices.size(); ++i) {
            self->Index(i, ir_indices[i]);
        }
        self->type = ir_tensor->type->value_type;
        self->tag = "Indexing";
        return self;
    }

    int64_t IndexSize() { return this->OperandSize() - 1; }
    std::shared_ptr<ir::Tensor> Index(int64_t i) { return this->GetOperand(1 + i); }
    void Index(int64_t i, std::shared_ptr<ir::Tensor> ir_index) {
        this->SetOperand(1 + i, ir_index);
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Indexing>(this->shared_from_this()));
    }

   public:
    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir
