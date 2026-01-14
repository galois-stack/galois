#pragma once

#include <string>

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class UnaryIntrinsic : public Instruction {
   protected:
    UnaryIntrinsic() = default;

   public:
    static std::shared_ptr<UnaryIntrinsic> Create(std::string intrinsic_name,
                                                  std::shared_ptr<Tensor> ir_oprand,
                                                  bool llvm_prefix = true) {
        GALOIS_ASSERT(intrinsic_name.size());
        std::shared_ptr<UnaryIntrinsic> self(new UnaryIntrinsic);
        self->OperandResize(1);
        self->intrinsic_name = intrinsic_name;
        self->Operand = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Operand = ir_oprand;
        self->type = ir_oprand->type;
        self->llvm_prefix = llvm_prefix;
        self->tag = "UnaryIntrinsic";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<UnaryIntrinsic>(this->shared_from_this()));
    }

   public:
    std::string intrinsic_name;
    bool llvm_prefix;
    OperandProperty Operand = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir
