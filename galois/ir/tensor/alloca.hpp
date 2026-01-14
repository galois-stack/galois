#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

enum struct MemoryType { Heap, Stack };

class Alloca : public Instruction {
   protected:
    Alloca() = default;

   public:
    static std::shared_ptr<Alloca> Create(std::shared_ptr<TensorType> ir_type,
                                          MemoryType memory_type = MemoryType::Heap) {
        std::shared_ptr<Alloca> self(new Alloca);
        self->type = ir_type;
        self->tag = "Alloca";
        self->memory_type = memory_type;
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Alloca>(this->shared_from_this()));
    }

    MemoryType memory_type;
};

}  // namespace galois::ir
