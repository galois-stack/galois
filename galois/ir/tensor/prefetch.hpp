#pragma once

#include "galois/ir/tensor/accessor.hpp"

namespace galois::ir {

class Prefetch : public Instruction {
   public:
    static std::shared_ptr<Prefetch> Create(std::shared_ptr<ir::Accessor> ir_address, int64_t rw,
                                            int64_t locality, int64_t cache_type) {
        std::shared_ptr<Prefetch> self(new Prefetch);
        self->OperandResize(1);
        self->Address(ir_address);
        self->rw = rw;
        self->locality = locality;
        self->cache_type = cache_type;
        self->tag = "Prefetch";
        return self;
    }

    std::shared_ptr<ir::Accessor> Address() const {
        return Cast<ir::Accessor>(this->GetOperand(0));
    }

    void Address(std::shared_ptr<ir::Accessor> ir_address) { this->SetOperand(0, ir_address); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Prefetch>(this->shared_from_this()));
    }

    int64_t rw;
    int64_t locality;
    int64_t cache_type;
};

}  // namespace galois::ir
