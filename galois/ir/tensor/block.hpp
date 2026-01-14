#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Block : public Tensor, public std::list<std::shared_ptr<Tensor>> {
   public:
    static std::shared_ptr<Block> Create() {
        std::shared_ptr<Block> self(new Block);
        self->tag = "Block";
        return self;
    }
    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Block>(this->shared_from_this()));
    }
};

}  // namespace galois::ir
