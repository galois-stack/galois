#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class GridIndex : public Tensor {
   public:
    static std::shared_ptr<GridIndex> Create(int64_t rank) {
        std::shared_ptr<GridIndex> self(new GridIndex);
        self->type = i64->Tile(rank);
        self->tag = "GridIndex";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<GridIndex>(this->shared_from_this()));
    }
};

}  // namespace galois::ir
