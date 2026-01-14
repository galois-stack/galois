#pragma once

#include "galois/ir/tensor/block.hpp"
#include "galois/ir/tensor/grid_index.hpp"

namespace galois::ir {

class Grid : public Tensor {
   public:
    static std::shared_ptr<Grid> Create(Eigen::VectorXi64 shape) {
        std::shared_ptr<Grid> self(new Grid);
        self->index = GridIndex::Create(shape.size());
        self->block = Block::Create();
        self->shape = shape;
        self->tag = "Grid";
        return self;
    }

    int64_t GetAffineDimSize() const { return this->shape.size(); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Grid>(this->shared_from_this()));
    }

    Eigen::VectorXi64 shape;
    std::shared_ptr<Block> block = nullptr;
    std::shared_ptr<GridIndex> index = nullptr;
    bool enable_multi_thread = false;
    bool unroll_grid = false;
};

}  // namespace galois::ir
