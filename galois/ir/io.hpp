#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::io {

class LoadBinary : public ir::Instruction {
   protected:
    LoadBinary() = default;

   public:
    static std::shared_ptr<LoadBinary> Create(std::shared_ptr<TensorType> ir_type,
                                              std::string filename) {
        auto self = std::shared_ptr<LoadBinary>(new LoadBinary);
        self->type = ir_type;
        self->filename = filename;
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<LoadBinary>(this->shared_from_this()));
    }

    std::string filename;
};

}  // namespace galois::ir::io
