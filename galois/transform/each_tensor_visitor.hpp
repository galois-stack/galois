#pragma once

#include <memory>

#include "galois/ir/tensor.hpp"

namespace galois::transform {

template <typename Value_>
class EachTensorVisitor : public ir::Visitor {
   protected:
    EachTensorVisitor() = default;

   public:
    static std::shared_ptr<EachTensorVisitor<Value_>> Create(
        std::function<void(std::shared_ptr<Value_>)> callback) {
        auto self = std::shared_ptr<EachTensorVisitor<Value_>>(new EachTensorVisitor);
        self->callback_ = callback;
        return self;
    }

    void Visit(std::shared_ptr<ir::Tensor> ir_tensor) override {
        if (auto value = Cast<Value_>(ir_tensor)) {
            callback_(value);
        }
    }

    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        for (auto tensor : ir_block->tensors) {
            this->Visit(tensor);
        }
    }

   private:
    std::function<void(std::shared_ptr<Value_>)> callback_;
};

}  // namespace galois::transform