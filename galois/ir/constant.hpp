#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir {

class Constant : public Tensor {
   public:
    virtual ~Constant() {}

    void Detach() override { this->instruction_with_index_list.clear(); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Constant>(this->shared_from_this()));
    }
};

class ConstantRealNumber : public Constant {
   protected:
    ConstantRealNumber() = default;

    virtual void ApplyVisitor(std::shared_ptr<Visitor> interpreter) {
        interpreter->Visit(Cast<ConstantRealNumber>(this->shared_from_this()));
    }
};

class ConstantInt : public ConstantRealNumber {
   protected:
    ConstantInt() = default;

   public:
    static std::shared_ptr<ConstantInt> Create(std::shared_ptr<TensorType> ir_type, int64_t value) {
        GALOIS_ASSERT(ir_type);
        std::shared_ptr<ConstantInt> self(new ConstantInt);
        self->type = ir_type;
        self->value = value;
        self->tag = "ConstantInt";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<ConstantInt>(this->shared_from_this()));
    }

   public:
    uint64_t value;
};

class ConstantFloat : public ConstantRealNumber {
   protected:
    ConstantFloat() = default;

   public:
    enum SpecialValue { None, Smallest, Largest, NaN, Inf };

    static std::shared_ptr<ConstantFloat> Create(std::shared_ptr<TensorType> type, double value) {
        GALOIS_ASSERT(type);
        std::shared_ptr<ConstantFloat> self(new ConstantFloat);
        self->type = type;
        self->value = value;
        self->tag = "ConstantFloat";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<ConstantFloat>(this->shared_from_this()));
    }

   public:
    double value;
    SpecialValue special_value = SpecialValue::None;
    bool is_negative = false;
};

}  // namespace galois::ir
