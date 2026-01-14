#pragma once

#include <algorithm>
#include <list>
#include <memory>
#include <numeric>
#include <regex>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "galois/assert.hpp"
#include "galois/helper.hpp"
#include "galois/ir/global_context.h"
#include "galois/ir/tensor_type.hpp"
#include "galois/ir/visitor.hpp"
#include "galois/named.hpp"
#include "galois/property.hpp"

namespace galois::ir {

class Block;
class Visitor;
class Instruction;

struct InstructionAndOperandIndex {
    std::weak_ptr<Instruction> instruction;
    int64_t operand_index;
};

inline bool operator==(galois::ir::InstructionAndOperandIndex lhs,
                       galois::ir::InstructionAndOperandIndex rhs) {
    return lhs.instruction.lock() == rhs.instruction.lock() &&
           lhs.operand_index == rhs.operand_index;
}

class Tensor : public Named, public std::enable_shared_from_this<Tensor> {
   protected:
    Tensor() {}

   public:
    std::shared_ptr<Block> ParentBlock() {
        if (auto parent = this->parent_block.lock()) {
            return Cast<Tensor>(parent)->ParentBlock();
        } else {
            return Cast<Block>(this->shared_from_this());
        }
    }

    bool IsInsideOf(std::shared_ptr<Block> ir_block) {
        if (auto parent = this->parent_block.lock()) {
            if (parent == ir_block) {
                return true;
            }
        }

        if (auto parent = this->parent_block.lock()) {
            return Cast<Tensor>(parent)->IsInsideOf(ir_block);
        }

        return false;
    }

    virtual void ApplyVisitor(std::shared_ptr<Visitor> interpreter) { GALOIS_UNREACHABLE; }

    virtual ~Tensor() {}

   private:
    bool is_finalized = false;

   public:
    std::shared_ptr<ir::TensorType> type = nullptr;
    std::unordered_map<std::string, std::list<std::string>> annotation_dict;
    std::list<InstructionAndOperandIndex> instruction_with_index_list;
    std::weak_ptr<Block> parent_block;
    std::shared_ptr<pir::Value> pir_value = nullptr;
    std::string tag = "Tensor";
};

class Instruction : virtual public Tensor {
   protected:
    Instruction() : Instruction(0) {}

    Instruction(int64_t operand_size) {
        this->tag = "Instruction";
        this->operands.resize(operand_size);
    }

   public:
    virtual void OperandResize(int64_t size) { return this->operands.resize(size); }

    virtual int64_t OperandSize() const { return this->operands.size(); }

    std::shared_ptr<Tensor> GetOperand(int64_t i) const {
        GALOIS_ASSERT(this->OperandSize() > i);
        return this->operands[i];
    };

    void SetOperand(int64_t i, std::shared_ptr<Tensor> ir_value) {
        GALOIS_ASSERT(this->OperandSize() > i);

        auto ir_old_value = this->operands[i];
        if (ir_old_value) {
            ir_old_value->instruction_with_index_list.remove(
                {Cast<Instruction>(this->shared_from_this()), i});
        }

        this->operands[i] = ir_value;
        if (ir_value)
            ir_value->instruction_with_index_list.push_back(
                {Cast<Instruction>(this->shared_from_this()), i});
    }

   protected:
    std::vector<std::shared_ptr<Tensor>> operands;
};

class OperandProperty : public Property<std::shared_ptr<Tensor>> {
   public:
    OperandProperty(std::shared_ptr<galois::ir::Instruction> ir_instruction, int64_t index)
        : Property<std::shared_ptr<Tensor>>(
              [ir_instruction, index]() { return ir_instruction->GetOperand(index); },
              [ir_instruction, index](std::shared_ptr<Tensor> ir_tensor) {
                  ir_instruction->SetOperand(index, ir_tensor);
              }) {}

    OperandProperty& operator=(std::shared_ptr<Tensor> ir_tensor) {
        this->set(ir_tensor);
        return *this;
    }
};

}  // namespace galois::ir
