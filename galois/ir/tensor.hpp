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
#include "galois/property.hpp"

namespace galois::ir {

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
            // GALOIS_ASSERT(Is<Block>(this->shared_from_this()));
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
        // GALOIS_ASSERT(ir_value);
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

class Accessor : public Instruction {
   public:
    static std::shared_ptr<Accessor> Create(std::shared_ptr<Tensor> ir_tensor,
                                            Eigen::MatrixXi64 transform_matrix,
                                            Eigen::VectorXi64 shift_vector) {
        std::shared_ptr<Accessor> self(new Accessor);
        self->OperandResize(1);
        self->Tensor(ir_tensor);
        self->transform_matrix = transform_matrix;
        self->shift_vector = shift_vector;
        self->type = ir_tensor->type->value_type;
        self->tag = "Accessor";
        return self;
    }

    bool IsReaded() {
        for (auto inst_with_index : this->instruction_with_index_list) {
            if (!Is<Write>(inst_with_index.instruction) || inst_with_index.operand_index == 0) {
                return true;
            }
        }

        return false;
    }

    bool IsWritten() {
        for (auto inst_with_index : this->instruction_with_index_list) {
            if (Is<Write>(inst_with_index.instruction) && inst_with_index.operand_index == 1) {
                return true;
            }
        }

        return false;
    }

    std::shared_ptr<Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Accessor>(this->shared_from_this()));
    }

   public:
    Eigen::MatrixXi64 transform_matrix;
    Eigen::VectorXi64 shift_vector;
};

class Indexing : public Instruction {
   public:
    static std::shared_ptr<Indexing> Create(std::shared_ptr<Tensor> ir_tensor,
                                            std::vector<std::shared_ptr<Tensor>> ir_indices) {
        std::shared_ptr<Indexing> self(new Indexing);
        self->OperandResize(1 + ir_indices.size());
        self->Tensor(ir_tensor);
        for (int64_t i = 0; i < ir_indices.size(); ++i) {
            self->Index(i, ir_indices[i]);
        }
        self->type = ir_tensor->type->value_type;
        self->tag = "Indexing";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    int64_t IndexSize() { return this->OperandSize() - 1; }
    std::shared_ptr<ir::Tensor> Index(int64_t i) { return this->GetOperand(1 + i); }
    void Index(int64_t i, std::shared_ptr<ir::Tensor> ir_index) {
        this->SetOperand(1 + i, ir_index);
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Indexing>(this->shared_from_this()));
    }
};

class ArithmeticInstruction : public Instruction {
   public:
    enum Operation {
        Add,
        Sub,
        Mul,
        Div,
    };

    static std::shared_ptr<ArithmeticInstruction> Create(Operation op,
                                                         std::shared_ptr<Tensor> ir_operand0,
                                                         std::shared_ptr<Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<ArithmeticInstruction> self(new ArithmeticInstruction);
        self->operation = op;
        self->OperandResize(2);
        self->SetOperand(0, ir_operand0);
        self->SetOperand(1, ir_operand1);
        self->type = ir_operand0->type;
        self->tag = "ArithmeticInstruction";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<ArithmeticInstruction>(this->shared_from_this()));
    }

   public:
    Operation operation;
};

class CompareInstruction : public Instruction {
   public:
    enum Operation {
        Equal,         // ==
        NotEqual,      // !=
        Less,          // <
        LessEqual,     // <=
        Greater,       // >
        GreaterEqual,  // >=
    };

    static std::shared_ptr<CompareInstruction> Create(Operation op,
                                                      std::shared_ptr<Tensor> ir_operand0,
                                                      std::shared_ptr<Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<CompareInstruction> self(new CompareInstruction);
        self->operation = op;
        self->OperandResize(2);
        self->SetOperand(0, ir_operand0);
        self->SetOperand(1, ir_operand1);
        if (ir_operand0->type->IsScalar()) {
            self->type = ir::bool_;
        } else {
            // For tensors, return tensor of booleans with same shape
            self->type = ir::TensorType::Create(ir::bool_, ir_operand0->type->shape);
        }
        self->tag = "CompareInstruction";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<CompareInstruction>(this->shared_from_this()));
    }

   public:
    Operation operation;
};

class SelectInstruction : public Instruction {
   public:
    static std::shared_ptr<SelectInstruction> Create(std::shared_ptr<Tensor> ir_condition,
                                                     std::shared_ptr<Tensor> ir_true_value,
                                                     std::shared_ptr<Tensor> ir_false_value) {
        GALOIS_ASSERT(ir_condition);
        GALOIS_ASSERT(ir_true_value);
        GALOIS_ASSERT(ir_false_value);
        GALOIS_ASSERT(ir_true_value->type == ir_false_value->type);

        // Condition must be boolean tensor with same shape as values (or broadcastable)
        if (ir_condition->type->IsScalar()) {
            GALOIS_ASSERT(ir_condition->type->DataType() == ir::bool_);
        } else {
            GALOIS_ASSERT(ir_condition->type->DataType() == ir::bool_);
            GALOIS_ASSERT(ir_condition->type->shape.size() == ir_true_value->type->shape.size());
            for (int64_t i = 0; i < ir_condition->type->shape.size(); ++i) {
                GALOIS_ASSERT(ir_condition->type->shape[i] == ir_true_value->type->shape[i] ||
                              ir_condition->type->shape[i] == 1 ||
                              ir_true_value->type->shape[i] == 1);
            }
        }

        std::shared_ptr<SelectInstruction> self(new SelectInstruction);
        self->OperandResize(3);
        self->Condition(ir_condition);
        self->TrueValue(ir_true_value);
        self->FalseValue(ir_false_value);
        self->type = ir_true_value->type;
        self->tag = "SelectInstruction";
        return self;
    }

    std::shared_ptr<Tensor> Condition() { return this->GetOperand(0); }
    void Condition(std::shared_ptr<Tensor> ir_condition) { this->SetOperand(0, ir_condition); }

    std::shared_ptr<Tensor> TrueValue() { return this->GetOperand(1); }
    void TrueValue(std::shared_ptr<Tensor> ir_true_value) { this->SetOperand(1, ir_true_value); }

    std::shared_ptr<Tensor> FalseValue() { return this->GetOperand(2); }
    void FalseValue(std::shared_ptr<Tensor> ir_false_value) { this->SetOperand(2, ir_false_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SelectInstruction>(this->shared_from_this()));
    }
};

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

    int64_t rw;          // 读(0)或写(1)
    int64_t locality;    // 时间局部性  0（无局部性）到 3（极高局部性）
    int64_t cache_type;  // 缓存类型 (0: 指令缓存, 1: 数据缓存)
};

/// @brief For liked fmla.4s v2, v0, v1[0] instruction
class VectorBroadcast : public Instruction {
   protected:
    VectorBroadcast() = default;

   public:
    /// TODO: 需要进一步处理
    static std::shared_ptr<VectorBroadcast> Create(std::shared_ptr<Tensor> ir_value,
                                                   std::shared_ptr<ir::TensorType> ir_type,
                                                   int64_t lane_id) {
        std::shared_ptr<VectorBroadcast> self(new VectorBroadcast);
        self->OperandResize(1);
        self->Vector(ir_value);
        self->lane_id = lane_id;
        self->type = ir_type;
        self->tag = "VectorBroadcast";
        return self;
    }

    std::shared_ptr<Tensor> Vector() { return this->GetOperand(0); }
    void Vector(std::shared_ptr<Tensor> ir_value) { this->SetOperand(0, ir_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<VectorBroadcast>(this->shared_from_this()));
    }

    int64_t lane_id;
};

class Write : public Instruction {
   public:
    static std::shared_ptr<Write> Create(std::shared_ptr<Tensor> value,
                                         std::shared_ptr<Tensor> accessor) {
        std::shared_ptr<Write> self(new Write);
        GALOIS_ASSERT(value->type == accessor->type);
        self->OperandResize(2);
        self->Tensor(value);
        self->Variable(accessor);
        self->tag = "Write";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() const { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<class Tensor> value) { this->SetOperand(0, value); }

    std::shared_ptr<class Tensor> Variable() const {
        return Cast<class Tensor>(this->GetOperand(1));
    }
    void Variable(std::shared_ptr<class Tensor> accessor) { this->SetOperand(1, accessor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Write>(this->shared_from_this()));
    }
};

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

    //    public:
    // std::list<std::shared_ptr<Tensor>> tensors;
};

class Input : public Tensor {
   public:
    static std::shared_ptr<Input> Create(std::shared_ptr<TensorType> ir_type) {
        std::shared_ptr<Input> self(new Input);
        self->type = ir_type;
        self->tag = "Input";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Input>(this->shared_from_this()));
    }
};

/// @brief An operator of tensors, which is liked as a node of ComputingGraph
class Operator : public Tensor {
   public:
    static std::shared_ptr<Operator> Create(std::shared_ptr<OperatorType> ir_operator_type) {
        std::shared_ptr<Operator> self(new Operator);
        self->type = ir_operator_type;
        self->block = Block::Create();
        std::transform(RANGE(ir_operator_type->ir_input_types), std::back_inserter(self->inputs),
                       [](std::shared_ptr<TensorType> ir_type) { return Input::Create(ir_type); });

        self->tag = "Operator";
        return self;
    }

    std::shared_ptr<OperatorType> GetOperatorType() { return Cast<OperatorType>(this->type); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Operator>(this->shared_from_this()));
    }

   public:
    std::shared_ptr<Block> block = nullptr;
    std::vector<std::shared_ptr<Input>> inputs;
    std::shared_ptr<pir::Function> pir_function = nullptr;
};

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

class Call : public Instruction {
   protected:
    Call() = default;

   public:
    static std::shared_ptr<Call> Create(std::shared_ptr<Operator> ir_operator,
                                        std::vector<std::shared_ptr<Tensor>> ir_inputs) {
        std::shared_ptr<Call> self(new Call);
        self->input_size = ir_inputs.size();
        self->OperandResize(1 + self->input_size);
        self->Operator(ir_operator);
        auto iter_inputs = ir_inputs.begin();
        for (int64_t i = 0; i < self->InputSize(); ++i, ++iter_inputs) {
            GALOIS_ASSERT(ir_operator->GetOperatorType()->ir_input_types[i] == ir_inputs[i]->type);
            self->Input(i, *iter_inputs);
        }
        self->type = ir_operator->GetOperatorType()->output_type;
        self->tag = "Call";
        return self;
    }

    std::shared_ptr<ir::Operator> Operator() { return Cast<class Operator>(this->GetOperand(0)); }
    void Operator(std::shared_ptr<ir::Operator> ir_operator) { this->SetOperand(0, ir_operator); }

    std::shared_ptr<Tensor> Input(int64_t i) { return this->GetOperand(1 + i); }
    void Input(int64_t i, std::shared_ptr<Tensor> ir_argument) {
        this->SetOperand(1 + i, ir_argument);
    }
    int64_t InputSize() { return this->input_size; }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Call>(this->shared_from_this()));
    }

   private:
    int64_t input_size;
    int64_t output_size;
};

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

class Free : public Instruction {
   protected:
    Free() = default;

   public:
    static std::shared_ptr<Free> Create(std::shared_ptr<Tensor> ir_tensor) {
        std::shared_ptr<Free> self(new Free);
        self->OperandResize(1);
        self->Tensor(ir_tensor);
        self->tag = "Free";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<class Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Free>(this->shared_from_this()));
    }
};

class Return : public Instruction {
   protected:
    Return() = default;

   public:
    static std::shared_ptr<Return> Create(std::shared_ptr<Tensor> ir_value) {
        std::shared_ptr<Return> self(new Return);
        self->OperandResize(1);
        self->type = ir_value->type;
        self->Tensor(ir_value);
        self->tag = "Return";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<class Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Return>(this->shared_from_this()));
    }
};

class UnaryIntrinsic : public Instruction {
   protected:
    UnaryIntrinsic() : Operand(nullptr, 0) {}

   public:
    static std::shared_ptr<UnaryIntrinsic> Create(std::string intrinsic_name,
                                                  std::shared_ptr<Tensor> ir_oprand,
                                                  bool llvm_prefix = true) {
        GALOIS_ASSERT(intrinsic_name.size());
        std::shared_ptr<UnaryIntrinsic> self(new UnaryIntrinsic);
        self->OperandResize(1);
        self->intrinsic_name = intrinsic_name;
        self->Operand = OperandProperty(
            Cast<Instruction>(self->shared_from_this()), 0
        );
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

    OperandProperty Operand;
};

class Builder;

inline std::vector<std::shared_ptr<TensorType>> GetTensorTypes(
    std::vector<std::shared_ptr<Tensor>> ir_tensors) {
    std::vector<std::shared_ptr<TensorType>> ir_types;
    for (auto ir_tensor : ir_tensors) {
        ir_types.push_back(ir_tensor->type);
    }
    return ir_types;
}

}  // namespace galois::ir
