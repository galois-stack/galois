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

template <typename Matrix_>
inline void RemoveRow(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (index < numRows)
        matrix.block(index, 0, numRows - index, numCols) = matrix.bottomRows(numRows - index);

    matrix.conservativeResize(numRows, numCols);
}

template <typename Matrix_>
inline void RemoveColumn(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (index < numCols)
        matrix.block(0, index, numRows, numCols - index) = matrix.rightCols(numCols - index);

    matrix.conservativeResize(numRows, numCols);
}

namespace galois::ir {

class Instruction;

struct InstructionAndOperandIndex {
    std::shared_ptr<Instruction> instruction;
    int64_t operand_index;
};

inline bool operator==(galois::ir::InstructionAndOperandIndex lhs,
                       galois::ir::InstructionAndOperandIndex rhs) {
    return lhs.instruction == rhs.instruction && lhs.operand_index == rhs.operand_index;
}

class Tensor : public Named, public std::enable_shared_from_this<Tensor> {
   protected:
    Tensor() {}

   public:
    /// @brief 释放不必要的依赖, 解除循环引用
    virtual void Detach() {
        // 只是解除依赖, 不是销毁数据,
        this->instruction_with_index_list.clear();
    }

    /// @brief 实例需要销毁前调用
    virtual void Finalize() {
        GALOIS_ASSERT(this->instruction_with_index_list.size() == 0);
        this->Detach();
        this->is_finalized = true;
    }

    std::shared_ptr<Block> ParentBlock() {
        if (!this->parent_block) {
            // GALOIS_ASSERT(Is<Block>(this->shared_from_this()));
            return Cast<Block>(this->shared_from_this());
        } else {
            return Cast<Tensor>(this->parent_block)->ParentBlock();
        }
    }

    bool IsInsideOf(std::shared_ptr<Block> ir_block) {
        if (this->parent_block == ir_block) {
            return true;
        }

        if (this->parent_block) {
            return Cast<Tensor>(this->parent_block)->IsInsideOf(ir_block);
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
    std::shared_ptr<Block> parent_block = nullptr;
    std::shared_ptr<pir::Value> pir_value = nullptr;
    std::string tag = "Tensor";
};

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

    void Finalize() override {
        Tensor::Finalize();

        for (int64_t i = 0; i < OperandSize(); ++i) {
            auto ir_old_value = this->operands[i];
            if (ir_old_value) {
                ir_old_value->instruction_with_index_list.remove(
                    {Cast<Instruction>(this->shared_from_this()), i});
            }
        }

        this->OperandResize(0);
    }

   protected:
    std::vector<std::shared_ptr<Tensor>> operands;
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

class Viewer : public Instruction {
   public:
    static std::shared_ptr<Viewer> Create(std::shared_ptr<Tensor> ir_tensor,
                                          Eigen::MatrixXi64 transform_matrix,
                                          Eigen::VectorXi64 shift_vector) {
        std::shared_ptr<Viewer> self(new Viewer);
        self->ir_tensor = ir_tensor;
        self->transform_matrix = transform_matrix;
        self->type = ir_tensor->type;
        self->shift_vector = shift_vector;
        self->tag = "Viewer";
        return self;
    }

    static std::shared_ptr<Viewer> Shift(std::shared_ptr<Tensor> ir_tensor,
                                         Eigen::VectorXi64 shift_vector) {
        auto tensor_rank = ir_tensor->type->shape.size();
        auto identity_matrix = Eigen::MatrixXi64::Identity(tensor_rank, tensor_rank);
        GALOIS_ASSERT(shift_vector.size() == tensor_rank);
        return Create(ir_tensor, identity_matrix, shift_vector);
    }

    static std::shared_ptr<Viewer> Stride(std::shared_ptr<Tensor> ir_tensor,
                                          Eigen::VectorXi64 stride_vector) {
        auto tensor_rank = ir_tensor->type->shape.size();
        GALOIS_ASSERT(stride_vector.size() == tensor_rank);
        Eigen::MatrixXi64 transform_matrix = Eigen::MatrixXi64::Zero(tensor_rank, tensor_rank);
        for (int64_t i = 0; i < tensor_rank; ++i) {
            transform_matrix(i, i) = stride_vector[i];
        }
        return Create(ir_tensor, transform_matrix, Eigen::VectorXi64::Zero(tensor_rank));
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Viewer>(this->shared_from_this()));
    }

    Eigen::MatrixXi64 transform_matrix;
    Eigen::VectorXi64 shift_vector;
    std::shared_ptr<Tensor> ir_tensor = nullptr;
};

class SliceView : public Instruction {
   public:
    static std::shared_ptr<SliceView> Create(std::shared_ptr<Accessor> ir_origin,
                                             Eigen::VectorXi64 shape) {
        GALOIS_ASSERT(ir_origin->Tensor()->type->shape.size() == shape.size());
        std::shared_ptr<SliceView> self(new SliceView);
        self->OperandResize(1);
        self->Origin(ir_origin);
        self->shape = shape;

        auto stride = ir_origin->Tensor()->type->stride;
        GALOIS_ASSERT(ir_origin->Tensor()->type->value_type);
        self->type = ir::TensorType::Create(ir_origin->Tensor()->type->value_type, shape, stride);
        self->tag = "SliceView";
        return self;
    }

    std::shared_ptr<Accessor> Origin() { return Cast<Accessor>(this->GetOperand(0)); }
    void Origin(std::shared_ptr<ir::Accessor> ir_accessor) { this->SetOperand(0, ir_accessor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SliceView>(this->shared_from_this()));
    }

    Eigen::VectorXi64 shape;
};

class SqueezeDimView : public Instruction {
   public:
    static std::shared_ptr<SqueezeDimView> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim) {
        std::shared_ptr<SqueezeDimView> self(new SqueezeDimView);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

        auto shape = ir_tensor->type->shape;
        auto stride = ir_tensor->type->stride;
        RemoveRow(shape, dim);
        RemoveColumn(stride, dim);
        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->dim = dim;
        self->tag = "Squeeze";
        return self;
    }

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SqueezeDimView>(this->shared_from_this()));
    }

    int64_t dim;
};

class SqueezeView : public Instruction {
   public:
    static std::shared_ptr<SqueezeView> Create(std::shared_ptr<Tensor> ir_tensor) {
        std::shared_ptr<SqueezeView> self(new SqueezeView);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

        int64_t valid_shape_size = 0;
        Eigen::VectorXi64 shape(ir_tensor->type->shape.size());
        Eigen::VectorXi64 stride(ir_tensor->type->stride.size());
        for (int64_t i = 0; i < ir_tensor->type->shape.size(); ++i) {
            if (ir_tensor->type->shape[i] != 1) {
                shape[valid_shape_size] = ir_tensor->type->shape[i];
                stride[valid_shape_size] = ir_tensor->type->stride[i];
                valid_shape_size++;
            }
        }
        shape.conservativeResize(valid_shape_size);
        stride.conservativeResize(valid_shape_size);
        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->tag = "Squeeze";
        return self;
    }

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SqueezeView>(this->shared_from_this()));
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

class Prefetch : public Instruction {
   public:
    static std::shared_ptr<Prefetch> Create(std::shared_ptr<ir::Accessor> ir_address) {
        std::shared_ptr<Prefetch> self(new Prefetch);
        self->OperandResize(1);
        self->Address(ir_address);
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
};

class Broadcast : public Instruction {
   protected:
    Broadcast() = default;

   public:
    static std::shared_ptr<Broadcast> Create(std::shared_ptr<Tensor> ir_value,
                                             Eigen::VectorXi64 shape) {
        std::shared_ptr<Broadcast> self(new Broadcast);
        self->shape = shape;
        self->OperandResize(1);
        self->Tensor(ir_value);
        self->type = TensorType::Create(ir_value->type, shape);
        self->tag = "Broadcast";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_value) { this->SetOperand(0, ir_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Broadcast>(this->shared_from_this()));
    }

    Eigen::VectorXi64 shape;
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

class VoidType : public TensorType {
   protected:
    VoidType() = default;

   public:
    static std::shared_ptr<VoidType> Create() {
        for (auto ir_type : global_context.created_types) {
            if (auto ir_void_type = Cast<VoidType>(ir_type)) {
                return ir_void_type;
            }
        }

        std::shared_ptr<VoidType> self(new VoidType);
        self->name = "void";
        self->fullname = "void";
        global_context.created_types.push_back(self);
        return self;
    }
};

class OperatorType : public TensorType {
   public:
    static std::shared_ptr<OperatorType> Create(
        std::vector<std::shared_ptr<TensorType>> ir_in_types,
        std::shared_ptr<TensorType> ir_out_types) {
        std::shared_ptr<OperatorType> self(new OperatorType);
        self->ir_input_types = ir_in_types;
        self->output_type = ir_out_types;
        self->name = "(";
        for (auto ir_in_type : ir_in_types) {
            self->name += ir_in_type->name + ",";
        }

        self->name += ") -> " + ir_out_types->name;
        self->fullname = self->name;
        return self;
    }

   public:
    std::vector<std::shared_ptr<TensorType>> ir_input_types;
    std::shared_ptr<TensorType> output_type;
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

    int64_t GetAffineDimSize() const {
        if (this->parent_grid && !this->is_local) {
            return this->shape.size() + this->parent_grid->GetAffineDimSize();
        } else {
            return this->shape.size();
        }
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Grid>(this->shared_from_this()));
    }

    Eigen::VectorXi64 shape;
    std::shared_ptr<Block> block = nullptr;
    std::shared_ptr<GridIndex> index = nullptr;
    std::shared_ptr<Operator> parent_operator = nullptr;
    std::shared_ptr<Grid> parent_grid = nullptr;
    bool enable_multi_thread = false;
    bool unroll_grid = false;

    bool is_local = true;
};

class PthreadBlock : public Block {
   public:
    static std::shared_ptr<PthreadBlock> Create() {
        std::shared_ptr<PthreadBlock> self(new PthreadBlock);
        self->tag = "PthreadBlock";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<PthreadBlock>(this->shared_from_this()));
    }
};

class BitCast : public Instruction {
   public:
    static std::shared_ptr<BitCast> Create(std::shared_ptr<Tensor> ir_value,
                                           std::shared_ptr<TensorType> ir_type) {
        GALOIS_ASSERT(ir_type);
        std::shared_ptr<BitCast> self(new BitCast);
        GALOIS_ASSERT(ir_value->type->bytes == ir_type->bytes);
        self->OperandResize(1);
        self->Tensor(ir_value);
        self->type = ir_type;
        self->tag = "BitCast";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() const { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<class Tensor> ir_value) { this->SetOperand(0, ir_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<BitCast>(this->shared_from_this()));
    }
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

class Alloca : public Instruction {
   protected:
    Alloca() = default;

   public:
    static std::shared_ptr<Alloca> Create(std::shared_ptr<TensorType> ir_type) {
        std::shared_ptr<Alloca> self(new Alloca);
        self->type = ir_type;
        self->tag = "Alloca";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Alloca>(this->shared_from_this()));
    }
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
    UnaryIntrinsic() = default;

   public:
    static std::shared_ptr<UnaryIntrinsic> Create(std::string intrinsic_name,
                                                  std::shared_ptr<Tensor> ir_oprand) {
        GALOIS_ASSERT(intrinsic_name.size());
        std::shared_ptr<UnaryIntrinsic> self(new UnaryIntrinsic);
        self->OperandResize(1);
        self->intrinsic_name = intrinsic_name;
        self->SetOperand(0, ir_oprand);
        self->type = ir_oprand->type;
        self->tag = "UnaryIntrinsic";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<UnaryIntrinsic>(this->shared_from_this()));
    }

    std::shared_ptr<Tensor> Operand() { return this->GetOperand(0); }
    void Operand(std::shared_ptr<Tensor> ir_oprand) { this->SetOperand(0, ir_oprand); }

   public:
    std::string intrinsic_name;
};

class Builder;

template <typename DataType, typename... Args>
inline std::shared_ptr<TensorType> CreateScalarType(Args... args) {
    auto ir_data_type = DataType::Create(args...);
    auto fullname = ir_data_type->name + "[]";
    for (auto ir_type : global_context.created_types) {
        if (ir_type->fullname == fullname) {
            return Cast<TensorType>(ir_type);
        }
    }

    std::shared_ptr<TensorType> self(new TensorType);
    self->value_type = nullptr;
    self->shape.resize(0);
    self->stride.resize(0);
    self->fullname = fullname;
    global_context.created_types.push_back(self);
    return self;
}
inline std::vector<std::shared_ptr<TensorType>> GetTensorTypes(
    std::vector<std::shared_ptr<Tensor>> ir_tensors) {
    std::vector<std::shared_ptr<TensorType>> ir_types;
    for (auto ir_tensor : ir_tensors) {
        ir_types.push_back(ir_tensor->type);
    }
    return ir_types;
}

}  // namespace galois::ir

template <>
struct std::hash<galois::ir::InstructionAndOperandIndex> {
    std::int64_t operator()(galois::ir::InstructionAndOperandIndex inst_with_idx) const noexcept {
        std::int64_t h1 =
            std::hash<std::shared_ptr<galois::ir::Instruction>>{}(inst_with_idx.instruction);
        std::int64_t h2 = std::hash<int64_t>{}(inst_with_idx.operand_index);
        // 这里哈希函数应该不重要, 应该不会导致性能问题
        return h1 ^ (h2 << 1);
    }
};
