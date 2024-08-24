#pragma once

#include <stdarg.h>

#include <algorithm>
#include <list>
#include <memory>
#include <numeric>
#include <regex>
#include <vector>

#include "Eigen/Dense"
#include "galois/assert.hpp"
#include "galois/helper.hpp"
#include "galois/ir/global_context.h"
#include "galois/named.hpp"

namespace prajna::ir {
class Value;
class Type;
class Function;
}  // namespace prajna::ir

namespace Eigen {
typedef Eigen::Matrix<int64_t, -1, -1> MatrixXi64;
typedef Eigen::Vector<int64_t, -1> VectorXi64;
typedef Eigen::RowVector<int64_t, -1> RowVectorXi64;
typedef Eigen::Vector<int64_t, 1> Vector1i64;
typedef Eigen::Vector<int64_t, 2> Vector2i64;
typedef Eigen::Vector<int64_t, 3> Vector3i64;
typedef Eigen::Vector<int64_t, 4> Vector4i64;
}  // namespace Eigen

typedef Eigen::Matrix<std::shared_ptr<prajna::ir::Value>, -1, -1> MatrixXprajna;
typedef Eigen::Vector<std::shared_ptr<prajna::ir::Value>, -1> VectorXprajna;
typedef Eigen::RowVector<std::shared_ptr<prajna::ir::Value>, -1> RowVectorXprajna;

namespace galois::ir {

class OperatorInstance;

class TensorType;

class Type : public Named {
   protected:
    Type() = default;

   public:
    virtual bool IsScalar() { return true; }

    virtual ~Type() {}

   public:
    // @ref https://llvm.org/docs/LangRef.html#langref-datalayout
    // bytes是多少可参阅datalyout的描述
    int64_t bytes = 0;
    std::shared_ptr<prajna::ir::Type> prajna_ir_type;
};

class Instruction;

struct InstructionAndOperandIndex {
    std::shared_ptr<Instruction> instruction;
    int64_t operand_index;
};

inline bool operator==(galois::ir::InstructionAndOperandIndex lhs,
                       galois::ir::InstructionAndOperandIndex rhs) {
    return lhs.instruction == rhs.instruction && lhs.operand_index == rhs.operand_index;
}

class RealNumberType : public Type {
   protected:
    RealNumberType() = default;

   public:
    int64_t bits = 0;
};

class FloatType : public RealNumberType {
   protected:
    FloatType() = default;

   public:
    static std::shared_ptr<FloatType> CreateImp(int64_t bits) {
        for (auto ir_type : global_context.created_types) {
            if (auto ir_float_type = Cast<FloatType>(ir_type)) {
                if (ir_float_type->bits == bits) {
                    return ir_float_type;
                }
            }
        }

        std::shared_ptr<FloatType> self(new FloatType);
        self->bits = bits;
        self->bytes = bits / 8;
        self->name = "f" + std::to_string(bits);
        self->fullname = "f" + std::to_string(bits);
        global_context.created_types.push_back(self);
        return self;
    }

    static std::shared_ptr<TensorType> Create(int64_t bits);
};

class IntType : public RealNumberType {
   protected:
    IntType() = default;

   public:
    static std::shared_ptr<IntType> Create(int64_t bits, bool is_signed) {
        for (auto ir_type : global_context.created_types) {
            if (auto ir_int_type = Cast<IntType>(ir_type)) {
                // if (Is<ir::CharType>(ir_type) || Is<ir::BoolType>(ir_type)) {
                //     continue;
                // }

                if (ir_int_type->bits == bits && ir_int_type->is_signed == is_signed) {
                    return ir_int_type;
                }
            }
        }

        std::shared_ptr<IntType> self(new IntType);
        self->bits = bits;
        self->bytes = (bits + 7) / 8;
        self->is_signed = is_signed;
        self->name = std::string(is_signed ? "i" : "u") + std::to_string(bits);
        self->fullname = std::string(is_signed ? "i" : "u") + std::to_string(bits);
        global_context.created_types.push_back(self);
        return self;
    }

   public:
    bool is_signed = true;
};

class VectorType : public Type {
   protected:
    VectorType() = default;

   public:
    static std::shared_ptr<VectorType> Create(std::shared_ptr<Type> value_type, int64_t size) {
        GALOIS_ASSERT(IsPowerOfTwo(size));
        for (auto ir_type : global_context.created_types) {
            if (auto ir_vectory_type = Cast<VectorType>(ir_type)) {
                if (ir_vectory_type->value_type == value_type && ir_vectory_type->size == size) {
                    return ir_vectory_type;
                }
            }
        }

        std::shared_ptr<VectorType> self(new VectorType);
        self->value_type = value_type;
        self->size = size;
        self->bytes = value_type->bytes * size;
        self->name = value_type->name + "[" + std::to_string(size) + "]";
        self->fullname = self->name;
        global_context.created_types.push_back(self);
        return self;
    }

   public:
    std::shared_ptr<Type> value_type = nullptr;
    int64_t size = 0;
};

enum struct Layout { RowMajor, ColumnMajor, View };

enum struct MemoryType { Host, Stack };

class TensorType : public Named, public std::enable_shared_from_this<TensorType> {
   public:
    static std::shared_ptr<TensorType> Create(std::shared_ptr<TensorType> value_type,
                                              Eigen::VectorXi64 shape,
                                              Layout layout = Layout::RowMajor) {
        for (auto ir_type : global_context.created_types) {
            if (auto ir_tensor_type = Cast<TensorType>(ir_type)) {
                if (ir_tensor_type->value_type == value_type &&
                    shape.size() == ir_tensor_type->shape.size() &&
                    shape == ir_tensor_type->shape && layout == ir_tensor_type->layout) {
                    return ir_tensor_type;
                }
            }
        }

        std::shared_ptr<TensorType> self(new TensorType);
        self->value_type = value_type;
        self->data_type = self->value_type->data_type;
        self->shape = shape;
        self->layout = layout;

        if (self->shape.size() == 0) {
            self->name = value_type->name;
            self->bytes = self->value_type->bytes;
        } else {
            if (layout == Layout::RowMajor) {
                self->stride.resize(shape.size());
                auto i = shape.size() - 1;
                self->stride[i] = 1;
                while (i > 0) {
                    i = i - 1;
                    self->stride[i] = shape[i + 1] * self->stride[i + 1];
                }
            } else {
                self->stride.resize(shape.size());
                self->stride[0] = 1;
                for (int64_t i = 1; i < shape.size(); ++i) {
                    self->stride[i] = shape[i - 1] * self->stride[i - 1];
                }
            }
            self->name = value_type->name + "[";
            for (auto i : shape) {
                self->name += std::to_string(i);
                self->name.push_back('x');
            }
            self->name.back() = ']';
            self->fullname = self->name;
            self->bytes = self->Size() * self->value_type->bytes;
        }

        global_context.created_types.push_back(self);
        return self;
    }

    std::shared_ptr<TensorType> ScalarType() {
        if (this->IsScalar()) {
            return this->shared_from_this();
        } else {
            return this->value_type->ScalarType();
        }
    }

    Eigen::VectorXi64 NormalizeShape() {
        if (this->IsScalar()) {
            return Eigen::VectorXi64::Ones(0);
        } else {
            auto value_type_normalize_shape = this->value_type->NormalizeShape();
            if (this->value_type->IsScalar()) {
                return this->shape;
            } else {
                return this->shape.array() * value_type_normalize_shape.array();
            }
        }
    }

    static std::shared_ptr<TensorType> CreateMatrixType(std::shared_ptr<TensorType> value_type,
                                                        int64_t rows, int64_t cols,
                                                        Layout layout = Layout::RowMajor) {
        Eigen::VectorXi64 shape(2);
        shape[0] = rows;
        shape[1] = cols;
        return TensorType::Create(value_type, shape, layout);
    }

    template <typename... Dims>
    std::shared_ptr<TensorType> Tensor(Dims... dims) {
        std::array<int64_t, std::tuple_size<std::tuple<Dims...>>::value> shape_array = {dims...};
        Eigen::VectorXi64 shape(shape_array.size());
        std::copy(RANGE(shape_array), shape.begin());
        return TensorType::Create(this->shared_from_this(), shape);
    }

    std::shared_ptr<TensorType> operator()(Eigen::VectorXi64 shape) {
        return TensorType::Create(this->shared_from_this(), shape);
    }

    int64_t Size() {
        int64_t sum = 1;
        for (int64_t i = 0; i < this->shape.size(); ++i) {
            sum *= this->shape[i];
        }
        return sum;
    }

    bool IsScalar() { return this->shape.size() == 0; }

   public:
    Eigen::VectorXi64 shape;
    std::shared_ptr<TensorType> value_type;
    std::shared_ptr<Type> data_type;
    Layout layout = Layout::RowMajor;
    Eigen::RowVectorXi64 stride;
    MemoryType memory_type = MemoryType::Host;
    int64_t bytes = 0;

    std::shared_ptr<prajna::ir::Type> prajna_ir_type;
};

class TensorTypePointer : public std::shared_ptr<TensorType> {
   public:
    TensorTypePointer(std::shared_ptr<TensorType> ir_type) : std::shared_ptr<TensorType>(ir_type) {}

    template <typename... Dims>
    TensorTypePointer operator()(Dims... dims) {
        std::array<int64_t, std::tuple_size<std::tuple<Dims...>>::value> shape_array = {dims...};
        Eigen::VectorXi64 shape(shape_array.size());
        std::copy(RANGE(shape_array), shape.begin());
        return TensorType::Create(*this, shape);
    }

    TensorTypePointer operator()(Eigen::VectorXi64 shape) {
        return TensorType::Create(*this, shape);
    }
};

class Viewer;

class Tensor : public Named, public std::enable_shared_from_this<Tensor> {
   protected:
    Tensor() {}

   public:
    static std::shared_ptr<ir::Tensor> Create(std::shared_ptr<TensorType> ir_type) {
        std::shared_ptr<ir::Tensor> self(new Tensor);
        self->type = ir_type;
        self->tag = "Value";
        return self;
    }

    virtual std::shared_ptr<Tensor> Clone() {
        GALOIS_UNREACHABLE;
        return nullptr;
    }

    virtual ~Tensor() {}

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

    std::shared_ptr<TensorType> GetTensorType() {
        auto ir_tensor_type = Cast<TensorType>(this->type);
        GALOIS_ASSERT(ir_tensor_type);
        return ir_tensor_type;
    }

    bool IsTensor() { return Is<TensorType>(this->type); }

    bool IsContinous() { return this->IsTensor() && !Is<Viewer>(this->shared_from_this()); }

   private:
    bool is_finalized = false;

   public:
    std::shared_ptr<ir::TensorType> type = nullptr;
    std::unordered_map<std::string, std::list<std::string>> annotation_dict;
    std::list<InstructionAndOperandIndex> instruction_with_index_list;

    std::shared_ptr<prajna::ir::Value> prajna_ir_value = nullptr;

    std::shared_ptr<OperatorInstance> inputted_operator = nullptr;
    std::shared_ptr<OperatorInstance> outputted_operator = nullptr;

    std::string tag = "Value";
};

class Constant : public Tensor {
   public:
    virtual ~Constant() {}

    virtual void Detach() { this->instruction_with_index_list.clear(); }
};

class ConstantRealNumber : public Constant {
   protected:
    ConstantRealNumber() = default;
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

   public:
    uint64_t value;
};

class ConstantFloat : public Constant {
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

    // std::shared_ptr<ir::Value> Clone(std::shared_ptr<FunctionCloner> function_cloner)
    // override {
    //     std::shared_ptr<ConstantFloat> ir_new(new ConstantFloat(*this));
    //     function_cloner->value_dict[shared_from_this()] = ir_new;
    //     return ir_new;
    // }

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

    std::shared_ptr<ir::Tensor> GetOperand(int64_t i) const {
        GALOIS_ASSERT(this->OperandSize() > i);
        return this->operands[i];
    };

    void SetOperand(int64_t i, std::shared_ptr<Tensor> ir_value) {
        GALOIS_ASSERT(ir_value);
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

    std::shared_ptr<Tensor> Clone() override {
        std::shared_ptr<Instruction> ir_new(new Instruction(this->OperandSize()));
        ir_new->CloneOperands(Cast<ir::Instruction>(this->shared_from_this()));
        return ir_new;
    }

    void CloneOperands(std::shared_ptr<ir::Instruction> ir_instrution) {
        for (int64_t i = 0; i < this->OperandSize(); ++i) {
            this->SetOperand(i, ir_instrution->GetOperand(i)->Clone());
        }
    }

    void Finalize() override {
        Tensor::Finalize();

        for (int64_t i = 0; i < OperandSize(); ++i) {
            auto ir_old_value = this->operands[i];
            if (ir_old_value) {
                // ir_old_value->instruction_with_index_list.remove(
                //     {Cast<Instruction>(this->shared_from_this()), i});
            }
        }

        this->OperandResize(0);
    }

   protected:
    std::vector<std::shared_ptr<Tensor>> operands;
};

class Write;

class AffineIndex : public Tensor {
   public:
    static std::shared_ptr<AffineIndex> Create(Eigen::MatrixXi64 transform_matrix,
                                               Eigen::VectorXi64 shift_vector) {
        std::shared_ptr<AffineIndex> self(new AffineIndex);
        self->tranform_matrix = transform_matrix;
        self->shift_vector = shift_vector;
        self->tag = "AffineIndex";
        return self;
    }

    Eigen::MatrixXi64 tranform_matrix;
    Eigen::VectorXi64 shift_vector;
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

    std::shared_ptr<ir::Tensor> Clone() override {
        std::shared_ptr<Accessor> ir_new(new Accessor(*this));
        return ir_new;
    }

   public:
    Eigen::MatrixXi64 transform_matrix;
    Eigen::VectorXi64 shift_vector;
    int64_t simd_size = 1;
    int64_t simd_shuffle = false;
};

class Viewer : public Tensor {
   public:
    static std::shared_ptr<Viewer> Create(std::shared_ptr<Tensor> ir_tensor,
                                          Eigen::MatrixXi64 transform_matrix,
                                          Eigen::VectorXi64 shift_vector) {
        std::shared_ptr<Viewer> self(new Viewer);
        GALOIS_ASSERT(ir_tensor->IsTensor());
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

    Eigen::MatrixXi64 transform_matrix;
    Eigen::VectorXi64 shift_vector;
    std::shared_ptr<Tensor> ir_tensor = nullptr;
};

class Slice : public Tensor {
   public:
    static std::shared_ptr<Slice> Create(std::shared_ptr<Accessor> ir_accessor_origin,
                                         Eigen::VectorXi64 shape) {
        std::shared_ptr<Slice> self(new Slice);
        GALOIS_ASSERT(ir_accessor_origin->Tensor()->IsContinous());
        self->origin = ir_accessor_origin;
        self->shape = shape;
        self->type = TensorType::Create(ir_accessor_origin->type, shape, Layout::View);
        self->tag = "Slice";
        return self;
    }

    std::shared_ptr<Accessor> origin;
    Eigen::VectorXi64 shape;
};

class ArithmeticInstruction : public Instruction {
    //    protected:
    // BinaryInstruction() = delete;
};

class Add : public ArithmeticInstruction {
   public:
    static std::shared_ptr<Add> Create(std::shared_ptr<ir::Tensor> ir_operand0,
                                       std::shared_ptr<ir::Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<Add> self(new Add);
        self->OperandResize(2);
        self->SetOperand(0, ir_operand0);
        self->SetOperand(1, ir_operand1);
        self->type = ir_operand0->type;
        self->tag = "Add";
        return self;
    }
};

class Sub : public ArithmeticInstruction {
   public:
    static std::shared_ptr<Sub> Create(std::shared_ptr<ir::Tensor> ir_operand0,
                                       std::shared_ptr<ir::Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<Sub> self(new Sub);
        self->OperandResize(2);
        self->SetOperand(0, ir_operand0);
        self->SetOperand(1, ir_operand1);
        self->type = ir_operand0->type;
        self->tag = "Sub";
        return self;
    }
};

class Mul : public ArithmeticInstruction {
   public:
    static std::shared_ptr<Mul> Create(std::shared_ptr<ir::Tensor> ir_operand0,
                                       std::shared_ptr<ir::Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<Mul> self(new Mul);
        self->OperandResize(2);
        self->SetOperand(0, ir_operand0);
        self->SetOperand(1, ir_operand1);
        self->type = ir_operand0->type;
        self->tag = "Mul";
        return self;
    }
};

class Div : public ArithmeticInstruction {
   public:
    static std::shared_ptr<Div> Create(std::shared_ptr<ir::Tensor> ir_operand0,
                                       std::shared_ptr<ir::Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<Div> self(new Div);
        self->OperandResize(2);
        self->SetOperand(0, ir_operand0);
        self->SetOperand(1, ir_operand1);
        self->type = ir_operand0->type;
        self->tag = "Div";
        return self;
    }
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
};

class Broadcast : public Instruction {
   protected:
    Broadcast() = default;

   public:
    static std::shared_ptr<Broadcast> Create(std::shared_ptr<ir::Tensor> ir_value,
                                             Eigen::VectorXi64 shape) {
        std::shared_ptr<Broadcast> self(new Broadcast);
        self->shape = shape;
        self->OperandResize(1);
        self->Tensor(ir_value);
        self->type = TensorType::Create(ir_value->type, shape);
        self->tag = "Broadcast";
        return self;
    }

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_value) { this->SetOperand(0, ir_value); }

    Eigen::VectorXi64 shape;
};

/// @brief For liked fmla.4s v2, v0, v1[0] instruction
class VectorBroadcast : public Instruction {
   protected:
    VectorBroadcast() = default;

   public:
    /// TODO: 需要进一步处理
    static std::shared_ptr<VectorBroadcast> Create(std::shared_ptr<ir::Tensor> ir_value,
                                                   int64_t lane_id) {
        std::shared_ptr<VectorBroadcast> self(new VectorBroadcast);
        self->OperandResize(1);
        self->Vector(ir_value);
        self->lane_id = lane_id;
        self->type = ir_value->type;
        self->tag = "VectorBroadcast";
        return self;
    }

    std::shared_ptr<ir::Tensor> Vector() { return this->GetOperand(0); }
    void Vector(std::shared_ptr<ir::Tensor> ir_value) { this->SetOperand(0, ir_value); }
    // Eigen::VectorXi64 shape;

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

    std::shared_ptr<ir::Tensor> Tensor() const { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> value) { this->SetOperand(0, value); }

    std::shared_ptr<ir::Tensor> Variable() const { return Cast<ir::Tensor>(this->GetOperand(1)); }
    void Variable(std::shared_ptr<ir::Tensor> accessor) { this->SetOperand(1, accessor); }

    std::shared_ptr<ir::Tensor> Clone() override {
        std::shared_ptr<Write> ir_new(new Write(*this));
        ir_new->CloneOperands(Cast<Instruction>(this->shared_from_this()));
        return ir_new;
    }
};

class Grid;

class Block : public Tensor {
   public:
    virtual ~Block() {}

   public:
    std::list<std::shared_ptr<Tensor>> values;
};

/// @brief An operator of tensors, which is liked as a node of ComputingGraph
class OperatorInstance : public Block {
   public:
    static std::shared_ptr<OperatorInstance> Create(
        std::vector<std::shared_ptr<TensorType>> ir_input_types,
        std::vector<std::shared_ptr<TensorType>> ir_output_types) {
        std::shared_ptr<OperatorInstance> self(new OperatorInstance);
        self->input_types = ir_input_types;
        self->output_types = ir_output_types;
        std::transform(RANGE(self->input_types), std::back_inserter(self->inputs),
                       [](std::shared_ptr<TensorType> ir_type) { return Tensor::Create(ir_type); });

        std::transform(RANGE(self->output_types), std::back_inserter(self->outputs),
                       [](std::shared_ptr<TensorType> ir_type) { return Tensor::Create(ir_type); });

        self->tag = "OperatorInstance";
        return self;
    }

   public:
    std::vector<std::shared_ptr<TensorType>> input_types;
    std::vector<std::shared_ptr<TensorType>> output_types;
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::shared_ptr<prajna::ir::Function> prajna_ir_function = nullptr;
};

class Grid : public Block {
   public:
    static std::shared_ptr<Grid> Create() {
        std::shared_ptr<Grid> self(new Grid);
        self->tag = "Grid";
        return self;
    }

    static std::shared_ptr<Grid> Create(Eigen::VectorXi64 shape) {
        std::shared_ptr<Grid> self(new Grid);
        self->shape = shape;
        self->tag = "Grid";
        return self;
    }

    int64_t GetAffineDimSize() const {
        if (this->parent_parallel && !this->is_local) {
            return this->shape.size() + this->parent_parallel->GetAffineDimSize();
        } else {
            return this->shape.size();
        }
    }

    Eigen::VectorXi64 shape;
    std::shared_ptr<OperatorInstance> parent_operator = nullptr;
    std::shared_ptr<Grid> parent_parallel = nullptr;
    VectorXprajna prajna_ir_index_vector;
    bool is_local = true;
};

class BitCast : public Instruction {
   public:
    static std::shared_ptr<BitCast> Create(std::shared_ptr<Tensor> ir_value,
                                           std::shared_ptr<TensorType> ir_type) {
        GALOIS_ASSERT(ir_type);
        GALOIS_ASSERT(ir_value->IsContinous());
        std::shared_ptr<BitCast> self(new BitCast);
        GALOIS_ASSERT(ir_value->type->bytes == ir_type->bytes);
        // auto ir_old_tensor_type = ir_value->type;
        // GALOIS_ASSERT(ir_old_tensor_type->Size() * ir_old_tensor_type->value_type->bytes
        // ==
        //                 ir_tensor_type->Size() * ir_tensor_type->value_type->bytes);
        self->OperandResize(1);
        self->Tensor(ir_value);
        self->type = ir_type;
        self->tag = "BitCast";
        return self;
    }

    std::shared_ptr<ir::Tensor> Tensor() const { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_value) { this->SetOperand(0, ir_value); }

    std::shared_ptr<ir::Tensor> Clone() override {
        std::shared_ptr<BitCast> ir_new(new BitCast(*this));
        ir_new->CloneOperands(Cast<Instruction>(this->shared_from_this()));
        return ir_new;
    }
};

inline std::shared_ptr<Tensor> CreateTensor(std::shared_ptr<TensorType> ir_value_type,
                                            Eigen::VectorXi64 shape,
                                            Layout layout = Layout::RowMajor) {
    auto self = Tensor::Create(TensorType::Create(ir_value_type, shape, layout));
    return self;
}

class Call : public Instruction {
   protected:
    Call() = default;

   public:
    static std::shared_ptr<Call> Create(std::shared_ptr<OperatorInstance> ir_operator,
                                        std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                                        std::vector<std::shared_ptr<ir::Tensor>> ir_outputs) {
        std::shared_ptr<Call> self(new Call);
        self->input_size = ir_inputs.size();
        self->output_size = ir_outputs.size();
        self->OperandResize(1 + self->input_size + self->output_size);
        self->OperatorInstance(ir_operator);
        auto iter_inputs = ir_inputs.begin();
        for (int64_t i = 0; i < self->InputSize(); ++i, ++iter_inputs) {
            self->Input(i, *iter_inputs);
        }
        auto iter_outputs = ir_outputs.begin();
        for (int64_t i = 0; i < self->OutputSize(); ++i, ++iter_outputs) {
            self->Output(i, *iter_outputs);
        }
        // TODO: 需要进一步处理
        self->type = nullptr;
        self->tag = "Call";
        return self;
    }

    std::shared_ptr<ir::OperatorInstance> OperatorInstance() {
        return Cast<class OperatorInstance>(this->GetOperand(0));
    }
    void OperatorInstance(std::shared_ptr<ir::OperatorInstance> ir_operator) {
        this->SetOperand(0, ir_operator);
    }

    std::shared_ptr<ir::Tensor> Input(int64_t i) { return this->GetOperand(1 + i); }
    void Input(int64_t i, std::shared_ptr<ir::Tensor> ir_argument) {
        this->SetOperand(1 + i, ir_argument);
    }
    int64_t InputSize() { return this->input_size; }

    std::shared_ptr<ir::Tensor> Output(int64_t i) {
        return this->GetOperand(1 + this->input_size + i);
    }
    void Output(int64_t i, std::shared_ptr<ir::Tensor> ir_argument) {
        this->SetOperand(1 + this->input_size + i, ir_argument);
    }
    int64_t OutputSize() { return this->output_size; }

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
};

class Builder;

class Kernel {
   public:
    virtual bool Match(std::vector<std::shared_ptr<Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) = 0;

    virtual void Build(std::vector<std::shared_ptr<Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) = 0;

    virtual ~Kernel() = default;
};

// inline std::shared_ptr<TensorType> ToType(std::string type_str) {
//     std::smatch sm;
//     // float32[100x200], $1是类型, $2位数, S3S4...是shape
//     std::regex e("([a-z]+)(\\d+)(\\[\\w*\\])*");
//     std::regex_match(type_str, sm, e);
//     GALOIS_ASSERT(sm.size() >= 4);
//     auto bits = std::stoi(sm[2]);
//     std::shared_ptr<TensorType> value_type = nullptr;
//     if (sm[1] == "float") {
//         value_type = FloatType::Create(bits);
//     }
//     if (sm[1] == "int") {
//         value_type = IntType::Create(bits, true);
//     }
//     if (sm[1] == "uint") {
//         value_type = IntType::Create(bits, false);
//     }
//     for (int64_t i = 3; i < sm.size(); ++i) {
//         auto shape_vec = split(sm[i].str(), 'x');
//         Eigen::VectorXi64 shape(shape_vec.size());
//         for (auto j = 0; j < shape.size(); ++j) {
//         d:
//             shape[j] = std::stoi(shape_vec[j]);
//         }
//         value_type = TensorType::Create(value_type, shape);
//     }
//     return value_type;
// }

inline std::shared_ptr<TensorType> FloatType::Create(int64_t bits) {
    auto ir_float_type = FloatType::CreateImp(bits);
    for (auto ir_type : global_context.created_types) {
        if (auto ir_tensor_type = Cast<TensorType>(ir_type)) {
            if (ir_tensor_type->IsScalar() && ir_tensor_type->data_type == ir_float_type) {
                return ir_tensor_type;
            }
        }
    }

    std::shared_ptr<TensorType> self(new TensorType);
    self->value_type = nullptr;
    self->data_type = ir_float_type;
    self->shape.resize(0);
    self->stride.resize(0);

    self->name = self->data_type->name + "[]";
    self->fullname = self->name;
    self->bytes = self->data_type->bytes;
    global_context.created_types.push_back(self);
    return self;
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
