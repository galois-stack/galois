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
#include "galois/named.hpp"

namespace Eigen {
typedef Eigen::Matrix<int64_t, -1, -1> MatrixXi64;
typedef Eigen::Vector<int64_t, -1> VectorXi64;
typedef Eigen::RowVector<int64_t, -1> RowVectorXi64;
typedef Eigen::Vector<int64_t, 1> Vector1i64;
typedef Eigen::Vector<int64_t, 2> Vector2i64;
typedef Eigen::Vector<int64_t, 3> Vector3i64;
typedef Eigen::Vector<int64_t, 4> Vector4i64;
}  // namespace Eigen

namespace prajna::ir {
class Value;
class Type;
class Function;
}  // namespace prajna::ir

typedef Eigen::Matrix<std::shared_ptr<prajna::ir::Value>, -1, -1> MatrixXprajna;
typedef Eigen::Vector<std::shared_ptr<prajna::ir::Value>, -1> VectorXprajna;
typedef Eigen::RowVector<std::shared_ptr<prajna::ir::Value>, -1> RowVectorXprajna;

namespace galois::ir {

namespace pir = prajna::ir;

class TensorType : public Named, public std::enable_shared_from_this<TensorType> {
   public:
    static Eigen::VectorXi64 GetStride(Eigen::VectorXi64 shape) {
        Eigen::VectorXi64 stride(shape.size());
        auto i = shape.size() - 1;
        stride[i] = 1;
        while (i > 0) {
            i = i - 1;
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        return stride;
    }

    static std::shared_ptr<TensorType> Create(std::shared_ptr<TensorType> value_type,
                                              Eigen::VectorXi64 shape) {
        return TensorType::Create(value_type, shape, TensorType::GetStride(shape));
    }

    static std::shared_ptr<TensorType> Create(std::shared_ptr<TensorType> value_type,
                                              Eigen::VectorXi64 shape,
                                              Eigen::RowVectorXi64 stride) {
        // 如果shape为0， 直接退化为value_type
        if (!shape.size()) {
            return value_type;
        }

        for (auto ir_type : global_context.created_types) {
            if (auto ir_tensor_type = Cast<TensorType>(ir_type)) {
                if (ir_tensor_type->value_type == value_type &&
                    shape.size() == ir_tensor_type->shape.size() &&
                    shape == ir_tensor_type->shape &&
                    stride.size() == ir_tensor_type->stride.size() &&
                    stride == ir_tensor_type->stride) {
                    return ir_tensor_type;
                }
            }
        }

        std::shared_ptr<TensorType> self(new TensorType);
        self->value_type = value_type;
        self->shape = shape;
        self->stride = stride;

        self->name = value_type->name + "[";
        for (auto i : shape) {
            self->name += std::to_string(i);
            self->name.push_back('x');
        }

        self->name.back() = ']';
        self->fullname = self->name;
        self->bytes = self->Size() * self->value_type->bytes;
        global_context.created_types.push_back(self);

        return self;
    }

    std::shared_ptr<TensorType> DataType() {
        if (this->IsScalar()) {
            return this->shared_from_this();
        } else {
            return this->value_type->DataType();
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
                // 需要同一维度的，NormalizeShape才有意义
                GALOIS_ASSERT(this->shape.size() == value_type_normalize_shape.size());
                return this->shape.array() * value_type_normalize_shape.array();
            }
        }
    }

    static std::shared_ptr<TensorType> CreateMatrixType(std::shared_ptr<TensorType> value_type,
                                                        int64_t rows, int64_t cols) {
        Eigen::VectorXi64 shape(2);
        shape[0] = rows;
        shape[1] = cols;
        return TensorType::Create(value_type, shape);
    }

    std::shared_ptr<TensorType> Tile(Eigen::VectorXi64 shape) {
        return TensorType::Create(this->shared_from_this(), shape);
    }

    template <typename... Dims>
    std::shared_ptr<TensorType> Tile(Dims... dims) {
        std::array<int64_t, std::tuple_size<std::tuple<Dims...>>::value> shape_array = {dims...};
        Eigen::VectorXi64 shape(shape_array.size());
        std::copy(RANGE(shape_array), shape.begin());
        return TensorType::Create(this->shared_from_this(), shape);
    }

    int64_t Size() {
        return std::accumulate(RANGE(this->shape), 1, [](int64_t x, int64_t y) { return x * y; });
    }

    int64_t NormalizeSize() {
        if (this->IsScalar()) {
            return this->Size();
        } else {
            return this->Size() * this->value_type->NormalizeSize();
        }
    }

    bool IsMatch(std::shared_ptr<TensorType> other) {
        return this->value_type == other->value_type && this->shape == other->shape;
    }

    std::shared_ptr<TensorType> DenseType() {
        return TensorType::Create(this->value_type, this->shape);
    }

    virtual bool IsScalar() { return this->shape.size() == 0; }

   public:
    Eigen::VectorXi64 shape;
    std::shared_ptr<TensorType> value_type;
    int64_t bytes = 0;
    Eigen::RowVectorXi64 stride;
    std::shared_ptr<pir::Type> pir_type = nullptr;
    bool enable_multi_thread = false;
    bool unroll_grid = false;
};

class RealNumberType : public TensorType {
   protected:
    RealNumberType() = default;

   public:
    int64_t bits = 0;
};

class FloatType : public RealNumberType {
   protected:
    FloatType() = default;

   public:
    static std::shared_ptr<FloatType> Create(int64_t bits) {
        for (auto ir_type : global_context.created_types) {
            if (auto ir_float_type = Cast<FloatType>(ir_type)) {
                if (ir_float_type->bits == bits) {
                    return ir_float_type;
                }
            }
        }

        std::shared_ptr<FloatType> self(new FloatType);
        self->value_type = nullptr;
        self->shape.resize(0);
        self->stride.resize(0);

        self->bits = bits;
        self->bytes = bits / 8;
        self->name = "f" + std::to_string(bits);
        self->fullname = "f" + std::to_string(bits);
        global_context.created_types.push_back(self);
        return self;
    }
};

class IntType : public RealNumberType {
   protected:
    IntType() = default;

   public:
    static std::shared_ptr<IntType> Create(int64_t bits, bool is_signed) {
        std::shared_ptr<IntType> self(new IntType);
        for (auto ir_type : global_context.created_types) {
            if (auto ir_int_type = Cast<IntType>(ir_type)) {
                if (ir_int_type->bits == bits && ir_int_type->is_signed == is_signed) {
                    return ir_int_type;
                }
            }
        }

        self->bits = bits;
        self->is_signed = is_signed;
        self->bytes = (bits + 7) / 8;
        self->name = std::string(is_signed ? "i" : "u") + std::to_string(bits);
        self->fullname = std::string(is_signed ? "i" : "u") + std::to_string(bits);
        global_context.created_types.push_back(self);
        return self;
    }

   public:
    bool is_signed = true;
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

class BoolType : public TensorType {
protected:
    BoolType() = default;

public:
    static std::shared_ptr<BoolType> Create() {
        for (auto ir_type : global_context.created_types) {
            if (auto ir_bool_type = Cast<BoolType>(ir_type)) {
                return ir_bool_type;
            }
        }

        std::shared_ptr<BoolType> self(new BoolType);
        self->value_type = nullptr;
        self->shape.resize(0);
        self->stride.resize(0);

        self->bytes = 1;  // 通常1字节存储bool
        self->name = "bool";
        self->fullname = "bool";

        global_context.created_types.push_back(self);
        return self;
    }
};

}  // namespace galois::ir
