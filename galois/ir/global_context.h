#pragma once

#include <list>
#include <memory>
#include <vector>

namespace galois {
class Named;
}

namespace galois::ir {

class Type;
class TensorType;

class GlobalContext {
   public:
    GlobalContext(int64_t target_bits);

    /// @brief 用于存储已经构造了的类型
    /// @note 需要使用vector来确保构造的顺序, 因为后面的codegen需要顺序正确
    std::list<std::shared_ptr<TensorType>> created_types;
};

extern GlobalContext global_context;
extern std::shared_ptr<TensorType> f16;
extern std::shared_ptr<TensorType> f32;
extern std::shared_ptr<TensorType> f64;
extern std::shared_ptr<TensorType> i8;
extern std::shared_ptr<TensorType> i16;
extern std::shared_ptr<TensorType> i32;
extern std::shared_ptr<TensorType> i64;
extern std::shared_ptr<TensorType> u8;
extern std::shared_ptr<TensorType> u16;
extern std::shared_ptr<TensorType> u32;
extern std::shared_ptr<TensorType> u64;
extern std::shared_ptr<TensorType> bool_;

}  // namespace galois::ir
