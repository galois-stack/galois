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
class TensorTypePointer;

class GlobalContext {
   public:
    GlobalContext(int64_t target_bits);

    /// @brief 用于存储已经构造了的类型
    /// @note 需要使用vector来确保构造的顺序, 因为后面的codegen需要顺序正确
    std::list<std::shared_ptr<Named>> created_types;
};

extern GlobalContext global_context;
extern TensorTypePointer f32;
extern TensorTypePointer i64;
extern TensorTypePointer bool_;

}  // namespace galois::ir
