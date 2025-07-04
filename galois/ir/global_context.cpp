#include "galois/ir/global_context.h"

#include "galois/ir/ir.hpp"

namespace galois::ir {

GlobalContext::GlobalContext(int64_t target_bits) {
    // this->created_types.clear();
}

GlobalContext global_context = GlobalContext(64);

std::shared_ptr<TensorType> f16(FloatType::Create(16));
std::shared_ptr<TensorType> f32(FloatType::Create(32));
std::shared_ptr<TensorType> f64(FloatType::Create(64));
std::shared_ptr<TensorType> i8(IntType::Create(8, true));
std::shared_ptr<TensorType> i16(IntType::Create(16, true));
std::shared_ptr<TensorType> i32(IntType::Create(32, true));
std::shared_ptr<TensorType> i64(IntType::Create(64, true));
std::shared_ptr<TensorType> u8(IntType::Create(8, false));
std::shared_ptr<TensorType> u16(IntType::Create(16, false));
std::shared_ptr<TensorType> u32(IntType::Create(32, false));
std::shared_ptr<TensorType> u64(IntType::Create(64, false));

std::shared_ptr<TensorType> bool_(BoolType::Create());
std::shared_ptr<TensorType> void_(VoidType::Create());

}  // namespace galois::ir
