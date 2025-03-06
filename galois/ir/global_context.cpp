#include "galois/ir/global_context.h"

#include "galois/ir/ir.hpp"

namespace galois::ir {

GlobalContext::GlobalContext(int64_t target_bits) {
    // this->created_types.clear();
}

GlobalContext global_context = GlobalContext(64);
std::shared_ptr<TensorType> f32(FloatType::Create(32));
std::shared_ptr<TensorType> f64(FloatType::Create(64));
std::shared_ptr<TensorType> i8(IntType::Create(8, true));
std::shared_ptr<TensorType> i64(IntType::Create(64, true));

}  // namespace galois::ir
