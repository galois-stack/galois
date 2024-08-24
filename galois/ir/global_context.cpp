#include "galois/ir/global_context.h"

#include "galois/ir/ir.hpp"

namespace galois::ir {

GlobalContext::GlobalContext(int64_t target_bits) {
    this->created_types.clear();
}  // namespace prajna::ir

GlobalContext global_context = GlobalContext(64);
TensorTypePointer f32(FloatType::Create(32));

}  // namespace galois::ir
