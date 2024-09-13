#include "galois/ir/global_context.h"

#include "galois/ir/ir.hpp"

namespace galois::ir {

GlobalContext::GlobalContext(int64_t target_bits) {
    // this->created_types.clear();
}

GlobalContext global_context = GlobalContext(64);
TensorTypePointer f32(CreateScalarType<FloatType>(32));
TensorTypePointer i64(CreateScalarType<IntType>(64, true));
TensorTypePointer bool_(CreateScalarType<IntType>(1, false));

}  // namespace galois::ir
