#pragma once

#include "tensor.hpp"
namespace galois::ir {

class Tensor;
class Constant;
class ConstantRealNumber;
class ConstantInt;
class ConstantFloat;
class Block;
class Grid;
class Instruction;
class ArithmeticInstruction;
class Operator;
class Input;
class Alloca;
class Free;
class Return;
class Prefetch;
class Write;
class VectorBroadcast;
class Call;
class UnaryIntrinsic;
class GridIndex;
class Accessor;
class Indexing;

namespace view {
class BitCast;
class Broadcast;
class Viewer;
class SqueezeDim;
class UnsqueezeDim;
class Slice;
class Squeeze;
class Flatten;
class Transpose;
}  // namespace view

namespace io {
class LoadBinary;
}

class Visitor : public std::enable_shared_from_this<Visitor> {
   public:
    virtual void Visit(std::shared_ptr<ir::Constant> ir_constant) {}
    virtual void Visit(std::shared_ptr<ir::ConstantRealNumber> ir_constant_real_number) {}
    virtual void Visit(std::shared_ptr<ir::ConstantInt> ir_constant_int) {}
    virtual void Visit(std::shared_ptr<ir::ConstantFloat> ir_constant_float) {}
    virtual void Visit(std::shared_ptr<ir::Tensor> ir_tensor) {}
    virtual void Visit(std::shared_ptr<ir::Input> ir_input) {}
    virtual void Visit(std::shared_ptr<ir::Block> ir_block) {}
    virtual void Visit(std::shared_ptr<ir::Grid> ir_grid) {}
    virtual void Visit(std::shared_ptr<ir::Accessor> ir_accessor) {}
    virtual void Visit(std::shared_ptr<ir::Indexing> ir_index) {}
    virtual void Visit(std::shared_ptr<ir::GridIndex> ir_grid_index) {}
    virtual void Visit(std::shared_ptr<ir::Instruction> ir_instruction) {}
    virtual void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) {}
    virtual void Visit(std::shared_ptr<ir::view::BitCast> ir_bit_cast) {}
    virtual void Visit(std::shared_ptr<ir::view::Broadcast> ir_broad_cast) {}
    virtual void Visit(std::shared_ptr<ir::Alloca> ir_alloca) {}
    virtual void Visit(std::shared_ptr<ir::Free> ir_free) {}
    virtual void Visit(std::shared_ptr<ir::Return> ir_return) {}
    virtual void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) {}
    virtual void Visit(std::shared_ptr<ir::Write> ir_write) {}
    virtual void Visit(std::shared_ptr<ir::VectorBroadcast> ir_vector_broadcast) {}
    virtual void Visit(std::shared_ptr<ir::Call> ir_call) {}
    virtual void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) {}
    virtual void Visit(std::shared_ptr<ir::view::Viewer> ir_viewer) {}
    virtual void Visit(std::shared_ptr<ir::view::SqueezeDim> ir_squeeze_dim_view) {}
    virtual void Visit(std::shared_ptr<ir::view::UnsqueezeDim> ir_unsqueeze_dim_view) {}
    virtual void Visit(std::shared_ptr<ir::view::Slice> ir_slice_view) {}
    virtual void Visit(std::shared_ptr<ir::view::Squeeze> ir_squeeze_view) {}
    virtual void Visit(std::shared_ptr<ir::view::Flatten> ir_flatten_view) {}
    virtual void Visit(std::shared_ptr<ir::view::Transpose> ir_transpose_view) {}
    virtual void Visit(std::shared_ptr<ir::Operator> ir_operator) {}
    virtual void Visit(std::shared_ptr<ir::io::LoadBinary> ir_load_binary) {}
};

}  // namespace galois::ir
