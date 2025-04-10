#pragma once

namespace galois::ir {

class Tensor;
class Block;
class Grid;
class ArithmeticInstruction;
class Operator;
class BitCast;
class Alloca;
class Free;
class Return;
class PthreadBlock;
class Write;
class VectorBroadcast;
class Broadcast;
class Call;
class UnaryIntrinsic;
class GridIndex;
class SqueezeDimView;
class SliceView;
class Accessor;
class SqueezeView;

class Visitor : public std::enable_shared_from_this<Visitor> {
   public:
    virtual void Visit(std::shared_ptr<ir::Tensor> ir_tensor) {}
    virtual void Visit(std::shared_ptr<ir::Block> ir_block) {}
    virtual void Visit(std::shared_ptr<ir::Grid> ir_grid) {}
    virtual void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) {}
    virtual void Visit(std::shared_ptr<ir::Operator> ir_operator) {}
    virtual void Visit(std::shared_ptr<ir::BitCast> ir_bit_cast) {}
    virtual void Visit(std::shared_ptr<ir::Alloca> ir_alloca) {}
    virtual void Visit(std::shared_ptr<ir::Free> ir_free) {}
    virtual void Visit(std::shared_ptr<ir::Return> ir_return) {}
    virtual void Visit(std::shared_ptr<ir::PthreadBlock> ir_pthread_block) {}
    virtual void Visit(std::shared_ptr<ir::Write> ir_write) {}
    virtual void Visit(std::shared_ptr<ir::VectorBroadcast> ir_vector_broadcast) {}
    virtual void Visit(std::shared_ptr<ir::Broadcast> ir_broadcast) {}
    virtual void Visit(std::shared_ptr<ir::Call> ir_call) {}
    virtual void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) {}
    virtual void Visit(std::shared_ptr<ir::GridIndex> ir_grid_index) {}
    virtual void Visit(std::shared_ptr<ir::SqueezeDimView> ir_squeeze_dim_view) {}
    virtual void Visit(std::shared_ptr<ir::SliceView> ir_slice_view) {}
    virtual void Visit(std::shared_ptr<ir::Accessor> ir_accessor) {}
    virtual void Visit(std::shared_ptr<ir::SqueezeView> ir_squeeze_view) {}
};

}  // namespace galois::ir
