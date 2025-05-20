#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir {

class CloneVisitor : public Visitor {
   private:
    CloneVisitor() = default;

   public:
    static std::shared_ptr<CloneVisitor> Create() {
        std::shared_ptr<CloneVisitor> self(new CloneVisitor);
        return self;
    }

    void Visit(std::shared_ptr<Block> ir_block) override {
        if (tensor_dict.count(ir_block)) {
            return;
        }

        auto ir_new = Block::Create();
        for (auto ir_tensor : *ir_block) {
            ir_tensor->ApplyVisitor(this->shared_from_this());
            ir_new->push_back(tensor_dict[ir_tensor]);
        }
        tensor_dict[ir_block] = ir_new;
    }

    void Visit(std::shared_ptr<Grid> ir_grid) override {
        if (tensor_dict.count(ir_grid)) {
            return;
        }

        ir_grid->block->ApplyVisitor(this->shared_from_this());
        auto ir_new = Grid::Create(ir_grid->shape);
        ir_new->block = Cast<Block>(tensor_dict[ir_grid->block]);
        tensor_dict[ir_grid] = ir_new;
    }

    void Visit(std::shared_ptr<Accessor> ir_accessor) override {
        if (tensor_dict.count(ir_accessor)) {
            return;
        }

        ir_accessor->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = Accessor::Create(tensor_dict[ir_accessor->Tensor()],
                                       ir_accessor->transform_matrix, ir_accessor->shift_vector);
        tensor_dict[ir_accessor] = ir_new;
    }

    void Visit(std::shared_ptr<GridIndex> ir_grid_index) override {
        if (tensor_dict.count(ir_grid_index)) {
            return;
        }

        auto ir_new = GridIndex::Create(ir_grid_index->type->shape[0]);
        tensor_dict[ir_grid_index] = ir_new;
    }

    void VisitOperands(std::shared_ptr<Instruction> ir_instruction) {
        for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
            ir_instruction->GetOperand(i)->ApplyVisitor(this->shared_from_this());
        }
    }

    void Visit(std::shared_ptr<ArithmeticInstruction> ir_arithmetic_instruction) override {
        if (tensor_dict.count(ir_arithmetic_instruction)) {
            return;
        }

        ir_arithmetic_instruction->GetOperand(0)->ApplyVisitor(this->shared_from_this());
        ir_arithmetic_instruction->GetOperand(1)->ApplyVisitor(this->shared_from_this());
        auto ir_new =
            ArithmeticInstruction::Create(ir_arithmetic_instruction->operation,
                                          tensor_dict[ir_arithmetic_instruction->GetOperand(0)],
                                          tensor_dict[ir_arithmetic_instruction->GetOperand(1)]);
        tensor_dict[ir_arithmetic_instruction] = ir_new;
    }

    void Visit(std::shared_ptr<BitCastView> ir_bit_cast) override {
        if (tensor_dict.count(ir_bit_cast)) {
            return;
        }
        ir_bit_cast->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = BitCastView::Create(tensor_dict[ir_bit_cast->Tensor()], ir_bit_cast->type);
        tensor_dict[ir_bit_cast] = ir_new;
    }

    void Visit(std::shared_ptr<Alloca> ir_alloca) override {
        if (tensor_dict.count(ir_alloca)) {
            return;
        }
        auto ir_new = Alloca::Create(ir_alloca->type);
        tensor_dict[ir_alloca] = ir_new;
    }

    void Visit(std::shared_ptr<Free> ir_free) override {
        if (tensor_dict.count(ir_free)) {
            return;
        }
        ir_free->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = Free::Create(tensor_dict[ir_free->Tensor()]);
        tensor_dict[ir_free] = ir_new;
    }

    void Visit(std::shared_ptr<Return> ir_return) override {
        if (tensor_dict.count(ir_return)) {
            return;
        }
        ir_return->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = Return::Create(tensor_dict[ir_return->Tensor()]);
        tensor_dict[ir_return] = ir_new;
    }

    void Visit(std::shared_ptr<PthreadBlock> ir_pthread_block) override {
        if (tensor_dict.count(ir_pthread_block)) {
            return;
        }
        GALOIS_UNIMPLEMENT;
    }

    void Visit(std::shared_ptr<Write> ir_write) override {
        if (tensor_dict.count(ir_write)) {
            return;
        }
        ir_write->Tensor()->ApplyVisitor(this->shared_from_this());
        ir_write->Variable()->ApplyVisitor(this->shared_from_this());
        auto ir_new =
            Write::Create(tensor_dict[ir_write->Tensor()], tensor_dict[ir_write->Variable()]);
        tensor_dict[ir_write] = ir_new;
    }

    void Visit(std::shared_ptr<VectorBroadcast> ir_vector_broadcast) override {
        if (tensor_dict.count(ir_vector_broadcast)) {
            return;
        }
        ir_vector_broadcast->Vector()->ApplyVisitor(this->shared_from_this());
        auto ir_new =
            VectorBroadcast::Create(tensor_dict[ir_vector_broadcast->Vector()],
                                    ir_vector_broadcast->type, ir_vector_broadcast->lane_id);
        tensor_dict[ir_vector_broadcast] = ir_new;
    }

    void Visit(std::shared_ptr<Call> ir_call) override {
        if (tensor_dict.count(ir_call)) {
            return;
        }
        this->VisitOperands(ir_call);
        std::vector<std::shared_ptr<Tensor>> ir_inputs;
        for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
            ir_inputs.push_back(tensor_dict[ir_call->Input(i)]);
        }
        auto ir_new = Call::Create(Cast<Operator>(tensor_dict[ir_call->Operator()]), ir_inputs);
        tensor_dict[ir_call] = ir_new;
    }

    void Visit(std::shared_ptr<UnaryIntrinsic> ir_unary_intrinsic) override {
        if (tensor_dict.count(ir_unary_intrinsic)) {
            return;
        }
        ir_unary_intrinsic->Operand()->ApplyVisitor(this->shared_from_this());
        auto ir_new = UnaryIntrinsic::Create(ir_unary_intrinsic->intrinsic_name,
                                             tensor_dict[ir_unary_intrinsic->Operand()]);
        tensor_dict[ir_unary_intrinsic] = ir_new;
    }

    void Visit(std::shared_ptr<SqueezeDimView> ir_squeeze_dim_view) override {
        if (tensor_dict.count(ir_squeeze_dim_view)) {
            return;
        }
        ir_squeeze_dim_view->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = SqueezeDimView::Create(tensor_dict[ir_squeeze_dim_view->Tensor()],
                                             ir_squeeze_dim_view->dim);
        tensor_dict[ir_squeeze_dim_view] = ir_new;
    }

    void Visit(std::shared_ptr<UnsqueezeDimView> ir_unsqueeze_dim_view) override {
        if (tensor_dict.count(ir_unsqueeze_dim_view)) {
            return;
        }
        ir_unsqueeze_dim_view->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = UnsqueezeDimView::Create(tensor_dict[ir_unsqueeze_dim_view->Tensor()],
                                               ir_unsqueeze_dim_view->dim);
        tensor_dict[ir_unsqueeze_dim_view] = ir_new;
    }

    void Visit(std::shared_ptr<SliceView> ir_slice_view) override {
        if (tensor_dict.count(ir_slice_view)) {
            return;
        }
        GALOIS_UNIMPLEMENT;
    }

    void Visit(std::shared_ptr<SqueezeView> ir_squeeze_view) override {
        if (tensor_dict.count(ir_squeeze_view)) {
            return;
        }
        ir_squeeze_view->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = SqueezeView::Create(tensor_dict[ir_squeeze_view->Tensor()]);
        tensor_dict[ir_squeeze_view] = ir_new;
    }

    void Visit(std::shared_ptr<FlattenView> ir_flatten_view) override {
        if (tensor_dict.count(ir_flatten_view)) {
            return;
        }
        ir_flatten_view->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = FlattenView::Create(tensor_dict[ir_flatten_view->Tensor()]);
        tensor_dict[ir_flatten_view] = ir_new;
    }

    void Visit(std::shared_ptr<TransposeView> ir_transpose_view) override {
        if (tensor_dict.count(ir_transpose_view)) {
            return;
        }
        ir_transpose_view->Tensor()->ApplyVisitor(this->shared_from_this());
        auto ir_new = TransposeView::Create(tensor_dict[ir_transpose_view->Tensor()],
                                            ir_transpose_view->dim0, ir_transpose_view->dim1);
        tensor_dict[ir_transpose_view] = ir_new;
    }

    void Visit(std::shared_ptr<Operator> ir_operator) override {
        if (tensor_dict.count(ir_operator)) {
            return;
        }
        tensor_dict[ir_operator] = ir_operator;
    }

    std::shared_ptr<ir::Tensor> Clone(std::shared_ptr<Tensor> ir_tensor) {
        ir_tensor->ApplyVisitor(this->shared_from_this());
        return tensor_dict[ir_tensor];
    }

    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> tensor_dict;
};

}  // namespace galois::ir
