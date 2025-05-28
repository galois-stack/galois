#pragma once

#include <queue>
#include <stack>

#include "galois/ir/ir.hpp"
#include "galois/op/creator.hpp"

namespace galois::op {
class MatrixMultiplyMicroKernel;
}  // namespace galois::op

namespace galois::ir {

class Builder : public std::enable_shared_from_this<Builder> {
   public:
    static std::shared_ptr<Builder> Create() {
        std::shared_ptr<Builder> self(new Builder);
        return self;
    }

    Eigen::VectorXi64 CreateMatrixShape(int64_t rows, int64_t cols) {
        Eigen::VectorXi64 shape(2);
        shape[0] = rows;
        shape[1] = cols;
        return shape;
    }

    template <typename Tensor_, typename... Args_>
    std::shared_ptr<Tensor_> Create(Args_&&... args) {
        auto ir_tensor = Tensor_::Create(std::forward<Args_>(args)...);
        static_assert(std::is_base_of<Tensor, Tensor_>::value);
        this->Insert(ir_tensor);
        return ir_tensor;
    }

    std::shared_ptr<ir::Block> CurrentBlock() {
        GALOIS_ASSERT(this->block_stack.size());
        return this->block_stack.top();
    }

    std::shared_ptr<ir::Grid> CurrentGrid() {
        GALOIS_ASSERT(this->grid_stack.size());
        return this->grid_stack.top();
    }

    std::shared_ptr<ir::Operator> CurrentOperator() {
        GALOIS_ASSERT(this->operator_stack.size());
        return this->operator_stack.top();
    }

    void Insert(std::shared_ptr<Tensor> ir_tensor) {
        GALOIS_ASSERT(this->iterator_stack.size());
        this->CurrentBlock()->insert(this->iterator_stack.top(), ir_tensor);
        ir_tensor->parent_block = this->CurrentBlock();
    }

    std::tuple<std::shared_ptr<Grid>, std::unique_ptr<ScopeGuard>> CreateGrid(
        Eigen::VectorXi64 shape) {
        auto ir_grid = this->Create<Grid>(shape);
        this->grid_stack.push(ir_grid);
        this->block_stack.push(ir_grid->block);
        this->iterator_stack.push(ir_grid->block->end());
        auto scope_guard = ScopeGuard::Create([&]() {
            this->grid_stack.pop();
            this->block_stack.pop();
            this->iterator_stack.pop();
        });
        return {ir_grid, std::move(scope_guard)};
    }

    std::tuple<std::shared_ptr<Block>, std::unique_ptr<ScopeGuard>> CreateBlock() {
        auto ir_block = this->Create<Block>();
        this->block_stack.push(ir_block);
        this->iterator_stack.push(ir_block->end());
        auto scope_guard = ScopeGuard::Create([&]() {
            this->block_stack.pop();
            this->iterator_stack.pop();
        });
        return {ir_block, std::move(scope_guard)};
    }

    std::tuple<std::shared_ptr<PthreadBlock>, std::unique_ptr<ScopeGuard>> CreatePthreadBlock() {
        auto ir_pthread_block = Cast<PthreadBlock>(this->Create<PthreadBlock>());
        this->block_stack.push(ir_pthread_block);
        this->iterator_stack.push(ir_pthread_block->end());
        auto scope_guard = ScopeGuard::Create([&]() {
            this->block_stack.pop();
            this->iterator_stack.pop();
        });
        return {ir_pthread_block, std::move(scope_guard)};
    }

    std::tuple<std::shared_ptr<Operator>, std::unique_ptr<ScopeGuard>> CreateOperator(
        std::shared_ptr<OperatorType> ir_operator_type, std::string name) {
        auto ir_operator = Operator::Create(ir_operator_type);
        ir_operator->name = name;
        ir_operator->fullname = this->operator_stack.size()
                                    ? ir_operator->name + this->operator_stack.top()->fullname
                                    : ir_operator->name;
        if (this->block_stack.size()) {
            this->Insert(ir_operator);
        }

        this->operator_stack.push(ir_operator);
        this->block_stack.push(ir_operator->block);
        this->iterator_stack.push(ir_operator->block->end());

        auto scope_guard = ScopeGuard::Create([&]() {
            this->operator_stack.pop();
            this->block_stack.pop();
            this->iterator_stack.pop();
        });
        return {ir_operator, std::move(scope_guard)};
    }

    std::shared_ptr<Accessor> CreateAccessor(std::shared_ptr<Tensor> ir_tensor) {
        auto ir_tensor_type = ir_tensor->type;

        auto grid_rank = this->grid_stack.empty() ? 0 : this->CurrentGrid()->shape.size();
        Eigen::MatrixXi64 transform_matrix =
            Eigen::MatrixXi64 ::Zero(ir_tensor_type->shape.size(), grid_rank);
        auto ir_accessor = this->Create<Accessor>(
            ir_tensor, transform_matrix, Eigen::VectorXi64::Zero(ir_tensor_type->shape.size()));
        return ir_accessor;
    }

    std::shared_ptr<Accessor> CreateIdentityAccessor(std::shared_ptr<Tensor> ir_tensor) {
        auto ir_tensor_type = ir_tensor->type;
        Eigen::MatrixXi64 transform_matrix = Eigen::MatrixXi64::Identity(
            ir_tensor_type->shape.size(), this->CurrentGrid()->shape.size());
        auto ir_accessor = this->Create<Accessor>(
            ir_tensor, transform_matrix, Eigen::VectorXi64::Zero(ir_tensor_type->shape.size()));
        return ir_accessor;
    }

    std::shared_ptr<ir::Constant> GetZero(std::shared_ptr<ir::TensorType> ir_type) {
        return this->GetConstant(ir_type, 0);
    }

    std::shared_ptr<ir::Constant> GetConstant(std::shared_ptr<ir::TensorType> ir_type, double v) {
        if (auto ir_float_type = Cast<FloatType>(ir_type)) {
            return this->Create<ir::ConstantFloat>(ir_type, v);
        } else if (auto ir_int_type = Cast<IntType>(ir_type)) {
            return this->Create<ir::ConstantInt>(ir_type, static_cast<int64_t>(v));
        } else {
            GALOIS_ASSERT(false);
        }

        return nullptr;
    }

    std::shared_ptr<ir::ArithmeticInstruction> Add(std::shared_ptr<ir::Tensor> ir_tensor1,
                                                   std::shared_ptr<ir::Tensor> ir_tensor2) {
        return this->Create<ir::ArithmeticInstruction>(ir::ArithmeticInstruction::Add, ir_tensor1,
                                                       ir_tensor2);
    }

    std::shared_ptr<ir::ArithmeticInstruction> Sub(std::shared_ptr<ir::Tensor> ir_tensor1,
                                                   std::shared_ptr<ir::Tensor> ir_tensor2) {
        return this->Create<ir::ArithmeticInstruction>(ir::ArithmeticInstruction::Sub, ir_tensor1,
                                                       ir_tensor2);
    }

    std::shared_ptr<ir::ArithmeticInstruction> Mul(std::shared_ptr<ir::Tensor> ir_tensor1,
                                                   std::shared_ptr<ir::Tensor> ir_tensor2) {
        return this->Create<ir::ArithmeticInstruction>(ir::ArithmeticInstruction::Mul, ir_tensor1,
                                                       ir_tensor2);
    }

    std::shared_ptr<ir::ArithmeticInstruction> Div(std::shared_ptr<ir::Tensor> ir_tensor1,
                                                   std::shared_ptr<ir::Tensor> ir_tensor2) {
        return this->Create<ir::ArithmeticInstruction>(ir::ArithmeticInstruction::Div, ir_tensor1,
                                                       ir_tensor2);
    }

    std::shared_ptr<ir::Call> Call(std::shared_ptr<ir::Operator> ir_operator,
                                   std::vector<std::shared_ptr<Tensor>> ir_inputs) {
        return this->Create<ir::Call>(ir_operator, ir_inputs);
    }

    std::shared_ptr<ir::Alloca> Alloca(std::shared_ptr<ir::TensorType> ir_tensor_type) {
        return this->Create<ir::Alloca>(ir_tensor_type);
    }

    std::shared_ptr<ir::Write> Write(std::shared_ptr<ir::Tensor> ir_value,
                                     std::shared_ptr<ir::Tensor> ir_variable) {
        return this->Create<ir::Write>(ir_value, ir_variable);
    }

    std::shared_ptr<ir::BitCastView> BitCastView(std::shared_ptr<ir::Tensor> ir_value,
                                                 std::shared_ptr<ir::TensorType> ir_type) {
        return this->Create<ir::BitCastView>(ir_value, ir_type);
    }

    std::shared_ptr<ir::Return> Return(std::shared_ptr<ir::Tensor> ir_value) {
        return this->Create<ir::Return>(ir_value);
    }

    template <typename Creator, typename... CreatorArgs>
    std::shared_ptr<Operator> CreateOperatorByCreator(
        std::vector<std::shared_ptr<TensorType>> ir_input_types, CreatorArgs... creator_args) {
        auto sp_creator = Creator::Create(creator_args...);
        auto ir_output_type = sp_creator->InferType(ir_input_types);
        auto ir_operator_type = OperatorType::Create(ir_input_types, ir_output_type);
        auto [ir_operator, op_scope] =
            this->CreateOperator(ir_operator_type, sp_creator->name + std::to_string(this->id++));
        std::vector<std::shared_ptr<Tensor>> ir_inputs;
        std::transform(RANGE(ir_operator->inputs), std::back_inserter(ir_inputs),
                       [](std::shared_ptr<Tensor> ir_input) { return ir_input; });
        sp_creator->Express(ir_inputs, this->shared_from_this());
        return ir_operator;
    };

    template <typename Creator, typename... CreatorArgs>
    std::shared_ptr<Tensor> ExpressCreator(std::vector<std::shared_ptr<Tensor>> inputs,
                                           CreatorArgs... creator_args) {
        std::vector<std::shared_ptr<TensorType>> ir_input_types;
        std::transform(RANGE(inputs), std::back_inserter(ir_input_types),
                       [](std::shared_ptr<Tensor> ir_tensor) { return ir_tensor->type; });
        auto ir_operator = this->CreateOperatorByCreator<Creator>(ir_input_types, creator_args...);
        return this->Call(ir_operator, inputs);
    };

   public:
    std::stack<std::shared_ptr<Grid>> grid_stack;
    std::stack<std::shared_ptr<Operator>> operator_stack;
    std::stack<std::shared_ptr<Block>> block_stack;
    std::stack<std::list<std::shared_ptr<Tensor>>::iterator> iterator_stack;

    size_t id = 0;

    std::list<std::shared_ptr<op::MatrixMultiplyMicroKernel>> matrix_multiply_kernel_queue;
};

}  // namespace galois::ir
