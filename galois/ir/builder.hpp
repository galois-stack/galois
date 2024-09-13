#pragma once

#include <queue>
#include <stack>

#include "galois/ir/ir.hpp"

namespace galois::ir {

class Builder {
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
        if (auto ir_grid = Cast<Grid>(ir_tensor)) {
            this->Insert(ir_grid->indices);
        }
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

    std::shared_ptr<ir::OperatorFunction> CurrentOperator() {
        GALOIS_ASSERT(this->operator_stack.size());
        return this->operator_stack.top();
    }

    void Insert(std::shared_ptr<Tensor> ir_tensor) {
        GALOIS_ASSERT(this->iterator_stack.size());
        this->CurrentBlock()->values.insert(this->iterator_stack.top(), ir_tensor);
        ir_tensor->parent_block = this->CurrentBlock();
    }

    std::tuple<std::shared_ptr<Grid>, std::unique_ptr<ScopeGuard>> CreateGrid(
        Eigen::VectorXi64 shape) {
        auto ir_grid = Cast<Grid>(this->Create<Grid>(shape));
        this->grid_stack.push(ir_grid);
        this->block_stack.push(ir_grid);
        this->iterator_stack.push(ir_grid->values.end());
        auto scope_guard = ScopeGuard::Create([&]() {
            this->grid_stack.pop();
            this->block_stack.pop();
            this->iterator_stack.pop();
        });
        return {ir_grid, std::move(scope_guard)};
    }

    std::tuple<std::shared_ptr<PthreadBlock>, std::unique_ptr<ScopeGuard>> CreatePthreadBlock() {
        auto ir_pthread_block = Cast<PthreadBlock>(this->Create<PthreadBlock>());
        this->block_stack.push(ir_pthread_block);
        this->iterator_stack.push(ir_pthread_block->values.end());
        auto scope_guard = ScopeGuard::Create([&]() {
            this->block_stack.pop();
            this->iterator_stack.pop();
        });
        return {ir_pthread_block, std::move(scope_guard)};
    }

    std::tuple<std::shared_ptr<OperatorFunction>, std::unique_ptr<ScopeGuard>> CreateOperator(
        std::vector<std::shared_ptr<TensorType>> ir_input_types,
        std::vector<std::shared_ptr<TensorType>> ir_output_types, std::string name) {
        auto ir_operator = OperatorFunction::Create(ir_input_types, ir_output_types);
        ir_operator->name = name;
        ir_operator->fullname = this->operator_stack.size()
                                    ? ir_operator->name + this->operator_stack.top()->fullname
                                    : ir_operator->name;
        if (this->block_stack.size()) {
            this->Insert(ir_operator);
        }

        this->operator_stack.push(ir_operator);
        this->block_stack.push(ir_operator);
        this->iterator_stack.push(ir_operator->values.end());
        this->temp_tensors_stack.push(std::vector<std::shared_ptr<Tensor>>());

        auto scope_guard = ScopeGuard::Create([&]() {
            this->operator_stack.pop();
            this->block_stack.pop();
            this->iterator_stack.pop();
            this->temp_tensors_stack.pop();
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
        ir_accessor->Indices(this->grid_stack.top()->indices);
        return ir_accessor;
    }

    std::shared_ptr<Accessor> CreateIdentityAccessor(std::shared_ptr<Tensor> ir_tensor) {
        auto ir_tensor_type = ir_tensor->type;
        Eigen::MatrixXi64 transform_matrix = Eigen::MatrixXi64::Identity(
            ir_tensor_type->shape.size(), this->CurrentGrid()->shape.size());
        auto ir_accessor = this->Create<Accessor>(
            ir_tensor, transform_matrix, Eigen::VectorXi64::Zero(ir_tensor_type->shape.size()));
        ir_accessor->Indices(this->grid_stack.top()->indices);
        return ir_accessor;
    }

   public:
    std::stack<std::shared_ptr<Grid>> grid_stack;
    std::stack<std::shared_ptr<OperatorFunction>> operator_stack;
    std::stack<std::shared_ptr<Block>> block_stack;
    std::stack<std::list<std::shared_ptr<Tensor>>::iterator> iterator_stack;
    std::stack<std::vector<std::shared_ptr<Tensor>>> temp_tensors_stack;

    std::list<std::shared_ptr<Kernel>> kernel_queue;
};

class OperatorCreator : public Named {
   public:
    virtual std::vector<std::shared_ptr<Tensor>> GetOutputs(
        std::vector<std::shared_ptr<Tensor>> ir_inputs, std::shared_ptr<Builder> ir_builder) = 0;

    virtual void AffineExpress(std::vector<std::shared_ptr<Tensor>> ir_inputs,
                               std::vector<std::shared_ptr<Tensor>> ir_outputs,
                               std::shared_ptr<Builder> ir_builder) = 0;
};

class CopyOperatorCreator : public OperatorCreator {
   public:
    std::vector<std::shared_ptr<Tensor>> GetOutputs(std::vector<std::shared_ptr<Tensor>> ir_inputs,
                                                    std::shared_ptr<Builder> ir_builder) override {
        auto input = ir_inputs.front();
        auto ir_tensor = ir_builder->Create<Tensor>(input->type);
        return {ir_tensor};
    }

    void AffineExpress(std::vector<std::shared_ptr<Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        if (ir_inputs[0]->type->IsScalar()) {
            ir_builder->Create<Write>(ir_inputs[0], ir_outputs[0]);
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_inputs[0]->type->shape);
        auto ir_input_accessor = ir_builder->CreateIdentityAccessor(ir_inputs[0]);
        auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_outputs[0]);
        this->AffineExpress({ir_input_accessor}, {ir_output_accessor}, ir_builder);
    }
};

// class AddOperatorCreator : public OperatorCreator {
//    public:
//     std::vector<std::shared_ptr<Tensor>> GetOutputs(std::vector<std::shared_ptr<Tensor>>
//     ir_inputs,
//                                                    std::shared_ptr<Builder> ir_builder) override
//                                                    {
//         auto input = ir_inputs.front();
//         auto ir_tensor = ir_builder->Create<Tensor>(input->type);
//         return {ir_tensor};
//     }

//     void AffineExpress(std::vector<std::shared_ptr<Tensor>> ir_inputs,
//                        std::vector<std::shared_ptr<Tensor>> ir_outputs,
//                        std::shared_ptr<Builder> ir_builder) override {
//         if (ir_inputs[0]->type->IsScalar()) {
//             ir_builder->Create<Write>(ir_builder->Create<Add>(ir_inputs[0], ir_inputs[1]),
//                                       ir_outputs[0]);
//         }

//         auto [ir_grid, scope_guard] =
//             ir_builder->CreateGrid(ir_inputs[0]->type->shape);
//         auto ir_input_accessor0 = ir_builder->CreateIdentityAccessor(ir_inputs[0]);
//         auto ir_input_accessor1 = ir_builder->CreateIdentityAccessor(ir_inputs[1]);
//         auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_outputs[0]);
//         this->AffineExpress({ir_input_accessor0, ir_input_accessor1}, {ir_output_accessor},
//                             ir_builder);
//     }
// };

}  // namespace galois::ir
