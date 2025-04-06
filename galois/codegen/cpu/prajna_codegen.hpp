#pragma once

#include <algorithm>
#include <cmath>

#include "galois/helper.hpp"
#include "galois/ir/ir.hpp"
#include "galois/transform/transform.hpp"
#include "prajna/ir/ir.hpp"
#include "prajna/lowering/ir_builder.hpp"

namespace galois::codegen::cpu {

namespace pir = prajna::ir;

class PrajnaCodegen {
   protected:
    PrajnaCodegen() = default;

   public:
    static std::shared_ptr<PrajnaCodegen> Create(
        std::shared_ptr<prajna::lowering::SymbolTable> prajna_symbol_table) {
        auto self = std::shared_ptr<PrajnaCodegen>(new PrajnaCodegen);
        auto pir_module = pir::Module::Create();
        auto pir_logger = prajna::Logger::Create("");
        pir_module->symbol_table = prajna_symbol_table;
        self->pir_builder =
            prajna::lowering::IrBuilder::Create(prajna_symbol_table, pir_module, pir_logger);
        return self;
    }

    void DeclareIntrinsic() {
        auto pir_i32_type = pir_builder->GetInt32Type();
        auto pir_i32_pointer_type = pir::PointerType::Create(pir_i32_type);
        auto pir_llvm_prefetch_function_type = pir::FunctionType::Create(
            {pir_i32_pointer_type, pir_i32_type, pir_i32_type}, pir::VoidType::Create());
        pir_builder->CreateFunction(prajna::ast::Identifier("llvm.prefetch"),
                                    pir_llvm_prefetch_function_type);
    }

    std::shared_ptr<pir::Type> EmitType(std::shared_ptr<ir::TensorType> ir_type) {
        if (ir_type->pir_type) {
            return ir_type->pir_type;
        }

        if (Is<ir::VoidType>(ir_type)) {
            ir_type->pir_type = pir::VoidType::Create();
            return ir_type->pir_type;
        }

        if (ir_type->IsScalar()) {
            if (auto ir_float_type = Cast<ir::FloatType>(ir_type)) {
                ir_type->pir_type = pir::FloatType::Create(ir_float_type->bits);
                return ir_type->pir_type;
            }

            if (auto ir_int_type = Cast<ir::IntType>(ir_type)) {
                ir_type->pir_type = pir::IntType::Create(ir_int_type->bits, ir_int_type->is_signed);
                return ir_type->pir_type;
            }

            GALOIS_TODO;
        }

        if (ir_type->shape.size() == 1 && ir_type->value_type->IsScalar() &&
            IsPowerOfTwo(ir_type->shape[0])) {
            ir_type->pir_type =
                pir::VectorType::Create(this->EmitType(ir_type->value_type), ir_type->Size());
            return ir_type->pir_type;
        } else {
            ir_type->pir_type =
                pir::ArrayType::Create(this->EmitType(ir_type->value_type), ir_type->Size());
            return ir_type->pir_type;
        }

        GALOIS_TODO;
        return nullptr;
    }

    void EmitGridIndexVector(std::shared_ptr<ir::GridIndex> ir_indices) {
        this->EmitType(ir_indices->type);
        ir_indices->pir_value = pir_builder->Create<pir::LocalVariable>(ir_indices->type->pir_type);
    }

    void EmitGrid(std::shared_ptr<ir::Grid> ir_grid) {
        GALOIS_ASSERT(!ir_grid->pir_value);
        if (this->grid_stack.size()) {
            ir_grid->parent_grid = this->grid_stack.top();
        }
        this->grid_stack.push(ir_grid);
        auto guard = ScopeGuard::Create([=]() { this->grid_stack.pop(); });

        std::unique_ptr<ScopeGuard> thread_guard;
        if (ir_grid->enable_multi_thread) {
            auto pir_thread_num = pir_builder->GetInt32Constant(12);
            auto pir_i32_function_type_i64 = pir::FunctionType::Create(
                {pir::IntType::Create(32, true)}, pir::IntType::Create(64, true));
            auto pir_thpool_init =
                pir_builder->GetIntrinsic("thpool_init", pir_i32_function_type_i64);
            auto pir_thread_pool = pir_builder->Create<pir::Call>(pir_thpool_init, pir_thread_num);
            this->pir_thpool = pir_thread_pool;
            auto pir_i64_function_type = pir::FunctionType::Create({pir::IntType::Create(64, true)},
                                                                   pir::VoidType::Create());
            thread_guard = std::move(ScopeGuard::Create([=]() {
                pir_builder->Create<pir::Call>(
                    pir_builder->GetIntrinsic("thpool_wait", pir_i64_function_type),
                    pir_thread_pool);
                pir_builder->Create<pir::Call>(
                    pir_builder->GetIntrinsic("thpool_destroy", pir_i64_function_type),
                    pir_thread_pool);
            }));
        }

        this->EmitGridIndexVector(ir_grid->index);

        for (int64_t i = 0; i < ir_grid->shape.size(); ++i) {
            auto pir_first_value = pir_builder->GetInt64Constant(0);
            auto pir_last_value = pir_builder->GetInt64Constant(ir_grid->shape[i]);
            auto ir_loop_block = pir::Block::Create();
            auto pir_scalar_index =
                pir_builder->Create<pir::LocalVariable>(pir_builder->GetInt64Type());
            pir_scalar_index->fullname = "idx";
            auto ir_for = pir_builder->Create<pir::For>(pir_scalar_index, pir_first_value,
                                                        pir_last_value, ir_loop_block);
            if (!ir_grid->pir_value) {
                ir_grid->pir_value = ir_for;
            }
            pir_builder->PushBlock(ir_loop_block);

            pir_builder->Create<pir::WriteVariableLiked>(
                pir_scalar_index, pir_builder->Create<pir::IndexArray>(
                                      ir_grid->index->pir_value, pir_builder->GetInt64Constant(i)));
        }

        for (auto ir_tensor : ir_grid->values) {
            this->EmitTensor(ir_tensor);
        }

        for (int64_t i = 0; i < ir_grid->shape.size(); ++i) {
            pir_builder->PopBlock();
        }
    }

    void EmitOperatorFunction(std::shared_ptr<ir::Operator> ir_operator) {
        this->operator_stack.push(ir_operator);
        auto gurad = ScopeGuard::Create([=]() { this->operator_stack.pop(); });
        std::list<std::shared_ptr<pir::Type>> pir_parameter_types;
        for (auto ir_input_type : ir_operator->GetOperatorType()->input_types) {
            pir_parameter_types.push_back(pir::PointerType::Create(this->EmitType(ir_input_type)));
        }

        std::list<std::shared_ptr<pir::Type>> pir_output_types;
        auto pir_out_type = this->EmitType(ir_operator->GetOperatorType()->output_type);
        if (prajna::Is<pir::VoidType>(pir_out_type)) {
            pir_output_types.push_back(pir_out_type);
        } else {
            pir_output_types.push_back(pir::PointerType::Create(pir_out_type));
        }

        std::shared_ptr<pir::FunctionType> pir_function_type;
        if (pir_output_types.size() == 1) {
            pir_function_type =
                pir::FunctionType::Create(pir_parameter_types, pir_output_types.front());
        } else {
            pir_function_type =
                pir::FunctionType::Create(pir_parameter_types, pir::VoidType::Create());
        }

        ir_operator->pir_function =
            pir_builder->CreateFunction(ir_operator->name, pir_function_type);
        ir_operator->pir_value = ir_operator->pir_function;

        pir_builder->CreateTopBlockForFunction(ir_operator->pir_function);
        auto prajna_guard = ScopeGuard::Create([=]() {
            pir_builder->PopBlock();
            pir_builder->function_stack.pop();
        });

        auto pir_function_parameters_iter = ir_operator->pir_function->parameters.begin();
        for (int64_t i = 0; i < ir_operator->inputs.size(); ++i) {
            ir_operator->inputs[i]->pir_value =
                pir_builder->Create<pir::DeferencePointer>(*pir_function_parameters_iter);
            (*pir_function_parameters_iter)->no_alias = true;
            (*pir_function_parameters_iter)->no_capture = true;
            (*pir_function_parameters_iter)->no_undef = true;
            // (*pir_function_parameters_iter)->readonly = true;
            ++pir_function_parameters_iter;
        }

        // for (int64_t i = 0; i < ir_operator->output_types.size();
        //      ++i, ++pir_function_parameters_iter) {
        //     ir_operator->outputs[i]->pir_value =
        //         pir_builder->Create<pir::DeferencePointer>(*pir_function_parameters_iter);
        //     (*pir_function_parameters_iter)->no_alias = true;
        //     (*pir_function_parameters_iter)->no_capture = true;
        //     (*pir_function_parameters_iter)->no_undef = true;
        // }

        for (auto ir_tensor : ir_operator->values) {
            this->EmitTensor(ir_tensor);
        }

        pir_builder->ReturnVoid();
    }

    void EmitWrite(std::shared_ptr<ir::Write> ir_write_accessor) {
        this->EmitTensor(ir_write_accessor->Variable());
        this->EmitTensor(ir_write_accessor->Tensor());
        pir_builder->Create<pir::WriteVariableLiked>(
            ir_write_accessor->Tensor()->pir_value,
            prajna::Cast<pir::VariableLiked>(ir_write_accessor->Variable()->pir_value));
    }

    void EmitArithmeticInstruction(
        std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) {
        this->EmitTensor(ir_arithmetic_instruction->GetOperand(0));
        this->EmitTensor(ir_arithmetic_instruction->GetOperand(1));

        auto ir_value_type = ir_arithmetic_instruction->type->DataType();
        GALOIS_ASSERT(Is<ir::RealNumberType>(ir_value_type));

        auto pir_binary_operation = pir::BinaryOperator::Operation::None;
        if (auto ir_add = Cast<ir::Add>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::FAdd;
            }
            if (Is<ir::IntType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::Add;
            }
        }
        if (auto ir_sub = Cast<ir::Sub>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::FSub;
            }
            if (Is<ir::IntType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::Sub;
            }
        }
        if (auto ir_mul = Cast<ir::Mul>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::FMul;
            }
            if (Is<ir::IntType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::Mul;
            }
        }
        if (auto ir_div = Cast<ir::Div>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                pir_binary_operation = pir::BinaryOperator::Operation::FDiv;
            }
            if (auto ir_int_type = Cast<ir::IntType>(ir_value_type)) {
                if (ir_int_type->is_signed) {
                    pir_binary_operation = pir::BinaryOperator::Operation::SDiv;
                } else {
                    pir_binary_operation = pir::BinaryOperator::Operation::UDiv;
                }
            }
        }
        GALOIS_ASSERT(pir_binary_operation != pir::BinaryOperator::Operation::None);

        ir_arithmetic_instruction->pir_value = pir_builder->Create<pir::BinaryOperator>(
            pir_binary_operation, ir_arithmetic_instruction->GetOperand(0)->pir_value,
            ir_arithmetic_instruction->GetOperand(1)->pir_value);
    }

    void EmitTensor(std::shared_ptr<ir::Tensor> ir_tensor) {
        if (!ir_tensor) {  // indices may nullptr
            return;
        }

        if (ir_tensor->pir_value) {
            return;
        }

        // if (auto ir_affine_index = Cast<ir::AffineIndex>(ir_tensor)) {
        //     this->EmitAffineIndex(ir_affine_index);
        //     return;
        // }

        if (auto ir_pthread_block = Cast<ir::PthreadBlock>(ir_tensor)) {
            this->EmitPthreadBlock(ir_pthread_block);
        }

        if (auto ir_slice = Cast<ir::SliceView>(ir_tensor)) {
            this->EmitSliceView(ir_slice);
            return;
        }

        if (auto ir_instruction = Cast<ir::Instruction>(ir_tensor)) {
            this->EmitInstruction(ir_instruction);
            return;
        }

        if (auto ir_operator = Cast<ir::Operator>(ir_tensor)) {
            this->EmitOperatorFunction(ir_operator);
            return;
        }

        if (auto ir_constant = Cast<ir::Constant>(ir_tensor)) {
            this->EmitConstant(ir_constant);
            return;
        }

        if (auto ir_grid = Cast<ir::Grid>(ir_tensor)) {
            this->EmitGrid(ir_grid);
            return;
        }

        if (Is<ir::GridIndex>(ir_tensor)) {
            // 在EmitGrid中处理
            return;
        }

        GALOIS_UNREACHABLE;
    }

    void EmitInstruction(std::shared_ptr<ir::Instruction> ir_instruction) {
        for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
            this->EmitTensor(ir_instruction->GetOperand(i));
        }

        if (auto ir_arithmetic_instruction = Cast<ir::ArithmeticInstruction>(ir_instruction)) {
            this->EmitArithmeticInstruction(ir_arithmetic_instruction);
            return;
        }

        if (auto ir_accessor = Cast<ir::Accessor>(ir_instruction)) {
            this->EmitAccessor(ir_accessor);
            return;
        }

        if (auto ir_write_accessor = Cast<ir::Write>(ir_instruction)) {
            this->EmitWrite(ir_write_accessor);
            return;
        }

        if (auto ir_prefetch = Cast<ir::Prefetch>(ir_instruction)) {
            this->EmitPrefetch(ir_prefetch);
            return;
        }

        if (auto ir_broadcast = Cast<ir::Broadcast>(ir_instruction)) {
            this->EmitBroadcast(ir_broadcast);
            return;
        }

        if (auto ir_vector_broadcast = Cast<ir::VectorBroadcast>(ir_instruction)) {
            this->EmitVectorBroadcast(ir_vector_broadcast);
            return;
        }

        if (auto ir_bit_cast = Cast<ir::BitCast>(ir_instruction)) {
            this->EmitBitCast(ir_bit_cast);
            return;
        }

        if (auto ir_call = Cast<ir::Call>(ir_instruction)) {
            this->EmitCall(ir_call);
            return;
        }

        if (auto ir_free = Cast<ir::Free>(ir_instruction)) {
            this->EmitFree(ir_free);
            return;
        }

        if (auto ir_alloca = Cast<ir::Alloca>(ir_instruction)) {
            this->EmitAlloca(ir_alloca);
            return;
        }

        if (auto ir_return = Cast<ir::Return>(ir_instruction)) {
            this->EmitReturn(ir_return);
            return;
        }

        if (auto ir_unary_intrinsic = Cast<ir::UnaryIntrinsic>(ir_instruction)) {
            this->EmitUnaryIntrinsic(ir_unary_intrinsic);
            return;
        }

        GALOIS_UNREACHABLE;
    }

    void EmitConstant(std::shared_ptr<ir::Constant> ir_constant) {
        if (auto ir_constant_float = Cast<ir::ConstantFloat>(ir_constant)) {
            ir_constant->pir_value = pir_builder->Create<pir::ConstantFloat>(
                ir_constant->type->pir_type, ir_constant_float->value);
            return;
        }

        if (auto ir_constant_int = Cast<ir::ConstantInt>(ir_constant)) {
            ir_constant->pir_value = pir_builder->Create<pir::ConstantInt>(
                ir_constant->type->pir_type, ir_constant_int->value);
            return;
        }

        GALOIS_UNREACHABLE;
    }

    std::shared_ptr<pir::Value> GetPrajnaPointerFromTensor(std::shared_ptr<ir::Tensor> ir_tensor) {
        if (auto pir_deference = prajna::Cast<pir::DeferencePointer>(ir_tensor->pir_value)) {
            return pir_deference->Pointer();
        } else {
            return ir_tensor->pir_value;
        }
    }

    void EmitAccessor(std::shared_ptr<ir::Accessor> ir_accessor) {
        this->EmitTensor(ir_accessor->Tensor());

        auto pir_linear_index =
            pir_builder->Create<pir::LocalVariable>(pir_builder->GetInt64Type());
        pir_builder->Create<pir::WriteVariableLiked>(pir_builder->GetInt64Constant(0),
                                                     pir_linear_index);

        // Add shift vector offset
        auto s_product_b = ir_accessor->Tensor()->type->stride * ir_accessor->shift_vector;
        pir_builder->Create<pir::WriteVariableLiked>(
            pir_builder->Create<pir::BinaryOperator>(pir::BinaryOperator::Operation::Add,
                                                     pir_linear_index,
                                                     pir_builder->GetInt64Constant(s_product_b)),
            pir_linear_index);

        // transform is valid
        if (ir_accessor->transform_matrix.size()) {
            auto s_product_a = ir_accessor->Tensor()->type->stride * ir_accessor->transform_matrix;
            RowVectorXprajna pir_s_product_a(s_product_a.size());
            std::transform(RANGE(s_product_a), pir_s_product_a.begin(), [=](int64_t value) {
                return pir_builder->Create<pir::ConstantInt>(pir_builder->GetInt64Type(), value);
            });

            auto pir_index = grid_stack.top()->index->pir_value;
            for (int64_t i = 0; i < pir_s_product_a.size(); ++i) {
                if (s_product_a[i] == 0) {
                    continue;
                }
                auto pir_scalar_index = pir_builder->Create<pir::IndexArray>(
                    pir_index, pir_builder->GetInt64Constant(i));
                auto pir_mul_tmp = pir_builder->Create<pir::BinaryOperator>(
                    pir::BinaryOperator::Operation::Mul, pir_s_product_a[i], pir_scalar_index);
                auto pir_sum_tmp = pir_builder->Create<pir::BinaryOperator>(
                    pir::BinaryOperator::Operation::Add, pir_mul_tmp, pir_linear_index);
                pir_builder->Create<pir::WriteVariableLiked>(pir_sum_tmp, pir_linear_index);
            }
        }

        auto pir_tensor_value_type = this->EmitType(ir_accessor->Tensor()->type->value_type);

        auto pir_tensor_pointer = pir_builder->Create<pir::BitCast>(
            this->GetPrajnaPointerFromTensor(ir_accessor->Tensor()),
            pir::PointerType::Create(pir_tensor_value_type));
        ;
        GALOIS_ASSERT(ir_accessor->Tensor()->type->value_type == ir_accessor->type);

        auto pir_tensor_pointer_var =
            pir_builder->Create<pir::LocalVariable>(pir_tensor_pointer->type);
        pir_builder->Create<pir::WriteVariableLiked>(pir_tensor_pointer, pir_tensor_pointer_var);

        auto pir_value_poitner = pir_builder->Create<pir::GetPointerElementPointer>(
            pir_builder->Create<pir::GetAddressOfVariableLiked>(pir_tensor_pointer_var),
            pir_linear_index);

        ir_accessor->pir_value = pir_builder->Create<pir::DeferencePointer>(pir_value_poitner);
    }

    void EmitPrefetch(std::shared_ptr<ir::Prefetch> ir_prefetch) {
        this->EmitAccessor(ir_prefetch->Address());
        auto pir_address = pir_builder->GetAddressOf(ir_prefetch->Address()->pir_value);

        static std::shared_ptr<pir::Function> pir_prefetch_function = nullptr;
        auto pir_i32_type = pir_builder->GetInt32Type();
        auto pir_i32_pointer_type = pir::PointerType::Create(pir_i32_type);
        if (!pir_prefetch_function) {
            auto pir_llvm_prefetch_function_type = pir::FunctionType::Create(
                {pir_i32_pointer_type, pir_i32_type, pir_i32_type, pir_i32_type},
                pir::VoidType::Create());
            pir_prefetch_function = pir::Function::Create(pir_llvm_prefetch_function_type);
            pir_prefetch_function->fullname = "llvm.prefetch";
            pir_prefetch_function->parent_module = pir_builder->module;
            pir_builder->module->functions.push_back(pir_prefetch_function);
        }

        std::list<std::shared_ptr<pir::Value>> pir_arguments;
        pir_arguments.push_back(
            pir_builder->Create<pir::BitCast>(pir_address, pir_i32_pointer_type));
        /*``address`` is the address to be prefetched, ``rw`` is the specifier
        determining if the fetch should be for a read (0) or write (1), and
        ``locality`` is a temporal locality specifier ranging from (0) - no
        locality, to (3) - extremely local keep in cache. The ``cache type``
        specifies whether the prefetch is performed on the data (1) or
        instruction (0) cache. The ``rw``, ``locality`` and ``cache type``
        arguments must be constant integers.*/
        pir_arguments.push_back(pir_builder->GetInt32Constant(0));
        pir_arguments.push_back(pir_builder->GetInt32Constant(0));
        pir_arguments.push_back(pir_builder->GetInt32Constant(1));
        pir_builder->Create<pir::Call>(pir_prefetch_function, pir_arguments);
    }

    void EmitBroadcast(std::shared_ptr<ir::Broadcast> ir_broadcast) {
        this->EmitTensor(ir_broadcast->Tensor());
        this->EmitType(ir_broadcast->type);

        std::list<std::shared_ptr<pir::Constant>> prajna_constant_zero_list;
        for (int64_t i = 0; i < ir_broadcast->type->Size(); ++i) {
            prajna_constant_zero_list.push_back(pir_builder->GetInt32Constant(0));
        }
        auto pir_constant_vector_zero_mask = pir_builder->Create<pir::ConstantVector>(
            prajna::Cast<pir::VectorType>(ir_broadcast->type->pir_type), prajna_constant_zero_list);

        auto pir_vector_tmp = pir_builder->Create<pir::LocalVariable>(ir_broadcast->type->pir_type);
        pir_builder->Create<pir::WriteVariableLiked>(
            ir_broadcast->Tensor()->pir_value,
            pir_builder->Create<pir::IndexArray>(pir_vector_tmp, pir_builder->GetInt64Constant(0)));

        ir_broadcast->pir_value =
            pir_builder->Create<pir::ShuffleVector>(pir_vector_tmp, pir_constant_vector_zero_mask);
    }

    void EmitVectorBroadcast(std::shared_ptr<ir::VectorBroadcast> ir_vector_broadcast) {
        this->EmitTensor(ir_vector_broadcast->Vector());
        this->EmitType(ir_vector_broadcast->type);

        std::list<std::shared_ptr<pir::Constant>> prajna_constant_lane_id_list;
        GALOIS_ASSERT(ir_vector_broadcast->type->shape.size() == 1);
        for (int64_t i = 0; i < ir_vector_broadcast->type->shape[0]; ++i) {
            prajna_constant_lane_id_list.push_back(
                pir_builder->GetInt32Constant(ir_vector_broadcast->lane_id));
        }
        auto pir_constant_vector_lane_id_mask = pir_builder->Create<pir::ConstantVector>(
            prajna::Cast<pir::VectorType>(ir_vector_broadcast->type->pir_type),
            prajna_constant_lane_id_list);

        ir_vector_broadcast->pir_value = pir_builder->Create<pir::ShuffleVector>(
            ir_vector_broadcast->Vector()->pir_value, pir_constant_vector_lane_id_mask);
    }

    void EmitBitCast(std::shared_ptr<ir::BitCast> ir_bit_cast) {
        this->EmitTensor(ir_bit_cast->Tensor());
        this->EmitType(ir_bit_cast->type);
        ir_bit_cast->pir_value =
            pir_builder->Create<pir::DeferencePointer>(pir_builder->Create<pir::BitCast>(
                pir_builder->Create<pir::GetAddressOfVariableLiked>(
                    prajna::Cast<pir::DeferencePointer>(ir_bit_cast->Tensor()->pir_value)),
                pir::PointerType::Create(ir_bit_cast->type->pir_type)));
    }

    void EmitAlloca(std::shared_ptr<ir::Alloca> ir_alloca) {
        auto ir_tensor_type = ir_alloca->type;

        std::list<std::shared_ptr<pir::Value>> pir_arguments = {
            pir_builder->GetInt64Constant(ir_tensor_type->bytes)};
        auto pir_function_type =
            pir::FunctionType::Create({pir::IntType::Create(64, true)},
                                      pir::PointerType::Create(pir::IntType::Create(8, false)));
        auto pir_aligned_alloc = pir_builder->GetIntrinsic("auto_aligned_alloc", pir_function_type);
        auto pir_tensor_pointer = pir_builder->Create<pir::BitCast>(
            pir_builder->Create<pir::Call>(pir_aligned_alloc, pir_arguments),
            pir::PointerType::Create(this->EmitType(ir_tensor_type)));
        ir_alloca->pir_value = pir_builder->Create<pir::DeferencePointer>(pir_tensor_pointer);
        return;
    }

    std::shared_ptr<pir::Value> GetPirValueOfTensor(std::shared_ptr<ir::Tensor> ir_tensor) {
        if (auto pir_deference_pointer =
                prajna::Cast<pir::DeferencePointer>(ir_tensor->pir_value)) {
            return pir_deference_pointer->Pointer();
        } else {
            return ir_tensor->pir_value;
        }
    }

    void EmitReturn(std::shared_ptr<ir::Return> ir_return) {
        auto pir_pointer_value = GetPirValueOfTensor(ir_return->Tensor());
        pir_builder->Create<pir::Return>(pir_pointer_value);
    }

    void EmitFree(std::shared_ptr<ir::Free> ir_free) {
        this->EmitTensor(ir_free->Tensor());
        ir_free->pir_value = pir_builder->Create<pir::Call>(
            pir_builder->GetIntrinsic(
                "free", pir::FunctionType::Create(
                            {pir::PointerType::Create(pir::IntType::Create(8, false))},
                            pir::VoidType::Create())),
            pir_builder->Create<pir::BitCast>(
                prajna::Cast<pir::DeferencePointer>(ir_free->Tensor()->pir_value)->Pointer(),
                pir::PointerType::Create(pir::IntType::Create(8, false))));
    }

    void EmitSliceView(std::shared_ptr<ir::SliceView> ir_slice) {
        this->EmitAccessor(ir_slice->Origin());

        Eigen::RowVectorXi64 stride(ir_slice->type->shape.size());
        // stride使用被slice的tensor
        ir_slice->type->stride.resize(ir_slice->type->shape.size());
        int64_t i = stride.size() - 1;
        int64_t j = ir_slice->Origin()->Tensor()->type->stride.size() - 1;
        GALOIS_ASSERT(i <= j);  // 如果slice的同时降维, 需要将stride也处理下
        for (; i >= 0; --j, --i) {
            stride[i] = ir_slice->Origin()->Tensor()->type->stride[j];
        }
        //
        auto ir_slice_type =
            ir::TensorType::Create(ir_slice->type->value_type, ir_slice->shape, stride);
        ir_slice->type = ir_slice_type;
        // 偏移地址
        ir_slice->pir_value = prajna::Cast<pir::VariableLiked>(ir_slice->Origin()->pir_value);
    }

    void EmitCall(std::shared_ptr<ir::Call> ir_call) {
        if (!ir_call->annotation_dict.count("enable_multi_thread")) {
            std::list<std::shared_ptr<pir::Value>> pir_arguments;
            for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
                pir_arguments.push_back(pir_builder->Create<pir::GetAddressOfVariableLiked>(
                    // 我们通过指针来传递参数， 所以需要用变量包装一下
                    pir_builder->VariableLikedNormalize(ir_call->Input(i)->pir_value)));
            }

            ir_call->pir_value =
                pir_builder->Create<pir::Call>(ir_call->Operator()->pir_value, pir_arguments);

            // TODO: warlaround
            auto ir_operator_type = ir_call->Operator()->GetOperatorType();
            if (!Is<ir::VoidType>(ir_operator_type->output_type)) {
                ir_call->pir_value = pir_builder->Create<pir::DeferencePointer>(ir_call->pir_value);
            }
        } else {  // async invoke
            std::list<std::shared_ptr<pir::Value>> pir_arguments;
            int64_t grid_argument_index;
            for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
                pir_arguments.push_back(pir_builder->Create<pir::GetAddressOfVariableLiked>(
                    prajna::Cast<pir::VariableLiked>(ir_call->Input(i)->pir_value)));
                if (ir_call->Input(i) == this->grid_stack.top()->index) {
                    grid_argument_index = i;
                }
            }

            std::list<std::shared_ptr<pir::Field>> pir_fields;
            int64_t num = 0;
            std::transform(RANGE(pir_arguments), std::back_inserter(pir_fields),
                           [&num](std::shared_ptr<pir::Value> pir_value) {
                               return pir::Field::Create("__field" + std::to_string(num),
                                                         pir_value->type);
                           });
            auto pir_async_args_struct_type = prajna ::ir::StructType::Create(pir_fields);
            auto pir_async_args_type = pir::PointerType::Create(pir_async_args_struct_type);
            auto pir_async_function_type =
                pir::FunctionType::Create({pir_async_args_type}, pir::VoidType::Create());
            auto pir_async_function = pir_builder->CreateFunction(
                prajna::ast::Identifier("__todo_async_call"), pir_async_function_type);
            // begin prajna_async_function
            {
                pir_builder->CreateTopBlockForFunction(pir_async_function);
                std::list<std::shared_ptr<pir::Value>> pir_inner_arguments;
                auto pir_async_parameter_deference = pir_builder->Create<pir::DeferencePointer>(
                    pir_async_function->parameters.front());
                std::transform(RANGE(pir_fields), std::back_inserter(pir_inner_arguments),
                               [=](std::shared_ptr<pir::Field> pir_field) {
                                   return pir_builder->Create<pir::AccessField>(
                                       pir_async_parameter_deference, pir_field);
                               });
                pir_builder->Create<pir::Call>(ir_call->Operator()->pir_value, pir_inner_arguments);
                this->PirFree(pir_builder->Create<pir::AccessField>(
                    pir_async_parameter_deference,
                    *std::next(pir_fields.begin(), grid_argument_index)));
                this->PirFree(pir_async_function->parameters.front());
                pir_builder->ReturnVoid();
                pir_builder->PopBlock();
                pir_builder->function_stack.pop();
            }
            // end prajna_async_function
            auto pir_async_args_struct =
                pir_builder->Create<pir::LocalVariable>(pir_async_args_type);
            pir_builder->Create<pir::WriteVariableLiked>(
                this->PirNew(pir_async_args_type->value_type), pir_async_args_struct);

            int64_t i = 0;
            for (auto [pir_field, pir_argument] : boost::combine(pir_fields, pir_arguments)) {
                if (i != grid_argument_index) {
                    pir_builder->Create<pir::WriteVariableLiked>(
                        pir_argument,
                        pir_builder->Create<pir::AccessField>(
                            pir_builder->Create<pir::DeferencePointer>(pir_async_args_struct),
                            pir_field));
                } else {
                    auto ir_new_indices = this->PirNew(ir_call->Input(i)->pir_value->type);
                    pir_builder->Create<pir::WriteVariableLiked>(
                        ir_call->Input(i)->pir_value,
                        pir_builder->Create<pir::DeferencePointer>(ir_new_indices));
                    pir_builder->Create<pir::WriteVariableLiked>(
                        ir_new_indices,
                        pir_builder->Create<pir::AccessField>(
                            pir_builder->Create<pir::DeferencePointer>(pir_async_args_struct),
                            pir_field));
                }
                ++i;
            }
            auto pir_i64_type = pir::IntType::Create(64, true);
            auto pir_function_type = pir::FunctionType::Create(
                {pir_i64_type, pir_i64_type, pir_i64_type}, pir::VoidType::Create());
            pir_builder->Create<pir::Call>(
                pir_builder->GetIntrinsic("thpool_add_work", pir_function_type),
                std::list<std::shared_ptr<pir::Value>>{
                    this->pir_thpool,
                    pir_builder->Create<pir::CastInstruction>(
                        pir::CastInstruction::Operation::PtrToInt, pir_async_function,
                        pir_i64_type),
                    pir_builder->Create<pir::CastInstruction>(
                        pir::CastInstruction::Operation::PtrToInt, pir_async_args_struct,
                        pir_i64_type)});
        }
    }

    void EmitUnaryIntrinsic(std::shared_ptr<ir::UnaryIntrinsic> ir_intrinsic) {
        auto ir_operand = ir_intrinsic->GetOperand(0);
        auto pir_operand_type = ir_operand->type->pir_type;
        auto pir_intrinsic_type = pir::FunctionType::Create({pir_operand_type}, pir_operand_type);
        auto llvm_intrinsic_name =
            "llvm." + ir_intrinsic->intrinsic_name + "." + pir_operand_type->name;
        auto pir_intrinsic = pir_builder->GetIntrinsic(llvm_intrinsic_name, pir_intrinsic_type);
        ir_intrinsic->pir_value =
            pir_builder->Create<pir::Call>(pir_intrinsic, ir_operand->pir_value);
    }

    void EmitPthreadBlock(std::shared_ptr<ir::PthreadBlock> ir_pthread_block) {
        auto ir_captured_tensors = transform::CaptureExternalTensors(ir_pthread_block);
    }

    std::shared_ptr<pir::Value> PirNew(std::shared_ptr<pir::Type> pir_type) {
        auto pir_function_type =
            pir::FunctionType::Create({pir::IntType::Create(64, true)},
                                      pir::PointerType::Create(pir::IntType::Create(8, false)));
        auto pir_malloc = pir_builder->GetIntrinsic("malloc", pir_function_type);
        return pir_builder->Create<pir::BitCast>(
            pir_builder->Create<pir::Call>(pir_malloc,
                                           pir_builder->GetInt64Constant(pir_type->bytes)),
            pir::PointerType::Create(pir_type));
    }

    void PirFree(std::shared_ptr<pir::Value> pir_value) {
        auto pir_function_type = pir::FunctionType::Create(
            {pir::PointerType::Create(pir::IntType::Create(8, false))}, pir::VoidType::Create());
        pir_builder->Create<pir::Call>(
            pir_builder->GetIntrinsic("free", pir_function_type),
            pir_builder->Create<pir::BitCast>(
                pir_value, pir::PointerType::Create(pir::IntType::Create(8, false))));
    }

   public:
    std::stack<std::shared_ptr<ir::Grid>> grid_stack;
    std::stack<std::shared_ptr<ir::Operator>> operator_stack;

    std::shared_ptr<pir::Value> pir_thpool = nullptr;
    std::shared_ptr<prajna::lowering::IrBuilder> pir_builder = nullptr;
};

}  // namespace galois::codegen::cpu
