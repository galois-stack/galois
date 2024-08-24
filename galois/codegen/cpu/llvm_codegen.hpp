#pragma once

#include <cmath>

#include "galois/helper.hpp"
#include "galois/ir/ir.hpp"
#include "prajna/ir/ir.hpp"
#include "prajna/lowering/ir_builder.hpp"

namespace galois::codegen::cpu {

using namespace ir;

class LlvmCodegen {
   public:
    LlvmCodegen(std::shared_ptr<prajna::lowering::SymbolTable> prajna_symbol_table) {
        // auto ir_symbol_table = prajna::lowering::SymbolTable::Create(nullptr);
        auto prajna_ir_module = prajna::ir::Module::Create();
        auto prajna_ir_logger = prajna::Logger::Create("");
        prajna_ir_module->symbol_table = prajna_symbol_table;
        this->prajna_ir_builder = prajna::lowering::IrBuilder::Create(
            prajna_symbol_table, prajna_ir_module, prajna_ir_logger);

        this->BindIntrinsics();
    }

    void DeclareIntrinsic() {
        auto prajna_ir_i32_type = this->prajna_ir_builder->GetInt32Type();
        auto prajna_ir_i32_pointer_type = prajna::ir::PointerType::Create(prajna_ir_i32_type);
        auto prajna_ir_llvm_prefetch_function_type = prajna::ir::FunctionType::Create(
            {prajna_ir_i32_pointer_type, prajna_ir_i32_type, prajna_ir_i32_type},
            prajna::ir::VoidType::Create());
        this->prajna_ir_builder->CreateFunction(prajna::ast::Identifier("llvm.prefetch"),
                                                prajna_ir_llvm_prefetch_function_type);
    }

    std::shared_ptr<prajna::ir::Type> EmitType(std::shared_ptr<ir::TensorType> ir_type) {
        if (ir_type->prajna_ir_type) {
            return ir_type->prajna_ir_type;
        }

        if (ir_type->IsScalar()) {
            if (auto ir_float_type = Cast<ir::FloatType>(ir_type->data_type)) {
                ir_type->prajna_ir_type = prajna::ir::FloatType::Create(ir_float_type->bits);
                return ir_type->prajna_ir_type;
            }
        }

        if (ir_type->shape.size() == 1 && ir_type->value_type->IsScalar() &&
            IsPowerOfTwo(ir_type->shape[0])) {
            ir_type->prajna_ir_type = prajna::ir::VectorType::Create(
                this->EmitType(ir_type->value_type), ir_type->Size());
            return ir_type->prajna_ir_type;
        } else {
            ir_type->prajna_ir_type =
                prajna::ir::ArrayType::Create(this->EmitType(ir_type->value_type), ir_type->Size());
            return ir_type->prajna_ir_type;
        }

        GALOIS_TODO;
        return nullptr;
    }

    void EmitGrid(std::shared_ptr<ir::Grid> ir_grid) {
        GALOIS_ASSERT(!ir_grid->prajna_ir_value);
        if (this->parallel_stack.size()) {
            ir_grid->parent_parallel = this->parallel_stack.top();
        }
        this->parallel_stack.push(ir_grid);
        auto guard = ScopeGuard::Create([=]() { this->parallel_stack.pop(); });

        int64_t parent_index_size = 0;
        if (!ir_grid->is_local) {
            parent_index_size = ir_grid->parent_parallel->prajna_ir_index_vector.size();
            ir_grid->prajna_ir_index_vector =
                VectorXprajna::Zero(parent_index_size + ir_grid->shape.size());
            ir_grid->prajna_ir_index_vector.topRows(parent_index_size) =
                ir_grid->parent_parallel->prajna_ir_index_vector;

        } else {
            ir_grid->prajna_ir_index_vector = VectorXprajna::Zero(ir_grid->shape.size());
        }
        for (int64_t i = 0; i < ir_grid->shape.size(); ++i) {
            auto prajna_ir_first_value = this->prajna_ir_builder->GetInt64Constant(0);
            auto prajna_ir_last_value =
                this->prajna_ir_builder->GetInt64Constant(ir_grid->shape[i]);
            auto ir_loop_block = prajna::ir::Block::Create();
            auto prajna_ir_scalar_index =
                this->prajna_ir_builder->Create<prajna::ir::LocalVariable>(
                    this->prajna_ir_builder->GetInt64Type());
            prajna_ir_scalar_index->fullname = "idx";
            auto ir_for = this->prajna_ir_builder->Create<prajna::ir::For>(
                prajna_ir_scalar_index, prajna_ir_first_value, prajna_ir_last_value, ir_loop_block);
            if (!ir_grid->prajna_ir_value) {
                ir_grid->prajna_ir_value = ir_for;
            }
            this->prajna_ir_builder->PushBlock(ir_loop_block);
            ir_grid->prajna_ir_index_vector[i + parent_index_size] = prajna_ir_scalar_index;
        }

        for (auto ir_tensor : ir_grid->values) {
            this->EmitTensor(ir_tensor);
        }

        for (int64_t i = 0; i < ir_grid->shape.size(); ++i) {
            this->prajna_ir_builder->PopBlock();
        }
    }

    void EmitOperatorInstance(std::shared_ptr<ir::OperatorInstance> ir_operator) {
        if (ir_operator->prajna_ir_value) {
            return;
        }

        this->operator_stack.push(ir_operator);
        auto gurad = ScopeGuard::Create([=]() { this->operator_stack.pop(); });
        std::list<std::shared_ptr<prajna::ir::Type>> prajna_ir_parameter_types;
        for (auto ir_input_type : ir_operator->input_types) {
            prajna_ir_parameter_types.push_back(
                prajna::ir::PointerType::Create(this->EmitType(ir_input_type)));
        }

        for (auto ir_output_type : ir_operator->output_types) {
            prajna_ir_parameter_types.push_back(
                prajna::ir::PointerType::Create(this->EmitType(ir_output_type)));
        }

        auto prajna_ir_function_type = prajna::ir::FunctionType::Create(
            prajna_ir_parameter_types, prajna::ir::VoidType::Create());

        ir_operator->prajna_ir_function =
            this->prajna_ir_builder->CreateFunction(ir_operator->name, prajna_ir_function_type);
        ir_operator->prajna_ir_value = ir_operator->prajna_ir_function;

        this->prajna_ir_builder->CreateTopBlockForFunction(ir_operator->prajna_ir_function);
        auto prajna_guard = ScopeGuard::Create([=]() {
            this->prajna_ir_builder->PopBlock();
            this->prajna_ir_builder->function_stack.pop();
        });

        auto prajna_ir_function_parameters_iter =
            ir_operator->prajna_ir_function->parameters.begin();
        for (int64_t i = 0; i < ir_operator->input_types.size(); ++i) {
            ir_operator->inputs[i]->prajna_ir_value =
                this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(
                    *prajna_ir_function_parameters_iter);
            (*prajna_ir_function_parameters_iter)->no_alias = true;
            (*prajna_ir_function_parameters_iter)->no_capture = true;
            (*prajna_ir_function_parameters_iter)->no_undef = true;
            (*prajna_ir_function_parameters_iter)->readonly = true;
            ++prajna_ir_function_parameters_iter;
        }

        for (int64_t i = 0; i < ir_operator->output_types.size();
             ++i, ++prajna_ir_function_parameters_iter) {
            ir_operator->outputs[i]->prajna_ir_value =
                this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(
                    *prajna_ir_function_parameters_iter);
            (*prajna_ir_function_parameters_iter)->no_alias = true;
            (*prajna_ir_function_parameters_iter)->no_capture = true;
            (*prajna_ir_function_parameters_iter)->no_undef = true;
        }

        for (auto ir_tensor : ir_operator->values) {
            this->EmitTensor(ir_tensor);
        }

        this->prajna_ir_builder->Create<prajna::ir::Return>(
            this->prajna_ir_builder->Create<prajna::ir::VoidValue>());
    }

    void EmitWrite(std::shared_ptr<ir::Write> ir_write_accessor) {
        this->EmitTensor(ir_write_accessor->Variable());
        this->EmitTensor(ir_write_accessor->Tensor());
        this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            ir_write_accessor->Tensor()->prajna_ir_value,
            prajna::Cast<prajna::ir::VariableLiked>(
                ir_write_accessor->Variable()->prajna_ir_value));
    }

    void EmitArithmeticInstruction(
        std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) {
        this->EmitTensor(ir_arithmetic_instruction->GetOperand(0));
        this->EmitTensor(ir_arithmetic_instruction->GetOperand(1));

        auto ir_value_type = ir_arithmetic_instruction->type->data_type;
        GALOIS_ASSERT(Is<RealNumberType>(ir_value_type));

        auto prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::None;
        if (auto ir_add = Cast<ir::Add>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::FAdd;
            }
            if (Is<ir::IntType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::Add;
            }
        }
        if (auto ir_sub = Cast<ir::Sub>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::FSub;
            }
            if (Is<ir::IntType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::Sub;
            }
        }
        if (auto ir_mul = Cast<ir::Mul>(ir_arithmetic_instruction)) {
            if (Is<ir::FloatType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::FMul;
            }
            if (Is<ir::IntType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::Mul;
            }
        }
        if (auto ir_div = Cast<ir::Div>(ir_arithmetic_instruction)) {
            if (Is<FloatType>(ir_value_type)) {
                prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::FDiv;
            }
            if (auto ir_int_type = Cast<IntType>(ir_value_type)) {
                if (ir_int_type->is_signed) {
                    prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::SDiv;
                } else {
                    prajna_ir_binary_operation = prajna::ir::BinaryOperator::Operation::UDiv;
                }
            }
        }
        GALOIS_ASSERT(prajna_ir_binary_operation != prajna::ir::BinaryOperator::Operation::None);

        ir_arithmetic_instruction->prajna_ir_value =
            this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
                prajna_ir_binary_operation,
                ir_arithmetic_instruction->GetOperand(0)->prajna_ir_value,
                ir_arithmetic_instruction->GetOperand(1)->prajna_ir_value);
    }

    void EmitTensor(std::shared_ptr<ir::Tensor> ir_tensor) {
        if (ir_tensor->prajna_ir_value) {
            return;
        }

        // if (auto ir_affine_index = Cast<ir::AffineIndex>(ir_tensor)) {
        //     this->EmitAffineIndex(ir_affine_index);
        //     return;
        // }

        if (auto ir_slice = Cast<ir::Slice>(ir_tensor)) {
            this->EmitSlice(ir_slice);
            return;
        }

        if (auto ir_instruction = Cast<ir::Instruction>(ir_tensor)) {
            this->EmitInstruction(ir_instruction);
            return;
        }

        if (auto ir_operator = Cast<ir::OperatorInstance>(ir_tensor)) {
            this->EmitOperatorInstance(ir_operator);
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
    }

    void EmitInstruction(std::shared_ptr<ir::Instruction> ir_instruction) {
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

        GALOIS_UNREACHABLE;
    }

    void EmitConstant(std::shared_ptr<ir::Constant> ir_constant) {
        if (auto ir_constant_float = Cast<ir::ConstantFloat>(ir_constant)) {
            ir_constant->prajna_ir_value =
                this->prajna_ir_builder->Create<prajna::ir::ConstantFloat>(
                    ir_constant->type->prajna_ir_type, ir_constant_float->value);
            return;
        }

        if (auto ir_constant_int = Cast<ir::ConstantInt>(ir_constant)) {
            ir_constant->prajna_ir_value = this->prajna_ir_builder->Create<prajna::ir::ConstantInt>(
                ir_constant->type->prajna_ir_type, ir_constant_int->value);
            return;
        }

        GALOIS_UNREACHABLE;
    }

    void EmitAccessor(std::shared_ptr<ir::Accessor> ir_accessor) {
        this->EmitTensor(ir_accessor->Tensor());

        auto prajna_ir_linear_index = this->prajna_ir_builder->Create<prajna::ir::LocalVariable>(
            this->prajna_ir_builder->GetInt64Type());
        auto s_product_b = ir_accessor->Tensor()->type->stride * ir_accessor->shift_vector;

        auto s_product_a = ir_accessor->Tensor()->type->stride * ir_accessor->transform_matrix;

        RowVectorXprajna prajna_ir_s_product_a(s_product_a.size());
        std::transform(RANGE(s_product_a), prajna_ir_s_product_a.begin(), [=](int64_t value) {
            return this->prajna_ir_builder->Create<prajna::ir::ConstantInt>(
                this->prajna_ir_builder->GetInt64Type(), value);
        });

        this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            this->prajna_ir_builder->GetInt64Constant(0), prajna_ir_linear_index);
        for (int64_t i = 0; i < prajna_ir_s_product_a.size(); ++i) {
            if (s_product_a[i] == 0) {
                continue;
            }
            if (s_product_a[i] == 1) {
                auto prajna_ir_sum_tmp =
                    this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
                        prajna::ir::BinaryOperator::Operation::Add,
                        this->parallel_stack.top()->prajna_ir_index_vector[i],
                        prajna_ir_linear_index);
                this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
                    prajna_ir_sum_tmp, prajna_ir_linear_index);
                continue;
            }
            auto prajna_ir_mul_tmp = this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
                prajna::ir::BinaryOperator::Operation::Mul, prajna_ir_s_product_a[i],
                this->parallel_stack.top()->prajna_ir_index_vector[i]);
            auto prajna_ir_sum_tmp = this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
                prajna::ir::BinaryOperator::Operation::Add, prajna_ir_mul_tmp,
                prajna_ir_linear_index);
            this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(prajna_ir_sum_tmp,
                                                                            prajna_ir_linear_index);
        }

        this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
                prajna::ir::BinaryOperator::Operation::Add, prajna_ir_linear_index,
                this->prajna_ir_builder->GetInt64Constant(s_product_b)),
            prajna_ir_linear_index);

        auto prajna_ir_tensor_value_type = this->EmitType(ir_accessor->Tensor()->type->value_type);
        auto prajna_ir_tensor_pointer = this->prajna_ir_builder->Create<prajna::ir::BitCast>(
            prajna::Cast<prajna::ir::DeferencePointer>(ir_accessor->Tensor()->prajna_ir_value)
                ->Pointer(),
            prajna::ir::PointerType::Create(prajna_ir_tensor_value_type));
        ;
        GALOIS_ASSERT(ir_accessor->Tensor()->type->value_type == ir_accessor->type);

        auto prajna_ir_tensor_pointer_var =
            this->prajna_ir_builder->Create<prajna::ir::LocalVariable>(
                prajna_ir_tensor_pointer->type);
        this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            prajna_ir_tensor_pointer, prajna_ir_tensor_pointer_var);

        auto prajna_ir_value_poitner =
            this->prajna_ir_builder->Create<prajna::ir::GetPointerElementPointer>(
                this->prajna_ir_builder->Create<prajna::ir::GetAddressOfVariableLiked>(
                    prajna_ir_tensor_pointer_var),
                prajna_ir_linear_index);

        // std::shared_ptr<prajna::ir::Value> prajna_ir_i64_value_address =
        //     this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
        //         prajna::ir::BinaryOperator::Operation::Add,
        //         this->prajna_ir_builder->Create<prajna::ir::CastInstruction>(
        //             prajna::ir::CastInstruction::Operation::PtrToInt,
        //             prajna_ir_tensor_pointer, this->prajna_ir_builder->GetInt64Type()),
        //         this->prajna_ir_builder->Create<prajna::ir::BinaryOperator>(
        //             prajna::ir::BinaryOperator::Operation::Mul,
        //             this->prajna_ir_builder->GetInt64Constant(prajna_ir_tensor_value_type->bytes),
        //             prajna_ir_linear_index));

        // std::shared_ptr<prajna::ir::Tensor> prajna_ir_value_poitner = nullptr;
        if (ir_accessor->simd_size == 1) {
            // prajna_ir_value_poitner =
            // this->prajna_ir_builder->Create<prajna::ir::CastInstruction>(
            //     prajna::ir::CastInstruction::Operation::IntToPtr,
            //     prajna_ir_i64_value_address,
            //     prajna::ir::PointerType::Create(prajna_ir_tensor_value_type));
        } else {
            GALOIS_TODO;
            // if (!ir_accessor->simd_shuffle) {
            //     auto prajna_ir_simd_type =
            //         prajna::ir::VectorType::CreateImp(prajna::Cast<prajna::ir::PointerType>(
            //                                            ir_accessor->Tensor()->prajna_ir_value->type)
            //                                            ->value_type,
            //                                        ir_accessor->simd_size);
            //     prajna_ir_data_ptr =
            //     this->prajna_ir_builder->Create<prajna::ir::CastInstruction>(
            //         prajna::ir::CastInstruction::Operation::IntToPtr, prajna_ir_data_address,
            //         prajna::ir::PointerType::Create(prajna_ir_simd_type));
            // } else {
            //     auto prajna_ir_simd_type =
            //         prajna::ir::VectorType::CreateImp(prajna::Cast<prajna::ir::PointerType>(
            //                                            ir_accessor->Tensor()->prajna_ir_value->type)
            //                                            ->value_type,
            //                                        ir_accessor->simd_size);
            //     std::list<std::shared_ptr<prajna::ir::Constant>> prajna_constant_zero_list;
            //     for (int64_t i = 0; i < ir_accessor->simd_size; ++i) {
            //         prajna_constant_zero_list.push_back(
            //             this->prajna_ir_builder->GetInt32Constant(0));
            //     }
            //     auto prajna_ir_constant_vector_zero_mask =
            //         this->prajna_ir_builder->Create<prajna::ir::ConstantVector>(
            //             prajna_ir_simd_type, prajna_constant_zero_list);
            //     prajna_ir_data_ptr =
            //     this->prajna_ir_builder->Create<prajna::ir::CastInstruction>(
            //         prajna::ir::CastInstruction::Operation::IntToPtr, prajna_ir_data_address,
            //         ir_accessor->Tensor()->prajna_ir_value->type);
            //     auto prajna_ir_vector =
            //         this->prajna_ir_builder->Create<prajna::ir::LocalVariable>(prajna_ir_simd_type);
            //     this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            //         this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(
            //             prajna_ir_data_ptr),
            //         this->prajna_ir_builder->Create<prajna::ir::IndexArray>(
            //             prajna_ir_vector, this->prajna_ir_builder->GetInt64Constant(0)));
            //     this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            //         this->prajna_ir_builder->Create<prajna::ir::ShuffleVector>(
            //             prajna_ir_vector, prajna_ir_constant_vector_zero_mask),
            //         prajna_ir_vector);
            //     ir_accessor->prajna_ir_value = prajna_ir_vector;
            //     PRAJNA_ASSERT(!ir_accessor->IsWritten());
            //     return;
            //     GALOIS_TODO;
            // }
        }

        ir_accessor->prajna_ir_value =
            this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(prajna_ir_value_poitner);
    }

    void EmitPrefetch(std::shared_ptr<ir::Prefetch> ir_prefetch) {
        this->EmitAccessor(ir_prefetch->Address());
        auto prajna_ir_address =
            this->prajna_ir_builder->GetAddressOf(ir_prefetch->Address()->prajna_ir_value);

        static std::shared_ptr<prajna::ir::Function> prajna_ir_prefetch_function = nullptr;
        auto prajna_ir_i32_type = this->prajna_ir_builder->GetInt32Type();
        auto prajna_ir_i32_pointer_type = prajna::ir::PointerType::Create(prajna_ir_i32_type);
        if (!prajna_ir_prefetch_function) {
            auto prajna_ir_llvm_prefetch_function_type =
                prajna::ir::FunctionType::Create({prajna_ir_i32_pointer_type, prajna_ir_i32_type,
                                                  prajna_ir_i32_type, prajna_ir_i32_type},
                                                 prajna::ir::VoidType::Create());
            prajna_ir_prefetch_function =
                prajna::ir::Function::Create(prajna_ir_llvm_prefetch_function_type);
            prajna_ir_prefetch_function->fullname = "llvm.prefetch";
            prajna_ir_prefetch_function->parent_module = this->prajna_ir_builder->module;
            this->prajna_ir_builder->module->functions.push_back(prajna_ir_prefetch_function);
        }

        std::list<std::shared_ptr<prajna::ir::Value>> prajna_ir_arguments;
        prajna_ir_arguments.push_back(this->prajna_ir_builder->Create<prajna::ir::BitCast>(
            prajna_ir_address, prajna_ir_i32_pointer_type));
        /*``address`` is the address to be prefetched, ``rw`` is the specifier
        determining if the fetch should be for a read (0) or write (1), and
        ``locality`` is a temporal locality specifier ranging from (0) - no
        locality, to (3) - extremely local keep in cache. The ``cache type``
        specifies whether the prefetch is performed on the data (1) or
        instruction (0) cache. The ``rw``, ``locality`` and ``cache type``
        arguments must be constant integers.*/
        prajna_ir_arguments.push_back(this->prajna_ir_builder->GetInt32Constant(0));
        prajna_ir_arguments.push_back(this->prajna_ir_builder->GetInt32Constant(0));
        prajna_ir_arguments.push_back(this->prajna_ir_builder->GetInt32Constant(1));
        this->prajna_ir_builder->Create<prajna::ir::Call>(prajna_ir_prefetch_function,
                                                          prajna_ir_arguments);
    }

    void EmitBroadcast(std::shared_ptr<ir::Broadcast> ir_broadcast) {
        this->EmitTensor(ir_broadcast->Tensor());
        this->EmitType(ir_broadcast->type);

        std::list<std::shared_ptr<prajna::ir::Constant>> prajna_constant_zero_list;
        for (int64_t i = 0; i < ir_broadcast->type->Size(); ++i) {
            prajna_constant_zero_list.push_back(this->prajna_ir_builder->GetInt32Constant(0));
        }
        auto prajna_ir_constant_vector_zero_mask =
            this->prajna_ir_builder->Create<prajna::ir::ConstantVector>(
                prajna::Cast<prajna::ir::VectorType>(ir_broadcast->type->prajna_ir_type),
                prajna_constant_zero_list);

        auto prajna_ir_vector_tmp = this->prajna_ir_builder->Create<prajna::ir::LocalVariable>(
            ir_broadcast->type->prajna_ir_type);
        this->prajna_ir_builder->Create<prajna::ir::WriteVariableLiked>(
            ir_broadcast->Tensor()->prajna_ir_value,
            this->prajna_ir_builder->Create<prajna::ir::IndexArray>(
                prajna_ir_vector_tmp, this->prajna_ir_builder->GetInt64Constant(0)));

        ir_broadcast->prajna_ir_value = this->prajna_ir_builder->Create<prajna::ir::ShuffleVector>(
            prajna_ir_vector_tmp, prajna_ir_constant_vector_zero_mask);
    }

    void EmitVectorBroadcast(std::shared_ptr<ir::VectorBroadcast> ir_vector_broadcast) {
        this->EmitTensor(ir_vector_broadcast->Vector());
        this->EmitType(ir_vector_broadcast->type);

        std::list<std::shared_ptr<prajna::ir::Constant>> prajna_constant_lane_id_list;
        GALOIS_ASSERT(ir_vector_broadcast->type->shape.size() == 1);
        for (int64_t i = 0; i < ir_vector_broadcast->type->shape[0]; ++i) {
            prajna_constant_lane_id_list.push_back(
                this->prajna_ir_builder->GetInt32Constant(ir_vector_broadcast->lane_id));
        }
        auto prajna_ir_constant_vector_lane_id_mask =
            this->prajna_ir_builder->Create<prajna::ir::ConstantVector>(
                prajna::Cast<prajna::ir::VectorType>(ir_vector_broadcast->type->prajna_ir_type),
                prajna_constant_lane_id_list);

        ir_vector_broadcast->prajna_ir_value =
            this->prajna_ir_builder->Create<prajna::ir::ShuffleVector>(
                ir_vector_broadcast->Vector()->prajna_ir_value,
                prajna_ir_constant_vector_lane_id_mask);
    }

    void EmitBitCast(std::shared_ptr<ir::BitCast> ir_bit_cast) {
        this->EmitTensor(ir_bit_cast->Tensor());
        this->EmitType(ir_bit_cast->type);
        ir_bit_cast->prajna_ir_value =
            this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(
                this->prajna_ir_builder->Create<prajna::ir::BitCast>(
                    this->prajna_ir_builder->Create<prajna::ir::GetAddressOfVariableLiked>(
                        prajna::Cast<prajna::ir::DeferencePointer>(
                            ir_bit_cast->Tensor()->prajna_ir_value)),
                    prajna::ir::PointerType::Create(ir_bit_cast->type->prajna_ir_type)));
    }

    void BindIntrinsics() {
        {
            auto function_type = prajna::ir::FunctionType::Create(
                {prajna::ir::IntType::Create(64, true)},
                prajna::ir::PointerType::Create(prajna::ir::IntType::Create(8, false)));
            this->prajna_ir_malloc_function = this->prajna_ir_builder->CreateFunction(
                prajna::ast::Identifier("malloc"), function_type);
            this->prajna_ir_malloc_function->annotation_dict["intrinsic"].push_back("malloc");
        }
        {
            auto function_type = prajna::ir::FunctionType::Create(
                {prajna::ir::PointerType::Create(prajna::ir::IntType::Create(8, false))},
                prajna::ir::VoidType::Create());
            this->prajna_ir_free_function = this->prajna_ir_builder->CreateFunction(
                prajna::ast::Identifier("free"), function_type);
            this->prajna_ir_free_function->annotation_dict["intrinsic"].push_back("free");
        }
    }

    void EmitAlloca(std::shared_ptr<ir::Alloca> ir_alloca) {
        if (ir_alloca->prajna_ir_value) {
            return;
        }

        auto ir_tensor_type = ir_alloca->type;
        if (ir_alloca->type->memory_type == ir::MemoryType::Host) {
            auto tensor_bytes = ir_tensor_type->Size() * ir_tensor_type->value_type->bytes;
            auto prajna_ir_tensor_pointer = this->prajna_ir_builder->Create<prajna::ir::BitCast>(
                this->prajna_ir_builder->Create<prajna::ir::Call>(
                    this->prajna_ir_malloc_function,
                    this->prajna_ir_builder->GetInt64Constant(tensor_bytes)),
                prajna::ir::PointerType::Create(this->EmitType(ir_tensor_type)));
            ir_alloca->prajna_ir_value =
                this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(
                    prajna_ir_tensor_pointer);
            return;
        }
        if (ir_alloca->type->memory_type == ir::MemoryType::Stack) {
            auto prajna_ir_tensor_pointer = this->prajna_ir_builder->Create<prajna::ir::Alloca>(
                this->EmitType(ir_alloca->type), this->prajna_ir_builder->GetInt64Constant(1));
            ir_alloca->prajna_ir_value =
                this->prajna_ir_builder->Create<prajna::ir::DeferencePointer>(
                    prajna_ir_tensor_pointer);
            return;
        }

        GALOIS_TODO;
    }

    void EmitFree(std::shared_ptr<ir::Free> ir_free) {
        this->EmitTensor(ir_free->Tensor());
        ir_free->prajna_ir_value = this->prajna_ir_builder->Create<prajna::ir::Call>(
            this->prajna_ir_free_function,
            this->prajna_ir_builder->Create<prajna::ir::BitCast>(
                prajna::Cast<prajna::ir::DeferencePointer>(ir_free->Tensor()->prajna_ir_value)
                    ->Pointer(),
                prajna::ir::PointerType::Create(prajna::ir::IntType::Create(8, false))));
    }

    void EmitSlice(std::shared_ptr<ir::Slice> ir_slice) {
        if (ir_slice->prajna_ir_value) {
            return;
        }

        this->EmitAccessor(ir_slice->origin);

        // stride使用被slice的tensor
        ir_slice->type->stride.resize(ir_slice->type->shape.size());
        int64_t i = ir_slice->type->stride.size() - 1;
        int64_t j = ir_slice->origin->Tensor()->type->stride.size() - 1;
        GALOIS_ASSERT(i <= j);  // 如果slice的同时降维, 需要将stride也处理下
        for (; i >= 0; --j, --i) {
            ir_slice->type->stride[i] = ir_slice->origin->Tensor()->type->stride[j];
        }
        ir_slice->type->layout = ir_slice->origin->Tensor()->type->layout;
        // 偏移地址
        ir_slice->prajna_ir_value =
            prajna::Cast<prajna::ir::VariableLiked>(ir_slice->origin->prajna_ir_value);
    }

    void EmitCall(std::shared_ptr<Call> ir_call) {
        if (ir_call->prajna_ir_value) {
            return;
        }

        for (int64_t i = 0; i < ir_call->OperandSize(); ++i) {
            this->EmitTensor(ir_call->GetOperand(i));
        }

        std::list<std::shared_ptr<prajna::ir::Value>> prajna_ir_parameters;
        for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
            prajna_ir_parameters.push_back(
                prajna::Cast<prajna::ir::DeferencePointer>(ir_call->Input(i)->prajna_ir_value)
                    ->Pointer());
        }
        for (int64_t i = 0; i < ir_call->OutputSize(); ++i) {
            prajna_ir_parameters.push_back(
                prajna::Cast<prajna::ir::DeferencePointer>(ir_call->Output(i)->prajna_ir_value)
                    ->Pointer());
        }

        ir_call->prajna_ir_value = this->prajna_ir_builder->Create<prajna::ir::Call>(
            ir_call->OperatorInstance()->prajna_ir_value, prajna_ir_parameters);
    }

   private:
    void SetPrajnaIrIndex() {}

   public:
    std::shared_ptr<prajna::lowering::IrBuilder> prajna_ir_builder = nullptr;

   private:
    std::stack<std::shared_ptr<ir::Grid>> parallel_stack;
    std::stack<std::shared_ptr<ir::OperatorInstance>> operator_stack;

    std::shared_ptr<prajna::ir::Function> prajna_ir_malloc_function = nullptr;
    std::shared_ptr<prajna::ir::Function> prajna_ir_free_function = nullptr;
};

}  // namespace galois::codegen::cpu
