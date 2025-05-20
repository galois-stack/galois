#pragma once

#include "galois/codegen/cpu/prajna_codegen.hpp"
#include "galois/helper.hpp"
#include "galois/ir/ir.hpp"
#include "galois/transform/reference_count_visitor.hpp"
#include "prajna/bindings/core.hpp"
#include "prajna/jit/execution_engine.h"
#include "thpool.h"

inline float relu6(float x) { return std::fminf(0.0f, x); }

namespace galois::jit {

class Engine {
   protected:
    Engine() = default;

   public:
    static std::shared_ptr<Engine> Create() {
        std::shared_ptr<Engine> self(new Engine);
        self->prajna_compiler = Engine::CreatePrajnaCompiler();
        return self;
    }

    static std::shared_ptr<prajna::Compiler> CreatePrajnaCompiler() {
        auto prajna_compiler = prajna::Compiler::Create(false);
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_init),
                                                   "thpool_init");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_add_work),
                                                   "thpool_add_work");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_wait),
                                                   "thpool_wait");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_destroy),
                                                   "thpool_destroy");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(aligned_alloc),
                                                   "aligned_alloc");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(malloc), "malloc");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(free), "free");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(auto_aligned_alloc),
                                                   "auto_aligned_alloc");
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(relu6),
                                                   "llvm.relu6.f32");
        return prajna_compiler;
    }

    template <typename Func>
    Func EmitOperatorSymbol(std::shared_ptr<ir::Operator> ir_operator) {
        if (!ir_operator->pir_value) {
            auto reference_count_visitor = transform::ReferenceCountVisitor::Create();
            ir_operator->ApplyVisitor(reference_count_visitor);

            auto prajna_codegen =
                codegen::cpu::PrajnaCodegen::Create(prajna_compiler->_symbol_table);
            prajna_codegen->EmitOperator(ir_operator);
            prajna_compiler->GenLlvm(prajna_codegen->pir_builder->module);

            transform::Each<ir::Tensor>(
                ir_operator, [&](std::shared_ptr<ir::Tensor> ir_tensor) { ir_tensor->Detach(); });
        }

        GALOIS_ASSERT(ir_operator->pir_value->llvm_value);
        return reinterpret_cast<Func>(
            prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));
    }

   private:
    std::shared_ptr<prajna::Compiler> prajna_compiler;
};

}  // namespace galois::jit
