#pragma once

#include "galois/codegen/cpu/prajna_codegen.hpp"
#include "galois/helper.hpp"
#include "galois/ir/ir.hpp"
#include "galois/transform/reference_count_visitor.hpp"
#include "prajna/bindings/core.hpp"
#include "prajna/jit/execution_engine.h"
#include "thpool.h"

#include <immintrin.h>

struct tile_config {
    uint8_t palette_id;  // 配置模式:0 1
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];  // 每个 tile 的列字节数
    uint8_t rows[16];    // 每个 tile 的行数
};

inline void InitTileConfig() {
    tile_config tileinfo{};
    tileinfo.palette_id = 1;
    tileinfo.start_row = 0;
    for (int i = 0; i < 8; ++i) {
        tileinfo.colsb[i] = 64;
        tileinfo.rows[i] = 16;
    }
    _tile_loadconfig(&tileinfo);
}


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
        prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(InitTileConfig),
                                                   "InitTileConfig");
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
        }

        GALOIS_ASSERT(ir_operator->pir_value->llvm_value);
        return reinterpret_cast<Func>(
            prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));
    }

   private:
    std::shared_ptr<prajna::Compiler> prajna_compiler;
};

}  // namespace galois::jit
