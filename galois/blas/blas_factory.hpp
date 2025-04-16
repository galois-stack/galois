#pragma once

#include "galois/galois.hpp"

namespace galois::blas {

class BlasFactory {
   public:
    static std::shared_ptr<BlasFactory> Create() {
        std::shared_ptr<BlasFactory> self(new BlasFactory);
        self->jit_engine = jit::Engine::Create();
        return self;
    }

    void GemmF32(float *p_mat_a, float *p_mat_b, float **pp_mat_c, int64_t m, int64_t k,
                 int64_t n) {
        auto ir_mat_type_a = ir::f32->Tile(m, k);
        auto ir_mat_type_b = ir::f32->Tile(k, n);
        auto ir_a_b_type_tuple = std::make_tuple(ir_mat_type_a, ir_mat_type_b);
        if (!gemm_symbol_dict.count(ir_a_b_type_tuple)) {
            auto ir_builder = ir::Builder::Create();
            auto ir_operator = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
                {ir_mat_type_a, ir_mat_type_b});
            auto gemm_optimizer = optimization::GemmOptimizer::Create();
            auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);
            gemm_symbol_dict[ir_a_b_type_tuple] =
                jit_engine->EmitOperatorSymbol<int64_t>(ir_gemm_operator);
        }

        auto fun_symbol = gemm_symbol_dict[ir_a_b_type_tuple];
        *pp_mat_c = reinterpret_cast<float *(*)(float *, float *)>(fun_symbol)(p_mat_a, p_mat_b);
    }

   private:
    std::map<std::tuple<std::shared_ptr<ir::TensorType>, std::shared_ptr<ir::TensorType>>, int64_t>
        gemm_symbol_dict;
    std::shared_ptr<jit::Engine> jit_engine;
};

}  // namespace galois::blas
