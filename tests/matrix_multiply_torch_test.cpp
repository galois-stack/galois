// matrix_multiply.cpp
#include "boost/scope/scope_exit.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestMatrixMultiply_Compare) {
    auto ir_data_type = ir::f32;
    auto ir_mat_type_a = ir_data_type->Tile(4, 1024);
    auto ir_mat_type_b = ir_data_type->Tile(1024, 2);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::MatrixMultiplyCreator>(
        {ir_mat_type_a, ir_mat_type_b});
    auto gemm_optimizer = optimization::GemmOptimizer::Create();
    auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_gemm_operator);

    auto normalize_m = ir_mat_type_a->NormalizeShape()[0];
    auto normalize_k = ir_mat_type_a->NormalizeShape()[1];
    auto normalize_n = ir_mat_type_b->NormalizeShape()[1];

    auto sp_aligned256_mem_a = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_k * ir::f32->bytes),
        [](void *p) { free(p); });
    auto sp_aligned256_mem_b = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_k * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });

    auto sp_aligned256_mem_c_torch = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });

    float *a_data = static_cast<float *>(sp_aligned256_mem_a.get());
    float *b_data = static_cast<float *>(sp_aligned256_mem_b.get());
    float *c_data_torch = static_cast<float *>(sp_aligned256_mem_c_torch.get());

    for (int i = 0; i < normalize_m * normalize_k; ++i) {
        a_data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
    for (int i = 0; i < normalize_k * normalize_n; ++i) {
        b_data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }

    // for (int i = 0; i < normalize_m * normalize_k; ++i) {
    //     a_data[i] = static_cast<float>(rand() % 11);
    // }
    // for (int i = 0; i < normalize_k * normalize_n; ++i) {
    //     b_data[i] = static_cast<float>(rand() % 11);
    // }
    std::fill(c_data_torch, c_data_torch + normalize_m * normalize_n, 0.0f);

    auto c_data = mat_mul_fun(static_cast<float *>(sp_aligned256_mem_a.get()),
                              static_cast<float *>(sp_aligned256_mem_b.get()));

    calculate_error(sp_aligned256_mem_a.get(), sp_aligned256_mem_b.get(),
                    sp_aligned256_mem_c_torch.get(), normalize_m, normalize_k, normalize_n);

    int length = normalize_m * normalize_n;
    int error_num = 0;
    for (int i = 0; i < length; i++) {
        if (error_num >= 10) {
            GALOIS_ASSERT(error_num == 0);
        }
        float result = c_data[i];
        float result_torch = c_data_torch[i];
        fmt::print("Index:{:2d} Galois:{:.6f} Pytorch:{:.6f} \n", i, result, result_torch);
        if (std::abs(result - result_torch) > 1e-5f) {
            error_num++;
        }
    }
}