#include <cstdlib>
#include <functional>
#include <chrono>
#include <Eigen/Dense>

#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

class MatrixMultiplyGemmTest : public testing::TestWithParam<
                     std::tuple<std::shared_ptr<ir::TensorType>, int64_t, int64_t, int64_t>> {
   public:
    void SetUp() override {
        auto [ir_data_type, m, k, n] = GetParam();

        auto ir_mat_type_a = ir_data_type->Tile(m, k);
        auto ir_mat_type_b = ir_data_type->Tile(k, n);

        auto ir_builder = ir::Builder::Create();
        auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
        auto ir_mat_type_c = ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

        auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                               {ir_mat_type_a, ir_mat_type_b});

        auto gemm_optimizer = optimization::GemmOptimizer::Create();
        auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

        this->jit_engine = jit::Engine::Create();
        mat_mul_fun = jit_engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_gemm_operator);

        auto normalize_m = ir_mat_type_a->NormalizeShape()[0];
        auto normalize_k = ir_mat_type_a->NormalizeShape()[1];
        auto normalize_n = ir_mat_type_b->NormalizeShape()[1];
        this->items = normalize_m * normalize_k * normalize_n;


        this->sp_aligned256_mem_a = std::shared_ptr<void>( std::aligned_alloc(32, normalize_m * normalize_k * ir_data_type->bytes), [](void *p) { free(p); });
        this->sp_aligned256_mem_b = std::shared_ptr<void>( std::aligned_alloc(32, normalize_k * normalize_n * ir_data_type->bytes), [](void *p) { free(p); });

    }

    void TearDown() override {
        fmt::print("Galois cost time: {}ns, galois flops: {:.04f}gops\n", galois_cost_time,
                   items * 2 / galois_cost_time);
    }

    std::function<void *(void *, void *)> mat_mul_fun;
    std::shared_ptr<galois::jit::Engine> jit_engine;

    std::shared_ptr<void> sp_aligned256_mem_a;
    std::shared_ptr<void> sp_aligned256_mem_b;

    double items;
    double galois_cost_time;
};

TEST_P(MatrixMultiplyGemmTest, TestMatrixMultiplyGemm) {
    // 执行 Galois 矩阵乘法
    auto t0 = std::chrono::high_resolution_clock::now();
    auto mat_ptr_c = mat_mul_fun(this->sp_aligned256_mem_a.get(), this->sp_aligned256_mem_b.get());
    auto t1 = std::chrono::high_resolution_clock::now();
    this->galois_cost_time = static_cast<double>((t1 - t0).count());
    // 释放内存
    free(mat_ptr_c);
}




 // ir::i16不支持需要修复,
INSTANTIATE_TEST_SUITE_P(MatrixMultiplyGemmTest, MatrixMultiplyGemmTest,
    testing::Combine(testing::Values(ir::i8),  //  f32
                     testing::Values(64, 512, 1024),              // m
                     testing::Values(64, 512, 1024),              // n
                     testing::Values(64, 512, 2014)));            // k
