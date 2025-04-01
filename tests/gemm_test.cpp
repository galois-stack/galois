#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestGemmF32) {
    using DataType = float;
    auto ir_mat_type_a = ir::f32->Tile(1536, 1024);
    auto ir_mat_type_b = ir::f32->Tile(1024, 1024);

    auto ir_builder = ir::Builder::Create();
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_mat_type_c =
        ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

    auto ir_operator_type = ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b}, ir_mat_type_c);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_mat_type_a, ir_mat_type_b});
    auto gemm_optimizer = optimization::GemmOptimizer::Create();
    auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<DataType *(*)(DataType *, DataType *)>(ir_gemm_operator);

    auto shape_a = ir_mat_type_a->NormalizeShape();
    auto shape_b = ir_mat_type_b->NormalizeShape();

    using EigenMatrixType = Eigen::Matrix<DataType, -1, -1, Eigen::RowMajor>;
    EigenMatrixType eigen_matrix_a = EigenMatrixType::Random(shape_a[0], shape_a[1]);
    EigenMatrixType eigen_matrix_b = EigenMatrixType::Random(shape_b[0], shape_b[1]);
    auto shape_c = ir_mat_type_c->shape;

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    auto eigen_matrix_f32_expect = (eigen_matrix_a * eigen_matrix_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen flops: {:.04f}gops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto p_mat_c = mat_mul_fun(eigen_matrix_a.data(), eigen_matrix_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    auto get_galois_re = [=](int64_t i, int64_t j) -> DataType {
        return p_mat_c[i * shape_c[1] + j];
    };

    int64_t error_count = 0;
    for (int64_t i = 0; i < shape_c[0]; ++i) {
        for (int64_t j = 0; j < shape_c[1]; ++j) {
            auto galois_re = get_galois_re(i, j);
            auto eigen_re = eigen_matrix_f32_expect(i, j);
            double epllise = std::max(std::abs(0.2 * eigen_re), 0.5);
            if (std::abs(galois_re - eigen_re) > epllise) {
                fmt::print("{}, {} | galois: {}, expect: {}\n", i, j, double(galois_re),
                           double(eigen_re));
                error_count++;
                if (error_count > 20) {
                    ASSERT_TRUE(false);
                }
            }
        }
    }

    free(static_cast<void *>(p_mat_c));
}

TEST(GaloisTests, TestGemmI8) {
    using DataType = int8_t;
    auto ir_mat_type_a = ir::i8->Tile(1536, 1024);
    auto ir_mat_type_b = ir::i8->Tile(1024, 1024);

    auto ir_builder = ir::Builder::Create();
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_mat_type_c =
        ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

    auto ir_operator_type = ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b}, ir_mat_type_c);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_mat_type_a, ir_mat_type_b});
    auto gemm_optimizer = optimization::GemmOptimizer::Create();
    auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<DataType *(*)(DataType *, DataType *)>(ir_gemm_operator);

    auto shape_a = ir_mat_type_a->NormalizeShape();
    auto shape_b = ir_mat_type_b->NormalizeShape();

    using EigenMatrixType = Eigen::Matrix<DataType, -1, -1, Eigen::RowMajor>;
    EigenMatrixType eigen_matrix_a = EigenMatrixType::Random(shape_a[0], shape_a[1]);
    EigenMatrixType eigen_matrix_b = EigenMatrixType::Random(shape_b[0], shape_b[1]);
    auto shape_c = ir_mat_type_c->shape;

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    auto eigen_matrix_f32_expect = (eigen_matrix_a * eigen_matrix_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen flops: {:.04f}gops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto p_mat_c = mat_mul_fun(eigen_matrix_a.data(), eigen_matrix_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    auto get_galois_re = [=](int64_t i, int64_t j) -> DataType {
        return p_mat_c[i * shape_c[1] + j];
    };

    int64_t error_count = 0;
    for (int64_t i = 0; i < shape_c[0]; ++i) {
        for (int64_t j = 0; j < shape_c[1]; ++j) {
            auto galois_re = get_galois_re(i, j);
            auto eigen_re = eigen_matrix_f32_expect(i, j);
            double epllise = std::max(std::abs(0.2 * eigen_re), 0.5);
            if (std::abs(galois_re - eigen_re) > epllise) {
                fmt::print("{}, {} | galois: {}, expect: {}\n", i, j, double(galois_re),
                           double(eigen_re));
                error_count++;
                if (error_count > 20) {
                    ASSERT_TRUE(false);
                }
            }
        }
    }

    free(static_cast<void *>(p_mat_c));
}

TEST(GaloisTests, TestGemmF16) {
    using DataType = Eigen::half;
    auto ir_mat_type_a = ir::f16->Tile(1024, 1024);
    auto ir_mat_type_b = ir::f16->Tile(1024, 1024);

    auto ir_builder = ir::Builder::Create();
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_mat_type_c =
        ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

    auto ir_operator_type = ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b}, ir_mat_type_c);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_mat_type_a, ir_mat_type_b});
    auto gemm_optimizer = optimization::GemmOptimizer::Create();
    auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<DataType *(*)(DataType *, DataType *)>(ir_gemm_operator);

    auto shape_a = ir_mat_type_a->NormalizeShape();
    auto shape_b = ir_mat_type_b->NormalizeShape();

    using EigenMatrixType = Eigen::Matrix<DataType, -1, -1, Eigen::RowMajor>;
    EigenMatrixType eigen_matrix_a = EigenMatrixType::Random(shape_a[0], shape_a[1]);
    EigenMatrixType eigen_matrix_b = EigenMatrixType::Random(shape_b[0], shape_b[1]);
    auto shape_c = ir_mat_type_c->shape;

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    auto eigen_matrix_f32_expect = (eigen_matrix_a * eigen_matrix_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen flops: {:.04f}gops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto p_mat_c = mat_mul_fun(eigen_matrix_a.data(), eigen_matrix_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    auto get_galois_re = [=](int64_t i, int64_t j) -> DataType {
        return p_mat_c[i * shape_c[1] + j];
    };

    int64_t error_count = 0;
    for (int64_t i = 0; i < shape_c[0]; ++i) {
        for (int64_t j = 0; j < shape_c[1]; ++j) {
            auto galois_re = get_galois_re(i, j);
            auto eigen_re = eigen_matrix_f32_expect(i, j);
            double epllise = std::max(std::abs(0.2 * eigen_re), 0.5);
            if (std::abs(galois_re - eigen_re) > epllise) {
                fmt::print("{}, {} | galois: {}, expect: {}\n", i, j, double(galois_re),
                           double(eigen_re));
                error_count++;
                if (error_count > 20) {
                    ASSERT_TRUE(false);
                }
            }
        }
    }

    free(static_cast<void *>(p_mat_c));
}

class GemmPerformanceTest
    : public testing::TestWithParam<
          std::tuple<std::shared_ptr<ir::TensorType>, int64_t, int64_t, int64_t>> {
   public:
    void SetUp() override {
        auto [ir_data_type, m, k, n] = GetParam();

        auto ir_mat_type_a = ir_data_type->Tile(m, k);
        auto ir_mat_type_b = ir_data_type->Tile(k, n);

        auto ir_builder = ir::Builder::Create();
        auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
        auto ir_mat_type_c =
            ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

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

        this->sp_aligned256_mem_a = std::shared_ptr<void>(
            std::aligned_alloc(32, normalize_m * normalize_k * ir_data_type->bytes),
            [](void *p) { free(p); });
        this->sp_aligned256_mem_b = std::shared_ptr<void>(
            std::aligned_alloc(32, normalize_k * normalize_n * ir_data_type->bytes),
            [](void *p) { free(p); });
    }

    void TearDown() override {
        fmt::print("Galois cost time: {}ns, galois glops: {:.04f}gops\n", galois_cost_time,
                   items * 2 / galois_cost_time);
    }

    std::function<void *(void *, void *)> mat_mul_fun;
    std::shared_ptr<galois::jit::Engine> jit_engine;

    std::shared_ptr<void> sp_aligned256_mem_a;
    std::shared_ptr<void> sp_aligned256_mem_b;

    double items;
    double galois_cost_time;
};

TEST_P(GemmPerformanceTest, TestMatrixMultiplyGemm) {
    // 执行 Galois 矩阵乘法
    auto t0 = std::chrono::high_resolution_clock::now();
    auto mat_ptr_c = mat_mul_fun(this->sp_aligned256_mem_a.get(), this->sp_aligned256_mem_b.get());
    auto t1 = std::chrono::high_resolution_clock::now();
    this->galois_cost_time = static_cast<double>((t1 - t0).count());
    // 释放内存
    free(mat_ptr_c);
}

// TODO: has bug when i8
// INSTANTIATE_TEST_SUITE_P(Large, GemmPerformanceTest,
//                          testing::Combine(testing::Values(ir::f64, ir::f32, ir::i32,
//                                                           ir::i8),      //  f32
//                                           testing::Values(500, 1000),   // m
//                                           testing::Values(500, 1000),   // n
//                                           testing::Values(500, 1000)),  // k
//                          galois::test::PrintTestName                    // 自定义测试名称
// );

// ir::i16不支持需要修复,
INSTANTIATE_TEST_SUITE_P(Large2, GemmPerformanceTest,
                         testing::Combine(testing::Values(ir::f64, ir::f32, ir::f16, ir::i32,
                                                          ir::i16,
                                                          ir::i8),      //  f32
                                          testing::Values(512, 1024),   // m
                                          testing::Values(512, 1024),   // n
                                          testing::Values(512, 1024)),  // k
                         galois::test::PrintTestName                    // 自定义测试名称
);
