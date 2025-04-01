#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

template <typename DataType>
class GetGaloisIrType;

template <>
class GetGaloisIrType<float> {
   public:
    static std::shared_ptr<ir::TensorType> GetType() { return ir::f32; }
};
template <>
class GetGaloisIrType<double> {
   public:
    static std::shared_ptr<ir::TensorType> GetType() { return ir::f64; }
};

template <>
class GetGaloisIrType<int8_t> {
   public:
    static std::shared_ptr<ir::TensorType> GetType() { return ir::i8; }
};
template <>
class GetGaloisIrType<int16_t> {
   public:
    static std::shared_ptr<ir::TensorType> GetType() { return ir::i16; }
};

template <>
class GetGaloisIrType<int32_t> {
   public:
    static std::shared_ptr<ir::TensorType> GetType() { return ir::i32; }
};

template <typename T>
class TestGemm : public testing::Test {
   public:
    using DataType = T;
    static std::shared_ptr<ir::TensorType> GetIrType() {
        return GetGaloisIrType<DataType>::GetType();
    }

    void SetUp() override {
        auto ir_mat_type_a = GetIrType()->Tile(1536, 1024);
        auto ir_mat_type_b = GetIrType()->Tile(1024, 1024);

        auto ir_builder = ir::Builder::Create();
        auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
        auto ir_mat_type_c =
            ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

        auto ir_operator_type =
            ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b}, ir_mat_type_c);
        auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                               {ir_mat_type_a, ir_mat_type_b});
        auto gemm_optimizer = optimization::GemmOptimizer::Create();
        auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

        this->jit_engine = jit::Engine::Create();
        this->mat_mul_fun =
            jit_engine->EmitOperatorSymbol<DataType *(*)(DataType *, DataType *)>(ir_gemm_operator);

        this->normalize_m = ir_mat_type_a->NormalizeShape()[0];
        this->normalize_k = ir_mat_type_a->NormalizeShape()[1];
        this->normalize_n = ir_mat_type_b->NormalizeShape()[1];
        this->items = normalize_m * normalize_k * normalize_n;

        this->eigen_matrix_a = EigenMatrixType::Random(this->normalize_m, this->normalize_k);
        this->eigen_matrix_b = EigenMatrixType::Random(this->normalize_k, this->normalize_n);
        this->shape_c = ir_mat_type_c->shape;
    }
    void TearDown() override {
        fmt::print("cost time: {}ns, eigen glops: {:.04f}gops\n", this->eigen_cost_time,
                   this->items * 2 / static_cast<double>(this->eigen_cost_time));

        fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", this->galois_cost_time,
                   this->items * 2 / static_cast<double>(this->galois_cost_time));
    }

    std::shared_ptr<jit::Engine> jit_engine;
    std::function<DataType *(DataType *, DataType *)> mat_mul_fun;
    double galois_cost_time = 0;
    double eigen_cost_time = 0;

    using EigenMatrixType = Eigen::Matrix<DataType, -1, -1, Eigen::RowMajor>;
    EigenMatrixType eigen_matrix_a;
    EigenMatrixType eigen_matrix_b;

    Eigen::VectorXi64 shape_c;

    int64_t normalize_m;
    int64_t normalize_k;
    int64_t normalize_n;
    double items;
};

TYPED_TEST_SUITE_P(TestGemm);

TYPED_TEST_P(TestGemm, MatrixMultiplyCorrectness) {
    auto t0_eigen = std::chrono::high_resolution_clock::now();
    auto eigen_matrix_f32_expect = (this->eigen_matrix_a * this->eigen_matrix_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    this->eigen_cost_time = static_cast<double>((t1_eigen - t0_eigen).count());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto p_mat_c = this->mat_mul_fun(this->eigen_matrix_a.data(), this->eigen_matrix_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    this->galois_cost_time = static_cast<double>((t1 - t0).count());

    auto get_galois_re = [=](int64_t i, int64_t j) -> TypeParam {
        return p_mat_c[i * this->shape_c[1] + j];
    };

    int64_t error_count = 0;
    for (int64_t i = 0; i < this->shape_c[0]; ++i) {
        for (int64_t j = 0; j < this->shape_c[1]; ++j) {
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

using ScalarTypes = ::testing::Types<double, float, int8_t, int16_t, int32_t>;

REGISTER_TYPED_TEST_SUITE_P(TestGemm, MatrixMultiplyCorrectness);

INSTANTIATE_TYPED_TEST_SUITE_P(GaloisGemmTests, TestGemm, ScalarTypes);

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
            galois::auto_aligned_alloc(normalize_m * normalize_k * ir_data_type->bytes),
            [](void *p) { free(p); });
        this->sp_aligned256_mem_b = std::shared_ptr<void>(
            galois::auto_aligned_alloc(normalize_k * normalize_n * ir_data_type->bytes),
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

auto ir_types = testing::Values(ir::f64, ir::f32, ir::f16, ir::i32, ir::i16, ir::i8);

INSTANTIATE_TEST_SUITE_P(Scalar, GemmPerformanceTest,
                         testing::Combine(ir_types,
                                          testing::Values(1),   // m
                                          testing::Values(1),   // k
                                          testing::Values(1)),  // n
                         galois::test::PrintTestName);

INSTANTIATE_TEST_SUITE_P(Large, GemmPerformanceTest,
                         testing::Combine(ir_types,                     //
                                          testing::Values(500, 1000),   // m
                                          testing::Values(500, 1000),   // k
                                          testing::Values(500, 1000)),  // n
                         galois::test::PrintTestName);

INSTANTIATE_TEST_SUITE_P(Large2, GemmPerformanceTest,
                         testing::Combine(ir_types,                     //
                                          testing::Values(512, 1024),   // m
                                          testing::Values(512, 1024),   // k
                                          testing::Values(512, 1024)),  // n
                         galois::test::PrintTestName);
