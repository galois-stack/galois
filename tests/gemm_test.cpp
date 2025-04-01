#include <cstdint>
#include <iostream>

#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "gtest/gtest.h"
#include "tests/galois_test.hpp"

template <int Bits, bool isFloat>
struct TensorTypeWrapper {
    static std::shared_ptr<galois::ir::TensorType> get() {
        if constexpr (isFloat) {
            return galois::ir::FloatType::Create(Bits);
        } else {
            return galois::ir::IntType::Create(Bits, true);
        }
    }

    // 定义 ValueType 映射到原生类型
    using ValueType = typename std::conditional_t<
        isFloat,
        std::conditional_t<
            Bits == 16, float, 
            std::conditional_t<Bits == 32, float, std::conditional_t<Bits == 64, double, void>>>,
        std::conditional_t<
            Bits == 8, int8_t,
            std::conditional_t<Bits == 16, int16_t,
                               std::conditional_t<Bits == 32, int32_t,
                                                  std::conditional_t<Bits == 64, int64_t, void>>>>>;
};

using F16Type = TensorTypeWrapper<16, true>;
using F32Type = TensorTypeWrapper<32, true>;
using F64Type = TensorTypeWrapper<64, true>;
using i8Type = TensorTypeWrapper<8, false>;
using i16Type = TensorTypeWrapper<16, false>;

template <typename T>
class TestGemm : public testing::Test {
   public:
    void SetUp() override {
        auto tensorType = T::get();
        auto ir_mat_type_a = tensorType->Tile(1536, 1024);
        auto ir_mat_type_b = tensorType->Tile(1024, 1024);

        auto ir_builder = ir::Builder::Create();
        auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
        auto ir_mat_type_c =
            ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

        auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                               {ir_mat_type_a, ir_mat_type_b});
        auto gemm_optimizer = optimization::GemmOptimizer::Create();
        auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

        this->jit_engine = jit::Engine::Create();
        this->mat_mul_fun =
            jit_engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_gemm_operator);

        this->shape_c = ir_mat_type_c->shape;


        this->normalize_m = ir_mat_type_a->NormalizeShape()[0];
        this->normalize_k = ir_mat_type_a->NormalizeShape()[1];
        this->normalize_n = ir_mat_type_b->NormalizeShape()[1];
        this->items = normalize_m * normalize_k * normalize_n;
        this->ir_data_type = tensorType;


        this->sp_aligned256_mem_a = std::shared_ptr<void>(
            galois::auto_aligned_alloc(normalize_m * normalize_k * ir_data_type->bytes),
            [](void *p) { free(p); });
        this->sp_aligned256_mem_b = std::shared_ptr<void>(
            galois::auto_aligned_alloc(normalize_k * normalize_n * ir_data_type->bytes),
            [](void *p) { free(p); });

        // 使用 Eigen::Map 初始化矩阵
        Eigen::Map<MatrixType> matrix_a(
            static_cast<ScalarType*>(sp_aligned256_mem_a.get()), normalize_m, normalize_k);
        Eigen::Map<MatrixType> matrix_b(
            static_cast<ScalarType*>(sp_aligned256_mem_b.get()), normalize_k, normalize_n);
        matrix_a.setOnes();
        matrix_b.setOnes();    
    }

    void TearDown() override {
        fmt::print("eigen cost time: {}ns, eigen flops: {:.04f}gops\n", this->eigen_cost_time,
                   this->items * 2 / static_cast<double>(this->eigen_cost_time));

        fmt::print("galois cost time: {}ns, galois flops: {:.04f}gops\n", this->galois_cost_time,
                   this->items * 2 / static_cast<double>(this->galois_cost_time));
    }

    std::shared_ptr<jit::Engine> jit_engine;
    std::function<void *(void *, void *)> mat_mul_fun;
    double galois_cost_time = 0;
    double eigen_cost_time = 0;
    double items;

    std::shared_ptr<void> sp_aligned256_mem_a;
    std::shared_ptr<void> sp_aligned256_mem_b;
    int64_t normalize_m;
    int64_t normalize_k;    
    int64_t normalize_n;
    Eigen::VectorXi64 shape_c;
    std::shared_ptr<ir::TensorType> ir_data_type ;

    using ScalarType = typename T::ValueType;  // T 的原生类型
    using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;  // 动态矩阵类型
};

using TensorTypes = ::testing::Types<F16Type, F32Type, F64Type, i8Type, i16Type>;

TYPED_TEST_SUITE_P(TestGemm);

TYPED_TEST_P(TestGemm, MatrixMultiplyCorrectness) {


    using ScalarType = typename TestFixture::ScalarType;
    using MatrixType = typename TestFixture::MatrixType;

    // Eigen 计算
    Eigen::Map<MatrixType> matrix_a(
        static_cast<ScalarType*>(this->sp_aligned256_mem_a.get()), this->normalize_m, this->normalize_k);
    Eigen::Map<MatrixType> matrix_b(
        static_cast<ScalarType*>(this->sp_aligned256_mem_b.get()), this->normalize_k, this->normalize_n);

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    MatrixType eigen_matrix_expect = (matrix_a * matrix_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    this->eigen_cost_time = static_cast<double>((t1_eigen - t0_eigen).count());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result_ptr = this->mat_mul_fun(this->sp_aligned256_mem_a.get(), this->sp_aligned256_mem_b.get());
    auto t1 = std::chrono::high_resolution_clock::now();
    this->galois_cost_time = static_cast<double>((t1 - t0).count());

    // 使用泛型类型获取结果
    auto get_result = [=](int64_t i, int64_t j) -> ScalarType {
        return static_cast<ScalarType*>(result_ptr)[i * this->shape_c[1] + j];
    };
    // 正确性验证
    int64_t error_count = 0;
    for (int64_t i = 0; i < this->shape_c[0]; ++i) {
        for (int64_t j = 0; j < this->shape_c[1]; ++j) {
            ScalarType result = get_result(i, j);
            ScalarType expect = eigen_matrix_expect(i, j);
            // 使用浮点比较，转换为 double 以避免精度问题
            double result_d = static_cast<double>(result);
            double expect_d = static_cast<double>(expect);
            if ((std::abs(result_d - expect_d)) /
                    std::max(std::abs(expect_d), std::abs(result_d)) > 0.1) {
                fmt::print("err pos: {},{}; {}, {}\n", i, j, result, expect);
                ++error_count;
                if (error_count > 100) {
                    std::terminate();
                }
            }
        }
    }

    free(static_cast<void *>(result_ptr));

}

REGISTER_TYPED_TEST_SUITE_P(TestGemm, MatrixMultiplyCorrectness);

INSTANTIATE_TYPED_TEST_SUITE_P(GaloisGemmTests, TestGemm, TensorTypes);

// TEST(GaloisTests, TestGemm) {
//     //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
//     auto ir_mat_type_a = f32->Tile(1536, 1024);
//     auto ir_mat_type_b = f32->Tile(1024, 1024);

//     auto ir_builder = ir::Builder::Create();
//     auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
//     auto ir_mat_type_c =
//         ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});

//     auto ir_operator_type = ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b},
//     ir_mat_type_c); auto ir_operator =
//     ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
//                                                            {ir_mat_type_a, ir_mat_type_b});
//     auto gemm_optimizer = optimization::GemmOptimizer::Create();
//     auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

//     auto jit_engine = jit::Engine::Create();
//     auto mat_mul_fun =
//         jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_gemm_operator);

//     auto shape_a = ir_mat_type_a->NormalizeShape();
//     auto shape_b = ir_mat_type_b->NormalizeShape();

//     Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
//     Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);
//     auto shape_c = ir_mat_type_c->shape;

//     auto t0_eigen = std::chrono::high_resolution_clock::now();
//     Eigen::MatrixRXf32 eigen_matrix_f32_expect = (eigen_matrix_f32_a *
//     eigen_matrix_f32_b).eval(); auto t1_eigen = std::chrono::high_resolution_clock::now();
//     fmt::print("cost time: {}ns, eigen flops: {:.04f}gops\n", (t1_eigen - t0_eigen).count(),
//                shape_a[0] * shape_a[1] * shape_b[1] * 2 /
//                    static_cast<double>((t1_eigen - t0_eigen).count()));

//     auto t0 = std::chrono::high_resolution_clock::now();
//     auto f32_c_ptr = mat_mul_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
//     auto t1 = std::chrono::high_resolution_clock::now();

//     fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", (t1 - t0).count(),
//                shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 -
//                t0).count()));

//     auto get_f32_c = [=](int64_t i, int64_t j) -> float { return f32_c_ptr[i * shape_c[1] + j];
//     };

//     int64_t error_count = 0;
//     for (int64_t i = 0; i < shape_c[0]; ++i) {
//         for (int64_t j = 0; j < shape_c[1]; ++j) {
//             if ((std::abs(get_f32_c(i, j) - eigen_matrix_f32_expect(i, j))) /
//                     std::max(std::abs(eigen_matrix_f32_expect(i, j)), std::abs(get_f32_c(i, j)))
//                     >
//                 0.1f) {
//                 fmt::print("err pos: {},{}; {}, {}\n", i, j, get_f32_c(i, j),
//                            eigen_matrix_f32_expect(i, j));
//                 ++error_count;
//                 if (error_count > 100) {
//                     std::terminate();
//                 }
//                 std::cout << std::endl;
//             }
//         }
//     }

//     free(static_cast<void *>(f32_c_ptr));
// }

TEST(GaloisTests, TestGemm) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_mat_type_a = f32->Tile(1536, 1024);
    auto ir_mat_type_b = f32->Tile(1024, 1024);

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
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_gemm_operator);

    auto shape_a = ir_mat_type_a->NormalizeShape();
    auto shape_b = ir_mat_type_b->NormalizeShape();

    Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);
    auto shape_c = ir_mat_type_c->shape;

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    Eigen::MatrixRXf32 eigen_matrix_f32_expect = (eigen_matrix_f32_a * eigen_matrix_f32_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen flops: {:.04f}gops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto f32_c_ptr = mat_mul_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    auto get_f32_c = [=](int64_t i, int64_t j) -> float { return f32_c_ptr[i * shape_c[1] + j]; };

    int64_t error_count = 0;
    for (int64_t i = 0; i < shape_c[0]; ++i) {
        for (int64_t j = 0; j < shape_c[1]; ++j) {
            if ((std::abs(get_f32_c(i, j) - eigen_matrix_f32_expect(i, j))) /
                    std::max(std::abs(eigen_matrix_f32_expect(i, j)), std::abs(get_f32_c(i, j))) >
                0.1f) {
                fmt::print("err pos: {},{}; {}, {}\n", i, j, get_f32_c(i, j),
                           eigen_matrix_f32_expect(i, j));
                ++error_count;
                if (error_count > 100) {
                    std::terminate();
                }
                std::cout << std::endl;
            }
        }
    }

    free(static_cast<void *>(f32_c_ptr));
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
