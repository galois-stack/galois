
#include <functional>

#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestMatrixMultiplyKernelExpand) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = ir::f16->Tile(8, 1)->Tile(3, 1)->Tile(1, 32)->Tile(4, 1)->Tile(10, 10);
    auto ir_ts_type_b = ir::f16->Tile(1, 8)->Tile(1, 1)->Tile(32, 1)->Tile(1, 4)->Tile(10, 10);

    auto ir_builder = ir::Builder::Create();
    ir_builder->matrix_multiply_kernel_queue.push_back(op::SimdMatrixMultiplyKernel::Create(128));
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});

    auto ir_grid = Cast<ir::Grid>(transform::GetInnerMostBlock(ir_operator));
    GALOIS_ASSERT(ir_grid);
    optimization::ExpandGrid(ir_grid);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun = jit_engine->EmitOperatorSymbol<int8_t *(*)(int8_t *, int8_t *)>(ir_operator);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXi8 eigen_matrix_i8_a = Eigen::MatrixRXi8::Zero(shape_a[0], shape_a[1]);
    Eigen::MatrixRXi8 eigen_matrix_i8_b = Eigen::MatrixRXi8::Zero(shape_b[0], shape_b[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ir_mat_c_ptr = mat_mul_fun(eigen_matrix_i8_a.data(), eigen_matrix_i8_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, galois flops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    free(ir_mat_c_ptr);
}

class TileMatrixMultiplyPerformanceTest
    : public testing::TestWithParam<
          std::tuple<std::shared_ptr<ir::TensorType>, int64_t, int64_t, int64_t>> {
   public:
    void SetUp() override {
        auto [ir_data_type, m, k, n] = GetParam();
        auto native_cpu_info = optimization::NativeCpuInfo::Create();
        auto mat_mul_tile_policy = optimization::MatrixMultiplyTilePolicy::Create();
        auto [ir_mat_type_a, ir_mat_type_b, mat_mul_kernel] =
            mat_mul_tile_policy->Tile(ir_data_type, native_cpu_info);
        ir_mat_type_a = ir_mat_type_a->Tile(m, k);
        ir_mat_type_b = ir_mat_type_b->Tile(k, n);

        auto ir_builder = ir::Builder::Create();
        ir_builder->matrix_multiply_kernel_queue.push_back(mat_mul_kernel);
        auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
        auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                               {ir_mat_type_a, ir_mat_type_b});

        this->jit_engine = jit::Engine::Create();
        mat_mul_fun = jit_engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_operator);

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
        fmt::print("cost time: {}ns, galois glops: {:.04f}gops\n", this->cost_time,
                   this->items * 2 / this->cost_time);
    }

    std::function<void *(void *, void *)> mat_mul_fun;
    std::shared_ptr<void> sp_aligned256_mem_a;
    std::shared_ptr<void> sp_aligned256_mem_b;
    std::shared_ptr<jit::Engine> jit_engine;

    double items;
    double cost_time;
};

TEST_P(TileMatrixMultiplyPerformanceTest, TestTilePolicy) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto mat_ptr_c = mat_mul_fun(this->sp_aligned256_mem_a.get(), this->sp_aligned256_mem_b.get());
    auto t1 = std::chrono::high_resolution_clock::now();
    this->cost_time = static_cast<double>((t1 - t0).count());
    free(mat_ptr_c);
}

// ir::i16不支持需要修复,
INSTANTIATE_TEST_SUITE_P(
    General, TileMatrixMultiplyPerformanceTest,
    testing::Combine(testing::Values(ir::f64, ir::f32, ir::f16, ir::i32, ir::i16, ir::i8),
                     testing::Values(16, 32, 64, 128), testing::Values(16, 32, 64, 128),
                     testing::Values(16, 32, 64, 128)),
    galois::test::PrintTestName);
