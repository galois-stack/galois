
#include <functional>

#include "boost/scope/scope_exit.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

class TileMatrixMultiplyPerformanceTest
    : public testing::TestWithParam<
          std::tuple<std::shared_ptr<ir::TensorType>, int64_t, int64_t, int64_t>> {
   public:
    void SetUp() override {
        auto [ir_data_type, m, k, n] = GetParam();
        auto native_cpu_info = optimization::NativeCpuInfo::Create();
        auto mat_mul_tile_policy = optimization::GemmTilePolicy::Create();
        auto [ir_mat_type_a, ir_mat_type_b, mat_mul_kernel] =
            mat_mul_tile_policy->Tile(ir_data_type, native_cpu_info);
        ir_mat_type_a = ir_mat_type_a->Tile(m, k);
        ir_mat_type_b = ir_mat_type_b->Tile(k, n);

        auto ir_builder = ir::Builder::Create();
        ir_builder->matrix_multiply_kernel_queue.push_back(mat_mul_kernel);
        auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
        auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                               {ir_mat_type_a, ir_mat_type_b});

        auto ir_register_tile_grid = optimization::GetInnerGrid3(ir_operator->block);
        optimization::ExpandGrid(ir_register_tile_grid);

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
    auto p_mat_c = mat_mul_fun(this->sp_aligned256_mem_a.get(), this->sp_aligned256_mem_b.get());
    boost::scope::scope_exit free_mem([p_mat_c] { free(p_mat_c); });
    auto t1 = std::chrono::high_resolution_clock::now();
    this->cost_time = static_cast<double>((t1 - t0).count());
}

// ir::i16不支持需要修复,
INSTANTIATE_TEST_SUITE_P(
    General, TileMatrixMultiplyPerformanceTest,
    testing::Combine(testing::Values(ir::f64, ir::f32, ir::f16, ir::i32, ir::i16, ir::i8),
                     testing::Values(16, 32, 64, 128), testing::Values(16, 32, 64, 128),
                     testing::Values(16, 32, 64, 128)),
    galois::test::PrintTestName);
