#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include "tests/galois_test.hpp"
#include "tests/test_utils.h"

class FillWithParamTest
    : public testing::TestWithParam<
          std::tuple<std::shared_ptr<ir::TensorType>, std::vector<int64_t>, float>> {
   public:
    void SetUp() override {
        auto [ir_data_type, shape, fill_value] = GetParam();
        this->ir_data_type = ir_data_type;
        this->shape = shape;  
        this->fill_value = fill_value;

        Eigen::VectorXi64 eigen_shape = Eigen::Map<Eigen::VectorXi64>(shape.data(), shape.size());

        auto ir_input_type = ir_data_type->Tile(eigen_shape);
        auto ir_builder = ir::Builder::Create();
        auto ir_fill_creator = op::FillCreator::Create();

        auto ir_operator =
            ir_builder->CreateOperatorByCreator(ir_fill_creator, {ir_input_type, ir_data_type});

        this->jit_engine = jit::Engine::Create();
        fill_fun = jit_engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_operator);

        auto normalize_m = ir_input_type->NormalizeShape()[0];
        auto normalize_k = ir_input_type->NormalizeShape()[1];
        this->items = normalize_m * normalize_k;
        this->sp_aligned256_mem_a = std::shared_ptr<void>(
            std::aligned_alloc(32, this->items * ir_data_type->bytes),
            [](void *p) { free(p); });
    }

    void TearDown() override {
        
        double galois_cost_time_seconds = galois_cost_time / 1e9;
        double bandwidth = items / (galois_cost_time_seconds * 1024 * 1024 * 1024);

        fmt::print("Galois cost time: {}ns, galois flops: {:.04f}gops, bandwidth: {:.04f}\n", galois_cost_time,
                   items / galois_cost_time, bandwidth);
    }

    std::function<void *(void *, void *)> fill_fun;
    std::shared_ptr<galois::jit::Engine> jit_engine;
    std::shared_ptr<void> sp_aligned256_mem_a;
    std::vector<int64_t> shape;  // 多维形状
    std::shared_ptr<ir::TensorType> ir_data_type;  

    double items;
    double galois_cost_time;
    float fill_value;
    
};

TEST_P(FillWithParamTest, TestFillWithParamTest) {

    // 执行 fill 算子
    auto t0 = std::chrono::high_resolution_clock::now();
    auto fill_ptr_c = fill_fun(this->sp_aligned256_mem_a.get(), &this->fill_value);
    auto t1 = std::chrono::high_resolution_clock::now();
    this->galois_cost_time = static_cast<double>((t1 - t0).count());

    // 验证
    void *raw_ptr = sp_aligned256_mem_a.get();

    if (auto float_type = std::dynamic_pointer_cast<ir::FloatType>(this->ir_data_type)) {
        if (float_type->bits == 32) {
            float *float_ptr = static_cast<float *>(raw_ptr);
            float float_fill_value = static_cast<float>(this->fill_value);  // 类型转换
            for (int i = 0; i < this->items; i++) {
                GALOIS_ASSERT(float_ptr[i] == float_fill_value);
            }
        } else if (float_type->bits == 64) {
            double *double_ptr = static_cast<double *>(raw_ptr);
            double double_fill_value = static_cast<double>(this->fill_value);  // 类型转换
            for (int i = 0; i < this->items; i++) {
                GALOIS_ASSERT(double_ptr[i] == double_fill_value);
            }
        }
    }
}



INSTANTIATE_TEST_SUITE_P(FillWithParamTest, FillWithParamTest,
                            testing::Combine(
                                            testing::Values(ir::f32),  // 数据类型
                                            testing::Values(
                                                std::vector<int64_t>{512, 512},           // 2D
                                                std::vector<int64_t>{512, 512, 3},        // 3D
                                                std::vector<int64_t>{512, 512, 3, 4}      // 4D
                                            ),  
                                            testing::Values(1.0f, 9.0f, 42.0f)),          // fill_value
                                            galois::test::PrintTestNameWithVector
                            );