#include <iostream>

#include "boost/scope/scope_exit.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

class GemmCacheFactory {
   public:
    using MatMulFunType = std::function<void *(void *, void *)>;

    static std::shared_ptr<galois::jit::Engine> GetEngine() {
        std::shared_ptr<galois::jit::Engine> engine = galois::jit::Engine::Create();
        return engine;
    }

    static MatMulFunType GetMatMulFunc(std::shared_ptr<ir::Operator> ir_operator) {
        auto input_a = ir_operator->inputs[0]->type;
        auto input_b = ir_operator->inputs[1]->type;

        std::string cache_key = GenerateCacheKey(input_a, input_b, "void");
        auto it = mat_mul_cache_.find(cache_key);
        if (it != mat_mul_cache_.end()) {
            std::cout << "Cache hit for key: " << cache_key << std::endl;
            return std::any_cast<MatMulFunType>(it->second);
        }

        auto engine = GetEngine();
        auto raw_fun = engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_operator);
        MatMulFunType mat_mul_fun = raw_fun;  // 转换为 std::function
        std::cout << "Cache miss, generated for key: " << cache_key << " at " << &mat_mul_fun
                  << std::endl;
        mat_mul_cache_[cache_key] = std::any(mat_mul_fun);
        return mat_mul_fun;
    }

    static bool HasMatMulFun(std::shared_ptr<ir::TensorType> type_a,
                             std::shared_ptr<ir::TensorType> type_b) {
        std::string cache_key = GenerateCacheKey(type_a, type_b, "void");
        return mat_mul_cache_.find(cache_key) != mat_mul_cache_.end();
    }

    static void ClearCache() { mat_mul_cache_.clear(); }

   private:
    // 修改 GenerateCacheKey，加入形状信息
    static std::string GenerateCacheKey(std::shared_ptr<ir::TensorType> type_a,
                                        std::shared_ptr<ir::TensorType> type_b,
                                        const std::string &signature) {
        auto data_type_a = type_a->DataType();
        auto data_type_b = type_b->DataType();

        // 构建 type_a 的形状字符串
        std::string shape_a_str;
        if (type_a->shape.size() > 0) {
            shape_a_str = std::to_string(type_a->shape[0]);
            for (int i = 1; i < type_a->shape.size(); ++i) {
                shape_a_str += "x" + std::to_string(type_a->shape[i]);
            }
        } else {
            shape_a_str = "scalar";  // 标量情况
        }

        // 构建 type_b 的形状字符串
        std::string shape_b_str;
        if (type_b->shape.size() > 0) {
            shape_b_str = std::to_string(type_b->shape[0]);
            for (int i = 1; i < type_b->shape.size(); ++i) {
                shape_b_str += "x" + std::to_string(type_b->shape[i]);
            }
        } else {
            shape_b_str = "scalar";  // 标量情况
        }

        // 组合数据类型、形状和签名
        return data_type_a->fullname + "_" + shape_a_str + "_" + data_type_b->fullname + "_" +
               shape_b_str + "_" + signature;
    }

    inline static std::unordered_map<std::string, std::any> mat_mul_cache_ = {};
};


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

        // 设置唯一的 fullname
        // ir_gemm_operator->fullname = "MatrixMultiply_gemm_" +
        //                             ir_data_type->fullname + "_" +
        //                             std::to_string(m) + "_" +
        //                             std::to_string(k) + "_" +
        //                             std::to_string(n);
        std::cout << "Operator fullname: " << ir_gemm_operator->fullname << std::endl;
        // 使用 GemmCacheFactory 获取 jit_engine 和 mat_mul_fun
        this->jit_engine = GemmCacheFactory::GetEngine();
        this->mat_mul_fun = GemmCacheFactory::GetMatMulFunc(ir_gemm_operator);

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
    auto p_mat_c = mat_mul_fun(this->sp_aligned256_mem_a.get(), this->sp_aligned256_mem_b.get());
    boost::scope::scope_exit free_mem([p_mat_c] { free(p_mat_c); });
    auto t1 = std::chrono::high_resolution_clock::now();
    this->galois_cost_time = static_cast<double>((t1 - t0).count());
}

auto ir_types = testing::Values(ir::f64, ir::f32);

INSTANTIATE_TEST_SUITE_P(Large, GemmPerformanceTest,
                         testing::Combine(ir_types,                //
                                          testing::Values(5),      // m
                                          testing::Values(10),     // k
                                          testing::Values(5, 6)),  // n
                         galois::test::PrintTestName);

INSTANTIATE_TEST_SUITE_P(Large2, GemmPerformanceTest,
                         testing::Combine(ir_types,                //
                                          testing::Values(5),      // m
                                          testing::Values(10),     // k
                                          testing::Values(5, 6)),  // n
                         galois::test::PrintTestName);