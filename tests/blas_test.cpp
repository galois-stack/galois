#include "galois/blas/blas_factory.hpp"
#include "galois_test.hpp"

TEST(GaloisTest, TestGemmF32) {
    int64_t m = 1000;
    int64_t k = 1000;
    int64_t n = 1000;

    std::vector<float> mat_a(m * k);
    std::vector<float> mat_b(k * n);

    auto blas_factory = blas::BlasFactory::Create();
    float *p_mat_c = nullptr;
    blas_factory->GemmF32(mat_a.data(), mat_b.data(), &p_mat_c, m, k, n);
    free(p_mat_c);

    auto t0 = std::chrono::steady_clock::now();
    blas_factory->GemmF32(mat_a.data(), mat_b.data(), &p_mat_c, m, k, n);
    auto t1 = std::chrono::steady_clock::now();
    free(p_mat_c);
    auto cost_time = (t1 - t0).count();

    fmt::print("cost time {}ns: {:.04f}gops\n", cost_time,
               (double(m) * k * n * 2) / static_cast<double>(cost_time));
}
