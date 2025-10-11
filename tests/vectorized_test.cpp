#include <cmath>
#include <cstdlib>
#include <memory>

#include "galois/op/fill.hpp"
#include "galois/optimization/VectorizedOptimizer.hpp"
#include "galois/optimization/VectorizedOptimizer_fill.hpp"
#include "tests/galois_test.hpp"

namespace {

// 将 std::vector 形式的形状转换成 Eigen 向量，方便构造张量类型。
Eigen::VectorXi64 MakeShape(const std::vector<int64_t> &shape_vec) {
    Eigen::VectorXi64 shape(shape_vec.size());
    for (size_t i = 0; i < shape_vec.size(); ++i) {
        shape(static_cast<int>(i)) = shape_vec[i];
    }
    return shape;
}

// 把形状编码成 `[d0xd1x...]` 的字符串，便于输出结果。
std::string ShapeToString(const std::vector<int64_t> &shape_vec) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < shape_vec.size(); ++i) {
        if (i) oss << 'x';
        oss << shape_vec[i];
    }
    oss << ']';
    return oss.str();
}

using Buffer = std::unique_ptr<float, decltype(&std::free)>;

// 申请对齐内存，确保与 JIT 输出的对齐策略一致。
Buffer AllocateAligned(int64_t elements) {
    return Buffer(static_cast<float *>(galois::auto_aligned_alloc(elements * sizeof(float))),
                  &std::free);
}

// 为输入缓冲区填充可重复的浮点序列，避免使用随机数增加噪音。
void FillBuffer(float *ptr, int64_t elements, float scale) {
    for (int64_t i = 0; i < elements; ++i) {
        ptr[i] = static_cast<float>((i % 257) * scale);
    }
}

enum class AddMode { Scalar, Vectorized, VectorizedFill };

// 根据模式构造不同的加法算子：纯标量或矢量化优化版本。
std::shared_ptr<ir::Operator> BuildAddOperator(const Eigen::VectorXi64 &shape, AddMode mode) {
    auto input_type = ir::f32->Tile(shape);
    auto builder = ir::Builder::Create();
    auto base = builder->CreateOperatorByCreator<op::AddCreator>({input_type, input_type});

    switch (mode) {
        case AddMode::Scalar:
            return base;
        case AddMode::Vectorized: {
            auto optimizer = optimization::VectorizedOptimizer::Create();
            return optimizer->Optimize(base);
        }
        case AddMode::VectorizedFill: {
            auto optimizer = optimization::VectorizedOptimizerFill::Create();
            return optimizer->Optimize(base);
        }
    }
    return base;
}

// 运行单次加法算子测试，打印耗时与带宽，方便比较不同模式。
void RunAddTest(const std::vector<int64_t> &shape_vec, AddMode mode) {
    auto shape = MakeShape(shape_vec);
    int64_t elements = 1;
    for (auto dim : shape_vec) {
        elements *= dim;
    }

    auto op = BuildAddOperator(shape, mode);
    auto jit_engine = jit::Engine::Create();
    auto add_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(op);

    auto lhs = AllocateAligned(elements);
    auto rhs = AllocateAligned(elements);

    FillBuffer(lhs.get(), elements, 0.25f);
    FillBuffer(rhs.get(), elements, 1.5f);

    auto bytes = static_cast<double>(elements) * 2 * sizeof(float);

    auto t0 = std::chrono::steady_clock::now();
    auto result = add_fun(lhs.get(), rhs.get());
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    const char* mode_suffix = "";
    switch (mode) {
        case AddMode::Scalar:
            mode_suffix = "(scalar)";
            break;
        case AddMode::Vectorized:
            mode_suffix = "(vec)";
            break;
        case AddMode::VectorizedFill:
            mode_suffix = "(vec_fill)";
            break;
    }
    fmt::print("Add{} {}: {:.3f} ms, {:.3f} GB/s\n", mode_suffix, ShapeToString(shape_vec),
               elapsed.count() * 1e3, bytes / elapsed.count() / 1e9);

    std::free(result);
}

}  // namespace

TEST(GaloisTests, TestAddScalar) {
    // 标量模式，作为矢量化优化的基线。
    std::vector<std::vector<int64_t>> shapes = {
        {1024000000},
        {8192, 8192},
        {2048, 1024},
        {512, 512, 256},
        {128, 128, 64, 16},
    };

    for (const auto &shape : shapes) {
        RunAddTest(shape, AddMode::Scalar);
    }
}

TEST(GaloisTests, TestAddVectorized) {
    // 直接使用 VectorizedOptimizer，比对性能提升。
    std::vector<std::vector<int64_t>> shapes = {
        {102400000},
        {8192, 8192},
        {2048, 1024},
        {512, 512, 256},
        {128, 128, 64, 16},
    };

    for (const auto &shape : shapes) {
        RunAddTest(shape, AddMode::Vectorized);
    }
}

TEST(GaloisTests, TestAddVectorizedFill) {
    // 使用带 Fill 的矢量化实现，验证初始化逻辑的正确性。
    std::vector<std::vector<int64_t>> shapes = {
        {102400000},
        {8192, 8192},
        {2048, 1024},
        {512, 512, 256},
        {128, 128, 64, 16},
    };

    for (const auto &shape : shapes) {
        RunAddTest(shape, AddMode::VectorizedFill);
    }
}
