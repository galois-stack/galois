#include "galois/op/slice.hpp"

#include <numeric>
#include <vector>

#include "tests/galois_test.hpp"

void BM_Slice(benchmark::State &state) {
    auto length = int64_t(state.range(0));

    // 编译期：开始build 算子的 ir
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto shape = ir_input_type->shape / 2;  // 裁剪尺寸的一半
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::SliceCreator>({ir_input_type}, shape);

    auto jit_engine = jit::Engine::Create();
    auto slice_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input_vec(length);
    for (auto _ : state) {
        slice_fun(input_vec.data());
    }

    // 统计基准测试处理的字节数和元素数量
    //  state.iterations() 记录了基准测试运行的次数。SetBytesProcessed(...) 计算
    //  处理的总字节数。 SetItemsProcessed(...) 计算 填充的总元素个数。
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
}

BENCHMARK(BM_Slice)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);

TEST(GaloisTests, TestSlice) {
    Eigen::VectorXi64 shape(1);
    shape << 8;  // 显式初始化

    auto ir_input_type = ir::f32->Tile(16);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::SliceCreator>({ir_input_type}, shape);

    auto jit_engine = jit::Engine::Create();
    auto slice_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input_vec(16);                  //  16 elements
    std::iota(input_vec.begin(), input_vec.end(), 1);  // [1, 2, ..., 16]
    auto result = slice_fun(input_vec.data());
    for (int i = 0; i < 8; i++) {           // 8 elements
        GALOIS_ASSERT(result[i] == i + 1);  // [1, 2, 3, 4, 5, 6, 7, 8]
    }
    delete[] result;
}
