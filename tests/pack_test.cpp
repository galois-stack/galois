#include "galois/op/pack.hpp"
#include <vector>
#include "tests/galois_test.hpp"

void BM_Pack(benchmark::State& state) {
    auto length = int64_t(state.range(0));
    // 待打包的张量形状
    auto ir_input_type = ir::f32->Tile(4, length / 4);  // f32[4, length/4]
    // 打包的张量形状
    auto ir_pack_type = ir::f32->Tile(4, 4)->Tile(1, length / 16);  // f32[1, length/16, 4, 4]

    auto ir_builder = ir::Builder::Create();
    auto ir_pack_creator = op::PackCreator::Create(ir_pack_type);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_pack_creator, {ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto pack_fun = jit_engine->EmitOperatorSymbol<float * (*)(float *, float *)>(ir_operator);

    // 设置输入张量：f32[4, length/4]
    std::vector<float> input(4 * (length / 4));  // 4 * (length/4) = length 个元素

    // 设置输出张量：f32[1, length/16, 4, 4]
    std::vector<float> output(1 * (length / 16) * 4 * 4);  // 1 * (length/16) * 4 * 4 = length 个元素

    for (auto _ : state) {
        // 调用 pack_fun
        pack_fun(input.data(), output.data());
    }
    // 统计基准测试处理的字节数和元素数量
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
}

BENCHMARK(BM_Pack)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);


TEST(GaloisTests, TestPack) {
    // 待打包的张量形状
    auto ir_input_type = ir::f32->Tile(4, 8);  // f32[4, 8]
    // 打包的张量形状
    auto ir_pack_type = ir::f32->Tile(4, 4)->Tile(1, 2);  // f32[1, 2, 4, 4]

    auto ir_builder = ir::Builder::Create();
    auto ir_pack_creator = op::PackCreator::Create(ir_pack_type);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_pack_creator, {ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto pack_fun = jit_engine->EmitOperatorSymbol<float * (*)(float *, float *)>(ir_operator);

    // 设置输入张量：f32[4, 8]
    std::vector<float> input(4 * 8);  // 4×8 = 32 个元素
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            input[i * 8 + j] = static_cast<float>(i * 8 + j);
        }
    }

    // 设置输出张量：f32[1, 2, 4, 4]
    std::vector<float> output(1 * 2 * 4 * 4);  // 1×2×4×4 = 32 个元素
    std::fill(output.begin(), output.end(), 0.0f);  // 初始化为 0

    // 调用 pack_fun
    float* output_ptr = pack_fun(input.data(), output.data());

    // 验证输出的值
    for (int b = 0; b < 2; b++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                //output_ptr stride = [32, 16, 4, 1]
                int output_idx = 0 * (2 * 4 * 4) + b * (4 * 4) + h * 4 + w;
                int input_row = (b * 4 + h) % 4;  // 按行顺序填充
                int input_col = w + (b * 4);      // 每块 4×4，列偏移
                int input_idx = input_row * 8 + input_col;
                std::string message = "打包后的值索引 [0, " + std::to_string(b) + ", " +
                                      std::to_string(h) + ", " + std::to_string(w) +
                                      "] 对应原张量值的索引 [" + std::to_string(input_row) +
                                      ", " + std::to_string(input_col) + "]";
                GALOIS_ASSERT(output_ptr[output_idx] == input[input_idx], message);
            }
        }
    }

    
}