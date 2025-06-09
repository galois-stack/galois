
#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestReLU) {
    int64_t length = 8;
    std::vector<float> float_vec = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -2.0f, -3.0f, -4.0f};
    auto ir_type = ir::f32->Tile(int64_t(float_vec.size()));
    auto ir_builder = ir::Builder::Create();

    auto ir_operator_type = ir::OperatorType::Create({}, ir_type);
    auto [ir_operator, scope] = ir_builder->CreateOperator(ir_operator_type, "relu");
    auto ir_input = ir_operator->inputs[0];
    auto ir_output = 
        ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_input}, "relu", false);
    ir_builder->Create<ir::Return>(ir_output);

    // 生成 JIT 引擎
    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    auto output_vec = model_fun(float_vec.data());

    for (int i = 0; i < length; ++i) {
        float expected = std::fmax(0.0f, float_vec[i]);
        EXPECT_NEAR(output_vec[i], expected, 1e-5)
            << "Mismatch at index " << i << ": input=" << float_vec[i];
    }
}