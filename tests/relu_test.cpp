#include "galois/ir/ir.hpp"
#include "galois/jit/engine.hpp"
#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestReLU) {
    int64_t length = 128;
    auto ir_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();

    auto ir_operator_type = ir::OperatorType::Create({ir_type}, ir_type);
    auto [ir_operator, scope] = ir_builder->CreateOperator(ir_operator_type, "relu");
    // fmt::print("ir_operator->inputs:{}  \n", ir_operator->inputs.size());

    auto ir_input = ir_operator->inputs[0];
    auto ir_output =
        ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_input}, "relu", false);
    ir_builder->Create<ir::Return>(ir_output);

    // 生成 JIT 引擎
    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> float_vec(length);
    for (int i = 0; i < length; ++i) {
        float_vec[i] = static_cast<float>(i);
    }

    auto output_vec = model_fun(float_vec.data());

    for (int i = 0; i < length; i++) {
        float expected = std::max(0.0f, float_vec[i]);
        GALOIS_ASSERT(output_vec[i] == expected);
    }

    free(output_vec);
}