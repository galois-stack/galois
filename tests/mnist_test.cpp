#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestMnist) {
    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir::f32->Tile(24 * 24, 1)}, ir::f32->Tile(10)),
        "mnist");  // 24应该改为相应的尺寸, 这里我直接把它展开了, 因为还没有reshape操作
    auto ir_input = ir_operator->inputs[0];
    auto ir_weights_type1 = ir::f32->Tile(10000, 576);
    auto ir_weights_type2 = ir::f32->Tile(10, 10000);
    auto ir_weight1 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type1, "weight1.bin");
    auto ir_weight2 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type2, "weight2.bin");
    auto ir_full1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_weight1, ir_input});
    auto ir_relu1 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_full1}, "relu6");
    auto ir_full2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_weight2, ir_relu1});
    auto ir_squeeze_view = ir_builder->Create<ir::SqueezeView>(ir_full2);  // 这里的维度需要改成1
    auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_squeeze_view});  // softmax
    ir_builder->Create<ir::Return>(ir_softmax);

    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<void (*)(float *, float *)>(ir_operator);

    // std::vector<float> input_vec(length);
    // float value = 5.0f;
    // fill_fun(input_vec.data(), &value);
    // for (int i = 0; i < length; i++) {
    //     GALOIS_ASSERT(input_vec[i] == value);
    // }
}
