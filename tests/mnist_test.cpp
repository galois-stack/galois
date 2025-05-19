#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestMnist) {
    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir::f32->Tile(28 * 28, 1)}, ir::f32->Tile(10)),
        "mnist");  // 24应该改为相应的尺寸, 这里我直接把它展开了, 因为还没有reshape操作
    
    // 输入
    auto ir_input = ir_operator->inputs[0];

    // 加载权重
    auto ir_weights_type1 = ir::f32->Tile(10000, 784);
    auto ir_weights_type2 = ir::f32->Tile(10, 10000);
    auto ir_weight1 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type1, "weight1.bin");
    auto ir_weight2 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type2, "weight2.bin");

    // 第一层： 全连接 + relu
    auto ir_full1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_weight1, ir_input});
    auto ir_relu1 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_full1}, "relu6");

    // 第二层： 全连接 + softmax
    auto ir_full2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_weight2, ir_relu1});
    auto ir_squeeze_view = ir_builder->Create<ir::SqueezeView>(ir_full2);  // 这里的维度需要改成1
    auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_squeeze_view});  // softmax
    ir_builder->Create<ir::Return>(ir_softmax);

    // 生成 JIT 引擎
    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<void (*)(float *, float *)>(ir_operator);

    std::vector<float> input_vec(28 * 28);
    float value = 5.0f;
    fill_fun(input_vec.data(), &value);
    for (int i = 0; i < 28 * 28; i++) {
        GALOIS_ASSERT(input_vec[i] == value);
    }

    // 执行模型
    float output_vec[10];
    model_fun(input_vec.data(), output_vec);

    // 打印输出
    for (int i = 0; i < 10; i++) {
        std::cout << output_vec[i] << " ";
    }
    std::cout << std::endl;
}
