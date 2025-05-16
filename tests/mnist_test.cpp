// #include "galois/op/fill.hpp"
// #include "tests/galois_test.hpp"

// TEST(GaloisTests, TestMnist) {
//     auto ir_builder = ir::Builder::Create();
//     auto [ir_operator, scope] = ir_builder->CreateOperator(
//         ir::OperatorType::Create({ir::f32->Tile(24 * 24, 1)}, ir::f32->Tile(10)),
//         "mnist");  // 24应该改为相应的尺寸, 这里我直接把它展开了, 因为还没有reshape操作
//     auto ir_input = ir_operator->inputs[0];
//     auto ir_weights_type1 = ir::f32->Tile(10000, 576);
//     auto ir_weights_type2 = ir::f32->Tile(10, 10000);
//     auto ir_weight1 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type1, "weight1.bin");
//     auto ir_weight2 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type2, "weight2.bin");
//     auto ir_full1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_weight1, ir_input});
//     auto ir_relu1 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_full1}, "relu6");
//     auto ir_full2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_weight2, ir_relu1});
//     auto ir_squeeze_view = ir_builder->Create<ir::SqueezeView>(ir_full2);  // 这里的维度需要改成1
//     auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_squeeze_view});  // softmax
//     ir_builder->Create<ir::Return>(ir_softmax);

//     auto jit_engine = jit::Engine::Create();
//     auto model_fun = jit_engine->EmitOperatorSymbol<void (*)(float *, float *)>(ir_operator);

    // std::vector<float> input_vec(length);
    // float value = 5.0f;
    // fill_fun(input_vec.data(), &value);
    // for (int i = 0; i < length; i++) {
    //     GALOIS_ASSERT(input_vec[i] == value);
    // }
// }

#include <iostream>
#include <vector>
#include "cnpy.h"
#include "Eigen/Dense"
#include "galois/ir/builder.hpp"
#include "galois/jit/engine.hpp"
#include "galois/op/op.hpp"

using namespace galois::ir;
using namespace galois::op;
using namespace galois::jit;

TEST(GaloisTests, TestMnist) {
    // 1. 加载权重
    auto npz = cnpy::npz_load("../../train/weights.npz");
    auto W1 = npz["W1"].as_vec<float>();
    auto b1 = npz["b1"].as_vec<float>();
    auto W2 = npz["W2"].as_vec<float>();
    auto b2 = npz["b2"].as_vec<float>();

    int input_dim = 28 * 28;
    int hidden_dim = b1.size();
    int output_dim = b2.size();

    // 2. 构建 IR
    auto builder = Builder::Create();
    // 定义张量类型
    auto input_type = f32->Tile(1, input_dim);
    auto W1_type    = f32->Tile(input_dim, hidden_dim);
    auto b1_type    = f32->Tile(1, hidden_dim);
    auto W2_type    = f32->Tile(hidden_dim, output_dim);
    auto b2_type    = f32->Tile(1, output_dim);

    // 将输入、权重和偏置作为函数参数创建 IR 输入
    auto input     = builder->Create<ir::Input>(input_type);
    auto W1_input  = builder->Create<ir::Input>(W1_type);
    auto b1_input  = builder->Create<ir::Input>(b1_type);
    auto W2_input  = builder->Create<ir::Input>(W2_type);
    auto b2_input  = builder->Create<ir::Input>(b2_type);

    // 第一层：全连接 + ReLU
    auto mat1 = builder->CreateOperatorByCreator<MatrixMultiplyCreator>({input, W1_input});
    auto bias1 = builder->CreateOperatorByCreator<BroadcastCreator>({b1_input}, Eigen::VectorXi64::Constant(2, hidden_dim));
    auto add1 = builder->CreateOperatorByCreator<AddCreator>({mat1, bias1});
    auto act1 = builder->CreateOperatorByCreator<UnaryInstrinsicCreator>({add1}, "relu");

    // 第二层：全连接
    auto mat2 = builder->CreateOperatorByCreator<MatrixMultiplyCreator>({act1, W2_input});
    auto bias2 = builder->CreateOperatorByCreator<BroadcastCreator>({b2_input}, Eigen::VectorXi64::Constant(2, output_dim));
    auto out = builder->CreateOperatorByCreator<AddCreator>({mat2, bias2});

    // 3. JIT 编译并执行
    auto engine = Engine::Create();
    // 生成接受 5 个输入指针的函数：input, W1, b1, W2, b2
    auto infer_fun = engine->EmitOperatorSymbol<float*(*)(float*, float*, float*, float*, float*)>(out);
    std::vector<float> input_data(input_dim, 1.0f);
    // 调用时传入数据与权重&偏置指针
    float* result = infer_fun(input_data.data(), W1.data(), b1.data(), W2.data(), b2.data());

    // 4. 打印输出
    for (int i = 0; i < output_dim; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    free(result);

    return 0;
} 
