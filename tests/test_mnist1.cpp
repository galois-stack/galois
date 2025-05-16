#include <iostream>
#include <fstream>
#include <vector>
#include "Eigen/Dense"
#include "galois/ir/builder.hpp"
#include "galois/jit/engine.hpp"
#include "galois/op/op.hpp"

using namespace galois::ir;
using namespace galois::op;
using namespace galois::jit;

int main() {
    // 1. 从二进制文件读取权重
    auto read_vec = [&](const std::string &path, std::vector<float> &vec) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        auto size = in.tellg();
        in.seekg(0, std::ios::beg);
        vec.resize(size / sizeof(float));
        in.read(reinterpret_cast<char*>(vec.data()), size);
    };
    std::vector<float> W1, b1, W2, b2;
    read_vec("../../train/weight1.bin", W1);
    read_vec("../../train/bias1.bin", b1);
    read_vec("../../train/weight2.bin", W2);
    read_vec("../../train/bias2.bin", b2);

    int input_dim = 28 * 28;
    int hidden_dim = b1.size();
    int output_dim = b2.size();

    // 2. 构建 IR
    auto builder = Builder::Create();
    // 定义张量类型
    auto input_type = f32->Tile(input_dim, 1);
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
    float value = 5.0f;
    fill_fun(input_data.data(), &value);
    for (int i = 0; i < length; i++) {
        GALOIS_ASSERT(input_vec[i] == value);
    }
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
