#include "galois/op/fill.hpp"
#include "mnist_helper.hpp"
#include "tests/galois_test.hpp"

std::vector<float> PreprocessImageData(uint8_t* p_image_data, size_t image_size) {
    std::vector<float> processed_image(image_size);
    const float mean = 0.1307f;
    const float std = 0.3081f;

    for (size_t i = 0; i < image_size; i++) {
        processed_image[i] = (static_cast<float>(p_image_data[i]) / 255.0f - mean) / std;
    }

    return processed_image;
}

TEST(GaloisTests, TestMnist) {
    MnistDataLoader mnist_data_loader("tests/data/mnist/t10k-images-idx3-ubyte.bin",
                                      "tests/data/mnist/t10k-labels-idx1-ubyte.bin");

    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir::f32->Tile(1, 28 * 28)}, ir::f32->Tile(10)), "mnist");
    auto ir_input = ir_operator->inputs[0];
    int image_size = mnist_data_loader.rows * mnist_data_loader.cols;
    int hide_layer_size = 128;
    int output_size = 10;
    auto ir_weights_type1 = ir::f32->Tile(image_size, hide_layer_size);
    auto ir_weights_type2 = ir::f32->Tile(hide_layer_size, output_size);
    // 加载权重
    auto ir_weight1 =
        ir_builder->Create<ir::io::LoadBinary>(ir_weights_type1, "tests/models/mnist/weight1.bin");
    auto ir_weight2 =
        ir_builder->Create<ir::io::LoadBinary>(ir_weights_type2, "tests/models/mnist/weight2.bin");
    // 第一层： 全连接 + relu
    auto ir_full1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_input, ir_weight1});
    auto ir_relu1 =
        ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_full1}, "relu", false);
    // 第二层： 全连接 + softmax
    auto ir_full2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_relu1, ir_weight2});
    auto ir_squeeze_view = ir_builder->Create<ir::SqueezeView>(ir_full2);  // 这里的维度需要改成1
    auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_squeeze_view});  // softmax
    ir_builder->Create<ir::Return>(ir_softmax);

    // 生成 JIT 引擎
    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    int count_right = 0;
    for (int i = 0; i < mnist_data_loader.image_count; i++) {
        auto preprocessed_image = PreprocessImageData(mnist_data_loader.GetImage(i), image_size);
        auto output_vec = model_fun(preprocessed_image.data());
        auto p_max = std::max_element(output_vec, output_vec + 10);
        auto predicted_digit = std::distance(output_vec, p_max);

        if (predicted_digit == mnist_data_loader.GetLabel(i)) {
            count_right++;
        }
    }

    ASSERT_GT(count_right, mnist_data_loader.image_count * 0.95);
}
