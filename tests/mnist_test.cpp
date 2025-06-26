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

TEST(GaloisTests, TestMnistLinear) {
    MnistDataLoader mnist_data_loader("tests/data/mnist/t10k-images-idx3-ubyte.bin",
                                      "tests/data/mnist/t10k-labels-idx1-ubyte.bin");

    auto ir_builder = ir::Builder::Create();
    int64_t batch_size = 1;
    auto [ir_operator, scope] =
        ir_builder->CreateOperator(ir::OperatorType::Create({ir::f32->Tile(batch_size, 28 * 28)},
                                                            ir::f32->Tile(batch_size, 10)),
                                   "mnist");
    auto ir_input = ir_operator->inputs[0];
    int image_size = mnist_data_loader.rows * mnist_data_loader.cols;
    int hide_layer_size = 128;
    int output_size = 10;
    auto ir_weights_type1 = ir::f32->Tile(image_size, hide_layer_size);
    auto ir_weights_type2 = ir::f32->Tile(hide_layer_size, output_size);

    auto ir_weight1 =
        ir_builder->Create<ir::io::LoadBinary>(ir_weights_type1, "tests/models/mnist/weight1.bin");
    auto ir_weight2 =
        ir_builder->Create<ir::io::LoadBinary>(ir_weights_type2, "tests/models/mnist/weight2.bin");
    // First layer: fully connected + relu
    auto ir_full1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_input, ir_weight1});
    auto ir_relu1 =
        ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_full1}, "relu", false);
    // Second layer: fully connected + softmax
    auto ir_full2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_relu1, ir_weight2});
    auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_full2});  // softmax
    ir_builder->Create<ir::Return>(ir_softmax);

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

TEST(GaloisTests, TestMnistCNN) {
    MnistDataLoader mnist_data_loader("tests/data/mnist/t10k-images-idx3-ubyte.bin",
                                      "tests/data/mnist/t10k-labels-idx1-ubyte.bin");
    
    auto ir_builder = ir::Builder::Create();
    // input [1, 28, 28] (channels, height, width)
    auto [ir_operator, scope] = ir_builder->CreateOperator(
            ir::OperatorType::Create({ir::f32->Tile(1, 28, 28)}, ir::f32->Tile(10)),
            "mnist_cnn");

    // [out_channels, in_channels, kernel_h, kernel_w]
    auto ir_conv1_type = ir::f32->Tile(32, 1, 5, 5);   // 5x5 convolution kernel
    auto ir_conv2_type = ir::f32->Tile(64, 32, 5, 5);  // 5x5 convolution kernel
    auto ir_fc1_type = ir::f32->Tile(4*4*64, 128);     // 4*4*64 x 128 = 1024 x 128
    auto ir_fc2_type = ir::f32->Tile(128, 10);
    auto ir_input = ir_operator->inputs[0];
    int image_size = mnist_data_loader.rows * mnist_data_loader.cols;

    auto ir_conv1_weights = ir_builder->Create<ir::io::LoadBinary>(
        ir_conv1_type, "tests/models/mnist/conv1_weight.bin");
    auto ir_conv2_weights = ir_builder->Create<ir::io::LoadBinary>(
        ir_conv2_type, "tests/models/mnist/conv2_weight.bin");
    auto ir_fc1_weights = ir_builder->Create<ir::io::LoadBinary>(
        ir_fc1_type, "tests/models/mnist/fc1_weight.bin");
    auto ir_fc2_weights = ir_builder->Create<ir::io::LoadBinary>(
        ir_fc2_type, "tests/models/mnist/fc2_weight.bin");
 
    // Conv1 + ReLU: [28,28,1] -> [12,12,32] (kernel=5, stride=2, (28-5)/2+1=12)
    auto ir_conv1 = ir_builder->ExpressCreator<op::Convolution3DCreator>({ir_input, ir_conv1_weights}, 2, 2, 0, 0);
    auto ir_relu1 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_conv1}, "relu", false);
        
    // Conv2 + ReLU: [12,12,32] -> [4,4,64] (kernel=5, stride=2, (12-5)/2+1=4)
    auto ir_conv2 = ir_builder->ExpressCreator<op::Convolution3DCreator>({ir_relu1, ir_conv2_weights}, 2, 2, 0, 0);
    auto ir_relu2 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_conv2}, "relu", false);
        
    // Flatten: [64, 4, 4] -> [1, 1024] 
    auto ir_flatten = ir_builder->Create<ir::view::BitCast>(ir_relu2, ir::f32->Tile(1, 4*4*64));
        
    // FC1 + ReLU
    auto ir_fc1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_flatten, ir_fc1_weights});
    auto ir_relu3 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_fc1}, "relu", false);
        
    // FC2 + Softmax
    auto ir_fc2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_relu3, ir_fc2_weights});
    auto ir_squeeze = ir_builder->Create<ir::view::Squeeze>(ir_fc2);
    auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_squeeze});
        
    ir_builder->Create<ir::Return>(ir_softmax);
        
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
    
    fflush(stdout);
}