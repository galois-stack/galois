#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"
#include <cmath> // 添加cmath头文件，支持isnan和isinf

void fill_fun(float* data, float* value) {
    for (int i = 0; i < 28 * 28; i++) {
        data[i] = *value;
    }
}

// 读取MNIST图像文件的函数
std::vector<uint8_t> read_mnist_images(const std::string& filename, int& num_images, int& rows, int& cols) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number); // 大端转小端
    
    if (magic_number != 0x00000803) {
        std::cerr << "Invalid MNIST image file: " << filename << std::endl;
        exit(1);
    }
    
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = __builtin_bswap32(num_images); // 大端转小端
    
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = __builtin_bswap32(rows); // 大端转小端
    
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cols = __builtin_bswap32(cols); // 大端转小端
    
    // 读取图像数据
    std::vector<uint8_t> images(num_images * rows * cols);
    file.read(reinterpret_cast<char*>(images.data()), images.size());
    
    return images;
}

// 读取MNIST标签文件的函数
std::vector<uint8_t> read_mnist_labels(const std::string& filename, int& num_labels) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    // 读取MNIST标签文件头
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number); // 大端转小端
    
    if (magic_number != 0x00000801) {
        std::cerr << "Invalid MNIST label file: " << filename << std::endl;
        exit(1);
    }
    
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels); // 大端转小端
    
    // 读取标签数据
    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), labels.size());
    
    return labels;
}

// 对图像进行处理，与PyTorch训练时的转换一致
std::vector<float> preprocess_image(const std::vector<uint8_t>& image, int index, int rows, int cols) {
    std::vector<float> processed(rows * cols);
    const float mean = 0.1307f;
    const float std = 0.3081f;
    
    for (int i = 0; i < rows * cols; ++i) {
        // 转换为[0,1]并进行标准化
        processed[i] = (static_cast<float>(image[index * rows * cols + i]) / 255.0f - mean) / std;
    }
    
    return processed;
}

TEST(GaloisTests, TestMnist) {
    std::string image_path = "../../data/MNIST/raw/t10k-images-idx3-ubyte";
    std::string label_path = "../../data/MNIST/raw/t10k-labels-idx1-ubyte";

    int num_images = 0, rows = 0, cols = 0;
    int num_labels = 0;

    std::vector<uint8_t> images = read_mnist_images(image_path, num_images, rows, cols);
    std::vector<uint8_t> labels = read_mnist_labels(label_path, num_labels);

    if (num_images != num_labels) {
        std::cerr << "Error: number of images and labels don't match!" << std::endl;
    }

    std::cout << "Loaded " << num_images << " test images of size " << rows << "x" << cols << std::endl;

    // 选择10张图像进行推理
    std::vector<std::vector<float>> processed_images;
    std::vector<int> labels_vec;
    for (int i = 0; i < 10; i++) {
        int image_index = i;
        std::vector<float> processed_image = preprocess_image(images, image_index, rows, cols);
        processed_images.push_back(processed_image);
        labels_vec.push_back(static_cast<int>(labels[image_index]));
    }
    

    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir::f32->Tile(28 * 28, 1)}, ir::f32->Tile(10)),
        "mnist");  // 24应该改为相应的尺寸, 这里我直接把它展开了, 因为还没有reshape操作
    
    // 输入
    auto ir_input = ir_operator->inputs[0];

    // 加载权重
    auto ir_weights_type1 = ir::f32->Tile(128, 784);
    auto ir_weights_type2 = ir::f32->Tile(10, 128);
    auto ir_weight1 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type1, "weight1.bin");
    auto ir_weight2 = ir_builder->Create<ir::io::LoadBinary>(ir_weights_type2, "weight2.bin");

    // 检查权重文件是否存在
    std::cout << "----------模型信息----------" << std::endl;
    {
        std::ifstream f1("weight1.bin", std::ios::binary);
        std::ifstream f2("weight2.bin", std::ios::binary);
        std::cout << "weight1.bin 存在: " << (f1.good() ? "是" : "否") << std::endl;
        std::cout << "weight2.bin 存在: " << (f2.good() ? "是" : "否") << std::endl;
        
        if (f1.good()) {
            f1.seekg(0, std::ios::end);
            size_t size1 = f1.tellg();
            std::cout << "weight1.bin 大小: " << size1 << " 字节" << std::endl;
            
            // 检查文件内容
            if (size1 > 0) {
                f1.seekg(0, std::ios::beg);
                std::vector<float> weights1(std::min(size_t(20), size1 / sizeof(float)));
                f1.read(reinterpret_cast<char*>(weights1.data()), weights1.size() * sizeof(float));
                
                std::cout << "weight1.bin 前几个值: ";
                for (size_t j = 0; j < weights1.size(); ++j) {
                    std::cout << weights1[j] << " ";
                }
                std::cout << std::endl;
                
                // 检查是否有不正常的值
                bool has_nan = false;
                bool has_inf = false;
                for (auto w : weights1) {
                    if (std::isnan(w)) has_nan = true;
                    if (std::isinf(w)) has_inf = true;
                }
                std::cout << "weight1.bin 包含NaN: " << (has_nan ? "是" : "否") << std::endl;
                std::cout << "weight1.bin 包含无穷大: " << (has_inf ? "是" : "否") << std::endl;
            }
        }
        
        if (f2.good()) {
            f2.seekg(0, std::ios::end);
            size_t size2 = f2.tellg();
            std::cout << "weight2.bin 大小: " << size2 << " 字节" << std::endl;
            
            // 检查文件内容
            if (size2 > 0) {
                f2.seekg(0, std::ios::beg);
                std::vector<float> weights2(std::min(size_t(20), size2 / sizeof(float)));
                f2.read(reinterpret_cast<char*>(weights2.data()), weights2.size() * sizeof(float));
                
                std::cout << "weight2.bin 前几个值: ";
                for (size_t j = 0; j < weights2.size(); ++j) {
                    std::cout << weights2[j] << " ";
                }
                std::cout << std::endl;
                
                // 检查是否有不正常的值
                bool has_nan = false;
                bool has_inf = false;
                for (auto w : weights2) {
                    if (std::isnan(w)) has_nan = true;
                    if (std::isinf(w)) has_inf = true;
                }
                std::cout << "weight2.bin 包含NaN: " << (has_nan ? "是" : "否") << std::endl;
                std::cout << "weight2.bin 包含无穷大: " << (has_inf ? "是" : "否") << std::endl;
            }
        }
    }
    std::cout << "---------------------------" << std::endl;

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
    
    for (int i = 0; i < 10; i++) {
        // 执行模型
        float output_vec[10];
        model_fun(processed_images[i].data(), output_vec);

        // 打印输出
        std::cout << "图像 #" << i << " (实际数字: " << labels_vec[i] << ") 的输出:" << std::endl;
        std::cout << "原始输出: ";
        for (int j = 0; j < 10; j++) {
            std::cout << output_vec[j] << " ";
        }
        std::cout << std::endl;
        
        // 找出概率最高的数字
        int predicted_digit = 0;
        float max_prob = output_vec[0];
        for (int j = 1; j < 10; j++) {
            if (output_vec[j] > max_prob) {
                max_prob = output_vec[j];
                predicted_digit = j;
            }
        }
        
        std::cout << "预测数字: " << predicted_digit << std::endl;
        std::cout << "实际数字: " << labels_vec[i] << std::endl;
        std::cout << (predicted_digit == labels_vec[i] ? "√ 正确" : "× 错误") << std::endl;
        std::cout << "----------------------------" << std::endl;
    }
    fflush(stdout);
}
