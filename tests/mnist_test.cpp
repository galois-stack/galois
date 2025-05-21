#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

// 读取MNIST图像和用户自定义图像的函数
class MnistDataLoader {
public:
    
    // 从MNIST测试集加载单张图像
    static std::vector<float> loadTestImage(int& label, int imageIndex = -1) {
        // MNIST数据集路径
        std::string image_path = "data/MNIST/raw/t10k-images-idx3-ubyte";
        std::string label_path = "data/MNIST/raw/t10k-labels-idx1-ubyte";
        
        // 打开图像文件
        std::ifstream image_file(image_path, std::ios::binary);
        if (!image_file) {
            std::cerr << "Error: Cannot open MNIST image file at " << image_path << std::endl;
            std::cerr << "Using random data instead" << std::endl;
            return generateRandomImage(label);
        }
        
        // 读取MNIST图像文件头
        uint32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;
        image_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        image_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        image_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        image_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        // 大端转小端
        magic_number = swapEndian(magic_number);
        num_images = swapEndian(num_images);
        rows = swapEndian(rows);
        cols = swapEndian(cols);
        
        // 如果未指定图像索引，则随机选择一个
        if (imageIndex < 0 || imageIndex >= static_cast<int>(num_images)) {
            srand(time(NULL));
            imageIndex = rand() % num_images;
        }
        
        // 加载标签
        label = loadLabel(label_path, imageIndex);
        
        // 读取选定图像的数据
        image_file.seekg(16 + imageIndex * rows * cols, std::ios::beg);  // 16字节头 + 图像数据偏移
        std::vector<uint8_t> image_data(rows * cols);
        image_file.read(reinterpret_cast<char*>(image_data.data()), image_data.size());
        
        // 预处理图像数据
        std::vector<float> processed_image = preprocessImageData(image_data);
        
        std::cout << "Loaded test image #" << imageIndex << ", label: " << label << std::endl;
        return processed_image;
    }

    static std::vector<float> loadCustomImage(const std::string& image_path) {
        std::cout << "Attempting to load custom image: " << image_path << std::endl;
        
        // 简单起见，假设输入已经是正确格式的784个浮点数的原始二进制文件
        // 在实际应用中，这里应该使用图像处理库（如OpenCV）来加载和处理图像
        std::ifstream file(image_path, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open image file: " << image_path << std::endl;
            return std::vector<float>();
        }
        
        std::vector<float> image_data(784);
        file.read(reinterpret_cast<char*>(image_data.data()), 784 * sizeof(float));
        
        // 应用标准化处理
        normalizeImage(image_data);
        
        std::cout << "Successfully loaded custom image" << std::endl;
        return image_data;
    }

private:
    // 大小端转换
    static uint32_t swapEndian(uint32_t val) {
        return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
               ((val & 0xFF0000) >> 8) | ((val & 0xFF000000) >> 24);
    }
    
    // 生成随机图像数据
    static std::vector<float> generateRandomImage(int& label) {
        std::vector<float> random_image(784);
        for (int i = 0; i < 784; i++) {
            random_image[i] = (static_cast<float>(rand() % 256) / 255.0f - 0.1307f) / 0.3081f;
        }
        label = -1;  // 未知标签
        return random_image;
    }
    
    // 加载标签
    static int loadLabel(const std::string& label_path, int imageIndex) {
        std::ifstream label_file(label_path, std::ios::binary);
        if (!label_file) {
            return -1; // 未知标签
        }
        
        // 跳过标签文件头
        uint32_t label_magic, label_count;
        label_file.read(reinterpret_cast<char*>(&label_magic), sizeof(label_magic));
        label_file.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
        
        // 大端转小端
        label_count = swapEndian(label_count);
        
        // 读取选定图像的标签
        label_file.seekg(8 + imageIndex, std::ios::beg);  // 8字节头 + 图像索引
        uint8_t label_value;
        label_file.read(reinterpret_cast<char*>(&label_value), 1);
        return static_cast<int>(label_value);
    }
    
    // 图像预处理
    static std::vector<float> preprocessImageData(const std::vector<uint8_t>& image_data) {
        std::vector<float> processed_image(image_data.size());
        const float mean = 0.1307f;
        const float std = 0.3081f;
        
        for (size_t i = 0; i < image_data.size(); i++) {
            processed_image[i] = (static_cast<float>(image_data[i]) / 255.0f - mean) / std;
        }
        
        return processed_image;
    }

    // 标准化图像数据
    static void normalizeImage(std::vector<float>& image_data) {
        const float mean = 0.1307f;
        const float std_dev = 0.3081f;
        for (auto& pixel : image_data) {
            pixel = (pixel / 255.0f - mean) / std_dev;
        }
    }
};

TEST(GaloisTests, TestMnist) {
    // 选择10张图像进行推理
    int image_num = 100;
    std::vector<std::vector<float>> processed_images;
    std::vector<int> labels_vec;
    for (int i = 0; i < image_num; i++) {
        int label;
        auto processed_image = MnistDataLoader::loadTestImage(label, i);
        processed_images.push_back(processed_image);
        labels_vec.push_back(label);
    }

    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir::f32->Tile(1, 28 * 28)}, ir::f32->Tile(10)),
        "mnist");  // 24应该改为相应的尺寸, 这里我直接把它展开了, 因为还没有reshape操作

    // 输入
    auto ir_input = ir_operator->inputs[0];

    // 加载权重
    auto ir_weights_type1 = ir::f32->Tile(784, 128);
    auto ir_weights_type2 = ir::f32->Tile(128, 10);
    auto ir_weight1 = ir_builder->Create<ir::io::LoadBinary>(
        ir_weights_type1, "weight1.bin");
    auto ir_weight2 = ir_builder->Create<ir::io::LoadBinary>(
        ir_weights_type2, "weight2.bin");


    // 第一层： 全连接 + relu
    auto ir_full1 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_input, ir_weight1});
    auto ir_relu1 = ir_builder->ExpressCreator<op::UnaryInstrinsicCreator>({ir_full1}, "relu6");

    // 第二层： 全连接 + softmax
    auto ir_full2 = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>({ir_relu1, ir_weight2});
    auto ir_squeeze_view = ir_builder->Create<ir::SqueezeView>(ir_full2);  // 这里的维度需要改成1
    auto ir_softmax = ir_builder->ExpressCreator<op::SoftmaxCreator>({ir_squeeze_view});  // softmax
    ir_builder->Create<ir::Return>(ir_softmax);

    // 生成 JIT 引擎
    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    int count_right = 0;
    for (int i = 0; i < processed_images.size(); i++) {
        // 执行模型
        auto output_vec = model_fun(processed_images[i].data());

    //     // 打印输出
        std::cout << "图像 #" << i << " (实际数字: " << labels_vec[i] << ") 的输出:" << std::endl;
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

        if (predicted_digit == labels_vec[i]) {
            count_right++;
        }
    }

    std::cout << "正确率: " << static_cast<float>(count_right) / processed_images.size() * 100 << "%"
              << std::endl;

    fflush(stdout);
}
