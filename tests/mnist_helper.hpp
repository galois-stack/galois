// 读取MNIST图像和用户自定义图像的函数
#pragma once

#include <boost/predef/other/endian.h>

#include "galois_test.hpp"

inline uint32_t BigEndianToHostEndian(uint32_t val) {
#ifdef BOOST_ENDIAN_LITTLE_BYTE
    return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | ((val & 0xFF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
#else
    return val;
#endif
}

class MnistDataLoader {
   public:
    MnistDataLoader(std::string image_path, std::string label_path) {
        this->LoadImages(image_path);
        this->LoadLabels(label_path);
    }

    void LoadImages(std::string image_path) {
        std::ifstream image_file(image_path, std::ios::binary);
        GALOIS_ASSERT(image_file.good());

        uint32_t magic_number;
        image_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        image_file.read(reinterpret_cast<char*>(&image_count), sizeof(image_count));
        image_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        image_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        magic_number = BigEndianToHostEndian(magic_number);
        GALOIS_ASSERT(magic_number == 2051);
        image_count = BigEndianToHostEndian(image_count);
        rows = BigEndianToHostEndian(rows);
        cols = BigEndianToHostEndian(cols);

        image_data.resize(image_count * rows * cols);
        image_file.read(reinterpret_cast<char*>(image_data.data()), image_data.size());
        image_file.close();
    }

    void LoadLabels(const std::string& label_path) {
        std::ifstream label_file(label_path, std::ios::binary);
        GALOIS_ASSERT(label_file.good());

        uint32_t label_magic, label_count;
        label_file.read(reinterpret_cast<char*>(&label_magic), sizeof(label_magic));
        label_file.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
        label_count = BigEndianToHostEndian(label_count);
        GALOIS_ASSERT(label_count == this->image_count);
        labels.resize(label_count);
        label_file.read(reinterpret_cast<char*>(labels.data()), label_count);
    }

    uint8_t* GetImage(int index) {
        GALOIS_ASSERT(index >= 0 && index < image_count);
        return image_data.data() + index * rows * cols;
    }

    uint8_t GetLabel(int index) {
        GALOIS_ASSERT(index >= 0 && index < image_count);
        return labels[index];
    }

   public:
    uint32_t image_count;
    uint32_t rows;
    uint32_t cols;

   private:
    std::vector<uint8_t> image_data;
    std::vector<uint8_t> labels;
};
