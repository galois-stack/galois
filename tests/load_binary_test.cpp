#include <filesystem>
#include <fstream>

#include "galois/ir/io.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestLoadBinary) {
    std::vector<float> float_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto ir_type = ir::f32->Tile(int64_t(float_vec.size()));

    auto temp_file_path = std::filesystem::temp_directory_path() / "test.bin";

    std::ofstream ofs(temp_file_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(float_vec.data()), float_vec.size() * sizeof(float));
    ofs.close();

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type = ir::OperatorType::Create({}, ir_type);
    auto [ir_operator, scope] = ir_builder->CreateOperator(ir_operator_type, "load_binary");
    auto ir_load_bin = ir_builder->Create<ir::io::LoadBinary>(ir_type, temp_file_path);
    ir_builder->Create<ir::Return>(ir_load_bin);

    auto jit_engine = jit::Engine::Create();
    auto load_binary_fun = jit_engine->EmitOperatorSymbol<float *(*)(void)>(ir_operator);

    auto result = load_binary_fun();
    for (int i = 0; i < float_vec.size(); i++) {
        GALOIS_ASSERT(result[i] == float_vec[i]);
    }

    free(result);
}
