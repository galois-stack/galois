#include <cassert>

#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestPackedMatrixMultiply_F32x4x1x4) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = ir::f32->Tile(4, 1)->Tile(2, 1)->Tile(1, 1024)->Tile(64, 1);
    auto ir_ts_type_b = ir::f32->Tile(1, 4)->Tile(1, 3)->Tile(1024, 1)->Tile(1, 64);
    ir_ts_type_a->value_type->enable_multi_thread = true;

    auto ir_builder = ir::Builder::Create();
    ir_builder->matrix_multiply_kernel_queue.push_back(
        op::VectorizedMatrixMultiplyKernel::Create(128));
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ir_mat_c_ptr = mat_mul_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, galois flops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    free(ir_mat_c_ptr);
}

TEST(GaloisTests, TestPackedMatrixMultiply_F32x8x1x8) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = ir::f32->Tile(8, 1)->Tile(2, 1)->Tile(1, 1024)->Tile(64, 1);
    auto ir_ts_type_b = ir::f32->Tile(1, 8)->Tile(1, 1)->Tile(1024, 1)->Tile(1, 64);
    ir_ts_type_a->value_type->enable_multi_thread = true;

    auto ir_builder = ir::Builder::Create();
    ir_builder->matrix_multiply_kernel_queue.push_back(
        op::VectorizedMatrixMultiplyKernel::Create(256));
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ir_mat_c_ptr = mat_mul_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, galois flops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    free(ir_mat_c_ptr);
}

TEST(GaloisTests, TestPackedMatrixMultiply_i8x16x1x16) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = ir::i8->Tile(16, 1)->Tile(1, 2048)->Tile(128, 1);
    auto ir_ts_type_b = ir::i8->Tile(1, 16)->Tile(2048, 1)->Tile(1, 128);
    ir_ts_type_a->value_type->enable_multi_thread = true;

    auto ir_builder = ir::Builder::Create();
    ir_builder->matrix_multiply_kernel_queue.push_back(
        op::VectorizedMatrixMultiplyKernel::Create(128));
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun = jit_engine->EmitOperatorSymbol<int8_t *(*)(int8_t *, int8_t *)>(ir_operator);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXi8 eigen_matrix_i8_a = Eigen::MatrixRXi8::Zero(shape_a[0], shape_a[1]);
    Eigen::MatrixRXi8 eigen_matrix_i8_b = Eigen::MatrixRXi8::Zero(shape_b[0], shape_b[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ir_mat_c_ptr = mat_mul_fun(eigen_matrix_i8_a.data(), eigen_matrix_i8_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, galois flops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    free(ir_mat_c_ptr);
}

TEST(GaloisTests, TestGemm) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = f32->Tile(512, 512);
    auto ir_ts_type_b = f32->Tile(512, 512);

    auto ir_builder = ir::Builder::Create();
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_ts_type_c =
        ir_packed_matrix_multiply_op_creator->InferType({ir_ts_type_a, ir_ts_type_b});

    auto ir_operator_type = ir::OperatorType::Create({ir_ts_type_a, ir_ts_type_b}, ir_ts_type_c);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});
    auto gemm_optimizer = optimization::GemmOptimizer::Create();
    auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_gemm_operator);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);
    auto shape_c = ir_ts_type_c->shape;

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    Eigen::MatrixRXf32 eigen_matrix_f32_expect = (eigen_matrix_f32_a * eigen_matrix_f32_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen flops: {:.04f}gops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto f32_c_ptr = mat_mul_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois flops: {:.04f}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    auto get_f32_c = [=](int64_t i, int64_t j) -> float { return f32_c_ptr[i * shape_c[1] + j]; };

    int64_t error_count = 0;
    for (int64_t i = 0; i < shape_c[0]; ++i) {
        for (int64_t j = 0; j < shape_c[1]; ++j) {
            if ((std::abs(get_f32_c(i, j) - eigen_matrix_f32_expect(i, j))) /
                    std::max(std::abs(eigen_matrix_f32_expect(i, j)), std::abs(get_f32_c(i, j))) >
                0.1f) {
                fmt::print("err pos: {},{}; {}, {}\n", i, j, get_f32_c(i, j),
                           eigen_matrix_f32_expect(i, j));
                ++error_count;
                if (error_count > 100) {
                    std::terminate();
                }
                std::cout << std::endl;
            }
        }
    }

    free(static_cast<void *>(f32_c_ptr));
}
