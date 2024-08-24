
#include <chrono>

#include "galois/galois.hpp"
#include "galois/graph/graph.hpp"
#include "galois/lowering/lowering.hpp"
#include "galois/op/affine_convertor.hpp"
#include "galois/op/op.hpp"
#include "gtest/gtest.h"
#include "prajna/bindings/core.hpp"

using namespace galois;
using namespace galois::ir;
using namespace std;

namespace Eigen {
typedef Matrix<float, -1, -1, Eigen::RowMajor || Eigen::Aligned16> MatrixRXf;
}

TEST(GaloisTests, TestMatrixMultiply) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = f32(4, 1)(2, 1)(1, 1024)(128, 1);
    auto ir_ts_type_b = f32(1, 4)(1, 2)(1024, 1)(1, 128);
    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    auto ir_input_a = graph::Input::Create(ir_ts_type_a);
    auto ir_input_b = graph::Input::Create(ir_ts_type_b);
    auto ir_matrix_multiply_op = std::make_shared<op::MatrixMultiplyCreator>();
    auto ir_mat_mul = graph::ComputeNode::Create(ir_matrix_multiply_op, {ir_input_a, ir_input_b});
    auto ir_module = graph::ComputeGraph::BuildComputeGraph(ir_mat_mul, "tmp_module");

    auto ir_affine_convertor = graph::AffineConvertor::Create();
    auto ir_operator = ir_affine_convertor->EmitModule(ir_module);
    auto prajna_compiler = prajna::Compiler::Create();
    auto llvm_codegen = std::make_shared<codegen::cpu::LlvmCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorInstance(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->prajna_ir_builder->module);
    auto tmp_fun = reinterpret_cast<void (*)(float *, float *, float *)>(
        prajna_compiler->GetSymbolValue("::tmp_module"));

    Eigen::MatrixRXf eigen_matrix_f32_a = Eigen::MatrixRXf::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf eigen_matrix_f32_b = Eigen::MatrixRXf::Ones(shape_b[0], shape_b[1]);
    Eigen::MatrixRXf eigen_matrix_f32_c = Eigen::MatrixRXf::Random(shape_a[0], shape_b[1]);

    // eigen_matrix_f32_c.setZero();
    auto t0 = std::chrono::high_resolution_clock::now();
    tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data(), eigen_matrix_f32_c.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois gemm flops: {}gflops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));
}

TEST(GaloisTests, TestGemm) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = f32(4, 1)(2, 1)(1, 1024)(128, 1);
    auto ir_ts_type_b = f32(1, 4)(1, 2)(1024, 1)(1, 128);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    auto ir_input_a = graph::Input::Create(f32(shape_a));
    auto ir_input_b = graph::Input::Create(f32(shape_b));
    auto ir_pack_op_a = op::PackCreator::Create(ir_ts_type_a);
    auto ir_pack_a = graph::ComputeNode::Create(ir_pack_op_a, {ir_input_a});
    auto ir_pack_op_b = op::PackCreator::Create(ir_ts_type_b);
    auto ir_pack_b = graph::ComputeNode::Create(ir_pack_op_b, {ir_input_b});
    auto ir_matrix_multiply_op = std::make_shared<op::MatrixMultiplyCreator>();
    auto ir_mat_mul = graph::ComputeNode::Create(ir_matrix_multiply_op, {ir_pack_a, ir_pack_b});
    auto ir_unpack_op_c = std::make_shared<op::UnpackCreator>();
    auto ir_unpack_c = graph::ComputeNode::Create(ir_unpack_op_c, {ir_mat_mul});
    auto ir_module = graph::ComputeGraph::BuildComputeGraph(ir_unpack_c, "tmp_module");

    auto ir_affine_convertor = graph::AffineConvertor::Create();
    auto ir_operator = ir_affine_convertor->EmitModule(ir_module);
    auto prajna_compiler = prajna::Compiler::Create();
    auto llvm_codegen = std::make_shared<codegen::cpu::LlvmCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorInstance(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->prajna_ir_builder->module);
    auto tmp_fun = reinterpret_cast<void (*)(float *, float *, float *)>(
        prajna_compiler->GetSymbolValue("::tmp_module"));

    Eigen::MatrixRXf eigen_matrix_f32_a = Eigen::MatrixRXf::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf eigen_matrix_f32_b = Eigen::MatrixRXf::Ones(shape_b[0], shape_b[1]);
    auto shape_c = Cast<TensorType>(ir_module->type)->shape;
    Eigen::MatrixRXf eigen_matrix_f32_c = Eigen::MatrixRXf::Random(shape_c[0], shape_c[1]);
    Eigen::MatrixRXf eigen_matrix_f32_expect = Eigen::MatrixRXf::Random(shape_c[0], shape_c[1]);

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    eigen_matrix_f32_expect = (eigen_matrix_f32_a * eigen_matrix_f32_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen gemm flops: {}gflops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    // eigen_matrix_f32_c.setZero();
    auto t0 = std::chrono::high_resolution_clock::now();
    tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data(), eigen_matrix_f32_c.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois gemm flops: {}gflops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    int64_t error_count = 0;
    for (int64_t i = 0; i < shape_c[0]; ++i) {
        for (int64_t j = 0; j < shape_c[1]; ++j) {
            if ((std::abs(eigen_matrix_f32_c(i, j) - eigen_matrix_f32_expect(i, j))) /
                    std::max(std::abs(eigen_matrix_f32_expect(i, j)),
                             std::abs(eigen_matrix_f32_c(i, j))) >
                0.1f) {
                fmt::print("err pos: {},{}; {}, {}\n", i, j, eigen_matrix_f32_c(i, j),
                           eigen_matrix_f32_expect(i, j));
                ++error_count;
                if (error_count > 100) {
                    std::terminate();
                }
                std::cout << std::endl;
            }
        }
    }
}
