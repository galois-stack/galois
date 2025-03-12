#include <cassert>
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "galois_test.hpp"

TEST(GaloisTests, TestPackedMatrixMultiply_F32x4x1x4) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = ir::f32->Tile(4, 1)->Tile(2, 1)->Tile(1, 1024)->Tile(64, 1);
    auto ir_ts_type_b = ir::f32->Tile(1, 4)->Tile(1, 3)->Tile(1024, 1)->Tile(1, 64);
    ir_ts_type_a->value_type->enable_multi_thread = true;

    auto ir_builder = ir::Builder::Create();
    ir_builder->kernel_queue.push_back(op::MatrixMultiplyKernel4x1x4::Create());
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *, float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ir_mat_c_ptr = tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, galois flops: {}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    free(ir_mat_c_ptr);
}

TEST(GaloisTests, TestPackedMatrixMultiply_i8x16x1x16) {
    //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = ir::i8->Tile(16, 1)->Tile(1, 2048)->Tile(128, 1);
    auto ir_ts_type_b = ir::i8->Tile(1, 16)->Tile(2048, 1)->Tile(1, 128);
    ir_ts_type_a->value_type->enable_multi_thread = true;

    auto ir_builder = ir::Builder::Create();
    ir_builder->kernel_queue.push_back(op::MatrixMultiplyKernelI8_16x1x16::Create());
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_packed_matrix_multiply_op_creator,
                                                           {ir_ts_type_a, ir_ts_type_b});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<int8_t *(*)(int8_t *, int8_t *)>(
        prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXi8 eigen_matrix_i8_a = Eigen::MatrixRXi8::Zero(shape_a[0], shape_a[1]);
    Eigen::MatrixRXi8 eigen_matrix_i8_b = Eigen::MatrixRXi8::Zero(shape_b[0], shape_b[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ir_mat_c_ptr = tmp_fun(eigen_matrix_i8_a.data(), eigen_matrix_i8_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, galois flops: {}gops\n", (t1 - t0).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 - t0).count()));

    free(ir_mat_c_ptr);
}

// TEST(GaloisTests, TestMatrixMultiply1) {
//     //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
//     auto ir_ts_type_a = f32(4, 1)(2, 1)(1, 1024)(64, 1)(2, 1)(10, 10);
//     auto ir_ts_type_b = f32(1, 4)(1, 2)(1024, 1)(1, 64)(1, 2)(10, 10);  // 一个比较合理的设计
//     ir_ts_type_a->value_type->enable_multi_thread = true;
//     auto shape_a = ir_ts_type_a->NormalizeShape();
//     auto shape_b = ir_ts_type_b->NormalizeShape();

//     auto ir_input_a = graph::Input::Create(ir_ts_type_a);
//     auto ir_input_b = graph::Input::Create(ir_ts_type_b);
//     auto ir_matrix_multiply_op = op::MatrixMultiplyCreator::Create();
//     auto ir_mat_mul = graph::ComputeNode::Create(ir_matrix_multiply_op, {ir_input_a,
//     ir_input_b}); auto ir_module = graph::ComputeGraph::BuildComputeGraph(ir_mat_mul,
//     "tmp_module");

//     auto ir_affine_convertor = graph::AffineConvertor::Create();
//     auto ir_operator = ir_affine_convertor->EmitModule(ir_module);
//     transform::Each<ir::Grid>(ir_operator, [](std::shared_ptr<ir::Grid> ir_grid) {
//         if (ir_grid->enable_multi_thread) {
//             transform::AsyncInvokeByThreadPool(ir_grid);
//         }
//     });

//     auto prajna_compiler = CreateCompiler();
//     auto llvm_codegen =
//         std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
//     llvm_codegen->EmitOperatorFunction(ir_operator);
//     prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
//     auto tmp_fun = reinterpret_cast<void (*)(float *, float *, float *)>(
//         prajna_compiler->GetSymbolValue("::tmp_module"));

//     Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
//     Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);
//     Eigen::MatrixRXf32 get_f32_c = Eigen::MatrixRXf32::Random(shape_a[0], shape_b[1]);

//     // get_f32_c.setZero();
//     auto t0 = std::chrono::high_resolution_clock::now();
//     tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data(), get_f32_c.data());
//     auto t1 = std::chrono::high_resolution_clock::now();

//     fmt::print("cost time: {}ns, galois flops: {}gops\n", (t1 - t0).count(),
//                shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 -
//                t0).count()));
// }

// TEST(GaloisTests, TestMatrixMultiply256) {
//     //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
//     auto ir_ts_type_a = f32(8, 1)(1, 512)(128, 1)(4, 1);
//     auto ir_ts_type_b = f32(1, 8)(512, 1)(1, 128)(1, 4);
//     // ir_ts_type_a->enable_multi_thread = true;
//     auto shape_a = ir_ts_type_a->NormalizeShape();
//     auto shape_b = ir_ts_type_b->NormalizeShape();

//     auto ir_input_a = graph::Input::Create(ir_ts_type_a);
//     auto ir_input_b = graph::Input::Create(ir_ts_type_b);
//     auto ir_matrix_multiply_op = op::MatrixMultiplyCreator::Create();
//     auto ir_mat_mul = graph::ComputeNode::Create(ir_matrix_multiply_op, {ir_input_a,
//     ir_input_b}); auto ir_module = graph::ComputeGraph::BuildComputeGraph(ir_mat_mul,
//     "tmp_module");

//     auto ir_affine_convertor = graph::AffineConvertor::Create();
//     auto ir_operator = ir_affine_convertor->EmitModule(ir_module);
//     transform::Each<ir::Grid>(ir_operator, [](std::shared_ptr<ir::Grid> ir_grid) {
//         if (ir_grid->enable_multi_thread) {
//             transform::AsyncInvokeByThreadPool(ir_grid);
//         }
//     });

//     auto prajna_compiler = CreateCompiler();
//     auto llvm_codegen =
//         std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
//     llvm_codegen->EmitOperatorFunction(ir_operator);
//     prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
//     auto tmp_fun = reinterpret_cast<void (*)(float *, float *, float *)>(
//         prajna_compiler->GetSymbolValue("::tmp_module"));

//     Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Random(shape_a[0], shape_a[1]);
//     Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Random(shape_b[0], shape_b[1]);
//     Eigen::MatrixRXf32 get_f32_c = Eigen::MatrixRXf32::Random(shape_a[0], shape_b[1]);

//     // get_f32_c.setZero();
//     auto t0 = std::chrono::high_resolution_clock::now();
//     tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data(), get_f32_c.data());
//     auto t1 = std::chrono::high_resolution_clock::now();

//     fmt::print("cost time: {}ns, galois flops: {}gops\n", (t1 - t0).count(),
//                shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 -
//                t0).count()));
// }

// TEST(GaloisTests, TestGemm) {
//     //  一种快捷写法, 需要用TensorTypePointer包装后才支持这种写法
//     auto ir_ts_type_a = f32(8, 1)(1, 512)(64, 1);
//     auto ir_ts_type_b = f32(1, 8)(512, 1)(1, 64);
//     // ir_ts_type_a->enable_multi_thread = false;

//     auto shape_a = ir_ts_type_a->NormalizeShape();
//     auto shape_b = ir_ts_type_b->NormalizeShape();

//     auto ir_input_a = graph::Input::Create(f32(shape_a));
//     auto ir_input_b = graph::Input::Create(f32(shape_b));
//     auto ir_pack_op_a = op::PackCreator::Create(ir_ts_type_a);
//     auto ir_pack_a = graph::ComputeNode::Create(ir_pack_op_a, {ir_input_a});
//     auto ir_pack_op_b = op::PackCreator::Create(ir_ts_type_b);
//     auto ir_pack_b = graph::ComputeNode::Create(ir_pack_op_b, {ir_input_b});
//     auto ir_matrix_multiply_op = op::MatrixMultiplyCreator::Create();
//     auto ir_mat_mul = graph::ComputeNode::Create(ir_matrix_multiply_op, {ir_pack_a, ir_pack_b});
//     auto ir_unpack_op_c = op::UnpackCreator::Create();
//     auto ir_unpack_c = graph::ComputeNode::Create(ir_unpack_op_c, {ir_mat_mul});
//     auto ir_module = graph::ComputeGraph::BuildComputeGraph(ir_unpack_c, "tmp_module");

//     auto ir_affine_convertor = graph::AffineConvertor::Create();
//     auto ir_operator = ir_affine_convertor->EmitModule(ir_module);
//     transform::Each<ir::Grid>(ir_operator, [](std::shared_ptr<ir::Grid> ir_grid) {
//         if (ir_grid->enable_multi_thread) {
//             transform::AsyncInvokeByThreadPool(ir_grid);
//         }
//     });

//     auto prajna_compiler = CreateCompiler();
//     auto llvm_codegen =
//         std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
//     llvm_codegen->EmitOperatorFunction(ir_operator);
//     prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
//     auto tmp_fun = reinterpret_cast<void (*)(float *, float *, float *)>(
//         prajna_compiler->GetSymbolValue("::tmp_module"));

//     Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Random(shape_a[0], shape_a[1]);
//     Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Random(shape_b[0], shape_b[1]);
//     auto shape_c = Cast<TensorType>(ir_module->type)->shape;
//     Eigen::MatrixRXf32 get_f32_c = Eigen::MatrixRXf32::Random(shape_c[0], shape_c[1]);
//     Eigen::MatrixRXf32 eigen_matrix_f32_expect = Eigen::MatrixRXf32::Random(shape_c[0],
//     shape_c[1]);

//     Eigen::setNbThreads(1);
//     auto t0_eigen = std::chrono::high_resolution_clock::now();
//     eigen_matrix_f32_expect = (eigen_matrix_f32_a * eigen_matrix_f32_b).eval();
//     auto t1_eigen = std::chrono::high_resolution_clock::now();
//     fmt::print("cost time: {}ns, eigen flops: {}gops\n", (t1_eigen - t0_eigen).count(),
//                shape_a[0] * shape_a[1] * shape_b[1] * 2 /
//                    static_cast<double>((t1_eigen - t0_eigen).count()));

//     // get_f32_c.setZero();
//     auto t0 = std::chrono::high_resolution_clock::now();
//     tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data(), get_f32_c.data());
//     auto t1 = std::chrono::high_resolution_clock::now();

//     fmt::print("cost time: {}ns, galois flops: {}gops\n", (t1 - t0).count(),
//                shape_a[0] * shape_a[1] * shape_b[1] * 2 / static_cast<double>((t1 -
//                t0).count()));

//     int64_t error_count = 0;
//     for (int64_t i = 0; i < shape_c[0]; ++i) {
//         for (int64_t j = 0; j < shape_c[1]; ++j) {
//             if ((std::abs(get_f32_c(i, j) - eigen_matrix_f32_expect(i, j))) /
//                     std::max(std::abs(eigen_matrix_f32_expect(i, j)),
//                              std::abs(get_f32_c(i, j))) >
//                 0.1f) {
//                 fmt::print("err pos: {},{}; {}, {}\n", i, j, get_f32_c(i, j),
//                            eigen_matrix_f32_expect(i, j));
//                 ++error_count;
//                 if (error_count > 100) {
//                     std::terminate();
//                 }
//                 std::cout << std::endl;
//             }
//         }
//     }
// }

TEST(GaloisTests, TestGemm0) {
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

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_gemm_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *, float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_gemm_operator->fullname));

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    Eigen::MatrixRXf32 eigen_matrix_f32_a = Eigen::MatrixRXf32::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf32 eigen_matrix_f32_b = Eigen::MatrixRXf32::Ones(shape_b[0], shape_b[1]);
    auto shape_c = ir_ts_type_c->shape;

    auto t0_eigen = std::chrono::high_resolution_clock::now();
    Eigen::MatrixRXf32 eigen_matrix_f32_expect = (eigen_matrix_f32_a * eigen_matrix_f32_b).eval();
    auto t1_eigen = std::chrono::high_resolution_clock::now();
    fmt::print("cost time: {}ns, eigen flops: {}gops\n", (t1_eigen - t0_eigen).count(),
               shape_a[0] * shape_a[1] * shape_b[1] * 2 /
                   static_cast<double>((t1_eigen - t0_eigen).count()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto f32_c_ptr = tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("cost time: {}ns, galois flops: {}gops\n", (t1 - t0).count(),
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


TEST(GaloisTests, TestFill) {

    // 1.创建张量类型
    auto ir_ts_type = ir::f32->Tile(4, 1)->Tile(2, 1);
   
   // 2.创建IR构建器和Fill算子，填充值为5
   auto ir_builder = ir::Builder::Create();

   auto ir_fill_op_creator = op::FillCreator::Create(ir_ts_type,5);

    // 3.推导类型
    auto ir_ts_type_c = ir_fill_op_creator->InferType({ir_ts_type});

    // 4.创建操作类型和操作符  //张量类型
    auto ir_operator_type = ir::OperatorType::Create({ir_ts_type},ir_ts_type_c);
    auto [ir_operator, operator_scope] =
        ir_builder->CreateOperator(ir_operator_type, "fill_tensor");

    // 对张量进行填充
    ir_fill_op_creator->AffineExpress(ir_operator->inputs, ir_builder);
    // 释放operator_scope
    operator_scope = nullptr;  

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<void (*)(float *)>(
    prajna_compiler->GetSymbolValue("::fill_tensor"));

    auto input = new float[8];
    // 传入一个 float*，与函数匹配
    tmp_fun(input); 
    GALOIS_ASSERT(input[0] == 5.0);
    GALOIS_ASSERT(input[4] == 5.0);
    GALOIS_ASSERT(input[7] == 5.0);

    // 释放动态数组
    delete[] input;  
    tmp_fun = nullptr; 

}



