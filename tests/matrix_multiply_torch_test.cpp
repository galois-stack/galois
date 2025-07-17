// matrix_multiply.cpp
#include "boost/scope/scope_exit.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestMatrixMultiply_Compare) {
    auto native_cpu_info = optimization::NativeCpuInfo::Create();
    auto mat_mul_tile_policy = optimization::GemmTilePolicy::Create();
    auto [ir_mat_type_a, ir_mat_type_b, mat_mul_kernel] =
        mat_mul_tile_policy->Tile(ir::f32, native_cpu_info);
    ir_mat_type_a = ir_mat_type_a->value_type;
    ir_mat_type_b = ir_mat_type_b->value_type;

    fmt::print("a: {}, b: {}\n", ir_mat_type_a->name, ir_mat_type_b->name);

    auto ir_builder = ir::Builder::Create();
    ir_builder->matrix_multiply_kernel_queue.push_back(mat_mul_kernel);
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_mat_type_c =
        ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});
    auto [ir_operator, scope_operator] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b, ir_mat_type_c}, ir::void_),
        "matrix_multiply");
    ir_packed_matrix_multiply_op_creator->ExpressInline(
        ir_operator->inputs[0], ir_operator->inputs[1], ir_operator->inputs[2], ir_builder);
    scope_operator = nullptr;

    transform::Repeat(ir_operator, 1);

    auto ir_operator_counter = ir::OperationCounter::Create();
    auto operations = ir_operator_counter->CountOperation(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<void (*)(void *, void *, void *)>(ir_operator);

    auto normalize_m = ir_mat_type_a->NormalizeShape()[0];
    auto normalize_k = ir_mat_type_a->NormalizeShape()[1];
    auto normalize_n = ir_mat_type_b->NormalizeShape()[1];

    auto sp_aligned256_mem_a = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_k * ir::f32->bytes),
        [](void *p) { free(p); });
    auto sp_aligned256_mem_b = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_k * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });

    auto sp_aligned256_mem_c = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });
    auto sp_aligned256_mem_c_torch = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });

    float* a_data = static_cast<float*>(sp_aligned256_mem_a.get());
    float* b_data = static_cast<float*>(sp_aligned256_mem_b.get());
    float* c_data = static_cast<float*>(sp_aligned256_mem_c.get());
    float* c_data_torch = static_cast<float*>(sp_aligned256_mem_c_torch.get());

    // for (int i = 0; i < normalize_m * normalize_k; ++i) {
    //     a_data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    // }
    // for (int i = 0; i < normalize_k * normalize_n; ++i) {
    //     b_data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    // }
    for (int i = 0; i < normalize_m * normalize_k; ++i) {
        a_data[i] = static_cast<float>(rand() % 11);
    }
    for (int i = 0; i < normalize_k * normalize_n; ++i) {
        b_data[i] = static_cast<float>(rand() % 11);
    }
    std::fill(c_data, c_data + normalize_m * normalize_n, 0.0f);
    std::fill(c_data_torch, c_data_torch + normalize_m * normalize_n, 0.0f);

    auto t0 = std::chrono::steady_clock::now();
    mat_mul_fun(sp_aligned256_mem_a.get(), sp_aligned256_mem_b.get(), sp_aligned256_mem_c.get());
    auto t1 = std::chrono::steady_clock::now();

    fmt::print("micro kernel: {:.04f}gops\n", operations / static_cast<double>((t1 - t0).count()));

    calculate_error(sp_aligned256_mem_a.get(), sp_aligned256_mem_b.get(), sp_aligned256_mem_c_torch.get(), normalize_m, normalize_k, normalize_n);

    int length = normalize_m * normalize_n;
    int error_num = 0;
    for (int i = 0; i < length; i++) {
        if(error_num >= 10){
            GALOIS_ASSERT(error_num==0);
        }
        float result = c_data[i];
        float result_torch = c_data_torch[i];
        fmt::print("Index:{:2d} Galois:{:.6f} Pytorch:{:.6f} \n", i, result, result_torch);
        if (std::abs(result - result_torch) > 1e-5f){
            error_num++;
        }
    }
}