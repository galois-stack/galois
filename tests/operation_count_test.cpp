
#include "galois/ir/operation_count_visitor.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestOperationCount) {
    auto ir_mat_type_a = ir::f32->Tile(4, 8)->Tile(3, 7);
    auto ir_mat_type_b = ir::f32->Tile(8, 4)->Tile(7, 5);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::MatrixMultiplyCreator>(
        {ir_mat_type_a, ir_mat_type_b});

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun = jit_engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_operator);

    auto normalize_m = ir_mat_type_a->NormalizeShape()[0];
    auto normalize_k = ir_mat_type_a->NormalizeShape()[1];
    auto normalize_n = ir_mat_type_b->NormalizeShape()[1];
    auto items = normalize_m * normalize_k * normalize_n * 2;

    auto operation_count_visitor = ir::OperationCounter::Create();
    auto operation_count = operation_count_visitor->CountOperation(ir_operator);
    ASSERT_EQ(items, operation_count);
}
