#include <iostream>
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"
#include "galois/transform/operation_count_visitor.hpp"

TEST(GaloisTests, TestOperationCount) {
        auto ir_mat_type_a = ir::f32->Tile(4, 8);
        auto ir_mat_type_b = ir::f32->Tile(8, 4);

        auto ir_builder = ir::Builder::Create();
        auto ir_operator = ir_builder->template CreateOperatorByCreator<op::MatrixMultiplyCreator>(
            {ir_mat_type_a, ir_mat_type_b});
        auto gemm_optimizer = optimization::GemmOptimizer::Create();
        auto ir_gemm_operator = gemm_optimizer->Optimize(ir_operator);

        auto jit_engine = jit::Engine::Create();
        auto mat_mul_fun =
            jit_engine->EmitOperatorSymbol<void *(*)(void *, void *)>(ir_gemm_operator);

        auto normalize_m = ir_mat_type_a->NormalizeShape()[0];
        auto normalize_k = ir_mat_type_a->NormalizeShape()[1];
        auto normalize_n = ir_mat_type_b->NormalizeShape()[1];
        auto items = normalize_m * normalize_k * normalize_n;

        auto operationCountVisitor = galois::transform::OperationCountVisitor::Create();
        ir_gemm_operator->ApplyVisitor(operationCountVisitor);
        auto operationCount = operationCountVisitor->GetOperationCount();

        std::cout << "operationCount: " << operationCount << std::endl;
        std::cout << "this->items * 2: " << items * 2 << std::endl;
        ASSERT_EQ(items * 2, operationCount);

    
}
