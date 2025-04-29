
#include <iostream>

#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestIRPrint) {
    auto ir_mat_type_a = ir::f32->Tile(4, 8);
    auto ir_mat_type_b = ir::f32->Tile(8, 4);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::MatrixMultiplyCreator>(
        {ir_mat_type_a, ir_mat_type_b});

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto mmul_output = ir_print_visitor->Print(ir_operator);
    std::cout << mmul_output << "\n";

}
