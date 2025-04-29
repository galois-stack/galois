#include <iostream>

#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/op/softmax.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestIRPrintMM) {
    auto ir_mat_type_a = ir::f32->Tile(4, 8);
    auto ir_mat_type_b = ir::f32->Tile(8, 4);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::MatrixMultiplyCreator>(
        {ir_mat_type_a, ir_mat_type_b});

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto mmul_output = ir_print_visitor->Print(ir_operator);
    std::cout << mmul_output << "\n";
}

TEST(GaloisTests, TestIRPrintFill) {
    auto ir_input_type = ir::f32->Tile(8, 16);
    auto ir_value_type = ir::f32;

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::FillCreator>(
        {ir_input_type, ir_value_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n\n";
}

TEST(GaloisTests, TestIRPrintFull) {
    auto ir_input_type = ir::i32->Tile(8, 64);
    auto ir_value_type = ir::i32;

    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::FullCreator>({ir_value_type}, ir_input_type);

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n\n";
}

TEST(GaloisTests, TestIRPrintPack) {
    auto ir_input_type = ir::f32->Tile(4, 8);             // f32[4, 8]
    auto ir_pack_type = ir::f32->Tile(4, 4)->Tile(1, 2);  // f32[1, 2, 4, 4]

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::PackCreator>(
        {ir_input_type}, ir_pack_type);

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n\n";
}

TEST(GaloisTests, TestIRPrintPadding) {
    auto ir_input_type = ir::f32->Tile(32, 32);
    Eigen::VectorXi64 shape(2);
    shape[0] = 32;
    shape[1] = 32;

    auto ir_builder = ir::Builder::Create();
    auto ir_padding_creator = op::PaddingCreator::Create(shape);
    auto ir_operator =
        ir_builder->template CreateOperatorByCreator<op::PaddingCreator>({ir_input_type}, shape);

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n";
}

TEST(GaloisTests, TestIRPrintSum) {
    auto ir_input_type = ir::f32->Tile(4);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->template CreateOperatorByCreator<op::SumCreator>({ir_input_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n";
}

TEST(GaloisTests, TestIRPrintSoftmax) {
    auto ir_vec_type = ir::f32->Tile(4);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->template CreateOperatorByCreator<op::SoftmaxCreator>({ir_vec_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n";
}

TEST(GaloisTests, TestIRPrintUnaryIntrinsic) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->template CreateOperatorByCreator<op::UnaryInstrinsicCreator>(
        {ir_input_type}, "sin");

    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n";
}
