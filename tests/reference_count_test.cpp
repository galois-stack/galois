#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/fill.hpp"
#include "galois/transform/reference_count_visitor.hpp"
#include "galois_test.hpp"

TEST(GaloisTests, TestReferenceCount) {
    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] =
        ir_builder->CreateOperator(ir::OperatorType::Create({}, ir::void_), "test_operator");
    auto ir_alloc = ir_builder->Alloca(ir::i8->Tile(1123));  // 一个特殊的尺寸以便检查内存泄漏
    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    ir_printer->Dump(ir_operator);
    fmt::print("---after optimization--- \n");
    auto ir_reference_visitor = transform::ReferenceCountVisitor::Create();
    ir_operator->ApplyVisitor(ir_reference_visitor);
    ir_printer->Dump(ir_operator);
}

TEST(GaloisTests, TestReferenceCountReturn) {
    auto ir_builder = ir::Builder::Create();
    auto ir_type = ir::i8->Tile(1123);
    auto [ir_operator, scope] =
        ir_builder->CreateOperator(ir::OperatorType::Create({}, ir_type), "test_operator");
    auto ir_alloc = ir_builder->Alloca(ir_type);
    ir_builder->Return(ir_alloc);
    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    ir_printer->Dump(ir_operator);
    fmt::print("---after optimization--- \n");
    auto ir_reference_visitor = transform::ReferenceCountVisitor::Create();
    ir_operator->ApplyVisitor(ir_reference_visitor);
    ir_printer->Dump(ir_operator);
}

TEST(GaloisTests, TestReferenceCountCall) {
    auto ir_builder = ir::Builder::Create();
    auto ir_type = ir::i8->Tile(1123);

    auto [ir_operator, scope] = ir_builder->CreateOperator(ir::OperatorType::Create({}, ir::void_),
                                                           "test_referece_count_call");
    std::shared_ptr<ir::Operator> ir_callee;
    {
        auto [ir_operator_local, scope_local] =
            ir_builder->CreateOperator(ir::OperatorType::Create({}, ir_type), "test_operator");
        auto ir_alloc = ir_builder->Alloca(ir_type);
        ir_builder->Return(ir_alloc);
        ir_callee = ir_operator_local;
    }
    std::vector<std::shared_ptr<ir::Tensor>> ir_arguments;
    ir_builder->Call(ir_callee, {});
    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    ir_printer->Dump(ir_operator);
    fmt::print("---after optimization--- \n");
    auto ir_reference_visitor = transform::ReferenceCountVisitor::Create();
    ir_operator->ApplyVisitor(ir_reference_visitor);
    ir_printer->Dump(ir_operator);
}
