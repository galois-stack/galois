#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/fill.hpp"
#include "galois/transform/free_after_last_use.hpp"
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

TEST(GaloisTests, TestReferenceCountBlockLastUse) {
    auto ir_builder = ir::Builder::Create();
    auto ir_type = ir::i8->Tile(1123);

    auto [ir_operator, scope] = ir_builder->CreateOperator(ir::OperatorType::Create({}, ir::void_),
                                                           "test_referece_count_call");
    auto ir_alloc = ir_builder->Alloca(ir_type);
    auto ir_alloc2 = ir_builder->Alloca(ir_type);
    {
        auto [ir_block, scope_block] = ir_builder->CreateBlock();
        auto ir_view = ir_builder->Create<ir::FlattenView>(
            ir_alloc);  // ir_view会引用ir_alloc, 我们目前的版本没考虑ir_view, 因为其会增加一次引用,
                        // 也会减少一次引用, 所以我们在block末尾free的时候不需要考虑view的引用关系
                        // 现在我们要free after last use, 那应该要考虑其引用关系,
        ir_builder->ExpressCreator<op::CopyCreator>({ir_view, ir_alloc2});  // 这是深拷贝,
        // ir_alloc在后面并未被使用, Free(ir_alloc)应该在这里插入,及时释放内存.
        // 这是我们需要改进的地方. 目前Free是在上层Block的末尾插入的.
        // 我们FreeTensor的时候写一个 “往后遍历的visitor”检测到最后一次使用ir_alloc时候插入
        // 我们先做上面这个简单版本的
        auto ir_alloc3 = ir_builder->Alloca(ir_type);
    }

    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    ir_printer->Dump(ir_operator);
    fmt::print("---after optimization--- \n");
    auto ir_reference_visitor = transform::ReferenceCountVisitor::Create();
    ir_operator->ApplyVisitor(ir_reference_visitor);
    ir_printer->Dump(ir_operator);
}
