#include "galois/transform/free_after_last_use.hpp"

#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/op.hpp"
#include "galois_test.hpp"

TEST(GaloisTests, TestFreeAfterLastUseSimple) {
    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope] = ir_builder->CreateOperator(ir::OperatorType::Create({}, ir::void_),
                                                           "test_free_after_last_use_simple");
    auto ir_alloc0 = ir_builder->Alloca(ir::i8->Tile(128));
    auto ir_alloc1 = ir_builder->Alloca(ir::i8->Tile(128));

    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    fmt::print("---before optimization--- \n");
    ir_printer->Dump(ir_operator);

    galois::transform::FreeAfterLastUse::Create(ir_operator->block);

    fmt::print("---after optimization--- \n");
    ir_printer->Dump(ir_operator);
}

TEST(GaloisTests, TestFreeAfterLastUseReturn) {
    auto ir_builder = ir::Builder::Create();
    auto ir_type = ir::i8->Tile(64);
    auto [ir_operator, scope] = ir_builder->CreateOperator(ir::OperatorType::Create({}, ir_type),
                                                           "test_free_after_last_use_return");
    auto ir_alloc0 = ir_builder->Alloca(ir_type);
    auto ir_alloc1 = ir_builder->Alloca(ir_type);
    auto ir_alloc2 = ir_builder->Alloca(ir_type);

    ir_builder->Return(ir_alloc1);
    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    fmt::print("---before optimization--- \n");
    ir_printer->Dump(ir_operator);

    galois::transform::FreeAfterLastUse::Create(ir_operator->block);

    fmt::print("---after optimization--- \n");
    ir_printer->Dump(ir_operator);
}

TEST(GaloisTests, TestFreeAfterLastUseCall) {
    auto ir_builder = ir::Builder::Create();
    auto ir_type = ir::i8->Tile(256);

    auto [ir_operator, scope] = ir_builder->CreateOperator(ir::OperatorType::Create({}, ir::void_),
                                                           "test_free_after_last_use_call");
    std::shared_ptr<ir::Operator> ir_callee;
    {
        auto [ir_operator_local, scope_local] =
            ir_builder->CreateOperator(ir::OperatorType::Create({}, ir_type), "test_operator");
        auto ir_alloc = ir_builder->Alloca(ir_type);
        ir_builder->Return(ir_alloc);
        ir_callee = ir_operator_local;
    }
    std::vector<std::shared_ptr<ir::Tensor>> ir_arguments;
    auto call = ir_builder->Call(ir_callee, {});
    (void)call;
    scope = nullptr;

    auto ir_printer = ir::IRPrinter::Create();
    fmt::print("---before optimization--- \n");
    ir_printer->Dump(ir_operator);

    galois::transform::FreeAfterLastUse::Create(ir_operator->block);

    fmt::print("---after optimization--- \n");
    ir_printer->Dump(ir_operator);
}
