#include "galois/op/operator_fusion.hpp"

#include <boost/scope/scope_exit.hpp>
#include <cmath>

#include "galois/ir/ir_print_visitor.hpp"
#include "galois/transform/transform.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestOperatorFusion) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();

    auto ir_op_type = ir::OperatorType::Create(
        {ir_input_type, ir_input_type, ir_input_type},  
        ir_input_type                                   
    );

    auto [ir_operator, scope] = ir_builder->CreateOperator(ir_op_type, "ElemWiseCalc");
    auto ir_input_0 = ir_operator->inputs[0];
    auto ir_input_1 = ir_operator->inputs[1];
    auto ir_input_2 = ir_operator->inputs[2];
    auto ir_output = ir_builder->Alloca(ir_input_type);
    auto ir_add = ir_builder->ExpressCreator<op::AddCreator>({ir_input_0, ir_input_1});
    auto ir_sub = ir_builder->ExpressCreator<op::SubCreator>({ir_add, ir_input_2});
    ir_builder->Write(ir_sub, ir_output);
    ir_builder->Return(ir_output);

    auto ir_print_visitor = ir::IRPrinter::Create();
    std::cout << "Original IR:\n";
    std::cout << ir_print_visitor->Print(ir_operator) << "\n\n";

    auto op_hierarchy =
        galois::transform::ExtractHierarchicalFromBlock<ir::Operator>(ir_operator->block);
    galois::transform::PrintHierarchy(op_hierarchy);

    galois::transform::OperatorFusion(ir_operator);
    auto ir_print_visitor2 = ir::IRPrinter::Create();
    std::cout << "Fused IR:\n";
    std::cout << ir_print_visitor2->Print(ir_operator) << "\n\n";

    auto jit_engine = jit::Engine::Create();
    auto operatorfusion_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *, float *)>(ir_operator);

    std::array<float, length> input1 = {3.1415f, 2.7182f, 1.1415f, 0.7182f,
                                        4.1415f, 5.7182f, 6.1415f, 7.7182f};
    std::array<float, length> input2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, length> input3 = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> real_result(length);
    for (size_t i = 0; i < length; ++i) {
        real_result[i] = (input1[i] + input2[i]) - input3[i];
    }

    float *result = operatorfusion_fun(input1.data(), input2.data(), input3.data());
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < length; ++i) {
        ASSERT_NEAR(result[i], real_result[i], 1e-5);
    }
}