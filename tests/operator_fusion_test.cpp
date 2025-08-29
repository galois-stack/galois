#include "galois/op/operator_fusion.hpp"
#include "galois/op/operator_fusion_no_opt.hpp"
#include "galois/transform/transform.hpp"
#include "galois/ir/ir_print_visitor.hpp"

#include <boost/scope/scope_exit.hpp>
#include <cmath>

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestOperatorFusion_v1) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionNoOptCreator>(
        {ir_input_type, ir_input_type, ir_input_type});
    auto ir_print_visitor = ir::IRPrinter::Create();
    auto output = ir_print_visitor->Print(ir_operator);
    std::cout << output << "\n\n";

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

TEST(GaloisTests, TestOperatorFusion_v2) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionNoOptCreator>(
        {ir_input_type, ir_input_type, ir_input_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    std::cout << "Original IR:\n";
    std::cout << ir_print_visitor->Print(ir_operator) << "\n\n";

    auto op_hierarchy = galois::transform::ExtractHierarchicalFromBlock<ir::Operator>(ir_operator->block);
    galois::transform::PrintHierarchy(op_hierarchy);

    auto ir_operator2 = galois::transform::OperatorFusionOpt(ir_operator);
    std::cout << "Fused IR:\n";
    std::cout << ir_print_visitor->Print(ir_operator2) << "\n\n";

    auto jit_engine = jit::Engine::Create();
    auto operatorfusion_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *, float *)>(ir_operator2);

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

TEST(GaloisTests, TestOperatorFusion_v3) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionNoOptCreator>(
        {ir_input_type, ir_input_type, ir_input_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    std::cout << "Original IR:\n";
    std::cout << ir_print_visitor->Print(ir_operator) << "\n\n";

    auto op_hierarchy = galois::transform::ExtractHierarchicalFromBlock<ir::Operator>(ir_operator->block);
    galois::transform::PrintHierarchy(op_hierarchy);

    galois::transform::ModifyOperators(ir_operator);
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
        real_result[i] = (input1[i] + input2[i]) - input3[i] ;
    }

    float *result = operatorfusion_fun(input1.data(), input2.data(), input3.data());
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < length; ++i) {
        ASSERT_NEAR(result[i], real_result[i], 1e-5);
    }
}