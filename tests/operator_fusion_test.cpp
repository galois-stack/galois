#include "galois/op/operator_fusion.hpp"
#include "galois/op/operator_fusion_no_opt.hpp"
#include "galois/transform/transform.hpp"

#include <boost/scope/scope_exit.hpp>
#include <cmath>

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestOperatorFusion_v1) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    std::vector<std::string> operations = {"add", "sub"};
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionCreator>(
        {ir_input_type, ir_input_type, ir_input_type}, operations);
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

TEST(GaloisTests, TestOperatorFusion_v3) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionNoOptCreator>(
        {ir_input_type, ir_input_type, ir_input_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    std::cout << "Original IR:\n";
    std::cout << ir_print_visitor->Print(ir_operator) << "\n\n";

    auto ir_graph = galois::framework::BuildComputingGraph::Create();
    ir_graph->Traverse(ir_operator);
    // ir_graph->PrintAllNodes();

    // auto all_operators = galois::transform::ExtractAllFromBlock<ir::Operator>(ir_operator->block);
    // std::cout << "Total operators extracted: " << all_operators.size() << std::endl;
    // for (size_t i = 0; i < all_operators.size(); ++i) {
    //     const auto& op = all_operators[i];
    //     if (!op) continue;  // 跳过无效指针
    //     std::cout << "Operator " << i << ":" << std::endl;
    //     std::cout << "  Name: " << op->name << std::endl;
    //     std::cout << "  Fullname: " << op->fullname << std::endl;
    //     std::cout << "  Input count: " << op->inputs.size() << std::endl;
    //     std::cout << "-------------------------" << std::endl;
    // }

    // auto all_calls = galois::transform::ExtractAllFromBlock<ir::Call>(ir_operator->block);
    // std::cout << "Total calls extracted: " << all_calls.size() << std::endl;
    // for (size_t i = 0; i < all_calls.size(); ++i) {
    //     const auto& call = all_calls[i];
    //     if (!call) {
    //         std::cout << "Call " << i << ": [nullptr]" << std::endl;
    //         continue;
    //     }
    //     std::cout << "Call " << i << ":\n" << ir_print_visitor->Print(call) << std::endl;
    // }

    auto op_hierarchy = galois::transform::ExtractHierarchicalFromBlock<ir::Operator>(ir_operator->block);
    galois::transform::PrintHierarchy(op_hierarchy);

    auto ir_operator2 = ir_graph->OperatorFusionOpt(ir_operator);
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

TEST(GaloisTests, TestOperatorFusion_v4) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionNoOptCreator>(
        {ir_input_type, ir_input_type, ir_input_type});

    auto ir_print_visitor = ir::IRPrinter::Create();
    std::cout << "Original IR:\n";
    std::cout << ir_print_visitor->Print(ir_operator) << "\n\n";

    auto ir_graph = galois::framework::BuildComputingGraph::Create();
    ir_graph->Traverse(ir_operator);
    // ir_graph->PrintAllNodes();

    auto op_hierarchy = galois::transform::ExtractHierarchicalFromBlock<ir::Operator>(ir_operator->block);
    galois::transform::PrintHierarchy(op_hierarchy);

    galois::transform::ModifyOperators(ir_operator);
    auto ir_print_visitor2 = ir::IRPrinter::Create();
    std::cout << "Original IR:\n";
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
        real_result[i] = (input1[i] + input2[i]);
    }

    float *result = operatorfusion_fun(input1.data(), input2.data(), input3.data());
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < length; ++i) {
        ASSERT_NEAR(result[i], real_result[i], 1e-5);
    }
}