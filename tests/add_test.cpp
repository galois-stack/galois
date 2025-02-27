#include "galois_test.hpp"

inline auto Add2(std::shared_ptr<graph::ComputeNode> node0,
                 std::shared_ptr<graph::ComputeNode> node1) {
    auto ir_ts_type = node0->type;
    auto ir_add_op = op::AddCreator::Create();
    auto ir_add = graph::ComputeNode::Create(ir_add_op, {node0, node1});
    return ir_add;
}

inline auto Full(Eigen::VectorXi64 shape, float value) {
    auto ir_ts_type = f32(199);
    auto ir_full_op = op::FullCreator::Create(ir_ts_type);
    auto ir_full = graph::ComputeNode::Create(ir_full_op, {});
    return ir_full;
}

TEST(GaloisTests, TestAdd) {
    // auto ir_1 = Full(Eigen::Vector1i64(2), 1.0f);
    // auto ir_2 = Full(Eigen::Vector1i64(2), 2.0f);

    // auto ir_add = Add2(ir_1, ir_2);

    // auto tmp_fun;
}
