#include "galois_test.hpp"

inline auto Full(Eigen::VectorXi64 shape, float value) {
    auto ir_ts_type = f32(199);
    auto ir_full_op = op::FullCreator::Create(ir_ts_type);
    auto ir_full = graph::ComputeNode::Create(ir_full_op, {});
    return ir_full;
}

TEST(GaloisTests, TestFull) {}
