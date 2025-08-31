#pragma once

#include <map>
#include <set>
#include <unordered_set>

#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/op.hpp"
#include "galois/transform/common.hpp"
#include "galois/transform/each.hpp"

namespace galois::transform {

inline void Split(std::shared_ptr<ir::Grid> ir_grid, std::int64_t dim_index,
                  std::int64_t splitted_dim_size) {
    GALOIS_ASSERT(dim_index < ir_grid->shape.size());
    Eigen::VectorXi64 new_affine_shape(ir_grid->shape.size() + 1);
    new_affine_shape.topRows(dim_index) = ir_grid->shape.topRows(dim_index);
    auto remained_dim_size = ir_grid->shape.size() - dim_index - 1;
    new_affine_shape.bottomRows(remained_dim_size) = ir_grid->shape.bottomRows(remained_dim_size);
    new_affine_shape[dim_index] = ir_grid->shape[dim_index] / splitted_dim_size;
    new_affine_shape[dim_index + 1] = splitted_dim_size;
    ir_grid->shape = new_affine_shape;

    Eigen::MatrixXi64 split_transform_matrix =
        Eigen::MatrixXi64::Zero(ir_grid->shape.size() - 1, ir_grid->shape.size());
    int64_t j = 0;
    for (int64_t i = 0; i < split_transform_matrix.cols(); ++i) {
        if (i == dim_index) {
            split_transform_matrix(dim_index, dim_index) = splitted_dim_size;
        } else {
            split_transform_matrix(j, i) = 1;
            ++j;
        }
    }

    Each<ir::Accessor>(ir_grid->block, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix =
            (ir_accessor->transform_matrix * split_transform_matrix).eval();
    });
}

inline void Swap(std::shared_ptr<ir::Grid> ir_grid, int64_t dim0, int64_t dim1) {
    std::swap(ir_grid->shape[dim0], ir_grid->shape[dim1]);

    Each<ir::Accessor>(ir_grid->block, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix.col(dim0).swap(ir_accessor->transform_matrix.col(dim1));
    });
}

inline void Tile(std::shared_ptr<ir::Grid> ir_grid, Eigen::VectorXi64 tile_dims) {
    GALOIS_ASSERT(ir_grid->shape.size() == tile_dims.size());
    ir_grid->shape.conservativeResize(ir_grid->shape.size() + tile_dims.size());
    for (int64_t i = 0; i < tile_dims.size(); ++i) {
        GALOIS_ASSERT((ir_grid->shape[i] % tile_dims[i]) == 0);
        ir_grid->shape[i] = ir_grid->shape[i] / tile_dims[i];
    }
    for (int64_t i = tile_dims.size(); i < ir_grid->shape.size(); ++i) {
        ir_grid->shape[i] = tile_dims[i - tile_dims.size()];
    }

    Eigen::MatrixXi64 tile_transform_matrix =
        Eigen::MatrixXi64::Zero(tile_dims.size(), ir_grid->shape.size());
    for (int64_t i = 0; i < tile_dims.size(); ++i) {
        tile_transform_matrix(i, i) = tile_dims[i];
        tile_transform_matrix(i, i + tile_dims.size()) = 1;
    }

    Each<ir::Accessor>(ir_grid->block, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix =
            (ir_accessor->transform_matrix * tile_transform_matrix).eval();
    });
}

inline void TileWithLayout(std::shared_ptr<ir::Grid> ir_grid, Eigen::VectorXi64 tile_dims) {
    GALOIS_ASSERT(ir_grid->shape.size() == tile_dims.size());
    ir_grid->shape.conservativeResize(ir_grid->shape.size() + tile_dims.size());
    for (int64_t i = 0; i < tile_dims.size(); ++i) {
        GALOIS_ASSERT((ir_grid->shape[i] % tile_dims[i]) == 0);
        ir_grid->shape[i] = ir_grid->shape[i] / tile_dims[i];
    }
    for (int64_t i = tile_dims.size(); i < ir_grid->shape.size(); ++i) {
        ir_grid->shape[i] = tile_dims[i - tile_dims.size()];
    }

    Eigen::MatrixXi64 tile_transform_matrix =
        Eigen::MatrixXi64::Zero(tile_dims.size(), ir_grid->shape.size());
    for (int64_t i = 0; i < tile_dims.size(); ++i) {
        tile_transform_matrix(i, i) = tile_dims[i];
        tile_transform_matrix(i, i + tile_dims.size()) = 1;
    }

    Each<ir::Accessor>(ir_grid->block, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix =
            (ir_accessor->transform_matrix * tile_transform_matrix).eval();
    });
}

inline std::shared_ptr<ir::Block> GetInnerMostBlock(std::shared_ptr<ir::Block> ir_block) {
    auto ir_block_iter = std::find_if(
        RANGE((*ir_block)),
        [&](std::shared_ptr<ir::Tensor> ir_tensor) { return Is<ir::Block>(ir_tensor); });
    if (ir_block_iter != ir_block->end()) {
        return GetInnerMostBlock(Cast<ir::Block>(*ir_block_iter));
    } else {
        return ir_block;
    }
}

// inline std::shared_ptr<ir::Grid> ExtractInnerGrid(std::shared_ptr<ir::Grid> ir_grid,
//   std::int64_t inner_dim_size) {
// auto ir_inner_grid = ir::Grid::Create(ir_grid->shape.bottomRows(inner_dim_size));
//     ir_grid->shape = (ir_grid->shape.topRows(ir_grid->shape.size() - inner_dim_size)).eval();

//     *ir_inner_grid->block = *ir_grid->block;
//     ir_inner_grid->is_local = false;
//     ir_inner_grid->parent_grid = ir_grid;
//     *ir_grid->block= {ir_inner_grid};
//     ir_inner_grid->name = "inner";

//     return ir_inner_grid;
// }

// inline void Vectorize(std::shared_ptr<ir::Grid> ir_grid, std::int64_t simd_size) {
//     bool is_valide = true;
//     Each<ir::Accessor>(ir_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
//         auto right_cols1 = ir_accessor->transform_matrix.rightCols(1);
//         auto tmp = right_cols1.bottomRows(1)(0, 0);
//         is_valide = is_valide && (tmp == 1 || (tmp == 0 && !ir_accessor->IsWritten())) &&
//                     right_cols1.topRows(right_cols1.size() - 1).isZero();
//     });

//     if (!is_valide) return;

//     ir_grid->shape.bottomRows(1)[0] /= simd_size;

//     Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
//         ir_accessor->transform_matrix.bottomRightCorner(1, 1)(0, 0) *= simd_size;
//         ir_accessor->simd_size = simd_size;
//         if (ir_accessor->transform_matrix.bottomRightCorner(1, 1).isZero()) {
//             ir_accessor->simd_shuffle = true;
//         }
//     });
// }

// inline void ExpandInstruction(std::shared_ptr<ir::Grid> ir_grid, int64_t copy_size) {
//     for (auto ir_value : Clone(*ir_grid->block)) {
//         if (!Is<ir::Write>(ir_value)) continue;

//         auto ir_value_iter = std::find(RANGE((*ir_grid->block)), ir_value);

//         for (int64_t i = 1; i < copy_size; ++i) {
//             auto ir_value_copy = ir_value->Clone();
//             EachTensor(ir_value_copy, [=](std::shared_ptr<ir::Tensor> ir_x) {
//                 if (auto ir_accessor = Cast<ir::Accessor>(ir_x)) {
//                     ir_accessor->shift_vector += (ir_accessor->transform_matrix.rightCols(1)
//                     * i); ir_accessor->transform_matrix.rightCols(1) *= copy_size;
//                 }
//             });

//             *ir_grid->blockinsert(ir_value_iter, ir_value_copy);
//         }

//         EachTensor(ir_value, [=](std::shared_ptr<ir::Tensor> ir_x) {
//             if (auto ir_accessor = Cast<ir::Accessor>(ir_x)) {
//                 ir_accessor->transform_matrix.rightCols(1) *= copy_size;
//             }
//         });
//     }

//     ir_grid->shape.bottomRows(1)[0] /= copy_size;
// }

// inline void LayerMemory(std::shared_ptr<ir::Operator> ir_operator) {
//     auto ir_grid = Cast<ir::Grid>(*ir_operator->blockfront());
//     auto ir_inner_grid = Cast<ir::Grid>(*ir_grid->blockfront());

//     std::multimap<std::shared_ptr<ir::Tensor>, std::shared_ptr<ir::Accessor>>
//         tensor_accessor_multimap;
//     Each<ir::Accessor>(ir_inner_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
//         tensor_accessor_multimap.insert({ir_accessor->Tensor(), ir_accessor});
//     });

//     std::map<std::shared_ptr<ir::Accessor>, std::shared_ptr<ir::Tensor>>
//     accessor_local_tensor_map;

//     int64_t i = 0;
//     for (auto iter = tensor_accessor_multimap.begin(); iter !=
//     tensor_accessor_multimap.end();
//          iter = tensor_accessor_multimap.upper_bound(iter->first)) {
//         auto ir_tensor = iter->first;

//         auto accessor_range = tensor_accessor_multimap.equal_range(ir_tensor);
//         bool is_readed = false;
//         bool is_written = false;
//         for (auto accessor_iter = accessor_range.first; accessor_iter !=
//         accessor_range.second;
//              ++accessor_iter) {
//             auto ir_accessor = accessor_iter->second;
//             GALOIS_VERIFY(ir_accessor->transform_matrix ==
//                           accessor_range.first->second->transform_matrix);
//             GALOIS_VERIFY(ir_accessor->shift_vector ==
//             accessor_range.first->second->shift_vector);

//             if (ir_accessor->IsReaded()) {
//                 is_readed = true;
//             }
//             if (ir_accessor->IsWritten()) {
//                 is_written = true;
//             }
//         }

//         if (is_written) continue;

//         auto ir_accessor_tmp = accessor_range.first->second;
//         Eigen::VectorXi64 local_tensor_shape =
//             ir_accessor_tmp->transform_matrix.rightCols(ir_accessor_tmp->transform_matrix.cols()
//             -
//                                                         ir_grid->shape.size()) *
//             ir_inner_grid->shape;

//         // Copy memory to local tensor
//         auto ir_local_tensor =
//             ir::CreateTensor(ir_accessor_tmp->Tensor()->type->value_type,
//             local_tensor_shape);

//         // TODO: 后面需要调整, 目前仅支持accessor一样的
//         auto ir_accessor = accessor_range.first->second;
//         if (is_readed) {
//             auto ir_load_grid = ir::Grid::Create(ir_inner_grid->shape);
//             ir_load_grid->is_local = false;
//             auto local_accessor_a = ir_accessor->transform_matrix;
//             local_accessor_a.leftCols(ir_grid->shape.size()).setZero();
//             auto ir_local_accessor =
//                 ir::Accessor::Create(ir_local_tensor, local_accessor_a,
//                 ir_accessor->shift_vector);
//             auto ir_global_accessor = ir::Accessor::Create(
//                 ir_accessor->Tensor(), ir_accessor->transform_matrix,
//                 ir_accessor->shift_vector);
//             auto ir_write_accessor = ir::Write::Create(ir_global_accessor,
//             ir_local_accessor); ir_load_grid->push_back(ir_write_accessor);
//             *ir_grid->blockpush_front(ir_load_grid);
//             ir_load_grid->parent_grid = ir_grid;

//             ir_load_grid->name = "copy" + std::to_string(i);
//         }

//         if (is_written) {
//             auto ir_store_grid = ir::Grid::Create(ir_inner_grid->shape);
//             ir_store_grid->is_local = false;
//             auto local_accessor_a = ir_accessor->transform_matrix;
//             local_accessor_a.leftCols(ir_grid->shape.size()).setZero();
//             auto ir_local_accessor =
//                 ir::Accessor::Create(ir_local_tensor, local_accessor_a,
//                 ir_accessor->shift_vector);
//             auto ir_global_accessor = ir::Accessor::Create(
//                 ir_accessor->Tensor(), ir_accessor->transform_matrix,
//                 ir_accessor->shift_vector);
//             auto ir_write_accessor = ir::Write::Create(ir_local_accessor,
//             ir_global_accessor); ir_store_grid->push_back(ir_write_accessor);
//             ir_store_grid->parent_grid = ir_grid;
//             *ir_grid->blockpush_back(ir_store_grid);

//             ir_store_grid->name = "store" + std::to_string(i);
//         }

//         ++i;
//         for (auto accessor_iter = accessor_range.first; accessor_iter !=
//         accessor_range.second;
//              ++accessor_iter) {
//             auto ir_accessor = accessor_iter->second;
//             ir_accessor->Tensor() = ir_local_tensor;
//             ir_accessor->transform_matrix.leftCols(ir_grid->shape.size()).setZero();
//         }

//         *ir_grid->blockpush_front(ir_local_tensor);
//     }
// }

// inline void LayerMemory2(std::shared_ptr<ir::Operator> ir_operator) {
//     auto ir_outer_grid = Cast<ir::Grid>(*ir_operator->blockfront());
//     auto ir_inner_grid = Cast<ir::Grid>(ir_outer_grid->front());

//     std::multimap<std::shared_ptr<ir::Tensor>, std::shared_ptr<ir::Accessor>>
//         tensor_accessor_multimap;
//     Each<ir::Accessor>(ir_inner_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
//         tensor_accessor_multimap.insert({ir_accessor->Tensor(), ir_accessor});
//     });

//     std::map<std::shared_ptr<ir::Accessor>, std::shared_ptr<ir::Tensor>>
//     accessor_local_tensor_map;

//     int64_t i = 0;
//     for (auto iter = tensor_accessor_multimap.begin(); iter !=
//     tensor_accessor_multimap.end();
//          iter = tensor_accessor_multimap.upper_bound(iter->first)) {
//         auto ir_tensor = iter->first;

//         auto accessor_range = tensor_accessor_multimap.equal_range(ir_tensor);
//         bool is_readed = false;
//         bool is_written = false;
//         for (auto accessor_iter = accessor_range.first; accessor_iter !=
//         accessor_range.second;
//              ++accessor_iter) {
//             auto ir_accessor = accessor_iter->second;
//             GALOIS_VERIFY(ir_accessor->transform_matrix ==
//                           accessor_range.first->second->transform_matrix);
//             GALOIS_VERIFY(ir_accessor->shift_vector ==
//             accessor_range.first->second->shift_vector);

//             if (ir_accessor->IsReaded()) {
//                 is_readed = true;
//             }
//             if (ir_accessor->IsWritten()) {
//                 is_written = true;
//             }
//         }

//         if (is_written) continue;

//         auto ir_accessor_tmp = accessor_range.first->second;
//         Eigen::VectorXi64 local_tensor_type_shape =
//             ir_accessor_tmp->transform_matrix.rightCols(ir_accessor_tmp->transform_matrix.cols()
//             -
//                                                         ir_outer_grid->shape.size()) *
//             ir_inner_grid->shape;
//         Eigen::VectorXi64 local_tensor_shape =
//             (ir_accessor_tmp->transform_matrix.leftCols(ir_outer_grid->shape.size()) *
//              ir_outer_grid->shape)
//                 .eval();
//         local_tensor_shape.array() /= local_tensor_type_shape.array();

//         // Copy memory to local tensor
//         auto ir_local_tensor_type = ir::TensorType::Create(
//             ir_accessor_tmp->Tensor()->type->value_type, local_tensor_type_shape);
//         auto ir_local_tensor = ir::CreateTensor(ir_local_tensor_type, local_tensor_shape);

//         // TODO: 后面需要调整, 目前仅支持accessor一样的
//         auto ir_accessor = accessor_range.first->second;
//         auto ir_copy_grid_outer = ir::Grid::Create(ir_outer_grid->shape);
//         ir_copy_grid_outer->is_local = true;
//         *ir_operator->blockpush_front(ir_copy_grid_outer);
//         *ir_operator->blockpush_front(ir_local_tensor);
//         Eigen::MatrixXi64 ir_accessor_outer_transform_matrix =
//             Eigen::MatrixXi64::Zero(local_tensor_shape.size(), ir_outer_grid->shape.size());
//         for (int64_t i = 0; i < ir_accessor_outer_transform_matrix.rows(); ++i) {
//             for (int64_t j = 0; j < ir_accessor_outer_transform_matrix.cols(); ++j) {
//                 if (ir_accessor->transform_matrix(i, j) != 0) {
//                     ir_accessor_outer_transform_matrix(i, j) = 1;
//                 }
//             }
//         }
//         auto ir_local_accessor_outer =
//             ir::Accessor::Create(ir_local_tensor, ir_accessor_outer_transform_matrix,
//                                  Eigen::VectorXi64::Zero(local_tensor_shape.size()));
//         ir_copy_grid_outer->push_back(ir_local_accessor_outer);

//         auto ir_copy_grid_inner = ir::Grid::Create(ir_inner_grid->shape);
//         ir_copy_grid_inner->is_local = false;
//         ir_copy_grid_outer->push_back(ir_copy_grid_inner);

//         // auto local_accessor_a = ir_accessor->transform_matrix;
//         // local_accessor_a.rightCols(ir_inner_grid->shape.size()).setZero();
//         // for (int64_t i = 0; i < )
//         // local_accessor_a.leftCols(ir_grid->shape.size()).setZero();
//         Eigen::MatrixXi64 ir_inner_accessor_transform_matrix = ir_accessor->transform_matrix;
//         ir_inner_accessor_transform_matrix.leftCols(ir_outer_grid->shape.size()).setZero();
//         auto ir_local_accessor_inner =
//             ir::Accessor::Create(ir_local_accessor_outer, ir_inner_accessor_transform_matrix,
//                                  Eigen::VectorXi64::Zero(local_tensor_type_shape.size()));

//         auto ir_global_accessor = ir::Accessor::Create(
//             ir_accessor->Tensor(), ir_accessor->transform_matrix, ir_accessor->shift_vector);
//         auto ir_write_accessor = ir::Write::Create(ir_global_accessor,
//         ir_local_accessor_inner); ir_copy_grid_inner->push_back(ir_write_accessor);

//         for (auto accessor_iter = accessor_range.first; accessor_iter !=
//         accessor_range.second;
//              ++accessor_iter) {
//             auto ir_accessor = accessor_iter->second;
//             Eigen::MatrixXi64 ir_accessor_outer_transform_matrix =
//                 Eigen::MatrixXi64::Zero(local_tensor_shape.size(),
//                 ir_outer_grid->shape.size());
//             for (int64_t i = 0; i < ir_accessor_outer_transform_matrix.rows(); ++i) {
//                 for (int64_t j = 0; j < ir_accessor_outer_transform_matrix.cols(); ++j) {
//                     if (ir_accessor->transform_matrix(i, j) != 0) {
//                         ir_accessor_outer_transform_matrix(i, j) = 1;
//                     }
//                 }
//             }
//             auto ir_local_accessor_outer =
//                 ir::Accessor::Create(ir_local_tensor, ir_accessor_outer_transform_matrix,
//                                      Eigen::VectorXi64::Zero(local_tensor_shape.size()));
//             ir_accessor->Tensor() = ir_local_accessor_outer;
//             ir_accessor->transform_matrix.leftCols(ir_outer_grid->shape.size()) =
//                 ir_accessor_outer_transform_matrix;
//         }
//     }
// }

inline bool IsUselessDim(std::shared_ptr<ir::Grid> ir_grid, int64_t dim_index) {
    bool useless = true;
    auto parent_dim_size = ir_grid->GetAffineDimSize() - ir_grid->shape.size();
    Each<ir::Accessor>(ir_grid->block, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        useless =
            useless && ir_accessor->transform_matrix.col(parent_dim_size + dim_index).isZero();
    });

    return useless;
}

inline void RemoveDim(std::shared_ptr<ir::Grid> ir_grid, int64_t dim_index) {
    GALOIS_ASSERT(IsUselessDim(ir_grid, dim_index));
    auto parent_dim_size = ir_grid->GetAffineDimSize() - ir_grid->shape.size();
    Each<ir::Accessor>(ir_grid->block, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        RemoveColumn(ir_accessor->transform_matrix, parent_dim_size + dim_index);
    });

    RemoveRow(ir_grid->shape, ir_grid->shape.size() - dim_index);
}

inline void RemoveUselessDim(std::shared_ptr<ir::Grid> ir_grid) {
    for (int64_t i = 0; i < ir_grid->shape.size(); ++i) {
        if (IsUselessDim(ir_grid, i)) {
            RemoveDim(ir_grid, i);
        }
    }
}

inline void Repeat(std::shared_ptr<ir::Operator> ir_operator, int64_t times) {
    Eigen::VectorXi64 grid_shape(1);
    grid_shape[0] = times;
    auto ir_repeat_grid = ir::Grid::Create(grid_shape);

    for (auto ir_value : *ir_operator->block) {
        ir_repeat_grid->block->push_back(ir_value);
    }

    ir_operator->block->clear();
    ir_operator->block->push_back(ir_repeat_grid);
}

inline void AsyncInvokeByThreadPool(std::shared_ptr<ir::Block> ir_block) {
    //     auto ir_captured_tensor_set = CaptureExternalTensors(ir_block);
    //     std::vector<std::shared_ptr<ir::TensorType>> ir_input_types;
    //     std::transform(RANGE(ir_captured_tensor_set), std::back_inserter(ir_input_types),
    //                    [](std::shared_ptr<ir::Tensor> ir_tensor) {
    //                        GALOIS_ASSERT(ir_tensor->type);
    //                        return ir_tensor->type;
    //                    });
    //     std::vector<std::shared_ptr<ir::TensorType>> output_types;
    //     auto ir_operator_function = ir::Operator::Create(ir_input_types, output_types);
    //     ir_operator_function->name = "__tmp_todo";
    //     ir_operator_function->fullname = ir_operator_function->name;
    //     std::unordered_map<std::shared_ptr<ir::Tensor>, std::shared_ptr<ir::Tensor>>
    //         captured_tensor_dict;
    //     auto ir_captured_tensor_set_iter = ir_captured_tensor_set.begin();
    //     for (int64_t i = 0; i < ir_captured_tensor_set.size(); ++i,
    //     ++ir_captured_tensor_set_iter) {
    //         captured_tensor_dict[*ir_captured_tensor_set_iter] =
    //         ir_operator_function->inputs[i];
    //     }

    //     ir_operator_function-> = std::move(ir_block[0]);
    //     EachTensor(Cast<ir ::Block>(ir_operator_function), [=](std::shared_ptr<ir::Tensor>
    //     ir_tensor) {
    //         if (auto ir_instruction = Cast<ir::Instruction>(ir_tensor)) {
    //             for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
    //                 auto ir_operand = ir_instruction->GetOperand(i);
    //                 if (ir_captured_tensor_set.count(ir_operand)) {
    //                     ir_instruction->SetOperand(i, captured_tensor_dict.at(ir_operand));
    //                 }
    //             }
    //         }
    //     });

    //     auto ir_builder = ir::Builder::Create();
    //     ir_block[0]clear();
    //     ir_block[0]push_back(ir_operator_function);
    //     ir_builder->block_stack.push(ir_block);
    //     ir_builder->iterator_stack.push(ir_block[0]end());
    //     std::vector<std::shared_ptr<ir::Tensor>>
    //     ir_captured_tensor_vec(RANGE(ir_captured_tensor_set)); auto ir_call =
    //     ir_builder->Create<ir::Call>(ir_operator_function, ir_captured_tensor_vec,
    //                                                 std::vector<std::shared_ptr<ir::Tensor>>{});
    //     ir_call->annotation_dict["enable_multi_thread"] = {};
}

// inline void CombineElementwiseOperators(std::shared_ptr<ir::Model> ir_model) {
//     for (auto ir_operator : ir_model->operators) {
//     }
// }

inline bool IsElementWiseOperator(const std::shared_ptr<ir::Operator>& op) {
    const std::vector<std::string> elemWiseKeywords = {"Add", "Sub", "Mul", "Div"};
    return std::any_of(
        elemWiseKeywords.begin(), elemWiseKeywords.end(),
        [&](const std::string& kw) { return op->name.find(kw) != std::string::npos; });
}

template <typename Tensor_>
inline std::vector<std::shared_ptr<Tensor_>> ExtractAllFromBlock(std::shared_ptr<ir::Block> block) {
    std::vector<std::shared_ptr<Tensor_>> result;
    if (!block) return result;

    for (auto& tensor : *block) {
        // if (auto target = std::dynamic_pointer_cast<Tensor_>(tensor)) {
        //     result.push_back(target);
        // }
        if (auto target = std::dynamic_pointer_cast<Tensor_>(tensor)) {
            result.push_back(target);
        }
        if (auto sub_op = std::dynamic_pointer_cast<ir::Operator>(tensor)) {
            auto sub_results = ExtractAllFromBlock<Tensor_>(sub_op->block);
            result.insert(result.end(), sub_results.begin(), sub_results.end());
        }
    }
    return result;
}

template <typename Tensor_>
struct TreeNode {
    std::shared_ptr<Tensor_> value;
    std::vector<TreeNode<Tensor_>> children;
};

template <typename Tensor_>
inline std::vector<TreeNode<Tensor_>> ExtractHierarchicalFromBlock(
    std::shared_ptr<ir::Block> block) {
    std::vector<TreeNode<Tensor_>> current_level;
    if (!block) return current_level;

    for (auto& tensor : *block) {
        if (auto target = std::dynamic_pointer_cast<Tensor_>(tensor)) {
            TreeNode<Tensor_> node;
            node.value = target;

            if constexpr (std::is_same_v<Tensor_, ir::Operator>) {
                node.children = ExtractHierarchicalFromBlock<Tensor_>(target->block);
            }

            current_level.push_back(node);
        }

        if (auto grid = std::dynamic_pointer_cast<ir::Grid>(tensor)) {
            auto grid_children = ExtractHierarchicalFromBlock<Tensor_>(grid->block);
            current_level.insert(current_level.end(), grid_children.begin(), grid_children.end());
        }
    }

    return current_level;
}

template <typename Tensor_>
void PrintHierarchy(const std::vector<TreeNode<Tensor_>>& nodes, int depth = 0) {
    std::string indent(depth * 2, ' ');
    for (const auto& node : nodes) {
        if (!node.value) continue;
        std::cout << indent << "Level " << depth << ": Operator '" << node.value->name << "'\n";
        if (!node.children.empty()) {
            PrintHierarchy(node.children, depth + 1);
        }
    }
}

inline void ReplaceTensorReference(std::shared_ptr<ir::Block> block,
                                   std::shared_ptr<ir::Tensor> old_tensor,
                                   std::shared_ptr<ir::Tensor> new_tensor) {
    for (auto& tensor : *block) {
        if (auto instr = Cast<ir::Instruction>(tensor)) {
            for (int64_t i = 0; i < instr->OperandSize(); ++i) {
                if (instr->GetOperand(i) == old_tensor) {
                    instr->SetOperand(i, new_tensor);
                }
            }
        }
    }
}

template <typename Tensor_>
inline std::shared_ptr<Tensor_> FindOperatorByName(std::shared_ptr<ir::Operator> parent_op,
                                                   const std::string& name) {
    auto ops = ExtractAllFromBlock<Tensor_>(parent_op->block);
    for (auto op : ops) {
        if (op->name == name) {
            return op;
        }
    }
    return nullptr;
}

inline std::shared_ptr<ir::Call> FindCallByName(std::shared_ptr<ir::Block> block,
                                                const std::string& target_name) {
    std::shared_ptr<ir::Call> result;
    Each<ir::Call>(block, [&](std::shared_ptr<ir::Call> call) {
        if (call->Operator()->name == target_name) {
            result = call;
        }
    });
    return result;
}

inline std::list<std::shared_ptr<ir::Tensor>>::iterator FindInBlock(
    std::shared_ptr<ir::Block> block, std::shared_ptr<ir::Tensor> tensor) {
    for (auto it = block->begin(); it != block->end(); ++it) {
        if (*it == tensor) {  // 比较指针是否指向同一个Tensor
            return it;
        }
    }
    return block->end();  // 未找到返回end()
}

template <typename Tensor_>
inline void FindInstrRecursive(std::shared_ptr<ir::Tensor> tensor,
                               std::shared_ptr<Tensor_>& single_instr,
                               std::list<std::shared_ptr<ir::Tensor>>::iterator& single_it,
                               std::shared_ptr<ir::Block> current_block) {
    if (auto target = Cast<Tensor_>(tensor)) {
        if constexpr (std::is_same_v<Tensor_, ir::ArithmeticInstruction>) {
            if (target->operation == ir::ArithmeticInstruction::Add) {
                single_instr = target;
                auto it = FindInBlock(current_block, tensor);
                if (it != current_block->end()) {
                    single_it = std::next(it);
                }
            }
        } else {
            single_instr = target;
            auto it = FindInBlock(current_block, tensor);
            if (it != current_block->end()) {
                single_it = std::next(it);
            }
        }
    }
}

template <typename Tensor_>
inline void FindInstrRecursiveInner(std::shared_ptr<ir::Tensor> tensor,
                                    std::shared_ptr<Tensor_>& single_instr,
                                    std::list<std::shared_ptr<ir::Tensor>>::iterator& single_it,
                                    std::shared_ptr<ir::Block> current_block) {
    if (auto block = Cast<ir::Block>(tensor)) {
        for (auto& sub_tensor : *block) {
            FindInstrRecursive<Tensor_>(sub_tensor, single_instr, single_it, block);
        }
    } else if (auto grid = Cast<ir::Grid>(tensor)) {
        for (auto& sub_tensor : *grid->block) {
            FindInstrRecursive<Tensor_>(sub_tensor, single_instr, single_it, grid->block);
        }
    }
}

inline void PrintFusibleChains(std::vector<std::vector<std::shared_ptr<ir::Call>>> fusibleChains) {
    std::cout << "发现 " << fusibleChains.size() << " 条可融合的element-wise操作链：" << std::endl;

    for (size_t i = 0; i < fusibleChains.size(); ++i) {
        const auto& chain = fusibleChains[i];
        std::cout << "  链 " << (i + 1) << "（包含 " << chain.size() << " 个操作）：" << std::endl;
        
        for (size_t j = 0; j < chain.size(); ++j) {
            const auto& call = chain[j];
            // 假设Operator()返回操作符对象，Name()返回操作名称
            std::cout << "    操作 " << (j + 1) << ": " << call->Operator()->name << std::endl;
        }
    }
}

inline void LoopFusion(std::shared_ptr<ir::Operator> root_op) {
    if (!root_op || !root_op->block) return;

    auto ir_printer = ir::IRPrinter::Create();
    auto ir_builder = ir::Builder::Create();

    std::vector<std::shared_ptr<ir::Call>> elemWiseCalls;
    Each<ir::Call>(root_op->block, [&](std::shared_ptr<ir::Call> call) {
        if (IsElementWiseOperator(call->Operator())) {
            elemWiseCalls.push_back(call);
        }
    });

    std::unordered_map<std::shared_ptr<ir::Tensor>, std::vector<std::shared_ptr<ir::Call>>> tensorConsumers;
    
    for (auto& call : elemWiseCalls) {
        for (int64_t i = 0; i < call->InputSize(); ++i) {
            auto input = call->Input(i);
            tensorConsumers[input].push_back(call);
        }
    }

    std::vector<std::vector<std::shared_ptr<ir::Call>>> fusibleChains;
    std::unordered_set<std::shared_ptr<ir::Call>> processed;

    for (auto& current : elemWiseCalls) {
        if (processed.count(current)) continue;

        std::vector<std::shared_ptr<ir::Call>> chain;
        
        while (current && !processed.count(current)) {
            processed.insert(current);
            chain.push_back(current);
            
            std::shared_ptr<ir::Call> nextCall = nullptr;
            auto output = current;
            if (tensorConsumers.count(output)) {
                for (auto& consumer : tensorConsumers[output]) {
                    if (!processed.count(consumer) && IsElementWiseOperator(consumer->Operator())) {
                        nextCall = consumer;
                        break;
                    }
                }
            }
            
            current = nextCall;
        }
        
        if (chain.size() > 1) {
            fusibleChains.push_back(chain);
        }
    }

    PrintFusibleChains(fusibleChains);

    auto call_add1 = fusibleChains[0][0];
    auto call_sub3 = fusibleChains[0][1];
    auto add1_op = call_add1->Operator();
    auto sub3_op = call_sub3->Operator();

    GALOIS_ASSERT(add1_op && sub3_op && call_add1 && call_sub3, "目标算子或调用未找到");

    auto orig_op_type = add1_op->GetOperatorType();
    std::vector<std::shared_ptr<ir::TensorType>> new_input_types = orig_op_type->ir_input_types;
    new_input_types.push_back(sub3_op->GetOperatorType()->ir_input_types[1]);

    auto new_op_type = ir::OperatorType::Create(new_input_types, orig_op_type->output_type);
    add1_op->type = new_op_type;

    add1_op->inputs.push_back(ir::Input::Create(sub3_op->GetOperatorType()->ir_input_types[1]));

    auto& add1_block = add1_op->block;

    auto accessor_it = add1_block->end();
    auto add_it = add1_block->end();
    auto write_it_inner = add1_block->end();
    auto write_it = add1_block->end();
    auto grid_it = add1_block->end();
    std::shared_ptr<ir::Accessor> accessor_instr;
    std::shared_ptr<ir::ArithmeticInstruction> add_instr;
    std::shared_ptr<ir::Write> write_instr_inner;
    std::shared_ptr<ir::Write> write_instr;
    std::shared_ptr<ir::Grid> grid_instr;

    for (auto& tensor : *add1_block) {
        FindInstrRecursiveInner(tensor, accessor_instr, accessor_it, add1_block);
        FindInstrRecursiveInner(tensor, add_instr, add_it, add1_block);
        FindInstrRecursiveInner(tensor, write_instr_inner, write_it_inner, add1_block);
        FindInstrRecursive<ir::Grid>(tensor, grid_instr, grid_it, add1_block);
    }
    for (auto& tensor : *root_op->block) {
        FindInstrRecursive<ir::Write>(tensor, write_instr, write_it, root_op->block);
    }

    GALOIS_ASSERT(accessor_instr, "accessor_instr内部结构不符合预期");
    GALOIS_ASSERT(add_instr, "add_instr内部结构不符合预期");
    GALOIS_ASSERT(write_instr_inner, "write_instr_inner内部结构不符合预期");
    GALOIS_ASSERT(grid_instr, "grid_instr内部结构不符合预期");
    GALOIS_ASSERT(write_instr, "write_instr内部结构不符合预期");

    ir_builder->grid_stack.push(grid_instr);
    ir_builder->block_stack.push(grid_instr->block);
    ir_builder->iterator_stack.push(accessor_it);
    auto input2_accessor = ir_builder->CreateIdentityAccessor(add1_op->inputs[2]);
    ir_builder->grid_stack.pop();
    ir_builder->block_stack.pop();
    ir_builder->iterator_stack.pop();

    auto sub_instr = ir::ArithmeticInstruction::Create(ir::ArithmeticInstruction::Sub, add_instr,
                                                       input2_accessor);

    ir_builder->grid_stack.push(grid_instr);
    ir_builder->block_stack.push(grid_instr->block);
    ir_builder->iterator_stack.push(std::next(accessor_it));
    ir_builder->Insert(sub_instr);
    ir_builder->grid_stack.pop();
    ir_builder->block_stack.pop();
    ir_builder->iterator_stack.pop();

    write_instr_inner->SetOperand(0, sub_instr);

    std::vector<std::shared_ptr<ir::Tensor>> new_call_inputs;
    for (int64_t i = 0; i < call_add1->InputSize(); ++i) {
        new_call_inputs.push_back(call_add1->Input(i));
    }
    new_call_inputs.push_back(call_sub3->Input(1));

    auto call_all = ir::Call::Create(add1_op, new_call_inputs);
    auto call_add1_it = std::find(root_op->block->begin(), root_op->block->end(), call_add1);

    ir_builder->block_stack.push(root_op->block);
    ir_builder->iterator_stack.push(std::next(call_add1_it));
    ir_builder->Insert(call_all);
    ir_builder->block_stack.pop();
    ir_builder->iterator_stack.pop();

    auto& block_l1 = root_op->block;
    block_l1->erase(
        std::remove_if(block_l1->begin(), block_l1->end(),
                       [&](const std::shared_ptr<ir::Tensor>& t) { return t == call_add1; }),
        block_l1->end());
    block_l1->erase(
        std::remove_if(block_l1->begin(), block_l1->end(),
                       [&](const std::shared_ptr<ir::Tensor>& t) { return t == sub3_op; }),
        block_l1->end());
    block_l1->erase(
        std::remove_if(block_l1->begin(), block_l1->end(),
                       [&](const std::shared_ptr<ir::Tensor>& t) { return t == call_sub3; }),
        block_l1->end());

    GALOIS_ASSERT(write_instr, "未在block中找到Write指令");
    write_instr->SetOperand(0, call_all);
}

inline std::shared_ptr<ir::Operator> OperatorFusionOpt(std::shared_ptr<ir::Operator> ir_operator) {
    ir_operator->block->clear();

    auto ir_builder = ir::Builder::Create();
    std::vector<std::string> operations = {"add", "sub"};

    auto sp_creator = op::OperatorFusionCreator::Create(operations);
    auto [ir_operator_fused, scope] =
        ir_builder->CreateOperator(ir_operator->GetOperatorType(), ir_operator->name + "_fused");
    std::vector<std::shared_ptr<ir::Tensor>> ir_inputs;
    std::transform(RANGE(ir_operator_fused->inputs), std::back_inserter(ir_inputs),
                   [](std::shared_ptr<ir::Tensor> ir_input) { return ir_input; });
    sp_creator->Express(ir_inputs, ir_builder);

    return ir_operator_fused;
}
}  // namespace galois::transform