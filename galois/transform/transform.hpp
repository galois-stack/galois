#pragma once

#include <map>
#include <set>

#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
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
    std::shared_ptr<Tensor_> value;  // 当前节点的Tensor_实例
    std::vector<TreeNode<Tensor_>> children;  // 子节点（来自value->block中的Tensor_）
};

template <typename Tensor_>
inline std::vector<TreeNode<Tensor_>> ExtractHierarchicalFromBlock(std::shared_ptr<ir::Block> block) {
    std::vector<TreeNode<Tensor_>> current_level;
    if (!block) return current_level;  // 空block直接返回空

    // 遍历block中直接包含的所有tensor
    for (auto& tensor : *block) {
        // 1. 若当前tensor是目标类型Tensor_，创建节点
        if (auto target = std::dynamic_pointer_cast<Tensor_>(tensor)) {
            TreeNode<Tensor_> node;
            node.value = target;

            // 2. 检查该Tensor_是否包含block（如Operator有block成员），若有则递归提取子节点
            // 这里以Operator为例，若Tensor_是其他含block的类型，可类似扩展
            if constexpr (std::is_same_v<Tensor_, ir::Operator>) {
                // 提取Operator->block中的Tensor_作为子节点
                node.children = ExtractHierarchicalFromBlock<Tensor_>(target->block);
            }
            // 若有其他含block的Tensor_类型（如自定义类型），可在此添加判断
            // 例如：else if constexpr (std::is_same_v<Tensor_, ir::CustomType>) { ... }

            current_level.push_back(node);
        }

        // 3. 处理其他可能包含block的非Tensor_类型（如Grid），避免遗漏嵌套的Tensor_
        // （若Grid中可能包含Tensor_，则递归处理其block）
        if (auto grid = std::dynamic_pointer_cast<ir::Grid>(tensor)) {
            auto grid_children = ExtractHierarchicalFromBlock<Tensor_>(grid->block);
            // 将Grid的block中提取的节点加入当前层级（因为Grid是block的直接子节点）
            current_level.insert(current_level.end(), grid_children.begin(), grid_children.end());
        }
    }

    return current_level;
}

template <typename Tensor_>
void PrintHierarchy(const std::vector<TreeNode<Tensor_>>& nodes, int depth = 0) {
    std::string indent(depth * 2, ' ');  // 每层缩进2个空格
    for (const auto& node : nodes) {
        if (!node.value) continue;
        // 打印当前节点信息（以Operator为例）
        std::cout << indent << "Level " << depth << ": Operator '" << node.value->name << "'\n";
        // 递归打印子节点
        if (!node.children.empty()) {
            PrintHierarchy(node.children, depth + 1);
        }
    }
}

}  // namespace galois::transform
