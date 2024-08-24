#pragma once

#include <map>

#include "galois/helper.hpp"
#include "galois/ir/ir.hpp"

namespace galois::transform {

inline void EachTensor(std::shared_ptr<ir::Block> ir_grid,
                       std::function<void(std::shared_ptr<ir::Tensor>)> callback);

inline void EachTensor(std::shared_ptr<ir::Tensor> ir_value,
                       std::function<void(std::shared_ptr<ir::Tensor>)> callback) {
    callback(ir_value);
    if (auto ir_instruction = Cast<ir::Instruction>(ir_value)) {
        for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
            EachTensor(ir_instruction->GetOperand(i), callback);
        }
    }
    if (auto ir_grid = Cast<ir::Block>(ir_value)) {
        // if (!ir_grid->is_local) {
        EachTensor(ir_grid, callback);
        // }
    }
}

inline void EachTensor(std::shared_ptr<ir::Block> ir_grid,
                       std::function<void(std::shared_ptr<ir::Tensor>)> callback) {
    callback(ir_grid);
    for (auto ir_value : ir_grid->values) {
        EachTensor(ir_value, callback);
    }
}

template <typename Value_>
inline void Each(std::shared_ptr<ir::Block> ir_grid,
                 std::function<void(std::shared_ptr<Value_>)> callback) {
    EachTensor(ir_grid, [=](auto ir_e) {
        if (auto ir_value_ = Cast<Value_>(ir_e)) {
            callback(ir_value_);
        }
    });
}

template <typename Value_>
inline void Each(std::shared_ptr<ir::OperatorInstance> ir_operator,
                 std::function<void(std::shared_ptr<Value_>)> callback) {
    for (auto ir_value : ir_operator->values) {
        if (auto ir_block = Cast<ir::Block>(ir_value)) {
            Each<Value_>(ir_block, callback);
        } else {
            callback(Cast<Value_>(ir_value));
        }
    }
}

template <typename Matrix_>
inline void RemoveRow(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (index < numRows)
        matrix.block(index, 0, numRows - index, numCols) = matrix.bottomRows(numRows - index);

    matrix.conservativeResize(numRows, numCols);
}

template <typename Matrix_>
inline void RemoveColumn(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (index < numCols)
        matrix.block(0, index, numRows, numCols - index) = matrix.rightCols(numCols - index);

    matrix.conservativeResize(numRows, numCols);
}

inline void ApplyTransformMatrix(std::shared_ptr<ir::Grid> op, Eigen::Matrix2Xi transform_matrix) {
    // auto t_matrix = transform_matrix.transpose();
    // op->shape =
    //     ((t_matrix * transform_matrix).Cast<double>().inverse() *
    //     op->shape.Cast<double>())
    //         .Cast<int>();

    // for (auto ir_instruction : op->instructions) {
    //     // if (auto ir_accessor = Cast<ir::Accessor>(ir_instruction)) {
    //     //     ir_accessor->transform_matrix = ir_accessor->transform_matrix * transform_matrix;
    //     // }
    // }
}

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

    Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix =
            (ir_accessor->transform_matrix * split_transform_matrix).eval();
    });
}

inline void Swap(std::shared_ptr<ir::Grid> ir_grid, int64_t dim0, int64_t dim1) {
    std::swap(ir_grid->shape[dim0], ir_grid->shape[dim1]);

    Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
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

    Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
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

    Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix =
            (ir_accessor->transform_matrix * tile_transform_matrix).eval();
    });
}

inline std::shared_ptr<ir::Grid> ExtractInnerGrid(std::shared_ptr<ir::Grid> ir_grid,
                                                  std::int64_t inner_dim_size) {
    auto ir_inner_parallel = ir::Grid::Create();
    ir_inner_parallel->shape = ir_grid->shape.bottomRows(inner_dim_size);
    ir_grid->shape = (ir_grid->shape.topRows(ir_grid->shape.size() - inner_dim_size)).eval();

    ir_inner_parallel->values = ir_grid->values;
    ir_inner_parallel->is_local = false;
    ir_inner_parallel->parent_parallel = ir_grid;
    ir_grid->values = {ir_inner_parallel};
    ir_inner_parallel->name = "inner";

    return ir_inner_parallel;
}

inline void Vectorize(std::shared_ptr<ir::Grid> ir_grid, std::int64_t simd_size) {
    bool is_valide = true;
    Each<ir::Accessor>(ir_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        auto right_cols1 = ir_accessor->transform_matrix.rightCols(1);
        auto tmp = right_cols1.bottomRows(1)(0, 0);
        is_valide = is_valide && (tmp == 1 || (tmp == 0 && !ir_accessor->IsWritten())) &&
                    right_cols1.topRows(right_cols1.size() - 1).isZero();
    });

    if (!is_valide) return;

    ir_grid->shape.bottomRows(1)[0] /= simd_size;

    Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) {
        ir_accessor->transform_matrix.bottomRightCorner(1, 1)(0, 0) *= simd_size;
        ir_accessor->simd_size = simd_size;
        if (ir_accessor->transform_matrix.bottomRightCorner(1, 1).isZero()) {
            ir_accessor->simd_shuffle = true;
        }
    });
}

inline void ExpandInstruction(std::shared_ptr<ir::Grid> ir_grid, int64_t copy_size) {
    for (auto ir_value : Clone(ir_grid->values)) {
        if (!Is<ir::Write>(ir_value)) continue;

        auto ir_value_iter = std::find(RANGE(ir_grid->values), ir_value);

        for (int64_t i = 1; i < copy_size; ++i) {
            auto ir_value_copy = ir_value->Clone();
            EachTensor(ir_value_copy, [=](std::shared_ptr<ir::Tensor> ir_x) {
                if (auto ir_accessor = Cast<ir::Accessor>(ir_x)) {
                    ir_accessor->shift_vector += (ir_accessor->transform_matrix.rightCols(1) * i);
                    ir_accessor->transform_matrix.rightCols(1) *= copy_size;
                }
            });

            ir_grid->values.insert(ir_value_iter, ir_value_copy);
        }

        EachTensor(ir_value, [=](std::shared_ptr<ir::Tensor> ir_x) {
            if (auto ir_accessor = Cast<ir::Accessor>(ir_x)) {
                ir_accessor->transform_matrix.rightCols(1) *= copy_size;
            }
        });
    }

    ir_grid->shape.bottomRows(1)[0] /= copy_size;
}

inline void LayerMemory(std::shared_ptr<ir::OperatorInstance> ir_operator) {
    auto ir_grid = Cast<ir::Grid>(ir_operator->values.front());
    auto ir_inner_parallel = Cast<ir::Grid>(ir_grid->values.front());

    std::multimap<std::shared_ptr<ir::Tensor>, std::shared_ptr<ir::Accessor>>
        tensor_accessor_multimap;
    Each<ir::Accessor>(ir_inner_parallel, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        tensor_accessor_multimap.insert({ir_accessor->Tensor(), ir_accessor});
    });

    std::map<std::shared_ptr<ir::Accessor>, std::shared_ptr<ir::Tensor>> accessor_local_tensor_map;

    int64_t i = 0;
    for (auto iter = tensor_accessor_multimap.begin(); iter != tensor_accessor_multimap.end();
         iter = tensor_accessor_multimap.upper_bound(iter->first)) {
        auto ir_tensor = iter->first;

        auto accessor_range = tensor_accessor_multimap.equal_range(ir_tensor);
        bool is_readed = false;
        bool is_written = false;
        for (auto accessor_iter = accessor_range.first; accessor_iter != accessor_range.second;
             ++accessor_iter) {
            auto ir_accessor = accessor_iter->second;
            GALOIS_VERIFY(ir_accessor->transform_matrix ==
                          accessor_range.first->second->transform_matrix);
            GALOIS_VERIFY(ir_accessor->shift_vector == accessor_range.first->second->shift_vector);

            if (ir_accessor->IsReaded()) {
                is_readed = true;
            }
            if (ir_accessor->IsWritten()) {
                is_written = true;
            }
        }

        if (is_written) continue;

        auto ir_accessor_tmp = accessor_range.first->second;
        Eigen::VectorXi64 local_tensor_shape =
            ir_accessor_tmp->transform_matrix.rightCols(ir_accessor_tmp->transform_matrix.cols() -
                                                        ir_grid->shape.size()) *
            ir_inner_parallel->shape;

        // Copy memory to local tensor
        auto ir_local_tensor =
            ir::CreateTensor(ir_accessor_tmp->Tensor()->type->value_type, local_tensor_shape);

        // TODO: 后面需要调整, 目前仅支持accessor一样的
        auto ir_accessor = accessor_range.first->second;
        if (is_readed) {
            auto ir_load_parallel = ir::Grid::Create();
            ir_load_parallel->is_local = false;
            ir_load_parallel->shape = ir_inner_parallel->shape;
            auto local_accessor_a = ir_accessor->transform_matrix;
            local_accessor_a.leftCols(ir_grid->shape.size()).setZero();
            auto ir_local_accessor =
                ir::Accessor::Create(ir_local_tensor, local_accessor_a, ir_accessor->shift_vector);
            ir_local_accessor->simd_size = ir_accessor->simd_size;
            ir_local_accessor->simd_shuffle = ir_accessor->simd_shuffle;
            auto ir_global_accessor = ir::Accessor::Create(
                ir_accessor->Tensor(), ir_accessor->transform_matrix, ir_accessor->shift_vector);
            ir_global_accessor->simd_size = ir_accessor->simd_size;
            ir_global_accessor->simd_shuffle = ir_accessor->simd_shuffle;
            auto ir_write_accessor = ir::Write::Create(ir_global_accessor, ir_local_accessor);
            ir_load_parallel->values.push_back(ir_write_accessor);
            ir_grid->values.push_front(ir_load_parallel);
            ir_load_parallel->parent_parallel = ir_grid;

            ir_load_parallel->name = "copy" + std::to_string(i);
        }

        if (is_written) {
            auto ir_store_parallel = ir::Grid::Create();
            ir_store_parallel->is_local = false;
            ir_store_parallel->shape = ir_inner_parallel->shape;
            auto local_accessor_a = ir_accessor->transform_matrix;
            local_accessor_a.leftCols(ir_grid->shape.size()).setZero();
            auto ir_local_accessor =
                ir::Accessor::Create(ir_local_tensor, local_accessor_a, ir_accessor->shift_vector);
            ir_local_accessor->simd_size = ir_accessor->simd_size;
            ir_local_accessor->simd_shuffle = ir_accessor->simd_shuffle;
            auto ir_global_accessor = ir::Accessor::Create(
                ir_accessor->Tensor(), ir_accessor->transform_matrix, ir_accessor->shift_vector);
            auto ir_write_accessor = ir::Write::Create(ir_local_accessor, ir_global_accessor);
            ir_global_accessor->simd_size = ir_accessor->simd_size;
            ir_global_accessor->simd_shuffle = ir_accessor->simd_shuffle;
            ir_store_parallel->values.push_back(ir_write_accessor);
            ir_store_parallel->parent_parallel = ir_grid;
            ir_grid->values.push_back(ir_store_parallel);

            ir_store_parallel->name = "store" + std::to_string(i);
        }

        ++i;
        for (auto accessor_iter = accessor_range.first; accessor_iter != accessor_range.second;
             ++accessor_iter) {
            auto ir_accessor = accessor_iter->second;
            ir_accessor->Tensor() = ir_local_tensor;
            ir_accessor->transform_matrix.leftCols(ir_grid->shape.size()).setZero();
        }

        ir_grid->values.push_front(ir_local_tensor);
    }
}

inline void LayerMemory2(std::shared_ptr<ir::OperatorInstance> ir_operator) {
    auto ir_outer_parallel = Cast<ir::Grid>(ir_operator->values.front());
    auto ir_inner_parallel = Cast<ir::Grid>(ir_outer_parallel->values.front());

    std::multimap<std::shared_ptr<ir::Tensor>, std::shared_ptr<ir::Accessor>>
        tensor_accessor_multimap;
    Each<ir::Accessor>(ir_inner_parallel, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        tensor_accessor_multimap.insert({ir_accessor->Tensor(), ir_accessor});
    });

    std::map<std::shared_ptr<ir::Accessor>, std::shared_ptr<ir::Tensor>> accessor_local_tensor_map;

    int64_t i = 0;
    for (auto iter = tensor_accessor_multimap.begin(); iter != tensor_accessor_multimap.end();
         iter = tensor_accessor_multimap.upper_bound(iter->first)) {
        auto ir_tensor = iter->first;

        auto accessor_range = tensor_accessor_multimap.equal_range(ir_tensor);
        bool is_readed = false;
        bool is_written = false;
        for (auto accessor_iter = accessor_range.first; accessor_iter != accessor_range.second;
             ++accessor_iter) {
            auto ir_accessor = accessor_iter->second;
            GALOIS_VERIFY(ir_accessor->transform_matrix ==
                          accessor_range.first->second->transform_matrix);
            GALOIS_VERIFY(ir_accessor->shift_vector == accessor_range.first->second->shift_vector);

            if (ir_accessor->IsReaded()) {
                is_readed = true;
            }
            if (ir_accessor->IsWritten()) {
                is_written = true;
            }
        }

        if (is_written) continue;

        auto ir_accessor_tmp = accessor_range.first->second;
        Eigen::VectorXi64 local_tensor_type_shape =
            ir_accessor_tmp->transform_matrix.rightCols(ir_accessor_tmp->transform_matrix.cols() -
                                                        ir_outer_parallel->shape.size()) *
            ir_inner_parallel->shape;
        Eigen::VectorXi64 local_tensor_shape =
            (ir_accessor_tmp->transform_matrix.leftCols(ir_outer_parallel->shape.size()) *
             ir_outer_parallel->shape)
                .eval();
        local_tensor_shape.array() /= local_tensor_type_shape.array();

        ir::Layout layout = ir::Layout::RowMajor;
        if (local_tensor_type_shape[0] == 4) {
            layout = ir::Layout::ColumnMajor;
        }

        // Copy memory to local tensor
        auto ir_local_tensor_type = ir::TensorType::Create(
            ir_accessor_tmp->Tensor()->type->value_type, local_tensor_type_shape, layout);
        auto ir_local_tensor = ir::CreateTensor(ir_local_tensor_type, local_tensor_shape);

        // TODO: 后面需要调整, 目前仅支持accessor一样的
        auto ir_accessor = accessor_range.first->second;
        auto ir_copy_parallel_outer = ir::Grid::Create();
        ir_copy_parallel_outer->is_local = true;
        ir_copy_parallel_outer->shape = ir_outer_parallel->shape;
        ir_operator->values.push_front(ir_copy_parallel_outer);
        ir_operator->values.push_front(ir_local_tensor);
        Eigen::MatrixXi64 ir_accessor_outer_transform_matrix =
            Eigen::MatrixXi64::Zero(local_tensor_shape.size(), ir_outer_parallel->shape.size());
        for (int64_t i = 0; i < ir_accessor_outer_transform_matrix.rows(); ++i) {
            for (int64_t j = 0; j < ir_accessor_outer_transform_matrix.cols(); ++j) {
                if (ir_accessor->transform_matrix(i, j) != 0) {
                    ir_accessor_outer_transform_matrix(i, j) = 1;
                }
            }
        }
        auto ir_local_accessor_outer =
            ir::Accessor::Create(ir_local_tensor, ir_accessor_outer_transform_matrix,
                                 Eigen::VectorXi64::Zero(local_tensor_shape.size()));
        ir_copy_parallel_outer->values.push_back(ir_local_accessor_outer);

        auto ir_copy_parallel_inner = ir::Grid::Create();
        ir_copy_parallel_inner->is_local = false;
        ir_copy_parallel_inner->shape = ir_inner_parallel->shape;
        ir_copy_parallel_outer->values.push_back(ir_copy_parallel_inner);

        // auto local_accessor_a = ir_accessor->transform_matrix;
        // local_accessor_a.rightCols(ir_inner_parallel->shape.size()).setZero();
        // for (int64_t i = 0; i < )
        // local_accessor_a.leftCols(ir_grid->shape.size()).setZero();
        Eigen::MatrixXi64 ir_inner_accessor_transform_matrix = ir_accessor->transform_matrix;
        ir_inner_accessor_transform_matrix.leftCols(ir_outer_parallel->shape.size()).setZero();
        auto ir_local_accessor_inner =
            ir::Accessor::Create(ir_local_accessor_outer, ir_inner_accessor_transform_matrix,
                                 Eigen::VectorXi64::Zero(local_tensor_type_shape.size()));

        auto ir_global_accessor = ir::Accessor::Create(
            ir_accessor->Tensor(), ir_accessor->transform_matrix, ir_accessor->shift_vector);
        auto ir_write_accessor = ir::Write::Create(ir_global_accessor, ir_local_accessor_inner);
        ir_copy_parallel_inner->values.push_back(ir_write_accessor);

        for (auto accessor_iter = accessor_range.first; accessor_iter != accessor_range.second;
             ++accessor_iter) {
            auto ir_accessor = accessor_iter->second;
            Eigen::MatrixXi64 ir_accessor_outer_transform_matrix =
                Eigen::MatrixXi64::Zero(local_tensor_shape.size(), ir_outer_parallel->shape.size());
            for (int64_t i = 0; i < ir_accessor_outer_transform_matrix.rows(); ++i) {
                for (int64_t j = 0; j < ir_accessor_outer_transform_matrix.cols(); ++j) {
                    if (ir_accessor->transform_matrix(i, j) != 0) {
                        ir_accessor_outer_transform_matrix(i, j) = 1;
                    }
                }
            }
            auto ir_local_accessor_outer =
                ir::Accessor::Create(ir_local_tensor, ir_accessor_outer_transform_matrix,
                                     Eigen::VectorXi64::Zero(local_tensor_shape.size()));
            ir_accessor->Tensor() = ir_local_accessor_outer;
            ir_accessor->transform_matrix.leftCols(ir_outer_parallel->shape.size()) =
                ir_accessor_outer_transform_matrix;
        }
    }
}

inline bool IsUselessDim(std::shared_ptr<ir::Grid> ir_grid, int64_t dim_index) {
    bool useless = true;
    auto parent_dim_size = ir_grid->GetAffineDimSize() - ir_grid->shape.size();
    Each<ir::Accessor>(ir_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        useless =
            useless && ir_accessor->transform_matrix.col(parent_dim_size + dim_index).isZero();
    });

    return useless;
}

inline void RemoveDim(std::shared_ptr<ir::Grid> ir_grid, int64_t dim_index) {
    GALOIS_ASSERT(IsUselessDim(ir_grid, dim_index));
    auto parent_dim_size = ir_grid->GetAffineDimSize() - ir_grid->shape.size();
    Each<ir::Accessor>(ir_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
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

inline void Repeat(std::shared_ptr<ir::OperatorInstance> ir_operator, int64_t times) {
    auto ir_repeat_parallel = ir::Grid::Create();
    ir_repeat_parallel->shape.resize(1);
    ir_repeat_parallel->shape[0] = times;

    for (auto ir_value : ir_operator->values) {
        ir_repeat_parallel->values.push_back(ir_value);
    }

    ir_operator->values.clear();
    ir_operator->values.push_back(ir_repeat_parallel);
}

// inline void CombineElementwiseOperators(std::shared_ptr<ir::Model> ir_model) {
//     for (auto ir_operator : ir_model->operators) {
//     }
// }

}  // namespace galois::transform
