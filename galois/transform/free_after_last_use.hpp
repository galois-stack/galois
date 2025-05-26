#pragma once

#include <unordered_map>
#include <unordered_set>

#include "galois/ir/tensor.hpp"
#include "galois/transform/common.hpp"

namespace galois::transform {

/// @brief 在变量最后一次被使用后立刻插入Free指令的优化
class FreeAfterLastUse {
   public:
    static void Create(const std::shared_ptr<galois::ir::Block>& ir_block) {
        auto NeedFree = [](const std::shared_ptr<galois::ir::Tensor>& ir_tensor) -> bool {
            return Cast<galois::ir::Alloca>(ir_tensor) || Cast<galois::ir::Call>(ir_tensor);
        };

        // 1. 收集所有候选变量
        std::vector<std::shared_ptr<galois::ir::Tensor>> ir_candidates;
        std::unordered_set<std::shared_ptr<galois::ir::Tensor>> ir_operator_defs;
        for (auto it = ir_block->begin(); it != ir_block->end(); ++it) {
            auto ir_tensor = *it;
            if (Cast<galois::ir::Return>(ir_tensor)) continue;
            if (Cast<galois::ir::Input>(ir_tensor)) continue;
            if (Cast<galois::ir::Block>(ir_tensor) || Cast<galois::ir::Operator>(ir_tensor) ||
                Cast<galois::ir::Grid>(ir_tensor)) {
                ir_operator_defs.insert(ir_tensor);
                continue;
            }
            ir_candidates.push_back(ir_tensor);
        }

        // 2. 找到 Return 指令以及其操作数
        std::list<std::shared_ptr<galois::ir::Tensor>>::iterator ir_return_it = ir_block->end();
        std::unordered_set<std::shared_ptr<galois::ir::Tensor>> ir_return_tensors;
        for (auto it = ir_block->begin(); it != ir_block->end(); ++it) {
            if (auto ir_ret = Cast<galois::ir::Return>(*it)) {
                ir_return_it = it;
                if (ir_ret->OperandSize() > 0) {
                    auto ir_ret_tensor = ir_ret->GetOperand(0);
                    if (ir_ret_tensor) ir_return_tensors.insert(ir_ret_tensor);
                }
                break;
            }
        }

        // 3. 记录需要插入 Free 的信息
        struct FreeRecord {
            std::shared_ptr<galois::ir::Tensor> ir_tensor;
            std::list<std::shared_ptr<galois::ir::Tensor>>::iterator insert_after;
        };
        std::vector<FreeRecord> ir_to_free;

        for (auto ir_tensor : ir_candidates) {
            if (!NeedFree(ir_tensor)) continue;
            if (ir_return_tensors.count(ir_tensor)) continue;
            if (ir_operator_defs.count(ir_tensor)) continue;
            // 利用找最后一次使用的指令
            auto& use_list = ir_tensor->instruction_with_index_list;
            if (use_list.empty()) {
                // 没有被使用，直接在定义后插入
                auto it = std::find(ir_block->begin(), ir_block->end(), ir_tensor);
                if (it != ir_block->end()) {
                    ir_to_free.push_back({ir_tensor, it});
                }
            } else {
                // 找到最后一次使用的指令
                auto last = use_list.back();
                auto use_inst = last.instruction.lock();
                auto last_use_it = std::find(ir_block->begin(), ir_block->end(), use_inst);
                if (last_use_it != ir_block->end()) {
                    ir_to_free.push_back({ir_tensor, last_use_it});
                } else {
                    // 若没找到，就在定义后插入
                    auto it = std::find(ir_block->begin(), ir_block->end(), ir_tensor);
                    if (it != ir_block->end()) {
                        ir_to_free.push_back({ir_tensor, it});
                    }
                }
            }
        }

        // 4. 插入 Free 指令
        std::sort(ir_to_free.begin(), ir_to_free.end(),
                  [&](const FreeRecord& a, const FreeRecord& b) {
                      return std::distance(ir_block->begin(), a.insert_after) >
                             std::distance(ir_block->begin(), b.insert_after);
                  });

        for (const auto& rec : ir_to_free) {
            auto next_it = std::next(rec.insert_after);
            if (next_it != ir_block->end() && Cast<galois::ir::Return>(*next_it)) {
                ir_block->insert(next_it, galois::ir::Free::Create(rec.ir_tensor));
            } else {
                ir_block->insert(next_it, galois::ir::Free::Create(rec.ir_tensor));
            }
        }

        // 5. 递归处理 ir_block/operator/grid
        Each<galois::ir::Block>(ir_block, [ir_block](std::shared_ptr<galois::ir::Block> sub_block) {
            if (sub_block != ir_block) {
                FreeAfterLastUse::Create(sub_block);
            }
        });
        Each<galois::ir::Operator>(ir_block, [](std::shared_ptr<galois::ir::Operator> ir_op) {
            if (ir_op->block) {
                FreeAfterLastUse::Create(ir_op->block);
            }
        });
        Each<galois::ir::Grid>(ir_block, [](std::shared_ptr<galois::ir::Grid> ir_grid) {
            if (ir_grid->block) {
                FreeAfterLastUse::Create(ir_grid->block);
            }
        });
    }
};

}  // namespace galois::transform
