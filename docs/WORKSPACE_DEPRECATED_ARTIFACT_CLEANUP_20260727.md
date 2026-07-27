# 2026-07-27 废弃产物清理审计

## 范围与规则

本轮只删除已经被 Step28-v12/v12.1 完全替代、且不在任何当前 policy、manifest、测试或结果重放依赖闭包中的文件。文档、有效负结果、冻结输入、当前正式结果和历史可复现依赖全部保留。删除采用显式路径清单，不按 Step 编号或日期通配。

删除前还发现 `docs/PROJECT_PROGRESS.md` 在 v12.1 同步清单冻结后被追加内容，导致唯一一项 manifest 哈希漂移。该文件已恢复到提交 `ab4f512` 的冻结字节；后续 M0 定义仍完整保存在 `docs/AI_RESEARCH_HANDOFF_20260722_ADDENDUM.zh.md` 和 `docs/STEP7_STEP28_M0_ROLE_DEFINITION_20260724.zh.md`。

## 删除内容

共删除 19 个 Git 跟踪文件，`52,444,198` bytes（约 `50.01 MiB`）。首轮清理 `scripts/`、`tests/` 下 77 个 Python 缓存文件（`1,851,774` bytes）；全量回归测试后又清理其新生成的 68 个缓存文件（`1,804,940` bytes），最终剩余 `__pycache__` 目录为 0。

| 分组 | 文件数 | 删除原因 |
| --- | ---: | --- |
| Step28-v10 派生训练结果 | 5 | v12 已完整替代；v12 只依赖并哈希绑定 v10 的 `world_truth`、`synthetic_items` 和 `model_inputs` |
| Step28-v10 guarded application | 5 | 已被审计修复版及最终 v12.1 空队列应用替代，不属于当前同步清单 |
| Step28-v10.1 guarded application | 7 | 已被 v12.1 完全替代，不属于当前同步清单 |
| 两个 v10 application policy | 2 | 只服务上述已退役应用；v12.1 policy 为完整独立合同，不继承它们 |
| Python `__pycache__` | 首轮 77；验收后 68 | 本地可再生缓存，不是科研产物；最终目录数为 0 |

删除的 v10 派生训练文件为：

```text
reports/step28_transferable_identity_history/v10_20260720/step28_v10_generation_summary.json
reports/step28_transferable_identity_history/v10_20260720/step28_v10_model_artifacts.json
reports/step28_transferable_identity_history/v10_20260720/step28_v10_parsed_occurrences.csv
reports/step28_transferable_identity_history/v10_20260720/step28_v10_synthetic_predictions.csv
reports/step28_transferable_identity_history/v10_20260720/step28_v10_training_summary.json
```

两个应用结果目录和 policy 为：

```text
reports/step28_transferable_identity_history/v10_guarded_application_20260720/
reports/step28_transferable_identity_history/v10_1_guarded_application_20260720/
schema/step28_transferable_identity_history_v10_guarded_application_policy.json
schema/step28_transferable_identity_history_v10_1_guarded_application_policy.json
```

所有被删文件仍可从本次清理前的 Git 历史恢复；科研结论继续由 Step28 文档保留。

## 明确保留

- Step28-v12/v12.1 全部结果、policy、自审和同步清单。
- v6–v11 的 17 个 `world_truth`、`synthetic_items`、`model_inputs` 最小历史文件。它们虽然来自失败沿革，但被 v12.1 的 61-file 同步清单逐文件哈希绑定，用于跨版本身份与状态审计，不能删除。
- `schema/step28_transferable_identity_history_v4_1…v11_policy.json` 与 `tests/test_step28_v11_application_contracts.py`。当前 v12 自审和通用回归测试仍以它们重放历史生成合同；v11 测试已明确跳过，不会冒充当前结果。
- Step7-v3.1、v4、v4.1 的正式负结果和中间 GPU 特征。v4.1/v4.2 仍逐文件依赖这些冻结输入。
- Step15、Step24、Step27 等有效历史负结果、消融、统计审计和当前 manifest 输入。旧不等于废弃，不能只按日期删除。

保留的历史 v10 生成 policy 仍声明上述五个派生输出的文件名，这是历史合同的一部分；当前 v12/v12.1 不读取这些派生输出。删除后的全量回归共执行 269 项测试，263 项通过、6 项为已撤销 v11 的预期跳过，未发现现行代码或结果重放依赖缺失。

## 不改变的科研事实

清理不改变任何标签、split、模型参数、阈值、预测或正式指标。Step28 当前仍是 v12 合成机制修正复现通过、v12.1 真实应用正向队列为 0；Step7-v4.2 仍未选出稳定 M0。
