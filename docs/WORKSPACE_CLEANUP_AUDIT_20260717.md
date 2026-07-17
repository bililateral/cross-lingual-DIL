# 工作区清理审计（2026-07-17）

## 1. 清理原则

本次清理采用显式路径 allow-list，不使用 `reports/*`、日期通配符或按 Step 编号整目录删除。每个候选在删除前都经过路径边界、Git 跟踪状态、当前代码与 policy 引用、manifest/summary 输出关系和替代版本检查。

以下内容明确保留：原始数据；canonical Step3-Step5 输入与冻结标签；模型目录；当前 Step7/9/11 正式结果；Step12/13 统计与漂移审计；有论文消融价值的正式负结果；Step16H/16I 审查与数据完整性证据；Step21-Step25 正式结果、manifest 和复现代码。

## 2. 删除结果

共删除 `308` 个文件，释放 `178,167,287` bytes，约 `169.91 MiB`。

| 分组 | 文件数 | 删除原因 |
| --- | ---: | --- |
| `scripts/tests` Python cache | 36 | 可重复生成的解释器缓存 |
| Step9 `codexbak.context_mismatch` | 1 | 旧上下文回滚副本，不是当前 manifest 输入 |
| Step15-v8 Linux console logs | 3 | 已有正式 summary/manifest；日志不承担结果契约 |
| Step15-v8 V2 result bundle | 227 | 已被修正且哈希闭合的 V3 reprofix bundle 取代 |
| Step16-v8 V2 readiness freeze | 12 | V2 materialization/canonical-split 实现已失效，当前代码只认 V3 |
| Step23 invalid V2 result | 12 | 全量 cross-field redaction 误触发，已由冻结的 V2.1 纠正结果取代 |
| TUApps off-mainline probe summary | 1 | 不是暗网 marketplace seller-pair benchmark 输入 |
| Unreferenced Step11 threshold CSV | 16 | 不在任何当前 summary `output_paths`、allow-list 或 manifest 中 |

## 3. 同步修正

1. 删除 `.gitattributes` 中仅服务于 Step16-v8 V2 readiness 的 Git LFS 路径。
2. 将 V8 设计文档中的“正式 V2 输入”改为 manifest-bound V3 readiness run。
3. 在当前实验设计和项目进度中记录 Step25 已完成且 `d1_candidate_eligible=false`，避免继续把它描述为待运行方法。
4. 保留历史 Step11 `20260517` manifest/audit 及其引用输出，因为它们仍承担历史审计来源，不属于本次未引用 CSV。

## 4. 未删除但需注意的状态

当前 Windows 工作区仍缺少 Linux V3 reprofix bundle 所引用的独立 upstream readiness root。该缺失已在 `docs/PROJECT_PROGRESS.md` 中记录；本次没有用失效 V2 冒充 V3，也没有删除 V3 的正式结果 bundle。若未来需要从 Windows 完整复现 V8，应从 Linux 同步 V3 freeze manifest 指定的整个 readiness root，并按 SHA-256 校验。

大量历史正式结果仍保留在 `reports/` 中。这些文件可能不是当前主模型，但仍是论文负结果、消融、统计复核或数据谱系证据，不能仅因日期较早而删除。后续清理必须继续遵循 manifest-aware、explicit allow-list 原则。

## 5. 结论

本次清理没有改变标签、split、训练特征、模型参数、阈值或任何正式实验数值。被删除对象均为缓存、回滚/日志、已被纠正版本取代的无效 bundle、主线外探测摘要或没有任何活动引用的 Step11 阈值表。
