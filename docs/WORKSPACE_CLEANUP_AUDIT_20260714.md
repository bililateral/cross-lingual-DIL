# 工作区清理审计（2026-07-14）

## 1. 审计范围与原则

本次只审计 `docs/`、`scripts/`、`schema/`、`reports/`。判断标准不是“当前 runner 不读取就删除”，而是：

1. 临时备份、dry-run、失败日志且已有正式产物替代，可删除；
2. 已明确拒绝进入暗网 seller-pair 主线的外部普通 spam 数据，只保留结论时可删除原始探测缓存；
3. 旧方法、负结果、消融、冻结清单、论文对照和数据构建谱系即使不再运行，也必须保留；
4. 任何被当前 manifest、policy、progress 文档或可复现实验链引用的文件，在解除引用前不能删除；
5. 本文档只列出候选，不在本次 v7 方法修改中自动删除用户已有科研产物。

## 2. 可直接删除的未跟踪临时文件

### 2.1 Step9 上下文错配自动备份

以下 7 个 JSON 是旧 summary 在输入上下文变化时生成的回滚副本，不是任何当前 manifest 的输入：

- `reports/step15_v6/baselines/step9/step9_few_shot_summary.codexbak.context_mismatch.20260713-112752.json`
- `reports/step9_few_shot_summary.codexbak.context_mismatch.20260704-103530.json`
- `reports/step9_few_shot_summary.codexbak.context_mismatch.20260705-172251.json`
- `reports/step9_few_shot_summary.codexbak.context_mismatch.20260705-205027.json`
- `reports/step9_few_shot_summary.codexbak.context_mismatch.20260709-153916.json`
- `reports/step9_few_shot_summary.codexbak.context_mismatch.20260710-185345.json`
- `reports/step9_few_shot_summary.codexbak.context_mismatch.20260710-193915.json`

### 2.2 Step16D/16E dry-run 产物

这些文件只记录未应用的预演，不是当前 Step5 freeze 或 v6/v7 输入：

- `reports/step16d_relaxed_silver_positive_topup_candidates.dry_run.csv`
- `reports/step16d_relaxed_silver_positive_topup_summary.dry_run.json`
- `reports/step16d_relaxed_silver_positive_topup_training_pairs.dry_run.csv`
- `reports/step16e_relaxed_silver_negative_balance_candidates.dry_run.csv`
- `reports/step16e_relaxed_silver_negative_balance_summary.dry_run.json`
- `reports/step16e_relaxed_silver_negative_balance_training_pairs.dry_run.csv`

## 3. 已拒绝进入主线，建议整组删除或移出仓库

### 3.1 外部 SpamDam 探测仓库

- 目录：`reports/external_source_probe/`
- 静态统计：57 个文件，约 21.6 MB；包含嵌套 `.git/`。
- 原因：这是普通短信/微博 spam 数据探测缓存，不是中文暗网 marketplace seller-pair 数据，不能补充主任务 ground truth，也不应随科研主仓库同步。

### 3.2 Step19 普通 spam/Weibo 派生数据

根目录共有 21 个、约 1.54 MB 的未跟踪文件，匹配以下前缀：

- `reports/step19_external_chinese_spam_*`
- `reports/step19_external_spamdam_weibo_*`
- `reports/step19_chinese_sockpuppet_data_search_audit_20260704.csv`
- `reports/step19_public_chinese_external_identity_sources_20260704.csv`
- `reports/step19_public_collection_source_leads_20260704.csv`

这些文件可作为“外部数据不满足主线 provenance”的一次性调查记录，但不应作为训练、验证、测试或 prospective holdout 输入。若要保留失败调查结论，只保留一份简短审计说明即可，原始派生 CSV/JSON 可以删除或移到仓库外归档。

## 4. 需先解除文档引用或确认恢复价值，再删除

### 4.1 大型 Step5 队列回滚副本

- `reports/step5_zh_target_strict_balanced_review_queue.codexbak.step16_positive_salvage_apply.20260704-121655.csv`

它未被 Git 跟踪，也不被当前 freeze 读取；但它是一次标签队列修改前的恢复点。当前 Step5 frozen labels、Step16F/16H 审计和 v6 manifest 均完整后，可移到仓库外冷归档，不建议在未确认备份策略前直接删除。

### 4.2 Step16G 完整运行日志

- `reports/step16g_full_rerun_20260710.log`

该日志未跟踪，但 `docs/PROJECT_PROGRESS.md` 仍引用它作为当时 `[10/10]` 完成证据。正式 summary/manifest 已足以复现结果时，可先把进度文档改为引用正式 manifest，再删除日志。

### 4.3 Step11 archive dry-run

- `reports/step11_archive_dry_run_20260517.csv`
- `reports/step11_archive_dry_run_20260517.json`

二者由 Git 跟踪，且 `docs/CURRENT_EXPERIMENT_DESIGN.md` 明确说明其管理用途。它们不参与当前 manifest-only audit，但删除前应同步更新文档并形成单独 commit；不应与 v7 方法改动混在同一提交中。

## 5. 明确保留，不属于垃圾文件

以下内容即使不是当前 v7 执行入口，也不能删除：

- `reports/step15_v6/` 与 `reports/step12_v6/`：v6 严格负结果、模型产物和统计证据，被 negative freeze manifest 哈希绑定；
- `reports/step16h_blind_review/`：v6 gold/silver sensitivity 与双审证据；
- v5/v5r/v6 的 scripts、schema、docs：论文消融、失败诊断和结果复现所需；
- Step7/Step9/Step11 历史正式 summary/predictions：除非先完成 manifest 引用闭包审计并迁移到版本化归档；
- 多版 Step13 文档：分别对应不同数据冻结边界，是历史概念漂移审计，不等同于临时快照；
- `docs/PROJECT_PROGRESS.md`、`CURRENT_EXPERIMENT_DESIGN.md`、`RESEARCH_PLAN.md`：前两者包含当前与历史边界，后者保存完整科研决策谱系；
- `schema/step5_v2_milestone_snapshot_policy.json` 和 `scripts/step5_snapshot_milestone.py`：名称含 snapshot，但属于可复现实验工具，不是生成的临时快照目录。

## 6. 审计结论

`docs/`、`scripts/`、`schema/` 中没有发现可以在不损伤复现性的前提下直接删除的临时代码或配置。明确的清理收益集中在 `reports/`：8 个 `codexbak` 共约 86.2 MB、9 个 dry-run/log/tmp 共约 16.9 MB、21 个被拒绝的 Step19 文件约 1.54 MB、外部探测仓库约 21.6 MB。部分分组包含需条件确认的恢复点或日志，因此不应一次性无差别删除。

推荐分两次提交：

1. 先删除第 2、3 节明确的未跟踪临时/拒绝数据；
2. 再单独处理第 4 节，先更新文档引用或完成仓库外冷归档，再删除 tracked 管理文件。
