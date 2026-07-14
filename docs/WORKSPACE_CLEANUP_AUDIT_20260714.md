# 工作区清理完成记录（2026-07-14）

## 1. 清理范围与边界

本次只清理 `reports/` 中已经确认过期、未被当前 manifest/policy/runner 使用，或已明确拒绝进入暗网 seller-pair 科研主线的产物。删除前逐项完成了：

1. 精确路径存在性与大小检查；
2. 解析后路径必须位于当前工作区的 `reports/` 内；
3. Git 跟踪状态检查；
4. `docs/`、`scripts/`、`schema/` 的引用闭包检查；
5. 对仍有文档引用的文件，先确定正式替代证据，再同步修改引用。

没有使用 `reports/*` 之类的宽泛删除规则，也没有删除当前训练输入、冻结标签、模型、正式 summary、预测、manifest、Step12 统计结果或 Step16H 审核证据。

## 2. 已删除内容

共删除 `95` 个文件，释放 `126,239,312` bytes，约 `120.4 MiB`：

| 分组 | 文件数 | 删除原因 |
| --- | ---: | --- |
| Step9 `codexbak.context_mismatch` 回滚副本 | 7 | 输入上下文变化时生成的旧 summary 备份；不被当前 manifest 使用 |
| Step16D/16E `dry_run` 产物 | 6 | 未应用的预演结果；不是当前 Step5 freeze 或 v6/v7 输入 |
| `reports/external_source_probe/` | 57 | 普通 spam/Weibo 外部仓库探测缓存，含嵌套 `.git/`；不属于暗网 marketplace seller-pair 数据 |
| Step19 spam/Weibo 派生文件 | 21 | 已判定 provenance 与科研主线不匹配，不能进入训练、验证、测试或 prospective holdout |
| Step5 review-queue 回滚副本 | 1 | 当前 freeze、Step16F/16H 和 v6/v7 manifest 已完整；该未跟踪恢复点不再被读取 |
| Step16G 完整控制台日志 | 1 | 正式 summary、结果 bundle 与哈希已提供持久证据，控制台日志不再承担复现契约 |
| Step11 archive dry-run 清单 | 2 | 历史归档预演，不参与 manifest-only audit；删除前已解除设计文档引用 |

删除的两个 Step11 archive dry-run 文件是本批次中仅有的 Git tracked 文件：

- `reports/step11_archive_dry_run_20260517.csv`
- `reports/step11_archive_dry_run_20260517.json`

其余 `93` 个文件均未受 Git 跟踪；删除后通过本文档保留清理范围、原因和数量记录。

## 3. 引用迁移

### 3.1 Step16G

`docs/PROJECT_PROGRESS.md` 不再依赖 `reports/step16g_full_rerun_20260710.log` 证明完成状态。持久证据改为：

- `reports/step16g_hard_negative_imbalance_summary.json`；
- 同步后的 Step9、Step15、Step12、Step13 正式结果 bundle；
- Step16G summary 中记录的输入哈希、安全检查和 expanded-freeze SHA-256。

### 3.2 Step11

`docs/CURRENT_EXPERIMENT_DESIGN.md` 已明确记录 archive dry-run 清单退役。当前规则保持不变：

- 论文审计必须使用显式 allow-list 或受控 manifest；
- 每个 summary 只能通过自身 `output_paths` 解析 CSV；
- 禁止 glob 整个 `reports/` 作为论文结果输入。

### 3.3 Step5 回滚副本

被删除的 Step5 review-queue 回滚副本没有活跃代码、policy 或进度文档引用。当前 frozen labels、Step16F/16H 审核和后续 manifest 均未依赖它。

## 4. 明确保留的科研产物

以下内容经过复核，仍具有复现、负结果或论文消融价值，未被本次清理触碰：

- `reports/step15_v6/`、`reports/step12_v6/` 及其 negative-freeze/hash-closed 证据；
- `reports/step16h_blind_review/` 双审与裁决证据；
- v5/v5r/v6/v7 的正式 scripts、schema、docs 和 manifest；
- Step7/Step9/Step11 的正式 summary、预测、模型和显式审计结果；
- 不同冻结边界对应的 Step13 概念漂移审计；
- 当前 Step5 frozen labels、Step16G 正式 summary/training-pair audit 和 v7/Step20 输入谱系。

## 5. 清理结论

本次删除的是临时备份、未应用 dry-run、一次性控制台日志、历史 archive 预演，以及已经确认不适合主线的普通 spam/Weibo 探测数据。清理没有改变任何有效数据 split、标签、特征、模型配置、统计结果或当前科研结论；它只缩小工作区噪声并移除容易被误当成现役输入的旧文件。
