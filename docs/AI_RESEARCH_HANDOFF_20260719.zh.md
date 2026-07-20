# 跨语言暗网马甲识别科研项目 AI 交接文档

更新时间：2026-07-19

> **2026-07-20 后续事实更新**：Step28 最终复核已撤销 v5、v6/v6.1 和 v11。v11 的阻断问题是使用 audit 标签删除 49 个冲突状态、把模型识别与后置保护层弃权混为一种“通过”，并保留 3 个全零特征。当前有效阶段必须分开表述：v12 是同一预定义合成生成器家族内的修正复现实验；v12.1 只是冻结 v12 后的独立现有数据应用。v12 主审计保留全部 1,280 行并让 842 个状态各占总权重 1，完整历史 AUC/AP 为 `0.749634/0.767197`，相对直接历史 AP 增益 `+0.073928`，199 次分块置乱均值 `0.498092`、经验 `p=0.005`。它不证明真实准确率或所有未见状态泛化。v12.1 排除全部 `1,259` 个历史已审核 pair UID 后评分 `2,689` 条未审核候选；`101` 条非零修正全部为负，正修正和盲审队列均为 `0`。该空结果与 v12 合成通过互不混淆。当前结论为 `PASS_SYNTHETIC_REPLICATION_REAL_APPLICATION_ABSTENTION`。详见 `docs/STEP28_TRANSFERABLE_IDENTITY_HISTORY_V12_CORRECTED_REPLICATION_20260720.zh.md`。本轮用户明确授权无 GPU 的 Step28 CPU 实验在 Windows 执行，因此下文 Windows/Linux常规职责边界对本轮被该明确授权覆盖；其他需要模型编码或 GPU 的实验仍遵守原边界。

当前分支：`method/step27-english-pretrained-synthetic-adaptation`

Step27-v1.1 实现基线提交：`ff4bc04 Repair Step27 frozen-source replay contracts`

本文用途：让新的 AI 在不依赖旧对话的情况下，准确掌握科研主线、数据边界、历史实验结论、当前代码状态、不可违反的科研约束以及下一步操作。

---

## 1. 新 AI 首先必须知道的结论

1. 本项目研究的是跨语言地下市场 seller-pair 马甲识别，不是普通文本相似度分类。
2. 核心问题是：只在英文身份对上学习的能力，能否迁移到中文 seller-pair，并通过少量中文训练支持进一步改善。
3. 中文目标域存在严重概念漂移：高文本相似经常表示模板复用、同类商品或公共数据，而不表示同一控制者。
4. 当前真正的瓶颈不是模型规模，而是中文强身份正例稀缺、训练正例中 silver/soft supervision 占比过高，以及公共联系方式/网址噪声难以区分。
5. 历史上多个高分来自不同数据边界、训练集 D0 或反复查看过的开发测试，不能直接横向比较，也不能当作最终论文结果。
6. Step27-v1 的数值结果因输入契约漂移被判定为工程无效，不是科学负结果。
7. 当前最新方法是 Step27-v1.1。代码已修复并通过静态审查，但尚未在 Linux 完成正式数值运行。
8. Windows 只负责读代码、修改、审计、静态测试和同步文件；模型编码、训练和统计数值运行只在 Linux 服务器进行。
9. 禁止把 synthetic、silver、closure-derived 或模型预测结果伪装成 benchmark gold truth。
10. 当前 valid 和 200 条 internal test 都已被历史开发消费。Step27-v1.1 不得重新打开它们做晋级判断。

---

## 2. 论文科研主线

### 2.1 最终任务

输入是两个地下市场 seller 的 pair-level 表征，输出是：

```text
P(same_controller)
```

即判断两个 seller account 是否由同一现实控制者经营。

最终系统仍然可以在 Step11/17 中把高可信 pair edge 组成图并形成 cluster，但图聚类只是发现与审计层，不是 ground truth 生成器。

### 2.2 跨语言逻辑

主线应始终保持：

1. 在英文 source domain 上训练身份识别能力。
2. 不使用中文训练标签，直接在中文上评估 zero-shot transfer。
3. 在冻结英文来源能力的基础上，加入受控中文 train 支持进行 target-domain adaptation。
4. 研究模板复用、主题相似和公共联系方式噪声造成的 evidence-type concept drift。
5. 通过 seller-component 隔离、固定边界和 prospective holdout 检验方法是否真实泛化。

如果某个后续模型完全抛弃英文来源能力、只在中文 train 上重新拟合，它只能算 target-domain baseline，不能作为跨语言迁移主方法。

### 2.3 可以支持的论文贡献方向

当前较稳妥的论文叙事不是“已经获得极高准确率”，而是：

- 构建具有严格证据分层和 seller-component 隔离的跨语言地下市场身份链接评估框架；
- 定量展示 semantic/template/contact evidence 在英文与中文之间发生的条件漂移；
- 证明若不设置 duplication、source-only、hard-negative 和 evidence-reliability controls，少样本适配容易产生方法论幻觉；
- 检验冻结英文来源模型下的受控中文半合成残差适配是否具有超出重复加权的训练价值；
- 若 Step27 仍失败，形成严格负结果：在独立正身份成分稀少时，文本视图增强不能替代新的身份信息。

---

## 3. Windows 与 Linux 的职责边界

### 3.1 Windows 端

允许：

- 阅读和修改 `docs/`、`schema/`、`scripts/`、`tests/`；
- Python compilation、AST、JSON/CSV 静态解析；
- 不触发模型推理的 contract/unit tests；
- Git 分支、提交和代码审计；
- 检查 Linux 同步回来的 summary、prediction、manifest 和 SHA-256。

禁止：

- 在 Windows 运行真实 embedding、模型训练、Step12 大规模 bootstrap 或正式数值实验；
- 因 Windows 模型目录不同而降低模型指纹检查；
- 用本地临时结果覆盖 Linux 正式结果。

### 3.2 Linux 端

Linux 是唯一正式数值运行环境，路径通常为：

```text
/home/yongpeng/cross-lingual
```

模型、CUDA、PyTorch、Transformers 等运行环境已在 Linux 配置。Windows 修改后的代码由用户手动同步到 Linux，Linux 无法稳定连接 GitHub，因此不要要求服务器执行 `git fetch origin`。

---

## 4. 当前冻结数据边界

### 4.1 英文来源训练集

Step27 冻结的 Step24 E5 LR/L2 source scorer 只使用英文：

| Split | Rows | Positive | Negative |
|---|---:|---:|---:|
| English source train | 401 | 116 | 285 |

该来源模型只有一个输入特征：

```text
identifier_redacted_e5_cosine
```

其参数、标准化统计、训练 pair UID hash 和 artifact hash 全部冻结。

### 4.2 中文 canonical 边界

| Split | Rows | Positive | Negative | 角色 |
|---|---:|---:|---:|---|
| train | 573 | 229 | 344 | 目标域适配与 grouped OOF |
| valid | 120 | 30 | 90 | 已消费的 representative development valid |
| internal test | 200 | 50 | 150 | 已消费的回顾性内部诊断集 |

三个 split 必须保持 seller-component overlap 为零。

### 4.3 中文训练正例的证据构成风险

当前 229 个 train positive 大致包括：

| Evidence type | Count |
|---|---:|
| direct identifier | 57 |
| component anchor | 29 |
| style/structural soft | 143 |

历史审计指出 canonical train 中约 `213/229` positive 带有 train-only silver 属性。它们可以作为低权重训练支持，但不能作为 benchmark gold，也不能进入 valid/test。

这解释了为什么一些模型在中文 train D0 或 train OOF 上表现较高，却无法稳定迁移到 corrected valid/test。

### 4.4 当前关键切片仍然不足

对 Step27 的严格 canonical 边界：

| Slice | valid | internal test |
|---|---:|---:|
| direct positive | 4 | 21 |
| component-anchor positive | 0 | 1 |
| public-contact/URL negative | 3 | 6 |

因此 direct/component/public-noise 的 valid/test 切片没有足够统计功效。任何 slice FPR/recall 只能作为 fail-closed 诊断，不能作为充分论文证据。

---

## 5. 标签与证据纪律

### 5.1 身份标签

```text
same_controller
different_controller
uncertain
```

`uncertain` 不得进入二分类指标。

### 5.2 正例证据强度

优先级从高到低：

1. 双侧 seller-facing direct identifier：PGP、Telegram、Jabber、email、钱包、唯一别名闭合等。
2. 经独立身份锚点支持的 seller component 传递证据。
3. style/structural soft positive，只能作为低置信训练支持或敏感性分析。

文本相似、同类商品、模板复用、公共 URL、victim/product data contact 不能单独证明 same controller。

### 5.3 不允许的标签回流

- Step11/17 cluster 只能生成 review candidate，不能直接变成 Step5 truth。
- synthetic row 不能写入 Step5 frozen labels。
- silver row 不能进入 benchmark valid/test。
- 模型高分不能作为 positive 标签依据。
- Agent/Codex 审核必须披露为 agent-assisted internal review，不能冒充独立人类双盲标注。

---

## 6. 从 Step1 到 Step27 的流程地图

### Step1-Step5：数据与监督基础

- Step1：定义 schema、字段和审计契约。
- Step2：语言/来源 split 与泄漏隔离。
- Step3：seller profile、item text、联系方式和 identity occurrence 抽取。
- Step4：构建 seller-pair candidate universe。
- Step5：人工/规则证据审查，冻结 positive/negative/uncertain 监督边界。

### Step7-Step13：跨语言建模与审计

- Step7：英文 source-only baseline、pair features 和中文 zero-shot evaluation。
- Step8：zero-shot transfer 结果整理。
- Step9：中文 support-ratio adaptation、LR/L2、residual 和历史 positive-pair mixup controls。
- Step10：模型与特征消融。
- Step11：中文 candidate graph、relation pruning 和 cluster discovery。
- Step12：grouped bootstrap、paired comparison、permutation 和晋级门槛。
- Step13：英文/中文 feature drift、conditional drift 和 slice audit。

### Step15-Step20：困难负例、证据融合和评估边界

- Step15 v2-v5：逐阶段加入 topic/template/public-noise hard negatives 的 curriculum 诊断。
- Step15 v7：identifier-redacted clean scorer，两阶段 evidence fusion。
- Step15 v8：occurrence-level contextual evidence expert；corrected internal test 有诊断改善，但 public noise 未解决，不能作为最终论文结果。
- Step16：证据候选挖掘、blind review、数据完整性和 readiness。
- Step17：relation reliability/noisy-edge filtering，只是图后处理，不提高 pair scorer 本身的 ROC-AUC/AP。
- Step18：曾尝试 final holdout，但正例过少，不能支撑最终结论。
- Step19：外部中文数据搜索；多数外部 spam/Telegram 数据与地下市场 seller-pair 主线不匹配，已降级或清理。
- Step20：未来 prospective holdout 的准备与冻结层。当前没有授权 Step27 直接使用旧 Step20 freeze 做确认性评估。

### Step21-Step27：合成、表示和冻结来源迁移

- Step21：文本变换 augmentation；相对 equal-weight duplication 基本为零增益。
- Step22：同 seller item-disjoint split augmentation；未超过 duplication control。
- Step23：item-level multi-instance distribution；放大主题/模板相似，冻结为负结果。
- Step24：content-independent authorship/style 表征；在 train D0 有信号，但在独立 valid 上不稳且被模板污染。
- Step25：template decontamination/copy-aware dual channel；严格收敛后仍为负结果。
- Step26：将冻结 Step24 English source scorer 盲重放到 corrected valid/test；预注册 gate 失败。
- Step27：冻结英文 E5 source scorer，研究中文半合成视图是否超过同有效权重 duplication control。当前主工作就在这里。

---

## 7. 关键历史结果及其正确解释

### 7.1 Step24

在中文 canonical train D0 上，source-only semantic+style 曾达到：

```text
AP = 0.802718
```

但 Step26 将同一冻结 artifact 应用于 corrected representative valid 后：

```text
ROC-AUC = 0.736296
AP      = 0.508495
PR-AUC  = 0.498554
```

matched Step15-v8 clean valid 为：

```text
ROC-AUC = 0.754074
AP      = 0.574855
PR-AUC  = 0.569806
```

Step24 primary 相对 v8 clean 的 valid AP delta 是 `-0.066360`，因此 Step26B 被阻断。结论是：跨语言 style signal 存在，但不稳定，并被 template/public-format similarity 污染。

### 7.2 Step25-v3.1

全部 constrained solver 在严格 KKT tolerance 下收敛，但主要方法只通过 `2/11` gates，伤害 direct/component positive，并恶化 template negative。它是有效的严格负结果，不能继续调同一 D0。

### 7.3 Step15-v8 contextual evidence

Step26 审计记录的 internal test 诊断：

```text
Step15-v8 clean       AP = 0.544139
contextual fusion     AP = 0.620525
```

这说明 occurrence-level direct evidence uplift 有价值，但 public-noise score 基本未改变，六条 public-noise negative 仍全部进入 top-50。不能声称 contextual fusion 已解决公共联系方式噪声。

### 7.4 关于历史 Step7/9/15 高分

项目经历过多次 Step5 refreeze、正负扩充、split 修正和 feature contract 变化。历史 `0.913/0.739` 等 Step15 数值属于旧 `zh_test=106` 边界，不能与当前 `573/120/200` 边界的结果直接比较，也不能据此选择当前模型。

---

## 8. 当前 Step27-v1.1 的实验设计

### 8.1 原 Step27-v1 为什么无效

Step15-v7/Step24 的真实 seller E5 输入是五个允许字段的非空值按固定顺序换行连接。Step27-v1 插入了人工 section headers，并重新编码了真实 seller，导致冻结 LR 实际接收到不同的 E5 feature distribution。

因此 v1 失败无法区分：

- augmentation 无效；
- 或输入表示被工程实现改变。

v1 必须冻结为 invalid engineering run，而不是 scientific negative result。

### 8.2 v1.1 的模型角色

| ID | 定义 |
|---|---|
| S0 | 冻结英文 Step24 E5 LR/L2，零中文参数拟合 |
| M0 | 真实中文 train residual baseline |
| M1 | 与 M2 使用相同父样本、行数和有效权重的 duplication control |
| M2 | parent-preserving synthetic-view residual，主要方法候选 |

另外有 learned-alpha、alpha-zero 和 silver sensitivity diagnostics，但它们不能晋级主方法。

### 8.3 合成设计

- primary 使用 16 个 non-silver positive parent pairs，分布在 13 个重算 seller components；
- 匹配 16 个 reviewed negative parents；
- 每个 parent 生成两个不改变身份语义的视图；
- 每 seed primary child cap 固定为 64；
- 共 10 个稳定性 seeds：`20260320` 到 `20260329`；
- silver direct/contact sensitivity 物理隔离，不能满足 primary gate；
- M1 与 M2 必须具有相同 parent set 和 effective-weight budget。

允许的变换只包括 section/segment rotation 和 layout punctuation normalization。禁止跨 parent 拼接、生成新身份、复制 identifier 或改变标签语义。

### 8.4 统计门槛

M2 在 train seller-component-grouped OOF 至少要满足：

1. `AP(M2)-AP(M1) >= 0.02`；
2. `AP(M2)-AP(M0) > 0`；
3. 10 seeds 中至少 8 个方向为正；
4. 相对 S0 的观察 AP 差和 component-bootstrap 95% 下界都不低于 `-0.01`；
5. direct/component recall 不显著下降；
6. template/public-noise FPR 不恶化；
7. synthetic shortcut/no-op/lineage 审计全部通过。

这里的 `-0.01` 是 v1.1 修复重放前冻结的工程诊断门槛，不是原 v1 的预注册门槛，也不是论文正式非劣效结论。

### 8.5 v1.1 的工程修复

- 真实 seller embedding 从冻结 Step15-v7 cache 按 UID 精确取子集，不重新编码；
- 真实 E5 pair cosine 必须在绝对误差 `5e-13` 内重放 Step24；
- Step24 policy、sync manifest、clean manifest、pair summary、中文 pair CSV 和 source artifact 独立哈希固定；
- semantic policy、v7 redaction policy 和两个 encoder producer 独立哈希固定；
- E5 model directory fingerprint 必须与冻结 v7 metadata 一致；
- feature manifest 必须匹配当前 producer/common/shared dependencies；
- variant no-op、child budget 不完整、UID namespace 混用、output manifest 不闭合均立即失败；
- 当前共享代码明确拒绝执行旧 v1 policy。

### 8.6 当前验证状态

Windows 已完成：

```text
47/47 Step27 contract tests PASS
8/8 config-only entry points PASS
Python compilation PASS
Bash syntax PASS
git diff --check PASS
```

未在 Windows 运行任何数值实验。

---

## 9. 当前 Git 和文件状态

### 9.1 当前分支与实现基线提交

```text
branch: method/step27-english-pretrained-synthetic-adaptation
Step27 implementation baseline: ff4bc04 Repair Step27 frozen-source replay contracts
```

远程分支在本交接文档编写时仍停在 `f592cdf`，所以 `ff4bc04` 尚未 push。交接文档可以位于后续 doc-only commit；接手时用 `git log -3 --oneline --decorate` 确认实际 HEAD。

### 9.2 `ff4bc04` 包含的 14 个文件

```text
docs/PROJECT_PROGRESS.md
docs/STEP27_ENGLISH_PRETRAINED_SYNTHETIC_ADAPTATION_PLAN_20260718.zh.md
docs/STEP27_V1_1_SOURCE_CONTRACT_REPAIR_20260719.zh.md
schema/step27_v1_1_exact_replay_policy.json
scripts/run_step27_v1_1_exact_replay_linux_20260719.sh
scripts/step12_step27_statistical_audit.py
scripts/step27_audit_synthetic_data.py
scripts/step27_build_pair_features.py
scripts/step27_build_sync_manifest.py
scripts/step27_common.py
scripts/step27_encode_profiles.py
scripts/step27_generate_train_only_views.py
scripts/step27_train_residual_models.py
tests/test_step27_english_pretrained_synthetic_contracts.py
```

### 9.3 工作区注意事项

`reports/` 中有大量历史同步结果和未纳入当前 commit 的变化。不要运行 `git reset --hard`、`git checkout --` 或批量清理来“恢复干净”，也不要把无关 reports 全部加入下一次提交。

当前已知与 Step27 提交无关的未跟踪文件：

```text
docs/STEP13_CONCEPT_DRIFT_AUDIT_STEP16C_REFREEZE_VALIDATION_20260709.md
```

不要误删或纳入 Step27 提交。

---

## 10. 新对话接手后的立即操作

### 10.1 先阅读这些文件

按顺序：

1. `docs/AI_RESEARCH_HANDOFF_20260719.zh.md`
2. `docs/PROJECT_PROGRESS.md`
3. `docs/STEP27_V1_1_SOURCE_CONTRACT_REPAIR_20260719.zh.md`
4. `docs/STEP27_ENGLISH_PRETRAINED_SYNTHETIC_ADAPTATION_PLAN_20260718.zh.md`
5. `schema/step27_v1_1_exact_replay_policy.json`
6. `scripts/run_step27_v1_1_exact_replay_linux_20260719.sh`

### 10.2 将 `ff4bc04` 的 14 个文件同步到 Linux

不要同步 Windows 当前 E5 模型目录。Windows 模型副本与冻结 v7 fingerprint 不一致。

冻结模型期望：

```text
file_count       = 9
total_size_bytes = 2261765101
files_sha256     = 74fb1ef756bd285a72c9f88297a7d027ff60c215be1c9d1f6969b3c70d19b2d6
```

Linux 如果不满足该指纹，必须恢复生成 Step15-v7 cache 时使用的精确模型快照，不能绕过检查。

### 10.3 Linux 一键运行

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step27_v1_1_exact_replay_linux_20260719.sh
```

runner 只运行 train OOF 工程诊断，不打开旧 valid/test。

### 10.4 应同步回 Windows 的唯一结果根目录

```text
reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719/
```

不要 glob 旧 Step27 目录，也不要把 v1 与 v1.1 summary 混合。

### 10.5 结果审查顺序

必须依次检查：

1. sync manifest 是否完整，文件数量/大小/SHA-256 是否全部一致；
2. exact real text/cache/pair cosine replay 是否通过；
3. E5 model/config/producer fingerprint 是否通过；
4. S0 是否复现：`ROC-AUC=0.7550015233065909`、`AP=0.6443826343928266`；
5. tokenizer truncation 和 synthetic feature displacement；
6. M1/M2 parent、child count、effective-weight 是否严格相等；
7. M2-M1、M2-M0、M2-S0 的 OOF AP 和 grouped bootstrap；
8. direct/component recall 与 template/public-noise FPR；
9. `technical_oof_gate_pass`，同时确认 `eligible_for_valid` 必须仍为 `false`。

即使技术 gate 通过，也不能用旧 valid/test 继续选择模型。

---

## 11. Step27 运行后的决策树

### 情况 A：工程重放失败

例如模型指纹、真实 cache、Step24 pair cosine、manifest 或 S0 复现失败。

结论：仍是工程失败。先修契约，不解释 M2 性能。

### 情况 B：工程重放通过，但 M2 不超过 M1

结论：当前半合成变换没有展示超出重复加权的训练信息。冻结为严格 train-OOF 负结果，不继续调当前 valid/test。

### 情况 C：M2 超过 M0，但不超过 M1

结论：增加这些 parent 的曝光/权重可能有用，但不能声称 synthetic transformation 有效。

### 情况 D：M2 同时超过 M1/M0，并满足 S0 和切片门槛

结论：只说明值得在新的冻结开发批次复现。下一步必须先冻结代码、policy、parent/fold/model/threshold/manifest，然后由 Step20 建设新的 Step27-specific prospective batch。

### 情况 E：未来 prospective holdout 也通过

只有此时才可以讨论对未见 seller components 的泛化和论文方法优越性。

---

## 12. 目前仍未解决的核心问题

1. 中文独立强身份正例 component 数量仍不足。
2. 中文 train positive 主要由 silver/soft supervision 构成，可能与 valid/test evidence composition 不匹配。
3. public-contact/URL noise 在 valid/test 中数量太少，无法稳定训练或评估 veto/expert。
4. semantic、style 和 item structure 都容易学习商品主题/模板，而非 controller identity。
5. 现有文本增强、同 seller split、multi-instance distribution、LoRA/NLI 等尝试均未展示可靠独立增益。
6. 多轮历史实验已经消耗旧 valid/test，必须转向新 prospective data，而不是继续在同一边界调权重。
7. synthetic data 可以增加训练视图，但不能增加独立身份数量，也不能修复 ground-truth scarcity。

---

## 13. 不应再做的事情

- 不要恢复 Step15 历史高分作为当前论文主结果。
- 不要把 Step24 train-D0 `AP=0.802718` 当作 corrected valid 性能。
- 不要继续调 Step25 C2、Step26B 或旧 Step15-v8 valid/test。
- 不要为了让 mixup 执行而人为扩负样本、制造类别不平衡。
- 不要编造市场来源、seller identity 或直接联系方式证据。
- 不要用外部普通 spam/微博/Telegram channel 数据冒充暗网市场 seller-pair benchmark。
- 不要把 synthetic row 当作新的真实中文马甲正例。
- 不要用测试集 ROC-AUC/AP 选择模型或阈值。
- 不要在 Step11 审计中 glob 整个 `reports/`；必须使用 explicit allow-list 和 summary 的 `output_paths`。
- 不要让旧 summary、CSV、snapshot 或不同数据 manifest 的结果互相覆盖。

---

## 14. 代码和输出纪律

1. 每个实验必须使用独立 versioned output root。
2. 写入已有 artifact 时必须验证 code/data manifest 完全一致；不同 manifest 不允许覆盖。
3. summary 必须记录 input/output paths、SHA-256、数据行数、label counts、split 和模型版本。
4. 排序指标和阈值指标必须分开解释：ROC-AUC/AP/PR-AUC 不依赖 threshold；ACC/F1/Recall 依赖冻结 threshold。
5. AP 是当前类别不平衡 pair verification 的主要排序指标；MAP/MRR 只有在预先定义了有意义的 query group 时才报告。
6. seller pair 不是独立同分布样本，bootstrap/fold 必须按 seller component 分组。
7. 当前 valid/test 只允许报告，不允许再参与选择。
8. Step11/17 只接收通过 Step12 且预注册的 scorer，并使用 explicit allow-list。

---

## 15. 给新 AI 的工作方式要求

1. 先读代码和 manifest，再判断，不要根据旧对话中的单个高分下结论。
2. 遇到 Linux 报错，先在 Windows 做静态复现或最小 contract test，再给用户重跑命令。
3. 不在 Windows 启动正式模型运行。
4. 修改前说明改哪些文件和原因；修改后必须运行 compilation、unit tests、JSON 解析和 Bash `-n`。
5. 使用 Git 分支和小范围 commit；不要提交无关 reports。
6. 任何科研主张都区分：工程通过、开发诊断通过、统计 gate 通过、prospective confirmation 通过。
7. 对结果保持客观：负结果可以冻结，不能为了正结果降低门槛、改标签或打开已消费 test。

---

## 16. 当前交接状态

- Step27-v1.1 代码修复：完成。
- 独立只读代理最终复核：完成，无剩余 blocker/high/medium。
- Windows contract tests：`47/47` 通过。
- Git commit：`ff4bc04`。
- Git push：尚未执行。
- Linux 正式数值运行：尚未执行。
- 下一动作：同步 `ff4bc04` 文件，确认 Linux 冻结 E5 指纹，运行 v1.1 runner，同步唯一输出根目录后做 manifest-first 审查。
