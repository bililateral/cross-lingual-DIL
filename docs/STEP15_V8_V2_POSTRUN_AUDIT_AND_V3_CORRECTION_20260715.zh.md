# Step15-v8 V2 重跑审计与 V3 修正说明

更新日期：2026-07-15

## 1. 审计结论

V2 返回文件同步完整，但 V2 的主验证集边界不合格，因此其验证指标、阈值和 Step12 晋级结论均不能作为论文依据。

同步 manifest 共记录 226 个文件、76,259,251 字节。逐文件检查路径、大小和 SHA-256 后，缺失与不一致文件均为 0。V2 效果差不是文件漏同步或传输损坏造成的。

V2 的训练折外选择结果为：

| 方法 | Train-OOF AP |
|---|---:|
| B0 | 0.651515 |
| B1 | 0.691656 |
| B2 | 0.718342 |
| B3 LR/L2 | 0.747101 |
| B3 RankNet | 0.789716 |

B3 RankNet 被训练集内按 seller component 分组的折外 AP 选中。但在 V2 代表性验证集上：

| 方法 | ROC-AUC | AP |
|---|---:|---:|
| B0 | 0.676232 | 0.618675 |
| clean selected | 0.719065 | 0.635463 |
| contextual fusion | 0.706887 | 0.634784 |

它没有达到预注册的 clean AP 增益、公共噪声假阳性率下降、fusion 不退化和 grouped bootstrap 非劣门槛。内部 200 条开发测试上 clean/fusion AP 为 0.534110/0.587025，但该测试只允许诊断，不能补救正式验证失败。

## 2. V2 的真实缺陷

V2 使用了旧 V7 representative-validation assignment。该 assignment 曾把 50 条 canonical `train` 行提到 `valid`。其中：

- 45 条为 `silver_train_only=1` 且 `benchmark_eligible=0`；
- 44 条为 positive，1 条为 negative；
- 它们占 V2 170 条主验证行的 26.5%；
- 它们占 V2 76 条验证正例的 57.9%。

这不是“训练时直接读取 test”的泄漏，但属于严重的评估边界污染：训练专用弱标签进入了主验证集，且阈值也在该验证集上选择。因此 V2 的验证指标既不能解释为 gold benchmark，也不能用于论文模型晋级。

## 3. 正确的 canonical 边界

当前 Step5 中文二分类监督数据为：

| Split | Positive | Negative | Total |
|---|---:|---:|---:|
| train | 229 | 344 | 573 |
| valid | 30 | 90 | 120 |
| internal development test | 50 | 150 | 200 |

Train 中有 371 条 train-only silver；它们只允许训练。Valid/Test 全部 `benchmark_eligible=1`，不存在 silver 行。

V3 将恢复 canonical split，因此预期主边界为：

- English source train：401；
- Chinese train：573；
- 总训练行：974；
- Chinese representative valid：120；
- Chinese internal development test：200，pair UID 集合保持不变；
- evidence-expert controls：train 60、valid 55，始终与主身份验证指标隔离。

## 4. V3 代码修正

1. `step15_v8_common.split_rows` 对 primary valid/test 增加 fail-closed 检查。任何 `benchmark_eligible!=1` 或 `silver_train_only=1` 行都会阻断训练。
2. readiness materializer 以 Step5 canonical `split_name` 为主身份样本唯一边界，不再沿用旧 assignment 的 `v7_split_name` 提升结果。
3. 新证据控制样本根据 canonical seller partition 动态判断可进入 train 或 valid，不能依赖旧 queue 的 split hint。
4. `step20_build_representative_validation.py` 保留完整 train supervision，但只有整个 seller component 均为 benchmark-eligible 时才允许从 train 移到 valid。
5. V3 输出使用新目录，绝不覆盖 V2。已有 V3 freeze 只有在全部冻结 payload 与重新计算结果逐字节一致时才允许复用。
6. V3 的补充 URL 审查派生产物单独写入 `reports/step15_v8/profile_url_control_review_v3_20260715/`；它读取并哈希绑定原 reviewer lane，但不会覆盖旧轮次的审查 summary/CSV。
7. 每条补充 public/victim-data URL control 会在两侧 seller 上物化 `external_url` occurrence，固定 `product_data_risk_context=1`、`direct_identity_eligible=0`。物化器逐 pair 复算 occurrence state；任何选中 public control 落入 `no_shared_identifier`、`ambiguous` 或 direct 状态都会阻断 freeze。
8. V3 artifact tests 不再默认读取失效的 V2 freeze。runner 先运行静态测试，V3 物化后再以 `STEP15_V8_READINESS_ROOT` 指向新 freeze 重跑全部 artifact/hash/count/state tests。
9. Step20 新 manifest 先写入 assignment CSV SHA-256，再计算 manifest self-hash；新建 evidence controls 的 `candidate_scope` 统一为 `evidence_expert_control`。

## 5. 公共网址噪声补齐

恢复 canonical valid 后，真实冻结输入测试确认：现有双审 context 队列只能提供 12 条 valid-compatible public-noise controls，固定门槛仍为 20，不能通过降低门槛规避。

为补齐差额，新增了一个不读取模型分数、不向审查者暴露 split 的 profile-URL 候选复核：

- 两条隔离审查 lane 分别判断 identity label、evidence type 和 confidence；每条决定保留实际 agent reviewer ID，不能把不同轮次的审查冒充成同一审查者；
- 仅接收两人完全一致的 `negative + public_contact_or_url_noise + high`；
- 程序重新读取原始 seller profile，要求同一 URL literal 确实出现在两侧；
- 两轮共审查 11 个候选提案；一组原先描述为“共享 DeepMix URL”的候选被自动检查发现两侧 `.onion` 域名不同，已从最终候选域剔除；
- 最终候选域为 10 条，8 条获得两位不同 reviewer 完全一致的 high-confidence public/victim-data URL negative 结论；
- 两条不确定或审查分歧候选不进入任何训练、验证或测试。

这 8 条样本均设置为 evidence-expert-only controls：不写为 Step5 gold，不进入 primary identity scorer，不计入主 benchmark，只用于验证 occurrence-level evidence expert 能否识别公共网址噪声。它们与 12 条原有 valid-compatible controls 合并后恰好达到固定 valid 配额 20。

## 6. 当前状态

本地仅执行了 Python 语法编译、JSON/CSV 静态解析和纯契约测试。Step15-v8 共发现 50 项：43 项无 V3 artifact 的静态/真实冻结输入契约均通过，7 项物化后 artifact 测试因 V3 尚未在 Linux 生成而明确跳过；Step15-v6/v7/v8 合并发现 99 项，92 项执行通过、同 7 项延后。Linux runner 在 V3 物化后会强制重跑全部 50 项，不能把当前 skip 解释为已验证 V3 输出。没有在 Windows 上运行模型、特征流水线或数值实验。

V3 尚无性能结果。只有 Linux 完整重跑后，才能回答：

1. B0-B3 在正确 valid 上哪一个胜出；
2. occurrence-level contextual fusion 是否真正降低 public-noise FPR；
3. 是否满足全部 Step12 晋级门槛；
4. 是否允许进入 Step20 和 Step11/17。

V2 目录必须保留为失效实验的审计证据，但不得再引用为当前有效结果。
