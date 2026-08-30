# Step28-v13 v1.12 正式中文合成数据集构建方案

> **历史关闭声明（2026-08-30）**：V1.12 已永久质量失败，本方案不再可执行。其专属策略、脚本和测试已从当前工作树清理，精确字节仅由 Git 历史保留；失败结论和保留边界以 `docs/STEP28_V13_V1_12_TERMINAL_FAILURE_AND_CLEANUP_20260809.zh.md` 为准。

日期：2026-08-03

状态：`DRAFT_IMPLEMENTATION_NO_SEED_OR_DATA_AUTHORIZATION`

## 1. 目标与结论边界

本方案的目标是生成一套可直接支持 M0/M1/M2/M3 训练与评估的 fresh 中文合成马甲数据集。科研问题仍是：冻结的真实英文 M0 能识别多少中文合成身份关系；在完全相同的基础分数、样本和训练程序下，正确 identity33 的 M2 是否稳定优于五个错配 identity33 的 M1。

数据通过只证明合成机制和工程边界成立，不证明真实中文地下市场外部有效性。旧 v1.2 仅保留历史字节；v1.3–v1.11 永久失败，代码、seed 和身份值均不得复用。

## 2. 固定数据形状

- split 顺序固定为 `train`、`development`、`audit_a`、`audit_b`；
- 每个 split 500 个相互隔离的 K28 world；
- 每个 world 为 12 个 controller：8 个 dyad、4 个 triad；
- 主分类使用全部 378 个无序 seller pair，固定 20 正、358 负；
- 每个 split 189,000 pair，10,000 正、179,000 负；
- train、development、Audit A 每 world 各有 84 个身份资产、每 split 42,000 个；Audit B 因冻结的未见状态机制每 world 为 89 个、每 split 44,500 个，不能把四个 split 错写成相同计数；
- 检索使用 28 个 seller 全部作为 query，每个 query 的 gallery 为其余 27 个 seller；
- C40 不进入训练、调参、阈值、主评估、模型挂载或发布成员。它若保留，只能在全部预测冻结后由私有诊断器计算。

## 3. 随机性和一次性仪式

只有正式源码闭包、依赖版本、两 world 写盘重放、500-world 数值捷径预检和全仓测试全部通过后，新的正式 prelock 才能授权 seed ceremony。

ceremony 按固定 split 顺序调用操作系统 CSPRNG 各一次，每次恰好生成 32 byte master seed。公开区只写 master 和 capability 的 SHA-256 承诺；原始 master 只写入 Git 忽略的 `private_custody/`。使用
`HMAC-SHA256(master, "step28-v13-v1.12" || 0x1f || split || 0x1f || role)`
派生互不复用的 structure、ID namespace、ID、text、identity bootstrap、identity remap、query 和内部兼容流。train 另派生 r01–r05 五把 M1 rewire key。

master 不进入生成器或模型进程。一个 split 生成器只挂载该 split 的 generator capability；一个 M1 进程只挂载一把 rewire key。ceremony、每个 split core 和每份 M1 都必须在第一次读取相应私密能力前，以 no-replace 方式写入持久 start receipt。start receipt 后若没有形成完整且可逐字节验证的 stage/完成件，该动作和整个 v1.12 永久失败；禁止补抽、换域、筛 seed 或从头重跑。若中断发生在完整确定性产物写成之后，只允许验证原字节并继续尚未完成的 manifest/发布步骤，不得重新抽 seed 或重新生成数据。

## 4. 新生成器边界

正式生成器只允许复用 v1.12 预仪式策略固定的 15 个成功 v1.2 纯生成源文件。旧 C40 构建器和所有失败版本实现都不在源码闭包中。另有一个只在 pre-seed 可信审计进程内运行的历史身份覆盖器：它读取已泄漏、仅作历史证据的 v1.2 四份 `identity_assets.jsonl`，按正式归一化规则重算哈希，公开回执只保留文件 pin、计数和集合摘要；这些原始身份值不进入新生成器、模型或新发布树。

该覆盖审计已明确证明：原身份禁用库 `112,996` 个哈希与旧 v1.2 的 `170,500` 个身份哈希零交集，并集为 `283,496`；三组哈希全部是当前 `915,996` 个只读禁用哈希的子集，缺失数均为 0。因此 `915,996` 不是只靠名称推断的“失败哈希”，而是当前生成器实际使用、同时覆盖原禁用库、成功 v1.2 和 v1.3–v1.11 失败运行的完整历史禁用集合。

每个 world 的固定顺序是：

1. 使用 split capability 生成 world/controller/seller/item 和结构；
2. 按 identity asset UID 固定顺序，为每个资产选择第一个可接受的身份值；
3. 候选不得命中上述 915,996 个完整历史禁用哈希、同次运行已分配哈希或身份无关可见文本；碰撞只推进该资产计数器，不更换 master；
4. 运行生产 Step3 parser，并与私有 slot/AST 计划逐行核对；
5. 标题只归一化，不替换登记身份串；描述只删除登记 context-guard 后缀；join UID 不作为自然语言扫描；
6. 从安全历史生成正确 identity33，从可见商品生成 seller profile；
7. 冻结完整 378 pair 输入后，私有验证器才从 controller membership 独立推导标签；
8. 原始生成表、公开投影、identity33、标签和回执均完成确定性重放及 schema/keyset 校验。

机制槽唯一键固定为 `(world_uid, mechanism_slot_uid)`；controller、seller、item、identity asset/value 等真实实体仍要求全局唯一。

## 5. 发布树和私有保管

公开 split 只包含：

- `observed/worlds.csv`
- `observed/sellers.csv`
- `observed/redacted_items.jsonl`
- `observed/seller_profiles.jsonl`
- `observed/complete_model_pair_endpoints.csv`
- `features/identity33_all_pairs.csv`
- `retrieval/queries.csv`
- train/development 的 `supervision/classification_labels.csv`
- 非连接审计回执、模型挂载白名单和 `split_manifest.json`

Audit A/B 的标签、qrels、controller、机制、身份资产、原始身份文本、生成 AST 和逐行私有审计只进入 Git 忽略的私有区。train 的五把 M1 key 和逐次映射能力同样保持私有；公开区只写不含逐行 join key 的计数、布尔值和哈希回执。

生成 stage 位于私有区。只有 split 所有门通过后才以 no-replace 原子发布公开目录；失败 stage、缓存和失败产物在记录失败边界及不可复用哈希后删除。正式成功私有 oracle 是复现和盲评资产，不属于垃圾文件。

seed ceremony、train/development execution lock、Audit A lock 和 Audit B lock 是四个依次收紧的权限层。train/development 全部发布后只能签发 `audit_a_generation_lock.json`，它只授权 Audit A；Audit A 完整生成、密封质量审计并发布后，才能另签 `audit_b_generation_lock.json`，后者只授权 Audit B。禁止一个 lock 同时授权两个 Audit。ceremony 任一不可恢复异常先写永久关闭回执，再删除私有失败 stage；start receipt 存在后，v1.12 不得再次抽 seed。正式 split 的 core、M1、finalize、quality 和 publish 均为单向状态；core/M1 start receipt 存在但对应 stage 不存在时必须失败关闭，完整 stage 或单侧 manifest 存在时只允许精确验证后续接，不得从头生成。

## 6. 训练前捷径与质量门

正式 seed 前必须在相同 500-world/189,000-pair 规模上用非正式设计 key 运行一次完整数值预检。优化器必须无 warning、目标有限、迭代未触顶且归一化梯度达到冻结容差；仅测试失败关闭语义不算通过。

正式 train/development 生成后还必须重新执行：

- 完整图、20/358 标签公式、跨 world/split keyset 和全局实体唯一性；
- UID/hash/行序及 14 项 label-free null-nuisance 特征的单项与组合攻击；
- 单项最大对称 ROC-AUC ≤ 0.52，组合对称 ROC-AUC ≤ 0.53；相对随机 AP 的点提升 ≤ 0.01，world-bootstrap 95% 上界提升 ≤ 0.015；
- 身份残留、controller/mechanism/label token、内部生成标记和未登记文本投影扫描；
- 新旧 UID、身份值哈希、完整字段和 seller-document 精确指纹交集为零；近重复链接统计按冻结阈值报告；
- 字符 n-gram 可预测性单独报告但不冒充 null-nuisance 门，因为合法作者风格本来就是 M0 的预期输入；
- 五个 M1 对每个 world 都是 378 行双射、零 fixed point、零端点重叠，并逐字节保持 identity33 整行多重集。

任何正式门失败都使 v1.12 失败，不允许据结果改阈值或换 seed。

## 7. 模型挂载和后续训练

M0 挂载只允许 seller profile 中依次排列的 `category_concat_top`、`signature_title_concat`、`title_concat_top`、`signature_description_concat`、`description_concat_top` 五个文本字段，以及只负责 join 的 pair endpoint。`profile_text`、seller/world/item UID 和其余元数据不得进入编码文本；M0 不得读取 synthetic 标签或 identity33。M1/M2 设计矩阵只允许冻结 `p0`、33 个数值身份特征和训练所需标签；UID、端点、行号、world 和文件顺序在拟合前删除。

M1-r01…r05 与 M2 使用相同 p0、样本、变换、权重、共享 L2、求解器和收敛门，唯一差异是错配或正确 identity33。development 只冻结阈值。M3-base/M3-joint 是直接中文训练强对照，不代替迁移对照。

分类报告 ROC-AUC、AP、梯形 PR-AUC、Precision、Recall、F1、Specificity、Balanced Accuracy、MCC、Brier、Log Loss、Recall@FPR=1% 和混淆矩阵；检索报告 MRR、MAP、Recall@1/3/5/10、NDCG@1/3/5/10。统计、置信区间和置乱单位均为 world。

## 8. 分阶段执行顺序

1. 完成正式生成、finalizer、custody、数值捷径预检和失败关闭测试；
2. 先完成历史 v1.2 身份哈希覆盖证明，再用非正式 key 完成两 world 真写盘/真消费重放与 500-world 数值预检；
3. 运行全仓测试，清理缓存，冻结正式源码和依赖闭包；
4. 发布只授权 seed ceremony 的 prelock；
5. 一次性生成四份 master commitment，发布不可覆盖 execution lock；
6. 生成并审计 train、development，物化五份 M1 结构回执；
7. 只有 train/development 全部门通过后才签发只授权 Audit A 的锁；Audit A 密封质量审计并发布后，才签发只授权 Audit B 的第二把锁；
8. finalizer 重读所有公开/私有成员，验证跨 split/旧版本隔离并发布根 manifest；
9. 根状态为 PASS 后才进入 M0 Linux 编码和 M1/M2/M3 训练。

在第 5 步之前，任何输出都只能称为设计或预检；在第 8 步之前，不能称正式中文合成数据集已经完成。

## 9. 当前实现进度（2026-08-03）

- 一世界完整生成和 train/development 两世界真写盘、真消费重放已通过；
- 500 train + 500 development、每 split 189,000 pair 的设计 key 精确统计捷径预检已通过；
- seed ceremony、capability custody、train/development execution lock、Audit A/B 后置授权、四 split 私有 stage、正式质量审计和根 finalizer 已进入源码审查与契约测试；
- 历史身份覆盖器已重算旧 v1.2 的 170,500 个身份哈希，并确认当前 915,996 禁用集合缺失数为 0；该证明将作为正式 prelock 的独立输入；
- seed ceremony、split core 和 train M1 已增加私密能力读取前的持久 start receipt；完整产物后的发布中断可精确续接，不完整动作禁止重启；
- 目前仍未创建正式 master、capability、正式数据行或模型结果；
- 上述设计回执必须在最终源码闭包提交后重新运行并与全仓测试一起写入 prelock，旧的开发中回执不得直接授权 ceremony。
