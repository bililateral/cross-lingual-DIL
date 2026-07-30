# Step 28-v13 正式训练与固定留出中文合成数据集发布合同

更新日期：2026-07-30  
当前状态：`FROZEN_READY_FOR_GENERATION / PRIVATE_KEY_CEREMONY_COMPLETE`  
目标版本：`v13_training_ready_v1_20260729`

## 1. 权威范围

本文档规定实际交付给 M0/M1/M2 的非冒烟数据集。此前
`dataset_smoke_v3` 永久保留为开发记录，禁止训练、调参、评估或并入本
版本。原
`STEP28_V13_SYNTHETIC_CHINESE_DATASET_BUILD_CONTRACT_20260727.zh.md`
继续保存设计演进；与本文冲突时，以本文及最终发布清单为准。

只有完成私钥仪式、四 split 正式生成和全部最终发布门后，本版本方可用于
M1/M2 训练、development 定阈值和 Audit A/B 固定留出评估。当前没有正式
数据字节，不能训练或评估。
`Audit` 是 split 名称，不等于本机已经实现人员盲法。当前 Windows 单用户
工作区只能做到目录分离、哈希封存和模型输入白名单，不能冒充独立 OS
账户、ACL、WORM 或第三方盲测证明。论文若主张盲评或外部不可篡改
custody，必须在模型、阈值和统计代码冻结后，由隔离 Linux 账户或第三方
对相同冻结字节补做部署见证；这不改变数据内容、模型或指标。

## 2. 科研比较

- M0：冻结的 `LightGBM + legacy18 + LaBSE` 英文来源完整流水线，不在合成
  中文上训练。
- M1：冻结 M0，加在 train 内按世界和 C40/非 C40 分层、端点不重合整行
  置乱的 33 维身份适配器；固定 5 个 seed。
- M2：冻结 M0，加使用正确 train 身份33维训练的同构适配器。
- M1/M2 推断都读取未置乱的 development/Audit 身份33维；差别只在训练
  矩阵。

## 3. 固定规模与隔离

| split | worlds | sellers | C40 | C40 正/负（每世界） | 用途 |
|---|---:|---:|---:|---:|---|
| train | 500 | 14,000 | 20,000 | 16/24 | 训练 M1/M2 |
| development | 500 | 14,000 | 20,000 | 10/30 | 阈值与校准 |
| audit_a | 500 | 14,000 | 20,000 | 10/30 | 主固定留出 |
| audit_b | 500 | 14,000 | 20,000 | 10/30 | 机制移位复核 |

四个 split 的 world、seller、item、identity、pair UID 与结构密钥必须零
交集。每世界固定 12 个 controller、28 个 seller、378 个完整无序 pair。
标签唯一公式为：

```text
label = int(controller(left) == controller(right))
```

禁止从英文标签、真实中文标签、M0 分数或 adapter 结果生成任何标签。

## 4. C40 正式抽样

旧文本触发 C40 已废止。它使正例主要来自共享联系方式、负例主要来自描述
相似，导致 `absdiff(item_count)` 单特征 AUC 约 0.57。

正式 C40 在所有文本和非身份字段生成完毕后执行。它不是标签盲候选抽样，
而是只适用于已知合成 controller oracle 的机制分层 case-control 抽样：

1. 在正例、负例内部独立抽样，不改变标签公式；
2. 每世界先至少覆盖 8 类正向身份机制和 7 类困难负例 flag；
3. 其余名额按固定 candidate key 的完整 HMAC 排序补齐；
4. 模型可见表只有 pair UID、world UID 和两个 endpoint；
5. label stratum、机制、flag、HMAC、selection role 和 rank 只进私有审计。

该数据集估计的是预注册机制分层 case-control 总体，不冒充地下市场的自然
pair 阳性率或自然校准分布。

500-world 冻结设计诊断：

- 旧 C40：最大 AUC `0.566523`，95% 上界 `0.577075`，失败；
- 机制分层 train 16/24：最大 AUC `0.507322`，上界 `0.514409`；
- 机制分层 eval 10/30：最大 AUC `0.506017`，上界 `0.515446`。

早期直接调用当时抽样实现、并经过真实解析和脱敏链的设计预检显示：

- development 125 worlds：最大 AUC `0.512046`，上界 `0.528742`；
- development 250 worlds：最大 AUC `0.514700`，上界 `0.526101`。

125 worlds 虽通过但离上界门槛仅 `0.001258`，因此不采用；后续先将
development 设计预检扩为 250 worlds，正式 training-ready 版本最终统一为
每个 split 500 worlds。上述两个数值属于旧
`development_smoke` 随机域的设计证据，不能替代显式
`training_ready` 模式的逐 split 精确预检。正式结构密钥生成前，必须用
设计专用结构 key、正式公开随机流和最终构建器重新跑 train/development/
Audit A/Audit B 的注册规模；不得据其选择正式 key。

诊断文件 SHA-256：

- `candidate_shortcut_diagnostic_500_v1_20260729.json`：
  `569d5342a6f212ab7cf15793a858f345e2ce62dfc8ca1f3a973875b4dfd46167`
- `candidate_mechanism_diagnostic_500_v1_20260729.json`：
  `a82a6683998db11f148abc12bd981467b10b60a4d985cb63a2d391b37170185d`

修复前历史实现合同
`b85798d7d8b446f32847f90aaf9e59db14a0e181164632c3c3ea6c17822ad73b`
下的四份 500-world 精确预检曾通过：

| split | 最大对称 AUC | world bootstrap 95% 上界 | 身份33维秩 | 全零列 | 报告 SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | 0 | `b47dff571746605f56d15a8ea12040b64f3759414dcccd76ff2948c992cf6562` |
| development | 0.508037 | 0.515723 | 29 | 0 | `0aa20c80d443b1c3b9803379f7450e08008b29ca14456c126d20e24055d76503` |
| audit_a | 0.509954 | 0.517763 | 30 | 0 | `02cd21f841e83ba0f79d9da546d0bdc074e7b38b0930b86618333e16355f3dff` |
| audit_b | 0.505748 | 0.513693 | 32 | 0 | `7a8f68e92f976585e2713cb3429dd4e32fbacdddf84d2cecaebd3d6007d8305e` |

train 报告明确记录 `no_all_zero_columns_required=true` 且通过。四个矩阵均
存在强相关和秩亏，属于预定义派生身份特征的冗余，不是 14 项元数据标签
捷径；后续 adapter 必须使用冻结正则化，报告条件数和秩，不得把单个共线
系数解释为独立因果效应。以上预检使用设计专用结构 key。2026-07-30 最终
审核随后发现：旧报告未强制绑定完整 checkpoint/OOF/bootstrap 证据，私钥
恢复路径也未完整重算禁用 commitment。修复这些代码后，构建器源码闭包和
科研实现合同已经改变，所以上表四份报告现只保留为历史诊断，已从 overlay
正式登记表撤下，不能再证明当前实现通过。当前实现必须重新跑四个
500-world 精确预检；不得根据旧数值挑选或重抽正式 key。

第二轮修复历史科研实现合同
`67b8c720e7287fb5742b5417a98be6578a87b657ca58858bc3992713510f660a`
曾重新完成四个 500-world 精确预检，并由当时的登记器核对原始输入、标签、
fold、三模型 OOF、bootstrap 摘要和阶段 checkpoint：

| split | 最大对称 AUC | world bootstrap 95% 上界 | 身份33维秩 | 全零列 | checkpoint | 历史报告 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | 0 | 5 | `6716cbe2aac9e220f8ae17a090ebe0aa0f9059ad457bea75105d02a198ace72d` |
| development | 0.508037 | 0.515723 | 29 | 0 | 5 | `a35373be8b01af52c93c2f726adf8e6a856b99a409f632ec81fe7d32a782b523` |
| audit_a | 0.509954 | 0.517763 | 30 | 0 | 6 | `8c7df397df286fd37afcbc3f3727422e55f1a6aac181a2b8b56d9cc5ae62ec6a` |
| audit_b | 0.505748 | 0.513693 | 32 | 0 | 6 | `9c856fa701d76f90c4021e1c0f3381a5112fb7899c22b59d1a69b7946c36f4bc` |

最终代码复审又发现，当时的登记器虽然逐项核对这些数组、重算 AUC、分位数
和固定位置的 bootstrap 统计，却没有从保存的 14 维输入、标签和 world fold
重新训练三个冻结审计模型，也没有逐元素重算全部 9,999 个 bootstrap
统计量。因此一组内部自洽但整体伪造的 OOF/bootstrap 仍可能通过，以上四份
报告不得继续充当当前实现的通过证据。

当前科研实现合同已升级为
`22e136c5bea376aedc68f784b148d4ab67b81216c69ccf34acd836a3710ce601`。
登记器现在会重新训练逻辑回归、梯度树和 RBF-SVM，逐元素比较三个 OOF
数组，再按冻结算法重算并逐元素比较全部 9,999 个 bootstrap 统计量；split
写盘阶段也会在发布前拒绝 Windows 重解析点。旧四份报告保留为历史科研
记录，不得混入现行登记。

v3 合同下现行四份 500-world 精确预检为：

| split | 最大对称 AUC | world bootstrap 95% 上界 | 身份33维秩 | 全零列 | checkpoint | 现行报告 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | 0 | 5 | `628bda9181b9063f2b9978f6d8b2e615f19c115cb0604a647f00976bed5aaec8` |
| development | 0.508037 | 0.515723 | 29 | 0 | 5 | `518dc19d26b82111fa12782863f6a58a07089e9a7a16d2ff3142d9c8e6a54563` |
| audit_a | 0.509954 | 0.517763 | 30 | 0 | 6 | `d9c2a03d32f7ed9316964bfbf9ac0c29bdb4b779b4816392a9699142256af2d6` |
| audit_b | 0.505748 | 0.513693 | 32 | 0 | 6 | `3dbcc4642ca0862e551df3ec5275e9bd3cc376ea6b2db183f0b23ebae23c8700` |

四份均低于 0.52/0.53 门并登记到 overlay。配置加载器在一个新进程中用
437.6 秒顺序重训四分区三模型、逐元素核对 OOF 和全部 9,999 次统计后，
返回 `PASS_CONFIG_VALIDATION`。Audit A/B 各另核对 2,000 个 query、
54,000 条 directed relation 和 54,000 条 qrel。第一次 v3 train v7
因运行命令漏传 checkpoint 前缀，虽完成数据审计但
`checkpointing_enabled=false`，只保留为无效诊断；现行 train 是完整
5-checkpoint 的 v8。Step28-v13 专项回归现为 91 项通过、1 项按既有
execution-blocked 声明跳过、0 失败；完整仓库回归为 360 项通过、7 项按
既有声明跳过、0 失败。当前仍须完成三路复审和 Git 基线冻结，正式私钥和
正式数据仍为 0。

## 5. 身份值与解析

正式 Telegram、蝙蝠和微信句柄使用类型前缀加 14 位小写十六进制主体。
禁止 unrestricted base36，因为其可能随机包含 `cvv`、`pwd` 等字符串，
被冻结 Step3 风险正则误判。完整正式候选池必须满足：

- 与真实身份值、Step28-v6–v12、smoke 值零交集；
- 同值固定 surface，跨 split 零复用；
- Step3 解析结果与生成前独立计划逐行、逐 flag 完全相等；
- 身份 surface 在 Step7-v4 脱敏文本中残留为 0；
- `must_ignore` 噪声不得进入身份图，也不得被脱敏删除。

## 6. 可直接消费的模型输入

M0 每 split 只允许挂载：

```text
observed/complete_model_pair_endpoints.csv
observed/redacted_items.jsonl
observed/seller_profiles.jsonl
```

Adapter 只允许读取冻结 M0 `p0`、C40 endpoint、一个 33 维矩阵和对应
train/development supervision。禁止 UID、market、机制、抽样审计、原始身份
文本、controller 或 derangement mapping 进入特征矩阵。

train 必须交付正确 M2 身份33维和 5 份 M1 destination-keyed 置乱矩阵。
每份 M1 必须与 M2 在每个 `(world,C40/non-C40)` 中保持完整 33 维联合向量
multiset 相同，且 source/destination pair 不共享 endpoint。

## 7. 检索任务

Audit A/B 每世界按固定 query key 选择 4 个 query seller，每个 query 的
gallery 为同世界其余 27 个 seller。相关性唯一公式为 controller 相等。
每个 audit 恰有 2,000 queries、54,000 directed relations。qrels 与分类
labels 一起做逻辑封存，并由模型输入白名单排除；本机同一用户仍可读取，
所以不得称为人员盲法。真正盲评时须在预测和统计代码锁定后由隔离保管方
解封。

基础构建合同第 16 节仍保留旧 250-world audit 的历史 `10,000/27,000`
计数，因为该基础文档已被 base policy 按字节固定；正式 training-ready
发布以本文 `20,000` 条分类 pair 和 `54,000` 条检索 relation 为准，禁止
把旧计数带入模型评估。

## 8. 发布前强制门

任一门失败即标记该 run 无效，不换 key、不删行、不降低阈值：

1. 四 split 行数、外键、UID 唯一性和哈希清单完全一致；
2. 标签及 qrels 由独立公式重算逐行相等；
3. train/development 的冻结14无关特征最大对称 AUC不高于 `0.52`，9,999
   次 world bootstrap 95% 上界不高于 `0.53`；
4. Audit A/B 同一门由发布 validator 验证，模型挂载只允许看到 PASS/FAIL
   和哈希；这属于逻辑封存，不是 OS 盲测证明；
5. 8 类正向机制、7 类困难负例在每个 world 均覆盖；
6. 正式 train 的 33 维均须有非零训练支持；development/Audit 因预注册
   G_A/G_B 机制移位可以出现全零列，但不得临时删列，并须逐 split 报告
   全零列、秩、重复列、条件数和零历史数量；
7. 五份 M1 全部通过端点不重合、双射、联合向量守恒；
8. M0/M1/M2 模型输入 allow-list 检查无额外列；
9. 生成器重放、完整单元测试和 `py_compile` 全部通过；
10. 发布目录以 no-replace 原子 rename 完成，禁止覆盖同名旧字节。

最终构建器不能只固定入口脚本。程序必须从入口递归解析仓库内静态与字面
动态 import，登记当前 33 个源码成员、成员数和包含每个文件 SHA-256 的
闭包哈希；入口、成员或闭包任一漂移均在生成前失败。另建立机器可读科研
合同，冻结 C40 estimand、M0/M1/M2、固定样本量、指标和推断边界；稳定
“科研实现合同”必须绑定它和全部发布工具，只排除报告登记、生命周期状态
与私钥承诺。四份精确预检报告必须绑定同一实现合同、构建器、源码闭包、
候选实现、基础 policy 和精确预检程序，并逐件绑定 checkpoint、14维输入、
OOF 分数及 bootstrap 统计证据，之后才能登记到发布 overlay。这样既避免
报告哈希与 overlay 自引用，也不允许叙述文档更新改变机器科研规则或掩盖
生成逻辑变化。

该发布是“通过预注册无关特征门后的生成器实现/随机实现”条件总体。门通过
只能排除这 14 项已登记元数据捷径超过阈值，不能证明不存在任何未知捷径，
也不能把 case-control 分布解释成真实地下市场自然分布。正式 split 若失
败，只能判定该 run 无效；禁止重抽 key、删行或降低门槛来换取通过。

## 9. 固定样本量与推断边界

父 draft 曾要求在未知 M0/M1/M2 score 分布、五个 M1 相关性、world ICC 和
hard-negative 误差结构的情况下，先假定这些量再做 5,000 次 Monte Carlo
选择 W。该 artifact 从未生成，父 policy 中路径和哈希始终为 null。为这些
未知量填入有利数字只会制造虚假功效，因此本 training-ready child 在任何
正式私钥生成前明确撤销该确认性选择合同。

四个 split 固定为最大可行的 500 个独立 world，不再声称由 80% 功效选择。
正式结果必须报告效应估计和预注册的 world-cluster 配对 bootstrap 区间；
禁止用“预先保证成功/失败”或“确认性功效已认证”表述，也禁止在看到结果后
增加样本。预先敏感性 artifact 只回答“在不同未知 paired-world 标准差下，
500 worlds 能检测多大差异”，不是实际 AP/bootstrap 的替代：

- artifact：
  `training_ready_fixed_sample_sensitivity_v1_20260730.json`；
- SHA-256：
  `cd8464d6efb9be16f98614a785dfefa628e35ea99f533fccc01527831f24a3bc`；
- 机器可读科研规则：
  `schema/step28_v13_training_ready_scientific_contract.json`。

## 10. 指标

分类必须分别报告：世界等权 weighted `average_precision_score` 定义的
AP、梯形 PR 曲线面积定义的 PR-AUC、ROC-AUC、precision、recall、F1、
balanced accuracy、specificity、混淆矩阵和校准指标。AP 与 PR-AUC 不得
互换；阈值仅由 development 冻结。检索必须报告 MRR、MAP@10、
Recall@1/5/10、Hits@1/5/10；同分按 gallery seller UID 的 UTF-8 字节升序
打破。主比较为 `M2-M0`、`M2-五个M1均值` 及最差 seed 差值，使用世界级
配对 bootstrap。

## 11. 更新纪律

正式结构密钥生成前可以修复实现并追加本节记录；密钥一旦生成，只允许修复
不改变数据字节的发布/校验问题。任何会改变 world、文本、C40、身份33维或
标签的修复必须升级 run ID，并将失败版本保留为无效研究记录。

- 2026-07-29 v1：废止旧文本触发 C40；冻结机制覆盖、split-specific
  16/24 与 10/30 抽样；修复 base36 句柄与 Step3 风险正则碰撞；明确正式
  训练可用与 publication custody 的边界。
- 2026-07-29 v2：发现旧设计诊断复制逻辑的 HMAC 域名与正式实现不同，
  改为直接调用正式 C40 实现做逐 split 精确预检；因 125-world
  development 的重采样上界余量过小，将其扩为 250 worlds，并重新冻结
  覆盖 1,250 worlds 的身份值盐池。
- 2026-07-29 v3：新增显式 `training_ready` 执行模式，禁止再借用
  `development_smoke` 名义；将 Audit A/B 的本机声明降为固定留出与逻辑
  封存；新增一次性分割密钥仪式、落盘 M1 独立重读、模型输入白名单和四
  split 最终发布审计。旧
  `step28_v13_metadata_shortcut_audit_lock.json` 明确记录为
  `FORMAL_EXECUTION_BLOCKED`，其父 policy 字节已被后续版本取代，故保持
  原字节不修改；对应旧锁测试在检测到该特定父哈希漂移时有说明地跳过，
  当前 shortcut 回归由 training-ready 端到端契约测试与本 policy 的实现
  闭包接管。
- 2026-07-29 v4：Audit B 500-world 精确预检暴露类型求解器重复从第 0 个
  topology 重放和一个固定背景容量不可行世界。producer 与独立 replay 均
  改为候选缓存的一次有序 DFS；高频 direct hub 在原 HMAC 选择容量不可行
  时，只按固定背景容量选择同 controller 集内的可行 seller tuple，不读取
  标签、C40、文本、缺失率或模型分数。另修复 G_B 的 89 identity assets/
  world 与 100 negative flags/world 被错误按 G_A 的 84/42 校验，以及阶段
  checkpoint 对两列 label 表误要求 `world_uid` 的问题。未触发回退的冻结
  样例 payload 哈希保持不变。
- 2026-07-29 v5：构建器新增递归 33 文件源码闭包、稳定科研实现合同、
  四 split 精确预检登记门和 split/release manifest 的实现来源绑定。旧
  Audit B v4 虽在修复后通过，但因其运行时尚未包含这一最终闭包合同，只
  保留为设计诊断；四个 split 必须用 v5 最终字节统一重跑。原始
  `AI_RESEARCH_HANDOFF_20260719.zh.md` 同时恢复为 v12 清单冻结字节，后续
  主线写入 `AI_RESEARCH_HANDOFF_20260722_ADDENDUM.zh.md`。
- 2026-07-29 v6：最终 train 500-world 预检后人工复核发现，旧代码把
  “train 身份33维无全零列”强制条件错误地限定为
  `generation_enabled=true`，导致设计预检虽报告无全零列，却没有真正启用
  失败门。现改为 500-world train 精确预检和正式生成一律强制；显式
  sub-500 design-key mini 因可能缺少稀有交互，只报告零列并受专用参数、
  非正式状态、设计 key 和规模上限四重约束。development/Audit 继续只报告
  而不临时删列。该次 train v3 报告保留为无效诊断，不能登记，四 split
  继续以修复后实现统一重跑。
- 2026-07-30 v7：将 sub-500 mini 豁免限制为非正式状态、设计 key、train
  split 和小于 500 worlds 四项同时成立；默认调用仍因全零列失败。最终
  80 项 Step28-v13 回归通过、1 项旧 execution-blocked lock 按精确父字节
  漂移声明跳过。随后四个 split 用同一实现合同全部完成 500-world 精确
  预检并通过 0.52/0.53 门；四份报告已写入 training-ready overlay 的精确
  登记表。同步文档和报告登记后，完整仓库回归为 349 项通过、7 项按既有
  声明跳过、0 失败；其中 6 项属于已撤销的 Step28-v11 测试类，1 项属于
  上述旧 execution-blocked lock。此时正式私钥承诺仍全部为 null，正式
  数据目录仍未生成。
- 2026-07-30 v8：最终三路审核判定 NO-GO。发现父 draft 的确认性 power
  artifact 从未生成，当前没有依据填写 M1 相关性、world ICC 或 score
  分布；本 child 因此在私钥前改为固定 500-world 估计性设计并发布敏感性
  artifact，禁止确认性功效/二元成功声明。另发现私钥恢复验证不完整、精确
  预检未强制绑定 checkpoint/OOF/bootstrap 证据、target release claim 被
  误读成当前 bytes-ready，以及 AP/PR-AUC、MAP/MAP@10 口径不清。相关实现
  和机器科研合同正在修复；旧四份精确预检已撤销正式登记，349/7/0 只属于
  修复前回归基线。正式私钥、正式数据和新实现 PASS 均仍为 0。
- 2026-07-30 v9：完成私钥恢复和精确预检证据链修复。已有私钥目录现在
  必须恰好包含四份分片私钥和一份公开 receipt 副本，逐项拒绝正式公开流、
  design-only 和已泄露 commitment，且目录、文件、receipt schema、自哈希
  与初始化器哈希必须全部精确；符号链接和 Windows 重解析点一律拒绝。新
  精确预检把原始 14 维投影、标签、fold、三模型 OOF 分数、9,999 个
  bootstrap 统计及阶段 checkpoint 逐件落盘并纳入报告清单；登记器重新
  计算 fold、三个 AUC、bootstrap 分位数、随机抽样矩阵哈希和固定哨兵
  replicate，任何已登记证据漂移均失败。分片清单和最终发布清单现显式
  写入机器科研合同引用，最终化器同时拒绝 JSONL 重复键和重解析点成员。
  当前科研实现合同 SHA-256 为
  `14970a98f2f9a7f37223d5b515874206063aa7b8b7102cf88c7b60748cc4dae5`；
  23 项针对性回归通过、0 失败。四份新 500-world 精确预检尚未运行，故
  overlay 登记仍为空，正式私钥和正式数据仍为 0。
- 2026-07-30 v10：第一次按 v9 运行的 train 500-world 报告摘要通过
  0.52/0.53 门，但登记器独立重放原始 checkpoint 时正确停止。原因不是
  模型或数据失败，而是 checkpoint 保留了生产标签表的规范字符串
  `"0"`/`"1"`，新登记器却错误要求 JSON 数字 `0`/`1`；因此 20,000 行
  全部被误判为类型漂移。现按生产标签 schema 要求规范字符串，并继续逐行
  与 int8 标签向量核对；修复后的登记器已能完整重放该旧 checkpoint 的
  其余证据。由于登记器属于稳定科研实现合同，旧 train v5 报告只保留为
  无效诊断，不能沿用或登记，四 split 必须按新合同重跑。当前科研实现合同
  SHA-256 更新为
  `67b8c720e7287fb5742b5417a98be6578a87b657ca58858bc3992713510f660a`；
  正式私钥和正式数据仍为 0。
- 2026-07-30 v11：按 v10 当前合同重新运行 train、development、Audit A
  和 Audit B，各 500 worlds。四份报告均通过 0.52/0.53 门，身份33维秩
  分别为 31/29/30/32，均无全零列；Audit A/B 额外绑定检索阶段证据。
  四份报告及合计 22 个 checkpoint 已写入 overlay 并由配置加载器一次性
  深验通过。当前仍须完成 Step28/full-repo 回归、三路最终复审和 Git
  基线冻结；在这些门完成前，正式私钥仪式与正式数据生成继续禁止。
- 2026-07-30 v12：登记四份报告并同步文档哈希后，Step28-v13 专项回归
  为 90 项通过、1 项旧 execution-blocked lock 按精确父字节漂移声明
  跳过、0 失败；完整仓库回归为 359 项通过、7 项按既有声明跳过、0
  失败。其余 6 项跳过仍来自已撤销的 Step28-v11 测试类。当前代码和证据
  进入三路只读复审，复审与 Git 基线冻结完成前不得生成正式私钥。
- 2026-07-30 v13：最终代码复审发现 v12 登记器只验证已提供 OOF 的内部
  指标一致性，且只重算固定位置的 bootstrap 统计，没有从 14 维输入重新
  拟合三个冻结审计模型，也没有重放全部 9,999 次统计。因此撤销 v12 四份
  报告的现行登记资格，但不删除历史报告。登记器现按冻结 world folds
  重新训练三模型、逐元素核对全部 OOF，并调用同一冻结自举函数逐元素核对
  全部统计；新增非哨兵 bootstrap 伪造和连同 AUC 一起伪造 OOF 的回归攻击
  测试。split writer 同时新增发布前重解析点拒绝。科研实现合同升级为
  `22e136c5bea376aedc68f784b148d4ab67b81216c69ccf34acd836a3710ce601`，
  overlay 精确预检登记已清空。v3 四分区重跑、完整回归和复审尚未完成，
  正式私钥与正式数据仍为 0。
- 2026-07-30 v14：第一次 v3 train v7 运行漏传 `--checkpoint-prefix`，
  主报告明确标记 `checkpointing_enabled=false` 且无 checkpoint manifest，
  因而判为不可登记的历史诊断；未事后补造证据。随后用全新名称完成
  train v8、development v8、Audit A v6、Audit B v9，共 2,000 worlds、
  80,000 条 C40 候选和 22 个 checkpoint。四份分别独立重训三模型并逐元素
  重放全部 9,999 次统计；统一 overlay 加载又在新进程中用 437.6 秒完整
  深验并返回 `PASS_CONFIG_VALIDATION`。四份报告现已登记，正式私钥与正式
  数据仍为 0；完整回归、三路复审和 Git 基线冻结仍待完成。
- 2026-07-30 v15：文档与报告登记同步后，Step28-v13 专项回归为 91 项
  通过、1 项按既有声明跳过、0 失败；完整仓库回归为 360 项通过、7 项按
  既有声明跳过、0 失败。新增的一项是 split writer 发布前拒绝重解析成员
  的回归测试；非哨兵 bootstrap 和伪造 OOF 攻击测试也通过。当前进入三路
  最终复审，复审和 Git 基线冻结完成前仍禁止正式私钥仪式。
- 2026-07-30 v16：三路只读终审均同意进入 Git 基线冻结；文档/指标审核、
  发布代码审核均无 blocker/high/medium/low，科研审核无
  blocker/high/medium，唯一 low 是必须在冻结记录中写明实跑命令、计数和
  环境，本条完成该记录。冻结前实跑
  `python -m unittest discover -s tests -p "test_step28_v13*.py"`：
  91 项通过、1 项既有声明跳过、0 失败；实跑
  `python -m unittest discover -s tests`：360 项通过、7 项既有声明跳过、
  0 失败。环境为 CPython 3.10.11、NumPy 2.2.6、
  scikit-learn 1.7.2、Windows AMD64。生成前基线以 576 个显式白名单文件
  提交为 Git `1a420b309ed269c84bb1c0a9874b3d884ce20469`；提交中无正式
  私钥、正式数据、`AGENTS.md`、论文 docx 或 Step7/Step24 结果。随后从
  已提交字节重新执行标准 `--validate-config-only`，590.5 秒后返回
  `PASS_CONFIG_VALIDATION`，四 split 均为 500 worlds。overlay 现只推进到
  `READY_FOR_KEY_CEREMONY`，仍保持 `generation_enabled=false`、四个私钥
  承诺和仪式回执为空；这允许下一步一次性私钥仪式，但尚不允许正式生成。
- 2026-07-30 v17：一次性四分片私钥仪式在 Git `46d145d` 的
  `READY_FOR_KEY_CEREMONY` overlay 上完成，公开回执 SHA-256 为
  `b421d905e15644d70b92fee9eaea3b653053b25899f8f0ea7b33279523d970d9`。
  回执状态为 `PASS_SPLIT_PRIVATE_KEY_CEREMONY`，四个承诺两两不同且与禁用
  承诺交集为 0，命令未返回原始密钥。私钥目录精确包含四个 split 密钥文件
  和一个回执副本，全部受 `.gitignore` 排除且 Git 跟踪数为 0；这只证明
  本地逻辑保管，不构成 OS custody 或人员盲法。公开承诺和回执现写回
  overlay，状态推进为 `FROZEN_READY_FOR_GENERATION` 且
  `generation_enabled=true`；正式四分片和最终发布清单仍为 0。
