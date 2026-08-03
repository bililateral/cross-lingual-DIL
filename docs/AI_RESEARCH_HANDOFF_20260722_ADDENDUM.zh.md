# AI 科研交接补充：Step 7-v4+ / Step 28-v13

更新时间：2026-08-01

本文件只记录 `docs/AI_RESEARCH_HANDOFF_20260719.zh.md` 之后的新事实。原交接文档和 `docs/PROJECT_PROGRESS.md` 已被 Step28-v12 冻结同步清单按字节哈希绑定，不得为更新叙述而改写。
因此 `PROJECT_PROGRESS.md` 顶部仍写 v12 是当时“current line”只属于冻结
历史快照；2026-08-01 的当前事实以本补充和本补充列出的专项合同/结果审计
为准。

## 当前活动阶段

Step 7-v3 已完成正式 GPU 打分，但没有区分出唯一编码器，也没有证明高分管线中的增益来自编码器。随后发现五个 tokenizer 下均有约六成卖家被 512-token 单窗口截断，因此 v3 只能作为失败的单窗口敏感性记录。其代码、policy、tests、runner 和结果文件已删除，只保留 `docs/STEP7_V3_CLEAN_SOURCE_SELECTION_20260722.zh.md`。

Step 7-v3.1 后来已完成正式 GPU 编码和修正后的折内特征/L2 选择，但没有区分出唯一编码器，也没有证明编码器增量。原始编码器比较中 E5 对 LaBSE 的主口径 AP 差仅 `+0.000309`，分组 bootstrap 95% 区间为 `[-0.040881, 0.043090]`，而且其他口径改由 LaBSE 排第一；完整管线第一名相对无编码器迁移特征对照反而为 `-0.024652`，95% 区间 `[-0.069212, -0.000299]`。这是编码器选择和归因的负结果，不能单独回答哪套完整英文来源流水线应成为当前定义的 M0。

Step 7-v4 已完成并冻结。它不再把 Step 3 卖家摘要误称为完整文本，而是回到两个固定原始快照，对有效边界内每条商品的完整标题和描述做身份删除、无遗漏共享分块和卖家两级聚合。v4 没有找到稳定 M0；作者风格编码器也没有提供可靠增量。

当前最新复核是 Step 7-v4.1。它完全删除 PCM、mStyleDistance 和 `stylometry22`，只复用 v4 冻结的 E5/LaBSE 完整商品分块聚合与 legacy18，在完全匹配的数据和分折上比较 L2 逻辑回归、RBF-SVM 与 LightGBM。所有选择、超参数、阈值和重复外层评估仍只用 401 条 train；已反复查看的 valid 只在两道不可变锁后作描述性诊断，历史 test 标签继续封存。

## 当前 M0 权威定义

当前 Step 7 → Step 28 主线中的 M0 是“只用真实英文马甲标签训练和选择，随后冻结并给 Step 28 提供基础分数的完整英文来源分类流水线”。它不是编码器名称，也不要求编码器必须显著超过 `legacy18`。只要完整流水线能按冻结的同一输入合同在 Step 28 数据上产生分数，`legacy18`、编码器特征及其融合都可以参加 M0 选择。

Step 28 固定比较同一个冻结来源模型：`M0` 不适配；`M1` 加五份按
world/C40 分层、整 33 维向量置乱且 pair 端点不重合的保守匹配对照模块；
`M2` 加只用合成中文身份训练集拟合的身份迁移模块。M1 不声称
controller-disjoint，不能描述成严格无身份关联的因果 null。主要方法价值
由 `M2-M1` 和 `M2-M0` 判断。编码器相对无编码器对照的增量属于独立消融，
只限制“编码器贡献”表述，不是 M0 资格门。

旧 Step27 文档中 `M0=真实中文 residual baseline` 只属于旧实验角色；Step 7 历史产物中的 `no_transfer_capable_m0` 只表示编码器没有通过额外归因门，不能解释为当前定义下不存在 M0。完整覆盖说明见 `docs/STEP7_STEP28_M0_ROLE_DEFINITION_20260724.zh.md`，其定义优先于早期文档中的同名符号。

## v3 历史结论与 v3.1 修复边界

- `same_market_bool`、`same_source_dataset_bool` 只作标签关联审计，不能进入任何可选择模型。
- 主 AP 在市场/来源分层内部计算后宏平均，防止分层标签比例或其文本代理制造赢家。
- v3 的原始分层 AP 排名为 GTE `0.733250`、MPNet `0.721820`、LaBSE `0.719920`、E5 `0.708868`、BGE-M3 `0.609077`；GTE 与 MPNet 的 bootstrap 区间跨零，且其他 AP 口径排名不一致，不能声称 GTE 最强。
- v3 的 GTE+迁移特征为 `0.911597`，但无编码器的仅迁移特征对照为 `0.911909`，所以编码器增量未获支持。
- v3.1 的五个原始编码器只比较完整字段分块的“字段等权余弦”；不训练分类头，不加迁移特征。
- v3.1 当时称为“M0 候选”的 `5 × 2 = 10` 项实际是编码器管线消融；两个无编码器模型只用于匹配归因，一个捷径模型只用于审计。该历史候选边界不能覆盖当前定义下的全部完整 M0 流水线。
- LR/L2 使用 Newton + Armijo 回溯，只有归一化最终梯度达到 `1e-9` 门槛才可记录收敛。
- E5 只作连续性报告，不改变排名或保送。
- 英文 test 已被历史开发使用，只能在验证选择冻结后作内部诊断。

v3.1 的历史 CPU tokenizer 预检为 `855` 个卖家、`4198` 个非空字段、`5378` 个共享块；其正式结果和失败结论保留在对应报告与协议中，不再作为当前实现。

## v4 当前冻结状态

- 原始结构审计在不读标签的情况下确认 Agora 两条“卖家”其实是长描述续行；对应独立 valid 组件固定隔离。有效边界为 `109,756` 条商品、`733` 条 pair、`853` 个卖家。
- 正式 CPU public/private 准备已经完成并通过逐项重放：`33,434` 个全局唯一清洗文本；train 保持 `401=116 正+285 负`，valid 为 `151=42 正+109 负`；test 未物化。
- 共享 token 预算固定为 `256`，不超过 LaBSE 的原生窗口；PCM、mStyleDistance、E5 的原生窗口保持 `512`，代码禁止向上改写模型窗口。实际 `SentenceTransformer.tokenize()` 与底层 tokenizer 必须逐块同 ID 哈希。
- `sentence-transformers` 固定为 `5.6.0`；真实 `BatchEncoding` 按映射接口读取。Linux 全量编码前必须逐个真实加载四模型，对最长共享块做两次字节一致的 CUDA 冒烟。
- 最小 GPU 包只含 `7` 个文件、`25,653,396` bytes；14 条禁止路径明确覆盖原始源、v4 私有文件和两份父标签，预期只回传 11 个紧凑结果文件，不发布 embedding 矩阵。
- v4 正式 GPU 编码已经完成：`33,434` 个唯一文本形成 `41,808` 个共享块，其中 `4,226` 个文本需要分块、最大 `23` 块；四个 tokenizer 超预算数均为 `0`，不发布 embedding 矩阵。
- 初次 CPU 选择因 Armijo 目标函数比较落入 float64 分辨率平台而误报未收敛；报错中的 `24.700000000000006` 只是 `24.7` 的正常浮点显示。独立数值补丁不改数据、特征、L2 网格、指标或 `1e-9` 梯度门槛，也不重跑 GPU。正式复跑的 `32,423` 次拟合全部按原门槛收敛，数值驻点兜底使用 `0` 次。
- 训练主口径暂列第一的是 `fusion__legacy_e5_stylometry`，但相对 `control__legacy18` 的分量等权 AP 只高 `0.000939`，95% 区间跨 0；同时超过所有候选的概率只有 `0.395`。去除完整文本克隆并完整重训后，它降到第 9，五个种子胜率为 `0`。
- 匹配的四编码器比较没有唯一赢家，预注册作者风格候选也没有稳定超过简单文体或 E5 对照。旧 valid 锁后诊断同样由 `legacy18` 优于训练赢家，历史 test 标签未读取。
- 正式状态为 `no_stable_unique_provisional_m0`：完整流水线赢家受克隆结构影响且排序不稳定，因而不能冻结任何候选为 M0。编码器和作者风格增量失败是另一个独立结论，不是 M0 资格条件。

当前协议：`docs/STEP7_V4_RAW_ITEM_AUTHORSHIP_SOURCE_SELECTION_20260722.zh.md`。

正式结果审计：`docs/STEP7_V4_RESULT_AUDIT_20260724.zh.md`。

## v4.1 无风格分类器复核

- 候选为 7 组无作者风格特征 × 3 类分类器，加 1 个空对照，共 22 项。主训练与完整去克隆训练分别执行 `43,286` 和 `43,498` 次嵌套拟合，全部通过数值合同。
- 包含克隆的完整 train 上，完整流水线 `LightGBM + legacy18 + LaBSE` 暂列第一，分量等权 AP 为 `0.944873`；`LightGBM + legacy18` 为 `0.939309`。二者差 `0.005564` 且 95% 区间 `[-0.009726, 0.022047]`，只能说明 LaBSE 独立增量未获支持，不能据此取消前者或后者的 M0 候选资格。
- 总体第一名五个种子只赢 `1/5`；相对次优的 AP 差为 `0.002114`，95% 区间跨 0；同时超过全部候选的概率仅 `0.4036`。
- 对相同 `legacy18`，完整集上 LightGBM 的 AP `0.939309` 显著高于 RBF-SVM `0.901967` 和 L2 `0.863341`，证明 v4 固定 L2 的比较范围确实过窄。
- 精确文本去克隆后剩 `286 = 27 正 + 259 负`。被删除的 115 条中有 89 个正例，即原正例的 `76.7%`，精确文本重合是明显标签捷径。原 LightGBM 赢家降至第 10，五个种子胜率为 0；L2 `legacy18+LaBSE` 改列分量 AP 第一，但其 pair AUC/AP 只有 `0.433004/0.259182`。
- 去克隆集只有 8 个含正例的连通分量，且 20 个正例集中在同一个 `40 = 20 正 + 20 负` 的巨型分量。分量等权和 pair 等权指标因此强烈分歧，不能依据其中较高者宣布成功。
- 旧 valid 上最佳 C0 继续优于训练暂列赢家，编码器增量未复现。两个落盘模型对 valid 的重放概率与开标签前盲算文件逐位一致，历史 test 标签未读取。
- 对当前 M0 选择有效的正式状态是 `no_stable_unique_current_best_style_free_pipeline`：完整模型排名不稳定，尚不能冻结 M0。历史字段 `no_transfer_capable_m0` 命名过度，只表示没有编码器候选通过相对 `legacy18` 的额外归因门，不等于当前定义下不存在 M0。

v4.1 正式审计：`docs/STEP7_V4_1_STYLE_FREE_CLASSIFIER_AUDIT_20260724.zh.md`。

下一步数据边界：在收集前冻结少量完整流水线候选，新收集互不连通、无精确文本克隆的真实英文正向身份分量，并作一次性前瞻确认。新的英文确认集直接选择最稳定的完整来源流水线；是否含编码器、以及编码器是否超过 `legacy18`，只作为并列消融报告。

## 2026-07-27：v4.2 新折分种子复跑

Step 7-v4.2 在 Windows CPU 上用五个从未用于 v4.1 的外层种子完整重跑相同 401 条 train、22 个候选、全部内层调参和去克隆消融。它没有读取旧 valid 或历史 test 标签，也没有改变候选、特征、网格或排序口径。

原 operational primary `LightGBM + legacy18 + LaBSE` 在完整集汇总中由第 1 降至第 2；新第 1 为 `LightGBM + legacy18 + E5 + LaBSE`。两者分量 AP 差仅 `0.002655`，95% 区间跨 0，新冠军只赢 `2/5` 个种子。去克隆后原 primary 排第 10，汇总第 1 改为 `L2 + legacy18`，但后者 pair AUC 仅 `0.351780`，并且也未赢得任何单种子。结论仍是 `no_stable_unique_current_best_style_free_pipeline`，当前没有正式选定 M0。

该复跑只增加同数据折分稳定性证据，不是独立数据确认。完整审计见 `docs/STEP7_V4_2_REPEAT_STABILITY_RESULT_AUDIT_20260727.zh.md`。

## 2026-07-27：废弃产物清理

已按显式 allow-list 删除未进入当前依赖闭包的 Step28-v10/v10.1 派生应用产物和本地 Python 缓存。v6–v11 中被 v12.1 同步清单哈希绑定的最小跨版本审计输入继续保留，不能因其历史结论失败而误删。完整清单和原因见 `docs/WORKSPACE_DEPRECATED_ARTIFACT_CLEANUP_20260727.md`。

## 2026-07-29：Step 28 operational M0 与正式中文合成数据

为继续 Step 28，现固定 `LightGBM + legacy18 + LaBSE` 为 operational
M0。这里固定的是从身份脱敏的完整 seller 文本、legacy18、LaBSE 到分类
概率的整条英文来源流水线，不是单独编码器。该选择提供可复现操作基线，
不推翻 Step 7-v4.2 的稳定性负结果，也不得写成“已由独立新英文数据证明
唯一最强”。权威角色定义见
`docs/STEP7_STEP28_M0_ROLE_DEFINITION_20260724.zh.md`。

`dataset_smoke_v3` 只保留为工程开发记录，禁止用于 M1/M2 训练、调参或
正式评估。正式版本由
`docs/STEP28_V13_TRAINING_READY_SYNTHETIC_CHINESE_RELEASE_CONTRACT_20260729.zh.md`
约束：train、development、audit_a、audit_b 各 500 个 world，每世界 28
个 seller、378 个完整 pair 和 40 个机制分层分类 pair。分类标签唯一公式
是 `int(controller(left)==controller(right))`；不得读取英文标签、真实中文
标签、M0 分数或 adapter 结果来造标签。Audit A/B 另各有每世界 4 个查询和
每查询 27 个同世界 gallery，用同控制者公式生成 qrels。

M0 在合成中文上保持冻结；M1 只用 train 内按 `(world,C40/非C40)` 分层、
端点不重合的五份整行身份33维置乱矩阵训练同构适配器；M2 用未置乱的
train 身份33维训练。development 只定阈值和校准，Audit A/B 是固定留出，
不是本机人员盲法。主比较为 `M2-M0`、`M2-M1均值` 和相对最差 M1 seed。

当前仍处于 `IMPLEMENTATION_LOCK_IN_PROGRESS`：正式四把私有结构密钥和
正式四 split 数据均尚未生成。构建器已固定递归发现的 33 文件源码闭包，
任何成员增删或字节漂移都会拒绝运行；所有使用旧构建器字节的精确预检只
是历史诊断。

截至 2026-07-30 最终审核前，历史科研实现合同
`b85798d7d8b446f32847f90aaf9e59db14a0e181164632c3c3ea6c17822ad73b`
下的四份 500-world 精确预检曾全部通过并登记到 training-ready overlay：

| split | 最大对称 AUC | world bootstrap 95% 上界 | 身份33维秩 |
|---|---:|---:|---:|
| train | 0.509703 | 0.516557 | 31 |
| development | 0.508037 | 0.515723 | 29 |
| audit_a | 0.509954 | 0.517763 | 30 |
| audit_b | 0.505748 | 0.513693 | 32 |

四个 split 均低于预注册的 `0.52/0.53` 捷径门，33 个身份特征均有非零
支持；train 的“不得有全零列”强制门已实际启用并通过。矩阵仍有明显共线
和秩亏，后续 adapter 必须使用冻结正则化并报告秩与条件数，不能把单个
系数解释成独立因果效应。Step28-v13 回归当前为 80 项通过、1 项旧
execution-blocked lock 因精确父字节漂移按声明跳过；完整仓库回归为
349 项通过、7 项按既有声明跳过、0 失败。7 项中 6 项来自已撤销的
Step28-v11 测试类，1 项为上述旧 lock。

上述结果不等于正式数据已经生成，也不等于 M1/M2 效果成功。

## 2026-07-30：最终审核 NO-GO 与 remediation

三路最终审核发现四项私钥仪式前阻断：

1. 父 draft 要求的确认性功效 artifact 从未生成，路径与哈希一直为 null；
2. 已有私钥目录的 recovery 只检查内部自洽，未完整拒绝公开、design-only
   或已泄露 commitment，也未强制目录和 receipt 精确 schema；
3. 精确预检摘要未强制绑定 checkpoint、OOF 分数和 bootstrap 原始统计，
   登记器无法拒绝人工伪造但字段自洽的 PASS 摘要；
4. target release claim 被误读成当前 bytes-ready，AP/PR-AUC 和 MAP/MAP@10
   口径也不够明确。

因此私钥仪式继续禁止，overlay 的四份精确预检登记已清空。旧四份报告和
349/7/0 回归只保留为修复前历史证据；源码闭包改变后必须全部重跑。

本 training-ready child 不会为未知的 M1 相关性、world ICC 和 score 分布
填入有利数字来伪造 5,000 次 Monte Carlo 功效。它在任何正式私钥前改为
固定四 split 各 500 world 的估计性设计：报告效应和 world-cluster 配对
bootstrap 区间，禁止“确认性功效已认证”或二元成功声明。固定样本量敏感性
artifact 为
`training_ready_fixed_sample_sensitivity_v1_20260730.json`，SHA-256 为
`cd8464d6efb9be16f98614a785dfefa628e35ea99f533fccc01527831f24a3bc`；
机器科研规则由
`schema/step28_v13_training_ready_scientific_contract.json` 冻结。

C40 同时明确为使用合成 controller oracle 在正负及机制 strata 内抽样的
case-control 设计，不再称为标签盲候选。它没有把 selection 字段暴露给
模型，但 estimand 仅限机制分层合成总体，不支持自然 prevalence、自然校准
或真实地下市场泛化声明。

截至 2026-07-30，本轮 remediation 已完成本地实现闭环，但仍未恢复
GO：私钥恢复会重算并拒绝正式公开流、design-only 和已泄露 commitment，
要求四份私钥加一份 receipt 的精确成员集、精确 schema、自哈希和非
symlink/非 Windows reparse 属性；精确预检现落盘并绑定 14 维原始投影、
标签、fold、三模型 OOF、9,999 个 bootstrap 统计和各阶段 checkpoint，
登记器会重算关键统计。分片/总发布清单也显式绑定机器科研合同，最终化器
拒绝 JSONL 重复键和重解析点成员。当前科研实现合同 SHA-256 为
`14970a98f2f9a7f37223d5b515874206063aa7b8b7102cf88c7b60748cc4dae5`；
23 项针对性回归通过、0 失败。新的四份 500-world 精确预检、完整回归、
三路复审、Git 基线冻结和私钥仪式均尚未完成；正式私钥、正式分片和最终
发布清单仍全部为 0。

随后第一次新 train 500-world 预检的摘要虽通过 0.52/0.53 门，登记器在
独立重放 checkpoint 时拒绝了它：生产标签证据按既有 schema 保存为规范
字符串 `"0"`/`"1"`，登记器却误要求 JSON 数字，导致 20,000 行类型检查
全部失败。现已修正为先要求规范字符串，再逐行转换并与 int8 标签向量
核对；旧 checkpoint 的其余证据已完整重放通过。由于该修复改变稳定科研
实现合同，旧 train v5 报告仅保留为无效诊断，不能登记，所有 split 仍须
重新运行。当前科研实现合同 SHA-256 为
`67b8c720e7287fb5742b5417a98be6578a87b657ca58858bc3992713510f660a`；
私钥仪式和正式数据生成继续禁止。

按科研实现合同
`67b8c720e7287fb5742b5417a98be6578a87b657ca58858bc3992713510f660a`，
四个 split 曾各自重新完成 500-world 精确预检，并由当时的登记器核对：

| split | 最大对称 AUC | world bootstrap 95% 上界 | 身份33维秩 | 全零列 | 历史报告 SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | 0 | `6716cbe2aac9e220f8ae17a090ebe0aa0f9059ad457bea75105d02a198ace72d` |
| development | 0.508037 | 0.515723 | 29 | 0 | `a35373be8b01af52c93c2f726adf8e6a856b99a409f632ec81fe7d32a782b523` |
| audit_a | 0.509954 | 0.517763 | 30 | 0 | `8c7df397df286fd37afcbc3f3727422e55f1a6aac181a2b8b56d9cc5ae62ec6a` |
| audit_b | 0.505748 | 0.513693 | 32 | 0 | `9c856fa701d76f90c4021e1c0f3381a5112fb7899c22b59d1a69b7946c36f4bc` |

四份报告均低于预注册 0.52/0.53 门；train/development 各有 5 个
checkpoint，Audit A/B 各有 6 个，共 22 个。报告登记后
`--validate-config-only` 曾返回通过。但最终代码复审发现，该登记器没有
从 checkpoint 的 14 维输入和标签重新训练三个冻结审计模型，只重算了所
提供 OOF 的 AUC；bootstrap 也只核对固定位置，而不是逐元素重算全部
9,999 个统计量。因此内部自洽的伪造 OOF/bootstrap 仍可能通过，以上报告
现仅为历史记录，不能证明当前实现通过。

随后回归结果为：Step28-v13 专项 90 项通过、1 项旧
execution-blocked lock 按精确父字节漂移声明跳过、0 失败；完整仓库
359 项通过、7 项按既有声明跳过、0 失败，其中另 6 项来自已撤销的
Step28-v11 测试类。这些计数只属于上述历史实现。

当前实现已升级为 v3 科研合同
`22e136c5bea376aedc68f784b148d4ab67b81216c69ccf34acd836a3710ce601`：
登记器从保存的 14 维输入、标签和冻结 world folds 重新训练逻辑回归、
梯度树、RBF-SVM，三个 OOF 数组逐元素核对；再调用同一冻结函数重算全部
9,999 个 world bootstrap 统计并逐元素核对。split 写盘阶段新增发布前
Windows 重解析点拒绝。非哨兵 bootstrap 篡改和连同 AUC 一起伪造 OOF
的攻击测试均已加入。v2 四份报告保留但已从 overlay 撤销。

v3 现行四份精确预检随后完成：

| split | 最大对称 AUC | world bootstrap 95% 上界 | 身份33维秩 | 全零列 | checkpoint | 现行报告 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | 0 | 5 | `628bda9181b9063f2b9978f6d8b2e615f19c115cb0604a647f00976bed5aaec8` |
| development | 0.508037 | 0.515723 | 29 | 0 | 5 | `518dc19d26b82111fa12782863f6a58a07089e9a7a16d2ff3142d9c8e6a54563` |
| audit_a | 0.509954 | 0.517763 | 30 | 0 | 6 | `d9c2a03d32f7ed9316964bfbf9ac0c29bdb4b779b4816392a9699142256af2d6` |
| audit_b | 0.505748 | 0.513693 | 32 | 0 | 6 | `3dbcc4642ca0862e551df3ec5275e9bd3cc376ea6b2db183f0b23ebae23c8700` |

四份均通过 0.52/0.53 门，每份又由登记器独立重训并完整重放；登记到
overlay 后，标准 `--validate-config-only` 新进程用 437.6 秒顺序深验
四份及 22 个 checkpoint，返回 `PASS_CONFIG_VALIDATION`。Audit A/B
各绑定 2,000 个 query、54,000 条 directed relation 和 54,000 条 qrel。
第一次 v3 train v7 因运行命令漏传 checkpoint 前缀，报告明确为
`checkpointing_enabled=false`，只保留为无效诊断，未事后补造 checkpoint；
现行 train 为 v8。Step28-v13 专项回归为 91 项通过、1 项按既有声明跳过、
0 失败；完整仓库回归为 360 项通过、7 项按既有声明跳过、0 失败。三路
只读终审随后均同意进入 Git 基线冻结：文档/指标审核和发布代码审核均无
blocker/high/medium/low，科研审核无 blocker/high/medium，唯一 low 是在
冻结记录中写明实跑命令、计数和环境，现已记录到正式发布合同 v16。576 个
显式白名单文件已提交为 Git
`1a420b309ed269c84bb1c0a9874b3d884ce20469`；其中没有正式私钥、正式
数据、`AGENTS.md`、论文 docx 或 Step7/Step24 结果。从该提交字节重新执行
标准 `--validate-config-only`，590.5 秒后返回
`PASS_CONFIG_VALIDATION`，四 split 均为 500 worlds。overlay 随后以独立
Git `46d145d94026021e8184e9db4a341e85e9d3871b` 推进到
`READY_FOR_KEY_CEREMONY`，仍为 `generation_enabled=false`。一次性四分片
私钥仪式已在该提交上完成；公开回执 SHA-256 为
`b421d905e15644d70b92fee9eaea3b653053b25899f8f0ea7b33279523d970d9`，
四个承诺两两不同、与禁用承诺交集为 0，原始密钥没有返回且私钥目录 Git
跟踪数为 0。现将公开承诺和回执写回 overlay，并推进到
`FROZEN_READY_FOR_GENERATION / generation_enabled=true`。这仍只是本地
逻辑保管，不是 OS custody 或人员盲法；正式分片和最终发布清单继续为 0。

## 2026-07-31：C40 行序失效、v1.2 修复与正式发布

`v13_training_ready_v1_20260729` 生成后，发布后逐行审查发现 C40 最终文件
行序错误复用了选择/补齐 HMAC 排名。由于正负分层池大小不同，文件位置可
预测标签；原 v1 虽有发布清单，也不得用于 M1/M2。第一次行序修复 v1.1
生成了四个 split，但最终比较器把 train/development 的公开统计审计路径
套到 Audit A/B，忽略后者按合同使用 `sealed_supervision/`，因此在 Audit A
拒绝封版，且没有生成发布清单。修正后的只读比较证明 v1.1 数据内容确实仅
改变三个行序相关文件，但旧产物绑定旧封版工具哈希，禁止事后补签。

正式修复版本为
`v13_training_ready_v1_2_order_repair_20260731`，实现合同 SHA-256 是
`ee4d249aaf421d4e6a6603e9e0ae779d9b3c384cef266e5234b0b0918800a831`。
它复用原四把结构 key，保持原 world、文本、C40 成员集合、identity33 和
`pair_uid -> label` 映射，不换 key、不重抽、不删行。新一轮四个 500-world
精确预检及 checkpoint 全量重放均通过：

| split | 最大对称 AUC | 95% 上界 | 身份33维秩 | 报告 SHA-256 |
|---|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | `aecac7e846a49b30053e06e057c7c66bd0bfc72ccc4ed0aaa2e2dd7a1b0551f6` |
| development | 0.508037 | 0.515723 | 29 | `9e51567bacd0dc0a4b471a3a8a4562e03cda73f508d98de3d07ee4e36d2fea53` |
| audit_a | 0.509954 | 0.517763 | 30 | `222835593b6f4f01076eb88556f2c41f29ffb49b7df1eada449bce9d6bf7f4b3` |
| audit_b | 0.505748 | 0.513693 | 32 | `ac810bf8999cec28da6ad0a3503dc9f0200800640b32cc9f60c35915b3faec43` |

四个正式 split 共 2,000 worlds、56,000 sellers、756,000 完整 pair 和
80,000 C40，磁盘约 4.824 GiB；Audit A/B 各有 2,000 query 和 54,000
relation/qrel。最终发布状态为
`PASS_DATASET_ONLY_READY_FOR_M0_M1_M2`。发布清单文件 SHA-256 为
`81ca7d9d2040d500b3bcb2ffc9af6aeb72c581754dbd075b94dd6cf8904b8275`，
父子等价报告文件 SHA-256 为
`3b0d1e29a713ce0402a14170d68936bd9a5b25f745321a5b988b1071817d2998`。
四个 split 都只有
`candidate_pairs.csv`、`candidate_sampling_audit.csv` 和
`world_generation_audit.jsonl` 三个文件字节改变，按主键语义一致，越界
变化为 0；全部跨 split UID、controller 和身份值交集为 0。

发布后逐行审计打开四个 split 共 202,071 条可见商品、56,000 份卖家档案
和 335,016 个计划身份槽，内部生成标记、原始身份 surface、规范身份值残留
及行序错误 world 均为 0。UID/文件位置组最大对称 AUC 为 train `0.508649`、
development `0.512257`；无关元数据组为 `0.505178/0.502243`。正式报告的
24 项 `visible_m0_proxy`（含 5 项 nuisance）在 train 的
gradient tree/logistic AUC 为 `0.570414/0.572021`，development 为
`0.572561/0.577379`；这是允许 M0 使用的共享文本/作者风格基线诊断，不是
真实迁移成绩。私有 AST/oracle 风格禁止模型挂载。逐行报告文件 SHA-256 为
`62bcc36cdfaadc5a257f7bfaaf0915b50b09b3e7902cc14efd349cf47c22390e`。

终审发现原逐行审计没有把整个正式文件树、500-world 精确集合和跨 world
连续块自身绑定成一个门。新增的独立发布树加固审计不导入生成器、预检器、
封版器或原逐行审计，哈希核对全部 160 个正式文件，并用标准库 HMAC
独立重放四 split 共 2,000 worlds、80,000 条 C40 的完整顺序；未解析
Audit 标签、qrel、controller、身份资产或 M1。状态为
`PASS_INDEPENDENT_FORMAL_RELEASE_TREE_HARDENING_AUDIT`，报告文件
SHA-256 为
`a053f71101b320868c8a52658bf78ae18bc3d67e84f71064f536a28d594fa3bd`，
自哈希为
`fbfe00737077f6422d24030e65840c447f9e1f314597ba3664171515f16580a7`。

v3 精确预检中的 `candidate_output_order_audit` 来自构建器返回收据，预检器
只核对固定字典，不是第三套独立 HMAC 重算；“预检通过”不得扩大解释为
“预检器独立验证”。正式 release 的独立顺序证据来自封版器磁盘重算和上述
独立发布树加固审计。冻结修复合同已被 release 哈希固定，不作事后改写，
本段用于收紧其解释边界。

完整结果与结论边界见
`docs/STEP28_V13_TRAINING_READY_ORDER_REPAIR_RESULT_AUDIT_20260731.zh.md`。
当前只证明数据字节可供冻结 M0、五个 M1 和 M2 使用；M0/M1/M2 尚未在该
正式数据上训练/评分，不能声称适配效果成功。Audit A/B 仍只是同一 Windows
用户下逻辑封存，人员盲法为 false。

父子等价证明完成后，已按显式路径清单删除失效 v1、未发布 v1.1 及绑定旧
实现的 order_repair_v1/v2 预检，共释放约 9.68 GiB。删除前把 v1 发布清单、
v1 四份 split manifest 和 v1.1 四份 split manifest 共 9 个文件原字节归档
并逐一验证 SHA-256。正式 v1.2、v3 预检、四份发布后审计、私钥保管目录、
release inputs、跟踪中的 development smoke 和全部历史文档均保留。清理
审计见 `docs/WORKSPACE_DEPRECATED_ARTIFACT_CLEANUP_20260731.md`。

由于失效 v1 完整数据已按合同在等价证明后删除，当前工作区可核验等价报告
和九份父 manifest 的哈希，但不能在不恢复 v1 全部原字节的前提下重新执行
父子语义比较；必须称为清理前完成并哈希封存的历史仪式。当前 v1.2 自身的
全部 160 文件、2,000 worlds 和 80,000 C40 已由上述独立加固审计重验。

提交前静态检查通过。首次全仓回归发现测试夹具把 5-world 设计期迷你构造
误留为 `generation_enabled=true`，生产门因此按预期拒绝；修复只把该测试
夹具显式标为非正式生成，没有修改正式数据、正式生成门或实现合同。针对性
18 项随后全部通过。最终 `python -m unittest discover -s tests` 共发现
381 项：374 项通过、7 项按既有声明跳过、0 失败，用时 798.904 秒。7 项
跳过由已撤销 Step28-v11 测试类的 6 项，以及已被 training-ready 测试替代
的 `dataset_smoke_v3` 不可执行旧锁 1 项组成。

## 2026-07-31：Step28 身份实验执行前锁与可见文本捷径控制

发布后攻击性复核确认，原 24 项 `visible_m0_proxy` 的 development AUC
约 `0.577` 低估了可见文本可学习性。现已把字符三元组 T_text 固定为只用
train 拟合、只在 development 报告的描述性探针；它不读取 UID、
`profile_text`、原始身份 surface 或 Audit A/B。正式结果为 train
`AUC=0.663437/AP=0.559696`，development
`AUC=0.627569/AP=0.346028`。一次性旧探针的
`0.628126/0.347326` 已原样披露，未据此调参。

该探针使用合成中文标签拟合，不能冒充冻结英文 M0，也不能把分数解释为
身份历史价值。生成器私有基础风格约 `AUC=0.91` 是模型不可见 oracle，不能
称作可见文本泄漏。当前处理不是重生成或抹掉作者风格，而是强制 M0、五个
M1 和 M2 在相同 `canonical_pair_uid` 上使用逐行完全相同的冻结 `p0`；
M1/M2 仅可在错配或正确对齐的 33 维身份矩阵上不同。科研结论只认
`M2-M0`、`M2-mean(M1)` 和最差 `M2-M1`，同时要求 M1 平均与 M0 的
90% TOST 区间完整落入 `[-0.01,+0.01]`。

新的执行前锁是
`schema/step28_v13_identity_transfer_experiment_policy.json`，状态为
`FROZEN_PREEXECUTION_CONTROLS_FORMAL_EXECUTION_BLOCKED`。它不改变 v1.2
任何正式数据字节或发布合同；正式 M0 评分、adapter 训练与 Audit 解封仍为
false。控制实现、剩余阻塞项和正式探针哈希见
`docs/STEP28_V13_IDENTITY_EXPERIMENT_PREEXECUTION_LOCK_20260731.zh.md`。
新增控制的 7 项针对性测试全部通过；最终全仓回归共 388 项，381 项通过、
7 项按既有声明跳过、0 失败，用时 819.420 秒。独立发布树加固审计重跑仍
得到原 self-hash，确认本次没有改写 v1.2 正式数据字节。

## 2026-08-01：模型训练就绪判定与直接中文训练基线

当前状态必须写成“数据就绪、正式训练未授权”。正式 v1.2 数据根目录、
160 文件发布树和 2,000-world/C40 顺序已通过发布与独立加固审计，可以作为
模型阶段冻结输入；数据集不需要重新生成。但现行身份实验锁仍为
`FROZEN_PREEXECUTION_CONTROLS_FORMAL_EXECUTION_BLOCKED`，其中 M0 正式
评分、adapter 正式训练、Audit A/B 解封均为 false。因此可以继续实现和
审核训练链，不能直接把一次训练称为正式 Step28 结果。

为正面回答“为什么不直接在合成中文标签上训练”，后继版本化策略必须在
Audit 解封前加入两个强基线：M3-base 复用相同公开预处理、legacy18 和冻结
LaBSE 表示，只用 synthetic train 标签训练目标域分类器；M3-joint 使用
同一 base 特征与正确对齐 identity33 联合训练单体分类器。两者只能在 train
内进行 world-grouped 选择，development 只冻结阈值和校准。若要声称模块
优于直接训练，必须预先冻结 M2 对 M3 的配对 world-bootstrap 比较；否则
只能并列报告数值。

现有 M0 是冻结表示加 LightGBM，不能把它描述成可端到端反向传播的 LaBSE
模型。真正微调 LaBSE 需要另立 M4-encoder，并先冻结可微 pair head、损失、
分块聚合和 GPU 搜索预算；当前未进入正式矩阵。正式训练前仍须完成
label-free 兼容夹具、CPU/GPU 策略、冻结 M0/C0 打分、M1/M2 求解器与完整
指标/bootstrap、M3 实现、Audit 授权链和提交后重验证。完整最新启动门见
`docs/STEP28_V13_MODEL_TRAINING_READINESS_20260801.zh.md`。

## 2026-08-02：full-378 v1.4 写盘失败，v1.5 合同已冻结

本节覆盖上面的 v1.3/v1.4 执行状态。v1.4 train 已完成 500/500 world 内存构造，但在写盘复核时失败：目标 JSONL 的绝对路径为 264 字符，原子写入使用长路径接口而直接 `Path.read_bytes()` 没有，因此文件实际存在却被误报缺失。只读复核又发现 split manifest 生产者写 v2、消费者仍要求 v1。train 未发布，development/Audit 未启动；v1.4 四份 seed、失败 stage 和 invalid marker 原样保留且禁止重试。

后继 `v13_training_ready_v1_5_20260802` 已修复长路径读/遍历/stat，并把 build manifest 和 execution lock 的版本分别收敛为单一权威常量。v1.4 失败 train 的 42,000 个身份值以哈希进入 `base_exclusions.v5`；与既有 283,496 个禁用哈希交集为 0，总计 325,496。v1.3/v1.4 共八个 master commitment 均进入禁用集合。

v1.5 两-world 预检、完整临时 public/private 写盘、manifest 消费、超 260 字符 Windows 路径和全部 Step28 后半段回归均通过。正式 Windows `Validate` 为 480 项：473 通过、7 声明跳过、0 失败。identity design self-hash 为 `d3515e7791f164b7a6a2ac55345c0306cec2fc82dcce2800a7649c0d58c67be5`；prelock self-hash 为 `04636dcf45bf295111d434aadf240425af451c268501073bee9eea094b21c589`。四份 seed 已按固定顺序一次性提交，原始密钥未公开；split、模型和结果仍全部为 0，下一步是 train/development 正式生成。

## 2026-08-02：v1.3 正式生成失败，v1.4 修复完成冻结前验证

本节覆盖上节“合同与随机性边界已冻结”的当前性。v1.3 四份一次性 seed
承诺已按序产生，但正式 train 在完成 500/500 world 的内存构造后被最终
聚合校验器以 `Mechanism assignment lineage collision` 拒绝。没有 split
发布；development 与 Audit A/B 未生成。私有失败标记与公开 seed 回执保留，
v1.3 按 one-shot/no-retry 合同永久 fail closed。

根因是 `mechanism_slot_uid` 表示每个 world 复用的模板槽名，旧校验器却要求
它跨完整 split 全局唯一。v1.4 只把该唯一键改为
`(world_uid, mechanism_slot_uid)`，并保留所有真实实体的全局唯一约束。新增
同 world 重复槽负测，以及两个真实 K28 world、756 pair 的端到端生成和
二次字节重放硬门；回执 self-hash 为
`ce77646a2b9314c55096434829e38f5f72a79bcbd1ff939b040585ee0f4622e0`。

新目标为 `v13_training_ready_v1_4_full378_scope_repair_20260802`。候选 identity
design self-hash 为 `6b629f139b34c464bc907cbe77f91488b4a39f82f1a5ae1f2a9bf7a83256f31f`，
draft prelock self-hash 为 `1ba6770eb09c65a4b150899fdd7e5885010fefae1027d8ec435ebf1978385944`。
冻结前全仓回归为 476 项：469 通过、7 声明跳过、0 失败。v1.4 正式
identity policy 与 prelock 已正式冻结；四份 seed 承诺已按固定顺序完成且原始
密钥未公开。v1.4 split、模型和效果目前仍为 0。

## 2026-08-01：训练前攻击复核撤销 v1.2/v2 启动资格

在任何正式 M0/M1/M2/M3 数值产生前，三路独立攻击复核发现三项足以阻断
训练的问题。第一，base policy 公开的 candidate key 可从可见 world/pair
UID 重算 C40 分层抽样；攻击在 train/development 的 AUC 约为
`0.89/0.86`。第二，即使候选 key 改为私有，40 条 C40 边形成的子图仍泄漏
类别；仅使用端点度数在 development 的 AUC 约为 `0.63`。第三，v1.2
Audit A/B 的 controller membership、机制、身份资产和候选审计已被 Git
跟踪，旧真值可以连接回任何复用 UID/文本/卖家历史的后继版本。目录名
`sealed_supervision` 不构成真实密封。

因此 v1.2 冻结字节和历史文档不改，但状态降为
`BLOCKED_PUBLIC_CANDIDATE_AND_AUDIT_CUSTODY_COMPROMISE`；v2 训练链从未执行，
现已禁止启动。后继版本固定为四 split 全量重生，并把每 world 的全部
`C(28,2)=378` 对作为主分类总体。每 split 为 500 worlds、189,000 pairs、
10,000 positives 和 179,000 negatives，随机 AP 基线为 `20/378`。完整图使
每 seller 度数恒为 27，从结构上删除 C40 选择与候选图捷径。C40 只能在所有
预测冻结后作为私有机制诊断，不能参与训练、阈值或主指标。

新 Audit A/B 必须使用全新 structure/UID/text/identity 随机域，从零生成且
相对 v1/v1.2 的 UID、身份值、卖家文档/profile/history 指纹不可连接；
labels、qrels、oracle 和逐行私有审计只进入 Git 忽略的 private custody。
五个 M1 改为完整 378-pair universe 内、不读标签的整 33 维端点不重合双射；
旧 C40 截距、L2、阈值、class weight、M1 分层及 20,000-row M3 配置全部失效。
正式修正案见
`docs/STEP28_V13_FULL378_FRESH_RELEASE_PREEXECUTION_AMENDMENT_20260801.zh.md`。
完成新数据、逐行捷径审计和 full-378 执行闭包前，当前状态始终为 NO-GO。

## 2026-08-01：Step28-v13 v2 执行实现完成，可以开始计算

上节“正式训练未授权”是 v1 锁当时的准确历史状态；不得删除或改写。其
后继 v2 已把全部登记 blocker 实际实现并另立新锁：
`schema/step28_v13_identity_transfer_experiment_policy_v2.json`，状态为
`FROZEN_PREEXECUTION_IMPLEMENTATION_COMPLETE`，canonical self-hash 为
`dc5be1258f5379864fb55eeb6493ce3f495760db05a245650fa0eb5c9dbd0c92`。
v1 文件仍保持 blocked 原字节，v2 只继承并关闭 blocker，不修改正式 v1.2
数据或历史结论。

已冻结并验证的执行链包括：

1. 英文 M0/C0 joblib、真实英文盲回放参考和 756,000 完整中文 pair 的
   label-free 公共投影；
2. 同时适配四个冻结 tokenizer 的完整文本分块，以及只接收匿名文本、匿名
   pair 和 LaBSE 的 Linux CUDA 工作区；
3. Step7 历史 LaBSE 兼容夹具。分块、重复 embedding 和历史六项聚合的
   12 位小数必须全部相等，任何一项不等立即失败；返回 bundle 同时绑定
   当前 GPU 策略和编码脚本哈希，禁止混入旧版本结果；
4. 冻结 M0/C0 前向评分；两个 base ×（五个 M1＋一个 M2）的 12 个
   train-only 33 维加法适配器；M1 只在 train 使用错配历史，推断时与 M2
   读取同一份正确目标 identity33，防止测试阶段人为压低 M1；
5. 四个 train-only 直接目标域 M3 LightGBM 对照，每个在 36 个候选中做
   5 折 world-grouped OOF 选择；M3 只持久化 JSON 兼容元数据和 LightGBM
   原生模型字符串，写盘前验证其预测与训练封装器 float64 字节完全相同，
   审计环境不反序列化 sklearn `1.7.2` estimator；
6. development-only 阈值（不拟合概率校准变换），AP/AUC/PR-AUC/F1/Recall/Precision 等分类指标、
   MRR/MAP@10/Recall@K/Hits@K 检索指标、hard-negative FPR 和 9,999 次
   配对 world bootstrap；
7. 在 Audit A 标签打开前共同冻结 A/B 盲预测；A 不通过则 B 保持封存，
   只有 A 通过才生成一次性 B 授权。

为补足原 release 中未公开的 hard-negative 诊断输入，已在不开分类标签、
qrel、controller 或机制标签的条件下从冻结观测数据重放，并把 A/B 各
20,000 行投影保存在 Git 忽略的 private custody。公开 v2 承诺 self-hash 为
`fede8eab3411e8cf61c03da29669efaa2cff32c2dd4630c05321dd66ac510675`；A/B
CSV SHA-256 分别为
`27a6f6127c4659c5f89d172a193fce7839aebcf249a08d604dbdee2fa0204530` 和
`f1df37a96c45353c76e59b1e6da3a906ea792905e0d2492648c976d546557538`。

执行前总验证器
`scripts/step28_v13_validate_identity_experiment_v2.py` 已逐文件核对正式
release、公共投影、夹具、英文模型依赖、诊断承诺和 25 项双向源码/传输
闭包，返回 `PASS_STEP28_V13_IDENTITY_V2_PREEXECUTION_CONTRACT`。其中
`.gitattributes` 也进入闭包，唯一 PowerShell runner 固定 LF，避免提交后
CRLF 转换造成跨 Windows/Linux 假漂移。

最终冻结实现上，40 项 Step28-v13 专项合同测试全部通过；全仓回归共 421
项，414 项通过、7 项按既有合同跳过、0 失败，用时 1103.765 秒。另用训练
环境生成仅含原生 LightGBM 字符串的 M3 夹具，再在 sklearn `1.7.1` 审计
环境以 warning-as-error 方式加载和预测，跨版本持久化检查通过。

当前科研状态必须准确表述为：**可以开始正式计算，但尚未计算。** Linux
LaBSE bundle、冻结中文 M0/C0 分数、M1/M2/M3 模型和 Audit 结果均尚未
产生。当前同一 Windows 用户逻辑封存仍令
`independent_blind_confirmatory_claim_authorized=false`；即使合成 Audit
通过，也不能替代新收集的真实中文 ground truth 或宣称真实市场外部有效性。
执行命令、环境和同步范围见
`docs/STEP28_V13_MODEL_TRAINING_READINESS_20260801.zh.md`。

## 2026-08-01：full-378 v1.3 实现终审完成，正式仪式尚未开始

本节是当前最新事实，优先于上节旧 v2“可以开始计算”的历史表述。旧
`schema/step28_v13_identity_transfer_experiment_policy_v2.json` 已因 v1.2
数据资格撤销和运行器源码漂移而 fail closed；旧私有诊断投影与过期 v2
执行树已清理，不得恢复为当前输入。v1.2 冻结字节和历史文档继续保留，
但没有 M0/M1/M2/M3 正式训练资格。

fresh 后继版本固定为
`v13_training_ready_v1_3_full378_fresh_20260801`。数据生成、完整 378-pair
主分类、全 28-query 检索、五份 train-only M1、M2、M3-base/M3-joint、
development M1 等价门、Audit A/B 顺序解封及全部指标/bootstrap 执行链
已经实现。Audit 数据生成另有一次性授权：必须先完成 train/development、
五份 M1、M1 独立性和公开捷径门，且 B 文本只能在 A 文本完成后生成。

终审新增三层深度重放回执。development 必须从冻结模型重算逐行预测、阈值、
指标和 9,999 次等价 bootstrap；Audit A/B 盲预测必须在真值打开前重算全部
分类与检索概率；每个 Audit 评估必须重算分类、检索、私有诊断和所有
bootstrap 数组。只验证文件哈希而不重算已不满足合同。Audit B 授权还必须
绑定 Audit A 的深度重放回执。

正式 identity design self-hash 为
`0b2ae09ed17cd68e313f4add60960f8e9fa192c0bd56ccb1a7702db7687ba5c1`，
75 项模型源码闭包已经冻结；正式 prelock self-hash 为
`23faeeda89cae653af8f1a2d363f436341bb826d9502bf676aa73a051130ccac`，
50 项数据源码闭包通过 config-only。正式 Windows `Validate` 入口实际运行
472 项，结果为 465 项通过、7 项既有声明跳过、0 项失败，用时 792.518 秒。最初 7 项失败/错误全部是
清理后仍要求旧结果、旧私有诊断或旧 v2 runner 的过期测试，现已改为验证
旧链 fail closed，没有恢复任何废弃产物。

最重要的当前边界：正式 identity policy 与 preceremony lock 已落盘；四份
master seed 承诺已按 train、development、Audit A、Audit B 固定顺序完成且
禁止重抽，原始密钥未公开。四个 v1.3 split、M0 分数、模型和效果报告仍为
0。当前只能称为“正式合同与随机性边界已冻结”，状态仍为 **NO-GO**。下一步
严格执行 train/development 生成与审计、Audit 文本单独授权、Audit A/B 生成、
M0 Linux 编码、Windows 训练和分阶段 Audit。最新启动说明见
`docs/STEP28_V13_MODEL_TRAINING_READINESS_20260801.zh.md`。

## 2026-08-03：full-378 v1.3–v1.11 全部关闭并清理

本节是当前最新事实，覆盖上文仍把 v1.5 或其他后继写成待执行版本的历史
状态。v1.3 至 v1.11 均已永久失败，当前不存在可用于 M0/M1/M2/M3 的 fresh
full-378 正式数据集，也没有任何正式模型结果。v1.11 虽生成四个 500-world
split，仍在发布前跨版本精确审计失败：审计器把 world 内模板槽
`mechanism_slot_uid` 错当成 split 全局唯一键。每个 split 的正确复合键
`(world_uid, mechanism_slot_uid)` 均为 6,000/6,000 唯一，world 内重复为 0；
这证明根因是审计合同作用域错误，但 one-shot 纪律仍禁止原地修补或发布。

删除前已把历代及 v1.11 实际生成的身份值压缩成 915,996 个不可逆禁用哈希；
文件 SHA-256 为
`f70611a4b5df7ddbded6784820026352c92952a0245fcb184b4e7c282c1447a0`，
self-hash 为
`6f60e294bdbcae1d1da3802acdf4095d605c8e633755fe689d4a71b67fece4d3`。
其中 v1.11 实际新增 170,500 个，不是先前估算的 168,000：Audit B 为
44,500，另外三份各 42,000。归档不含原始身份值或私钥。

失败 private custody、stage、失败版本专用代码和配置已按
`docs/STEP28_V13_FAILED_RUN_CLEANUP_20260803.zh.md` 的边界清理。成功 v1.2
发布树继续作为历史证据保留，但训练资格仍被撤销。后续不得通过恢复失败
payload 重建排除库；新版本只能读取并校验压缩哈希归档。当前主线在清理后
自主暂停，下一次 fresh 设计必须另立版本、四份新 seed 和新的冻结闭包。
实际删除约 20.11 GiB private custody，并按哈希清单删除 223 个失败版本专用
脚本、配置和测试；清单 SHA-256 为
`c6fbe552a67e533165e43d991ce5f2b2bee312ebf9455139ed61928159dccf19`。
清理重放通过；全仓回归为 381 项、374 通过、7 声明跳过、0 失败，用时
848.857 秒。

## 2026-08-03：v1.12 干净重启仅通过预仪式设计基线

本节是当前最新事实。v1.12 已从压缩失败历史重新起步，不恢复 v1.3–v1.11
已删除 payload，也不沿用其 seed。策略
`schema/step28_v13_v1_12_cleanroom_preceremony_policy.json` 的正式 seed、
正式数据、模型训练和 Audit 解封授权仍全部为 false；实现本身也明确不包含
正式 seed ceremony 或四 split 正式生成器。

非正式两世界预检共重放 756 条完整 pair、40 正/716 负、168 个身份资产、
756 行 identity33 和 10 张 M1 映射。915,996 个失败身份哈希、90 个禁用
master commitment、逐资产碰撞推进、Windows 长路径、world-scoped 机制槽、
最终 self-hash、登记后缀投影、join UID 分域和单一成员合同均通过。回执
SHA-256 为
`d21964a248e1138e65a654262e026c8c1457f8500e4915dbf5b83cdaba09d243`，
状态为 `PASS_DESIGN_ONLY_NO_FORMAL_AUTHORIZATION`。

全仓回归现为 396 项：389 通过、7 项既有声明跳过、0 失败，用时 809.034 秒。
当前正式数据行、正式密钥访问、科学指标和模型训练均为 0。下一步必须实现并
验证正式 custody/seed 流程、四 split 生成与发布审计，以及基于完整输入的真实
优化器收敛预检，再另行冻结正式授权；在此之前不得开始正式中文合成数据生成。
完整审计见
`docs/STEP28_V13_V1_12_PRECEREMONY_BASELINE_RESULT_20260803.zh.md`。
