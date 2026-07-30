# AI 科研交接补充：Step 7-v4 / v4.1

更新时间：2026-07-24

本文件只记录 `docs/AI_RESEARCH_HANDOFF_20260719.zh.md` 之后的新事实。原交接文档和 `docs/PROJECT_PROGRESS.md` 已被 Step28-v12 冻结同步清单按字节哈希绑定，不得为更新叙述而改写。

## 当前活动阶段

Step 7-v3 已完成正式 GPU 打分，但没有区分出唯一编码器，也没有证明高分管线中的增益来自编码器。随后发现五个 tokenizer 下均有约六成卖家被 512-token 单窗口截断，因此 v3 只能作为失败的单窗口敏感性记录。其代码、policy、tests、runner 和结果文件已删除，只保留 `docs/STEP7_V3_CLEAN_SOURCE_SELECTION_20260722.zh.md`。

Step 7-v3.1 后来已完成正式 GPU 编码和修正后的折内特征/L2 选择，但没有区分出唯一编码器，也没有证明编码器增量。原始编码器比较中 E5 对 LaBSE 的主口径 AP 差仅 `+0.000309`，分组 bootstrap 95% 区间为 `[-0.040881, 0.043090]`，而且其他口径改由 LaBSE 排第一；完整管线第一名相对无编码器迁移特征对照反而为 `-0.024652`，95% 区间 `[-0.069212, -0.000299]`。这是编码器选择和归因的负结果，不能单独回答哪套完整英文来源流水线应成为当前定义的 M0。

Step 7-v4 已完成并冻结。它不再把 Step 3 卖家摘要误称为完整文本，而是回到两个固定原始快照，对有效边界内每条商品的完整标题和描述做身份删除、无遗漏共享分块和卖家两级聚合。v4 没有找到稳定 M0；作者风格编码器也没有提供可靠增量。

当前最新复核是 Step 7-v4.1。它完全删除 PCM、mStyleDistance 和 `stylometry22`，只复用 v4 冻结的 E5/LaBSE 完整商品分块聚合与 legacy18，在完全匹配的数据和分折上比较 L2 逻辑回归、RBF-SVM 与 LightGBM。所有选择、超参数、阈值和重复外层评估仍只用 401 条 train；已反复查看的 valid 只在两道不可变锁后作描述性诊断，历史 test 标签继续封存。

## 当前 M0 权威定义

当前 Step 7 → Step 28 主线中的 M0 是“只用真实英文马甲标签训练和选择，随后冻结并给 Step 28 提供基础分数的完整英文来源分类流水线”。它不是编码器名称，也不要求编码器必须显著超过 `legacy18`。只要完整流水线能按冻结的同一输入合同在 Step 28 数据上产生分数，`legacy18`、编码器特征及其融合都可以参加 M0 选择。

Step 28 固定比较同一个冻结来源模型：`M0` 不适配，`M1` 加不含新增身份因果信息的匹配对照模块，`M2` 加只用合成中文身份训练集拟合的身份迁移模块。主要方法价值由 `M2-M1` 和 `M2-M0` 判断。编码器相对无编码器对照的增量属于独立消融，只限制“编码器贡献”表述，不是 M0 资格门。

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
`PASS_CONFIG_VALIDATION`，四 split 均为 500 worlds。当前 overlay 只推进
到 `READY_FOR_KEY_CEREMONY`，仍为 `generation_enabled=false`，四个私钥
承诺、正式分片和最终发布清单继续全部为 0。
