# 保留关系目标记忆核心的修订外审上下文

2026-10-06，Asia/Shanghai。用户最新原话：“保留这一核心，这是我们算法区别于基线的根本，这个改了，那我们的算法跟基线有什么区别呢？”随后：“那现在开始正式的修改吧，完成后送交外审，一定要交代清楚上下文”。用户另外明确授权本轮单CPU、10分钟累计墙钟、RSS2GiB、证据16MiB的手写核验，禁BGE/GPU/正式数据，失败停止不自动重试。此次上传对象是本轮实现、核验及直接相关的完整项目上下文；旧实验许可不转移。

## 当前审查对象

先完整阅读 `docs/SELLER_ALIAS_RELATION_REVISION.zh.md`，再追踪 `scripts/step28_relation_revision.py`、新测试和CPU监督入口，以及它们调用的原关系记忆和排序依赖。新阶段入口已用手写微型编码器实际执行，正式数据runner尚未接入。原关系核心、原正式runner和冻结policy没有改动；不要把新实现误认成原固定点实际运行来源。

新目标是当前BCE＋query-rank＋0.5已知top5-hard，历史原压缩二次Q权重1，加0.1完整查询logistic排序R。保留原关系头、d32、Dropout=0、epsilon=.001、存活参考刷新、Algorithm R、H/b/c/N迁移累加、精度及阶段流程。R是缓存真实标签监督，不是新教师蒸馏，不新增参考分数，绝不以移除Q或回归LOGIT0.1代替修复。0.1是无效果调参的待审固定点，不是证明过的最优权重。

当前B与历史平方代理已分开，统计精确性仅对定义的固定表示平方代理成立。R能看到部分Q投影盲区中的排序变化，不等于解决全部盲区或缓存外泛化。原Q的固定间隔压力、岭偏差、单群/联合映射差异和迁移累积误差继续存在。此次原生BGE新目标资源及新效果未验证，不能复用旧GPU证据冒充新目标准入。

## 阅读顺序和必要原件

1. 问题/纪律/用户决定：`docs/RESEARCH_DISCIPLINE.zh.md`、`docs/AI_RESEARCH_HANDOFF.zh.md`、`docs/RESEARCH_DISCUSSION.zh.md`。注意最新指令的时间范围，历史待许可、旧关闭和旧继续条件不能覆盖新修订授权，也不能授予新数据权限。
2. 新实现与证据：修订设计、新源码/测试、`reports/documentation/20261006/relation_revision/cpu/`。其中manifest是实际19份CPU来源，recovery_manifest核对5份回传载荷。`result.json`是Linux监督结果，`unittest.log`有全部5项实际结果，`wrapper_time.txt`计完整11.42秒。无正式数据或BGE；两阶段576次是微型手写更新。
3. 原设计和数学局限：`docs/SELLER_ALIAS_RELATION_MEMORY.zh.md`及EXECUTION/PILOT；`reports/documentation/20261004/relation_memory/external/page.json`为设计外审完整网页原文，external/unpacked含独立方程和有界盲区反例；该目录report为原主审及文献定位。不要把旧“未实现”文字当当前状态。
4. 原实际实现和准入：同目录implementation下原始实现外审、disposition、cpu_repair/gpu记录；pilot下接入外审/处置；`reports/documentation/20261005/relation_memory/`的入口最小修复和CPU证据。本轮不重审无变化的旧编排，但不掩盖新模块尚未接正式编排。
5. 为什么修订：`docs/SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md`、`reports/seller_alias_continual/20261006/relation_result/`的冻结source、实际job、完整指标/分数/计数、Linux保存结果复核；review下原始结果外审完整报告及主审勘误。这一结果有效，六条件0/6；本轮不改判。首阶段三域已较弱，最终旧域下降不等于所有域都遗忘更多，ΔF MAP条件区间跨0、CAB/C自身改善等反例保留。
6. 基线和历史：`docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md`、LOGIT_LOW及LOGIT_RISK_RESULT、`reports/seller_alias_continual/20261004/result_review/`原文。基线B的实际实现和0.1监督＋独立0.5MSE见包内旧源码。risk负结果保留，group-meta资源受限关闭，不能借本轮恢复。完整基线和候选保存矩阵沿用上轮272成员上下文的载荷，足够追溯旧结论。

## 独立判断与主张边界

研究旧域未见卖家的排序/基础识别与新群学习，仅中文合成受控标题/描述；三顺序不是三种子，valid已开发，不外推真实市场。48/12/20划分、28账号/378对/20正边、六原始群且1MiB等历史约束均在原合同。本轮不读test、owners、Audit真值、private_custody。包中没有正式正文、逐对真值、真实缓存、模型权重、凭据；手写标签明确是人工夹具。

新五项CPU检查不能代替科学判断。尤其统计扰动是内核夹具：证明Q可改变梯度和实际更新，不证明两份真实数据历史可产生相同缓存与指定统计，也不证明Q的效益。R的手写反例覆盖是局部；串行/联合核验共用生产组件的部分应与独立标准库目标、独立图和历史数学证据一起判断，不能计数认证。旧完整状态/BGE恢复证据仅适用于未改部分；新增R无持久状态，但新完整正式链仍待后续接入。

请区分数学/实现阻断、当前交付复现缺陷、合理后续建议、范围外扩展。不因核心保留便宣称创新，也不因组件已有便自动否定组合；需指出具体重合、贡献缺口和必要后续归因，不要求本轮启动全部消融。不能承诺超过基线。审查不授权Linux新预算、真实数据读取、正式训练或权重恢复。

## 包身份与历史时间

根 `manifest.json` 是本次完整载荷清单，按本次实际字节核对。旧 `context_inventory.json`、package/submission及可变docs旧哈希是历史送审记录，不要求当前交接与旧快照字节相同；冻结source和科研原件不改。旧结果文档“待审/权重保留”由后续review/report及当前交接解释，实际九权重已清理。这里提供完整原件是为了可追溯，不是让审查者无差别重复审核所有历史实验。

请保存完整中文审查原文及真正执行的独立检查代码/输出，提供带清单的可下载ZIP。没有Torch/BGE或未执行的检查如实说明；网页运行与Linux证据严格区分。请求网页GPT 6 Pro，前端可见6/Pro，后台身份未独立认证。
