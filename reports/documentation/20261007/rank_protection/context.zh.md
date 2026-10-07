# 本轮方案审议入口与完整背景

2026-10-07。本轮是新方案讨论／外审，不是实现验收或训练启动。先读docs/SELLER_ALIAS_RANK_PROTECTION_PROPOSAL.zh.md，再按下列入口核实背景。

## 用户当前决定

最新原话：“继续按之前的要求修改，我允许你按你自己的想法来修改，可以修改算法核心，但必须经过我和外审审批。有问题就问”。用户允许提出核心变更，先前不得改关系目标记忆核心的限制在本轮被这一新决定取代；具体机制、公式、执行预算尚未获用户批准。用户与外审双审必须满足，不把本次讨论回复当成正式训练许可。

此前性能目标为超越LOGIT0.1，不要求所有指标全胜；用户进一步选择“旧域保持为主，兼顾整体排序提升”。论文目标仍需实质方法贡献，不接受只把已有方法换名字。研究对象是旧分布未见卖家与新群体学习，不是记住训练控制者或追踪新别名。

## 证据地图

1. 当前规则、研究状态、全部最新用户决定：docs/RESEARCH_DISCIPLINE.zh.md、AI_RESEARCH_HANDOFF.zh.md、RESEARCH_DISCUSSION.zh.md。
2. 最新结果、你刚完成的外审及主审：docs/SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md；reports/seller_alias_continual/20261007/function_memory_result/review/report.zh.md、recovery.json、submission.json。此次九权重已依用户10:04:38指令删除，无加载／恢复依赖；留代码和所有小证据，不以外审旧保管快照撤销这一事实。
3. **history/function_memory_result_external_evidence.zip**：你刚交付的原始结果外审ZIP，18,615,222字节，SHA256 170424de77f259b973bdf6e29bc0aadde76501eceeff337824bb36b7b97f5c8a。含43,908字节完整原文、五份独立脚本／输出、全部统计和原始输入original_input.zip。主审已验证39成员／38载荷；本轮可复用，不必重跑。
4. 上述original_input.zip为18,320,431字节、SHA a0c8ae95371cbf1ff0667171f6e0909b56509ddadda9601263093b57d1d7a72d，含最新171项非权重清单中的162本体、23正式冻结来源、全部结果／日志／指标／分数／计数、LOGIT0.1对应矩阵以及历史入口。省略9 Memory本体和模型，不含正文／逐对标签／凭据。
5. original_input.zip中history/function_memory_implementation_external_evidence.zip为13,204,125字节，SHA545ec223fcf2b7408568310a19589c8fc203a721ab8a015e48c9d8435f3a589a，含完整函数记忆实现外审、R1/F1原件及更早输入／方案／结果审查。仅按需要回溯，不递归重跑全部历史。
6. LOGIT0.1实际已执行来源37项放在reports/seller_alias_continual/20261004/logit_low_result/source/。同时附该结果inventory.json和实际共享Memory／update／runner等活动来源供追踪。**LOGIT的update实际在step28_er_weight.py**，不存在独立step28_logit_weight.py；来源eval、学生当前与历史train，旧教师来自最初入缓存阶段且存活时不刷新。
7. 文献定位：docs/SELLER_ALIAS_LITERATURE.zh.md及新方案第5节。本轮主执行者仅对四项一手来源作有限核查，未认证新颖性；不能把检索没找到当原创性证明。

## 需要你作出的判断

对首选“完整LOGIT0.1＋教师正确次序的截断单向最坏违反保护”作严格设计审查，尤其是：它是否仅为已知hard negative／margin distillation的普通组合；新项能否有独立作用；与保留MSE的张力、Dropout差异和六群泛化风险；是否值得交用户批准做一个固定候选。

允许否决首选；若否决，最多提出一个你认为更有依据的替代，须有完整计算定义、资源／可见信息及相对LOGIT的差异，并解释所针对的是已证实现象还是假设。不要为了看起来新颖增加一组无关组件，不强迫维持已被用户允许修改的旧累计核心，不复活已关闭group-meta。若替代需要改6群、1MiB、数据选择权或计算预算，显式标出新审批项。

## 执行边界

本轮网页仅讨论、查一手文献和读取已有附件。不执行项目脚本，不运行新的数值测试／玩具实验，不训练、不载模、不索取文本／标签／Memory／权重，不接服务器，不自动追加候选。数学推导与手算反例可以给出并明确其性质。现有保存证据可阅读；已有完整结果核验不重复执行。产出详细中文方案审查原文和可下载证据ZIP，包含实际阅读范围／文献来源、结论、必要修订和未决问题。没有执行就明确说没有执行，不伪造测试结果。

新方案尚未实现。旧实现／结果审查是历史已关闭证据，本次不能称“新算法实现已通过”。主执行者会独立处置后将收敛方案和具体CPU／GPU／正式训练合同呈用户审批。
