# 有限历史数据下的马甲候选排序：文献重叠与论文定位

初次外审修订：2026-09-27；定向补充：2026-09-30。本页保留原文献外审及主审结论，补入当前数据版本、用户已选择的历史供给方式和FDR直接近邻。尚未建立方法创新，也不是新实验合同。[原实际回复](../reports/seller_alias_papers/20260927/continual_position/review_reply.json)、[原逐项主审处置](../reports/seller_alias_papers/20260927/continual_position/disposition.json)。提交与等待历史保留在[原回执](../reports/seller_alias_papers/20260927/continual_position/submission.json)，9月30日补充范围见[当前方案主审](../reports/documentation/20260930/continual_plan/review/disposition.json)。

用户要求：整理直接相关文献，判断研究问题能否形成论文、与已有研究是否重叠。投稿目标、历史数据较少的前提及先完成受控主线的范围，沿用[用户回答记录](RESEARCH_DISCUSSION.zh.md)。当前文献编号S01—S17、完整题名及本地阅读范围见[复核来源目录](../reports/seller_alias_papers/20260927/continual_position/review_sources.json)；[初次来源清单](../reports/seller_alias_papers/20260927/continual_position/sources.json)按原字节保留为历史，不替代本次修订。

## 1. 当前判断

**该问题有继续发展为投稿论文的价值，但已有研究重叠明显；目前尚未建立可主张的方法创新。**

1. 用商品文本关联暗网卖家账号已有直接工作。
2. 在分布变化、历史记忆有限时保持检索能力，已有持续文本检索和终身重识别研究。
3. 多文本身份表示、选择困难／多样旧样本、按身份保存、保持旧关系或排序，以及利用共享对象选择训练样本，均有先例。把这些方法迁移到中文合成马甲任务，不自动产生新的算法贡献。
4. 多条商品文本构成账号、同控／异控关系监督、新阶段身份不重叠、旧分布**未参与训练的卖家**评价，是本项目的重要设定。多记录和未见身份本身已有直接先例；这些条件的交集也不能证明需要新算法。当前还没有一个完整的新机制可以与最近方法作算法等价或差异判断。

“值得继续”是执行者基于下文文献的判断，不等于“已经足够投稿”，也不是CCF B／C录用预测。本轮定向检索未发现完全覆盖上述所有条件的同一方案；检索不穷尽，任务条件的交集也不能代替方法新颖性论证。

## 2. 我们究竟研究什么

研究问题可暂表述为：

> 在可用历史训练数据较少、新卖家群体分批到来的条件下，如何学习新的中文合成卖家分布，同时保持旧分布留出卖家的马甲候选排序与基础识别能力？

模型只看到屏蔽后的商品标题／描述。它学习两个账号的同控分数，用来排列同组候选；没有不断扩大的卖家身份分类头。当前是每查询27候选、保证1—2个同控候选。不同阶段不是同一控制者不断新增马甲的跟踪任务；旧能力评价也不是重新识别训练时记住的旧身份。

历史数据较少的背景已经确认。2026-09-30用户进一步选择：**首轮按模型无关的预定随机规则保留最多6个完整拟合群，历史样本和附属状态总计≤1MiB。** 两个重放方法使用同一成员与抽取序列，淘汰后不读回；这不授权算法自主选择有利历史。下文有关学习记忆选择的贡献设想仍以另行允许该选择权为条件。[实际回答](../reports/documentation/20260930/continual_plan/clarifications.json)。

后续比较须披露能够访问的历史原始输入、监督关系、历史输出／教师模型及其他持久统计。相同的最终样本数或字节数，不代表获得了相同的历史选择机会。首轮已选机制之外的容量和访问方式仍需具体合同。

基础排序、校准及独立test已完成；用户接受实际增益作为后续基础，原事前test合同8/9不改。当前处于[持续学习方案审议](SELLER_ALIAS_BGE_CONTINUAL.zh.md)，尚未运行新BGE持续学习实验。具体决定见[讨论记录](RESEARCH_DISCUSSION.zh.md)。

## 3. 最直接的近邻与重叠

| 原始论文 | 已经做到的内容 | 与我们的差别及对创新的约束 |
| --- | --- | --- |
| [S01 VendorLink，ACL 2023](https://aclanthology.org/2023.acl-long.481/) | 以同一卖家的多条广告及卖家间广告相似度构造归一化账号相似度，关联潜在马甲，并研究低资源市场适应。 | 应用目标及多记录账号证据汇总均有重叠。所读任务没有把有限历史信息下的顺序旧／新能力保持作为核心比较；换中文编码器或合成数据不能单独建立新方法主张。 |
| [S02 L²R，CIKM 2023](https://jiafengguo.github.io/2023/CIKM2023-L2R.pdf) | 为训练查询保存有限旧负例，以正样本优势度及查询条件下的多样性选择、更新记忆；另有兼容学习及排序对齐。 | **最需要正面对照的方法近邻之一。**它复用初始标注查询和正文档，接收新无标签文档。式(8)／表3的L²R没有兼容约束，仍与当前重编码设定相关；表4的L²R vanilla已经固定旧文档表示，不能混称。 |
| [S03 ContEA，ISWC 2022](https://arxiv.org/html/2207.11436) | 增长知识图谱上的持续实体对齐；重放受新增三元组影响的旧种子对齐，并利用高置信新对齐。 | “只复用有用的旧匹配关系”已有先例。其可用历史图、邻域及对齐信息必须纳入适用条件，不能直接等同于只保留有限旧商品文本和关系的合同。 |
| [S04 CLUDA-ReID，CVPR 2022](https://arxiv.org/pdf/2112.06632) | 从标注源域进入无标签目标域，按身份／伪身份保存样本，并分别保持旧、新批次内的特征关系。 | 身份记忆及关系保持已有先例。它约束特征相似度矩阵，不等同于对已知真实同控边做监督式保持；本轮未审计补充材料中的全部身份配额实现。 |
| [S05 DwoPP，BMVC 2022](https://arxiv.org/pdf/2210.01600) | 不保留旧训练样本，但保留旧教师模型；在当前episode上移除特定蒸馏分布中的正类。 | 无旧样本不等于无历史信息。当前监督度量目标仍学习正对；不能推导任何正对约束都无益，也不能把有记忆比较方法的预算写到它头上。 |
| [S06 CREAM，KDD 2026](https://arxiv.org/html/2601.02708v2) | 动态流的聚类记忆保留样本、原型及统计；§4.3.1式(7)按共享文档的覆盖与重叠选择训练查询，再进行自监督更新。 | **共享对象的覆盖选择已有具体机制。**这里是记忆内的训练查询选择，不能直接等同于严格字节预算下的原始关系端点保存；聚类裁剪也不自动保证固定总容量。 |

CREAM出版身份由[作者机构记录](https://pure.korea.ac.kr/en/publications/cream-continual-retrieval-on-dynamic-streaming-corpora-with-adapt/)另行核实。上述任务与机制的不同，需要在比较中明确映射，不自动形成算法创新。

### L²R为何比泛泛的持续分类文献更重要

L²R把每个训练查询的旧负例记忆用于后续文本排序，与“留下有代表性的旧候选”直接相关。它不是简单选分数最高的负例；其记忆槽按查询分配，初始标注查询／正文档持续可用，比较历史信息时不能把后者视作免费。本项目若采用总量受限的多条商品文本与账号关系保存，资源单位和监督到达方式会不同。**这种不同目前只是比较设定上的差别，尚未证明需要新算法。**

同样，不能把L²R的旧索引兼容性与其负例选择混为一谈：兼容目标不是当前任务的必要组成，记忆选择却直接相关。论文中的“AP”指跨阶段平均表现，不能与本项目成对Average Precision直接对表。

### 外审后补读的四项直接近邻

| 文献 | 原文核对及其约束 |
| --- | --- |
| [S14 LUAR，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.70/) | §2以文本集合学习身份表示并作监督对比训练；§3按作者划分，§4研究跨域检索。多记录文本与训练未见身份评价已有先例，但该文不能被说成已经解决本项目的持续学习问题。 |
| [S15 DiCGRL，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.237/) | §3.3通过共享实体端点激活旧图三元组，并筛选相关语义成分更新。它能查询历史图，且新旧边可共享实体；本项目阶段身份不重叠，不能为套用方法改变该边界。 |
| [S16 ERD，CVPR Workshops 2022](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Wang_Incremental_Meta-Learning_via_Episodic_Replay_Distillation_for_Few-Shot_Image_Recognition_CVPRW_2022_paper.html) | 作者稿的Cross-task episodic training及ERD段落已核对：旧样本episode、新旧混合episode及查询相对支持集的分布蒸馏已有实现目标。默认随类别增长的记忆与固定总量变体必须区分。出版位置是工作坊。 |
| [S17 VeriDark，NeurIPS 2022 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html) | 作者稿§3及§4.4已核对：暗网相关文本的同作者验证、作者隔离评价及账号标签噪声均已有直接讨论。它不提供本项目现实同控真值，也不意味着本研究转向作者归属或新基准建设。 |

### 其他必须纳入边界判断的工作

- [S07 KDD 2019卖家账号匹配](https://www.contrib.andrew.cmu.edu/~nicolasc/publications/Tai-KDD19.pdf)已有成对打分及层次聚类。加入传递性、分组或图表示不自动等于新贡献。
- [S08 MIR，NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)选择预期最受新更新干扰的旧样本；[S09 GSS，NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html)用梯度信息维护多样记忆；[S10 DER／DER++，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)结合重放与历史输出保持。取出样本、维护缓存和训练约束是不同步骤，不能合并成含糊的“智能记忆”创新点。
- [S11 类别增量作者归属，CogMI 2024](https://arxiv.org/pdf/2408.08900)表明文本身份与持续学习的组合已有研究；其不断增加作者类别的分类任务不同于本项目未见控制者的成对判断。
- [S12 QDC](https://proceedings.mlr.press/v330/goswami26a.html)的旧索引兼容目标不是当前要求，但作者稿§4.2在当前查询和文档上进行新旧模型特征蒸馏，仍与历史知识保持相关。是否比较该训练机制取决于未来干预，不因此新增索引任务。作者稿为2025、PMLR卷330页面为2026。
- [S13 VLADR，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Vision-Language_Attribute_Disentanglement_and_Reinforcement_for_Lifelong_Person_Re-Identification_CVPR_2026_paper.html)继续研究跨实例属性关系保持。泛称“关系蒸馏”仍不够具体。

以上是文献对应关系，不是已完成的本项目方法复现或新基线清单。不同数据、候选池和监督下的论文分数不能直接与我们的MAP比较。

## 4. 哪些能称为工作，哪些还不能称为创新

| 做法／主张 | 当前判断 |
| --- | --- |
| 使用BGE中文编码器，调学习率、加入已知困难负例排序损失 | 是必要的基础模型改进；按当前实现，没有据此建立新算法主张。 |
| 使用少量旧样本重放，加历史输出或排序约束 | 方法家族已有大量直接近邻，不能作为唯一创新。 |
| 按账号保存、把共享账号的关系放在一组 | 可能有利于实现与预算利用，但身份记忆已有工作；分组本身不保证额外的关系学习。 |
| 声称首次“中文暗网＋有限记忆＋持续学习” | 当前检索不足以支持首次主张；改语言或应用场景不能替代方法贡献。 |
| 在相同可用历史信息下，提出区别于近邻的机制，并解释它为什么更能保持旧排序与新学习 | 有望成为论文贡献；尚无已完成的方法与验证证据。 |

### 值得继续查证的贡献问题

一个待证问题是：**在明确的历史信息条件下，多记录账号的保存成本、关系复用与查询排序目标之间，是否存在现有方法及其直接适配尚未解决的具体困难？**目前没有证据保证这种困难必然存在，也尚未提出解决它的新算法。

多条关系可能共享同一账号，同一文本也能服务多条关系；这种依赖本身已有先例。若之后允许在线选择历史数据，可研究保存内容与训练目标的联系；若可用历史子集外生给定，只能研究允许子集内的利用方式。需要可说明、可区分的实质方法差异，但不要求每个组成算子都从未出现；已有部件的非平凡结合也可能形成贡献。按真实字节公平计量属于比较规范，本身不构成算法创新。

如果之后考虑关系分组，必须解释收益是否来自关系机制，还是正负比例、负例难度、重复次数、账号覆盖或额外计算的变化。只有在明确机制后，才制定最小必要的区分性对照。此处保留待证问题，不提前冻结算法、对照数量、损失系数或训练预算。

在打分与权重不依赖其他边、目标仅为独立成对损失之和的条件下，分组求和可合并为重加权的成对目标；不能凭分组名称声称新增关系推理。这是目标形式的条件判断，不保证随机梯度训练轨迹相同，也不直接适用于存在候选竞争、查询归一化或其他跨对计算的目标。本次没有据此判定当前排序训练等价或失效。

## 5. 形成投稿论文还缺少什么

以下是执行者的证据判断，不是刊会录用承诺，也不新增当前实验门槛。

1. **现象确实存在。**在验收后的底座上，顺序学习是否损害旧分布留出卖家的能力？少量随机记忆是否已经足够？旧底座上的负结果不能代替这个答案。不能按已观察结果回选有利顺序、生成条件或预算来追求预定胜负；前瞻确认并明确标为合成控制因素的分布变化可以研究。当前已确认顺序与数据不变。
2. **方法差异明确。**用输入、可用历史信息、记忆更新、样本选择和训练目标逐项说明相对最近工作的实质变化。若只是套用现有方法，应如实按应用研究定位；这尚不满足用户已确认的方法创新目标。
3. **增益来自所称机制。**同样的监督、底座、候选池和历史信息边界下比较，披露实际存储与训练／选择开销；预算敏感性及机制对照由未来合同前瞻安排，不能事后只报告有利预算。
4. **旧／新能力与识别保护一起报告。**不能通过牺牲新群体学习换取旧排序，也不能把排序改善自动称为概率预测改善。具体标准与不确定性在未来阶段确认。
5. **验证边界可信。**本研究目前只支持中文合成受控任务。相同生成机制的留出群体有助于检验该机制内的泛化，不能证明真实中文市场有效性。27候选且保证存在同控候选，也不支持全库或拒识结论。

合成同控标签表示生成规则定义的控制者归属；文本相似、共同作者、现实共同控制之间的对应关系未由本研究直接验证。屏蔽显式标识符也不证明剩余信号必然是稳定文风。本次没有读取正式文本，不能声称已发现模板捷径或泄漏。

新群体到来本身不自动意味着输入分布变化。早期9月9日[数据合同](SELLER_ALIAS_CONTINUAL_DATA.zh.md)主要改变经营品类／商品内容；当前BGE输入实际沿用9月10日[表达变化版本](SELLER_ALIAS_EXPRESSION_DATA_RESULT.zh.md)，保留商品主题混合变化并叠加公共表达覆盖概率变化，不能概括为“表达机制保持”，也不能称纯表达单因素因果实验。后续解释应对应实际使用版本，不把身份变化或随机分组冒称真实时间漂移，不因此改动生成器。旧文献外审原件及修订前全文保留。

合成数据会使方法的适用范围和外部有效性成为论文的重要限制，尤其要说明哪些现象来自生成假设。用户已经选择先完成受控主线，本次不擅自加入真实数据、大候选库或无同控查询实验。是否扩展，由受控主线证据与后续投稿定位共同决定。

**当前建议是继续原主线，并把创新论证集中到一个有证据的具体失败机制。**基础验收闭环后，以本文近邻为约束设计持续学习，先观察新BGE上的学习／遗忘及简单方法表现。若简单方法已解决问题，或拟议改进与近邻等价，应调整贡献主张，而不是给已有方法换名称。

### 9月30日补充：阶段末输出保持的直接近邻

[Measuring and regularizing networks in function space，ICLR 2019](https://arxiv.org/pdf/1805.08289)的§3.1—3.1.1保存少量旧样本，以及它们在原任务训练结束时的预测，再约束新旧输出的距离。DER正文也明确将FDR的任务边界输出与DER的训练轨迹输出区分。当前拟议LOGIT臂使用完整群、原始logit MSE及本任务监督排序损失，是已有思想的具体适配；“阶段末保存分数”不能单独作为创新。输出对象、距离和缓存分配不同，也不能直接声称等同原版FDR。

主执行者本次核对了上述FDR§3.1—3.1.1及DER§3.1相关原文；未复现其代码。新增原文访问范围记录在[本轮来源补充](../reports/documentation/20260930/continual_plan/review/primary_sources.json)，不回写9月27日S01—S17的历史阅读声明。首轮不因此强制新增FDR训练臂。

## 6. 阅读范围、历史审查与9月27日边界

本轮按暗网账号关联、持续文本检索、持续实体对齐、终身重识别及通用有限记忆五条线定向检索，核查到2026年的可访问原始资料；检索式、逐篇实际阅读位置和获取限制记在来源清单。核心六篇已读相关任务和方法原文；补充文献按清单区分正文阅读与摘要核查。没有将任何论文源码宣称为已复现或审计通过。

同时亲自重读2026-09-07选题外审、方案外审和2026-09-09数据方案外审的实际回复，复核其适用范围：按身份保存／关系一致性已有先例；共同锚点分组在可加成对损失下不自动增加关系约束；旧分布留出身份与历史身份记忆不同；随机划批不能充当真实时间域。这些是已存在的论证约束，不是本轮新发现的代码故障。原始路径记于来源清单，既有处置与关闭状态保持。

本次未连接Linux、未读取新的实验数据或标签、未加载模型，也未修改正在运行的18份冻结来源。只编辑文献定位、来源记录及非冻结文档入口；采用文档／链接／散列核查，不执行科研测试。

## 7. 本次外审与主执行者处置

用户明确“外审已结束”后恢复结果读取。实际回复ID为`6f968888-1076-4b49-a167-412832c3b6fa`，完整渲染正文8618字符及`source_register.json`附件均已保存并亲自读完。外审的阅读声明与主执行者实际阅读分别记录；没有把其论文阅读或声明当作本地复现证据。

| 意见 | 本地主审处置 |
| --- | --- |
| F1 多记录文本／未见身份近邻不足 | 补读LUAR与VeriDark，补充VendorLink账号广告集合机制；明确这些设定已有先例。 |
| F2 共享对象／端点机制已有工作 | 补读CREAM式(7)及DiCGRL§3.3；区分记忆内训练选择、历史端点保存及历史图可用性，不以共享端点认证创新。 |
| F3 组合与分组不能自动证明创新 | 补读ERD并保留成对可加目标的条件限制；说明实质新组合可以构成贡献，不要求所有部件原创。 |
| F4 历史信息来源边界 | 采纳披露原始输入、监督、输出／教师及统计的建议；未冻结容量或选择权限。 |
| F5 QDC机制不能全部排除 | 核对其§4.2并补充当前数据特征蒸馏的关联；当前不增加索引兼容任务。 |
| F6 作者／账号／控制者真值不同 | 核对VeriDark§4.4，明确合成同控真值和真实共同控制的区别；未宣称发现本项目泄漏。 |
| F7 顺序与分布变化表述 | 澄清允许前瞻批准的受控变化，继续禁止为方法排名事后改条件；对应已确认的经营品类／商品内容变化轴。 |
| F8 提交文本链接格式缺陷 | 实际网页用户消息确有重复URL、转义括号及正文混入；本地原始提示没有同类字符变形。保留两种原件，采用S01—S17稳定目录恢复可追溯性，不声称历史消息已被修复或重新提交。 |

另已精确区分L²R式(8)／表3与vanilla兼容版本，补充CLUDA的伪身份／特征关系及DwoPP保留教师的边界。初次来源清单中“Vanilla method remains relevant”措辞易误读，适用的明确说明以本次复核目录为准。

外审所谓`SCIENTIFIC_BLOCKER`主要针对“已经找到方法空白／创新成立”的更强主张；本项目当前未作该主张，本次补充也没有证明新机制成立。F8属于已确认的提交呈现与来源定位缺陷，已用原件留存及清晰目录处理。没有由本次文献审查确认的活动训练阻断；扩大数据范围、增加索引任务或要求复现全部近邻均不执行。未新增研究设计，不重复送审未变化证据。完整证据、处置和剩余阅读限制见主审记录。

## 8. 2026-10-23：商品文本卖家关联比较文献补查

用户本轮要求为2022—2026年、CCF A/B或北大中文核心，重点是利用商品标题和描述关联卖家账号的具体方法；不以数据集论文、任务不匹配论文或同一方法的多个版本补足3篇。本节只补充书目、输入对象和检索状态，不批准新增实验、改变模型输入或比较合同，也不覆盖上面的历史审查与运行状态。

资格按用户提供的[CCF目录](../CCF推荐目录.pdf)和[北大核心目录](../北大中文核心期刊目录.pdf)核对：CCF PDF第22页将TrustCom列为C类，第23页将ICICS、PETS列为C类；北大核心PDF第28页列有《计算机工程》《信息安全研究》，未列《信息安全学报》，两刊不能混认。页码指PDF物理页。

### 新增核查与保留边界

| 文献与正式来源 | 输入／具体任务的证据 | 本轮筛选状态 |
| --- | --- | --- |
| [VendorLink，ACL 2023主会](https://aclanthology.org/2023.acl-long.481/) | 正文§2、§3、§4.1—4.2：标题与描述用`[SEP]`拼接，卖家监督训练后提取广告表示，计算广告间余弦相似度，并用账号内部自相似度归一化账号间分数；有[作者代码](https://github.com/maastrichtlawtech/VendorLink)。 | **目前唯一已确认同时满足本轮年份、刊会和直接任务要求的方法候选。**具体中文适配与复现仍未执行，不能写成已复现。 |
| [You May Not Be Your Username: Cross-Market Vendor Alias Attribution with Automatic Multi-Signal Evaluation，TrustCom 2025](https://doi.org/10.1109/TRUSTCOM66490.2025.00129) | 引文索引和[DBLP书目](https://dblp.org/rec/conf/trustcom/KwanTPSM25.html)核实题名、作者、年份及1119—1126页；本轮未获得全文，不能仅从题名推定各信号及训练细节。 | 新增直接任务线索，但**CCF C**，不计入；未将其写成可复现或纯文本方法。 |
| [Multi-Identity Recognition of Darknet Vendors Based on Metric Learning，Applied Sciences 2024，14(4):1619](https://www.mdpi.com/2076-3417/14/4/1619) | 已获取全文：商品文字与图片特征，经PCA／WGCCA等处理，用孪生网络和度量学习识别同一卖家的多个账号；正文与参考文献回溯到eDarkFind等前序方法。 | 确为具体账号关联方法，但刊物不满足所给目录资格，且完整方法使用图片；不计入。 |
| [Link Prediction-Based Multi-Identity Recognition of Darknet Vendors，ICICS 2023:317—332](https://link.springer.com/chapter/10.1007/978-981-99-7356-9_19) | Springer摘要及参考文献：商品文本、图片、写作风格及异构关系，图神经网络与孪生网络预测账号是否属于同一供应者；作者为Futai Zou、Yuelin Hu、Wenliang Xu、Yue Wu。尚未取得全文完成代码级复现判断。 | 新增直接任务方法，但**CCF C**，且原方法有额外模态；不计入。 |
| [Follow Your Hidden Traces: Underground Forum-Based Darknet Market Sybil Detection，ICWS 2025:218—224](https://doi.org/10.1109/ICWS67624.2025.00035) | [作者机构摘要](https://pure.korea.ac.kr/en/publications/follow-your-hidden-traces-underground-forum-based-darknet-market-/)明确：筛选马甲相关论坛帖子，抽取多个账号的信息，再核验账号是否存在于市场。 | ICWS为B类、目标涉及卖家马甲，但输入依赖额外论坛线索，不能直接放入仅有屏蔽后商品标题／描述的实验；不计入本轮3个直接比较方法。 |
| [Detect Your Fingerprint in Your Photographs: Photography-based Multi-Feature Sybil Detection，PoPETs 2023(1):244—263](https://petsymposium.org/popets/2023/popets-2023-0015.php) | 官方摘要：从商品照片提取图像哈希、类别、复原及OCR文字特征，融合后匹配卖家；其中“text”是照片中的文字。 | 图片依赖与本项目输入不符；所给目录中的PETS条目也是C类，不能因摘要含text便计作商品文本方法。 |
| [暗网网页用户身份信息聚合方法，计算机工程2023，49(11):187—194,210](https://www.ecice06.com/CN/10.19678/j.issn.1000-3428.0066805) | 已读正文及37条参考文献：ConRE围绕身份实体对、其上下文及实体类型做共指关系抽取；不是从两个卖家的商品集合估计同控关系。 | 北大核心且为方法论文，但目标对象与所需输入不同。将其改成商品对分类器会改变论文方法，不能据此声称直接复现；不计入。 |
| [去除主题偏差的暗网论坛作者归属分析方法，信息安全学报2026，11(3):18—33](https://xuebao.iie.ac.cn/ch/reader/view_abstract.aspx?file_no=20260302&flag=1) | 已获取正文：基于论坛帖子、作者／主题分组的表示学习和去偏约束；不是持续学习实验，也没有商品标题／描述卖家关联的直接验证。 | 《信息安全学报》不在所给北大核心目录中；不与《信息安全研究》混同，不计入。 |
| [FCA-XLNet-BiGRU Multi-task Framework for Darknet Transactions，KSEM 2025会议／2026卷](https://link.springer.com/chapter/10.1007/978-981-95-3072-4_8) | Springer摘要明确任务为闭集交易分类、开放集检测和低资源适配；引用VendorLink并不使其成为卖家账号关联方法。 | 任务不匹配，不计入。在线日期2025-10-11与卷引用年份2026分别保留。 |

### 实际追踪路径和未解决缺口

- 沿VendorLink正文的前序方法追到[eDarkFind（WWW 2020）](https://doi.org/10.1145/3366423.3380263)、[KDD 2019账号匹配](https://doi.org/10.1145/3292500.3330763)、[uStyle-uID（WWW 2019）](https://doi.org/10.1145/3308558.3313537)，再通过OpenAlex的`cites`查询筛查它们2022—2026年的被引文献；前序论文年份不合本轮要求，不能充作新论文。VendorLink正式版与arXiv版、同一工作的学位论文归为同一方法。
- 本轮OpenAlex记录分别为VendorLink `W4385570400`、eDarkFind `W3012857244`、KDD 2019 `W2953027522`、uStyle-uID `W2988701063`、2024度量学习 `W4391948814`；接口筛选日期为2022-01-01至2026-12-31，返回结果仍须按实际出版日期、原文和目录核对。没有把检索结果数量当作独立方法数量，也没有把索引未收录解释为文献不存在。
- 中文路线核对ConRE参考文献，并追查作者、相邻成果与可公开检索的引用线索；《暗网高质量威胁情报获取技术与实现》（计算机工程2026）虽与上述上海交大团队相关，但其任务是主题筛选、实体抽取与情报时新性评估，不是卖家账号配对。未获得CNKI／万方的完整被引导出，不能宣称中文被引已穷尽。
- **严格清单目前为1/3，另2个空缺仍未补齐。**这是当前已核实证据的范围，不是断言全球不存在另外两篇。数据集、通用作者归属、商品分类、图片指纹、额外身份标识符抽取以及C类论文均不冒充合格直接比较方法。没有新训练、数据读取或基线实现；不能把“未研究持续遗忘”写成“已经证明会遗忘”。

## 9. 2026-10-23后续：放宽刊会范围后的直接方法核查

用户随后将刊会要求放宽为“2022—2026年、CCF A/B或知名期刊会议”，仍要求商品标题／描述用于卖家账号关联。第8节保留当时较严格条件下的结果；新条件下不再仅因TrustCom、ICICS为C类直接排除。以下是文献事实与待核实边界，不是新增比较方案、实施授权或复现完成声明。

### 两篇有正文依据的直接任务候选

1. **DarkBIP-FW：Decoding the shadows: multi-modal identity profiling in darknet markets using latent behavior feature fusion**。Xuan Zhang、Pengfei Li、Xin Tong、K. P. Chow、DeYu Yuan、Si Chen、Bo Jin；*Discover Applied Sciences*，2025，7:656，2025-06-17发表，[DOI及出版社全文](https://link.springer.com/article/10.1007/s42452-025-07217-5)。本轮实际核读§4.2.3、§5、§6.1—6.6与参考文献。§4.2.3将中文商品标题和描述拼接，由`bert-base-chinese`编码；§5使用卖家标签嵌入及LabelSimCos损失；§6.5.2明确包含**DarkBIP (Only Text)**，不是本项目自行臆造的消融。§6.6.2另以卖家嵌入的两两余弦相似度筛选疑似多账号，故不能把此文概括成只有商品分类或活跃度分组。须分开三类证据：Only Text的结果是已见卖家分类，疑似账号关联案例没有独立报告Only Text版本，尚未验证本项目未见身份／同控真值设定。文本消融与原文关联规则能否组合成可复现比较臂仍需明确聚合、标签嵌入和评价适配，不能借用完整多模态案例的效果。
2. **Multi-Identity Recognition of Darknet Vendors Based on Metric Learning**。Yilei Wang、Yuelin Hu、Wenliang Xu、Futai Zou；*Applied Sciences*，2024，14(4):1619，2024-02-17发表，[DOI及出版社页面](https://www.mdpi.com/2076-3417/14/4/1619)、[出版社全文PDF](https://mdpi-res.com/d_attachment/applsci/applsci-14-01619/article_deploy/applsci-14-01619-v2.pdf?version=1708313082)。已核读全文。§3.2.1分别对标题、描述提取冻结BERT、Doc2Vec和写作风格特征，再聚合为账号特征；§3.2.3降维，§3.3用共享参数孪生网络及对比损失学习账号距离，目标就是不同账号是否属于同一卖家。完整模型还含ResNet50图片特征；§3.2.2明确无图片商品用零向量。因此存在按文字输入研究其缺图版本的具体依据，但“所有商品均缺图”的文本实验仍属适配，正文未给出独立纯文本消融成绩。英文风格特征、784维降维、标题Jaccard预筛及用户名／伪标签流程不能不加说明地移植到当前中文固定候选集与已知标签合同。

两篇的任务都直接涉及卖家身份／多账号，具备比数据集论文或泛作者归属更具体的比较依据；二者的文本适用证据强度不同。刊物资格仍如实标注：均非本轮已核对的CCF A/B条目，不能把出版社知名等同于刊物达到顶刊档次。[Discover Applied Sciences官网](https://link.springer.com/journal/42452)列明ESCI、Scopus、Ei Compendex收录；本轮不将其写成SCIE或A/B期刊。不因放宽范围便宣称已找到另外两篇无需适配、可原样复现的纯文本方法。尚未找到并核验这两篇的官方实现。

### TrustCom 2025的新证据与暂留问题

[You May Not Be Your Username](https://doi.org/10.1109/TRUSTCOM66490.2025.00129)已由[作者机构发表清单](https://www.cpp.edu/faculty/santanamarin/publications.shtml)、Crossref及IEEE公开页面元数据交叉核实；正确IEEE记录为[11354938](https://ieeexplore.ieee.org/document/11354938/)，作者Wesley H. Kwan、Lynn K. Takahashi、Nathan Pham、Apurva Sista、Ericsson Marin，1119—1126页。相较第8节，本轮已取得正式摘要：模型在一个市场的带卖家标签数据上训练，对未见市场商品预测；Oracle通过用户名、图片相似度和文本签名交叉核验预测。**Oracle是摘要所述评价机制，不能据此断言预测模型一定依赖图片，也不能反向断言模型仅用标题／描述。**全文和官方代码仍未取得，具体模型、输入与账号聚合尚不清楚，暂不将其计为可落地的已核实纯文本比较方法。

另外复核了ICICS 2023的Springer摘要、参考文献及OpenReview书目；后者为书目入口，未获得全文，不冒称已读方法。其图结构和图片依赖未解决，因此没有拿它与2024同团队论文直接拼成两个已经独立核实的文本方法。扩展的取证路线中，Single-Vendor Marketplaces相似度论文主要使用网页结构／元数据，Busting up Monopoly涉及市场监测及加密货币关联，Pathway toward the Surface Web涉及追踪代码／站点连接；均未据此补足商品文本账号关联名额。

本轮只检索公开论文、核对书目与方法正文并更新本记录；未发送作者邮件、读取项目实验数据、连接Linux、实现或运行新的比较方法。已获全文的候选可进入后续复现可行性核查，但不把候选数量写成已完成比较数量。

## 10. 2026-10-23：持续检索目标与作者、犯罪学及2026年来源补查

用户明确要求设定goal，在VendorLink之外补齐另外两个2022—2026年、CCF A/B或知名刊会的商品标题／描述卖家账号关联方法。目标已建立并保持进行中；检索渠道不限，年份、输入、任务和刊会条件不因此放宽。第9节的“候选”是相关文献线索，不代表名额已补齐：**DarkBIP-FW仅作相关参考，不计入；Applied Sciences 2024度量学习方法的刊会资格与纯文本适配证据仍不足，也不计入。当前已确认清单仍为VendorLink一项，另外两项空缺。**数据库收录、影响因子或出版社品牌不能单独证明符合用户所说的知名刊会。

本次补查所得的一手证据与限定结论如下：

| 来源 | 核查所得 | 对名额的影响 |
| --- | --- | --- |
| Saxena博士论文[Connecting online criminal behavior with machine learning](https://arxiv.org/abs/2605.04080) | 已取得公开全文，核对出版物清单与章节对应关系；暗网商品方法为VendorLink，另有IDTraffickers、MATCHED及伦理准则。MATCHED的正式来源为[Findings of ACL 2025，4334—4373](https://aclanthology.org/2025.findings-acl.225/)，不因学位论文中笼统的main-track表述改写正式书目。 | 学位论文中的VendorLink不另计一个方法；IDTraffickers、MATCHED围绕陪侍广告，虽含身份识别／验证及文本基线，但不能充作两项已在商品卖家账号任务核实的方法。 |
| [The Network of Online Stolen Data Markets: How Vendor Flows Connect Digital Marketplaces](https://academic.oup.com/bjc/article/62/6/1518/6503727)，British Journal of Criminology，2022，62(6):1518—1536 | 出版社正文“Digital trace data on the darkweb”明确以卖家别名完全匹配确认同一卖家，再构造市场迁移网络；同时承认换名和冒名的局限。另定位到[CrimRxiv存档](https://www.crimrxiv.com/pub/orvi17zc/release/1)，其下载访问未成功，未冒称已下载该存档全文。 | 不是以商品文字识别不同卖家账号的算法，不计入。 |
| [Stayin’ Alive: How Global Stolen Data Markets Thrive on Telegram](https://www.usenix.org/conference/usenixsecurity26/presentation/marjanov)，USENIX Security 2026，5473—5492 | 已查看[作者机构公开PDF](https://www.andrew.cmu.edu/user/nicolasc/publications/Marjanov-USENIXSec26.pdf)的研究问题与方法范围：盗售数据类型分类、渠道发现、语言／地区与频道存续分析。 | 刊会与年份符合，但没有所需商品文本卖家账号配对任务，不计入。 |
| TrustCom 2025作者公开资料 | 复核[作者实验室项目页](https://www.cpp.edu/calsys/projects.shtml)、[研究人员页](https://www.cpp.edu/calsys/researchers.shtml)及[DMDb公开说明](https://www.cpp.edu/calsys/database.shtml)。项目页确有Vendor Alias Attribution，但没有补足架构、训练目标和账号打分细节；已核实实验室GitHub组织当前没有公开仓库。IEEE公开PDF入口也未取得全文。 | 保留为直接相关待核实线索；未把项目简介当作代码或方法全文，也未申请数据库、联系作者或冒用机构访问。其Oracle属于评价信号的边界仍按第9节。 |
| [NLP-based techniques for Cyber Threat Intelligence](https://doi.org/10.1016/j.cosrev.2025.100765)，Computer Science Review 58 (2025):100765 | 已取得[作者机构49页公开全文](https://iris.uniupo.it/retrieve/ee584c74-d1f4-4602-9d83-a7df99b9e55f/15%29CosRev_NLP.pdf)，定向核查暗网商品、身份及关联段落。涉及5个市场、7911卖家的文献[130]实际为2019年《Dark-net ecosystem cyber-threat intelligence (CTI) tool》，不能依据综述年份把它算作2025年新方法。 | 作为追索来源，综述本身及该2019年工作均不计入。 |

本次还扩展检索了seller entity resolution、seller account linkage、vendor attribution及中文商家／卖家身份关联等表述；返回的商品匹配、销售预测、论坛作者归属与身份标识符聚类均未据此补位。宽检索的部分接口输出存在截断和无关结果，不声称已逐篇穷尽筛查，更不将“尚未找到”写成“相关论文不存在”。后续继续追查直接方法的公开全文、官方实现和近期引用；只有补足输入、训练、账号表示与打分及刊会证据，才改变名额状态。本节仅更新公开文献核查记录，不产生新实验或改变既有研究合同。

## 11. 2026-10-23：用户提供两篇全文后的方法核实

用户随后提供ICICS论文集 `C:\Users\35734\Downloads\Documents\978-981-99-7356-9.pdf` 及TrustCom论文 `C:\Users\35734\Downloads\Documents\You_May_Not_Be_Your_Username_Cross-Market_Vendor_Alias_Attribution_with_Automatic_Multi-Signal_Evaluation.pdf`。已实际读取前者317—332印刷页（PDF第332—347页）和后者方法、实验、相关工作与参考文献，并以页面图像复核GraphSAGE／对比损失公式及TrustCom分类、账号聚合规则。**第9—10节“未取得全文”的状态到此解除；原文输入与当前任务的差别仍存在，取得PDF不等于补齐两个合格名额。**全文由用户本地提供，未上传出版社PDF或论文集。

### ICICS 2023：GraphSAGE与孪生网络的账号对预测

[Link Prediction-Based Multi-Identity Recognition of Darknet Vendors](https://doi.org/10.1007/978-981-99-7356-9_19)，Futai Zou、Yuelin Hu、Wenliang Xu、Yue Wu，LNCS 14252，317—332页。原文研究对象确为跨市场不同账号是否属于同一卖家，不能把它归为数据集论文或普通商品分类。

- **实际输入（§3.2，322—324页）：**每条商品分别处理标题与描述，提取BERT、PV-DM Doc2Vec及81维写作风格特征；账号侧对商品特征求均值，并另将同一账号的标题／描述串接，用PV-DBOW学习账号文档特征。图片用ResNet-50生成1000维特征，多图取均值；账号再聚合其图片。PCA或WGCCA将融合特征降到784维。原文未报告独立纯文本消融，也未给出2024年后续论文中那条“缺图置零”的规则，不能将后文规则倒填到本篇。
- **图结构（§4.1，325—326页）：**四类节点是卖家、商品、发货地、商品类别；商品属于类别、从发货地发出、由卖家出售，发货地和类别采用one-hot特征。待预测的是卖家之间的同一身份关系；正文说明以前三类关系及节点属性预测该关系，不能自行把待测同控标签当作传播边。原方法因此还需要标题／描述以外的类别、发货地与图片信息。
- **训练与分数（§4.2—4.3，327—328页）：**按关系聚合邻居的GraphSAGE生成账号表示 `h_a`、`h_b`，同一套孪生网络映射为 `g(h_a)`、`g(h_b)`，按两者距离判断是否同控。正文没有明确距离的具体范数，不能凭习惯补成已核实的欧氏距离。损失为 `1/(2N) Σ[y_i d_i² + (1-y_i) max(margin-d_i,0)²]`，同时反传更新GraphSAGE和孪生网络；同控对拉近，异控对至少推离margin。这是明确的账号对学习方法，与2024年无该异构图的度量学习方法不能简单当作同名重复版本。
- **结果及复现限制（§3.3、§5）：**21个英文市场、185460条清洗后商品、8507个账号；以跨市场同用户名起始标注，标题词集合Jaccard阈值0.2过滤负例，再用预测分歧触发人工重标并重复划分。评价按账号对分90%／10%并做十折，并未据此证明控制者身份严格不重叠。表4报告Accuracy 99.75%、F1 0.990；表7对旧uStyle-uID／dStyle-GAN引用各自数据上的成绩，作者明确未在同一数据集复现两者，不能表述为受控的同数据优势。正文未完整给出网络层数、最终margin和判定阈值等细节，公开数据仓库也不等于方法实现仓库。

因此，本篇已从“缺全文线索”推进为**已核实的直接账号关联方法，但不满足当前纯标题／描述输入下的原样比较条件**。删掉图片和两类属性节点后如何保留原方法，需要明确的文本适配及验证；本轮没有实施或默认批准该适配。

### TrustCom 2025：多模态SVM的已学卖家类别匹配

[You May Not Be Your Username: Cross-Market Vendor Alias Attribution with Automatic Multi-Signal Evaluation](https://doi.org/10.1109/TRUSTCOM66490.2025.00129)，Wesley H. Kwan、Lynn K. Takahashi、Nathan Pham、Apurva Sista、Ericsson Marin，1119—1126页。该篇同样是卖家跨市场别名关联的具体方法论文，且与ICICS方法不同。

- **商品特征（§IV-B，1121—1122页）：**标题／描述的字符3—6 gram TF-IDF表示词汇内容；词性序列4—7 gram TF-IDF与大小写、标点、长度等统计表示风格；ResNet-50表示商品图片。三路分别PCA降维。因此图片确实进入预测模型，不能再将其概括为仅用于Oracle评价。
- **两个分类配置（§IV-B）：**MMM将三路经MCCA投影、拼接后输入一个SVM；ESV分别训练三路SVM，对预测概率等权平均。ESV含独立文本分类器这一事实，不等于论文报告了移除其他模态后的纯文本关联效果。两配置是同篇工作中的变体，不能占两个论文名额。
- **账号聚合与匹配（§IV-B4、§V-B）：**每个源市场卖家是一个类别；少于20条商品的账号合并成`-1`噪声类。对目标账号的各商品输出类别概率并求均值，取最高概率类别，超过相应阈值才报告匹配。用符号重述为 `s(b→a)=mean_{x∈商品(b)} p(a|x)`，其中`a`必须是训练时已经建立的源市场卖家类别。这不是给任意两个新身份账号直接输出同控分数；“未见市场”不等同于本项目的“未见控制者身份”。SVM核、概率校准等完整实现细节尚无官方代码补证。
- **评价及数字边界（§IV-C—D、§V-C）：**在市场1、9、10分别训练后应用于其余24市场，Oracle以用户名、图片、描述LCS的三信号交集构造评价依据。Oracle是评价机制，不是可替代预测模型的第二方法；CPR是Oracle正例的人工确认比例，也不是独立身份真值完整性证明。表5的跨市场F1依源市场及配置分别为MMM的0.62／0.43／0.64、ESV的0.56／0.52／0.68，**0.68是表中最高值，不能写成25个市场的总体平均F1**。没有独立纯文本账号关联结果或任意未见身份对的评分验证。

因此，本篇已从“输入不清”推进为**输入、训练和账号匹配规则已核实，但多模态输入与已学身份类别输出都需要适配**。仅保留文本SVM仍不能自动解决两端控制者均未见时的账号对评分，不能借用原文0.68或“open-world”一词宣称它已经适用于当前合同。

### 刊会及其他新线索

本地CCF目录将TrustCom和ICICS列在网络与信息安全C类；二者不冒称A/B。用户最新条件允许“知名期刊会议”，因而不再只按C类标签排除，但仍披露实际层级。[TrustCom官方会议页](https://ieee-aiplus-2025.org/callFoePapers/Trustcom-CallforPaper.html)记载2025年为第24届；[ICICS出版社论文集](https://link.springer.com/book/10.1007/978-981-99-7356-9)及[南开大学会议报道](https://cc.nankai.edu.cn/2023/1201/c13291a530468/page.htm)确认2023年为第25届、38篇长文与6篇短文从181篇投稿中选出。这些是领域会议历史和评审依据，不将办会届数或出版社品牌自动等同于顶会资格；本轮不靠重新包装刊会层级补位。

此外发现[作者公开简介](https://github.com/princia05/PRINCIA)中的《Cross-Market Vendor Profiling in Darknet Marketplaces Using Semantic Similarity》：作者称在IEEE ICASI 2026展示，使用26000余条记录、Sentence-BERT和余弦相似度。[会议官网](https://2026.icasi-conf.net/)及[议程](https://2026.icasi-conf.net/agenda/)可核实2026年4月14—17日京都会议，但当前精确标题检索、Crossref标题检索和作者公开仓库未补足该篇正式出版记录、方法全文／实现及满足“知名刊会”的证据；仅作新线索，不计入。另一个[学生项目仓库](https://github.com/Islam-dabbour/Vendor-Linking-in-Darknet-Marketplaces-Text-vs-Hybrid-ML)自述TF-IDF／BERT／混合模型比较，没有正式论文出处，不能当作TrustCom官方代码或合格论文。[LLM-HetGDT（Findings of ACL 2025）](https://aclanthology.org/2025.findings-acl.725/)检测Twitter贩毒参与者，不解决商品文本卖家账号关联，予以排除。

本次记录解除两篇论文的全文获取缺口，但**当前仍仅VendorLink计入，另外两项未补齐，goal未完成**。在获知两篇的具体适配缺口后，用户明确选择“保持当前输入和任务要求，继续找可直接比较的方法”，因此不将文本适配版转成已批准候选，也不重新询问是否放宽这一条件。本节是文献原文事实与本项目既有输入／输出边界的核对，不采纳新比较臂、不修改合同、不运行适配实验，也不宣称已通过新的方法外审。

## 12. 2026-10-23：保持原输入与任务后的引用追查

继续沿近期论文参考文献、[DNM研究目录](https://gwern.net/doc/darknet-market/dnm-archive/index)、作者公开资料与正式出版入口追索，并补查vendor re-identification、seller entity resolution及中文卖家／多身份关联等表述。本轮新增核查如下；目录、索引和检索摘要只用于定位，具体输入／任务判断回到原文。

| 新核查对象 | 原始证据与适用结论 |
| --- | --- |
| [NBC-Softmax: Darkweb Author Fingerprinting and Migration Tracking](https://arxiv.org/abs/2212.08184)，2022 | 已读[全文](https://arxiv.org/pdf/2212.08184)§3.6—4.3：以帖子文字、星期及子论坛上下文组成episode，沿用SYSML架构，加入NBC-Softmax对比损失；实验为Silk Road、Silk Road 2、Agora、Black Market Reloaded的**讨论论坛帖子**。表2虽有去时间、去图上下文的消融，仍不是商品标题／描述卖家账号实验。作者给出[代码](https://github.com/gayanku/NBC-Softmax)，但本轮没有查到符合条件的正式刊会版本，不因有代码或2022年预印本而计入。公开索引全文检索另返回T5-CSBoost，题目与摘要为LLM生成文本指纹识别，也不是卖家账号方法；不把索引引用计数当作已穷尽全部被引文献。 |
| [Identifying Similar Users Between Dark Web and Surface Web Using BERTopic and Authorship Attribution](https://doi.org/10.3390/electronics14010148)，Electronics，2025 | 出版社正文描述先用BERTopic找两类网络中的相近话题，再以作者风格相似度关联用户；研究对象为一个明网论坛与两个暗网论坛。即使是文本方法，也没有补足商品标题／描述卖家关联证据，不计入。 |
| [Multimodal Threat Actor Profiling on the Tor Network: Techniques, Datasets, and Ethical Challenges](https://www.ijcaonline.org/archives/volume187/number20/pativada-2025-ijca-925278.pdf)，International Journal of Computer Applications，2025 | 已读公开全文；文章是综述和概念框架，案例为假设性合成示例。相关商品方法引向uStyle（2019）和eDarkFind（2020）；不能把本文年份转移给这些旧方法，也不能将概念框架当作已核实可复现的新增账号对算法。 |
| Cross-Market Vendor Profiling in Darknet Marketplaces Using Semantic Similarity，作者称ICASI 2026展示 | 继续查[会议历届论文集入口](https://2026.icasi-conf.net/past-conferences-proceedings/)及议程，当前页面未提供该篇正式出版记录／全文。官网允许投稿manuscript或abstract，因此展示本身不足以证明正式全文发表，也不能反向断言该篇一定只是摘要。OpenAlex精确短语查询未补到记录，DBLP请求遇到反机器人页；后者不是“零篇”证据。输入、账号聚合、未见身份评价及刊会资格仍未补足，维持未核实线索。 |
| Automated Models for Darknet Monitoring and Criminal Attribution | 检索发现该标题与卖家重新识别有关，但目前仅有第三方书目将其列为《Demystifying the Darknet: Technologies, Threats, and Trends》的59—84页章节；[出版社图书页](https://www.igi-global.com/book/demystifying-darknet-technologies-threats-trends/400777)可对应电子ISBN 9798337397573。未取得该章方法全文或符合条件的刊会出处，且不同入口的出版年份／状态表述不一致；不将摘要中的准确率当作已验证结果，不计入。 |

本轮还发现一处会误导引用追查的表述：[DarkBIP-FW原文](https://link.springer.com/article/10.1007/s42452-025-07217-5)§2.2把文献[27—29]概括为利用商品描述作跨市场身份关联，但回查实际来源，[27]为2016年论坛写作风格识别，[28]为2022年[Tracking Discourse Influence in Darknet Forums](https://arxiv.org/abs/2202.02081)（论坛话语影响分析技术报告），[29]为[SYSML，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.548/)（论坛作者关联）。它们不能据此成为2022—2026年的商品卖家文本关联方法。引用链中的年份、研究对象及方法作用须以被引原文为准。

后续补查作者团队及近期参考文献：直接读取[上海交大团队成果页](https://netsec.sjtu.edu.cn/)，商品卖家关联条目仍指向已核实的ICICS 2023与Applied Sciences 2024；同团队LLM-TIKG（Computers & Security 2024，DOI `10.1016/j.cose.2024.103999`）的[作者仓库说明](https://github.com/Netsec-SJTU/LLM-TIKG-dataset)明确是从威胁情报文本抽取实体和关系，不能因刊会合格就替代卖家账号任务。另取得ICWS 2025《Follow Your Hidden Traces》的[Crossref参考文献元数据](https://api.crossref.org/works/10.1109/ICWS67624.2025.00035)，其中市场方法线索回到eDarkFind、uStyle、KDD 2019及PoPETs 2023图片方法，未补出新的合格文本方法；本轮获得的是书目数据，不冒称取得该篇方法全文。

检索出现的2022年Dark Vendor Profiling（DVP）介绍也已回查：[Tennessee Tech原始发布页](https://publish.tntech.edu/index.php/PSRCI/article/view/673)明确日期为2020-05-11，实际方法融合写作风格、图片和商品属性。2022年报道不能将2020年成果变成近期论文，此线索不计入。

ICASI线索补查发现[官方论文集模板页](https://2026icasi.wind-tech.com.tw/proceeding-template/)同时列出IET和Taylor & Francis出版路径，因而不将“未找到IEEE DOI”单独当作未发表证明。已据此补查IET、Taylor & Francis、IEEE标题入口和Crossref完整标题查询，仍未取得该篇正式书目／全文；作者公开展示说明可以交叉印证展示主张，但不补足账号评分、未见身份实验或刊会资格证据。保留这一具体缺口，未反复请求用户放宽条件或自行联系作者。

**本轮没有新增合格名额，仍为VendorLink一项、另外两项空缺。**这表示已核查范围内尚未找到另外两个满足条件的方法，不表示证明此类论文不存在。继续检索时保留用户已明确的输入与未见身份任务，不重复推荐已排除的多模态适配、论坛方法或旧年份工作；只有新的原文证据才能改变其状态。本轮仅更新公开文献记录，未访问实验数据、连接Linux、联系作者或改变活动比较实验。

最后补查其他作者、机构／预印本入口及GitHub公开README的精确标题，未补得ICASI候选的全文或官方实现。新发现的[darshankumar97/DARK_WEB-Vendor_Profiling](https://github.com/darshankumar97/DARK_WEB-Vendor_Profiling)已实际读取唯一源码[vendor.py](https://github.com/darshankumar97/DARK_WEB-Vendor_Profiling/blob/main/vendor.py)：将商品名称与描述拼接，使用TF-IDF一／二元词组和K-Means作商品聚类，按现有`Vendor Name`分组展示统计；没有两个账号的同控分数、账号关联监督或未见身份评价。仓库未给出对应论文，作者身份与ICASI论文的对应关系也未核实，因此既不能当作论文官方实现，也不能由商品聚类替代账号关联。未执行该源码或下载其数据。

**本阶段结束时检索受阻，目标未完成（后续恢复检索见第13节）。**引用追查（提交`c25efbb9`）、作者及出版路径补查（`00731c57`）和本轮公开实现核查后，同一缺口连续三轮未解除：没有另外两个满足既定条件且有足够原始证据的方法；尚存的ICASI直接线索仍缺正式出版／方法证据，其他具体线索已排除或不能计入。现有可追查来源暂不能继续推进资格核实，需要新的相关论文全文、可确认对应论文的官方实现或正式出版记录等外部证据；取得后仍须逐项核验，不能预先认定合格。这是当时证据条件下的受阻判断，不是不存在此类论文的证明，不更改用户条件、不计作完成，也不授权联系作者或开展适配实验。

## 13. 2026-10-23：恢复检索，补查法证语言分析与卖家关联线索

目标恢复为active后，核对现有记录并补查卖家画像、广告语言痕迹、关联卖家档案及其引用链；输入、任务、年份与刊会条件不变。以下是本轮新取得的证据，不把新增排除项算作合格名额。

| 新核查对象 | 一手证据、阅读范围与结论 |
| --- | --- |
| [Analysing the digital transformation of the market for fake documents using a computational linguistic approach](https://doi.org/10.1016/j.fsisyn.2022.100287)，Clara Degeneve、Julien Longhi、Quentin Rossy，Forensic Science International: Synergy 5 (2022):100287 | 已通过[Europe PMC公开全文XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9596726/fullTextXML)实际读取§3—5。摘要虽提出能否联系不同卖家，§3.2明确商品描述经尝试后被排除；实际分析商品标题，另分析卖家个人简介。IRaMuTeQ作下降层次分析（DHA）、对应因子分析（CFA）及词共现图；§3.4／4.4按卖家ID聚合词频、人工逐轮排除离群卖家，再依据投影位置分组。标题分析最后保留67／86个卖家，四组对应不同商品供应类型，并非四个同控身份。没有报告同控账号对训练目标、身份打分或未见身份验证。因此它是商品／卖家经营类型的探索性分组，不能因为摘要中有“link distinct vendors”就计作马甲关联方法；无须靠刊会层级弥补任务不符。 |
| [Illuminating the dark web market of fraudulent identity documents and personal information: An international and Australian perspective](https://doi.org/10.1016/j.forsciint.2024.112203)，Forensic Science International 363 (2024):112203 | 从上一篇被引记录定位，核查[出版社索引正文](https://www.sciencedirect.com/science/article/pii/S0379073824002846)的方法段。研究White House Market和Empire Market，确实比较跨市场卖家，但账号对应依据为清理格式后的PGP公钥与用户名比较，商品标题／描述用于市场与产品分析，未形成所需的纯商品文本身份打分方法。直接网页／机构PDF打开出现403，不冒称已取得完整PDF；上述具体关联规则有出版社返回正文支持。依赖本项目不可用的身份标识符，不计入。 |
| [Identification of Linked Seller Profiles on Marketplaces through Behavioral Data Analysis and Machine Learning](https://www.hse.ru/en/edu/vkr/1159983183)，Polina Trubkina，HSE大学2026年硕士论文 | 大学原始条目明确是学位论文，描述了卖家对二分类、Logistic Regression／Random Forest／LightGBM及后续图特征／邻域表示。虽然研究多账号卖家，输入还包括商品组合同步性、经营规模相似性及广告文字重合；页面称经营规模是关键预测特征。当前只核实大学条目，未取得方法全文、官方实现、未见身份拆分或对应合格刊会版本。作者名与完整题名追索未补足正式版本，不能把学位论文或摘要效果直接计为所需论文。 |
| [Darknet Market Vendor Analysis](https://www.darkowl.com/blog-content/darknet-market-vendor-analysis/)，DarkOwl，2026-06-03 | 企业原始文章以账号全部商品的归一化类别分布构造指纹，作相似性检索与经营模式分析；正文明确此类指纹是相似信号，不能独自证明身份归属。没有给出合格刊会论文、标题／描述账号对学习规则或独立身份验证。因此仅为产业线索，不计入。 |
| [Connecting Illegal Vendors on Darknet Markets: Responsible Authorship Attribution to Link and Connect Online Cybercrimes](https://github.com/maastrichtlawtech/CyberCrimelinker) | 团队公开README把它定位为教学工作坊，列出的对应发表成果仍为VendorLink（ACL 2023）与IDTraffickers（EMNLP 2023）。虽有Agora账号识别／检索notebook，不能把教学材料当作另一个已发表的新方法，也不能重复计算VendorLink。只读说明，未执行notebook或下载数据。 |

引用追查的实际范围：OpenAlex对2022年法证论文（`W4297200383`）的2022-01-01至2026-10-23被引查询返回7项。逐项筛查书目后，进一步核查上述2024年市场研究，以及[WIREs Forensic Science 2024综述](https://doi.org/10.1002/wfs2.1528)的原始摘要和[同作者2026年法语论文](https://doi.org/10.26034/la.cfs.2026.8332)的期刊摘要／参考文献；后者研究爱情诈骗案例之间重复话术的传播，未转成商品卖家账号方法。该索引返回范围不是穷尽所有被引论文的证明；首次沙盒请求连接失败，不作为零篇结果。

另定位到ICSPIS 2024《Data Analysis of Dark Web Marketplaces using Machine Learning》，[IEEE记录](https://ieeexplore.ieee.org/document/10812633)、DOI `10.1109/ICSPIS63676.2024.10812633`及[作者所在大学成果页](https://rvu.edu.in/faculty/dr-sarasvathi-v/)可对应题名。当前IEEE返回页未提供方法正文，尚未取得原始全文、账号对评分或满足刊会条件的证据，保留待核实，不以第三方摘要补齐资格。

**该轮仍为VendorLink一项、另外两项空缺，结束时goal保持active且未完成；后续核查见第14节。**新增证据进一步区分了商品类型分组、基于标识符的跨市场对应、行为特征反欺诈与本项目的纯商品文本未见身份关联。该轮留下的下一步是ICSPIS原始全文／作者公开成果及相关正式发表记录；不改变比较合同、不开展实验。

## 14. 2026-10-23：ICSPIS引用链与跨市场标识符方法核查

用户询问ICASI 2026题名后，已再次核对[Princia D’Souza的公开介绍](https://github.com/princia05/PRINCIA)与[会议官网](https://2026.icasi-conf.net/)。题名和作者展示主张未变，也没有新增正式书目、全文或账号打分证据；该答复没有增加合格候选。本轮继续处理第13节留下的ICSPIS线索，复用已经取得的出版社元数据，不将重复核对计作新方法。

**ICSPIS 2024的阅读范围已推进到作者摘要与引用链，尚未取得方法全文。**[作者Pratham Rao的公开发表介绍](https://in.linkedin.com/in/prathamrao-)给出摘要，研究重点是Tor爬虫、按商品类别及地理来源分析卖家经营模式；本轮读到的是搜索返回的作者页面内容，不是完整论文。它没有给出商品文本账号对目标、聚合／打分规则或未见身份验证，因而不能升级为可直接比较方法；也不据此断言未读全文中绝无其他分析。题名和作者另有[会议正式议程](https://edas.info/web/icspis2024/program.html)对应。

取得的[Crossref原始元数据](https://api.crossref.org/works/10.1109/ICSPIS63676.2024.10812633)包含12条出版社提交的DOI参考文献，abstract字段为空。这是书目证据，不等于读过12篇全文。由其中近期条目及其直接相关引用，补充核查如下：

| 来源与阅读范围 | 对本任务的具体判断 |
| --- | --- |
| [A Comparative Analysis of Models for Dark Web Data Classification](https://link.springer.com/chapter/10.1007/978-981-97-0180-3_20)，2024，出版社摘要 | TF-IDF／BOW与SVM、Naive Bayes、RNN用于色情、枪支、毒品、黑客等内容类别分类，未形成卖家同控账号对方法。 |
| [An Analysis of Topic Modeling Approaches for Unlabeled Dark Web Data Classification](https://link.springer.com/chapter/10.1007/978-3-031-69201-7_12)，2024，出版社摘要 | 对爬取HTML文字比较LDA、LSA、Top2Vec、BERTopic、NMF，由专家评价主题；任务是内容主题识别，不是账号身份关联。 |
| [Unveiling Challenges and Solutions with Intelligence in the Dark and Deep Web](https://link.springer.com/chapter/10.1007/978-3-031-70018-7_42)，2024，出版社摘要 | 概述暗网／深网、监测及机器学习用途；没有提供可核实的商品文本卖家关联算法，不能用框架性描述补位。 |
| [Analysis of Security Mechanisms of Dark Web Markets](https://www.cs.kent.ac.uk/people/staff/ba284/home/Papers/SecurityMechanisms-EICC2024-AAM.pdf)，EICC 2024，DOI `10.1145/3655693.3655700`；以及同团队[Secure in the dark? An in-depth analysis of dark web markets security](https://link.springer.com/article/10.1007/s10207-025-01015-1)，IJIS 2025 | 实际核读前者摘要、相关工作、方法和参考文献，后者出版社摘要与引言／相关工作。研究网站、账号和金融安全机制；账号安全指密码、PIN、MFA等，不是马甲身份学习。前者相关工作[2]、[19]分别为2019、2018年工作；[13]引向下列ARES 2023论文，已继续核查。 |
| [Cheaper than you thought? A dive into the darkweb market of cyber-crime products](https://www.ricardoyaben.com/assets/papers/ares-2023.pdf)，Dimitrios Georgoulias、Ricardo Yaben、Emmanouil Vasilomanolakis，ARES 2023，DOI `10.1145/3600160.3605012` | 已读取作者提供PDF的摘要、引言和§4.4。确实分析跨市场卖家，但图10交叉匹配用户名，图11按相同PGP公钥连接账号；原文同时指出相同用户名未必属于同一身份。商品描述相同是后续观察，不是以标题／描述产生账号对分数的算法。因此依赖本项目不可用的身份标识符，不计入；没有把“跨市场”措辞当作输入符合的证据。 |
| [Dark Ending: What Happens when a Dark Web Market Closes down](https://www.scitepress.org/PublishedPapers/2023/116816/)，ICISSP 2023，DOI `10.5220/0011681600003405`；[作者大学记录](https://kar.kent.ac.uk/100540/) | 出版社和大学摘要明确分析六个市场关闭前的变化与预警特征，未提供所需卖家账号关联方法。部分PDF入口超时／403，未据此声称全文已读或论文不存在。 |
| [Research on Intelligence Mining of Illegal Underground Internet Production on Anonymous Network: Taking Personal Information Trading as an Example](https://doi.org/10.1109/ITNEC56291.2023.10081964)，ITNEC 2023 | DOI由上述出版社参考文献指向，书目线索可对应；IEEE入口和题名／DOI检索仍未取得可用原始方法正文，不能核实其是否包含本任务。保留“证据不足”，不伪装成已证实任务不符，也不把ResearchGate页面中其他推荐论文的摘要当作本篇摘要。 |

**目前仍仅VendorLink计入，另外两个方法未补齐。**恢复检索后的法证文献追查、ICASI核实及本轮ICSPIS引用追查，均未解除同一缺口：没有另外两个同时具备合格发表出处、纯商品文本输入、账号对方法与未见身份适用证据的方法。具体可定位来源已核查到上述范围；余下直接线索仍需要新的正式出版记录、可读方法全文或能确认对应论文的官方实现，现有材料不足以继续完成资格判断。这是本次恢复检索后的受阻状态，不是证明此类论文不存在，也不影响其他已授权科研工作。目标未完成；不放宽用户条件、不再反复请求接受文本适配、不联系作者、不改比较合同或开展实验。

## 15. 2026-10-23：继续引用追查，补足IEEE原始摘要证据

用户指出前述停留于受阻和排除清单的处理未达到要求。本轮继续执行原检索要求；第14节末的受阻叙述是该轮结束时判断，不能替代继续追查，也不能把取得公开证据的责任推给用户。合格名额仍按独立方法计数，不能用搜索次数、排除项数量或同一论文的不同入口补位。

重新检索VendorLink及早期市场关联方法、2024年度量学习方法的被引书目，并追查作者机构成果页。OpenAlex联合引用查询在2022-01-01至2026-10-23范围返回63条书目；这只是该索引、该查询的返回范围，包含重复版本和不相关论文，不是63篇方法全文，也不证明检索穷尽。新增直接线索GraySniffer与Software Solutions的出版社参考文献已取得，分别35条、15条；参考文献回查没有产生新的合格方法。

检索工具未能读取的IEEE页面，经本地Playwright正常公开网页入口实际读到了下列原始摘要／开放引言。需要登录的正文没有读取，不把摘要核实写成全文核实；本轮专用标签页已关闭。

| 来源 | 本轮补足的一手证据与资格判断 |
| --- | --- |
| [GraySniffer: A Cliques Discovering Method for Illegal SIM Card Vendor Based on Multi-Source Data](https://ieeexplore.ieee.org/document/10945061)，TrustCom 2024，DOI `10.1109/TrustCom63139.2024.00065` | IEEE摘要明确输入为群聊、论坛和对话；RoPE与BiLSTM抽取实体属性，构建情报库，再由GNN及自编码器发现售卡团伙。报告3792节点、25602关系与12个团伙。它研究多个卖家的组织关系，不是证明多个账号属于同一卖家的商品文本配对方法；依赖额外关系数据，不计入。没有取得完整方法正文。 |
| [Software Solutions for Identifying Users Behind Social Media-Based Drug Trafficking](https://ieeexplore.ieee.org/document/11380142)，ICEC2NT 2025，DOI `10.1109/ICEC2NT65402.2025.11380142` | 会议日期2025-09-03—04，IEEE收录日期2026-02-19，分别保留。原始摘要与开放引言说明用POS、关键词、上下文及CNN图片模型检测社交媒体涉毒活动，图片训练集为Roboflow毒品检测数据；88%是检测准确率，不能写成卖家账号关联准确率。题名中的“Identifying Users”不足以证明同控配对，不计入。 |
| [Research on Intelligence Mining of Illegal Underground Internet Production on Anonymous Network: Taking Personal Information Trading as an Example](https://ieeexplore.ieee.org/document/10081964)，ITNEC 2023 | 第14节仅有书目线索，本轮补到IEEE摘要和开放引言：主动／被动节点发现、爬虫采集，再按时间、空间、类型、主题分析个人信息交易。已读部分没有商品文本账号表示、配对目标或同控评分，不能升级为直接比较方法；没有声称未读全文中绝无其他分析。 |
| [Data Analysis of Dark Web Marketplaces using Machine Learning](https://ieeexplore.ieee.org/document/10812633)，ICSPIS 2024 | 从作者介绍推进到IEEE原始摘要、开放引言；出版社明确以自建Tor爬虫获取市场数据，按商品类别、地理来源分析卖家经营模式。现有证据没有给出账号对关联算法，保留不计入；全文仍需登录，未取得。 |
| [Follow Your Hidden Traces: Underground Forum-Based Darknet Market Sybil Detection](https://ieeexplore.ieee.org/document/11169719)，ICWS 2025 | IEEE原始摘要和开放引言再次明确：筛选论坛帖子、抽取马甲卖家信息、核验市场存在性，论坛是额外输入；不改变第8、12节判断。另查[作者实验室成果页](https://issl.korea.ac.kr/publications)与[作者主页](https://yrkim.oopy.io/)，未取得该篇全文下载入口。正式[会议议程](https://services.conferences.computer.org/2025/wp-content/uploads/sites/3/2025/07/SERVICES_webprogram7-5.pdf)将其列为short paper（CWS_SHT_65），不能省略论文类型。 |

另沿引用链读取ECAI 2024《Entity Matching Across Small Networks Using Node Attributes》的[作者机构摘要](https://publications.idiap.ch/publications/show/5369)，并定位到[作者原稿](https://publications.idiap.ch/attachments/papers/2024/Ahmadi_ECAI-PAIS2024_2024.pdf)的索引片段：UGC-DeepLink使用通话／论坛交互形成的小网络及节点属性，结合图结构学习和文本信息匹配实体；不是仅用商品标题、描述关联卖家，不能因ECAI刊会符合就补位。新的中文作者线索指向[《面向暗网抑制的普适性安全理论研究构想和成果展望》](https://doi.org/10.12454/j.jsuese.202400800)，工程科学与技术2025年第57卷第1期；已读摘要和开篇研究构想，其流量识别、连接性预测、陷阱节点溯源方向没有提供本项目所需商品账号对方法，不以综述／研究构想充数。

**本轮没有新增合格方法，仍是VendorLink一项、两个空缺。**实际进展是新增线索和若干原先未能访问来源的一手核实；它不等于完成目标，也不构成不存在其他合格文献的证明。ICASI 2026仍是缺正式出版与方法证据的线索，不升级为比较方法。检索、全文追查与资格判断继续由执行者承担；本次只更新文献事实与阅读范围，不改变实验合同、运行或比较方案。

## 16. 2026-10-23：保留L²R，第二个持续学习对照的GCR核查

用户明确保留L²R，要求继续找2022—2026年、CCF A/B或知名刊会的第二个持续学习方法；允许适配商品标题／描述账号关联，但须保留原方法核心、公平比较抗遗忘。持续学习组不要求原论文也研究暗网卖家，静态卖家方法组的资格不随之改变。

**当前选型：L²R已由用户确认保留；GCR的ER+GCR完整群适配已完成网页外审和独立主审，确定为第二个推荐对照。持续学习组选型2/2，静态卖家组仍为VendorLink一项、另两项未补齐。**完成的是方法选型，尚未实施或复现；不把新方法加入已冻结五方法运行，不更改原固定随机记忆合同。具体独立处置见第16.5节。

### 16.1 正式身份与实际阅读

Rishabh Tiwari、Krishnateja Killamsetty、Rishabh Iyer、Pradeep Shenoy：**GCR: Gradient Coreset based Replay Buffer Selection for Continual Learning**，CVPR 2022，99—108，DOI：[10.1109/CVPR52688.2022.00020](https://doi.org/10.1109/CVPR52688.2022.00020)。正式发表年份2022，预印本首发2021不改变该身份；CVPR在所给CCF目录人工智能部分为A类。

已通读CVF正式开放版10页正文（含实验、局限和参考文献）及4页补充材料，逐页核对第102—103页式(4)和算法1—2、补充§A抽样算法、§C／表9的ER+GCR。保留[正式论文](../reports/seller_alias_papers/20261023/continual_baseline/gcr.pdf)和[补充材料](../reports/seller_alias_papers/20261023/continual_baseline/gcr_supp.pdf)。公开入口：[作者项目页](https://gradientcoreset.github.io/)、[CVF正文](https://openaccess.thecvf.com/content/CVPR2022/papers/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.pdf)、[CVF补充](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Tiwari_GCR_Gradient_Coreset_CVPR_2022_supplemental.pdf)。

### 16.2 原方法与适配依据

式(4)联合选择有限子集和非负权重，使加权回放梯度接近可用集合的梯度，并正则化权重。OMP选择后，权重必须实际用于训练；下一轮目标沿用存活旧记忆及其权重，不能恢复淘汰数据。当前候选池由reservoir维护，训练包含旧记忆／当前候选池的自适应抽样。

完整GCR另含历史logit蒸馏与监督对比损失。**本次明确建议作者补充§C／表9验证的ER+GCR组合，不冒称完整GCR配置。**正文§4.1说明GradApprox可用于其他指定回放损失，补充给出实际ER组合证据。原图像实验不是本项目效果证据。

以下为主执行者的适配推论，尚未执行：

| 组成 | 账号关联适配及边界 |
| --- | --- |
| 监督目标 | 沿用共同BGE编码、账号打分和监督目标，以完整训练群选样并求梯度，保留候选排序；不能只抽孤立账号对并删掉排序项。 |
| GradApprox／OMP | 用当前候选群和存活历史群做梯度匹配、非负权重拟合、加权回放；只按梯度范数排序、选后权重全置1或只记录不更新均不算实现。 |
| 记忆递推 | 延续旧权重，保留有界候选池及相应抽样／更新；不读回淘汰历史、不接触未来或评价标签。 |
| 原逐类配额 | 完整群具有固定关系构成，不能给群伪造身份类别。按群选样须披露为适配差异，不声称逐字复现图像逐类配额。 |

该机制不依赖图像、商品类别或发货地，也不要求新增伪标签或固定卖家分类输出，故可以保留未见身份账号排序任务。梯度选择依据必须是允许的训练监督，不使用评价身份或结果。

### 16.3 公平性与实现证据

- 同输入、监督、划分、起点、顺序／种子和开发预算；共同观察旧域保持、新域学习及最终排序，不以少学习换取低遗忘。
- 计量文本、标签、权重等全部历史状态，并单列当前候选池、峰值内存和梯度选择开销。原论文候选池与回放池各K，§7承认额外内存，不能把候选池当作免费空间。
- L²R和GCR须有各自选样权。强制与当前所有臂使用同一Algorithm R成员会删掉它们的核心机制。新比较须在共同容量／访问范围内允许算法选样；这不改动活动实验。
- [作者代码](https://github.com/google-research/google-research/tree/master/gradient_coresets_replay)已静态读取：`gcr.py`权重进入回放损失及反向更新，`gradmatchstrategy.py`调用OMP，`dataselectionstrategy.py`采用输出层／最后线性层梯度表示。文本适配须明确梯度空间并直接求实际打分头导数，不能把分类头特征乘积公式照搬到记录池化模型。
- 当前公开实现的具体差异／缺陷：论文逐类等额、代码默认按类比例；`coreset.py`仍导入`continual_learning_rishabh`旧包名；`gradmatchstrategy.py`传`perClass=True`，被调用签名却是`per_class`。因此没有声称官方仓库开箱可运行。实施时须固定论文／代码取舍并作最小修复及真实接线核验。
- 权重使用还有一个与当前小批训练直接相关的必要条件：若每次只回放一群，再除以该批权重和，所得权重会被自身消去。选样目标、全局归一化、抽样概率及实际梯度必须一致，不能把“张量里有权重”当成加权回放生效。式(4)左侧使用旧权重定义目标，右侧拟合新的实际权重，不能未经推导重复乘旧权重。

### 16.4 其他线索与外审状态

DwoPP（BMVC 2022）与未见身份的持续度量学习直接相关；所给CCF目录将BMVC列为C类，且方法保留旧教师。本轮已有A类适配候选，优先GCR。DGC（ICML 2024，[正式出版页](https://proceedings.mlr.press/v235/lin24a.html)）取得18页全文，定向读§3.1—3.3／算法1：递推历史梯度校正量并保留参考参数，额外状态与现行极小历史预算衔接更复杂，暂不优先。OCS（ICLR 2022）只作检索线索，未完成全文核查，不作排除或效果结论。

19:35曾把模型菜单无障碍快照中的“已达到使用上限”误读成GPT-6 Pro不可用。用户追问后，19:43复查可见界面及DOM，确认两条提示为`sr-only`文本，并未绑定当前GPT-6选项；界面显示剩余8%。该错误判断已撤回。19:46通过指定本地Playwright单次成功提交[公开方法适配审查](https://chatgpt.com/c/6adb48e9-5e88-83e9-a68d-1d728a23faf5)，出现“正在回应”及“停止”，可见GPT-6／Pro；[实际请求](../reports/seller_alias_papers/20261023/continual_baseline/review/request.zh.txt)只含公开论文／代码和通用适配论证，无项目文件、数据、实验结果、运行参数或凭据。以[回执](../reports/seller_alias_papers/20261023/continual_baseline/review/submission.json)记录后续完成状态；此时未取得最终审查，不写已通过。本轮未SSH或运行科研实现。

### 16.5 外审回收、独立主审与最终选型

2026-10-23 20:09（Asia/Shanghai）从原会话取得完整最终回复，页面已无可见停止按钮；[完整页面原文、链接及21段公式TeX](../reports/seller_alias_papers/20261023/continual_baseline/review/external_review.json)已保留，无返回附件。外审结论为“推荐ER+GCR的完整群任务适配”，未发现已成立的选型阻断；并要求明确单位／梯度／权重／容量定义。这里记录的是要求GPT-6 Pro、提交时可见GPT-6／Pro与实际收到的回复，不推定后台模型身份。

主执行者独立复核范围：复用先前全文阅读，本轮重读本地CVF正文102—103页式(4)、算法1—2、式(7)及补充§A、§C／表9；在线CVF再次请求403不影响已保留全文的核对。另亲自重读官方`select_subset.py`、`coreset.py`、`gcr.py`、`gradmatchstrategy.py`、`dataselectionstrategy.py`的相关路径，并核对本地`step28_record_replay.py`的`account_logits/objective/backward_group/update`。没有运行官方代码、训练或重新计算实验指标，以下数学对应与静态发现不冒称运行效果。

| 审查事项 | 独立判断与处置 |
| --- | --- |
| 作者ER+GCR版本是否真实存在 | 补充§C／表9直接成立，正文§4.1允许指定回放损失。采用此版本有原文依据；不将其冒称带蒸馏及监督对比的完整GCR。 |
| 完整群与原目标能否保留 | 本地实际目标以完整群重建账号分数、成对监督和候选排序，群外数据不进入该群损失。群可作为GradApprox原子；选择和回放均须保留完整目标。群内正负构成一致不等价于论文逐类等额，明确取消逐类配额，不增造群身份类别、正负双池或配额机制。 |
| 梯度空间 | 统一选择时点参数，计算每群完整目标对真实打分头全部参数的导数；旧群重新求当前梯度，只继承旧权重。头梯度是参数子空间近似，不保证编码器全梯度匹配；实际训练仍更新共同设定允许更新的编码器和头。 |
| 权重递推 | 令原始群梯度为g，目标b=Σ历史旧权重×g+Σ当前g；选择S及非负新绝对权重v，最小化‖b−Σ_S v×g‖²+λ‖v‖²。字典列不再乘旧权重，新v保存后直接作用于原始群损失；不重复乘旧权重、不随意把每轮权重归一到均值1而保留新候选权重1。 |
| 单群权重抵消 | 官方批内除以权重和的写法，单群时确实得到wL/w=L。本适配采用按抽样群数平均的带权回放（单群即a_G L_G，当前候选a=1、历史a=保存权重），外层系数另按比较定义；不以批内权重和抵消。保留池间p=n_t/n及池内均匀抽样，计数统一按群，实际期望为p/|C|Σ_C L+(1−p)/|M|Σ_M wL，不冒称全历史均匀平均风险。空池启动须在实施时明确，不能抽空池。 |
| 公开代码额外权重问题 | 已独立确认：`CustomLoss`把旧权重乘入损失，策略对所得带权梯度做OMP，`coreset.py`又直接写回OMP系数；未见转换回未加权损失绝对系数的步骤。这是与论文目标的静态语义差异，不是已观察的训练失败。另CPU OMP传lam=0、GPU传self.lam，不能默认两分支都落实式(4)正则。后续按上述论文定义实现并定向核验，不以修正包名和关键字冒称复现完成。 |
| 预算与评价 | 允许各方法自己的选样；如果声称等总数据记忆，当前候选池的文本／标签也必须计入总额。如果采用等历史记忆预算，则另报当前池及峰值，不能混称等总内存。L²R保留的查询／正例同样计量。共同评价旧域保持、新域学习和最终排序；不把低遗忘单独当作胜出。 |

**独立主审结论：推荐ER+GCR完整群适配作为第二个持续学习对照，当前无已成立的选型阻断。**不要求先做图像基准复现、额外消融、全编码器梯度对照或新增种子才能完成选型；这些不属于本轮必要交付。后续实施只需证明原目标、选择／权重链条、真实更新及资源边界确实生效，再按获准协议运行；本轮没有授予新比较／训练预算，也未改变活动实验。

按用户最初对比角度，最终已选与空缺如下。组名采用“是否专门包含抗遗忘机制”，避免在未做本任务比较前把“未解决／解决了遗忘”写成已证实结论。

| 组别与名额 | 选取结果 | 作用与边界 |
| --- | --- | --- |
| 暗网卖家账号关联1 | VendorLink，ACL 2023，CCF A | 商品标题／描述→广告表示→账号集合归一化相似度，直接关联卖家马甲；原论文不以持续抗遗忘为核心。保留为任务方法对照，中文复现尚未做。 |
| 暗网卖家账号关联2、3 | **仍缺两项，不计作完成** | ICICS 2023需要图片、类别和发货地；TrustCom 2025需要图片且匹配已学卖家类别；用户未接受这些适配替代。ICASI 2026仍缺正式出版与方法证据；论坛、商品分类、数据集论文不补位。 |
| 持续学习1 | L²R，CIKM 2023，CCF B | 用户已保留；使用式(8)／表3不加兼容约束的版本，保留支持负例PSS、查询条件多样性ISD及记忆更新。账号查询／同控正候选／异控负候选为任务映射，查询与正例历史不免费；不是表4已固定旧表示的vanilla版本。 |
| 持续学习2 | ER+GCR完整群适配，CVPR 2022，CCF A | 保留OMP梯度匹配、正则与非负权重、记忆递推、候选池和实际加权回放；按本节明确选择单位及权重差异。外审与独立主审已完成，未实施或报告本任务效果。 |

因此，**当前方法选型为静态卖家组1/3、持续学习组2/2，总计3个已选方法、2个空缺**。两项持续学习对照分别覆盖文本排序中的负例记忆选择和训练梯度保持，两者可用于检验所提方法相对于已有抗遗忘机制的收益；选中它们本身不能证明本项目方法有效，也不预设输赢。

## 17. 2026-10-24：固定持续学习组后续查静态卖家方法

用户本轮明确保留L²R与ER+GCR；第16节的持续学习组选型现为**用户已确认2/2**。本轮只续查暗网卖家组，要求2022—2026年、CCF A/B或知名期刊会议，实际输入仅商品标题／描述，输出能关联训练未见身份的卖家账号。将多模态模型删掉其他模态、将固定卖家分类器另改配对器、只把未知类别拒识称为未见身份关联，均不能据此计入。没有新论文完成符合这些条件的全文确认，**本轮新增合格方法0项；VendorLink仍为1/3，另两项未补齐。**

本轮继续从VendorLink及已知多身份方法的相关文献、跨市场卖家研究、法证／犯罪学刊物和中英文商品卖家关联检索推进。下表记录新增或补强的一手证据；“读了方法段”不写成“通读全文”，只取得摘要的论文也不据此断言全文绝无其他方法。

| 来源与实际阅读范围 | 能确认的内容及本轮处置 |
| --- | --- |
| [Identifying & linking Darknet vendors through NLP applications](https://clin2022.tshdresearch.org/identifying-linking-darknet-vendors-through-nlp-applications/)，CLIN32 2022；完整官方报告摘要 | 作者Saxena、Spanakis、Van Dijck与VendorLink重合；内容为BERT监督卖家训练、广告嵌入相似度、七市场迁移／马甲关联与知识蒸馏。官方页面没有提供独立于VendorLink的第二套方法全文；按同一研究线的早期报告处理，不把报告和ACL论文占两个名额。 |
| [Decoding hidden darknet networks: What we learned about the illicit fentanyl trade on AlphaBay](https://doi.org/10.1111/1556-4029.15341)，Journal of Forensic Sciences 2023；从作者中心公开19页PDF定向读§3.3.3、§5及相关前后段落 | §3.3.3（PDF第9页）以Gephi构造卖家与商品、芬太尼／暗语的二部网络，边不表示两个账号同控。§5（PDF第15页）讨论以卖家名和PGP公钥跨市场追踪。它没有在已核对的方法中给出仅标题／描述的未见身份账号对评分，不能将“隐藏网络解码”当作马甲配对算法。[作者中心全文入口](https://ccybers.org/wp-content/uploads/2024/07/Journal-of-Forensic-Sciences-Maras-at-al.-2023-Decoding-hidden-darknet-networks-What-we-learned-about-the-illicit-fentanyl__.pdf)。网页已读到上述原文；本地下载超时，未声称已保管PDF或通读19页。 |
| [The Network of Online Stolen Data Markets: How Vendor Flows Connect Digital Marketplaces](https://academic.oup.com/bjc/article/62/6/1518/6503727)，The British Journal of Criminology 2022，62(6):1518–1536；出版社正文的Digital Trace Data与Analytic Approach及相关讨论 | 原文明确要求卖家别名精确相同才判为同一卖家，并承认不同别名／冒用别名的限制。ERGM建模的是市场之间卖家流动，不是从商品文字识别账号同控；不能把市场网络的边当成账号对分数。此前机构摘要线索本轮补到原始方法正文。 |
| [Distinguishing Sellers Reported as Scammers on Online Illicit Markets Using Their Language Traces](https://www.mdpi.com/2226-471X/9/7/235)，Languages 2024，9(7):235；出版社原始摘要与公开重印本的摘要页 | 确实使用卖家主页／商品描述等语言线索，但目标是区分被举报诈骗的卖家与信誉卖家，并使用Dread报告；不是同控账号关联。未取得并通读完整方法，不作全文核实合格声明，也不将这类状态分类作为身份配对方法。 |
| [From Sign-Up to Multi-million Revenues: A Deep Dive Into Vendors on Darknet Marketplaces](https://link.springer.com/chapter/10.1007/978-3-032-00635-6_13)，IWCC／ARES 2025 workshops，pp.221–238；出版社摘要、出版身份及参考文献 | 研究卖家入驻条件、市场佣金与收入；现有公开内容没有可直接比较的商品文本账号对算法。参考文献继续指向2019年Adversarial Matching等已知工作。正文为订阅内容，本轮没有取得；记录为缺乏方法证据，不把工作坊冒称ARES主会论文，也不凭题名推定它完全没有身份分析。 |
| [Automatic User Profiling in Darknet Markets: a Scalability Study](https://arxiv.org/abs/2203.13179)，2022；作者原始摘要 | 研究文本风格人口属性画像（年龄／性别）及规模变化，不是卖家账号同控。所见记录为预印本，未取得符合要求的正式刊会证据，不计入。 |
| [FCA-XLNet-BiGRU Multi-task Framework for Darknet Transactions](https://link.springer.com/chapter/10.1007/978-981-95-3072-4_8)，KSEM 2025，书卷标©2026；重新核对出版社摘要 | 原摘要的closed-set classification／open-set detection描述交易类别识别，不能据“open-set”一词推导为对训练未见卖家进行账号配对。仍未取得完整方法正文，维持不计入，不写成已完成全文排除。 |

另定位到2025年综述[NLP-based techniques for Cyber Threat Intelligence](https://doi.org/10.1016/j.cosrev.2025.100765)的出版社与作者机构公开版本；实际读到的相关段落将其[144]描述为暗网市场威胁分类，尚未从中取得新的商品账号关联方法证据，未声称核读全部引用。2026年《Automated Models for Darknet Monitoring and Criminal Attribution》只取得书籍章节的书目／二手摘要线索，没有合格刊会或完整方法证据；不引用其中准确率作为本项目可比较效果，不计入。

本次结果不证明满足全部条件的另外两篇论文不存在；也没有找到足以将两个空缺升级为正式对照的证据。已知ICICS 2023、TrustCom 2025、多模态度量方法和ICASI 2026的限制仍按第8—15节处理，不重复包装为新候选。本轮只补公开文献事实、阅读范围与用户决定，没有新增科研方案／效果结论，不重新外审既定持续学习选型；未实施、训练、连接Linux、追加预算或改变活动合同。
