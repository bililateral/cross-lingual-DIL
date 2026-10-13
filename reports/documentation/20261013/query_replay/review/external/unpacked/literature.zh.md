# 一手文献、实际阅读范围与定向对应

检索／访问日期：2026-10-13。以下均为作者论文或正式会议论文，不以二手概述、搜索摘要或任务名称代替方法核对。`review.zh.md`中的[L1]—[L10]对应本表。

“取得全文入口”表示浏览工具能够返回该论文正文，并不表示每页都逐行审查。实际阅读以各项列出的范围为准；没有复现论文代码／实验、重新核验论文全部理论或建立穷尽查新。公开PDF已调用网页截图查看指定方法页。为避免不必要转载，本回收包不附论文全文，只附来源、阅读记录和本次对照意见。

## [L1] ConSlide

**ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis**。

一手全文：`https://arxiv.org/html/2308.13324`

实际阅读：方法第4节，重点§4.2的Breakup-Reorganize Rehearsal、Algorithm 1，以及§4.3的跨层对齐；本次返回正文约第72—137行。未系统审查实验表、全部消融或实现代码。

关键对应：从记忆中的区域／patch特征按任务与同类标签重组袋，说明同类重组式持续重放不是新原则。其跨尺度对齐是层级表示间的约束，不能直接当作C_q的原账号内中心化响应R。差别在输入分配、被保存函数和保持投影，不在医学与文本任务名。

## [L2] C²Aug

**Contrastive Cross-Bag Augmentation for Multiple Instance Learning-based Whole Slide Image Classification**。

一手全文：`https://arxiv.org/html/2508.03081`

实际阅读：§3.1—§3.4，重点§3.2的同类跨袋多视图融合／实例扩展压缩、§3.3的EMA教师与袋级对比、§3.4的原型分组对齐及式(8)；正文约第67—128行，并定向回看约第156—160行。未审查作者代码或全文实验统计。

关键对应：同类输入增强与教师对齐已有直接组合。其分组由学习原型组织，使用概率对齐，不是根据原自然账号做u的中心化，也不是在冻结出生表上按有限总体b加权。不能仅以“同类增强＋教师”宣称全流程等价，亦不能把该组合原则写成C_q首创。

## [L3] PseMix

**Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole Slide Image Classification**。

一手全文：`https://arxiv.org/html/2306.16180`

实际阅读：方法第III节，尤其表型分层、无放回实例分配、伪袋划分和Algorithm 1；袋级掩码混合式(4)、目标混合式(6)及其周边；正文约第118—231行。未系统复核实验与代码。

关键对应：原袋内部的结构化无放回划分、伪袋混合与分布保持已有。它的混合目标和伪袋单位不同于C_q的原监督槽标签，也没有本稿固定自然查询、排除其全部来源和解析响应保持的完整运算。

## [L4] MoCHi

**Hard Negative Mixing for Contrastive Learning**，NeurIPS 2020论文。

一手全文：`https://papers.nips.cc/paper_files/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf`

实际阅读：§3.2相关设定、§4.1和§4.2主要算法／公式及部分§4.3；实际截图查看印刷第4、5、6页。未逐项分析返回页上的实验图值或全文附录。

关键对应：对每个查询，从难负例嵌入构造归一化混合负例；另有把查询本身混入负例的变体。因此查询条件化混合已有，但不是原记录守恒、同控分量内按原槽联合分配，也没有C_q的旧表投影保持。

## [L5] SupReMix

**SupReMix: Supervised Contrastive Learning for Medical Imaging Regression with Mixup**；取得页面明确标注arXiv:2309.16633v4，2025-03-23。

一手全文：`https://arxiv.org/html/2309.16633`

实际阅读：§3.1—§3.3，式(1)—(6)及相关定义，尤其正文第151—157行的anchor-exclusive正例；亦查看摘要、问题设置和方法理论段的说明，未逐项重证附录定理或核验实验结果。

关键对应：式(1)的负例混合可以含锚点；式(3)的正例混合由两个非锚点嵌入构成，连续标签插值等于锚点标签。**固定锚点／排除锚点混合并非本稿首创。** 但非锚点特征凸组合不等于排除查询账号所有记录后构造整行27个候选，更不等于C_q的出生表R。

## [L6] FDR

**Measuring and regularizing networks in function space**。

一手全文：`https://arxiv.org/html/1805.08289`

实际阅读：§2的函数空间距离定义，§3.1的任务序列与功能保持，正文约第56—72、101—118行。未独立审查全文实验、实现或全部拓展讨论。

关键对应：保留旧输入上的网络输出并在之后约束函数变化是直接先例；任务边界输出与本稿出生阶段固定T有相近来源。C_q仍须解释为什么选择特定的原账号边际投影，而不能把存输出／匹配响应本身视为新方法。

## [L7] DER／DER++

**Dark Experience for General Continual Learning: a Strong, Simple Baseline**，NeurIPS 2020论文。

一手全文：`https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf`

实际阅读及截图：印刷第3、4页，§3、Algorithms 1/2、式(5)/(6)及§3.1相关比较。未独立核验全文实验、代码或附录。

关键对应：式(5)为当前监督加历史logit平方匹配；式(6)另加历史标签监督。论文区分DER的训练轨迹输出与FDR的任务边界输出。C_q可直接归类为固定投影Φ(K)的响应匹配，但DER本身未指定本稿的自然查询候选生成和有限总体投影。

## [L8] ReMix（本轮补充定向核读）

**ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification**。

一手全文：`https://arxiv.org/html/2207.01805`

实际阅读：§2.1—§2.3，正文约第81—138行；包括原型／协方差压缩、同类袋选择、最近原型匹配以及追加、替换、插值、协方差增强。未审查全文实验或作者实现。

关键对应：维护类身份的跨袋混合已有更具体的机制。ReMix会增强查询袋、使用原型匹配；本稿则固定完整自然查询，另一端不做最近原型匹配，而按原槽大小联合分配。这是运算上的差别，不是术语替换的证明。

## [L9] RankDistil（本轮补充定向核读）

**RankDistil: Knowledge Distillation for Ranking**，AISTATS 2021，PMLR 130。

一手全文：`https://proceedings.mlr.press/v130/reddi21a/reddi21a.pdf`

实际阅读及截图：印刷第3、4、5页；§2的重点排序定义，§3的Algorithm 1，教师Top-p与学生负例筛选、§3.1的排序分布目标及§3.2的二分类版本。未把§3.3全文、附录证明或全部实验计为已审。

关键对应：直接保持教师排序／重点候选有成熟先例。C_q的R却是数值幅度保持，top5-hard来自沿用标签监督，不是RankDistil式教师排序抽样；不能声称R只保持名次，也不宜凭“排序任务”认定损失逐式相同。

## [L10] Logit Standardization（本轮补充定向核读）

**Logit Standardization in Knowledge Distillation**。

一手全文：`https://arxiv.org/html/2403.01427`

实际阅读：§3与§4.1—§4.3、Algorithms 1/2及中心化／尺度标准化公式，正文约第76—204行。未完整重证全文理论，也未执行作者代码／复核效果。

关键对应：通过平移及尺度处理再蒸馏相对响应是已有研究方向。本稿v只在原候选账号内减均值，不除标准差；R保留幅度，且加有限总体权重，故不等于该标准化KL。该区别也说明C_q不能把“相对”解释为纯名次或尺度自由。

## 检索范围与未建立的结论

本轮沿已列方法名称核对一手正文，并定向检索过anchor-exclusive mixing、query-conditional mixing、centered response distillation、MIL/ReMix及ranking distillation等组合。部分泛关键词检索返回词典、旅行或化学等不相干结果，未据其摘要形成结论；使用准确论文名取得上述方法正文。

实际取得的十项全文入口没有出现阻止阅读相关方法段的访问失败。Files对本次ZIP的正文读取失败属于附件解析限制，与论文全文访问无关。没有声称取得未列的论文全文、审查所有2026年文献，或验证搜索索引完整性。

在上述已读方法段中，没有找到与C_q全部核心运算逐式相同的论文。这不是首创证明。普通自然账号内bootstrap的有放回方差式由本次直接推导和人工枚举给出，没有为“普通bootstrap”虚构一篇专门对应本项目的论文。已有结构蒸馏的项目处置仅作为输入包内的关闭背景阅读，不冒称本轮重新取得并全文审查那些论文。
