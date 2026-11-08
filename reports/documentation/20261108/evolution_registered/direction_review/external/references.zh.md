# 公开原始文献与外推边界

检索日期：2026-11-08。仅用于研究方向推理；没有下载或执行任何作者代码，也没有运行项目实验。以下“核读”指实际打开原文并读取所列相关章节，不声称全文复现。

## L1. VendorLink

Saxena, V., Rethmeier, N., van Dijck, G., & Spanakis, G. (2023). *VendorLink: An NLP approach for Identifying & Linking Vendor Migrants & Potential Aliases on Darknet Markets*. ACL 2023, 8619–8639.

- 论文主页：[ACL Anthology](https://aclanthology.org/2023.acl-long.481/)
- 原文：[PDF](https://aclanthology.org/2023.acl-long.481.pdf)
- 作者官方实现：[maastrichtlawtech/VendorLink](https://github.com/maastrichtlawtech/VendorLink)
- 核读位置：§3预处理与每账号广告条件；§4.1分类训练；§4.2表征及归一化相似度；§4.3低资源迁移；§5.2末四层选择；Appendix A.2训练设置；官方README的verification、identification及adaptation段。
- 支持点：账号分类监督后以广告表征进行卖家文本相似度分析，有直接应用和方法来源。
- 外推差异：本案4/2/2、阶段重建head、固定gallery、六群Memory和连续遗忘均不是原文协议；原文潜在别名不构成本案真实同控标签。40epoch只是原文部分架构基线的设置，不能证明本案最优。
- 审查判断：V可保留为明确声明的顺序检索适配；不能只借名称，也不应改成作者累积分类器后声称更忠实原文。

## L2. NoveltyTask

Varshney, N., Gupta, H., Robertson, E., Liu, B., & Baral, C. (2023). *A Unified Evaluation Framework for Novelty Detection and Accommodation in NLP with an Instantiation in Authorship Attribution*. Findings of ACL 2023, 1794–1818.

- 主页：[ACL Anthology](https://aclanthology.org/2023.findings-acl.113/)
- 原文：[PDF](https://aclanthology.org/2023.findings-acl.113.pdf)
- 核读位置：摘要与任务定义；§3.4 Methods for Novelty Accommodation；结果分析中“Catastrophic Forgetting…”及Figure 5；已知样本抽样混合的说明。
- 支持点：新增作者标签的反馈学习可伴随旧类损失，混入旧样本在其任务中有帮助。
- 外推差异：Amazon评论作者分类，含新颖性检测与反馈；类别输出、F1、数据规模与本案多记录固定池检索不同。不能据其结果声称Evolution必然遗忘。
- 审查判断：这是应用动机的先例，不是本案遗忘门的外部资格证明。

## L3. AA-CIL

Rahgouy, M., et al. (2024). *Towards Effective Authorship Attribution: Integrating Class-Incremental Learning*. 本审核读固定版本arXiv:2408.08900v1，不额外推定发表状态。

- 原文：[HTML v1](https://arxiv.org/html/2408.08900v1)
- 版本页：[arXiv v1](https://arxiv.org/abs/2408.08900v1)
- 核读位置：§IV-B作者分批与文档60/20/20；§IV-C六/十阶段设置；§V方法类别；§VI-B.2遗忘、§VI-B.3冻结特征提取器。
- 支持点：作者不断加入可作为持续学习问题；冻结与微调的收益受分批设置影响。
- 外推差异：推理涉及旧新作者类别的分类；本案V仅用分类训练encoder、推理靠gallery检索。文档量、阶段数和候选变化不同，不能照搬其下降幅度或性能门。
- 审查判断：支持对比不同更新选择的必要性，不证明本案需要更多epoch或更强干预才能出现“正确”结果。

## L4. GEM

Lopez-Paz, D., & Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning*. NeurIPS 2017.

- 官方原文：[PDF](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)
- 核读位置：§2 Backward Transfer与式(3)；§3式(5)后关于固定历史预测和允许旧任务改善的讨论。
- 支持点：旧任务随继续学习改善属于正向后向迁移。严格固定旧预测会限制这种改善，与允许旧损失下降的约束不同。
- 外推差异：本文分类任务和约束算法不等于H；H是有限权重软MSE，不能说它必然禁止正迁移。当前负收益机制仍未隔离。
- 审查判断：F<0不该被当成必须修复的实验失败；不用为本次方向问题新增GEM实验。

## L5. DER / DER++

Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). *Dark Experience for General Continual Learning: a Strong, Simple Baseline*. NeurIPS 2020.

- 官方原文：[PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf)
- 官方摘要：[NeurIPS](https://papers.neurips.cc/paper_files/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
- 核读位置：摘要与引言的ER对照动机；§3式(5)–(6)、Algorithms 1–2；突变时历史logit偏差的讨论。
- 支持点：reservoir回放、历史输出平方约束与额外标签监督构成已有机制家族；ER是有信息量的必要参考。
- 外推差异：DER储存优化轨迹中的分类logits，H储存群出生时的账号对原始评分；输出、采样单位、模型和任务不同。文中系数经验不能为本案0.1/0.5或六群占比作证明。
- 审查判断：完整H的结构保留不等于保持项有增量价值；Full–ER应比扩充更多协议优先。

## 没有得到文献支持的本案数字

以上来源均未验证本案的5pp/3pp/2pp、MAP .60/.55、96×3×2、三seed足够功效、24累计小时或30轨迹必要性。40epoch有历史使用先例，但不等于四条fit的新任务已验证。它们必须分别归类为用户价值尺度、探索投入规则、资源限制或待验证设计选择。

## 文献如何使用

文献支持“这个问题可能有意义”和“这些机制已有先例”，不能代替新任务的实测前提。报告关于冻结θ1对照、一次结束承诺、Full–ER优先及停止投资的判断，是结合本包负证据作出的审查建议，不冒称任何论文已替本项目得出结论。
