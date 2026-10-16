# 外部一手来源及实际核读范围

本记录是对本次实际网页阅读的人工索引，不是网页原始抓取档案，也不是论文全文阅读证明。访问任务日期为2026-10-16。只读作者arXiv HTML；本次未打开/分析PDF，未复现实验、未下载作者代码。论文中的其他数据集收益不作为本项目预期效果证据。

| 编号 | 作者稿 | 实际方法定位 | 本次使用的有限结论 |
|---|---|---|---|
| W1 | Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives，arXiv:2201.11736，https://arxiv.org/html/2201.11736 | Methods / InfoNCE / Log_out Positives，式(1)–(2)；Log_in式(3)；RINCE式(4)–(5)及命名表。工具文本约120–155行。 | 式(2)逐正例、分母仅该正例加负例，原文求和；本提案按m平均、tau=1。完整RINCE层级/多温度未采用。 |
| W2 | Supervised Contrastive Learning，arXiv:2004.11362，https://arxiv.org/html/2004.11362 | §3.2.2式(2)/(3)，正例外/内平均、同类对齐及困难样本讨论。工具文本约124–147行。 | 原目标的分数形式合法，out/in不能混淆；归一化表示几何不能直接替代本任务MLP参数分析。 |
| W3 | Tuned Contrastive Learning，arXiv:2305.10675，https://arxiv.org/html/2305.10675 | §3.1式(2)–(6)及局限讨论；§3.2式(7)–(15)。工具文本约99–152行。 | 批评分母的理论讨论存在；实际TCL还加入k1/k2和额外项。本提案未采用，也不继承其基准收益。 |
| W4 | Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval，arXiv:2007.12163，https://arxiv.org/html/2007.12163 | §4/§4.1，式(6)/(7)、温度与梯度区间。工具文本约114–132行。 | sigmoid平滑AP是另一路线；S_pos不是该目标或精确AP优化。 |
| W5 | Decoupled Contrastive Learning，arXiv:2110.06848，https://arxiv.org/html/2110.06848 | 摘要、§3 Proposition2、附录0.A.2。工具文本约50、96–127、332–341行。 | DCL移除当前正例分母项；本方案保留当前正例，不能混称。 |

本次没有对S贡献背景内所有其他文献再做全文检索。背景中的近邻与关闭结论按实际读到的报告/主审继承；当前独立核查范围是上述5个直接关系，不宣称排除所有未检索文献。
