# 一手文献、实际阅读与未读范围

11篇独立论文；DER、DER++分别核查，但不虚增论文数。以下是外审正文书目的完整副本。仅作有范围的方法比较，不宣称截至2026-10-13的完备文献检索。

[L1] Thomas Gärtner, Peter A. Flach, Adam Kowalczyk, Alexander J. Smola. **Multi-Instance Kernels**（ICML 2002）。作者 PDF：<https://alex.smola.org/papers/2002/GaeFlaSmoKow02.pdf>。实读 §3.1–3.3、式(1)–(5)，§4 部分；PDF 第3页截图确认公式及归一化。未重审全部核理论及实验。

[L2] Manzil Zaheer et al. **Deep Sets**（2017）。<https://arxiv.org/html/1703.06114>。实读 §2.2–2.3 的表示定理条件及 distribution kernels；未完整审读附录证明。用于区分一般集合表示与 S 的特定双均值。

[L3] Pietro Buzzega et al. **Dark Experience for General Continual Learning: a Strong, Simple Baseline**（2020）。<https://arxiv.org/html/2004.07211>。实读 §3、式(5)(6)、算法1/2、§3.1。比较缓存、来源时点、监督与输出保持，不复算实验。

[L4] Yanyan Huang et al. **ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis**。作者版 arXiv:2308.13324v1：<https://arxiv.org/html/2308.13324>。实读 §3、§4.1–4.2，§4.3 仅定位及相关片段，不算完整审读；重点 BuRo 的保存和重组。CVF ICCV 2023 正式 PDF 打开失败，未核出版版逐字差异。

[L5] Jiawei Yang et al. **ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification**。arXiv:2207.01805v1：<https://arxiv.org/html/2207.01805>。实读 §2.1–2.3 的原型、同类匹配及四种操作；不复核全量性能／成本。

[L6] Pei Liu et al. **Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole Slide Image Classification**。arXiv:2306.16180v3：<https://arxiv.org/html/2306.16180>。实读 §III-A2、III-B1–3、算法1、混合及目标公式；不把表型边际近似保持解读成联合分布保证。

[L7] Xianrui Li et al. **Advancing Multiple Instance Learning with Continual Learning for Whole Slide Imaging**（CVPR 2025）。作者版 arXiv:2505.10649v1：<https://arxiv.org/html/2505.10649v1>。实读 §3 问题定义相关段、§4.1–4.2 及框架片段。CVF 正式 PDF 和页面打开失败；方法判断仅依据作者版，不虚称逐字核对出版版。

[L8] Rafael A. Rivera-Soto et al. **Learning Universal Authorship Representations**（EMNLP 2021）。ACL 正式 PDF：<https://aclanthology.org/2021.emnlp-main.70.pdf>。实读摘要／引言、§2、§3，PDF 第2页截图查看模型；未复算结果。集合对比、文档抽样和未见作者划分是本轮使用点。

[L9] Xiao Hui Tai, Kyle Soska, Nicolas Christin. **Adversarial Matching of Dark Net Market Vendor Accounts**（KDD 2019）。作者 PDF：<https://www.contrib.andrew.cmu.edu/~nicolasc/publications/Tai-KDD19.pdf>。实读 §5 开头、5.1、5.2，定位至 5.3 开头；PDF 第5页截图复核 5.1／5.2；只作公开特征卖家匹配的应用／模型路线比较，不讨论或恢复群聚类选题。

[L10] Bo Zhang et al. **Contrastive Cross-Bag Augmentation for Multiple Instance Learning-based Whole Slide Image Classification**。arXiv:2508.03081v1（2025-08-05）：<https://arxiv.org/html/2508.03081>。本轮新增；实读 §1、§3.1–3.5（方法正文、式(1)–(9)，不审插入其间的性能表）、§4.2 保留原袋的解释段。未独立核实出版状态、实验及官方代码。

[L11] Ari S. Benjamin, David Rolnick, Konrad P. Kording. **Measuring and regularizing networks in function space**。arXiv:1805.08289v3：<https://arxiv.org/html/1805.08289>。本轮新增方法级核查；实读 §3.1–3.1.1 的工作记忆与原任务末输出，不据摘要猜缓存细节；未比较其全部优化算法。

## 检索筛选但未作方法级判断

CoMEL、AGLR-CL、Replay-free Multi-patches、FEAT和部分2024—2026作者验证线索只在检索标题／摘要层筛选，未据其标题认定与S等价或不等价。未读代码和未访问论文不能支持“没有其他先例”。站点访问问题另列，不隐藏在“已读”条目里。
