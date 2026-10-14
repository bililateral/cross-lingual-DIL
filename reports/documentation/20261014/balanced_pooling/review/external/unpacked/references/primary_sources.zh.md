# 本轮一手来源、实际阅读与访问边界

检索与阅读日期：2026-10-14。以下仅覆盖本次直接近邻，不是完备性声明。本文使用的是作者原文或正式论文，不以博客、聚合页面或搜索摘要支持技术结论。

## L1：DeepEMD 原始作者版

Chi Zhang, Yujun Cai, Guosheng Lin, Chunhua Shen. *DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover’s Distance and Structured Classifiers*.

实际可读版本：arXiv:2003.06777v1，2020-03-15，11页。

原文：<https://arxiv.org/pdf/2003.06777v1>

实读：§3.1–3.4（运输、集合分数、端到端求导、质量生成），§3.5相关结构分类器说明；物理第6页表2；方法公式(1)–(3)、(9)–(10)。PDF物理第3–6页进行了页面图像核查。表2明确列有 Full Connections / Average 与 EMD / Equal，因此本提案的“均匀质量集合匹配”不能单独视为新机制。本文不复算其准确率，不将其视觉任务收益移植到卖家账号。

辅助读过不带版本号的作者扩展版HTML：<https://arxiv.org/html/2003.06777>，重点 IV-A/B/C/D 及相应平均／等质量对照。本次没有把未版本化HTML与v1逐字比对，也没有独立确认其每一次修订的出版对应；核心近邻判断以明确版本的原始PDF为准。

未能读取的CVF正式入口（不能据此声称已读正式版全部方法）：

<https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.html>

正式PDF尝试地址：

<https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf>

PDF两次open返回Internal Error；后续正式HTML返回403。原始作者PDF访问成功，未绕过受限页面，也不掩盖访问失败。

## L2：Cuturi，Sinkhorn原始OT论文

Marco Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS 2013，正式论文PDF，9页。

原文：<https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf>

实读：§2–4，运输多面体、KL与独立耦合、熵约束、式(2)、引理2、Algorithm 1及收敛／停止说明。PDF物理第2–6页中的相关定义／算法／图示经截图核查。未重验全文实验及全部附录证明。

用途：核对本提案属于标准熵正则OT与矩阵缩放族，并区分最优计划、运输线性项、含熵最优值。论文式(2)中的输出定义与本提案F不完全相同；不能只复制库函数名称或距离定理。

## 不属于本轮一手重读的材料

输入包中的旧S外审文献表已阅读全文，用于识别历史贡献边界。那11篇论文的旧阅读记录不是本轮对11篇原文的新阅读。本次没有执行论文官方代码或项目代码，没有重算论文数据，没有为判定新颖性执行额外科研实验。

没有查全相关研究，因此不作“首次”“没有任何相同方案”或截至今日完备检索的声明。两个直接一手近邻已经足够支持本轮“普通适配、非新OT理论”的有限结论；并不足以证明未来任何任务化组合都不可能有贡献。
