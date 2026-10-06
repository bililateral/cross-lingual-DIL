# 本轮增量一手来源查证记录

日期：2026-10-06。范围：新增B/R与保留Q的直接科研定位；未重做全部历史文献审查。以下为实际打开／定位的一手论文，未用二手概述替代公式核对。

| 文献 | 官方全文 | 本轮定位 | 核对用途 |
|---|---|---|---|
| Burges等，Learning to Rank using Gradient Descent，ICML2005 | https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf | §3式(1)–(3)，正文p3 | R单项的确定偏好标签特例 |
| Buzzega等，Dark Experience for General Continual Learning: a Strong, Simple Baseline，NeurIPS2020 | https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf | §3式(6)，p3；算法2，p4 | 当前监督、函数输出保持与缓存真标签监督的组合先例 |
| Pan等，Continual Deep Learning by Functional Regularisation of Memorable Past，NeurIPS2020 | https://proceedings.neurips.cc/paper/2020/file/2f3bbb9730639e9ea48f309d9a79ff01-Paper.pdf | §3.3式(8)，p6 | 相关函数二次约束的近邻；不是§3.2 |

主审已自行打开上表三份原文并定位相关公式，移机核对使用上表永久URL。抓取失败或无效的扩展结果没有用于结论。

本文件只保存来源位置和用途，未打包长篇抓取文本。主报告§8给出本候选的重合与非等同判断；数学反例和数值结果来自本次独立人工例，不归属这些论文，也不是其效果复现。
