# 当前实验设计入口

本页只索引合同，执行状态与授权补充见[当前交接](AI_RESEARCH_HANDOFF.zh.md)。合同记录该轮形成时的状态，其“待批准”等历史措辞须结合后续明确决定解释；科学公式、访问、预算和判据仍按对应冻结来源。

| 合同 | 研究内容及角色 |
|---|---|
| [BGE首轮持续学习](SELLER_ALIAS_BGE_CONTINUAL.zh.md) | 已完成问题诊断及SEQ／ER1／LOGIT1；后续扩展继承其明确适用设置 |
| [ER0.5／0.25](SELLER_ALIAS_ER_WEIGHT.zh.md) | 已完成有限权重比较，按原规则选0.25；[结果](SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md) |
| [ER0.1单点](SELLER_ALIAS_ER_LOW.zh.md) | 仅新增0.1，主比较对0.25，完整报告对SEQ；[结果与核验](SELLER_ALIAS_ER_LOW_RESULT.zh.md) |
| [匹配LOGIT0.25](SELLER_ALIAS_LOGIT_WEIGHT.zh.md) | 复用SEQ／ER0.25，仅新增匹配历史监督的LOGIT，MSE系数独立为0.5；[结果与核验](SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md) |

新方法、确认性多种子和独立最终留出尚需具体合同。本轮harness重构没有选择新算法、放宽原23项或增加访问／预算。常设边界见[科研纪律](RESEARCH_DISCIPLINE.zh.md)，历史设计全文见[历史入口](AI_RESEARCH_HANDOFF.zh.md#历史与证据)。
