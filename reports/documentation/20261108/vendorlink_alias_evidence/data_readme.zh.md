# 本轮最终候选证据读取说明

本轮没有生成可训练的真实马甲数据集。以下是用户已批准的“证据不足则保存候选与缺口”交付，不是正式标签、划分或评价候选池，不应被训练／评价入口自动载入。

Linux目录：`/home/yongpeng/cross-lingual/reports/documentation/20261108/vendorlink_alias_evidence/final_evidence/`。原始账号、链接、指纹及声明语境仅留Linux；Windows只保存聚合、来源和说明。

| 文件 | 含义 |
|---|---|
| candidate_relations.jsonl | 1013条最终弱关联组；candidate保留筛查后成员及原弱标签，grams_source保留原Grams行号、公开字段和匹配依据；verified_control=false表示未建立验证，**不表示已验证不同控** |
| accounts.jsonl | 恰好这些组的2275个成员账号及原记录数／不同文本数／观察时间元数据；没有复制商品正文作为训练载荷 |
| claim_candidates.jsonl | 主执行者实际阅读的3个不同名有向声明候选及原来源语境；其来源为audit03，同轮audit04没有改变声明召回；不是穷尽所有声明 |
| manual_relation_decisions.jsonl | 对上述3项的逐条判读：单向未签名声明，保留支持性线索，未升级为已核实同控 |
| header_candidates.jsonl | 44个身份页头部词命中及语境；均为同一通用冒名警示，不是账号控制权验证记录 |
| html_sources.jsonl | 15922个实际三市场身份来源页的原路径、字节和SHA；只用于来源追溯，不是训练数据 |

最终文件身份及机械整理方式见final_evidence_manifest.json。现有筛查候选按group_id连接未改写的原来源记录，账号取这些候选成员的并集；4个判读／来源文件按原字节移动，不改变其历史SHA。报告中的239条公钥边界、41条签名片段等最终计数和限制保留在audit04汇总；可重新生成的全量提取中间件按用户要求删除，原始商品包和SQL／HTML仍在。

数据尚未完成模型文本清洗、标签证据遮蔽、近重复处理、关系可靠性核实、已见用途逐账号映射或训练／开发／测试隔离；以上数量不能当独立确认规模。不同candidate组不是已验证负例，组内也未自动获同控真值。若未来要做来源弱标签实验，须先明确这一主张及新合同；恢复历史脚本不构成执行许可。

一次性脚本最小恢复版本位于本地Git `7eef50b3f0ba0ccb1de7790155c5aecf6ae059d7`，五个精确版本与大小/SHA见source_recovery.json；只在Windows使用Git。两端本轮脚本及中间件现已删除，精确范围见cleanup_windows.json／cleanup_linux.json；六个最终候选文件和原始输入保留。旧H、旧VendorLink及中文实验源码不在本轮清理范围内。
