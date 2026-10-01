# 失败与修订记录

完整解释见REVIEW.zh.md第3、11节。以下均保留原稿和原始stdout/stderr，未用最终成功摘要覆盖。

| 阶段 | 记录 | 失败/修订 | 责任与影响 |
|---|---|---|---|
|本次|logs/00_inventory.*、scripts/00_inventory.py|IndentationError，退出1；01修订为v2后02通过|审查者脚本缩进，未产生不完整核验结论|
|本次|logs/06_supplied_auditor_reproduction.*|CPU24不在当前允许集合，OSError22，退出1|审查者参数错误，不是项目算法错误|
|本次|logs/07_supplied_auditor_correct_cpu.*|实际允许CPU0；未修改Q源码；退出0|只修执行参数|
|本次|logs/08_report_crosschecks_v1.*|退出0且148数值通过；目录映射诊断误报|审查者导航逻辑；第一版全部输出保留|
|本次|logs/09_revise_report_links.*、10_report_crosschecks_v2.*|仅加入清单目录子树映射；最终仍真实缺1个observation链接|不改任何数值或正式源码|
|历史网页|原包HIST/external/logs/independent_v1/v2/v3.*|6/2/2→9/1/0→10/0/0|已关闭历史参考实现修正；本次未重跑|
|历史网页|原包HIST/external/logs/independent_metrics*.|初稿接口形状不对，修正后132手工指标通过|历史证据，不是正式标签重算|
|历史主审|原包HIST/disposition.json、primary_numeric_correction.json|P1数值示例来源错配、P2首项命令时间JSON缺失、GBK解码修复|已披露关闭，不重开、不编造原始记录|
|项目回传|原包A/return_sync.json、connection_status.json|过早解包、完成后SSH访问失败|不是正式训练异常；清理状态未独立验证|

主数值参考scripts/02_independent_results_v1.py和训练证据参考scripts/03_independent_training_evidence_v1.py均首次实际执行通过，没有失败后调整终点、浮点容差或统计公式。
