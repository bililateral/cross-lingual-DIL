# 外置勘误与说明：冻结输入不改

## 1. 已知非阻断导出语义缺陷

`analysis/summary.json:residuals.<B_S|B_Q|B_C_q|B_C_b>[epoch1..6].violations_total`共24个0，无本轮科学约束定义。它们来自`step28_query_result_audit.py:199`对缺失键的默认求和；原144条teacher_errors不存在`violations`字段。

这些值一律解释为“未定义／不适用”，不得解释为满足约束或零次违规。它们不参与本次运行有效性、R接入、排序、概率、分类或27标签的判定。用户送审已说明该限制；本外置说明完成解释层处置，无需修改冻结summary、重算结果或补建新总门。

## 2. 形成时保管状态

按本条用户补充，当前为已获具体上传／删除授权且五权重已删除。包内原先“待许可／仍保留”保留形成时事实；本审查只独立核对冻结身份记录，没有观察服务器删除，详见evidence/state_addendum.zh.txt。

## 3. 本工作区自身参考输出澄清

`reference/run01/execution_evidence.json:training.B_S.mean_R`最初默认汇总为0.0。S正式日志没有response字段，该值不是实测R=0。澄清后的解释文件为`reference/reporting_clarification/execution_evidence.corrected.json`，取null并标NOT_LOGGED。S/Q共同诊断R仍按描述量保留；这不影响任何统计重算或正式输入。原run01与源脚本保留以供复现，澄清脚本和日志同时提供。

没有发现必须修改正式统计数值、重训或补授权数据读取才能接受本次结论的事项。
