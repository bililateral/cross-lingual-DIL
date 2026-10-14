# C_q四臂A→B独立结果外审返回包

先读`review.zh.md`（完整正文含全部27项诊断与48项概率区间附录；同字节纯文本副本为review.zh.txt）。结论：本固定运行及保存分析可接受，未建立C_q相对完整S的主要排序收益；无需要改正式数值或重训的具体缺口。已知violations_total默认零是非阻断导出语义缺陷，外置说明为不适用。

## 包内结构

`input/original_input.zip`是本次收到的4,808,851字节冻结原包副本；SHA256与用户提交一致。未改变其任何成员，原生失败日志与形成时状态保留。

`reference/`提供独立参考核对源代码、原始输出、环境、stdout/stderr/exit、失败账、保管记录核对与自身报告字段澄清。主参考结果的“PASS”是算术／记录一致性检查，不是科学总验收门。读取原始标签、训练、模型或Memory均为0；未连接项目服务器。

`evidence/`提供357项逐文件阅读层级／行范围、工具请求记录、模型身份边界及用户最新保管状态补充。原始357成员大小／SHA／CRC清单在reference/run01/input_members.tsv和input_identity.json。后续用户删除补充是用户提供的状态，不被冒称为本审查独立现场见证。

`appendices/`分列27项诊断、48项概率对比和四臂分类表；完整精度数据在reference/run01/*.tsv。`findings.json`和`minimal_corrections.zh.md`区分实际发现、分类及最小处置。

注意使用`reference/reporting_clarification/execution_evidence.corrected.json`解释S的正式R字段：缺失R不等于实测R=0。原run01及源脚本为透明可复现而保留。

## 身份清单规则

`return_manifest.json`及`return_manifest.tsv`列出返回ZIP内除这两份清单自身以外的所有载荷大小和SHA256，避免自引用散列。ZIP外的`Cq_AB_external_review_20261014.receipt.json`另列最终ZIP和两份内部清单的大小／SHA，以及可直接阅读的报告。回执自身的身份在最终交付消息列出，不声称任何清单可以包含自身最终散列。

本包不含新模型、正式文本／标签、Memory载荷、封存真值、owners、qrels或凭据。嵌套原包中的资格手写例子与程序源码不是新增正式数据权限。无需运行任何项目脚本即可阅读结论和全部证据。
