# 本次校准正式结果审查证据

review_report.zh.md 为完整报告。本包是本次网页实际工作，不是上轮332成员实现证据的重新命名。

- scripts/independent_saved_audit.py：不导入项目模块，仅保存数值审查；无需模型、标签、SciPy拟合。
- logs/*/：每次实际命令、单CPU线程设置、版本、stdout、stderr和退出码。即使stderr为空也保存。
- outputs/independent_v1/：首次独立数值运行PASS；所有输出为保存结果的派生统计，不是新增真值或训练。
- outputs/supplied_replay/：提交入口在新目录的实际输出；与本审查独立实现分开。
- outputs/report_edge_checks/：显示表格及s1 A映射阈值舍入检查。93个新增预测正例不是93个已知误报。
- history/submitted/：项目原始首次失败、修正与有效日志的字节副本。逐文件SHA可与用户原包核对。
- history/error_and_revision_history.json：区分项目两次失败、审查者最初清单位置探查，以及前阶段已有参考修订。

初次清单探查在统一捕获器创建之前出错；其原Python体已保留并原样重捕获。该捕获日志明确是重现，不冒充最早进程日志。独立数值脚本本次未修订或放宽容差。

代码需要用户本次结果包的解压项目树，不需要未上传正式文本、CSV、owners或模型。包内不包含这些资产。只读JSON中保存的计数不允许被用于反推出个体真值。

SHA256SUMS.json列出除清单自身以外的全部成员；外层ZIP的摘要在独立receipt中。报告中的现存Linux权重边界不构成网页重新检查服务器的声明。
