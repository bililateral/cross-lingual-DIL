# 本条重新上传结果包：独立审查证据

这不是旧PASS的拷贝；所有`*.execution.json`对应本条附件新解压目录上的实际执行。独立主脚本复用并检查审查者既有实现，来源与路径修改见`script_provenance.json`。保存原脚本与原日志，不为使路径美观而事后改写。

先读`review_report.zh.md`和`findings.json`，再核对`submission_verification.json`、`independent_analysis/independent_results.json`、`additional_saved_checks.json`、`report_integrity_results.json`。

`submitted_entry_outputs/`为本次复跑执行者入口的新输出，原提交analysis不变；`independent_analysis/`为不导入项目模块的独立实现输出。`*.stdout.log`、`*.stderr.log`、`*.resource.log`、`*.exit.txt`、`*.execution.json`保留命令、版本与退出状态。交互式元数据显示的一次错误单独说明，不能据此称科学复算失败或声称所有交互调用从未报错。

复核独立结果只需在已有NumPy环境调用：

```text
python independent_result_audit.py --root <ranking_result.zip解压根目录> --out <新输出目录>
```

该入口仅使用保存分数／矩阵／计数／清单及标准库／NumPy，不加载模型、不解析正式标签、不执行训练。`run_logged.py`保留网页实际绝对路径，重现时可直接运行上述带参数入口；不需要复建项目GPU环境。源ZIP未重复打包，权重、正式原始文本和标签不在本证据包。

原附件SHA和全部成员记录见`submission_verification.json`；本包各文件大小及SHA见`evidence_inventory.json`（不包含清单自身，避免递归）。CSV为普通审计数据，不含公式或可执行内容。主要验收只看固定E6 C−A，不从额外比较选择最佳配置。
