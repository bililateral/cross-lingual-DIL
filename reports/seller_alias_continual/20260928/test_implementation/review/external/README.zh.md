# 本次固定s0 test实施前外审证据

**所有新生成文本、CSV标签、NumPy分数和3128字节微型Torch权重均为手工核验输入／产物，不是正式heldout。未调用正式execute，未读取原生BGE、正式模型或封存监督。**

先读 `test_implementation_review.zh.md`。实际主结果在 `outputs/independent_v2/independent_results.json`；首次失败必须同时读 `outputs/independent/independent_results.json`、`logs/independent_v1/`、`outputs/threshold_reference_diagnosis.json` 和 `outputs/reference_revision.diff`。

## 运行记录

- `logs/supplied_audit/`：15个原手工用例实际复跑，退出0。
- `logs/independent_v1/`：10项参考中的阈值比较失败，9通过1失败，退出1；没有覆盖。
- `logs/threshold_diagnosis/`：实际重现float32/Python阈值降精度，验证生产路径已float64化，退出0。
- `logs/independent_v2/`：仅修正参考后10项通过，退出0。
- `logs/source_verification/`、`logs/final_source_checks/`：开头与末尾实际SHA、原件字节核查。
- `logs/runner_help/`、`logs/audit_help/`、`logs/bash_syntax/`：只读帮助及Bash语法，均退出0。

每个日志目录保留完整stdout.log、stderr.log、command.json、execution.json和exit_status.txt。expected故障注入分支的异常类型、消息及恢复状态也在对应outputs JSON；它们不算未计划项目失败。

## 源码与目录

`submitted_source/`是当前获准包117个成员的原字节副本，没有从其他附件补齐数据。独立程序用项目函数作被测对象，标量指标／计数／九判据／频数bootstrap独立实现；冻结概率浮点表达式特意一致。

`scripts/independent_test_audit.py`是v1实际源码，`..._v2.py`是v2实际源码，不能用后者覆盖前者。脚本常量B对应本次实际根 `/mnt/data/test_pretest_audit_20260928`。其他环境需在新的隔离目录映射submitted_source为source、保留evidence布局，或仅调整路径并记录，不应调用原正式execute作为测试替代。实际命令日志无需、也没有为显示可移植性重写。

输出目录不覆盖；保存完整两版及四处故障恢复所需产物，所以有重复手工矩阵和抽样文件。没有剔除不利参考结果。旧项目CPU两轮真实日志仍在submitted_source原相对目录中；它们不是网页原生BGE新执行事实。

`evidence_inventory.json`记录除自身与SHA256SUMS外每个文件的大小与SHA；SHA256SUMS另含inventory自身摘要。包外receipt记录ZIP大小、摘要和成员核查，不设循环自摘要。
