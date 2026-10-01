# 本次独立外审材料说明

这是2026-09-28 calibration_review.zip的新合同／实现审查材料，不是正式校准结果。

原附件SHA-256：77ab3765d0928a231fadae7cbb9d92049e0343c4a7518ea5478a0b2909bcf51d。

先读包根的calibration_review_report.zh.md，再读review_findings.json和independent_outputs/independent_results.json。

## 执行记录

每个*_run或independent_run_v*目录保留command.json、execution.json、原始stdout/stderr、exit_status.txt、GNU time。v1／v2的exit=1和失败输出没有删除；最终v3 exit=0。两次失败属于审查者极端概率数值参考语义的修订，项目文件、指标定义及1e-12指标容限未变。

所有numpy数组产物均为审查者手工流程输出或测试用参考矩阵。模型ID形如s0_d也只是手工夹具的接口标识，不能当成正式模型。文件中的任何手工acceptance不是正式17项结果。

本次没解码原附件中的正式分数／指标数组；文件级SHA检查不等于数值解析。没有正式标签、文本、模型或GPU操作。

## 回读与重现布局

脚本按下列关系定位：

```
工作区/
  calibration_review.zip
  calibration_independent_review_20260928/
    submission/             # 用户本次原附件解压成员
    evidence/               # 本包evidence目录
```

审查者命令原样保存在日志，项目已有环境没有安装／升级。复跑时应使用另一新工作区，将原附件解压到submission，只复制最终独立脚本、guarded_supplied_audit.py、verify_submission.py、run_capture.py及provenance_metadata_checks.py到新的evidence；输出目录须不存在，不能覆盖已保留证据。

示例（对应当时环境中已有python）：

```
ROOT="$(pwd)/calibration_independent_review_20260928"
python "$ROOT/evidence/run_capture.py" supplied_audit_run python -B "$ROOT/evidence/guarded_supplied_audit.py"
python "$ROOT/evidence/run_capture.py" independent_run_v3 python -B "$ROOT/evidence/independent_calibration_checks.py"
```

以上从工作区父目录执行，使用绝对脚本路径；具体原始命令见command.json。未请求主执行者现在重跑；这里仅解释回读／复核所需布局。原项目正式execute入口没有运行。

## 文件清单

包根SHA256SUMS.json包含报告及所有evidence成员的相对路径、字节和SHA-256（不含清单自身）。完整性检查只核验产物，不重新触碰正式标签或模型。
