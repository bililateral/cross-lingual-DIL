# 本证据包的范围与回读

本包是网页独立审查证据，不是项目Linux或正式研究结果。`REVIEW.zh.md`为完整报告；`source/`是用户本次52来源加清单的原样解压（结束时逐字节复核）；`numbered_sources/`为带行号阅读副本。

`scripts/`含所有实际执行参考、诊断、修订脚本及diff。`logs/`中每个名称都有原stdout、原stderr、exit和包含完整命令/环境/时间的JSON。`outputs/`含本次输出，尤其：

- `submission_verification.json`、`metadata_budget.json`、`final_source_verification.json`；
- `independent_v1/independent_results.json`：6过、2失败、2错误；
- `independent_v2/independent_results.json`：9过、1失败；
- `independent_v3/independent_results.json`：10过、0失败；
- `reference_v1_diagnosis.json`、`reference_v2_diagnosis.json`及两个revision JSON；
- `independent_metrics.json`与`metrics_reference_revision.json`。

第一版主参考Softplus分支/混合dtype数值与两个自身API错误、第二版中间舍入遗漏、附加指标参考首次2D接口错误均保留。没有改项目源码、指标1e-12校验容限或性能零容差。源码成稿前普通拼写编辑不计一次测试，所有实际失败运行均有原记录。

`outputs/`的群标签/文本/参考分数/模型文件都是手工或TinyEncoder夹具。门检查的`.pt`中有刻意构造的非模型字节；`actual_micro`中的`.pt`才是真实微型Torch检查点。它们都不是原生BGE，不能当正式结果或服务器实物。端点公式的随机矩阵也只用于聚合公式核验。

## 复跑限制与路径

原运行路径为`/mnt/data/bge_continual_external_audit`；原命令完整保留。主参考是`independent_checks_v3.py --out <新的空目录>`；提供的22测试命令在`source/`根运行。不安装环境，不执行正式`execute`或原生BGE核验入口。

为逐字重放，可在一份新工作副本里维持原目录结构，务必使用新输出目录；若放置其他路径，审查者脚本顶部的ROOT常量需改为该工作根，另存修改及SHA。不要修改`source/`；不要覆盖本包已有输出后把它们称为新运行。未运行的方便性包装器没有被编造为已验证命令。

官方接口实际阅读位置见`outputs/official_sources.json`。本次没有重审旧正式结果或新开展论文创新搜索。最终文献/接口引用不构成任何项目数据访问。

`artifact_inventory.json`与`SHA256SUMS.txt`涵盖除它们自身外的全部载荷；这是避免自指散列，不是漏项。ZIP整体身份另见外部receipt。所有审核建议需本地主执行者逐项核对；本包不能自动作为正式阶段授权或原生CPU通过回执。
