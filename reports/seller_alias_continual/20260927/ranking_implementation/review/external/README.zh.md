# 本轮网页独立核查证据

本包对应 `ranking_review.zip`，SHA256 为 `bad88d2c83feef2d03f7c545519ab134930c9b50dd81c99d6e3b849008cf5203`。

总判定见 `ranking_review_report.zh.md`。37项手工／微型测试通过，正式训练与新监督仍未授权。没有模型权重或正式标签／文本。`independent_results.json` 内的检查点哈希属于已经清理的**微型测试模型**，其嵌入配置是来源元数据，不是正式运行结果。

## 已执行的源码和日志

| 路径 | 用途 |
|---|---|
| run_contracts.py / contracts.* / contract_results.json | 复跑提交的15项本轮测试与12项直接继承测试 |
| independent_tests.py / independent.* / independent_results.json | 本审查独立编写的10项测试及原始结果 |
| verify_package_and_cpu.py / package_cpu.* / package_cpu_crosscheck.json | 不导入项目模块的来源、CPU原始记录及手工损失复算 |
| final_verify.py / final_verify.* / final_source_check.json | 全61成员原件不变、14份脚本语法及测试退出码复核 |
| source_verification.json | 最初60份来源逐文件检查 |
| review_summary.json | 审查范围和结果的结构化汇总 |
| evidence_manifest.json | 本包其他文件的大小与SHA；清单不列自身 |

脚本保持实际执行时源码，没有在测试后为表述美化改写。使用当前既有Python3.13.5、NumPy2.3.5和PyTorch2.10.0+cpu，未安装依赖。

原执行路径：

```text
附件：/mnt/data/ranking_review.zip
解包根：/mnt/data/ranking_review_source
核查输出：/mnt/data/ranking_reviewer_evidence
前一结果包（只读取10份同名继承源码以比对字节）：/mnt/data/pooling_result_review.zip
```

实际命令核心：

```bash
python -u -B /mnt/data/ranking_reviewer_evidence/run_contracts.py
python -u -B /mnt/data/ranking_reviewer_evidence/verify_package_and_cpu.py
python -u -B /mnt/data/ranking_reviewer_evidence/independent_tests.py
python -u -B /mnt/data/ranking_reviewer_evidence/final_verify.py
```

两套测试由 `/usr/bin/time -v -o ...resource.log` 包裹，stdout/stderr分别重定向，命令退出码保存于exit.txt。测试脚本内部固定单CPU亲和性、PyTorch线程1/inter-op线程1；网页使用CPU版PyTorch，脚本不执行GPU。复核者可以直接审读源码和日志，不需要为了接受报告重复运行。

为在其他路径复现，需将脚本顶端 ROOT/OUT 及输入ZIP路径映射到上述相同文件的副本；不要把这些路径改为项目正式数据目录。包核查器会比对前一结果包的十份源码，没有读取此前正式盲分数或指标。

## 范围辨别

T1使用独立NumPy／标量梯度参考；T2/T3/T9使用自建微型编码器；T4使用提交方明确模拟文件夹具，54份分数都不是正式分数；T5用公开手工真值和人为分数做故障恢复；T6/T7/T8纯手工数组；T10显式模拟train_one、原生preflight和GPU名称，是父流程预算／次数测试，不是9次真实模型训练。

各脚本外层禁止正式public_inputs、attach_labels、通用Archive和原生base.load_model；特定手工测试临时替换相应接口，输入始终来自自建夹具。T9实际调用新运行器保存入口，但只有每角色一个手工群，不包含BGE权重或正式144/36/60群规模。

所有保存恢复测试权重都处于自动删除的临时目录。记录中的18权重/54分数“损坏拒绝”仅是模拟故障路径的实测；本轮正式文件还不存在，不能写成实物通过。提交的四次真实BGE手工更新记录经独立核对，不是网页重新执行。
