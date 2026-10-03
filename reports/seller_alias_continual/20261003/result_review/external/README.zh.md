# result_review_evidence.zip：独立结果审查证据

本包对应2026-10-03对ER λ=0.1与匹配LOGIT0.25两项已完成正式开发实验的独立审查。完整中文裁定见根 `REVIEW.zh.md`；其副本也在 `evidence/REVIEW.zh.md`。本包不含未获准的正式文本、正式标签、缓存正文、模型权重或test/owners；原提交包内的已开放小结果原样保留。

## 1. 先读什么

1. 根 `REVIEW.zh.md`：结论、115项表、原生/网页证据边界和失败修订。
2. `evidence/agents/independent_metrics/numerical_v1/summary.json`、`independent_all_115_checks.csv`：五组独立判定23/23、10/23、23/23、16/23、4/23。
3. 同目录 `independent_all_endpoints_22metrics.csv`、`independent_five_comparisons_22metrics.csv`、`independent_pooled_classification.csv`：完整数值而非仅正面摘要。
4. `evidence/root_reference/direct_index_bootstrap.json`：另一个显式R公式/直接抽群/独立分位数参考。
5. 各 `evidence/agents/*/notes.md`：实际源语义、原生证据、分数/校准与覆盖细节。
6. `evidence/execution_index.json`、各 `runs/*/run.json`：精确原执行命令、环境、UTC起止、退出码及原stdout/stderr摘要；原始流和实际执行源码快照在相同目录。

## 2. 输入与证据完整性

`input/result_review.zip`就是收到的 `result_review(1).zip` 的同一字节内容，只使用获准原名保存：

- 8,729,802字节；613成员。
- SHA-256：`87ff5ae41173ea9f68172736a11bd90bf54846fc4a2f0843b12edaa2965b9f7c`。

`EVIDENCE_MANIFEST.json`列出证据载荷成员的大小和SHA；它自身及其摘要文本作为两个明确例外，避免自引用哈希。`EVIDENCE_MANIFEST.sha256`给出清单摘要。外层ZIP的大小、成员数和SHA另见交付消息，不将外层ZIP摘要写回ZIP内部形成递归。

`evidence/input/received_members.json`覆盖原收到的613成员；`archive_integrity.json`保留最初带时间的身份核对，`final_validation.json`保留封包前重查。`combined_frozen_source_coverage.json`核对ER31与LOGIT34的实际内容阅读责任和字节对应，不把同字节复用冒充再读一次。

## 3. 重放环境与可移植入口

本次真正运行使用Python3.12.14、NumPy2.3.5、CPU0，各BLAS/OMP线程1，未安装包或加载Torch。项目原生Python3.10.19/Torch2.9.1+cu130/CUDA13.0证据来自原提交，不能与网页复算混同。

以下是供复查使用的可移植等价命令，**不是要求用户替审查者执行**；本次已执行的精确绝对命令保留在run.json。下面只需要已有Python和NumPy，只读取保存小结果，不会训练或解析正式标签。

在本证据ZIP的解压目录执行，先核原包并展开到相邻input目录：

```bash
python - <<'PY'
from pathlib import Path
import hashlib, zipfile
p=Path('input/result_review.zip')
assert p.stat().st_size==8729802
assert hashlib.sha256(p.read_bytes()).hexdigest()=='87ff5ae41173ea9f68172736a11bd90bf54846fc4a2f0843b12edaa2965b9f7c'
with zipfile.ZipFile(p) as z:
    assert len(z.infolist())==613
    assert z.testzip() is None
    z.extractall('input')
PY
```

独立全量重算：

```bash
review_root="$(pwd)"
python -B evidence/src/run_record.py \
  --label replay_independent_metrics \
  --cwd "$review_root" \
  -- python -B "$review_root/evidence/agents/independent_metrics/independent_metrics.py" \
  --project "$review_root/input/project" \
  --output "$review_root/replay_output/independent_metrics"
```

记录器自动选择当前允许的一个CPU并设置单线程。标签名或输出目录已使用时应改新名称，保持旧运行证据；不要覆盖证据包中已有数值。该独立程序不import项目指标/选择/区间函数。

按根REQUEST提供的项目审计重放（另有独立公式，不把这一步当独立结论）：

```bash
review_root="$(pwd)"
review_cpu="$(python -c 'import os; print(min(os.sched_getaffinity(0)))')"
python -B evidence/src/run_record.py \
  --label replay_project_low \
  --cwd "$review_root/input/project" \
  -- python -B scripts/step28_er_weight_audit.py \
  --study low --project . --source-root . \
  --job reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job \
  --baseline reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job \
  --inventory reports/seller_alias_continual/20261003/er_low_result/return_inventory.json \
  --output "$review_root/replay_output/project_low" --cpu "$review_cpu"

python -B evidence/src/run_record.py \
  --label replay_project_logit \
  --cwd "$review_root/input/project" \
  -- python -B scripts/step28_er_weight_audit.py \
  --study logit --project . \
  --source-root reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace \
  --job reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job \
  --baseline reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job \
  --inventory reports/seller_alias_continual/20261003/logit_weight_result/return_inventory.json \
  --output "$review_root/replay_output/project_logit" --cpu "$review_cpu"
```

其余辅助参考源码保留实际执行时的路径和版本；有的依据本次绝对workspace路径定位输入。若移植这些辅助脚本，只修改明确的project/output路径并另存自己的修订，不宣称修改后的源码就是本次执行原件。需要本次精确源码时查看对应 `source_snapshots/`。核心全量数值程序和两个项目审计入口已支持上面的可移植参数。

## 4. 如何读失败与修订

完整保留六类失败的原运行：CPU参数错误、ER/LOGIT参考误以为有未提供的旧日志、手写参考uint8溢出、取证参考的错误不变量/无依据阈值、报告生成器缺括号。各自修订和覆盖收窄见报告第8节、分工notes、diff和原streams。

这些是审查者工具失败或错误假设，不是正式实验失败；复跑同一例不是新增例数，预期异常注入也不是生产事故。原生CPU手写夹具仅有声明的真实更新，Adam计数设置为288不能冒充真实288步。

交互式文件阅读没有逐条制造额外进程计时回执；全部实质审计、数值参考、失败修订与主要整理程序已保留真实运行记录。没有补造未执行的命令或日志。

## 5. 结论不可越过的边界

正式真值依赖的AP/MAP/Brier等来自原一次收集，本次只从获准保存值独立组合；没有猜标签。校准充分统计量/仿射log_loss核验依赖原保存NLL，只验证条件一致性。原生权重/缓存目标正文和梯度向量未给，不冒称网页再次执行。

ER0.1可按原对ER0.25规则作开发选择；LOGIT主对手仍ER0.25。两者对SEQ分别10/23和16/23，区间跨零不能改判观察均值不下降。单s0、三个固定顺序、60开发群的条件bootstrap不是独立最终验证、三训练种子、参数最优或方法创新。
