# 两项定向静态反例

本文件是源码执行顺序与机械算术的推导记录，不是项目Linux复现回执。审查者未导入项目模块、运行测试、创建交接测试子进程、发信号或操作正式campaign。

## HPAUSE-01：目录已创建但尚无子进程

前置是合法运行状态：旧dispatch已有两个活动worker，一个健康空槽，下一条J仍为pending，资源随后满足。

| 顺序 | 原代码/外部冻结点 | 保存dispatch | 直接children | 文件系统 |
|---|---|---|---|---|
| 1 | 旧循环之前保存RUNNING | 两active，J pending | 两旧worker | J目录不存在 |
| 2 | frozen history dispatch:221完成job.mkdir | 未变 | 未变 | J目录已存在 |
| 3 | Popen:227之前旧父收到SIGSTOP | 未变 | 未变 | J目录存在，可还带console/spec |
| 4 | pause.freeze:114–127检查 | 前后相同 | 与保存active完全相同 | 不检查pending目录 |
| 5 | 两旧worker排空、旧父退役 | J仍pending | 无存活旧worker | J目录仍在 |
| 6 | 显式恢复；新dispatch:239 mkdir | 准备领取J | 尚未启动J | FileExistsError，转STOP_NO_RETRY |

该反例不假设文件损坏、外部篡改、重复开训或模型异常。当前18:50快照有三个活动worker，不证明未来冻结时永远没有一个健康空槽。最小修复是在STOPPED后、提交交接前核对pending任务目录及训练阶段边界，不一致则恢复旧父并拒绝本次交接。

## HPAUSE-02：暂停使辅助后处理的旧预留过时

当前控制器:162–170在进入dispatch前计算generation/analyze时限；dispatch的pause会在真正启动前等待，control.Ledger.pause仍计主动检查。

| 数量 | 秒 |
|---|---:|
| 原辅助cap | 14400 |
| row创建时已用 | 1000 |
| 当时row.seconds=cap-used-120 | 13280 |
| 暂停期间主动检查累计费用 | 121 |
| 恢复时已用 | 1121 |
| 使用旧row的预留合计 | 14401 |
| 恢复时实际余额 | 13279 |
| 按相同原公式重算的任务时限 | 13159 |

旧预留合计超cap而拒绝，实际余额尚充足。纯sleep不计费的合同并不能消除主动费用；不需要等待真实121秒来核验。只应将原本动态分配余额的辅助后处理时限延后到真正派发点，且让预留和deadline共用新值；训练6小时和确认独立桶均不改。

最终attempt03中的测试Ledger.control为空操作，因此其暂停/CPU后处理通过不覆盖这个反例。必要补核可用可控假账本，仍属于CPU接口反例，不是科研计算。
