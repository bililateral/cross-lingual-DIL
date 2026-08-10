# Step28-v13 v1.13 拆分事务实现外部父锚回执

## 回执范围

本回执只固定第四阶段乙的一世界开发烟雾事务、恢复、封存与清理实现字节。它不授权正式种子、正式能力派生、正式候选生成、正式数据生成、模型训练或指标计算，也不证明正式五百世界、多拆分语义、Linux 目录持久性或第三方环境安全。

回执位于实现提交的后继提交中，且不被机器策略、源码守卫、事务实现或合同测试读取。由此它是四文件内部闭包之外的父锚，而不是闭包成员自证。

## 被固定的父提交

- 提交甲：`dbafb62f91a51b057b5a4846b8028de4076c7c1c`
- 提交甲树：`c7ea90eafe62b0682722e87bf780bf2adab95358`
- 提交主题：`feat: add Step28 v1.13 split transactions`

提交甲中的实现闭包如下：

| 角色 | 规范路径 | 字节数 | SHA-256 |
|---|---|---:|---|
| 机器策略 | `schema/step28_v13_v1_13_split_transaction_policy.json` | 8,972 | `b282f62496e65df214870236df05e294d00f4a1117d7dbbcbe839372a8f7bff0` |
| 源码守卫 | `scripts/step28_v13_v1_13_source_guard.py` | 12,339 | `00b3b24ef5ad099175142e36545fea98acf1e142636385c68159c272fcf42136` |
| 事务实现 | `scripts/step28_v13_v1_13_split_transaction.py` | 97,248 | `58ce5d52e1dfb97090acc9c3f1a3499822050fd99612362742fe8a1b4b049fcf` |
| 非发现式合同测试 | `tests/step28_v13_v1_13_split_transaction_contracts.py` | 56,832 | `63792333f78a3f1692af0a5fb8cd3914a747d630879d652d01e10af30c93641f` |

机器策略的规范自哈希为 `2ea3c38e2f0b1a10feaad50ef751406c7a67f18dfb23d88efdbb80e4d667128e`。提交甲之后，针对上述四条路径执行工作树差异检查为空，工作树文件的字节数和 SHA-256 也逐项等于本表。

## 审查与测试证据

网页端 GPT-5.6 Sol 极高推理经过九轮实际附件攻击审查。最后一轮独立核对最终四文件，给出：

- `Blocker 0 / High 0 / Medium 0 / Low 0`
- `IMPLEMENTATION CODE GO, EXTERNAL ANCHOR PENDING`

该结论之前实际关闭了普通发现入口字节码抢占、发现期字节码冲突、平台相关跳过数硬编码、第三方／项目命名空间抢占和合同直接执行旁路。父锚完成只关闭其结论中的最后一项实现来源问题。

权威合同命令为：

```powershell
python -I -S -B scripts\step28_v13_v1_13_source_guard.py --focused-tests
```

最终结果为 60 项、370.870 秒、全部成功；另有 1 项因本机 Windows 普通用户没有符号链接权限而声明跳过。合同直接脚本执行会在任何导入与测试开始前非零退出。第一至第四阶段甲的 114 项最近一次回归为 244.212 秒全部成功。普通全仓发现按机器合同不包含、也不认证上述 60 项；它运行 602 项、用时 1681.523 秒，其中 594 项通过、7 项既有跳过、1 项固定历史失败且没有运行错误。唯一失败是冻结 Step28-v12.1 清单要求运行开始时 219,186 字节的持续更新进度文档等于 199,490 字节历史快照。

测试退出后，项目 `.pyc` 文件和 Step28-v13 v1.13 烟雾临时目录均为 0。

## 非复用与后继提交约束

承载本回执的提交乙不得修改上表四份提交甲实现字节。提交乙完成后必须验证：

```powershell
git diff dbafb62f91a51b057b5a4846b8028de4076c7c1c..HEAD -- schema/step28_v13_v1_13_split_transaction_policy.json scripts/step28_v13_v1_13_source_guard.py scripts/step28_v13_v1_13_split_transaction.py tests/step28_v13_v1_13_split_transaction_contracts.py
```

输出必须为空。若不为空，本回执立即失效，必须新建父提交和新回执，不得改写本回执冒充原锚。

父锚完成后，正式种子、正式能力、正式候选、正式数据行、质量回执、模型和指标仍全部为 0。下一步只能设计并另行冻结正式多拆分权限、前序拆分封存、正式环境证明与 Linux 目录持久性；不得把本开发烟雾对象直接提升为正式数据。
