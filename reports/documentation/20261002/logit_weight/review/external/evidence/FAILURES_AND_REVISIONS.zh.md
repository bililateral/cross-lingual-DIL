# 失败、修订与证据边界

## 有完整原始流的两次独立参考失败

1. `10_path_v1`：审查者自行编写的 `independent/path_reference_v1.py:145` 多一个右括号，解析时抛出 `SyntaxError: unmatched ')'`，退出1，任何代码均未执行。`path_reference_v2.py` 只移除此括号；差异保留为 `evidence/path_v1_to_v2.diff`。`11_path_v2` 实际退出0。未修改提交项目。
2. `12_access_identity_v1`：审查者在 `access_identity_reference_v1.py:15` 错把来源列表顺序相等当成来源记录相等，断言失败退出1；尚未执行后面的手写更新与访问账本探针。`13_diagnose_source_order` 独立证明两边都是34项、无重复、逐路径完整记录相同，仅列表顺序不同。v2 改为先检查唯一性、再按路径比较完整记录（仍严格比较大小与SHA），`14_access_identity_v2` 实际退出0。完整v1/v2/诊断程序及 `evidence/access_identity_v1_to_v2.diff` 均保留。

这两项是本次审查者参考代码的可复现性缺陷，均已修复，不是被审项目的缺陷，也不作为重开既有问题的理由。未删除、覆盖失败源码或失败原始流。

## 未经统一日志封装的早期探索

最初ZIP解压成功后，一次探索错误地寻找 `submission/source_inventory.json`，产生 `FileNotFoundError`；随后定位到真实的 `reports/documentation/20261002/logit_weight/review/source_inventory.json`。另外，一条组合阅读命令末尾执行 `ls -l /usr/local/bin/python* /usr/bin/python*`，因 `/usr/local/bin/python*` 不存在而退出2；实际Python位于 `/opt/pyvenv/bin/python`，并不影响之后执行。

这两次早期探索未通过 `run_command.py` 记录，不能提供经同一机制保存的原始完整stdout/stderr字节和精确起止时间；本说明是对可见操作的复述，不伪装成当时生成的原日志。Files的两次ZIP内容搜索均无索引结果，随后使用已挂载实物，不将检索失败解读为附件缺失。

全部被列入 `COMMAND_LEDGER.json` 的15次身份核验、指定CPU命令、独立参考及末次完整性核验都有真实原始流、命令参数、工作目录、环境覆盖、起止时间、退出码及流的大小/SHA。最终参考结果全部通过，指定28例无失败、错误或跳过。

## 预期的负向用例不是程序失败

39项盲门错误注入、23项判据逐项破坏、无效target、超1MiB、访问次数重试拒绝以及“18新+45复用先保存后统计故障”均是有意构造的拒绝/恢复测试。每项最终参考程序正常退出0；这些不得计入真实正式训练故障，也不得冒充新增39个unittest。原始微型模型/缓存/标签由手写fixture生成；本次没有正式文本、正式标签、正式模型、test或owners访问。
