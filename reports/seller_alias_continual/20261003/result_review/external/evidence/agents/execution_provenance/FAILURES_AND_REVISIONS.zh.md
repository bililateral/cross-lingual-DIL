# 执行证据独立参考失败与修订

第一次记录运行 `execution_provenance_v1` 实际退出 1：1699 条内部检查中 13 条失败，原 stdout/stderr、起止时间及版本源码均保留。该次数是逐文件/逐字段审计数量，不是 1699 个科研实验。

1. 12 条 `budget_summary_matches_training` 对两个不同时点的整个 memory summary 作相等断言。实际训练日志先记录旧 auxiliary，checkpoint 随后生成新的 stage/RNG/map auxiliary，`train` 再赋给 memory 并保存 `_budget.json`。因此 members、draw_count、draw_stage、seen、with_logits、references、reference_origins 均保持，只有 auxiliary_serialized_bytes / serialized_bytes / sha256 必然会变。逐份诊断显示完整序列化字节差恰等于 auxiliary 字节差；新 auxiliary 字节与对应 point 独立规范序列化相等。v2 改为核对这些因果不变量和字节差，未改任何项目输入、数值或预算。
2. 原生 encoder 历史梯度范数比为 2.50000358787728；v1 自设的 |ratio−2.5|<2e−6 未通过。该断言没有来自合同或原实现的依据：原核验是逐元素梯度差，已保存材料只有 float32 独立归约的 norm/SHA，没有向量。v2 不放宽数值公差以强行通过，而删除这个越界认证断言，把两项 norm ratio 原值仅作诊断。逐元素比例与 LOGIT 梯度相加均明确只能引用原生程序记录，本网页无法由 norm/SHA 重算。

诊断原件为 `runs/execution_provenance_diagnosis/{run.json,stdout.txt,stderr.txt}`。v1 的 13 条失败不是正式训练/测试失败，也不是证实缓存被刷新或历史监督不按比例。失败不能隐藏，v2 的限定核验通过也不能倒写 v1。
