# 匹配ER0.25的LOGIT合同及最小实现：独立审查证据

先读 [完整中文REVIEW](REVIEW.zh.md)，再按 [命令与原始流索引](COMMANDS.zh.md) 检索。

原收到ZIP：2,084,038字节，304成员，SHA-256 `ff678c2f9687ced6fa9888eca55064ca617c98d35bf371e082e7cf321f9ed08a`。`original/logit_review.zip`保留原字节，`submission/`全部304成员未修改。

网页指定28个不同单测全部通过，两个help、两个bash -n通过；5个最终独立参考通过。完整记录命令15次，其中审查者参考初版2次失败、修订后均通过。没有确认被审项目的科学阻断/待修复可复现性缺陷，但原生BGE、CUDA/BF16与新正式结果未运行、未认证；没有访问项目Linux、正式文本/标签/缓存、正式模型、test/owners，没有安装依赖。

**重要：`evidence/statistics_reference/`中的新LOGIT分数及`path_reference/`中的模型/缓存均是手写微型fixture。新正式LOGIT效果未知，绝不可把这些夹具输出作为科研结果。** 只有明确标出的既有 `quarter_minus_seq` 是对包内45套小型既有数据的观察复算。

文件导航：

- `independent/`：全部审查者独立代码，包括失败v1、最终v2、诊断、报告构建与打包脚本。
- `evidence/01_*`至`15_*`：命令、环境覆盖、实际起止、退出码与原stdout/stderr。
- `evidence/math_reference/`、`diff_reference/`、`statistics_reference/`、`path_reference/`、`access_identity_reference_v2/`：完整数值/案例/夹具证据。
- `evidence/FAILURES_AND_REVISIONS.zh.md`：全部已知失败、修订及早期探索未完整日志化的限制。
- `evidence/HISTORY_READING_INDEX.json`：相关历史全文及处置的阅读范围。
- `evidence/received_members_304.json`：收到原成员逐文件大小/SHA。
- `FILE_MANIFEST.json`、`SHA256SUMS.txt`：除索引自身外全交付文件的大小/SHA；自身摘要和外层ZIP摘要见外部交付回执。

可报告模型名：GPT-6 Astra Pro；没有独立后端标识，不能认证为用户指定GPT 6 Pro，自述不是认证证据。
