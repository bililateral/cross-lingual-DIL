# 本次结果独立外审证据

首先阅读REVIEW.zh.md。此包不包含额外项目标签、模型权重或完整历史文本；input内是用户本次上传ZIP的逐字节副本。没有任何新增训练或项目连接。

主要独立参考为scripts/02_independent_results_v1.py和03_independent_training_evidence_v1.py，均不导入项目模块。05_report_crosschecks_v2.py是最终正文/工作点诊断；v1原稿及输出保留。Q的补充重跑不替代独立参考。

输出最终版本：outputs/report_crosscheck_v2；根目录同名report_*保留第一版诊断，不应误用其已更正的目录链接误报。

实际命令保留绝对路径/mnt/data/bge_input与/mnt/data/bge_review_evidence，与本次运行环境一致。COMMANDS.md及logs/*.json给出精确参数、cwd和依赖环境。它们是执行证据，不是要求用户在本地重跑；复现需先按相同路径解压input原ZIP，防止覆盖已有文件。原始失败也列入COMMANDS，不能把失败命令当推荐操作。

MANIFEST.json给各文件大小/SHA；SHA256SUMS.txt覆盖全部载荷及MANIFEST。清单自身不自哈希。最终打包receipt在ZIP外，记录ZIP大小/SHA、退出和耗时。
