本返回包的主裁决是 report.zh.txt 和 findings.json。

阅读建议：先读报告第一、三、九节及 CPU 第七节；再按 evidence_index.tsv / evidence/source_extracts.zh.txt 定位原附件源码。完整范围见 reading_scope.json/tsv。contributions 是辅助原意见，已由主审裁决；若措辞不同，以完整报告和 findings 为准。

机械核对程序只用标准库，对文件大小/SHA、ZIP/tar成员、AST函数文本、保存字段和计数/预算算术操作。它们不导入项目模块、不运行项目测试、不重算指标或bootstrap。mechanical_checks.py 与 supplemental_mechanical_checks.py 顶部配置输入/输出路径；复核者可改为其本地附件提取目录和独立输出目录。程序中39/6/45的控制流计数由静态人工裁决显式列出，不冒称执行跟踪。

原附件未改动。本次审查六次机械失败见 execution/failures.json；三次打包前检查失败的当时程序、必要清单/调试字段保存在 execution/failed_package_attempt01、02、03。项目attempt02失败与一次旧候选包准备失败的必要原件副本在 evidence/original_failures。所有原件副本按字节核对。source snapshots 的全体46文件身份在 mechanical/cpu_attempts_and_snapshots.json 中，本返回包无需再嵌套原2MB输入ZIP。

manifest.json列出除自身和 members.tsv 之外的所有返回载荷。members.tsv列出所有载荷及manifest的字节/SHA，不列自身以避免循环哈希。外层同名 .identity.json 和 .members.tsv 对最终ZIP全部成员再次机械列明。ZIP身份不写入内部manifest，避免自引用。

本次交付仅在当前会话提供下载。没有执行科研、连接项目主机、读受禁数据或向其他目的地上传。当前模型选择/后台路由未被独立验证，报告已区分用户请求、环境自述和未观察事实。
