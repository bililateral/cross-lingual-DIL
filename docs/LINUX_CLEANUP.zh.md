# Linux 项目清理记录

## 当前清理与保留状态（2026-09-30）

本轮两端清理已完成：Windows58份冗余原文件112,594,084字节，Linux1个中转包18,784字节；详见[存储与恢复](STORAGE.zh.md)和[完成回执](../reports/maintenance/20260930/project_cleanup/completion.json)。持续学习已完成作业的必要共享／对照状态（10月1日更新）、9月27日排序实验18份保留权重、必要正负结果与原始外审证据均保留。

**下述9月10日／14日的12份旧文本模型恢复说明已过时。** 这些训练权重后来纳入9月26日用户授权的全量训练权重删除；当前不能再按早期迁移回执从Linux或Windows恢复。删除前路径、大小和SHA及实际回执保留在[全量清理](../reports/maintenance/20260926/trained_weights/cleanup.json)；不能把当年的归档描述当现存副本证明。新排序权重保留例外与活动状态不受这次旧删除影响。

## 9月10日／14日原始清理过程（历史）

**当前归档位置（2026-09-14更新）：**用户随后要求大型科研载荷优先留Linux。下述12个原模型已重新上传、核对原散列并删除Windows副本，当前从Linux恢复到Windows；详见[存储与恢复](STORAGE.zh.md)。下方9月10日的清理过程与原始回执作为历史保留。

2026-09-10，按用户要求，仅清理 `/home/yongpeng/cross-lingual` 内可确认无须重复驻留的文件。清理未改变研究方案或结果，属于已授权的文件整理，免外审。

删除旧文本实验 `20260907_200309` 的 12 份 Linux 推理模型副本，共 **22,633,965,684 字节（21.08 GiB）**；分配块为 22,634,053,632 字节，观察到文件系统可用空间同量增加。项目 `du` 显示由约 67 GiB 降到约 46 GiB。这是 Linux 磁盘节省量，Windows 归档继续占用原有空间。

删除前 Windows 与 Linux 各对全部十二文件重算 SHA-256，与原 point.json 及彼此一致。Windows 核验 16.266 秒，Linux 核验 16.295 秒；Linux 核验在后台推理结束后以 nice 15、idle I/O 优先级运行，未载入模型。精确路径删除 2.357 秒，未递归删除目录；所有文件归属当前账号且解析后在指定项目根内。

这些文件是有效负结果的必要归档，**不是无效科研证据**。只移除 Linux 重复副本，Windows 保留原字节及原相对路径，旧实验的分数、清单、代码与审查证据均保留。未来若在 Linux 复现旧文本模型，先从 Windows 项目根按下方清单恢复十二文件并核对原 SHA-256；不要重新训练冒充原字节。当前 population 实验的十二模型、当前诊断所用六模型和 `models/` 均未删除。

扫描未发现常见临时后缀文件、旧文件名迁移残留或 Linux 独有脚本。脚本合计约 8 MiB，历史命令入口仍关系复现，未仅按日期旧或无 import 引用删除。此前已从 Windows 移除、仅在 Linux 归档的数据，以及 private/sealed 输入均保留；只核对文件元数据，不读取封存内容。未修改其他用户文件、系统、环境或项目根外路径。

证据：

- [Windows 归档逐文件核验](../reports/linux_cleanup/20260910/windows_archive.json)
- [Linux 逐文件核验](../reports/linux_cleanup/20260910/remote_verification.json)
- [删除路径与恢复位置](../reports/linux_cleanup/20260910/deletion_intent.json)
- [实际删除与释放空间](../reports/linux_cleanup/20260910/deletion.json)
- [删除后核对](../reports/linux_cleanup/20260910/verification.json)

库存清单仅用于清理审计；当前推理输出在扫描后继续产生属于预期变化，不把它们算作清理修改。
