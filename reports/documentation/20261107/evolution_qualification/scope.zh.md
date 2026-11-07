# Evolution 数据资格盘点

用户于2026-11-07选择核查Evolution是否适合作为完整H的暗网马甲应用场景，并确认本次限定CPU盘点。此前VendorLink的有效负结果及英文ABC前置停止不改。本次不证明真实马甲、遗忘或方法收益。

允许下载Zenodo记录10171217的data-and-readme.zip（332606424字节，MD5 e144a1a0bdb9a8bc6bf831a84e0b501e）。只解析README和market/listings.tsv、vendors.tsv、categories.tsv、scrapes.tsv。检查字段覆盖、重复与编号冲突、账号有效文本容量、各类别/时期覆盖和账号交叉；账号名、PGP只供屏蔽及关系线索计数，不输出具体账号，不把同名或同公钥直接当作同控标签。不读取论坛、反馈或关系网络载荷。

Linux既有py310，独立目录reports/documentation/20261107/evolution_qualification；单CPU worker、8GiB RAM、5GiB磁盘。解析累计进程墙钟最多1800秒；下载/传输累计活动墙钟最多3600秒，失败保留费用，等待用户决定不计计算时间。不载模、不使用GPU、不划分train/dev/test、不训练，不操作当前活动实验。失败或超限留证停止；正常完成仅回传汇总与来源，不回传原始文本、账号或PGP。

第一次Linux直接下载于服务器13:47—13:48发生TLS EOF（curl退出35，耗时5.004320秒，接收0字节、未创建包）。Windows只读HEAD检查遇到DNS错误；本地Playwright浏览器可以访问官方页面。用户另行明确允许一次浏览器中转补救：同一官方包下载到Windows临时目录，经SFTP传到原Linux独立目录，在Linux校验官方MD5后删除Windows临时包；Windows只做文件中转，不解压或运行科研计算。再次失败停止，预算不重置。

交付为实际文件资格汇总、清洗口径/局限、是否值得进一步研究的判断及实耗。数据盘点不授予遗忘预实验或正式训练；具体方法/阶段选择需另立前瞻方案。公开论文及字段依据：https://arxiv.org/html/2311.11878v1 。官方包/许可：https://zenodo.org/records/10171217 （API显示CC-BY-4.0）。
