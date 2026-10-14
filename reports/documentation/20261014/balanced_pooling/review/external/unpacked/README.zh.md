# 交付导航

本包为2026-10-14“针对完整S的双边覆盖记录匹配评分”的独立静态外审。结论为值得进入实施合同准备，定位是常规OT任务适配／机制对照，不是已建立新算法或优于S的结果，也不是执行许可。

主正文：`review.zh.md`；`review.zh.txt`与其同字节，方便纯文本阅读。逐项发现见正文第九节；唯一实质文字澄清与原有待落地事项见`minimal_revision.zh.txt`。

范围与来源：`reading_scope.zh.md` / `reading_scope.json`列出每个输入文件实际行范围；`references/primary_sources.zh.md`列明原始论文版本、公式/表格阅读及访问失败。身份与权限边界在正文第二节。

原始证据：`evidence/input/`包含输入ZIP字节核对；`evidence/matrix_check/`包含自写手工矩阵输入、源码、实际子进程捕获器，以及唯一attempt01的环境、执行时间、output、stdout/stderr。程序没有项目依赖、正式数据或模型。

初稿和更正：`drafts/review.v01.zh.md`保持首次保存字节；`evidence/operations/version_events.jsonl`、`changes.zh.md`和patch记录定稿变化。数值源、输入及attempt01均未改动。`operations.zh.md`为操作摘要，不冒充未保存的完整浏览器录屏。

字节清单：`MANIFEST.tsv`列出本ZIP内除清单自身以外的全部相对路径、字节数和SHA256。它自排除，不能自证自身散列。ZIP外的`balanced_pooling_review_delivery_receipt.json`给出ZIP与所有成员（包括MANIFEST.tsv）的散列、CRC/覆盖核对及原输入身份；这是本实例交付回执，不是第三方签名或用户Linux接收证明。

原输入ZIP单独返回链接，不重复解包复制到输出包。没有访问用户服务器、执行项目科研代码、载模、正式数据/Memory读取、训练或委派子agent。
