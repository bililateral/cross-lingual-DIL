# G_half 外审返回包

先读 `report.zh.md`。本包结论是：当前合同、最小实现和两轮来源绑定 CPU 增量可接受，未发现必须先修复的当前阻断；GPU 六调用和实际投影尚未执行，正式准入尚不成立。**本包不含、也不是 NATIVE／FORMAL 批准门。**

## 内容

`report.zh.md` 是完整中文报告。`findings.json` 是分类意见；`reading_scope.json` 逐项说明原输入 78 成员的字节身份与实际阅读层次；`evidence/traceability.json` 对应合同、源码、证据和未验证范围；`execution_ledger.json` 区分项目提供的证据与网页实际操作。

`checks/` 保存唯一网页数表检查的完整自写源码、AST 原样提取的生产函数、手写输入、原始数值／梯度／独立参考和结果 JSON；`logs/pure_checks.*` 是其真实 stdout／stderr／exit0。没有实例化模型、Memory 或 optimizer，没有执行项目训练／GPU／CPU测试入口。

`tools/`、`evidence/`、`logs/` 还保留输入身份、来源继承、根路径映射、回收／去重核对及其原始输出。首版身份路径联结未匹配的记录保留，不覆盖为第二版；一次审查者元数据脚本 TypeError 的精确源码与真实工具转录也保留。最初 ZIP 交互式查看／解包未另存逐命令日志，后续机械复核有独立保存的脚本与重定向输出；不混称二者。普通静态读文件显示不逐条另造执行日志。

构建元数据时工具响应在成功汇总之后另显示了 `TERM environment variable not set.` 和清屏转义。该构建脚本的重定向 stderr 为空、exit0，输出文件正常；工具尾部另存 `logs/build_metadata.tool_tail.txt`。来源未独立定位，不作为项目错误或数值检查失败。

## 复核边界

原输入 ZIP 由项目保留，本返回包不重复包含它。必要原项目 CPU 日志和冻结源码仍通过报告中的原包路径引用；返回包仅保留本次产生的审查文件和实际数值检查所提取的少量纯函数。

复查原包身份时应使用原始 322949 字节、78 成员、SHA256 `f0f09f50daddbcf948bf9717edcb9b204fc19ea8f8a6d1ecc74b353f09f403e9` 的输入。原 77 载荷的逐项身份已经记录在 `evidence/input_identity.json`。

这些审查脚本使用当时网页容器的明确 `/mnt/data/g_half_input` 和 `/mnt/data/g_half_review` 路径；不是项目命令。需要自行复核时，只应在独立副本调整根路径，保留原输出，不在正式项目根执行，不导入／运行 G_half 的资格或训练入口，也不因此新增项目预算。脚本未附安装命令或新依赖环境方案。

## 清单口径

`MANIFEST.payload.json` 和 `MANIFEST.payload.tsv` 列出生成清单前的全部返回载荷，排除这两个清单自身，避免自散列循环。ZIP 外部的 `g_half_review_members.tsv` 则列出 ZIP 的**每个成员**，包括两份内部清单。`g_half_review_delivery.json` 给 ZIP 大小／SHA、完整成员身份和 CRC／逐项检查结果。ZIP、外部完整成员表、交付回执作为相互配套的传输产物；不要求 ZIP 包含自身散列回执。

除上述传输封装必然置于 ZIP 外部的清单／回执外，所有已经形成的必要审查报告、结构化元数据、计算输入输出、源码和失败记录均在 ZIP 内。
