# 实际阅读范围

以下行号来自原文件LF文本。22个ZIP成员及两份单独补充均逐项列出。全文仅表示本次阅读，不代表重做全部历史实现审查。

| 路径 | 总行数 | 实际范围/方式 | 限定 |
|---|---:|---|---|
| `input/context/docs/RESEARCH_DISCIPLINE.zh.md` | 183 | 全文 1–183 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/docs/SELLER_ALIAS_G_HALF.zh.md` | 180 | 全文 1–180 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md` | 209 | 全文 1–209 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/docs/SELLER_ALIAS_H_CONSISTENCY_NATIVE_AMENDMENT.zh.md` | 32 | 全文 1–32 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/documentation/20261013/s_contribution/report.zh.txt` | 239 | 全文 1–239 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/documentation/20261013/s_contribution/review/primary_disposition.zh.txt` | 71 | 全文 1–71 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/documentation/20261016/h_consistency/review/primary_disposition.zh.txt` | 54 | 全文 1–54 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt` | 27 | 全文 1–27 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/seller_alias_continual/20261013/crossfit_retention_result/review/primary_disposition.zh.txt` | 78 | 全文 1–78 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/seller_alias_continual/20261014/query_replay_result/review/primary_disposition.zh.txt` | 59 | 全文 1–59 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/seller_alias_continual/20261016/g_half_result/report.zh.txt` | 118 | 全文 1–118 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/seller_alias_continual/20261016/g_half_result/review/primary_disposition.zh.txt` | 38 | 全文 1–38 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/reports/seller_alias_continual/20261016/h_consistency_result/analysis/audit.json` | 21 | 全文 1–21 | 全文读取已有核算回执，未运行其审计器或重新计算正式统计。 |
| `input/context/reports/seller_alias_continual/20261016/h_consistency_result/report.zh.txt` | 163 | 全文 1–163 | 全文读取合同/纪律或背景报告/主审；其链接的未附原件不算已读取，不全量重审历史。 |
| `input/context/scripts/step28_alias_ranking.py` | 222 | 定向正文 1–184 | 完整读1–184，重点89–129的top5与hard；185–222仅函数定位，不称全读。未执行。 |
| `input/context/scripts/step28_chinese_base.py` | 880 | 定向正文 1–193 | 完整读1–193：输入/配置与原rank、标签结构；其余只做符号或关键词定位，不称正文全读。未执行。 |
| `input/context/scripts/step28_g_half.py` | 137 | 全文 1–137 | 全文静态阅读，未import或执行项目模块；无新S_pos/H_pos实现。 |
| `input/context/scripts/step28_h_consistency.py` | 136 | 全文 1–136 | 全文静态阅读，未import或执行项目模块；无新S_pos/H_pos实现。 |
| `input/context/scripts/step28_record_replay.py` | 332 | 全文 1–332 | 全文静态阅读，未import或执行项目模块；无新S_pos/H_pos实现。 |
| `input/input_manifest.json` | 130 | 全量机械解析 1–130 | 所有21载荷字段逐项验证大小/SHA与精确成员集合，不把清单核验当科学阅读或资格。 |
| `input/proposal.zh.txt` | 94 | 全文 1–94 | 完整原静态提案；当前用户消息仅更新许可/关闭/保管时点，不改公式或双臂建议。 |
| `input/request.zh.txt` | 14 | 全文 1–14 | 完整原静态审查请求。 |
| `originals/review.zh(7).md` | 322 | 全文 1–322 | 当前会话提供的完整补充文本已逐节读；只继承H_cons结果审查边界，不把它当S_pos/H_pos审查。 |
| `originals/primary_disposition.zh.txt` | 62 | 全文 1–62 | 当前会话提供的完整补充文本已逐节读；只继承H_cons结果审查边界，不把它当S_pos/H_pos审查。 |

## 未附或未执行

52份S/H保存对照及baseline_sources.json本体未附；不重新核对远端可用。未附完整旧S/G/F合同及全runner/模型构造/优化器构造/评价依赖、H_cons原生raw receipts、删除回执或远端现场不称已读。未打开模型/Memory、正式文本或标签、owners/private；未运行项目模块、BGE、优化器、资格/训练/正式评价/重采样。

## 外部来源与新手写材料

五份作者HTML仅直接方法/公式段，范围见external_sources.zh.md。未打开PDF或复现论文基准；其他近邻定位继承已读S贡献报告，不声称重新全文核查。新手写数学脚本及输入输出全量产生/核查，见checks/handwritten_results.json；其PASS只限分数坐标代数，不是新实现或原生资格。

## 阅读输出的完整性说明

部分长终端展示曾被工具截短，随后补读了base 165–193、record 1–45、H_cons 47–95及其他有必要的未显示段。没有以截断摘要代替上述声明的正文范围。原ZIP先经Files检索未返回可解析正文，随后只读已挂载原ZIP并解包；不是据空检索推断没有材料。
