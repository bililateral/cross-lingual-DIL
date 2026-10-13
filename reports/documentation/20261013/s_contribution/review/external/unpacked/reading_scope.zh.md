# 实际阅读／未读范围

本清单独立于旧审查人的阅读范围。全部29成员做字节身份核对；“全文取出”“逐字节散列”和“语义审读”不等同。`audit/reading_log.jsonl` 是部分取出操作日志，不是全部工具记录或逐行理解的证明。早期全文阅读、后续 sed/nl 定向阅读与本清单合并描述；未读部分不会被说成已核查。

项目源码从未作为模块导入或执行；仅审查者自写的文本／ZIP核对和有理数手工例程序运行。

| 成员 | 行数 | 语义状态／实读行 | 说明 |
|---|---:|---|---|
| `context/AGENTS.md` | 24 | 完整语义阅读；1–24 | 本轮纪律与服务器／数据权限；不是重新审核全部工程。 |
| `context/RESEARCH_DISCIPLINE.zh.md` | 179 | 定向语义阅读；1–50 | 后续仅标题／边界定位，不主张全文复审。 |
| `context/SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md` | 45 | 完整语义阅读；1–45 | 历史完整 C/S 合同背景，不重新诊断 C。 |
| `current/report.zh.txt` | 239 | 完整语义阅读；1–239 | 全文阅读，关键 §2、§3.9、§5–8 定向复读；部分全文显示早于 read_source.py 日志。 |
| `current_source/step28_continual_expression_data.py` | 369 | 定向语义阅读；84–136、216–280 | render 与 make_group 的控制者／账号条件及逐记录使用；其余函数未完整读。 |
| `current_source/step28_record_replay.py` | 332 | 完整差异核对；无全文语义行号声明 | 全文件与冻结 core 比较；四个 diff hunk 实读，未变行依据冻结全文，不冒称第二次逐行复读。 |
| `evidence/attribution_correction.zh.txt` | 17 | 完整语义阅读；1–17 | 原主审勘误及其适用范围。 |
| `evidence/attribution_external.zh.txt` | 363 | 定向语义阅读；47–90、129–155、306–338 | 执行资格、结果归因与结论／新颖性边界；其他只标题或关键词定位。 |
| `evidence/attribution_primary.zh.txt` | 89 | 完整语义阅读；1–89 | 闭合独立主审、结果与后续权重删除时点。 |
| `evidence/attribution_summary.json` | 761 | 仅字节身份核对；无全文语义行号声明 | 没有具体矛盾，未解析 JSON、未读取指标数值、未重算统计。 |
| `evidence/crossfit_result_primary.zh.txt` | 78 | 完整语义阅读；1–78 | 仅复用 C_cf 结果边界与处置，不读评分载荷或反向证明 S 最优。 |
| `evidence/record_static_external.zh.txt` | 547 | 定向语义阅读；9–29、88–121、263–294 | 有时间范围的背景结论、集合与重组相关讨论；其余不作本轮复审。 |
| `evidence/record_static_primary.zh.txt` | 116 | 定向语义阅读；1–12、64–116 | 历史定位及 S／自然分布相关条目，不恢复旧 C。 |
| `evidence/record_static_reading_scope.zh.txt` | 1017 | 仅字节身份核对；无全文语义行号声明 | 没有把前人 1017 行阅读清单当成本轮实读。 |
| `frozen_source/docs/SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md` | 97 | 完整语义阅读；1–97 | 归因合同、三臂定义及结果解释边界。 |
| `frozen_source/schema/step28_alias_ranking_policy.json` | 80 | 完整语义阅读；1–80 | 排序项与冻结口径，未启动任何合同测试。 |
| `frozen_source/schema/step28_bge_continual_policy.json` | 199 | 完整语义阅读；1–199 | 冻结训练、模型与记忆配置；不是全项资格复演。 |
| `frozen_source/schema/step28_chinese_base_policy.json` | 247 | 定向语义阅读；1–126 | 输入／模型与训练配置；请求全文输出曾截断，后半未完整审读。 |
| `frozen_source/schema/step28_record_attribution_policy.json` | 22 | 完整语义阅读；1–22 | 真实归因三臂及更新范围。 |
| `frozen_source/schema/step28_record_replay_policy.json` | 30 | 完整语义阅读；1–30 | 底层记录 replay 目标、维度、内存及 chunk。 |
| `frozen_source/scripts/step28_alias_ranking.py` | 222 | 定向语义阅读；89–129 | BCE／查询排序／hard-top5 的直接路径；1–185 曾取出，未把全部范围列为逐行复审。 |
| `frozen_source/scripts/step28_bge_continual.py` | 321 | 定向语义阅读；1–135 | 配置、阶段学习率、optimizer 步数及所需入口；后续未变功能未读全。 |
| `frozen_source/scripts/step28_chinese_base.py` | 880 | 定向语义阅读；89–193 | BGE loader、record_texts、原监督与排序；余下代码未重审。 |
| `frozen_source/scripts/step28_continual_population.py` | 210 | 定向语义阅读；1–40 | 模型、head 与 AdamW；1–130 曾取出但完整语义审读仅声明直接依赖段。 |
| `frozen_source/scripts/step28_continual_population_data.py` | 258 | 定向语义阅读；66–104、198–258 | 数据组、记录结构、Algorithm R、序列化与调度；不加载正式群。 |
| `frozen_source/scripts/step28_record_attribution_run.py` | 630 | 定向语义阅读；1–275 | 加载、真实 update、train_stage、阶段缓存／保存／恢复检查、同初始权重与分支；276–630 未全读。 |
| `frozen_source/scripts/step28_record_replay.py` | 330 | 完整语义阅读；1–330 | 本轮主要实现对象；全部函数，A0/A1、目标、VJP、更新、参考、缓存。 |
| `input_manifest.json` | 174 | 完整结构解析／字节核对；无全文语义行号声明 | 解析全部28载荷成员并逐一核大小与SHA；清单自身不自引用。 |
| `request.zh.txt` | 18 | 完整语义阅读；1–18 | 包内请求；同时使用用户本条完整请求。 |

## 未做及无法独立重演

没有模型、正式记录、训练／评价标签、Memory与F/O载荷，也未取外部项目数据。既有Linux原生资格、参数实际位移及已完成结果来自明确标识的旧证据，不是本轮工作容器的实测。本包并非可脱离未附依赖与模型直接重跑的完整项目归档；本轮没有因而要求补齐整个工程。

`attribution_summary.json` 未作内容回查；没有重新计算三臂指标、bootstrap或任何统计。旧静态全文、旧阅读清单及当前生成器其余功能未递归重审。

论文的版本、实读小节、未读范围与站点失败见 `references.zh.md`、`references.json` 及 `audit/access_issues.json`。没有运行论文官方代码，没有用摘要或论文标题判数学等价。
