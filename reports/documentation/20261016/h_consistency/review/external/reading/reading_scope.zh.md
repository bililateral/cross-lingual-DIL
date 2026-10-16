# 逐文件实际阅读范围

行号按原ZIP内UTF-8文本从1计数。全部载荷通过字节/SHA机械校验；机械读全文件不等于全文语义审查。
只读本次包内历史报告，不声称读取其引用的旧大包、正式数组、模型或Memory。源代码只阅读，没有导入或执行。

|编号|原包成员|实际阅读行|范围|用途|
|---|---|---|---|---|
|E01|`proposal.zh.txt`|L1–62|全文|唯一当前机制、公式、反例、比较门、预算与权限。|
|E02|`request.zh.txt`|L1–14|全文|包内审查范围与交付要求。|
|E03|`context/H_result.zh.txt`|L1–118|全文|已开放旧H结果报告；只复用，不重算底层正式数组。|
|E04|`context/H_result_external.zh.md`|L1–301|全文|旧H结果外审原文及历史边界；不冒充本轮重新执行其脚本。|
|E05|`context/H_result_primary.zh.txt`|L1–38|全文|旧外审主审处置、目标false、最新权重删除和Memory保管状态。|
|E06|`context/H_contract.zh.md`|L1–180|全文|原H完整计算、执行、配对、恢复、评价与权限合同。|
|E07|`context/H_static_primary.zh.txt`|L1–43|全文|原H静态主审及固定混合的贡献/风险界限。|
|E08|`context/Cq_contract.zh.md`|L1–249|全文|自然查询、来源排除、Rq和历史系数等与本候选的直接差别。|
|E09|`context/research_discipline.zh.md`|L1–181|全文|当前科研权限、证据继承与禁止越权边界。|
|E10|`source/step28_g_half.py`|L1–133|全文|H块导数、A0/A1监督与D0、回调、pooling身份、评分和更新入口。|
|E11|`source/step28_g_half_run.py`|L1–673|全文|日程、旧S/G来源门、日志重建、恢复、valid门、统计、资源投影及旧资格谓词。|
|E12|`source/step28_record_replay.py`|L1–332|全文|分配与完整K、旧C/S区别、proxy/leaf显式VJP、一次更新、Memory机制静态阅读。|
|E13|`source/step28_alias_ranking.py`|L1–222|全文|实际rank/hard来源、各视图自身top5、查询映射。|
|E14|`source/step28_chinese_base.py`|L1–193、L260–430|定向|L1–193：模型/池化上下文及实际rank公式；L260–430：必要执行/状态上下文。其余未作全文审查。|
|E15|`source/step28_g_half_policy.json`|L1–78|全文|旧H固定阶段、权重、数值、预算与判据，作为未来增量边界。|
|E16|`manifest.json`|L1–95|全文清单/机械核对|15项载荷源路径、大小/SHA及静态范围。|

完整字节数、SHA与原项目路径见`reading_scope.json`和`input_sources.tsv`。
P01/P02仅定向阅读方法正文和相应PDF页面截图，详见`../evidence/public_sources.json`。
M01为`../checks/results.json`、实际输入、源码与梯度数组；M02为`../checks/attempt01/`原始执行记录。
实际已执行的数学/文件检查与未验证边界详见正文第九、十节。
