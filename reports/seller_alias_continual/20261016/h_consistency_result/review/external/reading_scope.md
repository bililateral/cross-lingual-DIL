# 实际阅读范围与证据定位

本文件区分“实际按内容审阅”“程序读取/核算”“仅机械哈希”和“未执行”，不把275载荷哈希匹配声称为275文件全量科研重审。原包从容器只读解包；Files无ZIP可解析正文，未以网页搜索替代文件。

## 1. 主要文档/源码实际审阅

下表给出本次已审阅的必要范围。带原始行号节录在source_excerpts；完整原件仍在input_original.zip。9999表示读取到文件末，表中均已截到实际行数。冻结副本与evidence当前副本逐项机械比对，不重复宣称独立阅读两遍。

| 文件 | 行范围 | 内容/用途 |
|---|---|---|
| `report.zh.txt` | 1–163 | 全文；逐表独立校对、结论/权限/费用/限制审阅 |
| `request.zh.txt` | 1–17 | 全文；当前结果外审范围 |
| `audit.py` | 1–373 | 全文；统计、N/A、分项、诊断及输出；原样执行 |
| `evidence/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md` | 1–209 | 合同目标/轨迹/统计/权限/资格；按当前授权解释历史状态 |
| `evidence/docs/SELLER_ALIAS_H_CONSISTENCY_NATIVE_AMENDMENT.zh.md` | 1–32 | 全文；舍入总目标修订、9调用累计账及免审边界 |
| `evidence/docs/SELLER_ALIAS_G_HALF.zh.md` | 20–96 | 继承H评分及未变比较范围，不重审旧项目 |
| `evidence/scripts/step28_h_consistency.py` | 1–136 | 全文；H/JS、live学生与mixture、目标、共享更新及自然A0评分 |
| `evidence/scripts/step28_h_consistency_run.py` | 1–268 | 全文；两根对照、分项、原生修订/入口与共享编排 |
| `evidence/scripts/step28_record_replay.py` | 159–208, 271–283, 327–332 | 共享VJP/真实更新接口、固定历史抽样、摘要；其余不全量重审 |
| `evidence/scripts/step28_g_half_run.py` | 177–205, 219–365, 368–545, 549–610, 616–710 | 逐步更新与诊断、A/B连续性、两恢复、门及valid顺序、两根来源/统计、执行/资源；只读，未执行 |
| `evidence/scripts/step28_balanced_pooling_run.py` | 595–614, 648–680 | 逐查询定义和群统计/N/A传播；未重审F求解器 |
| `evidence/scripts/step28_replay_diagnostics.py` | 78–208 | capture、中性/残差摘要、九项诊断统计 |
| `evidence/scripts/step28_alias_calibration.py` | 1–90, 139–146 | 正仿射校准和接受条件；未重新拟合 |
| `evidence/scripts/step28_bge_continual.py` | 70–82 | 实际学习率次序和日程定义；补核v1错误定位 |
| `evidence/scripts/step28_continual_population_data.py` | 247–258 | 固定seed键及排列函数 |
| `context/reports/documentation/20261016/h_consistency_implementation/qualification/objective_outcome.zh.txt` | 1–34 | 全文；原生补充实际结果与独立补核，窄范围继承，不冒充raw receipt重验 |
| `context/reports/documentation/20261016/h_consistency_implementation/review/objective_primary.zh.txt` | 1–10 | 全文；用户免审后主执行者独立处置 |
| `context/reports/documentation/20261016/h_consistency_implementation/review/primary_disposition.zh.txt` | 1–32 | 全文；原合并审查、R01关闭及适用范围 |
| `context/reports/seller_alias_continual/20261015/balanced_pooling_result/review/primary_disposition.zh.txt` | 1–51 | 保存对照既有结果处置及损失遗忘方向；不重算F/G |
| `context/reports/seller_alias_continual/20261016/g_half_result/review/primary_disposition.zh.txt` | 1–38 | 旧H结果范围和关闭结论；不载旧模型 |

## 2. 保存证据的结构读取、数组核对与身份核验

审查者补核实际JSON/NumPy读取165个文件；完整逐文件清单在 `reviewer_independent_v3/programmatic_read_scope.json#/json_numpy_read`，没有把“print摘要”当逐元素读取。NPY均allow_pickle=False。275个载荷的哈希/大小及审查前后不变性另在同文件hash_verified及input_identity_final.json。

实际覆盖：两端点×三角色的新结果和S/H四端点×三角色的矩阵/查询/计数；44来源及其冻结副本；52旧保存对照和8个不变直接来源；当前及旧两日志的固定日程；当前2×288分项；partition/manifest/gate/access/completion；12诊断及其数组、9诊断统计；两次校准及保存full/inference评分；接收/分析清单、wrapper、console和analysis_wallclock。formal_start.json和各JSON的必要字段也作为结构证据阅读。

原audit.py另报告229个其直接核验的唯一文件，计数范围与全包275不同，二者不矛盾。来源文件摘要全部核查不意味着所有测试、verify模块及历史实现全部重跑。

## 3. 仅继承/仅哈希、不声称完整实读的部分

context中 `review/external/artifacts/review.zh.md` 保留且哈希绑定；本次没有把原外审全文逐项重审，直接审阅其主处置/R01及新补充的必要闭合链。未变测试文件/资格verify辅助文件/其余公共模块按来源身份继承，不宣称全部手动读完或再次执行。未改动旧S/G/F求解器与旧历史全文不递归审查。

源代码中出现模型、Memory、正式数据或私有路径，只是元数据/程序文本，不会沿这些路径尝试读取载荷。正式来源中的blind_gate只静态阅读，不调用它。

## 4. 明确未执行/无法从本包独立验证

没有标签级重评AP、PR-AUC、ROC-AUC、Recall@FPR、Brier、log-loss及逐查询排名的原子真实性；没有从记录载荷重新形成A1；没有载模恢复/训练或重演每次JS反事实作用；没有加载Memory核实内部原子表；没有重新拟合calibration最优解；没有重新确认远端当前进程/文件custody或操作系统级读访问。

gate所指的CPU receipt、GPU attempt03 result与wallclock原始文件本包缺省，只有其元数据/绑定摘要与关闭报告。原生作用/Adam/恢复数字明确归于窄范围继承，不将此结果审查冒充新原生资格。

只目视检查原analysis/comparison.png。原脚本生成的comparison.pdf保留为输出，未额外渲染或审阅PDF。未访问任何PDF外部网页。没有为新论文定位搜索文献或为显著性增加训练。

## 5. 最终报告证据编号

E01：合同88–106行 + evaluation比较/success + 独立CSV。
E02：baseline_sources、execution/gate/manifest/partition/updates/access及共享run入口。
E03：audit统计和N/A + 独立索引脚本/1650行/读取清单。
E04：三角色matrix/query/count、mapping、full/inference分数及校准源码。
E05：initial/forgetting/old_change及修正后的损失方向。
E06：12capture、overfitting.json和诊断源码。
E07：H_cons目标/共享VJP/原生补充/原主处置和objective_outcome。
E08：wrapper/console/completion budget/policy及原analysis_wallclock。

本文所称“独立”是限定于已经交付的保存证据及重新实现的统计组合；不是独立的原始数据生成、标签评价或第二次实验。
