# 逐文件实际阅读与核对范围

本清单区分语义阅读、定向追踪与字节身份核对；不把29份来源的哈希一致写成29份均逐行重新审核。行号对应原输入文件，不对应渲染后的排版行。
对JSON“全文”指完整读取／结构解析，A_F更新表另明确程序逐行检查哪些字段。未读取任何包外正式数据、模型、Memory或盲预测数组。

|包内路径|本轮范围|
|---|---|
|`AGENTS.md`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`docs/RESEARCH_DISCIPLINE.zh.md`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`docs/SELLER_ALIAS_BALANCED_POOLING.zh.md`|定向：§3（36–80）、§4/5中的调用、保存／恢复及统一盲门条款，§7/8/9（173–258）。作为旧合同定位复用，不声明重开全合同审查。|
|`reports/documentation/20261014/balanced_pooling_implementation/qualification/formal_gate.json`|全文JSON解析；29来源逐项比较、CPU/GPU资格引用与scope字段；不跟随包外私有路径。|
|`reports/documentation/20261014/balanced_pooling_implementation/qualification/ledger.json`|全文；CPU旧尝试及累计91秒。|
|`reports/documentation/20261014/balanced_pooling_implementation/qualification/native_disposition.zh.txt`|全文；复用原主审的原生8次、364秒、恢复作用和投影，不声称读取未附native result原件。|
|`reports/documentation/20261014/balanced_pooling_implementation/review/external/unpacked/review/report.zh.md`|定向：31–115数值、接线／权限相关讨论及标题定位；其余以本包主审关闭记录复用，不重审或复刻外审全文。|
|`reports/documentation/20261014/balanced_pooling_implementation/review/primary_disposition.zh.txt`|全文；识别旧缺口已关闭及适用范围，不递归重审全部旧历史。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/authorization.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/cpu_ledger.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt01/handmade_diagnostic.py`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt01/result.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt01/stdout.txt`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt01/wrapper.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt02/result.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt02/stdout.txt`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/attempt02/wrapper.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/receive.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/endpoint_inventory.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job.console.txt`|全文日志及traceback读取；对照阶段完成日志、B_F末条240、最终248失败；不把缺失逐步日志补造为原件。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job.wrapper.txt`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/access.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/execution.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/failure.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/resource.json`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/run/updates/A_F.json`|全文JSON解析；288行逐条校验step及solver_summary原门槛，抽取指定步累计最大值／块数；没有重新计算训练损失或梯度。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/handmade_diagnostic.py`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/receive_verification.json`|全文结构解析；15份回传清单中7份包内原件身份逐项匹配；8份缺席文件仅识别清单事实，35诊断名不当成35份正文。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/report.zh.txt`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`reports/seller_alias_continual/20261015/balanced_pooling_failure/review/request.zh.txt`|全文读取；新失败提案／授权边界／固定手写代码或对应结果、wrapper、stdout、ledger、inventory。代码仅读取，除另行记录的原solve隔离手写核对外未执行项目入口。|
|`schema/step28_alias_ranking_policy.json`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`schema/step28_balanced_pooling_policy.json`|全文结构和值；solver、8次资格与runtime边界。|
|`schema/step28_bge_continual_policy.json`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`schema/step28_chinese_base_policy.json`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`schema/step28_record_replay_policy.json`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/run_step28_balanced_pooling_linux_20261014.sh`|全文1–19；当前86395秒timeout、新起点和退出码。未执行脚本。|
|`scripts/step28_alias_calibration.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_alias_ranking.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_balanced_pooling.py`|全文；求解、异常、Envelope、身份、目标、更新、预测与诊断分派。隔离手写执行仅solve，不执行训练入口。|
|`scripts/step28_balanced_pooling_run.py`|全文；本轮重点35–74、186–338、360–558、699–776，检查来源、计时／投影、实际训练和恢复、盲门、准入。未调用正式入口。|
|`scripts/step28_balanced_pooling_verify.py`|全文；重点90–227、232–327，固定8次结构、实际诊断和两类恢复、CPU选择与累计预算。未执行该资格入口。|
|`scripts/step28_bge_continual.py`|仅定位当前导入／学习率等直接依赖；其余未变来源复用，不声明逐行重审。|
|`scripts/step28_bge_continual_evaluate.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_bge_continual_run.py`|定向：与parse_once、旧入口读账和更新时钟相关的定义定位；不执行、不重新审查整个训练器。|
|`scripts/step28_chinese_base.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_continual_expression_run.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_continual_population.py`|定向：state_digest及182–211 save_state/restore_state；明确optimizer=None仍保留metadata。|
|`scripts/step28_continual_population_data.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_continual_population_evaluate.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_continual_population_run.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_crossfit_retention.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_record_replay.py`|定向：159–219的真实分块反向、optimizer调用顺序及出生原表；234–332的Memory出生／抽取／串行化有关路径。|
|`scripts/step28_record_replay_numerics.py`|仅定位checkpoint_reserve及临时文件生命周期接口；其余按旧处置复用。|
|`scripts/step28_record_replay_run.py`|定向：60–104 Budget起点、snapshot、硬限和磁盘reserve；其他旧编排按已关闭范围复用。|
|`scripts/step28_record_replay_verify.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`scripts/step28_replay_diagnostics.py`|定向：18–43 RNG/neutral，61–129实际capture，尤其97–107的CPU live/teacher A0/A1投影。其余评价模块按未变范围复用。|
|`tests/test_step28_balanced_pooling.py`|定向：111–307的原数值／真实梯度、诊断恢复；372–411的失败与reserve；486–571编排fixture与计数边界。未运行本测试类，更未把mock计数当真实调用。|
|`tests/test_step28_record_replay.py`|仅检查本包大小／SHA及（适用时）29正式来源的身份衔接；未作本轮逐行语义复审。作为未变依赖或背景，不据此新增事实和授权。|
|`review_manifest.json`|全文JSON解析；58项大小、SHA、实际ZIP成员集合逐项机械核对。|

## 包外边界

未附的8份回收原件、原生result原件、正式K/P/Q、模型／Memory和预测数组没有因清单含路径而成为本轮读取材料。未跟随这些路径。
公共文献只核对Altschuler等Algorithm 2及Schmitzer的缩放稳定／收敛讨论；没有重开研究创新检索。
外审附加计算：handcheck.py运行原solve两次固定3×3例及零控制；rounded_reference.py仅对同两个例子的FP64／FP32输入进行闭式计算。均非修复实现、非项目Linux/BGE资格。
