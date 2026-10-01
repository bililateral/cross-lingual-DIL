# 固定模型的缓存内外诊断

2026-09-15，用户已明确采用[缓解计划](SELLER_ALIAS_MITIGATION_PLAN.zh.md)中的9模型诊断。Linux推理与一次Windows train评价均已完成；结果及解释边界见[固定模型诊断结果](SELLER_ALIAS_REPLAY_PROBE_RESULT.zh.md)。9模型、0训练，train本阶段许可已消费；结果外审及主执行者复核完成，D1子计时文字勘误已闭合；原结果与来源未变，本诊断有效闭环，详见结果文档。

## 实现范围

入口scripts/step28_continual_replay_probe.py，政策schema/step28_continual_replay_probe_policy.json，编排scripts/run_step28_replay_probe_linux_20260915.sh。仅使用原表达数据、原6个shared/SEQ推理权重和本轮3个ER权重。恢复时优化器为None；先检查文件/状态摘要、原point元数据及模型状态，再精确重放全部60群valid原分数，随后评分对应首域60个train群，末尾核对模型状态未变化。三顺序共9×(60+60)=1080群前向，无新训练或checkpoint。

复用未改变的core.load_model/restore_state/score、原公开输入装配及指标实现，不引入新分词、池化、模型结构、阈值或更新。读取历史manifest及保存指标用于映射；不会调用旧run/evaluate或反序列化缓存train标签。模型原始权重均留Linux。

分组基于首域结束的6个初始缓存群：即使第二次装入后被淘汰，仍属于曾重放群；其余54群从未被重放。公开ID顺序明确保存，并与原metadata核对。报告先分别对6/54群求均值，再按三个顺序等权，不按全60群混合加权。保留初始shared绝对值、两方法从shared到最终的变化、ER−SEQ差及两集合差异。后续保留/淘汰和抽样次数只作描述。

Windows先检查9点、来源、输入ID、模型映射、原valid全分数精确一致及0标签/0更新记录，再写access.json、执行一次新的train二值CSV解析。使用UID映射对齐，而非依赖解析器行顺序。完整群采用原22项指标，另加稳定的未截断BCE mean(logaddexp(0,logit)−y*logit)，与原用于概率评价的clipped log_loss明确区分。valid只复用保存的22指标，不解析其标签。test/owners不读。

## 本地核验与局限

8项针对性测试通过、0跳过，实际日志见reports/seller_alias_continual/20260915/replay_probe_implementation/tests.log。覆盖淘汰群仍算曾重放、公开行序、6/54不等组与三顺序等权手算、错误模型/valid重放先于train评分失败、9点完整性/身份/文件变化拒绝、失败完整门不解析标签、一次train-only解析及BCE=log(2)、不同群真值且解析顺序颠倒时UID仍正确配对。两次开发中的夹具错误已修正并记入checks.json：缺少测试valid路径；新增测试时旧断言位置错移。未影响生产结果或旧来源。

实际Windows运行public context核对了9份模型映射、三顺序分组、旧结果完整性和公开输入，无新增监督/模型加载。CLI帮助及两新Python文件AST检查通过。这里没有声称真实GPU推理已经执行；未变推理核心沿用既有实际证据，新9模型原valid精确重放是本次正式入口的必要检查，不另加例行smoke。外审环境实跑与本地验证分开记录。

## 资源、外审与下一步

批准的上限1小时、256MiB新增输出、一张空闲GPU和一个CPU计算线程；完整预计15—25分钟。入口另要求至少12GiB空闲显存与16GiB可用主存，资源不足等待；不停止他人进程或修改共享环境。Bash使用timeout 1h、原py310、离线模型与单线程环境；日志、起止时间、退出码位于独立job，Python输出在job/run。输出盘点包括父job日志；末尾外层回执及评价/外审文件仍须在最终记录中单列，不把盘点或最终占用冒称精确瞬时峰值。当前未做本诊断Linux资源探测/部署/加载。

本轮外审须核对解释边界、分组、模型恢复及分数映射、监督门和评价。批准配置不等于模型效果承诺：缓存内优势不能唯一证明过拟合或随机选择缺陷；缓存自身没有识别提升时，不能用在线BCE下降称保持了马甲关系。没有新成功门、参数搜索或自动方法扩展。

原计划要求实现外审通过后汇报并暂停；用户现已明确“linux直接连就行”，本次阶段已恢复。已完成的随机ER结果外审及17文件977369字节收尾同步另见replay_evaluation/20260915_125920/review_sync_completion.json；该复制不属于本诊断GPU运行。

## 本轮外审及恢复记录

2026-09-15本轮固定模型诊断实现外审及主执行者复核完成：实际网页回复6b3b0cf0-dd28-4e4b-8383-8db5be2e59c3，页面消息model_slug=gpt-6-pro；正文自述Astra Pro的矛盾原样保留，不用正文自述改写网页属性。无确证科研/复现缺陷，16来源未变；本地8项测试通过，外审报告15项通过，补充7项脚本因浏览器下载连接关闭未取得，不声称本地读过或执行过。亲自复核9模型公开映射、6/54实际抽样、原valid保存指标和BCE手算。用户随后明确“linux直接连就行”，恢复本次9模型、0训练诊断阶段，无须再次暂停。14:14现场GPU0 RTX5090/580.178.04，空闲32063MiB、利用率0、无计算进程，资源满足。当前准备同步与推理，尚未模型加载或解析新标签；1小时/256MiB、1GPU/1CPU线程及Windows新train一次、valid/test/owners零解析边界不变。

2026-09-15 14:18:13，部署30份525228字节逐项大小/SHA一致（18复制、12复用），原py310公开context及Bash语法/CLI核对通过。用户恢复后，GPU0后台启动replay_probe_execution/20260915_141813/job，launcher PID11539；0训练，Windows本阶段train许可尚未消费。当前推理中，以progress/exit及回传验证为完成依据。
