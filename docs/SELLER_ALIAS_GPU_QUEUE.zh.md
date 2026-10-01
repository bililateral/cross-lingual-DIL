# 历史GPU队列与当前运行入口

截至2026-10-01，持续学习正式训练与唯一valid已完成：06:06:13退出0、6048更新、实际15小时25分51秒，剩余0（原预计18—26小时，上限36小时）。09:24:59核对两个作业进程已退出。结果已回传及独立核查，实际结果外审和主审现已闭环，详见[结果报告](SELLER_ALIAS_BGE_RESULT.zh.md)。原GPU队列和本轮启动器均不重新执行，当前没有等待启动的已授权新训练。

## 9月24日及之前的运行历史

2026-09-24：本页旧“三模型优先、随后敏感性”队列和后续独立中文BGE作业均已结束。中文训练15:31:07—20:18:48、exit0、3456更新；原启动器PID2902148及训练PID2902159均已退出。运行目录为reports/seller_alias_continual/20260924/chinese_execution/20260924_152642/job。valid评价与回传已完成，见[中文结果](SELLER_ALIAS_CHINESE_RESULT.zh.md)。

中文训练启动前GPU空闲，未新建等待监听，也未恢复旧队列。15:34启动快照确认实际更新；当时配置见[中文合同](SELLER_ALIAS_CHINESE_BASE.zh.md)；当前后续顺序以页首和[科研交接](AI_RESEARCH_HANDOFF.zh.md)为准。不要执行下方旧队列的启动指令。

## 已结束队列的原始记录


2026-09-24当前：队列已按原顺序完成。三模型作业9月23日20:46:27—22:54:41退出0，敏感性22:56:34—9月24日01:34:41退出0；2592／3240更新。队列completion记录22:56:36已确认敏感性入口启动并删除等待脚本，其`sensitivity_training_complete=false`只是当时启动快照；本次以敏感性job退出0和完整manifest确认最终完成。旧PID102561、2871273、2871284检查时已不存在。最新训练、评价与解释见[三模型结果](SELLER_ALIAS_BASE_RESULT.zh.md)、[敏感性结果](SELLER_ALIAS_SENSITIVITY_RESULT.zh.md)，本页下方等待状态均为历史。未新建监听、重启训练或干预其他用户。

2026-09-23 21:32实查（仅代表该时刻快照）：基础模型作业于20:46:26自动启动，已运行45分40秒。LaBSE完成864／864次更新；清单记录纯训练1231.642秒（约20分32秒），第3／6轮推理模型均存在，运行器记录完整模型与Adam恢复和推理模型重载通过，本次仅核对清单与文件元数据，不冒充最终整轮验收。E5-large日志已记录384／864次更新，BGE-M3尚未启动；合计已记录1248／2592次更新，该比例不是耗时进度。队列BASE_TRAINING_RUNNING，原监听PID102561及训练PID2871284存活，GPU0 RTX5090训练进程占用13014MiB。原权重敏感性无launch.json/job，仍待三模型完整成功后执行；三模型全局completion.json尚不存在，valid正式评价尚未执行。访问记录为train_parse_attempts=1，development／heldout／owners均为0。本次只读监控，未改训练／队列、未加载模型或另读标签。7份53094字节小型证据已回传并逐文件大小／SHA核对一致，见[本次状态](../reports/seller_alias_continual/20260923/training_status/20260923_213100/status.json)、[回传核验](../reports/seller_alias_continual/20260923/training_status/20260923_213100/sync.json)和[进展复盘](RESEARCH_PROGRESS_20260923.zh.md)。原[21:10快照](../reports/seller_alias_continual/20260923/training_status/status.json)保留；下方旧“当前／等待／未启动”均按其日期作为历史。

2026-09-19 16:12实查：原PID102561仍正常等待，已运行约28小时；GPU0空闲3999MiB，计算进程仍为其他用户PID41338。最近保存状态16:11:52为WAITING_RESOURCES、review_ready=true、resource_ready=false。基础模型目录不存在，原敏感性无launch.json/job，正式训练仍未启动；旧PID45825不存在。27份批准来源及监听／放行文件与前次记录一致。本次仅状态检查和清理盘点，两端均无新确认可删除文件，未重启或修改队列。证据见[实查](../reports/maintenance/20260919_161100/linux_status.json)和[汇总](../reports/maintenance/20260919_161100/summary.json)。

2026-09-18 15:12再次实查：PID102561仍为原监听、由init接管，状态WAITING_RESOURCES；最近状态15:11:16，外审条件已满足，GPU0仍有其他用户PID41338、仅余3999MiB。基础模型与敏感性运行的launch.json及job均不存在，正式更新0；旧PID45825不存在。本次按用户要求检查并清理两端临时副本，没有重启监听、运行训练／评价或解析标签。基础模型9份与敏感性21份批准来源均匹配，去重后连同监听和审查放行文件共27份在清理前后SHA不变。见[状态实查](../reports/maintenance/20260918_150800/linux_status_after.json)与[清理汇总](../reports/maintenance/20260918_150800/summary.json)。下方13:34状态为先前检查。

2026-09-18用户在正式提交网页GPT6 Pro外审后，要求GPU可用时先运行三模型正式训练，再运行原基线权重敏感性实验。随后再次明确：“新的监听等到外审通过并且你这边修复完成后再开始。”这是本次等待顺序及自动启动的明确授权，取代此前原敏感性优先、保持旧等待进程不变的安排；不改变两项科学实验的配置、标签范围或资源预算。

当前新监听PID102561已在补充GPT6 Pro外审及三真实模型CPU核验闭合后，于13:33:10从SIGSTOP恢复，未新建或重启。13:34:05实查状态继续刷新，GPU仍有PID41338、可用3999MiB，WAITING_RESOURCES；两正式实验未启动，正式标签／更新0。旧PID45825不存在；原脚本备份仅供追溯。先前12:14初始放行后，用户要求实质复核发现池化API兼容缺陷，已最小修复，详见[实质复核](SELLER_ALIAS_BASE_IMPLEMENTATION.zh.md)。

## 固定顺序与边界

1. 收到网页GPT6 Pro正式回复并完成主执行者逐项复核，所有必要修复经适用检查后，保存与最终源码匹配的外审完成记录，再启动一次新监听。若有科研阻断未闭合，监听仍不启动。
2. 新监听另外要求`base_review_ready.json`中实际回复ID、`gpt-6-pro`身份、主执行者复核完成状态及批准来源；文件不存在时绝不启动训练。这是防止误启动的来源／审查条件，不是提前放行。
3. 每60秒只读检查GPU0；无计算进程、可用显存至少24GiB、主存至少16GiB、磁盘至少32GiB，连续两次满足后核对最终基础模型来源，调用`run_step28_base_model_linux_20260918.sh`。三模型共2592更新，完整训练作业仍为8小时／32GiB上限。
4. 监听等待三模型Bash实际退出0，并检查完整2592更新、三模型清单、来源、完成回执及六推理模型核验回执对应。失败或结果不完整则停止，保留证据，不自动重试或启动敏感性实验。
5. 成功后重新检查空闲GPU和主存，磁盘下限用原敏感性24GiB；再次连续两次满足并核对原21份批准来源，才调用原`run_step28_sensitivity_linux_20260916.sh`。两配置、3240更新、5小时／24GiB上限及原训练目录保持。
6. 观察到原敏感性`job/run/startup.json`且启动进程仍活着，监听才自删并退出，原训练独立后台继续；这只证明入口启动，不冒称其CPU检查或正式训练完成。

等待耗时不计入各自训练作业上限。两阶段均只允许一次启动尝试，监听自身亦不自动重启。调度器只读资源、源码与结果清单，不解析任何正式train／valid／test／owners标签，不加载模型，不自动调用评价入口。三模型valid评价与敏感性valid评价仍必须遵守各自完整门和一次性范围，不能用队列调度授权重复评价。

## 实现、检查与实际状态

新调度代码为[scripts/step28_wait_gpu.py](../scripts/step28_wait_gpu.py)，控制流检查为[tests/test_step28_wait_gpu_contracts.py](../tests/test_step28_wait_gpu_contracts.py)。这是原用户授权自动启动的调度修改，不改科研计算来源，不为机械编排单独增加重复网页外审；基础模型科研实现外审仍须完成。主执行者亲自核对了实际阻断条件、顺序、单次启动、失败分支、资源检查和自删时机。

Linux原py310、单CPU线程、禁用GPU计算，10项检查通过0跳过，unittest计时0.021秒。包括未完成外审时GPU空闲也不放行、两次连续满足／中间失败重置、来源不同拒绝、一次性实际无害Bash子进程、三模型完成／失败分支、先后顺序、资源再检查、最终自删及重启拒绝。真实子进程仅在临时目录写入环境值，无模型加载或训练；其余控制流使用手工回执和mock，不当作实际科研运行。

队列记录目录：`reports/seller_alias_continual/20260918/base_queue/20260918_113800`。新基础模型预备目录：`reports/seller_alias_continual/20260918/base_execution/20260918_113800`；原敏感性目录：`reports/seller_alias_continual/20260916/sensitivity_execution/20260916_192101`。原21份批准来源及批准清单SHA-256 `a0acd5e65bed29fcc10b8314b53c69dd2c649bb9112c9949e18b61278bf1dac2`在本次Linux逐项核对一致。

11:37停止旧等待后，新监听始终未启动，直至实际GPT6 Pro最终回复及R1／R2修复全部核验完成。12:14:01新监听首次状态显示review_ready=true、resource_ready=false、连续满足0次。新进程PID102561的用户、命令、父进程和存活状态已核实；base及sensitivity的launch.json均不存在。

检查回执已原样回传：见[Linux部署状态](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/deployment.json)、[十项检查日志](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/tests.log)、[旧等待停止记录](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/replacement.json)和[回传核对](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/return_sync.json)。8份记录与2份当前来源SHA均核对一致；检查完整进程0.12秒，不计为训练耗时。

启动证据：[审查放行记录](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/base_review_ready.json)、[进程与资源实查](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/startup_verification.json)、[回传SHA核对](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/startup_return_sync.json)。六份启动记录两端一致；新监听启动晚于修复完成，不把监听存活称为正式训练开始。

此次暂停和恢复保持同一PID及一次启动标记。原批准来源回执已原字节另存`base_review_ready_before_native_fix.json`，活动`base_review_ready.json`已绑定最终九来源及补充审查回复；Windows保存旧快照不覆盖，当前活动回执原字节回传为`base_review_ready_after_native_fix.json`。没有重置队列或启动尝试。暂停实证见[暂停记录](../reports/seller_alias_continual/20260918/base_implementation/semantic_reaudit_pause.json)，恢复实证见[恢复核验](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/native_resume_verification.json)、[当前来源](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/base_review_ready_after_native_fix.json)及[五文件回传核对](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/native_resume_return_sync.json)。
