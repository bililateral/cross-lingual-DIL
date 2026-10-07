# 本次函数目标累计关系记忆：完整结果外审上下文

2026-10-07。对象20261006_231800已经完成；请审已完成结果和主执行者解释，不重新签发训练许可。用户当前要求“去linux服务器上同步训练结果回来仔细分析”。用户此前批准单候选完整范围、要求保留累计统计＋小缓存迁移核心、优先旧域保持兼顾整体排序；已授权完整上下文网页讨论与结果外审。用户“不用送审了，可以直接开始训练的话就开始吧”仅免除上次R1/F1修复与原生后的增量准入，不冒充那次实际送审，也不免除本次合同第7节的结果外审。

## 阅读地图

1. `docs/SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md`为当前主分析；本目录`tables.zh.md`在上一级，给八端点22指标、三顺序/阶段/域MAP/AP、训练标量及合并counts。先核对这些主张，不重审所有历史。
2. 研究问题：有限原始历史记忆下新卖家群体学习及旧分布未见卖家的候选排序/基础识别。中文合成主题与表达同时变化，输入为屏蔽标题/描述，无身份特征。每群28账号、378对、20正358负、每query27候选且有正例；A/B/C各48fit、12cal、20valid。不是同卖家跨阶段别名追踪、真实市场或纯表达因果研究。封存test/owners/Audit/private_custody未开放。
3. 本轮冻结源位于`reports/seller_alias_continual/20261007/function_memory_result/source/`，23项；其合同/policy为实际执行身份，当前通用文档不改变它。独立训练入口来自Linux项目根，源码冻结副本、execution/gate/source_inventory严格对应。新增结果核验脚本`script/`实际为包根`scripts/step28_function_memory_result_verify.py`，不在原23项训练源内，不修改训练结果。
4. 同目录`job/`、wrapper/console、authorization、gate、inventory为真实执行结果：23:19:15—02:45:15，exit0，1728新增更新，9逻辑点，28指标/计数集，valid盲门前0/后1。九权重与九Memory本体不上传，只附原大小/SHA；非权重171文件清单包括这些Memory的记录，因此附件不声称是整个服务器的字节副本。分数和矩阵无逐对真值标签；群级混淆计数与指标为已开放结果。
5. 对手路径：冻结JSON的`reference.linux_root`对应包内`reference.local_root`即`reports/seller_alias_continual/20261004/logit_low_result/job/evaluation`。只附本次必需27套矩阵/counts及两份collected，shared三点在reference子目录。collected中其它历史臂不属于本次依赖，不要求补取。另附基线completion/before_valid/access/wrapper及独立主分析既有记录。基线权重此前已依法删除，不重训。
6. `verification/result.json`为Linux保存结果独立核验：单CPU禁GPU，1.26秒，9504统计数字容差1e−12；不加载模型、正文或标签。`syntax_error.txt`是本次新分析脚本第一遍括号语法错误0.02秒的真实日志，解释器未开始读结果；修正后成功。该脚本只分析已保存证据，不是再次执行合同手写CPU准入、正式实验或训练恢复。核验报告首域空序列均值/范数0仅占位，主报告明确不当作首域训练实测。

## 完整历史链与已关闭问题

`history/function_memory_implementation_external_evidence.zip`为原外审完整87成员包，13204125字节，SHA545ec223fcf2b7408568310a19589c8fc203a721ab8a015e48c9d8435f3a589a。内含完整报告、原输入、审查者实际脚本/输出与manifest；原输入含函数目标设计讨论完整证据及其原输入，继续包含原版/失败修改版/基线历史结果和核心取舍。可以按需解开定位，不要求再跑全部旧数学。

直接补附`reports/documentation/20261006/function_memory/report.zh.md`、修后CPU原件、失败简要/删除事实、原生一步结果、native_gate与正式gate。原外审R1 watchdog退出、F1证据类型缺陷均修复；9/9修后CPU、一次BGE native27.58秒、448×256当前与历史最大形状、D-only三探针、97项模式观察通过。没有伪称增量外审发生。函数核心/固定合同未改；数学审查只说明定义与接线，不证明泛化收益或创新。

历史：LOGIT0.1仍为强基线；原硬关系记忆固定点负结果，B＋Q＋.1R修改版更差且已按用户指令删除活动代码/权重（冻结记录保留）；group-meta为资源不可行关闭；risk固定点负结果。不要把历史待授权、等待、旧保管状态当作现行指令。当前本轮权重未获删除许可，继续保留。

## 重点核查与边界

固定总式B_current+.1B_history+.5D，完整baseline头/共享首域/规范FP16/直通/d129/.001。五条件：O MAP区间下端>0、final_all MAP点差>0、A2 MAP/Z MAP/O AP各自区间下端≥−.01。全部0/5，O MAP差−.035606、final_all差−.029906。代价预算为用户前瞻接受的绝对1个百分点，不是已证明无害界；三顺序不是三种子，已开发valid不是独立确认，条件区间不含重训/开发选择不确定性。

请重点判断：首域完全匹配后，A2落后与额外遗忘并存是否支持当前解释；初始D≈0、随后D标量小与所有步clip并存不能推断梯度比例；CAB六群全A但累计N96保留C历史，不能把缓存缺域作为全部失败定因；ABC/CAB第二阶段已落后，发生在第二次累计运输生效于第三阶段之前；识别误报减少与漏检增加、合并precision和群宏precision相反；历史版本恢复只是描述性，不是因果消融；本配置失败不等于整个方法家族无效。

本次允许你只用附件进行必要独立统计和数学检查；禁止要求或读取原始正文、标签、test/owners/Audit、模型/Memory、私有资产；不安装新环境、不运行项目训练/原生入口、不联网获取私有材料。没有新增训练/调参/消融许可。必须区分已证实阻断、当前交付复现缺陷、非阻断后续建议及过度设计，给具体位置与最小处理。提供完整中文原文及实际审查脚本、输出、manifest ZIP；如实写执行范围，不冒充Linux/GPU复跑。
