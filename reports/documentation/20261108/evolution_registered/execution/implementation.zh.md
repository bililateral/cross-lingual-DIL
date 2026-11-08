# Registered六轨迹实现与必要增量证据

2026-11-08。承接[用户许可](authorization.json)和[具体执行范围](scope.zh.md)。本轮已实现双路P六条、冻结扩库评分与一次全批开标；正式数据构建、载模训练、评分尚未执行，等待这次具体实现增量的外审及主审闭合。无需再次申请六条/双路/14GiB的普通执行许可；新包外发仍按具体包许可。

## 实际接线

- `step31_evolution_registered_data.py`从唯一资格缓存固定保留确认身份、导出开发4/2/2，校验全开发记录/正文和严格角色日期；fit、盲query/gallery、私有qrels分别保存。语言样本固定一次，每批29/32门进入真实`dataset()`启动检查。确认没有正文导出或训练入口。
- `step31_evolution_registered_control.py`实际campaign用原Runner双槽派发六条；各worker调用原`execute`，显式E1/E2/E3和12群进入当前fit调用、随机种子、学习率调度和保存。模型使用旧H1/V1，旧数学核心未改。三处共享文件仅显式接收domains、worker_entry及caps参数；旧默认值保持，Linux隔离部署不覆盖旧活动workspace。
- 六条训练完整后，实际`score_one`使用每条四个端点，全部15矩形含冻结3格，共90格；H调用原记录表示与成对head后作原四记录聚合，V逐记录单位化后求视图均值并用原归一化公式。所有旧/新query和gallery在同一端点重编码。
- 实际`analyze`先核六条训练、六份完整评分和90格身份/形状/数值/源对应，再一次读新开发qrels；计算并保存逐query及全部十指标，调用固定P门。输出仅STOP_LEARNING、STOP_NO_JOINT_FORGETTING或PREMISE_MET_AWAIT_USER_MECHANISM_DECISION，所有路径automatic_next_phase=false。
- 原14GiB整PID/13GiB allocator、2核/槽、32GiB RSS和40GiB准入承诺、失败槽退出/健康续领均复用真实Runner。新Ledger与Runner使用同一3h/1h/1h桶，训练任务1800秒帽、评分600秒帽、辅助分析300秒帽；父监测和失败照计。实际CLI在review_closed=false时拒绝，并未创建正式dataset或campaign。

## 原始核验与边界

所有执行均在Linux既有py310，手写输入和随机TinyEncoder，不加载预训练或GPU。累计CPU子账17/180秒，当前辅助549/3600秒＝资格352＋保守准备/传输/收尾180＋本轮17；training/gpu均0。各文件为原始小证据，费用不因失败清零。

| 运行 | 观察 | 费用与覆盖 |
|---|---|---|
| CPU01 | 失败；TinyEncoder只返回四个hidden states，缺少应独立列出的embedding层，旧末四层表示检查正确拒绝 | 外层2.8233秒，计3秒；原console/lifecycle、H失败回执及cpu01_source.py保留。未改正式模型来迁就夹具 |
| CPU02 | 修正夹具后五项PASS | 外层4.4297秒，计5秒；真实H/V各三阶段小更新、当前fit顺序/保存恢复、矩形与旧函数及标量公式一致、日期/重复/确认预留、并列排序与错标签拒绝、实际两CPU子进程失败退槽且健康继续领取 |
| CPU03 | 真实score_one/analyze全流程PASS | 外层7.7364秒，计8秒；实际六任务/90矩形/1824768逻辑分数，冻结同候选子集分数一致，少一份评分不开标，完整一次开标后不接M。训练完成元数据及端点刻意为合成夹具，不冒称完成8640正式更新；实际训练接线由CPU02覆盖 |
| 入口约束检查 | PASS_EXPECTED_REVIEW_GATE_REJECTION | 计1秒；全部14生产源可解析，实际CLI拒绝尚未审查的pending gate，正式目录仍不存在 |

源码直接对照时发现初稿V矩形编码遗漏原逐记录单位化，已在CPU02/03前补齐并经两条独立比较核对。CPU01在更早的夹具检查处停止，不能写成当时V已验证。初稿源码可由保留的deployment.zip及deployment_manifest.json恢复；CPU01失败测试源码另存cpu01_source.py。CPU02/03实际来源、当时final_sources和最后入口增量entry_sources分别保留，不以新哈希冒接旧执行。

最后入口只补CPU子帽准入和数据/语言入口两核亲和性，未变CPU02/03覆盖的数学、数据或评分函数；以实际CLI和解析核对补该差异，不追加GPU/全历史回归。五项/四项PASS只证明手写输入上的对应行为，不能证明正式数据语言、Evolution遗忘或完整H优势，也不能替代正式资源/来源检查与实现外审。

## 现场和保管

15:29:41只读现场见observation.json：中文H仍PAUSED_IDLE，7完成、0活动、56待派；监管1412412的属主/创建时刻/命令/cwd匹配原身份。GPU无计算进程，来源资格缓存已核对大小/SHA。此为时点观察，正式启动前须复查，不动中文暂停状态或未开放效果。

Linux独立目录为`/home/yongpeng/cross-lingual/reports/documentation/20261108/evolution_registered/execution/`，workspace/scripts保留实际来源。38份必要小证据及清单已经SCP回传并逐项核对，CPU02/03的16份当时源码与Windows匹配；最终入口来源另见entry_sources。真实资格缓存仍只在Linux；本轮没有打开96条正式语言正文、确认文本或旧qrels。手写CPU权重/评分不进入审查包；必要原始证据和精确失败来源保留，临时载荷在审查与诊断完成后按既有纪律收尾。
