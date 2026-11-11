# 六条M接续：增量核验与主审

2026-11-11。承接用户明确“改成4%”及同配置三个seed继续ER/Full的决定，按[补充](scope.zh.md)实施。同一FIT12轮已确认免审范围延续，不新开网页轮次，不冒称本修改经过网页复审。原P判定、来源和费用未覆盖。

## 实际改变及证据

仅三个生产模块变化，Git `c0fadd35`：policy把GB/GC均值要求接为.04，另保存Full−ER条件自身，原效果幅度/保护/3.5pp资格仍单列；control以具名原P来源和旧账进入M-only流程，主循环、worker和CLI均不允许C或重跑P；evaluate仅把无真值应用的V来源改为原P最终模型，Full使用新M来源。新运行身份与旧P隔离，资源计数根仍为全部fit_reference，旧8.41GB输出不从32GiB帽中消失。

`continued_baseline`实际读取/核对旧gate、batch、decision、completion、budget、results，确认原失败状态、六个P配置、共享manifest/模型、额度及累计费用一致。新4pp核查用于本次M准入，不写回P。正式六条仍由原tasks/make_jobs生成，完整训练→完整评分→统一分析；原Runner负责双路、预算、OOM退出/健康槽续领。M完成标为EXPLORATORY_M_COMPLETE，具体原style资格与各效果条件另外报告；不是把运行完整标成效果通过。

本次增加的训练验收只检查既定行为：ER/Full后两个市场的每次更新确有历史消费，历史文本和出生教师计数为正，缺失则拒绝完成。该检查不修改训练运算，也不能单靠数量证明历史项梯度正确；后者使用下列未变代码和原生证据。ER也沿原流程计算/保存出生教师，但其h_objective不启用蒸馏项，Full才以β加入，未把ER改成另一种算法。

## 一次必要CPU核验

Linux既有py310、两个CPU核、手写数据/元数据，12:39:09结束，四组全部通过，0错误/0失败。生命周期2.122029秒，向上计3秒；采样RSS峰值563408896字节。见[cpu01/receipt.json](cpu01/receipt.json)及stderr。没有正式账号/文本、预训练模型、GPU、训练或confirm读取。

1. 实际decide/pipeline验证4pp边界，P小遗忘时仅调用M一次，保存真实分项并保持原P值，缺seed拒绝；不以伪造decide返回值证明接续。
2. 实际旧来源读取及Ledger验证身份、旧失败和费用承接；重置训练账/修改绑定结果均拒绝。原P文件前后字节相同，六条10h完整预约可容纳。
3. 实际worker走到M训练入口，C入口拒绝；此项对模型/训练函数使用停止哨兵，仅证明新的编排接线，不声称执行了真实模型训练。应用路径核对V指向原P、Full指向新M。
4. 实际任务和预约为六个固定seed/方法及10h，无显存硬帽；正式训练完成验收接受完整手写计数，ER/Full历史消费缺失均拒绝。

临时源码8514字节、SHA256 `6b45cceaa2ccde88fca43e589164aac16b8e4ef0edcdfd23b2b00e0e62d33543`，精确恢复入口`c0fadd35:tests/verify_vendorlink_mechanism.py`。核验充分后不扩展测试；待精确恢复/依赖核对完成后两端清理，正式来源13份保留。

## 未变行为的复用和裁决

新旧deployment逐份比较，account_methods/run/io、step29三模块、队列与step30三个模块共十份字节不变；本次evaluate除应用来源路径外的encode/rectangular/score/analyze未变。原[GPU01](../../implementation/gpu01_receipt.json)含实际BERT当前CE和历史项反传、参数更新及恢复；[原实现主审](../../implementation/review/primary_disposition.zh.md)、[RMS增量](../../diagnosis/review/primary_disposition.zh.md)和[FIT12实际评分](../../reference/report.zh.md)在各自覆盖范围继续有效，不重做GPU探针。

独立检查实际execute：ER/Full在stage>1抽取存活Memory并调用update_account；后者把current及history反传到encoder/pair；Full的历史目标是α监督+β出生评分误差，RMS辅助CE只在current路径；teacher在每条记录进入保留记忆时产生，幸存教师不刷新。原P/H_SEQ没有该历史路径。当前样本/优化器/随机流与固定H母体保持。

主审认为新增接续/评价要求已落到实际入口，必要增量证据充分，可进入明确获准M。正式长轨迹的历史计数和影响仍须随本轮实际执行核实，不能用CPU和旧GPU证明尚未发生的Full效果。

原账训练66942/GPU1305/辅助1198秒，另承接P结果汇总1秒、本次CPU3秒及部署/来源核对保守辅助30秒，新gate初始账应为66942/1305/1232秒，CPU子账6秒已含辅助；Runner再按原Ledger另计30秒收尾，不重置原失败/费用。旧gate、batch、结果与账、十份未变源码及共享模型身份在部署/启动前核对，显存/RAM/磁盘满足后单次启动；资源不足由唯一控制器等待，失败不重试。
