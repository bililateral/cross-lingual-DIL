# 分阶段精确梯度：实现与实际尝试

2026-10-04，Asia/Shanghai。用户明确要求保持原数学定义，尝试分阶段计算完整梯度。本次已实现、核对小模型并在Linux尝试唯一原生路径；**微型等价性通过，BGE在第一个支持群构建二阶图时OOM，资源准入仍未通过。** 本轮没有新增正式数据/标签/缓存/旧权重访问，没有正式效果结果。

## 实现及直接证据

新增`step28_group_meta_staged.py`按两个支持群、一对query群、两个支持群精确HVP顺序计算`gs+beta*(v-Hs.T@(A*v))`，原`objective/lookahead`继续作直接参考。历史项保持0.1基础监督+0.5 MSE，query保持当前+rho历史，固定率不变，全参数交叉Hessian保留。支持重算复用每角色seed和原FP32/eager/microbatch32前向。每个loss诊断转数值；HVP函数独立作用域释放图；None梯度保留；结束将组装梯度交原共享`apply_gradient`，一次clip、一次AdamW。没有使用offload、checkpoint或一阶近似。

原方法仅提取原外层更新主体，旧GPU原生夹具仅允许显式传入update函数；全群输入与warm/两步安排未变。新增原生入口确实传入staged.update，不是仅记录分阶段名。新部署26来源大小/SHA全部匹配，py310 AST/Bash检查通过；修复前后来源、每次关键代码快照和部署资源均已留存。

## 发生了什么

| 尝试 | 实际时段 | 包装计费秒 | 结果 |
|---|---|---:|---|
| tiny_first | 18:25:00—18:25:05 | 6 | 8项中7通过、1 ERROR；unused旧Adam状态测试夹具触发直接参考的现有计数一致门 |
| tiny_repair | 18:27:22—18:27:30 | 8 | 修复夹具后8项全部通过，0失败/错误/跳过 |
| staged_first | 18:27:59—18:28:10 | 11 | warm完成，首个meta第一个支持群的create_graph=True求导OOM；成功元更新0 |

计费按包装原始seconds.txt，首次起止文字显示5秒但单独计费字段6秒，保留原件及保守6秒口径，不倒改日志。新消费25秒，连同前轮101秒累计126秒/3600秒；算术剩余3474秒，不自动追加尝试。原授权唯一同定义技术修复已消费；未重试原生或增加路径。

首次错误是测试安排不兼容已有`adam_step`约束：完全unused参数的旧计数288保持不变，其他参数更新至289，因此直接参考报计数不一致。将已有unused/zero矩的Adam行为独立放到标量夹具验证，生产完整群继续原计数门；不改算法、容差。新原生开始前8项实际通过，不能把首次失败抹去。

微型证据包含原六项回归以及新增两项：含交叉非线性和完整MSE的手算中央差分、None/零梯度及已有Adam矩跳过；四群含dropout在三种实现检查设置下的全参数梯度、loss、模型/RNG不污染及相同起点连续两步模型/Adam一致性。完整梯度容差rtol2e-5/atol2e-6，模型/Adam rtol1e-5/atol1e-7；不是逐位等价，也不等于原生BGE双反向通过。beta0用于边界测试，放大率用于检出二阶差异，均非效果调参。

## 原生失败位置和限制

原始stdout依次记录support_0_done、support_1_done、query_0_done、query_1_done，随后hvp_0_start。对应已经完成四个完整群的一阶梯度，并释放fast/query图；进入HVP前allocated=6599675392字节。支持一阶阶段结束allocated回到5293389824字节，query一阶阶段结束为7905960960字节。这是阶段间活跃显存观察，不等于各阶段精确峰值。

失败堆栈位于`one_hvp`的第一条`autograd.grad(loss, theta, create_graph=True)`，还未取得该支持群的可微梯度，更未完成HVP或外层Adam。CUDA申请20MiB失败，最后区间peak allocated=28586800640字节（26.62353GiB）、reserved=30064771072字节（28GiB）；RSS峰值2649920KiB（2.52716GiB）。不能说仅需再加20MiB，也不能归因唯一为碎片或证明所有等价实现不可行。

Python已保存OutOfMemoryError result.json；随后stderr另有`terminate called without an active exception`，实际包装退出134，不伪写为干净exit1。18:29:18保存的退出后资源显示GPU空闲32063MiB、利用率0，无本次计算进程，workspace顶层没有转储文件；不据此断言系统全局没有转储。包装计时包括退出清理。

结论：本次按群分阶段确实推进到了四群一阶之后，旧跨群支持图同时存活的失败点已越过；**单群二阶图在当前限制内仍无法完成。** 不继续叠加其他优化，不将资源失败说成效果负结果或机制正确性的反证。原生两步和正式训练仍未准入，无法据本次不完整步估算1728步正式时长。

## 原件

[returned.zip](returned.zip)：83665字节、59成员、SHA-256 `cd98d9dd2ebc7f2c564d35c83e32a710dc97d7235d38e952a8822fb235404241`；58载荷大小/SHA全部匹配。展开原始记录见linux/；初始26来源及原合同快照在initial_deployment.zip，失败源码未覆盖。新增方案见[执行补充](../../../../docs/SELLER_ALIAS_GROUP_META_STAGED.zh.md)。外审状态另见submission.json；此报告不预称外审通过。
