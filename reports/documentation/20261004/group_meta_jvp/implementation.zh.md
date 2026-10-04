# JVP完整二阶校正：实现与单次尝试结果

2026-10-04，Asia/Shanghai。用户对明确的一条JVP路径及原剩余3474秒预算回复“那开始尝试吧”。本次已实现、完成微型核验和唯一BGE原生尝试：**十项微型检查通过；BGE在首个支持群JVP变换内部前向OOM，零成功元更新。已停止，没有重试、切换算法或叠加优化。**

## 实现与证据范围

`step28_group_meta_jvp.py`通过真正的`torch.func.jvp(torch.func.grad(loss))`求支持HVP。固定方向仍为全参数Av，参数值和方向detach以不连接外围历史，损失核心保持可微；未调用通过两次反向模拟JVP的`autograd.functional.jvp`。

复用staged一阶支持/查询、临时参数、累加器和共享apply_gradient。staged只增加HVP回调及函数变换所需的标量损失/日志/随机流分离；默认两次反向路径仍保留，并在本次旧八项中回归。JVP生成unused零不能置used=True，参与掩码仍由普通支持/查询一阶梯度取得。每支持重算在变换外fork_rng及设原角色seed，检查只读buffer，保留原分词/微批次/前向/损失验证。JVP预算检查移至群/HVP边界，包装每秒监控，未取消预算保护。

实际新入口：新包装jvp模式→共享staged_check选择JVP模块→原native的update_function→jvp.update→staged.update/gradient→transformed_hvp→jvp.hvp。输出完整梯度再一次clip/Adam。原生堆栈确实经过该链条，不是只添加了未调用函数。旧直接objective/lookahead、模型、损失、手写输入及原Adam主体不变。

## 实际运行

| 尝试 | 时段（+08:00） | 包装计费 | 程序内部耗时 | 退出与结果 |
|---|---|---:|---:|---|
| tiny_first / jvp_tiny | 19:35:46—19:36:00 | 14秒 | 11.02416秒 | 0；10项通过，0失败/错误/跳过 |
| native_first / jvp | 19:37:07—19:37:18 | 11秒 | 9.06202秒 | 1；OutOfMemoryError，0次成功元更新 |

新增25秒，旧126秒，累计**151/3600秒**，算术余量3449秒不自动授权再试。没有技术修复或重跑。实际科研计算已结束，剩余运行时长0；不从不完整步外推正式1728步时长。

部署28来源逐项匹配，既有py310 AST与新包装bash -n通过；19:35部署及19:36原生前各保存资源/CPU47空闲采样，GPU无计算PID。19:38:28退出后GPU空闲32063MiB、利用率0，计算及本项目进程为空；workspace顶层无文件，不作全系统转储保证。

### 微型检查具体证明什么

旧六项加staged两项复用；新增两项直接复用既有标量与四群夹具，替换为JVP实现，并在每次HVP调用前用独立两次反向计算参考，fork_rng恢复后让实际JVP用相同dropout流。逐参数比较每个支持HVP，再执行既有完整梯度、None/零掩码、旧Adam矩、中央差分和逐步同起点两步实际更新核对。三种设置仍为检出用放大率、拟议原生固定点及beta0，不是效果搜索；容差未修改。

HVP比较的None参考转换为数值零仅用于数学张量比较；最终梯度None与零仍由原测试精确区分，独立Adam夹具验证unused旧计数不变。实际JVP返回值要求不带外围requires_grad。

本次没有新的非空buffer变更动态测试；沿用当前只读buffer拒绝策略和已审证据边界。新增完整群Tiny为CUDA、28账号/378关系、70商品/140文本、microbatch4；标量CPU float64、旧六项主要CPU。两步为沿候选路径逐步同起点比较，不是两条独立轨迹累计漂移；一次warm后置288不等于正式首域恢复。通过没有证明BGE完整HVP或全算子forward AD覆盖。

### 原生失败的位置和解释

原生四群保持每群28账号、99商品、198文本、378关系、标题21/描述82 token、microbatch32；FP32/eager、无checkpoint/offload，1e-5/1e-3/rho.1/beta1。warm实际完成；stdout依次记录support_0_done、support_1_done、query_0_done、query_1_done，随后hvp_0_start，没有hvp_0_done或outer_update_start。

traceback进入`torch.func.jvp`→`torch.func.grad`→完整支持损失的函数式BGE前向，在BERT intermediate.dense的`F.linear`处申请42MiB失败。不能沿用旧报告“在create_graph=True求导处失败”的定位：**本次在首个支持群JVP变换内的BERT前向阶段失败，尚未取得该HVP。** 这既不证明所有后续算子都支持forward AD，也不能仅按API名称推断实际资源已减少。

首次meta区间peak allocated=28,331,461,120字节（26.38572931GiB），peak reserved=30,064,771,072字节（28GiB）；进程RSS高水位2,650,404KiB（2.52762222GiB）。这是不完整meta区间，不是完成HVP的峰值，也不与旧路径作成功效率比较。失败42MiB不是完整更新缺口；碎片、张量、切向量、图及工作区各自占用尚未分解。result已保存，本次实际退出1；没有预算监视器终止记录，没有上次的terminate文字。

此固定执行路径在当前预算下未通过资源准入。数学/微型实现支持与原生资源失败分开；没有正式效果结果，创新限制不变。按已承诺边界停止本轮资源探索，保留当前尚无同规格BGE GPU完整元更新成功证据的事实。

## 原始材料

[returned.zip](returned.zip)为61729字节、44成员，SHA-256 `5ee8d874212fb65b019945d05995b9312296f278511a70e49660c7b7676947ca`。43载荷大小/SHA经Windows机械核对全部匹配；原始运行、源码快照、部署/原生前/退出后记录展开于linux/。源码、授权、合同和审查状态各独立保留。本报告不预称外审通过。
