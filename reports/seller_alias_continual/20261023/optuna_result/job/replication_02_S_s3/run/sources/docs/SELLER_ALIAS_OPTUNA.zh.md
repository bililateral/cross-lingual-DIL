# 三臂有限 Optuna 实施合同

2026-10-18。科学范围以[冻结方案](../reports/documentation/20261018/optuna_plan/proposal.zh.txt)、[预算补充](../reports/documentation/20261018/optuna_plan/budget_amendment.zh.txt)和[实施规格](../reports/documentation/20261018/optuna_plan/implementation_scope.zh.txt)为准，后两者解释原方案的执行细节。本文件只导航实际接线，不复制已有全部条款。

用户已批准[最小实现及手写 CPU 核验](../reports/documentation/20261018/optuna_implementation/authorization.json)：Linux 既有 py310，累计 300 秒、1线程、RSS 2 GiB、证据 64 MiB，包含导入和失败。禁止本阶段读取正式文本/标签、载入 BGE/旧 Memory、使用 GPU、正式搜索或复验。新代码不授予运行权限；具体实现外审包尚需自己的外发许可。

实际方法与冻结 trial 身份在 [step28_optuna.py](../scripts/step28_optuna.py)。完整 S 使用原 rank/FP32 均值；S_pos 使用逐正例 rank；H_pos 使用相同新 rank 和既有固定 H pooling。lambda 进入 current/history 两视图；LR multiplier 每个实际 optimizer 调用重新设置，不被旧常数覆盖。初始化、日程/dropout/regroup、Memory 根分别按实施规格，划分及诊断根不随训练 seed 改动。每 run 独立模型、Adam 和带身份的 Memory。正式恢复拒绝其他 run/参数。

[单 run 入口](../scripts/step28_optuna_run.py)复用既有 576 更新/864 群呈现、2端点/两种恢复、12捕获的实际训练生命周期。旧公共 train_stage 增量只将配置中明确的 schedule seed 传入 schedule；既有默认不变。初始门独立构造相应 seed 的 head，检查实际 head 与空 Adam 矩；s0 仍比较旧初始模型，默认 LR 也比较完整初始 Adam 身份。A 训练仍无缓存，A 端点才构建本 run Memory。盲门重新检查实际日志、日程、恢复分数、教师/Memory、参数及完整诊断，默认锚点额外逐位比较旧 A/B raw 分数。

[调度控制](../scripts/step28_optuna_control.py)和[生产入口](../scripts/step28_optuna_search.py)调用三个独立持续 Optuna 4.6.0 sampler，按原轮转逐一执行真实子进程；没有模拟结果回填正式 trial 的入口。每臂 trial0=(1,1)、trial1—3随机、trial4—11 TPE，12完整点后按(U,mean improvement,-trial number)精确选定。新 seed 九 run 均退出且 valid=0 后才集中解析一次。任一失败/超时/非有限值停止整个调度，保留全部账，SQLite不授予重启。

搜索 run 自 launch 前到子进程及所属 workers 退出、全评价完成，单次9000秒；复验 run 自 launch 前到各自盲完成退出，退出后的等待不算训练耗时。共享 C300/E2400/A600 分开计费，不能互借。父进程对子进程组设实际硬截止；控制与共享工作也有独立定时器。每个 run 资源检查同时覆盖整个 campaign 输出，峰值64 GiB、小证据2 GiB；search 权重生命周期须写入未来明确许可，复验18权重保留。

完整指标/计数/N/A、两端点/三角色、逐 seed 对比及三 seed 平均、旧域变化/遗忘、诊断和六格方向门分别保存。search 只把 B末 raw 双域 MAP 反馈控制器。开发结果与新 seed 均不能冒称最终独立确认；未来新 AB80群来源、隔离、预算仍需单独合同。所有45条轨迹的保存统计由独立分析入口核对；该核对不重新解析标签、不载模。

[增量资格入口](../scripts/step28_optuna_verify.py)与[手写参考/负例](../tests/test_step28_optuna.py)覆盖实际目标梯度/Adam/LR、随机根与自身恢复、调度和退出/计时。原生六调用的 Q0/QB 共同状态切换仅属明确手写 fixture，不能放宽正式恢复；QB计数336不是336次真实训练。原生实现虽可供静态审查，当前没有执行许可或新实测。

原生入口未来要求源码绑定的外审、主审、CPU和明确 NATIVE 许可；正式入口还要求明确 FORMAL 数据/预算/生命周期许可、六调用每臂真实证据和新时间资格。旧约38080秒双臂保守投影不可改贴9000秒单次 PASS。默认保留旧1.25系数/1800秒余量；新 P_search/P_rep 及共享 C/E/A 必须有具体测量依据且分别不超限。时间资格不足须报告，不能启动 trial0 试运气或从已见正式耗时倒推新门。
