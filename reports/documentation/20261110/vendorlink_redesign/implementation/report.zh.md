# 账号CE适配与有限调度：实现证据

2026-11-10。实际实现及必要Linux核验已完成；本轮完整实现外审和[独立主审](review/primary_disposition.zh.md)已闭合，没有必修代码或新增测试。用户最新已授权闭合后正式开始，见[范围补充](scope.zh.md)。启动前实际来源、资源和费用核对后沿来源绑定gate进入D；实际运行及真实更新另以启动回执证明。本报告只说明工程作用，不证明学习、遗忘或外部比较已达标。

## 1. 实际计算与入口

| 要求 | 实际接线与直接依据 |
|---|---|
| CE改善H账号学习，但保留H多记录评分 | `step35_vendorlink_account_methods.AccountModel.backward_h`沿原`records()`得到1536维z，保持两通道归一化拼接后除√2。原pair/BCE/rank/hard及自然/重组目标照旧；当前账号CE使用同一z的leaf，按当前全部K类的记录均值反向，H＋CE梯度合并后回到同一次encoder图。历史分支不加CE，辅助分类头不参与最终评分 |
| γ0/1唯一干预，完整历史结构不删 | D只有H_base/H_CE，γ=0/1；P/M固定H_CE。SEQ无历史、ER回放、Full回放＋出生评分保持，α/β、Memory上限和出生教师保留。原H梯度与CE梯度、当前及总encoder梯度分别留证；不拿loss非零替代编码器作用 |
| 辅助头与优化状态 | 当前域新建辅助Linear及AdamW，初始化全过程隔离Torch随机流；原encoder/pair AdamW跨域保留。所有实际参数合并裁剪一次，原optimizer每步只更新一次，γ1辅助optimizer更新一次。保存恢复同时包含原/辅助optimizer、类映射、模型、Memory、RNG及游标 |
| D确实包含A和B冷启动 | `policy.tasks`显式携带`domains`，`run.execute`、`validate_training`及`evaluate.score`均消费同一字段。Dream单域不会误训AlphaBay，也不会从A读取gallery/query；D按A/B的V_CLS均值选择V0/V1，H评价仍为集合评分 |
| 阶段与质量保护 | D八任务，P的H_SEQ/V各三seed51/52/53，M的ER/Full各三seed；C复用12条完整轨迹各自的stage0—3完整端点集合，不新增训练。3.5pp、逐域初学2pp/5pp、绝对能力、新域增量、Full初学/新学及对外部方法的组合门由`policy.decide`实际决定接续；完整训练后才评分，完整评分后才分析。M/C组件增量需两阶段都通过 |
| OOM续领与无GPU硬cap | 复用`step30_evolution_control.Runner`，只增加默认关闭的`continue_after_oom`；新控制器显式开启。实际worker把真实CUDA OOM与其他RuntimeError区分落盘，父控在子进程退出并释放后退休该槽，健康槽继续当前批独立任务；失败任务不重入，无健康槽则不自动补worker，缺项阻断下游。非OOM及共享故障保持各自停止语义。新配置不调用allocator配额接口，整PID GPU cap为None；5/17GiB只用于H/V派发预留，最多一条V同时运行 |
| 预算与来源 | `policy.CAPS`及`control.make_jobs`使用88h=training80/GPU4/aux4；原全帽预约和生命周期实耗计费复用。入口绑定实际13份运行源码、协议、scope、主审、原角色manifest与CPU/GPU证据；实现阶段许可关闭；本轮审查闭合后按最新明确启动授权开启。仅把新代码同步到独立implementation/workspace，没有覆盖Linux共享root或中文活动源码 |

六个新运行模块为`step35_vendorlink_account_{io,policy,methods,run,evaluate,control}.py`。只恢复原step34中本次确实需要的分组、端点、评分和阶段编排行为，不恢复旧OOM接续包装器，不新增数据构建器、训练恢复队列、调参框架或常驻监听。通用Runner新参数默认False；其他合同既有GPU上限仍按其原limits执行。

## 2. 原生核验与来源

| 核验 | 范围和结果 | 源码恢复点 |
|---|---|---|
| [CPU01](cpu01_receipt.json) | 4项通过，零失败/错误。独立普通autograd＋手写logsumexp CE参照实际leaf/分块更新；γ0与原H更新逐位相同；γ1改变encoder梯度。验证辅助头重置/原optimizer保留、全状态恢复；手写Dream冷训练＋实际Q2/G4评分；D/P门失败及缺seed阻断；80h末两任务预约可容纳；实际CPU子进程经过真实Runner，单槽OOM、健康槽续领与缺项阻断通过 | Git `cc8dd742`；[CPU01部署身份](cpu01_sources.json) |
| [GPU01](gpu01_receipt.json) | 唯一一次真实BERT GPU probe，17.5477秒。44记录、两通道各256token、946记录对、875类、1536维表示；共4次手写更新，包含γ0/γ1对照及保存恢复下一步。CE改变encoder参数，原pair/history均进入更新；恢复后模型与下一步日志逐位一致。没有读取真实账号/商品或任何角色载荷 | Git `1b3caa2c`；[增量源码身份](gpu01_sources.json) |
| [CPU02](cpu02_receipt.json) | 只补1项受影响入口检查，零失败/错误。主审发现`permit`沿用旧CPU回执`exit_code`字段，与本轮status/tests/errors/failures格式不同；修为实际格式。源码绑定许可正例/失败回执反例，以及真实`main`的CUDA OOM/普通RuntimeError分类均通过。不重跑CPU01或GPU | Git `f7e1f51f`；[增量源码身份](cpu02_sources.json) |

CPU共5项、两个有界作业，实际4.6817＋2.1886秒，向上取整计5＋3=8秒；GPU17.5477秒计18秒。未发生原生测试失败；CPU02修正的是主审发现的接口缺口，不伪报一次失败运行。CPU调度样例中预设的非零OOM子任务是故障注入，不是正式GPU OOM，其时间已含在CPU01生命周期内，不重复计费。

GPU观察：PyTorch峰值allocated=3255583744字节（3.032GiB），reserved=3584032768字节（3.338GiB），整PID采样峰值4265607168字节（3.973GiB），RSS采样峰值3745185792字节（3.488GiB）。CE开启前后tracked encoder参数更新最大差2.50041485e-5；γ1当前encoder梯度范数0.363649、γ0为0.017459。采样整PID/RSS不是连续峰值保证；这次单路短测不冒称双路长训实测。既有V成功训练路径未改变，按方案复用其证据，没有重做V资格。

原角色manifest5126字节、共享BERT六文件与19项初始部署源经过元数据/大小/SHA核对，见[预检](preflight.json)、[绑定](bindings.json)。只核对既有身份，没有重新解析数据或重新资格盘点。原文本角色资格、H表示/历史目标及V科学路径的适用证据从[方案§9](../proposal.zh.md#9-后续实现仅补哪些证据)和原已结案实现审查复用；本次实测只覆盖新增差异，不把旧PASS当新代码PASS。

## 3. 费用、收尾和结论

[费用账](budget.json)保存CPU8秒、GPU18秒，另将Linux元数据核对、源验证及收尾保守计入aux120秒（保守费用，非测得科研计算时长），本轮总计146秒、training=0。CPU子账包含在aux128秒中，不再相加；未来正式campaign沿原Ledger另计一次30秒最终收尾预留。历史20261109的71450秒及失败账不挪用、不清零。核验受本次CPU≤1200秒/单次GPU≤1800秒及原4h桶约束。

短测自身的`probe_latest.pt`在恢复比较完成后删除，删除前身份为1316372272字节/SHA `0afd89d68b91c358a6826c6f3083b533cbd0d6a23c120b052f86f7579cabee12`；没有形成可用于正式实验的训练端点。一次性CPU/GPU脚本按科研纪律§9在精确Git恢复及依赖检查后双端退役，证据与恢复点保留；具体以[收尾回执](cleanup.json)为准。正式13份运行源码保留，未来来源只绑定这些模块，不依赖已退役测试脚本。

主审结论：当前设计、实际接线及必要核验充分，本轮实现外审已回收并独立结案，无必修代码或追加资格。用户最新条件性正式启动授权有效，不新增候选、数据核查或重复GPU资格。当前实现经外审接受不等于科学目标已通过；实际启动与进展由交接和运行回执记录。即使代码正确，CE能否追平V_CLS、H_SEQ是否达到3.5pp遗忘、Full能否保旧且学新仍须由正式阶段回答，不保证得到预期正结果。
