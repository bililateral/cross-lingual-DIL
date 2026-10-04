# 完整群隔离的泛化重放：手写实现范围

2026-10-04。用户已要求按科研纪律尝试实现此前讨论的候选，并明确回复“按此预算执行手写核验”。此文件只定义本轮实现与手写核验；不授权正式训练、正式文本或标签访问，不宣称方法创新或效果已成立。新增预算为既有Linux py310、一个空闲CPU核、禁用GPU、总计算≤2小时、单进程内存≤64GiB、证据≤128MiB，失败修复计入总预算；见本轮authorization.json。

## 问题与目标

历史训练数据极少、新卖家群体不断到来时，学习当前群体，并保持旧分布未见卖家的候选排序和基础识别。现有结果提示缓存拟合与泛化可能分离，未证明过拟合是唯一退化原因。完整群隔离用于检验一次更新对其他训练群的影响；外层群仍为训练数据，不是未见测试卖家。

## 唯一候选及精确计算

四个完整群 Cs/Hs/Cq/Hq 分别来自当前拟合/历史存活缓存/另一当前拟合/另一历史缓存。四群UID、账号UID和商品UID不重叠。每群仍28账号、378无向关系；身份与域不进文本模型。监督、候选竞争、基础损失沿用LOGIT0.1。

Ls(theta) = Lbase(Cs;theta) + 0.1 Lbase(Hs;theta) + 0.5 mean((z(Hs;theta)-saved_target(Hs))^2)

theta' = theta - A grad_theta Ls(theta)

Lq(theta') = Lbase(Cq;theta') + rho Lbase(Hq;theta')

J(theta) = Ls(theta) + beta Lq(theta')

A为显式指定编码器/头内层学习率的对角矩阵；这是SGD前瞻，并非带裁剪/权重衰减/动量的真实Adam下一步。精确梯度为 grad Ls + beta (I-Hs A) grad_theta' Lq；在Top5集合和ReLU活动区固定的局部求导，不对离散候选选择求导。生产实现保留create_graph二阶路径，无detach一阶替代。正式优化器只对J执行一次裁剪和一次AdamW，阶段学习率与Adam连续计数沿用父机制。

beta=0仅作为退化核验对照，关闭外层计算；不把它称为元学习。rho/beta/内层学习率无正式默认值，由Settings显式传入。Tiny夹具使用(.03,.04,.3,.7)，原生手写夹具使用(.0001,.01,.3,.7)，顺序为内层编码器lr/头lr/rho/beta；这些数只检验运算，未据效果选择，不是正式实验候选搜索。

## 实现及边界

新增scripts/step28_group_meta.py。GroupScorer包装原标题/描述→BGE→分通道统计→对称头路径，通过torch.func.functional_call替换完整参数字典；保留原模型对象和优化器状态。原实现文件不改。外层基础损失保留BCE/rank/hard，不另加风险分布损失。

本轮采用float32、eager attention、关闭activation checkpointing。原因是二阶导数与函数式快参数必须实际支持；checkpoint在functional_call退出后重算可能使用原参数，不能以显存优化引入伪计算。此配置与既有bf16/checkpoint基线并不计算配对，未来正式比较需统一或明确对照，不能直接把当前实现视为正式GPU就绪。

原生手写首次因既有Transformers缺少运行时set_attn_implementation退出，修为加载时指定eager。随后microbatch=4在完整内层create_graph触及64GiB地址空间上限，未完成元更新；原源码/日志保留。后继只把原生手写文本microbatch改为32，以减少逐小批产生的参数梯度图；所有账号、记录、参数和二阶项保留。该执行配置改变dropout随机数消费分组，不能称与旧批大小逐位等价，也不是正式实验配对设置。

microbatch=32随后成功构建完整目标，但原生核验脚本硬编码的SentenceTransformer模块名与实际命名不同，停在fast参数探针定位；修为按原参数对象身份反查实际名称。此为核验脚本错误，未执行该次正式元更新，不改变生产目标；完整失败留证。

train_stage只接收当前48拟合群和六群自包含LOGIT缓存，不接收Archive、校准或valid。当前support沿用288步日程，query为每轮48群日程循环移一位，各群每阶段support/query各出现6次。历史support沿用父draw，query从其余5群独立无状态抽取。四种dropout种子独立固定。stage2/3各288次正式更新，1152群梯度呈现（不是旧576），二阶重算另计成本。源阶段与阶段末编排由未来正式合同定义；本轮没有接入正式数据入口。

记忆复用Algorithm R及来源阶段eval logit：六完整群、总历史及附属状态≤1MiB，旧存活target不刷新，淘汰不重读。新增持久设置纳入auxiliary序列化；query随机流由已有seed/order/stage/step重建，无额外历史档案或教师模型。阶段一没有历史群，仍应使用既有基础训练；不得填假历史群。

## 最小核验证据

- 独立标量反例：support更新改善自身却恶化query；精确解析/有限差分验证元梯度，并识别detach内层或未使用快参数的变异。
- 四个手写28账号群：真实文本链上核对外层梯度和二阶校正到达encoder/head；改query改变实际梯度；beta=0对齐原LOGIT梯度。
- 生产update：临时过程不改参数/Adam/RNG，一次正式clip/step；完整model/Adam保存恢复后相同下一步。
- 阶段调用spy核对288步群隔离、不同非默认参数透传、参考按UID绑定、存活旧目标不刷新、记忆预算。每UID使用不同手写参考值，以识别错绑；这些不是来源阶段真实模型输出，spy不称288次真实训练。
- Linux既有BGE原生CPU：一次真实warm后显式将Adam计数1置288，一次真实元更新；编码器首层query与隐藏头探针分开核对外层梯度和Hessian校正及实际参数变化。不是正式训练效果或完整288步证明。

所有样例手写、禁止正式数据加载。预算及失败修复范围已由本轮用户确认；技术失败保留源码/日志，修复限同一计算定义和核验预算，不能自动换成一阶或头部近似。

## 新颖性与未来比较

MER：https://nlp.stanford.edu/projects/mer/；La-MAML：https://arxiv.org/abs/2007.13904；MLDG：https://arxiv.org/abs/1710.03463。一阶展开Lq(theta')约为Lq(theta)-grad Lq^T A grad Ls，是已有梯度对齐/元泛化结构，不能因整群或LOGIT组合声称首创。本轮实现机制候选，不承诺论文创新已成立。

后续须比较LOGIT0.1、相同群呈现的普通联合训练和最直接元学习重放适配，区分额外数据/计算与群隔离机制；具体比较、超参数、种子及训练预算另定。六个query缓存不是真正未见身份，不能保证旧域泛化；BCE/rank代理不保证MAP/FPR。

## 外审范围

网页GPT 6 Pro使用本地Playwright：审查主线、与MLDG/La-MAML等实质差异；按四群供给→真实快参数→外层损失→二阶梯度→单次正式更新追查伪实现；核查反例、训练/评价模式、恢复、记忆与额外计算公平性。旧基础/LOGIT已关闭证据按未变范围复用。手写通过不替代正式训练准入。新审查包完成后按具体内容确认上传范围。


## 2026-10-04 外审后补充

本轮外审与主审已完成，见[处置及勘误](../reports/documentation/20261004/group_meta/report.zh.md)。唯一生产修正是beta0阶段群呈现应记576，正beta仍1152；完整目标与二阶路径不改。局部导数范围包括固定Top5、ReLU及abs活动分支。原实现说明中head范数误抄的正确值为0.92513164，原件保留。Linux六项增量微型核验通过；仍不宣称已具有非平凡方法创新或未见卖家收益，也不据此启动正式训练。
