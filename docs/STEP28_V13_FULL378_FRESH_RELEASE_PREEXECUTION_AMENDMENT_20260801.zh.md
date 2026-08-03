# Step28-v13 full-378 fresh release 执行前修正案

日期：2026-08-01
状态：`FORMAL_CONTRACT_AND_SEEDS_FROZEN_OPEN_SPLITS_NOT_GENERATED`

## 1. 当前结论

正式训练继续暂停。冻结数据集
`v13_training_ready_v1_2_order_repair_20260731` 保留原字节和历史审计价值，
但撤销其 M0/M1/M2/M3 正式训练资格。未产生任何正式模型效果结果。

撤销原因有三项，且彼此独立：

1. 公开 policy 中的候选密钥可重算 C40 分层抽样。仅用公开 UID 和该密钥，
   train/development 的标签攻击 AUC 分别约为 `0.89/0.86`。
2. C40 的 40 条候选边本身形成标签相关子图。仅用两端在候选图中的度数，
   development AUC 约为 `0.63`；换私钥不能消除此捷径。
3. v1.2 Audit A/B 的 controller membership、机制、身份资产和候选抽样明细
   已进入 Git 历史。“sealed”只是目录名；复用旧 UID、文本或卖家历史无法
   恢复真正的盲测边界。

旧 v1.2 必须表述为
`BLOCKED_PUBLIC_CANDIDATE_AND_AUDIT_CUSTODY_COMPROMISE`，不得通过删字段、
改 manifest 或重新排序恢复资格。

## 2. 新主分类总体

后继数据版本固定为
`v13_training_ready_v1_3_full378_fresh_20260801`。四个 split 全量重生，
每个 split 仍含 500 个相互隔离的 world；每个 world 有 28 个 seller，主分类
使用全部 `C(28,2)=378` 个无序 pair：

- 每个 world 固定 20 个同控制者正例、358 个异控制者负例；
- 每个 split 共 189,000 pair、10,000 正例、179,000 负例；
- 自然正例率与随机 AP 基线均为 `20/378=0.0529100529`；
- 每个 seller 的候选图度数恒为 27，每条边共同邻居恒为 26。

C40 完全退出训练、调参、阈值、主评估和 model mount。若保留机制诊断，
只能在所有模型预测冻结后由私有评估器按已冻结规则生成；不得影响主结论。
检索不再抽 4 个 query，而是每个 world 的 28 个 seller 全部轮流作为 query，
其余 27 个 seller 为 gallery。

## 3. 一次性随机性与跨版本隔离

生成前必须先冻结代码、合同、阈值和密钥用途。随后每个 split 一次性抽取一个
32-byte master seed，先落入 Git 忽略的私有保管区，再公开 SHA-256 承诺。
固定 HMAC-SHA256 域从 master 派生互不复用的 structure、ID namespace、ID、
文本、identity value 等生成子流；train 另外派生 r01--r05 五个 M1 rewire
子钥。ceremony 随即把子钥按用途分别封装：生成进程只挂生成 capability，
每个 M1 进程只挂自己的一把 rewire key；master 永不进入生成或模型进程。
任何生成或审计失败都使该版本失败，禁止换 seed、换域或筛选 seed 后重跑。
pre-ceremony lock 永不回写；抽取后另立 no-replace receipt 与 post-ceremony
execution lock 绑定承诺。

新四个 split 相对仍存在且已暴露的 v1.2，其 world、seller、item、pair、
query、relation、identity UID/值交集必须为零；卖家完整文档、profile 和
history 的去 UID 内容投影精确指纹交集必须为零。已删除的 v1 只按现存哈希
证据说明，不虚构逐行重放。Audit A/B 还必须通过以旧公开数据及旧 oracle
为攻击者能力的冻结近似链接检查；普通作者风格可迁移不等于一对一泄漏。

## 4. 公开与私有边界

公开 release 可以包含经删除身份的商品、卖家档案、完整 378 对端点投影、
pair key、真实 identity33、非连接审计收据，以及 train/development 的
full-378 标签。冻结 M0 projector 可读取文本与端点，但不得读取标签或
identity33。

Audit A/B 的 labels、qrels、controller、机制、身份资产、原始身份文本、
生成 AST 和逐行私有审计必须只存在于 `private_custody/`；不得被 Git 跟踪，
不得列入公开 manifest，也不得挂载到模型进程。M1 的五份错配矩阵和映射同样
留在训练私有能力区；每个拟合进程一次只挂一份数值矩阵。

可训练 adapter 的实际设计矩阵只能含冻结 `p0` 和 33 个数值身份特征。
world/pair/seller UID、端点、行号和文件顺序只能用于受信 join 或聚类统计，
必须在 `fit/predict` 前删除。禁止按 world 强制 top-20、配额解码或利用已知
正例数后处理预测。

## 5. M0/M1/M2/M3

- M0：只用真实英文标签训练并冻结的
  `LightGBM + legacy18 + LaBSE`，对新四 split 的全部 378 对重新打分。
- M1：五个独立对照。每个 train world 在完整 378-pair universe 内对整行
  identity33 做 endpoint-disjoint 双射；不读标签、controller 或 C40。
- M2：与 M1 使用完全相同的 pair、p0、权重、变换、正则和收敛策略，只把
  错配 identity33 换成正确对齐 identity33。
- M3-base/M3-joint：只用新 synthetic train 标签的直接中文训练强对照，
  必须按 full-378 重新选择超参数，旧 20,000-row 结果和配置均不得复用。

adapter 固定使用
`a=I(any(z!=0)); phi=a*(z/scale-mu)`：`scale` 和 active-row、world-equal
均值 `mu` 只能由当前 train fold 拟合；无截距，零 history 的 `phi` 严格全
零，M1/M2 共享同一 fold 的 `mu/scale`。不做标签重采样或 class weight。
L2 只用 fresh train 的 world-grouped OOF、固定候选网格和固定平局规则选出
一个共享值，然后同用于五个 M1 与 M2；development 只冻结全局阈值，不能
选择 L2、权重、变换或模型结构。

类别先验改变后，旧 C40 截距、阈值、class weight、L2 和 AP 门槛全部失效。
核心身份增益以 `M2-mean(M1)` 和五个 `M2-M1_r` 为主；`M2-M0` 同时包含
目标域校准影响，只作辅助解释。

## 6. 训练前 GO 门

只有以下条件全部满足才允许正式训练：

1. 四 split 均严格为 500 个 K28 完整图，无缺边、重边或自环，标签公式逐行
   零差异；Audit 只发布私有验证 PASS 和不可逆承诺。
2. 新旧 UID/身份值/精确文本与卖家历史指纹交集为零；Audit 近似链接攻击通过
   预注册门。
3. train 拟合、development 一次评估的 UID/hash/行序/公开常量/元数据攻击：
   单项最大对称 AUC 不超过 0.52，world-bootstrap 95% 上界不超过 0.53；
   组合模型 AUC 不超过 0.53；以 `0.0529100529` 为 AP 基线，点 uplift 不
   超过 0.01，world-bootstrap 95% 上界 uplift 不超过 0.015。
4. 禁止身份残留、controller/mechanism/label token 和内部生成标记逐行出现；
   字符 n-gram 总体可预测性单独报告，因为合法作者风格本来就是 M0 信号。
5. 五个 M1 在每个 world 都满足 378 行双射、零 fixed point、零端点重叠、
   33 维联合向量多集逐字节相同。source-label 到 destination-label 的 AUC、
   AP 和 world-bootstrap 区间必须完整报告，但不得用其筛 seed 或重抽数据；
   正式 null 资格由结构硬门、M1 对 M0 的预注册 TOST 及 M2 胜过每个 M1
   共同判断。
6. 实际模型 mount 与设计矩阵 schema 精确通过，私钥、UID、端点、oracle、
   Audit 真值和多份 M1 能力均不可达。
7. 新 M0 投影、CPU/GPU 同步、模型、分数、阈值、盲预测和评估链全部绑定
   v1.3 release/hash；旧 v2 链必须 fail closed。

主分类至少报告 ROC-AUC、AP、梯形 PR-AUC、Precision、Recall、F1、balanced
accuracy、MCC、Brier 和 Recall@FPR=1%；阈值只在 development 全局冻结。
检索报告 MRR、MAP、Recall@1/3/5/10 和 NDCG@1/3/5/10，并以 world 为 bootstrap、
置乱和置信区间单位。不得把 189,000 个 pair 当作相互独立样本。

## 7. 当前执行状态

本修正案冻结的是设计与执行合同，不是数据通过证明。fresh full-378 数据生成、
M0/M1/M2/M3、development 等价门和分阶段 Audit 代码已经实现；正式 identity
design self-hash 为
`0b2ae09ed17cd68e313f4add60960f8e9fa192c0bd56ccb1a7702db7687ba5c1`，
正式 prelock self-hash 为
`23faeeda89cae653af8f1a2d363f436341bb826d9502bf676aa73a051130ccac`。

正式 Windows `Validate` 入口实际运行 472 项，结果为 465 项通过、7 项既有
声明跳过、0 项失败，用时 792.518 秒。development、Audit A/B 盲预测和 Audit 指标均要求从冻结
输入逐行重算并生成深度重放回执，不能再以浅层文件哈希代替计算重放。

正式 identity policy/prelock 已冻结；train、development、Audit A、Audit B
四份一次性 seed 承诺已按固定顺序完成且禁止重抽，原始密钥未进入公开输出。
尚未执行的是四 split 正式生成与审计、Linux M0 编码、Windows 模型训练和
Audit A/B。因此正式 v1.3 split、模型和结果仍均为 0；完成数据门前状态始终为
**NO-GO**。

## 8. 2026-08-02 v1.3 失败与 v1.4 修复

v1.3 的四份 seed 承诺已冻结，但 train 在 500/500 world 后因机制模板槽被
错误要求跨 world 全局唯一而 fail closed；没有正式 split 发布，禁止原地修补
或复用 seed。v1.4 把唯一性改为 `(world_uid, mechanism_slot_uid)`，增加同
world 冲突负测和两个完整 world、756 pair 的真实生成重放门。冻结前全仓
476 项测试为 469 通过、7 声明跳过、0 失败。当前 v1.4 identity design
候选 self-hash 为
`6b629f139b34c464bc907cbe77f91488b4a39f82f1a5ae1f2a9bf7a83256f31f`，
draft prelock self-hash 为
`1ba6770eb09c65a4b150899fdd7e5885010fefae1027d8ec435ebf1978385944`；
正式 policy/prelock、seed、split、模型与效果仍为 0。

## 9. 2026-08-02 v1.4 写盘失败与 v1.5 后继

v1.4 在 train 500/500 world 构造后未能发布。失败点不是数据机制指标，而是 Windows 持久化实现：长路径安全写入后的复核使用普通 `Path.read_bytes()`，在 264 字符路径上误报缺失。后续静态对表还发现 split manifest 生产者为 v2、消费者为 v1。v1.4 必须按 one-shot 纪律永久关闭，不能原地修补或复用 seed。

后继运行固定为 `v13_training_ready_v1_5_20260802`。修复包括长路径安全读/遍历/stat、manifest/execution-lock 单一版本源、真实两-world 写盘与正式消费者回归。v1.4 已生成但未发布的 42,000 个身份值全部以哈希加入 v1.5 排除库，原始值不公开；v1.4 四个 master commitment 也加入禁用集合。

v1.5 identity design、shortcut、near-link 和 prelock 已冻结；prelock self-hash 为 `04636dcf45bf295111d434aadf240425af451c268501073bee9eea094b21c589`。正式 `Validate` 为 480 项、473 通过、7 跳过、0 失败。四份 seed 已按固定顺序一次性提交且原始密钥未公开；写下本节时 split 仍为 0。
