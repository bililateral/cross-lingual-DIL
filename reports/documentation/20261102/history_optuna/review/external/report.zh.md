# H_pos_tuned 两项历史系数有限 Optuna：3-worker 方案与实现定向外审

审查日期：2026-11-02。审查对象仅为本次 history_optuna_review.zip 及用户随请求提供的网页状态勘误。此报告不是此前 H/S 方案 A 的结果审查。

## 一、结论与处置优先级

本轮没有发现新的 SCIENTIFIC_BLOCKER：两项历史系数确实进入真实目标、encoder/head 梯度及 Adam 更新；候选 U 的实际 RAW 来源、三顺序一次反馈、固定选择规则和新确认统计链均有实质实现，不能判作“只有组件存在”。

发现两项已证的 REPRODUCIBILITY_DEFECT：

| 编号 | 结论 | 严重性与是否阻断 | 最小处置 |
|---|---|---|---|
| HOPT-R01 | GPU 准入混用先读 free 与后读自家占用，可能重复抵扣同一份预留 | 中等；应阻断当前资源准入实现的正式放行 | 本轮新控制器采用不因自家用量增长而返还的整槽预留信用，保留实际资源门；用已有手写 CPU 入口核对交错观察反例 |
| HOPT-R02 | 最终结算可被 monitor.close 异常跳过；最后落盘缺有限尾部覆盖 | 低；不否定科学方案，但正式记账封板前应补齐 | 局部 finally 保证清理失败也结算，并在原辅助桶内一次有记录地预提有限尾部 |
| CPU02 | 汇总测试 fixture 写错层级 | 已按原件关闭 | 保留 FAIL、修前版本及费用 |
| CPU04 | CPU 包装缺既有 CUBLAS 环境设置 | 已按原件关闭 | 保留 FAIL、修前版本及费用 |
| 方法、反馈、盲门与统计 | 本轮未发现新的已证科学错误或正常路径必现字段错误 | 可接受其静态实现结论 | 保持现有范围，不追加科研试验 |

上述两项是本轮实际代码的可达反例，不是声称已经发生了新的 GPU 事故、超预算训练或错误科研结果。正式门当前关闭是正确边界，不是缺陷。本报告不替主执行者宣布修复完成，不授予正式运行、数据开放或资源预算。

用户最新勘误优先：implementation.zh.txt 第33行关于 Pro“已达到使用限额”的旧判断不作为本次不可用证据；隐藏辅助文字没有证明当前 GPT-6 项被禁用。完整包的具体上传许可已经给出，不恢复历史“待上传许可”为当前阻断。本轮也没有独立取得网页后台路由证据。

## 二、输入身份与审查方法

已核对：

- ZIP：514462 字节。
- SHA256：39952612d10e561014d0de8586fbf9bff55ba98396d17616976e28fcfafc1118。
- 133 个成员：132 载荷＋input_manifest.json。
- 清单覆盖精确，无缺项、额外载荷、重复成员或不安全成员路径；132 载荷的实际大小和 SHA 均与清单一致，ZIP CRC 校验通过。
- 当前 58 项来源与 formal_gate.sources、CPU05 receipt.sources 逐项对应；CPU05 receipt 的实际大小/SHA 也与 formal_gate 的引用相符。

清单/SHA证明本次读取的字节身份，不证明代码科学正确，也不替代真实模型恢复。

实际操作限于解包、字节/SHA/JSON结构核对、源码和既有原件读取、静态差异及手算。没有导入或运行项目模块、tests、训练、模型、指标、bootstrap、生成器或 NNLS；没有连接项目 Linux，没有读取未提供的正式文本、标签、owners、权重或 Memory。

以下位置均以输入 ZIP 解包根目录为基准，行号对应本包冻结字节。为便于回收，采用这些简称：

| 简称 | 文件 |
|---|---|
| D/ | reports/documentation/20261102/history_optuna/ |
| M | scripts/step28_history_optuna.py |
| R | scripts/step28_history_optuna_run.py |
| C | scripts/step28_history_optuna_control.py |
| E | scripts/step28_history_optuna_evaluate.py |
| Q | scripts/step28_formal_queue.py |
| T | tests/test_step28_history_optuna.py |
| CPU | scripts/run_step28_history_optuna_cpu.py |

## 三、HOPT-R01：自家显存预留会因混时点观察被重复抵扣

**类别：REPRODUCIBILITY_DEFECT。已证；正式派发前必须修。**

### 证据与实际接线

Q第87行先调用 nvmlDeviceGetMemoryInfo 读取 free，第90行随后读取进程占用。Q第102、104行用后一次的自家进程占用，扣减已有槽的未兑现预留，并与前一次的 free 比较：

~~~text
pending = sum(max(0, S - owned_usage))
required = S + pending
ready = old_free >= required
~~~

C第183行确实调用这条资源路径，准入后由第195行创建 worker。这不是没有被生产入口调用的旧 helper。D/scope.zh.txt 第28行和 implementation.zh.txt 第8行明确承诺覆盖尚未分配 CUDA 的已派发槽预留。

### 可手算反例

令每槽 S=10.25 GiB，已有一个自家 worker；外部占用在整个反例中保持不变。

| 时点 | 观察或变化 |
|---|---|
| t0 | 自家 worker 尚未分配 CUDA；先读 free=16 GiB |
| t1 | 同一个自家 worker 分配8 GiB |
| t2 | 后读自家进程占用=8 GiB |

代码得到：

~~~text
pending = 10.25 - 8 = 2.25 GiB
required = 10.25 + 2.25 = 12.5 GiB
old_free = 16 GiB
ready = True
~~~

但 t2 的真实 free 已为8 GiB；旧槽还可增长2.25 GiB，新槽还需要10.25 GiB。同一8 GiB既留在较早的 free 中，又从较晚的 pending 中扣除了。

因此，这不是“外来进程在准入之后突然抢卡”的普通边界，而是准入函数自身对自家容量的重复计数。它可能在实际余量不足时启动额外 worker，浪费已批准预算并触发 OOM、槽退休或候选不完整。本轮没有正式 GPU 运行，不能把反例写成已观测事故。

### 最小修法

不改变三槽、9.25 GiB allocator、10.25 GiB进程阈值、候选数量或预算。修复可以只落在本轮独立控制器的准入适配中，不修改既有活动链的冻结文件。

一种局部且可解释的方案是“空闲时冻结可用容量，并按完整槽记账”：

~~~text
active 为空、旧进程组完整释放时：
    C = 此时实际观察到的空闲 GPU 容量
R = 已派发且尚未完整释放的槽数 * S

新增槽要求：
    C - R >= S
    且最新观测 free >= S
    且原 RAM / disk / deadline 等条件通过

派发即 R += S。
只有完整释放后才 R -= S。
active 非空时，不按 NVML 自家当前 usage 返还或增发容量信用。
~~~

在“该段内外部占用不增加、每槽实际不超过 S、释放槽不再持有 GPU”这些明确条件下，现存实际占用不超过 R，新增后的完整预留不超过 C。自家任意分配/释放不会重复返还同一份信用。反例中 C=16、R=10.25，C−R=5.75<S，必拒；C至少30.75 GiB时仍可容纳三槽。

这会更保守；不能借此声称共享 GPU 资源已变成原子锁。外部占用变化、NVML观测延迟和瞬时超限仍应保留为实际资源边界；观察到外部变化不能以自家用量抵扣为由增发信用。原现场不足门、进程阈值和失败规则继续适用。

仅交换两次 NVML 调用顺序不构成一般修复：自家释放时会出现对称不一致。夹取两次 free 也不应被描述为严格原子快照。

最小新增证据是现有手写 CPU 调度检查中：上述16→8交错、对称释放、正常三槽、完整释放后返还信用。无需新增 GPU 运行或资源平台。本报告只提出修法，没有实施或运行这些检查。

## 四、HOPT-R02：最终费用结算存在有限缺口

**类别：REPRODUCIBILITY_DEFECT。已证；LOW。不是科学目标阻断。**

### 已成立的主体

C第90行的 Ledger.control 在循环中会把上一轮 save/check 消耗结转到下一次 tick。C第140行在进程组释放后把子任务真实 elapsed 加到训练/确认/辅助桶，之后的验证由第157行计入辅助。活跃子任务的轮询睡眠不另计控制费，无 active 的资源等待计辅助。因此不应笼统说“整个并行账都漏算”或“验证双计入训练”。

### 剩余缺口

1. C第340行 monitor.close 若抛错，直接进入第342行异常分支，第341行最终 ledger.control 被跳过。
2. 正常末次 control 的时间截点在其自身 save/check 之前；C第349行最终 completion 写入也没有下一次 tick。
3. 当前 prior auxiliary=89，未另记有限末次尾部预提。

这是静态异常分支和末次截点的明确事实，不是已经超出280小时的证据，也没有依据夸大未计时长。

### 最小处置

- 在本地嵌套 finally 中保证关闭 NVML 失败也尝试结算，保留原清理异常，不能吞掉变成成功。
- 在既有辅助4小时内一次登记有限尾部保守预提，或采用已有外层结算方法；不增加总预算。可直接参考包内10月23日 primary_disposition.zh.txt 第10行旧I02的同桶30秒预提做法，但必须标明这是保守记账，不是假装精确耗时。
- 保持成功回执只在清理和费用检查通过后写出的正确顺序。

只需针对末次落盘成本及 close 抛错分支做少量手写行为检查；不需要无限递归自计时、通用账务系统或科研重跑。

## 五、两系数与真实训练路径：本轮可以接受

### 实际数学定义

M第111、118、121、125行实现的目标可回收为：

~~~text
L_sup(G) = 0.5 * sum_{view in {A0,A1}} [
    BCE(view) + lambda_rank * Rank_positive(view) + 0.5 * HardTop5(view)
]

L_current = L_sup(current)

L_history = alpha * L_sup(history)
          + beta * mean((H(current_record_logits, A0)
                         - H(saved_birth_record_logits, A0))^2)

alpha in [0.03,0.3], log search; anchor alpha=0.1
beta  in [0.1,1.0], log search; anchor beta=0.5
lambda_rank = 1.3051794547562026
lr_multiplier = 1.2832655111666686

H(x) = 0.5*mean(x) + 0.5*(logsumexp(x)-log(len(x))), tau=1
~~~

没有把历史项误写为 alpha*(supervision+beta*MSE)，没有把教师换到重组A1，也没有把当前监督乘以 alpha。H汇聚和逐正例排序仍由既有 g_half / positive_ranking 实现。

### 损失确实作用于真实参数

M第134行将新 Objective 注入继承 Method。step28_continual_five.py 第203、209行把该 objective 交给真实 record.backward_group；step28_record_replay.py 第165、169、176、180行依次执行真实encoder表示、目标对完整记录对表的梯度、真实head分块反传、再向encoder反传。five.py第213、217行对全参数裁剪并执行Adam。

独立CPU参考没有复用同一个objective来“自证”：T第24行的手写公式、独立池化/正例排序，和第109行直接完整图更新用于对照。CPU05/objectives.json 中，改变参数后删除历史监督、删除教师、错用旧系数的梯度最大差分别为：

~~~text
0.008532012974866755
0.00482197168382575
0.004740516693535904
~~~

CPU05/updates.json 保存的完整图对照最大参数差为1.862645149230957e-09；encoder和head确实变化，恢复后同一步的模型/Adam摘要及日志完全一致。这支持关键运算已经产生作用，不仅是metadata和选择张量存在。

“当前群权重仍1”是目标系数的陈述。总体梯度仍共用原 clip_norm=1，历史权重变化可通过共同裁剪影响最终更新；不能再把它解释成当前项对最终参数位移完全不受影响。

### 出生教师、Memory、恢复

记录级Memory保留逻辑仍在 step28_record_replay.py 第248、269行之间的 retain 路径：新入选群生成本轨迹出生评分，留存旧群的表SHA不变，淘汰同步删除表和origin；没有读回淘汰档案。全局6群及1 MiB载荷限制沿用，包裹的spec身份包含alpha/beta。

M第75行身份包含两系数；five.py第120、145、253行分别检查Memory、模型metadata和恢复。T第134、139行实际拒绝不同alpha的模型和Memory。

继承 three-stage入口的当前群、阶段cal、Memory切换、连续Adam、288/864更新和1440群呈现，与R第129行完成门检查的字段一致。每阶段真实保存/恢复/重评分和排序核对仍由five_run.checkpoint执行。未发现新接口使训练完成后必然因这些字段或路径报错。

CPU更新测试只做了3次生产Adam和1次独立参考；初始Adam step=624是手工构造。不能将其改写为已跑过864步完整轨迹。

## 六、三worker、失败隔离与来源：除R01/R02外主体可接受

C第301、304、305、306、309、310、311行形成真实顺序：

~~~text
同一候选 ask
→ ABC/BCA/CAB三条完整训练
→ 全部完成并释放
→ 一次联合旧dev评价
→ 父进程重核实际反馈
→ 一次tell
→ trials持久化
→ 条件退役该候选权重
~~~

不是同时提出三个候选，也不是每成功一个顺序就提前tell。R第115行核对整批spec精确列表，C第211行核对全部receipt成功。失败不会选取成功子集作为完整候选。

九复验由共同pending队列续领。C第155行退休失败槽，第172行跳过退休槽；健康槽继续。CPU05/three_workers.json 与T第149、174、182行相符：两个真实九任务队列、首三个子进程时间重叠、故障队列1失败8成功、失败槽只执行其原任务；另真实制造leader退出而孩子存活并清理。它不是三个初始进程的空壳测试。

完整进程组管理在C第46、61行；没有只按leader exit认定完成。子任务费用计至完整释放，验证记辅助，完整额度在派发前预留。270小时=最多(36搜索+9复验)×6小时；另外确认6小时、辅助4小时，共280累计任务小时。并行任务墙钟相加，不除以三，也不是GPU利用小时。

磁盘方面，新R第94行采用Q第139行的并发消失容忍扫描；共同campaign的checkpoint.lock由five_run第104行flock保护，预留后才保存。权重实际直接写.pt，未发现大模型临时文件被误算进2 GiB小证据的新路径。父进程检查总RSS/目录，worker检查deadline、campaign总量、小证据和系统余量。无需重开11月1日已关闭的扫描竞态。

C第20行whole-chain检查实际核对FOREVER和VendorLink两批gate绑定、整链完成、各自精确动作集合、费用、失败和未退出进程。包内observation只是其保存时点，不能当本次实时状态。没有新listener被启动。

## 七、实际U与12候选：旧“反馈自洽不等于来源正确”争点已实质补足

E第63行development从匹配的公共分区取得旧dev群顺序，只按规则一次attach_labels。第88、93行把实际端点保存分数按真实域行索引交给真实metric_set。第101行verified_feedback只读最终stage3的RAW三域指标，核验spec、development_ids、实际域、该域20群ID和原RAW文件大小/SHA；第52行metric_matrix核对22列、群/query形状及检索十列与query均值的对应。

~~~text
U(theta)
 = (1/3) * sum_order [
     (1/3) * sum_actual_domain [
       (1/20) * sum_group MAP_raw(stage3, order, domain, group)
     ]
   ]
~~~

C第306行在tell前重求，之后12候选完成又由第316行逐一重核，才冻结赢家。候选权重退役后，该再核只需保留的分数、指标、query、日志和校准，不要求重新载入已删模型。

这不是第二套独立指标重算，但已满足“从实际保存RAW群/query指标重新求U并绑定来源”的要求。不能只把Search.tell接收的三个数当作全部证据，也不应沿用10月18日N05开放状态否认本轮新增的真实链。

M第147行Search用既有Optuna4.6.0、TPE seed2026110203、startup4、EI24、NopPruner，锚点真实enqueue。pending未完成不得ask；12候选硬限；精确平局依次偏好原参数、较早trial；失败后不继续ask/selected。C第322行原参数胜出时不重复九复验或生成新确认。

测试确实使用实际Optuna，但反馈是手写值。RAW绑定测试patch了complete_batch，且三个域采用相同内容的测试分数；它不能独立证明异值域数值置换永远被测试抓住。当前生产代码的实际域切片正确，这一测试边界不构成已证错域缺陷，也无须因此新增整套评价平台。

## 八、原H27、新确认与统计：静态接线可接受

### 来源与计数

R第188行references在任何新训练前由C第297行调用，核对固定原H gate/completion、九个唯一(order,seed)、864更新、三端点、source、恢复标记，以及实际模型和mapping文件身份；缺失不重训。E第44行按新旧相同order/seed核对partition和pretrained_archive完全一致。

本包没有这些模型载荷；本轮不能声称原H27已实际加载或新九轨迹已完成。

新确认root2026110201与namespace h_history_confirmation_20261102进入固定生成器的ID、身份偏好和文本随机流，不是只更名。E第149、151、155行实际调用生成/验证并检查ID覆盖，owners在生成核验后封存；这是将来获准动作，CPU05没有运行生成器。

计数相符：

~~~text
新9轨迹 + 原H9轨迹 = 18轨迹
18*3 = 54推理状态恢复
18*6 = 108域阶段格
18*6*40 = 4320群RAW前向
108*3角色 = 324群指标矩阵
新确认独立群数 = 3*40 = 120
~~~

E第214行实际restore，第220行实际score，第221行校准。第229行完整格门和第232行全部矩阵复核后，第237行保存all_blind_complete，再由第238行登记一次pairs尝试，之后才读取标签。原access在读前登记并拒绝同角色重开；失败不清账。校准在每个新40×378矩阵上检查排序和精确ties。

CPU05调用了实际confirmation函数，保留真实restore/score/calibration/save/align/metrics/access；替换的是TinyEncoder构造、手写public群和CUDA设置。54/4320/324是真实手写入口计数，不是正式数据或GPU结果；三评分角色也不增加独立群样本数。

### 22指标与N/A

底层群指标的22列为：

~~~text
average_precision, trapezoidal_pr_auc, roc_auc, recall_at_fpr_1pct,
brier, log_loss, precision, recall, f1, specificity, balanced_accuracy, mcc,
map, mrr,
recall_at_1, recall_at_3, recall_at_5, recall_at_10,
ndcg_at_1, ndcg_at_3, ndcg_at_5, ndcg_at_10
~~~

five_evaluate.metric_set第47、50、53行保留真实群指标、28query×10检索明细及20正/358负计数对应，并把无定义Precision/MCC置NaN/null。E第273行用普通mean传播N/A，给defined_groups和群数；另算合并四计数后的Precision/MCC，不用其替代群平均。

当前输出精确层级是“每格四计数＋pooled Precision/MCC”；没有另外四项pooled分类率，也没有O/F各符号时点的整套合并率。范围原文可按现实现理解。最小建议是报告把这一层级写清，不泛称所有层级合并统计均已产生；若主执行者原意确需其他率，只能从已保存计数作必要代数，不由外审自动扩项。

### 聚合、配对与主判据

E第279、282行先把贡献放在真实A/B/C轴，再对九条固定轨迹等权。定义为：

~~~text
O = [M(3,1)+M(3,2)]/2
A_acq = [M(1,1)+M(2,2)]/2
F = A_acq - O
N = [M(2,2)+M(3,3)]/2
Z = M(3,3)
F_first = M(1,1)-M(3,1)
final_mean = [M(3,1)+M(3,2)+M(3,3)]/3
~~~

E第297行一次建立PCG64、seed2026110202的(5000,3,40)共同索引；相同实际域群跨方法、轨迹、角色和字段配对。九轨迹先等权再抽群，不把重复模型评分当独立样本。相关NaN继续传播。

唯一主判据为：

~~~text
lower_95pct_CI[
  RAW final_mean(H_pos_history) - RAW final_mean(H_pos_tuned)
] > 0
~~~

它支持固定选定配置在九路径、同机制新群上的最终排序提升。不能由此自动宣称遗忘减少、每域/顺序/seed无下降、总体非劣或真实市场有效。F需与A_acq、O并列解释；低取得也会使F变小。Brier/log_loss保留字面差分，正F在这些低值更好的列上不应称作遗忘加重。

当前5000次区间只在confirmation=True的新确认分支生成；旧dev回顾是点值/差值，不应写成已产出旧dev区间。scope第21、23行可以作这一有限解释，不必为此新增重采样。

旧dev已用于开发，seed4/5/6已见，新群仍来自同一合成机制；这些限制已写入scope和summary。该轮是额外有限开发后的联合系数配置比较，不能把alpha和beta各自的独立作用分解出来，也不能覆盖旧H−S主要门失败。这里没有理由追加新O/F成功门或按结果再调门。

## 九、条件退役与关闭的正式门

C第236行retire_search_weights只在候选三条完成、恢复/开发评价/真实反馈检查、tell与trials落盘后被调用。它限定该候选models/stage1/2/3.pt，核对属主、单硬链接、非symlink、实际大小/SHA，先清单后逐项删除与回执。失败候选不进入；新复验27和原H27不在删除调用范围。

删除后保留SHA不是可恢复模型；当前不再依赖搜索权重继续训练或做新评分，保存的分数/query等可支持后续选择核查，故此条件退役本身合理。若将来出现新的实际诊断依赖，应在退役前处理，不能把该约定改成所有权重永久保存。

当前formal_gate中approved=false、review_closed=false、actions=[]，train_data/new_confirmation_data/retire_search_weights均false。R第52行先拒绝，再解析实际来源和数据；T第219行把sources设为一旦调用就报错，确证关闭门在来源/data/CUDA/输出之前生效。

未来用户正式批准、实际源/数据元数据绑定、旧H实物、前序链完整退出和现场资源仍是未完成前提；不把它们称为隐瞒，也不因CPU PASS替用户开启正式门。

## 十、CPU原件与可复用证据

| 批次 | 原件状态 | 本轮处置 |
|---|---|---|
| CPU01 | 6项PASS | 保留；原11秒另补3秒尾部 |
| CPU02 | 8项中汇总fixture错误，FAIL | saved[identity]误写顶层；修为saved["runs"][identity]，原FAIL与来源保留 |
| CPU03 | 8项PASS | 同一修正后通过，13秒 |
| CPU04 | 9项中实际确认入口因CUBLAS配置拒绝，FAIL | 未绕过数值门；CPU包装补环境变量，14秒保留 |
| CPU05 | 当前9项PASS | 33.339328592秒，向上取整34秒；当前58来源绑定 |

总账：

~~~text
(11+3)+(11+3)+13+14+34 = 89 / 1200秒
~~~

CPU05峰值记录830844928字节。CPU01/02原计时遗漏来源/版本捕获，accounting按回执与stdout时点差加时间戳余量保守补记；原件没有追改。CPU03起这些步骤在计时内。

除了真实子进程与手写模型，测试还辨别错系数恢复、缺历史/教师、错域、错score SHA、缺盲格、缺整条前序、仍有前序进程、后处理覆盖训练dispatch等反例。不能凭“9 PASS”把未经覆盖的NVML时序和最终cleanup结算当已测。

可复用10月18日关于真实反馈绑定/收尾的经验，10月23日未变H运算、Memory、连续Adam、恢复与激活重算证据，以及11月1日文件扫描和健康槽续领经验。旧S/H/LOGIT三路最大形状短测不是当前三个H完整GPU运行；旧OOM及失败不因本轮静态接受而消失。6小时单条是硬限，不是完成时长保证。没有必要由此追加全部GPU组合或新原生资格。

## 十一、实际阅读范围

本次审查包含主审与三个并行分审。未将132份字节核对冒称132份全文语义复审。

**完整语义读取：**

- 本轮scope、implementation、request、正式门及CPU05源绑定；accounting/budget/receive和CPU01—05 stdout、receipt及分项原件。
- 四个history_optuna模块全文、72行CPU入口、464行新测试。
- 方法直接依赖全文：continual_five、continual_five_run、optuna、record_replay、positive_ranking、g_half、continual_population。
- 调度直接依赖全文：formal_queue、formal_wait、risk_study。
- 包内10月18日Optuna外审及主审、10月23日实现外审及主审/资格处置/资源投影、11月1日formal_chain主审。
- CPU01—04相关旧控制器、CPU入口和测试以全文静态diff对照当前版；重复分项原件与CPU05还核对了逐文件字节一致性。

**按真实调用链定向读取：**

- five_evaluate的access、metric_set、公开确认群和field_terms；
- comparison_evaluate的校准、完整格门和interval；
- population_evaluate的22列、分类/检索/group_metrics；
- balanced_pooling_run的query_metrics和undefined_metrics；
- expression生成器的随机ID/身份/文本路径、validator关系核验、public loader；
- population_data的Group/align_labels/Memory，Chinese base的分区和模型输入，bge_continual的数值设置/学习率/Supply；
- replay_diagnostics的RNG/模式中性上下文，以及手写TinyEncoder/独立池化/排序测试依赖；
- native02原件中的实际操作/资源/恢复/耗时/账务字段；重复source/pretrained档案数组未逐字重读；
- observation/predecessor_process的固定命令、身份和时点信息。

未对导入闭包中的L²R/GCR/其他旧方法源码重新全文语义审计；本轮无新矛盾指向它们。政策/词库等未变载荷的字节身份已核对，未重新审阅全部词条。没有重新通读原论文、重做文献选择、重审旧45条全部实验或此前H/S结果包。

完整逐文件阅读级别和身份另列于同包reviewed_files.json；输入核对见input_verification.json。没有本轮科研结果，也未验证现场H27、数据实物、GPU吞吐/峰值、完整三H长期并发或前序链当前状态。

## 十二、建议与不应扩入的工作

合理但不阻断科学定义的说明：

1. 明确旧dev无本轮bootstrap区间、新确认才有；写清pooled输出层级。
2. 写明当前监督系数1与全局梯度裁剪并不矛盾。
3. 新主门只支持最终RAW MAP方向，F和取得水平独立解释；保留旧H−S失败。
4. 修R01/R02后更新本轮实际来源和受影响证据，不把原CPU05自动改称修后版本通过。

OUT_OF_SCOPE_OVERDESIGN：

- 新增候选、seed、全六顺序、HPO维度或算法臂；
- 为本次两处资源/记账修订增加GPU科研运行、所有共存组合、真实模型资格或数据生成；
- 再建评价器、通用调度/审批/恢复/账务平台；
- 重读owners/正式标签、全历史回归或无指向重算；
- 为最终MAP主门追加事后O/F成功门，或将全部权重要求永久保存。

当前可接受方法、反馈和评价链的静态结论；正式实现封板仍需主执行者独立处置HOPT-R01与HOPT-R02，并保留全部原证据。后续是否实施、核验和正式运行，依用户已有及将来明确授权决定。
