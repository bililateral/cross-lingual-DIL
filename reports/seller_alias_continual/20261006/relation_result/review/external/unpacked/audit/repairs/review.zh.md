# 关系目标记忆结果外审：P1/P2/P3修复与实际准入来源子审

本次读取 `relation_memory_result_review` 结果包，先读context/request、前轮pilot原始审查与disposition、10月5日修复报告/CPU原件/deployment/preflight/gate，再读冻结source中的正式入口、Bash及集成测试。旧“尚未修复／尚未训练”按历史时点理解；10月5日明确用户执行指令适用于本次最小修复后同一固定候选，不要求重复批准。

**子审结论：P1/P2/P3均已在正式执行前最小修复，并绑定到本次实际训练来源。没有发现这些问题造成当前结果无效或要求重训的证据。本次有效完成应依据completion.json及其资源／访问账，不应依据纯统计evaluation.json；实际材料已经这样区分。** 本结论不代替保存矩阵、更新日程及指标复算分工。

## 已修问题及直接位置

本节行号指冻结 `source/scripts/step28_relation_memory_run.py`。

| 问题 | 实际修复与证据 | 影响及最小处置 |
|---|---|---|
| P1 纯统计可错误晋升／恢复不继承资格 | 454—457行把统计标记改为`STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION`，合取仅为`observed_continuation_checks_pass`；478—495行完整统计后先作带16KiB余量检查，核访问账，写completion后再次检查，失败则删receipt。498—527行恢复要求原job/gate/sources、合法`POSTPROCESS_IO_ONLY`失败账、28套矩阵，继承失败账和包装墙钟最大值及资源峰值；先备份失败账并写恢复进行中状态；CLI566—569不再直接finalize | 已修。当前completion独立存在并绑定实际evaluation哈希；纯统计不含worth建议，资源有效后才给worth=false。恢复路径无训练、模型或标签解析。无新增必要修复 |
| P2 失败快照与写账失败跳过停止 | 73—140行保存进度、累计耗时、RSS/CUDA/磁盘观察峰值；snapshot不调用预算断言、目录扫描或CUDA；persist使用锁和临时文件替换。143—152行watchdog无论写账成功与否都在finally调用_exit(2)。155—183行普通异常/TERM尽力保存快照，限制可恢复失败类别。243—245行在full与inference文件共存时强制量字节后再删full | 已修。没有承诺完全不可写磁盘或SIGKILL能留完整账；Bash保留外层退出与墙钟。本次实际exit0与completion/resource相符，无结果阻断 |
| P3 点号job路径漏计日志 | 116—117行使用`with_name(root.name + suffix)`，与冻结Bash12—15的字符串追加一致；恢复读取包装也用同一语义（511行） | 已修。独立原样表达式复测job.v1的两份10B日志得到20B，无遗漏 |

恢复资格的gate检查在461—475行，比对23来源、job、runtime/supervision，验证原生核心及修后集成证据。正常execute在530—556行先准入、再训练与完整盲门、最后调用complete。Bash保持新job、一次execute、外层86395秒TERM＋5秒KILL，未新增监听、训练恢复或自动重试。

## 修复没有改变固定科学对象

将本次23份冻结文件与本会话review(10)逐项比较，**只有正式入口和对应集成测试变化**。其余21项，包括核心、policy、Bash、两份合同以及评价/采样依赖，逐字节一致。入口diff所见是预算/状态/异常路径、进度、临时checkpoint共存量尺和状态名称；`comparisons`的六项公式、真实训练目标、参考配对及继续条件未改。

核心仍为`4d2fc05c945470a0e22fd563eed93e6b346904a7c1138cb0029406f09676b044`；本次原生证据记录的核心身份相同。正式入口为`213119e1205ed7cc765cc44f8a6fb8c3822198da25a28ca99a7d81b29a0bc094`，集成测试为`90b8ab990a6b2d4a7f7dd66357e8bc081b1cdac4c0de44386c90692809a580d5`。

所以本次仍是d32、epsilon=.001、当前与历史各1、s0三个到达顺序共2592更新的候选，不能把入口修复说成改出另一算法，也不需要因入口改动重跑未改的原生BGE一步。

## 来源绑定及实际完成证据

网页独立标准库脚本重新计算source中23项大小与SHA，和以下六处完整来源集合逐项相等：gate、deployment、修后CPU/result、job/execution、collected、evaluation。gate实际文件的5,478字节及SHA又与deployment及job/execution的记录一致。gate指向native、修后CPU、主审及前轮外审四份原件的大小/SHA均吻合。身份相等只是证明运行证据对象相同，不替代源码和数值审查。

实际资源preflight为10月5日15:05:48，记录GPU0空闲32,063MiB、host available263,556,362,240B、无GPU计算/项目作业；正式job/execution保存了启动时再次核验的资源。15:08:27的执行观察已经处于ABC第一阶段第27步、CPU affinity=[0]、train=1/valid=0，说明这是启动时观察而非结果后的伪完成摘要。

本次实际完成材料：

- `job/evaluation/evaluation.json`：`STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION`，不存在`worth_matched_replay`，有`observed_continuation_checks_pass=false`。
- `job/completion.json`：`COMPLETE_RELATION_FIXED_POINT`，2592更新、train=1/valid=1/heldout=0/owners=0、`worth_matched_replay=false`；绑定evaluation实际377,530字节及SHA。
- 包装：2026-10-05 15:06:32至18:57:22，exit0，完整墙钟13,850秒。completion记录13,849.124秒，resource末次13,849.126秒；口径区分清楚。
- completion观察峰值：产物15,679,024,129B，RSS6,900,240,384B，CUDA allocated8,465,105,408B、reserved9,367,977,984B，均低于约定上限。
- 当前job交付未见failure/recovery回执；没有发现经过预算失败再以纯统计状态晋升的材料迹象。

## 修后CPU覆盖及网页最窄独立检查

修后LinuxCPU原件记录3项通过、0失败/错误/跳过，57.853513秒，包装59.59秒、最大RSS855,796KiB，2592小模型更新。测试24—127沿真实手写分发、九端点保存恢复、盲门、校准、28套矩阵，新增无剩余预算拒绝和有余量统计恢复。测试128—148覆盖点号路径、超限后快照、失败账再次写错仍停止及TERM留账。

保存替身在测试44—50先真正保存，再扰动一个参数、一个Adam moment及Torch RNG；生产checkpoint220—232随后实际加载full/inference，依赖的restore_state193—209对加载后model/Adam摘要比对，restore_rng65—73实际set并比对恢复后的CPU/CUDA RNG。这里不是仅把same-object再次传入no-op恢复。该CPU夹具明确替换公共输入/标签、BGE载模与预训练身份，以及恢复例的gate资格提供者；没有执行真实APPROVED gate/GPU preflight，不扩张成正式数据训练。

网页本轮没有导入Torch、项目runner或真实模型。为隔离前轮真实故障点，从冻结源码AST提取**未修改**的complete/watchdog函数体，用显式手写预算/写盘异常注入，仅检查控制流：

1. 纯统计给observed=true，但receipt前资源检查拒绝：不会产生completion。
2. receipt写入后末次资源检查拒绝：已有completion被删除。
3. 正常注入路径才签发完成并persist。
4. watchdog记失败账时再次OSError，仍观察到_exit(2)调用。
5. 直接执行冻结Budget.check中的原样paths表达式，job.v1两份日志被正确计入20B。

这些是标准库控制语义核验，不证明实际资源测量本身，不是项目Linux/GPU或完整三测重跑，也不把前轮网页核验冒充修后执行。源码及实际输出在本目录两组`.py`/`.json`中保留。

## 分级裁决

- **已修：** P1/P2/P3，及原生核心身份与本次来源绑定。
- **真正结果阻断：** 本子审覆盖范围内未发现。
- **当前交付复现缺陷：** 未发现影响上述修复或完成资格判定的缺口。无法网页重跑真实模型/缓存并非本次要求，也不能借此补做标签读取或新训练。
- **非必要后续建议：** 本轮无需另建审计平台、扩大回归或重跑未改核心。后续若研究其他配置，再按新授权审查；不影响本固定点的结果接收。

低秩盲区、迁移近似、单步不代表全程，以及整套配置收益不能自动归为创新，继续沿用。这里核实正式计算有资格被接收，不能由此宣布效果良好；效果方向由保存矩阵的独立复算判断。
