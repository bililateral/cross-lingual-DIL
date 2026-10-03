# LOGIT0.25 源码、历史目标和独立手写参考审查笔记

本笔记是联合结果外审的一项分工，审查者任务名 `logit_source`。只审读附件中的源代码、合同、既有小型结果和手写 CPU 捕获向量；没有连接服务器、安装依赖、加载模型/权重、读取正式文本/标签/缓存正文或 test/owners，也没有重新训练。本笔记不独立裁定其他分工负责的全部 22 指标、5000 次区间和五组 23 项。

## 1. 本分工结论

在实际读取和复算的范围内，没有确认 LOGIT0.25 的科学阻断或当前项目实现缺陷。实际 LOGIT 来源根是 `project/reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/`，下文代码行号均指此根。不能拿包根 ER0.1 的旧函数替代它。

LOGIT 更新的实际表达为：

`BCE_c + Rank_c + 0.5 Hard_c + 0.25 BCE_h + 0.25 Rank_h + 0.125 Hard_h + 0.5 mean_378((z_h−target)^2)`。

历史 hard 的最终系数为 0.125，MSE 的系数为 0.5。当前群与历史群各一次训练模式群前向，历史监督与 MSE 使用同一历史预测张量，累加两次反向得到的梯度后只执行一次全局裁剪和一次 AdamW 更新。不存在额外除以 1.25 或 MSE 再乘 0.25 的路径。

只以给定 fitting group IDs、种子协议和独立 Python Random 实现，已重建六段共 1728 个当前群和 1728 个历史群序列，逐项等于新 LOGIT 日志和包内首轮 ER 日志。三个阶段二缓存分别有 4、2、0 个首域幸存群和 2、4、6 个新增群；旧幸存目标摘要保持不变，新增目标的阶段和模型摘要均对应本 LOGIT 路径阶段二终点，12 个新目标摘要均不同于旧 LOGIT1 的对应新目标摘要。

**证据边界：** 目标向量和正式权重未提供且禁止请求，本网页不能亲自把原生模型重新前向生成正式 target，也不能检验真实训练的每个参数梯度。上述目标结论由真实源代码因果顺序、保存的来源/摘要/阶段消费记录和独立 ID 重建共同支持。逐元素原生梯度增量验证仅存在于原生程序执行后保存的摘要与最大差，网页不能凭摘要冒充再次逐元素计算。

## 2. 来源和阅读覆盖

本包未见单独同名 `freeze.json`；本次正式作业的 `run/manifest.json.source_files` 完整保留 34 项冻结来源，policy/decision 和实际来源全在包内。未提供某个历史 freeze 文件名不等于来源失配；独立程序逐项核对这些实际字节/大小/SHA。

相对 ER0.1 的 31 项来源，有 24 项相同、7 项修改、3 项新增：

- 修改：两个 ER shell wrapper、`step28_er_weight.py`、`step28_er_weight_run.py`、`step28_er_weight_check.py`、`step28_er_weight_evaluate.py`、`tests/test_step28_er_weight_contracts.py`。
- 新增：LOGIT 合同、policy、decision。

本 agent 全文审读上述 10 项以及 4 项共同直接依赖：`step28_bge_continual.py`、`step28_alias_ranking.py`、`step28_continual_population.py`、`step28_continual_population_data.py`。其余 20 项经逐字节来源记录相同确认后，明确复用 ER 来源分工的全文阅读责任；没有声称两套相同大文件被此 agent 再次全文读完。另按实际需要读取了共同 checkpoint/restore/原首域 target 生成链、base score、handmade fixture 的准确范围。

还全文阅读了根 REQUEST、AGENTS、科研纪律、LOGIT 实现外审原文 513 行、LOGIT 主审全文、LOGIT 当前结果报告 212 行、当前原生 CPU stdout/stderr、正式作业原始 resource/start/end 记录。没有把旧外审报告内“尚未运行”形成时快照当成当前正式实验未完成。`AI_RESEARCH_HANDOFF.zh.md` 不在附件内，当前范围以用户和根 REQUEST 为准。

逐文件完整路径、大小、SHA、全文/部分/跨 agent 复用状态在 `read_coverage.json`。独立程序访问的是小 JSON 和 source，记录在 `results_v2/read_accesses.json`；`source_correspondence.json` 为实际 LOGIT 路径与包根文件对应。与包根同字节的 25 个文件包括额外放在根 docs 的 LOGIT 合同，因此这一“25”不是与 ER31 同源的“24”。

## 3. 输入、目标、梯度、更新：实际代码链

|对象|证据位置|判断|
|---|---|---|
|确定研究干预与对手|LOGIT 合同全文；policy:7–24,62–100,132–174；decision 全文|仅新 LOGIT 两阶段，0.25 完整历史监督 +0.5 MSE；对手 quarter 和 seq，没有 tenth 路由|
|群监督与供给|`step28_er_weight.py:160–168`；`step28_bge_continual.py:65–100`；runner:192–211,240–252|由 dispatcher 供给当时当前域 48 fitting/12 calibration；恢复路径注册为阶段二，不重新请求阶段一 fitting|
|base 与 hard|`step28_alias_ranking.py:89–129`；base objective 由 ER 分工全文审读|378 无向边 BCE；28 查询 rank；每查询已知负例 top5，索引 detach、所选 logit 活图反传|
|合法 target|`step28_er_weight.py:178–187,204–208`|target 必须 NumPy float32、378 长、有限；由 `torch.tensor` 生成常量目标，不保留 teacher 参数图|
|训练与系数|同文件:191–215|先 `model.train()`；每群独立 seed；当前完整 loss backward，历史 `weight*base + logit_weight*MSE` backward，共用历史预测图|
|梯度累加/裁剪/更新|同文件:192,209,223–228|zero 一次，两个反向，clip 一次，Adam step 一次；日志 total 用保留标量还原，各项 float32 分解须另列数值容差|
|阶段学习率|`step28_bge_continual.py:50–53`；update:188–190|29 步 warmup 到 1e−5，后续 259 步线性衰减，第 288 步 encoder LR=0，head=1e−3|
|参数组和全状态|`step28_continual_population.py:14–40,157–210`|对称特征 `abs(u−v),u*v`；AdamW encoder/head 两组；保存/恢复完整 state_dict 和优化器 moments/groups/scalars|

全局裁剪和 AdamW 使参数变化不能简单解释为历史更新缩小四分之一。阶段最后一步 encoder 有有限非零梯度但 LR=0、参数不变是设计行为；head 仍更新，不能将这一点当作训练失败。总梯度范数不提供当前/历史梯度冲突的夹角证据。

独立更新程序只构造数学小例，并未以假的模型/优化器冒充真实 PyTorch 更新。原生实际更新行为的支持来自上面实际代码、原先已审微型测试、当前原生 CPU 原件和正式 trace，由不同证据分别说明范围。

## 4. 首域恢复与 target 的生命周期

原共同首域来源链是 `step28_bge_continual_run.py:426–440`：首域训练 → checkpoint 保存完整 model/Adam/RNG/校准映射 → 无参数更新地建立 ER 与 LOGIT 缓存。LOGIT target 经 `ranking.score` 到 `base.score:223–230`，这里明确 `model.eval()` 和 `torch.inference_mode()`。因此来源是共同首域 eval 模型，不是某个已走完第二域的旧 LOGIT 模型。

新恢复链 `step28_er_weight_run.py:201–206,218–245` 先验证原首域 checkpoint/缓存记录，`prior.restore_branch` 真实实现（`step28_bge_continual_run.py:230–241`）载回 model、Adam、RNG，要求 Adam288，再在新路径对 60 群完整首域盲分数精确回放。复用首映射；LOGIT 初始缓存成员要求与 ER 相同、origin 都为 1。全状态恢复保留 state_dict buffers；打分采用 eval，不以训练 dropout 生成 target。

在新阶段二终点，runner:249–277 先 checkpoint、校准/保存，再 retain。`Memory.retain`（`step28_bge_continual.py:130–154`）只维护当前存活群，删除被淘汰引用，新增最终保留群才调用 score；旧 target 数组对象不重算，新的 float32 row 另 copy 保存。`Memory.draw:164–169` 返回复制的 target，避免更新函数改写缓存。`to_bytes/from_bytes:171–208` 包括所有 target、origin、随机状态、first_map、auxiliary；完整长度受 1MiB 限制，并检查往返字节一致。

当前保存记录的独立核对：

|顺序|首域幸存/新增 target|阶段三可用群（A,B,C）|目标链|
|---|---:|---|
|ABC|4 / 2|4,2,0|4 个旧 hash 不变，2 个新 hash 对应本路径阶段二模型|
|BCA|2 / 4|0,2,4|2 个旧 hash 不变，4 个新 hash 对应本路径阶段二模型|
|CAB|0 / 6|6,0,0|无首域 C 幸存；6 个新 hash 对应阶段二 A 模型|

CAB 自然淘汰所有首域群是这次随机 Algorithm R 的结果，不是缺陷，不应事后补平衡历史群。此事实会限制本轮留存机制的解释，但不改变冻结配对比较。`target_and_schedule_results.json` 保留具体 ID/hash 对应和 origin；没有缓存正文。

## 5. 独立数值参考

成功程序 `independent_logit_reference_v2.py` 不导入项目代码、不依赖 Torch。它对手写四个三元组和八个二元组关系构造 28 账号、378 边、20 正边，分别写出 BCE、rank、top5 hard 标量及解析梯度。只把该明确手写关系用于原生 CPU fixture 捕获的分数，不推测任何正式标签。

### 5.1 原生手写 CPU 保存分数的独立公式复算

|数值|网页 NumPy 独立值|
|---|---:|
|当前 base|4.3014722373463075|
|历史 base|4.302146913200568|
|ER0.25 总目标|5.37700896564645|
|LOGIT 原始 MSE|0.0001965899888976513|
|LOGIT 总目标|5.377107260640899|
|LOGIT 记录总目标|5.377107209948008|

两个总目标的最大差约 `5.0699e−8`；所有检查项最大差 `2.751858265703788e−7` 来自当前 rank 的 float32/float64 计算差。MSE 最大差 `1.209871485223675e−11`。不能把其他审计“跟踪指标最大差 7.11e−15”用于描述这些 float32 目标误差。

### 5.2 独立梯度公式的手写有限差分

使用独立、无并列的合成 current/history/target logit，解析

`∂L/∂z_c = ∂L_base/∂z_c`，

`∂L/∂z_h = 0.25 ∂L_base/∂z_h + (z_h−target)/378`。

18 个坐标的中心有限差分最大绝对差 `8.289082562701333e−10`；再让 7 个手写线性参数同时影响 current/history，7 个参数中心差分最大差 `1.052762321762657e−9`。把 MSE 系数错设为 0.125，logit 梯度最大差为 `0.0033223176895629094`，明显大于参考数值误差。完整数值与坐标在 `independent_loss_results.json`。

这证明本独立公式能分辨本轮最关键的系数错误，没有再次运行实际 BGE Jacobian 或 AdamW；后者不在本网页允许的重算范围。

## 6. 正式评价代码的路由审读

`step28_er_weight_evaluate.py:31–76` 明确只创建 seq/quarter/logit_quarter 三个视图；比较顺序由 policy 固定为 LOGIT−quarter、quarter−SEQ、LOGIT−SEQ。LOGIT 分支只输出两个比较检查 flag，没有调用 `select_configuration` 自动替换 ER。ER0.1 是否在另一合同入选不会改变此处对手。

runner:294–422 的盲门除了 source/hash，还检查三恢复起点、六阶段、每段 288 行、配对日程、MSE/历史权重/总目标、Adam/LR、target origins 和消费缓存；读取六点各四数组，当前域 12 群校准、60 群盲分数和两种仿射输出都要完整。正式 `execute:505–513` 先盲门→before_valid→唯一标签收集→18 套指标/计数→统计。finalize:131–162 复用 shared/SEQ 27 套和 ER0.25 后两阶段 18 套共45套，再做统计。完整 bootstrap/指标数值由联合审查其他独立分工负责。

当前结果报告明确保留相对 SEQ 的 N/Z MAP 等观察负差，写明跨零不等于通过非退化守卫，且没有用 23/23 对 ER 取代对 SEQ 的 16/23。它将 FDR/DER 类适配作为已知机制、把 dropout 方差作用计入解释，不声称新算法。这里对报告措辞的判断不替代其他分工对这些数值的独立确认。

## 7. 原件、失败与范围分类

本分工原生 CPU 的执行只是阅读附件证据：2026-10-02 15:37:17—15:44:20，原 stderr 中 28 个测试记录和 BGE API FutureWarning；警告不等于失败。`step28_er_weight_check.py:34–39` 明确每臂只有一次真预热后将 Adam counter 置288，再一次真加权更新，共4次；不能说完成原生288次预热。实际参数梯度增量 reference 针对两个指定参数张量，记录的最大差不是全参数逐元素独立证明。

本网页程序执行记录位于本目录 `runs/`：

1. `logit_source_independent_v1`，退出 1：错误假定旧 ER0.25 的六份 `run/updates/*.json` 也被附上。首份读取产生 FileNotFoundError，保留源码及原 stderr。旧 manifest 仅留这些日志的 path/bytes/SHA，不含日程正文。
2. `logit_source_independent_v2`，退出 0：将该特定旧日志直比标成未验证；仍独立重建并核对全部新 LOGIT/首轮 ER 序列。没有伪造旧日志、改变项目或请求新数据。修订 diff 为 `v1_to_v2.diff`。
3. `logit_source_coverage`，退出 0：生成阅读覆盖和源版本对应。它是证据整理，不计成科研数值测试。

每条记录包含原 argv、环境线程设置、允许 CPU/选定 CPU、完整 stdout/stderr、UTC 起止、退出码及实际源快照；NumPy2.3.5、Python3.12.14 的网页证据与原生 Python3.10.19/Torch2.9.1+cu130 证据分开。所有输出留在此分工目录，不改附件。

|类别|本分工裁定|
|---|---|
|SCIENTIFIC_BLOCKER|未在所审 LOGIT 干预、恢复、target 生命周期、ID 随机供给及手写公式中确认。最终联合裁定需合并指标/统计和执行证据分工。|
|REPRODUCIBILITY_DEFECT（项目）|未确认必须修改当前科研实现的问题。旧日志正文未附限制的是对旧 ER0.25 的直接逐行重验，不是新实验配对代码的已证伪缺陷。|
|REPRODUCIBILITY_DEFECT（本审查工具，已修正）|v1 错误假定旧日志在包内；已在 v2 明确边界并保留失败和修订原件。|
|未验证范围|未重算正式模型输出/target、正式监督指标标签、逐元素原生参数梯度和权重恢复；旧 ER0.25 六份旧 trace 正文不在包中，不能声称直接读过。可结合已附原审查/主审、源因果链、共享起点、现有完整新 trace 判断，不新增索要材料。|
|OUT_OF_SCOPE_OVERDESIGN|不增加参数搜索、seed、教师/特征缓存、再训练、标签访问、系统加固或新成功线；不为 source 文档命名和历史缺件设置新审批流程。|

本分工只建议最终交付准确区分上述证据边界，无需为了本笔记中未确认的假设风险改动冻结实现或重做实验。
