# 匹配ER0.25的SEQ／ER／LOGIT：合同与最小实现独立外审

**审查日期：2026年10月2日。范围：收到的冻结源代码、直接相关历史证据、网页Linux CPU手写/微型验证。不是正式训练结果审查。**

## 1. 结论、模型身份与原提交身份

**结论：在本次实际审读与CPU验证覆盖范围内，没有发现需要修改当前提交后才能进入既定主审／必要原生CPU核验的SCIENTIFIC_BLOCKER或项目REPRODUCIBILITY_DEFECT。合同和当前最小实现正确表达了“在匹配ER0.25上增加独立0.5原始logit MSE”的比较。** 这个结论不等于原生BGE核验已通过，不等于正式六端点结果有效，更不等于LOGIT会胜过ER或SEQ。

指定的28个不同unittest全部实际通过，无跳过；两个CLI `--help`与两个Bash `bash -n`均退出0。五个最终独立参考程序通过，包含非生产损失公式构造的解析导数、三分量梯度／统一裁剪／手算AdamW、完整实际微型288步、39种六点门错误注入、实际域条件bootstrap及先保存后恢复。审查者自己有两次参考程序失败，完整失败源码、原始流、诊断和修订均保留；没有据此修改被审项目。

### 1.1 能报告和不能认证的身份

本会话可报告的模型名称为 **GPT-6 Astra Pro**。没有暴露可供独立验证的后端模型标识，因而**不能认证这次实际后端就是用户指定的“网页GPT 6 Pro”**；模型自述、历史报告中的名称以及页面选择器都不是独立后端证明。本报告不以“指定型号已获认证”为有效性前提。

### 1.2 确实收到并打开的原包

|项目|独立实测|
|---|---|
|原文件|`logit_review.zip`|
|大小|**2,084,038字节**|
|ZIP成员|**304个，名称无重复**|
|SHA-256|`ff678c2f9687ced6fa9888eca55064ca617c98d35bf371e082e7cf321f9ed08a`|
|解压载荷合计|6,193,896字节|
|ZIP CRC|通过；未发现错误成员|
|审查结束时原ZIP、保留副本|均与收到的字节完全相同|
|解压提交|304个文件全部仍与ZIP内原字节相同；额外文件0；项目改动0|

收到的原ZIP原样保存在 `original/logit_review.zip`；全部成员位于 `submission/`。`evidence/received_members_304.json`含每一原成员的大小及SHA。`evidence/identity.json`保存初次核验；`evidence/final_integrity.json`保存末次核验。以下代码位置均指 `submission/` 内原始文件，未通过行号重排或格式化修改。

## 2. 从用户问题界定本次实验，而不是从测试PASS倒推问题

待回答的问题是：在有限历史训练群供给、相同原首域状态、相同后续群与随机流的条件下，历史完整监督系数固定0.25，加入独立系数0.5的历史原始logit MSE，能否改善旧分布留出卖家的候选排序，同时满足原冻结的基础识别和新域保护观察条件。新到达的是不同卖家群体，不是跟踪同一控制者在三个阶段的同一身份；每群28账号、每查询27候选的中文合成任务，不外推为真实市场效果。

本轮只新训练LOGIT0.25，复用SEQ和ER0.25。三个顺序ABC/BCA/CAB都用s0，续训第二、第三阶段，每阶段288更新：`3×2×288=1728`新更新，每更新当前/历史各一群，合计3456群梯度呈现。六个新训练端点；每端点12群训练内校准原分数、60群valid原分数及两种校准valid分数，共24个分数数组；由后面18套新指标/计数配合45套已有指标/计数完成三组比较。

主比较是LOGIT0.25−ER0.25，沿原23项；ER0.25−SEQ和LOGIT0.25−SEQ必须各自完整报告。**“相对ER为正”不推出“相对SEQ为正”；不自动替换ER，不调λ、MSE或seed，不以赢SEQ为完成条件。** 结果好坏均结束，不能把本轮诊断包装为原创算法、最优超参数或多种子稳健结论。

已阅读本轮 `decision.json` 的完整确认、合同、policy及实现说明。既定顺序是low独立任务先正式开始、本轮外审及主审后做必要CPU核验、等low释放资源后部署正式任务；没有新设同范围启动审批，也没有调用正式入口验证这个顺序。

## 3. 清单、冻结身份、最小差异与历史承接

### 3.1 清单和当前34来源

`reports/documentation/20261002/logit_weight/review/source_inventory.json`列出303项载荷，自身不自哈希，故与ZIP304成员不矛盾。独立逐文件检查303项大小和SHA全部相同，无遗漏、无额外载荷。

当前有效冻结为 `reports/documentation/20261002/logit_weight/freeze.json`，其34项来源全部与实际文件一致。实际执行 `step28_er_weight.sources(p)` 得到的34项，按唯一路径比较完整记录，也与freeze一致；初次对列表顺序作过强断言的审查者错误已在第12节单独记录。当前policy SHA-256：

`2aaf41724055ed3dff93ad65a8be2cc4e83c8f35d4515dace34bcdc44b081deb`

原首轮18个来源与原manifest仍完全一致。旧low的31项freeze和旧weight的28项freeze是历史字节身份，不要求与修改后的当前七个文件相等，也不将这种有意差异当作污染。原baseline的5项钉住记录与weight_reference的5项钉住记录全部独立核对大小/SHA一致，包括各自execution、manifest、partition、collected和evaluation。

用户说明Linux正在执行的low仍用旧31来源、本地新代码未同步。这属于用户提供的部署状态；本次没有连接项目Linux，**没有重新观测其当前PID、进度或完成状态**，更没有把新字节覆盖到正在运行的环境。

### 3.2 实际diff，而非只检查文件名

`independent/diff_reference_v1.py`逐hunk检查 `implementation.diff` 的新侧内容与当前文件相同，然后独立逆向还原旧侧七个文件。七个旧文件的大小与SHA均精确等于历史low冻结的旧字节。当前三项新增冻结来源是本轮合同、policy、用户决定。

|既有文件|修改前字节|当前字节|逆向还原与low冻结旧字节|
|---|---:|---:|---|
|`scripts/run_step28_er_check_linux_20261001.sh`|1127|1133|完全一致|
|`scripts/run_step28_er_weight_linux_20261001.sh`|1412|1418|完全一致|
|`scripts/step28_er_weight.py`|11761|13610|完全一致|
|`scripts/step28_er_weight_check.py`|12543|15227|完全一致|
|`scripts/step28_er_weight_evaluate.py`|10859|11503|完全一致|
|`scripts/step28_er_weight_run.py`|31753|36238|完全一致|
|`tests/test_step28_er_weight_contracts.py`|31069|43884|完全一致|


提交声称的基点为 `646ed9e226ab2e5993f762ac1014201841873cfc`。包中没有该Git对象，所以能认证的是“实际新侧差异及还原旧字节与既有冻结吻合”，**不能独立认证Git对象本身**；没有为此要求Linux Git或增加仪式。还原文件和逐文件新旧大小/SHA位于 `evidence/diff_reference/`。

### 3.3 直接相关历史外审全文及处置

|已完整阅读的历史外审/处置（submission下）|行数|字节|
|---|---:|---:|
|`reports/documentation/20260930/continual_plan/implementation/review/external/REVIEW.zh.md`|328|31103|
|`reports/documentation/20261001/er_weight/review/external/REVIEW.zh.md`|288|33574|
|`reports/documentation/20261002/er_low/review/external/REVIEW.zh.md`|298|32669|
|`reports/seller_alias_continual/20261002/er_weight_result/review/external/REVIEW.zh.md`|519|52410|
|`reports/documentation/20261001/er_weight/review/report.zh.md`|52|8646|
|`reports/documentation/20261002/er_low/review/report.zh.md`|36|7162|
|`reports/documentation/20261002/er_low/review/primary_check.json`|16|785|
|`reports/seller_alias_continual/20261002/er_weight_result/review/report.zh.md`|24|5026|


这些材料确实按全文读过，不把历史执行者PASS替代本次计算。历史外审者的标量/数组哈希混用、NumPy布尔JSON序列化、错误预计计数、F公式手写值错误、假设不存在的字段、fixture阶段/RNG错误等，均按附带最终修订和主审处置保持关闭，不重新列为本次项目缺陷。

本轮实现说明引用的一份首轮结果主审路径 `reports/seller_alias_continual/20261001/bge_continual_result/review/report.zh.md`不在收到的包内。本报告不声称读过这份缺失文件；已使用包内可见的原始小型结果和现有完整相关外审核对承接，不因此要求重开旧审查或补做正式实验。

## 4. 实际损失及三分量更新：独立验证的核心

令某完整群的目标为：

`L_base = BCE + Rank + 0.5 Hard`。

BCE对28账号的378个无向配对取均值；Rank对28个查询取均值，分别是27候选的log-sum-exp减该查询真实正例logit均值；Hard对每查询已知负例中当前分数最高的5个，计算与真实正例的softplus差并平均。选择top5索引不反传，但被选中的实际logit值参与反传。手写群保持20正配对、358负配对及原查询正例结构，不拿随机二值标签破坏关系约束。

本轮目标应展开为：

```
L = BCE_c + Rank_c + 0.5 Hard_c
  + 0.25 BCE_h + 0.25 Rank_h + 0.125 Hard_h
  + 0.5 × mean((z_h(train, ξ_h) − target_h(eval, frozen))²)
```

所以历史hard的最终系数0.125是正确的，**MSE的系数0.125则是错误的**。当前目标、历史目标各自按原群内规则归一化；没有再除以1.25，没有只给历史BCE加权，没有把MSE塞入整个历史目标后再乘0.25。

### 4.1 源码计算链

|核对事项|实际代码位置|审查结论|
|---|---|---|
|原BCE/Rank语义|`scripts/step28_chinese_base.py:165–191`|原378边/28查询归一化保持|
|Hard及完整base目标|`scripts/step28_alias_ranking.py:89–129`|top5已知负例；完整BCE+Rank+0.5Hard|
|入参、系数、target拒绝|`scripts/step28_er_weight.py:171–190`|合法群、阶段、Adam步数；float32的378个有限target；错误输入在zero/forward/step前拒绝|
|当前与历史前向|同文件`:191–206`|各一次群logits前向，分别使用配对的dropout随机种子|
|完整历史权重与独立MSE|同文件`:203–212`|先0.25完整历史监督，再对同一历史predicted张量加0.5MSE；不多做一次历史群前向|
|无梯度target|同文件`:207–210`|由NumPy缓存新建不带梯度tensor；target不成为可训练变量|
|累积、统一裁剪、一次更新|同文件`:212–237`|当前反向后历史组合反向，累加梯度；一次全局clip，一次AdamW.step|
|学习率|`scripts/step28_bge_continual.py:50–54`|encoder前29步warmup后线性衰减，阶段288末步为0，head始终0.001|

群级“两次forward”不是声称原生编码器内部只执行两个微批或没有激活检查点重算；证据针对的是当前/历史群logits计算接口是否重复调用，恰好对应MSE不需要第三次历史群计算这一要求。

### 4.2 非生产公式构造的解析参考

`independent/math_reference_v1.py`没有用生产objective生成期望梯度：独立NumPy实现BCE、Rank、top5 Hard及对378个logit的解析导数，再做六个坐标的有限差分。期望参数梯度使用这些独立logit导数穿过实际微型网络作VJP；网络和自动微分仍是PyTorch，并非另写第二个网络/自动微分系统。AdamW的矩、偏差修正、decoupled decay和参数更新则另外以NumPy float64计算。

手写logit参考：BCE=0.8410926306012743，Rank=4.233575500295191，Hard=2.0271363121562502，完整base=6.08823628697459。对应生产double精度损失值误差为0，解析logit梯度最大误差`1.3877787807814457e-17`，有限差分最大误差`6.576792608778415e-10`。

实际微型当前base=4.326027433472308，历史base=4.325096748310743，MSE=0.46326616545095045。由三项独立组成的总目标=5.638934703275469，实际记录=5.638934642076492；差异来自float32实际路径，非系数变化。

|场景|总目标参考值|实际记录总目标|裁剪因子|裁剪前梯度最大误差|裁剪后梯度最大误差|
|---|---:|---:|---:|---:|---:|
|unclipped|5.63893470327547|5.63893464207649|1|2.23517418e-08|2.23517418e-08|
|active_clip|5.63893475904624|5.63893461227417|0.0262757126796|8.94069672e-08|5.29934525e-10|
|last_step|5.63893470327547|5.63893464207649|1|2.23517418e-08|2.23517418e-08|


每种场景实测调用数均为：群forward=2、zero_grad=1、clip=1、Adam step=1。主动裁剪场景的因子为0.026275712679581755；将MSE错设为0.125，未裁剪梯度会偏离正确值约0.17587955，明显高于实际误差，因此参考能分辨关键错误，不只是“两个实现同时写错而一致”。

手算AdamW与实际float32更新相比，跨场景参数最大误差`1.852763937826296e-7`，一阶矩最大误差`2.3283064365386963e-9`，二阶矩最大误差`4.8152308692022555e-12`；阶段末步encoder LR=0且参数不变，head仍改变，Adam计数到576。参考容差是数值核验容差，不修改原23项零容忍的科研判断条件。

这里为检查末步和偏差修正，先做一次真实微型warm update，再显式把Adam计数夹具从1设到318或575；**没有冒称实际进行了318/575个历史warm更新**。此外，λ=1、MSE=0.5的旧LOGIT路径与新函数在另一个完整实际更新上，模型与Adam状态逐值/逐字节相同。

### 4.3 target确实无梯度，而非只看配置

`access_identity_reference_v2.py`捕获了实际新建的target tensor：数量1，叶张量、`requires_grad=False`、`grad=None`，传入的NumPy历史向量未改变。Inf、requires-grad Tensor和Python list三类无效reference均在模型/优化器改变之前拒绝，实际zero/forward/step调用数为0。这与提交六个新用例的无效reference测试互补。

### 4.4 对科学含义的必要限定

teacher是eval时冻结的历史原始logit，student使用train dropout；MSE包含对dropout随机性的惩罚，不应解释为纯粹的确定性函数保持。其期望可分解为均值偏离平方加随机输出方差；合同已经承认该设置，因此不是新增阻断。全局裁剪和AdamW是非线性更新过程，“历史梯度乘0.25”也不等于“历史参数变化精确变成四分之一”，不能从系数或梯度范数直接推导遗忘缓解。

## 5. 共同首域、缓存来源与真实微型续训

### 5.1 首域绑定的实际出处

原不变的 `scripts/step28_bge_continual_run.py:426–440`先完成共同首域训练并保存model/Adam/RNG/首映射，然后在没有中间参数更新的同一个状态上分别生成ER和LOGIT缓存；LOGIT首域target在该共同模型的eval路径生成。后续新分支的 `scripts/step28_er_weight_run.py:181–239`验证三个原首域full checkpoint、共同模型摘要、Adam=288、首映射和原LOGIT缓存摘要，恢复原full状态后才续第二域。它没有从ER0.25的第二域权重或旧LOGIT1第二域状态开始。

原首域实际模型和缓存向量不在本包；这里能独立确认的是源码因果顺序、钉住的小型身份记录及微型恢复行为，不能宣称已用原始BGE权重重新算出那些正式target。这个未验证范围是用户明确限定，不是要求提前读正式模型的理由。

### 5.2 第二域末target只给新保留群生成

`step28_er_weight_run.py:249–277`在本分支完成第二域更新和checkpoint后，切eval，对reservoir最终留下的**新增当前域群**生成target；原缓存幸存群target保留，不刷新。附带记录对照本分支第二域模型摘要和新增群集合，第三域消费新缓存。`step28_bge_continual.py:109–219`负责完整群、样本容量和序列化；没有从原LOGIT1第二域端点复制新target。

独立微型路径实际运行了288次新更新，观测576次群forward，Adam从显式夹具288到576。一个真实warm update之后把计数1改成288这一夹具事实有记录；后面的288步是真实执行，不是把末计数直接写成576。

checkpoint实际保存并回放12群校准和60群valid手写分数，共72群，模型/Adam/RNG状态完全一致；保存的推理checkpoint再恢复一致。缓存中4个旧幸存群`fit_A_08 / fit_A_12 / fit_A_43 / fit_A_05`的target逐值不变；2个新群`fit_B_00 / fit_B_02`由本分支第二域末eval模型生成。完整缓存124,573字节，其中辅助状态21,286字节；增加足以超过1MiB的辅助载荷实际触发拒绝。

重新序列化、恢复缓存和状态后，第三域下一次历史群抽取相同；两份微型分支实际做下一步更新后，模型和Adam逐字节相同，Adam为577。证据：`evidence/path_reference/results.json`及其微型checkpoint、缓存文件。**这里是真实一个微型阶段端点，不是六个原生BGE正式端点。**

### 5.3 群、dropout、Adam配对与随机缓存的实际分布

`train_stage`在 `step28_er_weight_run.py:43–97`复用原当前群序列，历史群从原状态恢复后的相同rng抽取，当前和历史dropout流分别按既定种子派生；三顺序各两阶段，总1728当前群和1728历史群都与原ER及ER0.25记录配对。

独立参考另写canonical JSON/SHA派生seed、六次epoch shuffle和Algorithm R，然后逐项比对；没有用生产schedule函数替代期望值。完整序列和对应dropout seed保存在 `evidence/path_reference/paired_schedule_1728.csv`。阶段末Adam分别576和864，不重置Adam。

重构出的第三阶段可用六群来源：

|顺序|A群|B群|C群|
|---|---:|---:|---:|
|ABC|4|2|0|
|BCA|0|2|4|
|CAB|6|0|0|

CAB中第一域C的历史群自然全部被reservoir淘汰。这不是随机算法错误，也不应事后平衡群分布来“修复”结果；匹配比较需保持它。它说明本轮结论受有限随机供给和单s0约束，而不是三次独立seed证据。双方容量都是6群/1MiB，不意味着LOGIT缓存字节数、wall time或eval计算与ER严格相同；额外target和生成成本属于该处理的一部分。

## 6. 新study、六点24数组门、valid访问与保存顺序

### 6.1 新study贯穿入口

`step28_er_weight.py:28–93`明确区分weight/low/logit，来源集合、policy摘要、with_logits、记录列随study改变。运行CLI和检查CLI都把`--study`传到实际函数；两个Bash也传递study，不会默认退回old weight。checkpoint、manifest、收集和统计继续携带当前policy及λ/MSE，不用历史low/weight freeze充当当前身份。

正式总入口 `step28_er_weight_run.py:454–527`只作源码审读，未执行。实际顺序为：既有审查/原生CPU证据及预算核对 → train →完整blind_gate→before_valid记录→一次valid标签解析→collect→finalize。`before_valid`沿用的通用状态名称不是它执行了旧study的证据；决定分支的是实际policy/来源/元数据传递，本轮已追踪这些值，没有提出为命名重构系统。

### 6.2 六端点门不是只检查manifest中的PASS

`step28_er_weight_run.py:294–422`要求六个端点、六份288行训练日志、三个起点和三个新缓存，核对实际source/policy、配对群与dropout/Adam、完整历史分解及总目标、每步学习率、MSE系数及缓存target来源，再读取每个端点的四个真实数组检查形状、dtype、有限性、映射关系和rank不变。

每端点四数组分别是12×378校准raw、60×378 valid raw及两份60×378校准valid；前二者float32，校准valid float64。校准映射只用当前训练内12群（4536配对、240正例），不使用valid标签；首映射继承原首域。完整六点门成功才返回18个评价角色。

独立参考建立完整六点门夹具后，分别注入39种错误，全部实际被拒绝。24种是在六点×四数组的每个数组中写NaN，并同步更新文件哈希，证明不是只检查一个旧digest；其他包括MSE误乘0.25、历史rank/hard漏加权、引用旧LOGIT1第二域target、train teacher、遗漏新增群、首缓存误用ER、旧幸存target实际值被刷新并同步修改缓存摘要、第三域消费错误reference、dropout不配对、Adam重置、末步LR不归零、缺第六点、旧study、总更新少1。各实际异常文本见附录C和 `path_reference/results.json`。

这里的六点完整性与错误拒绝使用明确的手写fake checkpoint身份夹具，不能替代将来对六个原生BGE正式文件的门核验；但实际调用的是生产门，不是另写一个只返回PASS的模拟门。

### 6.3 一次访问门与18新+45旧保存恢复

`step28_bge_continual_run.py:76–87`先记解析尝试再调用loader。独立手写loader刻意抛错时，账本已记为1，第二次调用在loader执行前被拒绝，train/development分别检验；heldout/test/owners非法split在loader之前拒绝。这里使用的是显式手写回调，本次正式train/valid标签解析均为0，没有消费项目允许的正式次数。

源码 `step28_continual_expression_run.py:58–107`的public输入只取train/development items和相应群元数据；正式标签由后面的受控attach_labels解析，未发现偷偷读取test/owners的必要路径。本次没有实际调用正式public_inputs或attach_labels。

新结果 `collect:425–451`先保存18套60×22指标矩阵及各自count，再写完整collected。`step28_er_weight_evaluate.py:116–188`先把45套钉住的既有小型矩阵/count复制保存，再做bootstrap与统计；旧45套来自共同首域/SEQ的27套及匹配ER后两域18套，不重训、不重读旧标签。

独立程序实际走完18新手写结果收集与45既有复制，在bootstrap故意失败：18+45套矩阵及count全部已经存在，evaluation尚未生成。随后只从这些保存输入恢复，将正式parser、模型加载和标签接口设为调用即失败，恢复仍成功，调用数0；重复finalize输出字节一致，全部输入摘要不变。需要的原参考小型manifest/身份记录仍必须可读；不把该恢复能力宣传成脱离所有被钉住元数据的独立结果包。

## 7. 22指标、七类端点、三组比较和原23项

### 7.1 22指标不是“全部等于排序”

实际列依次为：

`average_precision, trapezoidal_pr_auc, roc_auc, recall_at_fpr_1pct, brier, log_loss, precision, recall, f1, specificity, balanced_accuracy, mcc, map, mrr, recall_at_1, recall_at_3, recall_at_5, recall_at_10, ndcg_at_1, ndcg_at_3, ndcg_at_5, ndcg_at_10`。

AP以召回增量加权precision，不是梯形PR面积。独立四分数手写例子给出AP=5/6=0.8333333333333333，而梯形PR-AUC=19/24=0.7916666666666666，ROC-AUC=0.75，1%FPR下Recall=0.5，生产函数与独立实现一致。

固定分类阈值是logit≥0；其precision/recall/F1/specificity/balanced accuracy/MCC由保存count重算。群宏平均与跨群合并计数后再算的pooled结果不是同一个量，统计保留两种口径，没有用宏precision冒充pooled precision。正斜率校准不改变14个排序/曲线指标，校准会影响概率损失及固定0阈值分类指标。主角色使用stage-cal的概率/分类指标和raw的14个排序/曲线指标；另外raw、stage-cal、first-cal完整保存。

### 7.2 实际域到达位置正确映射

令 `R_m(t,j)`表示方法m完成第t个阶段时，在该顺序第j个到达域上、按对应实际域留出群计算的指标。独立直接公式与生产 `step28_bge_continual_evaluate.py:91–138`逐项比较：

|端点|公式|
|---|---|
|O|`(R(3,1)+R(3,2))/2`|
|N|`(R(2,2)+R(3,3))/2`|
|Z|`R(3,3)`|
|F_first|`R(1,1)−R(3,1)`|
|F|`[(R(1,1)−R(3,1))+(R(2,2)−R(3,2))]/2`|
|G|`[(R(2,2)−R(1,2))+(R(3,3)−R(2,3))]/2`|
|final_all|`[R(3,1)+R(3,2)+R(3,3)]/3`|

Brier和log_loss在F_first/F/G中翻转符号统一改善/遗忘方向，O/N/Z绝对概率损失不翻转。不能漏除以2，不能把实际域A永远当作首域；BCA和CAB按各自到达位置映射实际域。

### 7.3 Bootstrap保持既定条件范围

PCG64种子20260930，生成实际域A/B/C各20群、5000次有放回整群抽样；同一实际域的同一抽样索引同时用于所有方法、顺序、阶段和角色，然后平均三个顺序。没有把三个顺序当成三个seed重抽，也没有重训模型、重拟合校准器或重选缓存。

独立程序自行构造实际域权重场、抽样频数加权及按排序线性插值求2.5%/97.5%分位；与生产保存的draws完全一致。置信区间是**固定已训练模型、固定seed/顺序、固定valid群来源条件下的群重采样区间**，不代表训练随机性或真实市场总体无劣性。

### 7.4 原23项逐项保持，没有另立成功线

下列差值均为候选−对应控制；主比较控制是ER0.25，另外两组各有自己的控制SEQ。主角色与对照raw的额外概率保护不得混淆。

|编号|原判断条件|
|---|---|
|1|O MAP平均差严格大于0|
|2|O MAP条件95%区间下界严格大于0|
|3|O Recall@5平均差严格大于0|
|4|N MAP平均差大于等于0|
|5|N Recall@5平均差大于等于0|
|6–9|O AP、ROC-AUC差≥0；Brier、log_loss差≤0|
|10–11|O候选主角色的Brier、log_loss，相对控制raw的差≤0|
|12–15|N AP、ROC-AUC差≥0；Brier、log_loss差≤0|
|16–17|N候选主角色的Brier、log_loss，相对控制raw的差≤0|
|18–23|Z MAP、Recall@5、AP、ROC-AUC差≥0；Brier、log_loss差≤0|

独立逐一破坏23项，每次恰好对应一项失败。F/F_first/G/final_all完整报告，但不擅自追加为通过门；22指标全部报告，也不声称23项保障了所有22指标、每个域、每个顺序或总体无劣性。三个顺序的数值和O MAP正向顺序数是描述，不改写为三种子资格。

`step28_er_weight_evaluate.py:45–76`实际构造LOGIT0.25−ER0.25、ER0.25−SEQ、LOGIT0.25−SEQ三组，LOGIT分支输出两个独立判断flag而没有selection自动替换调用。

### 7.5 独立全量小型统计参考的数值证据

18套新手写分数×60群×22指标，共23,760个指标值独立复算，最大误差`3.3306690738754696e-16`。63套×60群×6个count派生指标=22,680个值，其中45套既有对应16,200个值；未重读这些既有结果的正式标签。

完整输出包括1,848个端点/角色/指标摘要、594个比较/指标摘要、三组完整23项，以及5,346行stage CSV。统计参考记录62,018次数值比较，最大绝对误差`6.661338147750939e-16`；这些是计算核对数量，不是62,018个独立科学样本。

故障恢复、独立draws、所有单群矩阵/count、全端点及全比较结果、23个逐项判据破坏名目均保留在 `evidence/statistics_reference/`。**其中LOGIT0.25的新分数由手写夹具构造，不是本轮新训练结果，不能拿其通过数或负差值写进科研效果表。**

## 8. 能从既有小型结果说什么，以及不能承诺什么

这里唯一作为真实既有观察复算的比较是 **ER0.25−SEQ**。独立重算仍为 **4/23通过**，通过的是O/N的Brier、log_loss相对raw SEQ的四项；其余未通过。下表不是新LOGIT结果：

|既有ER0.25−SEQ|平均差|条件95%区间|
|---|---:|---|
|O MAP|−0.004599553080780601|[−0.015904316897120435, +0.006791038350746349]|
|N MAP|−0.027483750468518715|[−0.035225815433494664, −0.019692009848518356]|
|Z MAP|−0.031082193543860975|[−0.043345612549187215, −0.018853500088938376]|

O MAP三个顺序差分别为ABC−0.015390462369616276、BCA+0.011118501704401429、CAB−0.009526698577126957。它们是三个固定顺序，不是三seed结论。旧LOGIT1−ER1的历史结果也不能代替当前匹配ER0.25的消融。

另有一个**明确手写的逻辑反例**：构造LOGIT相对ER的O MAP为+0.05、相对SEQ为−0.05，并相应构造其余指标。实际比较器得到23/23相对ER、0/23相对SEQ，两个flags一真一假且不产生selection。这证明代码能表达“增量有效但仍不如SEQ”，不是预测正式结果一定如此。

本轮正式LOGIT0.25结果未知。不能预先承诺MSE有效，不能把只胜ER解释成整体胜SEQ；同样不能因为相对SEQ未赢就要求增补超参数或拒绝结束此次已冻结诊断。

## 9. 实际命令、环境、起止时间与用例

### 9.1 网页CPU环境

Python `/opt/pyvenv/bin/python`，版本3.13.5；Linux 6.18.44 x86_64、glibc2.41；NumPy2.3.5、PyTorch2.10.0+cpu。CUDA不可用并设置 `CUDA_VISIBLE_DEVICES=`。没有安装依赖；没有加载原生BGE；transformers和sentence-transformers未安装。本次不声称复刻项目Python3.10.19/Torch2.9.1+cu130/RTX5090环境。

所有记录命令设定 `PYTHONPATH=scripts:tests`、`PYTHONDONTWRITEBYTECODE=1`、OMP/MKL/OpenBLAS/NumExpr线程为1；独立数值程序将Torch线程设为1并绑定允许集的最小CPU，本环境为CPU0。原指定unittest有线程限制，但未额外taskset绑核，不把它错误报告为已由同一方式固定CPU0。

### 9.2 用户指定的实际命令

```bash
PYTHONPATH=scripts:tests python -B -m unittest test_step28_er_weight_contracts -v
python -B scripts/step28_er_weight_run.py --help
python -B scripts/step28_er_weight_check.py --help
bash -n scripts/run_step28_er_weight_linux_20261001.sh
bash -n scripts/run_step28_er_check_linux_20261001.sh
```

均在 `/mnt/data/logit_audit/submission` 执行。Bash只有语法检查，CLI只有help，**没有执行Bash正文、native、check的run或正式execute**。unittest实际运行28个不同用例，框架报告30.418秒，外层墙钟32.701895秒；错误、失败、跳过均0。stdout 3,088字节、stderr 4,602字节，完整保留；unittest把用例结果写在stderr是正常输出，不能据此判定失败。

### 9.3 全部记录命令

日期均为2026-10-02。下列是实际UTC时间，换为Asia/Singapore加8小时；不是预计耗时。完整命令argv、环境覆盖和stdout/stderr大小/SHA见 `COMMAND_LEDGER.json`、CSV和 `COMMANDS.zh.md`。

|记录|开始 UTC|结束 UTC|墙钟秒|退出码|
|---|---|---|---:|---:|
|`01_identity`|2026-10-02T06:24:53.906768+00:00|2026-10-02T06:24:54.522471+00:00|0.615721|0|
|`02_unittest`|2026-10-02T06:25:27.466793+00:00|2026-10-02T06:26:00.168668+00:00|32.701895|0|
|`03_run_help`|2026-10-02T06:26:11.368988+00:00|2026-10-02T06:26:12.034886+00:00|0.665918|0|
|`04_check_help`|2026-10-02T06:26:12.578353+00:00|2026-10-02T06:26:13.294288+00:00|0.715954|0|
|`05_run_bash_syntax`|2026-10-02T06:26:13.902678+00:00|2026-10-02T06:26:13.906506+00:00|0.003830|0|
|`06_check_bash_syntax`|2026-10-02T06:26:14.553043+00:00|2026-10-02T06:26:14.556823+00:00|0.003784|0|
|`07_math_v1`|2026-10-02T06:32:38.408612+00:00|2026-10-02T06:32:41.182015+00:00|2.773423|0|
|`08_diff_v1`|2026-10-02T06:33:02.373969+00:00|2026-10-02T06:33:03.039785+00:00|0.665834|0|
|`09_statistics_v1`|2026-10-02T06:34:59.564944+00:00|2026-10-02T06:35:08.902696+00:00|9.337771|0|
|`10_path_v1`|2026-10-02T06:37:25.473211+00:00|2026-10-02T06:37:26.038944+00:00|0.565749|1|
|`11_path_v2`|2026-10-02T06:37:39.145755+00:00|2026-10-02T06:37:50.441088+00:00|11.295353|0|
|`12_access_identity_v1`|2026-10-02T06:39:47.382066+00:00|2026-10-02T06:39:49.203695+00:00|1.821649|1|
|`13_diagnose_source_order`|2026-10-02T06:40:05.248791+00:00|2026-10-02T06:40:05.864279+00:00|0.615506|0|
|`14_access_identity_v2`|2026-10-02T06:40:26.181501+00:00|2026-10-02T06:40:28.701497+00:00|2.520014|0|
|`15_final_integrity`|2026-10-02T06:44:11.472919+00:00|2026-10-02T06:44:12.138666+00:00|0.665767|0|


15次记录命令中13次退出0、2次退出1；后两次都是审查者参考初版，修订后实际重跑通过。没有隐藏失败、以未运行充当成功或把预期拒绝当成新科研结果。日志封装以前的两次探索错误缺少独立保存的精确时标/原始完整流，见第12节和 `FAILURES_AND_REVISIONS.zh.md`，不伪造补全。

## 10. 五个最终独立参考及证据定位

|独立程序|实际验证重点|最终原始日志|数值/案例输出|
|---|---|---|---|
|`independent/math_reference_v1.py`|解析loss导数；三分量参数梯度；手算clip/AdamW；末步LR；旧LOGIT逐值对齐|`evidence/07_math_v1/`|`evidence/math_reference/`|
|`independent/diff_reference_v1.py`|完整diff新侧及逆向还原七旧文件|`evidence/08_diff_v1/`|`evidence/diff_reference/`|
|`independent/statistics_reference_v1.py`|独立22指标/七端点/5000抽样/三比较/23项；18新45旧先保存后恢复|`evidence/09_statistics_v1/`|`evidence/statistics_reference/`|
|`independent/path_reference_v2.py`|独立1728配对schedule；真实微型288步；checkpoint/cache恢复；39盲门错误|`evidence/11_path_v2/`|`evidence/path_reference/`|
|`independent/access_identity_reference_v2.py`|34来源和原18来源；一次解析账本；target实际无梯度；非法reference前置拒绝|`evidence/14_access_identity_v2/`|`evidence/access_identity_reference_v2/`|

这些参考各自含完整源码，不只提供PASS字符串。辅助identity、失败版本、差异、source-order诊断、命令封装、末次完整性检查及本报告构建/打包程序也随包交付。提交的原28例和原源码另外完整保留，互不覆盖。

## 11. 分类裁决、受影响主张及最小处置

|类别|本轮发现/界限|受影响主张|证据和最小处置|
|---|---|---|---|
|SCIENTIFIC_BLOCKER|**未确认项目级阻断**|当前系数、相同历史供给和状态下LOGIT增量比较可由实现回答|第4–8节的独立数学、路径、统计证据支持；不需要为进入原定主审/必要CPU改项目|
|REPRODUCIBILITY_DEFECT（被审项目）|**未确认尚待修复缺陷**|当前冻结代码、入口和小型统计重现|303载荷/34来源/原18源、7文件diff、28例和独立参考通过|
|REPRODUCIBILITY_DEFECT（审查者工具R1，已修复）|独立路径v1多余右括号|这份参考初版不能运行，不能声称初版通过|`path_reference_v1.py:145`；原exit1；只删括号，v2重跑exit0；不改项目|
|REPRODUCIBILITY_DEFECT（审查者工具R2，已修复）|独立来源v1错误比较列表顺序|初版来源检查的失败不代表来源字节变动|`access_identity_reference_v1.py:15`；34项逐路径完全相同；唯一性+完整记录比较修复，v2 exit0|
|未验证范围，不是违反先后顺序|原生BGE四次手写更新/MSE梯度探针、CUDA/BF16、真实首域向量、六正式端点和一次正式标签门尚未实际运行|不能宣称原生环境/正式效果已验证|保持现有主审后必要CPU和low释放后正式流程；不要求本网页提前跑，不重复请求同范围授权|
|OUT_OF_SCOPE_OVERDESIGN|系统/网络加固、Linux Git、额外状态机/签名仪式、更多seed/λ/MSE、新增成功线和重复启动审批|会改变用户已确认诊断或扩大工程范围|本轮不提出、不实施；不作为放行条件|

本报告对“未发现缺陷”的限定是实际覆盖范围，不把哈希和单测当成对未知正式结果的担保。既定原生CPU证据若后来失败，应如实按失败本身处理；不能用本次网页微型PASS跳过它，也不需要为尚未按顺序运行而提前判阻断。

## 12. 所有本次失败、修订与未完成取证

**R1：独立路径参考v1，exit1。** 解析时`SyntaxError: unmatched ')'`，位于第145行，多一个右括号；没有开始任何代码执行。v2只删该括号，原v1、完整stderr和一字符diff保留；v2实际真实288步/39门测试通过。日志目录`10_path_v1`、`11_path_v2`。

**R2：独立来源参考v1，exit1。** 第15行将 `m.sources(p)` 与freeze的列表直接比较。独立诊断证明双方34项、无重复、逐路径大小/SHA/完整记录一致，只是顺序不同；因此这是参考断言错误，不是冻结破坏。v2先查重复，再按路径比较完整记录，并保留全部原严谨字段，实际通过。日志目录`12_access_identity_v1`、`13_diagnose_source_order`、`14_access_identity_v2`；原源码及diff均保留。

初次探索曾错误寻找根目录 `source_inventory.json`，得到FileNotFoundError；随后定位实际嵌套路径。另一次组合阅读命令末尾`ls /usr/local/bin/python* /usr/bin/python*`因第一个通配路径不存在退出2，随后以真实 `/opt/pyvenv/bin/python`运行。它们发生在统一记录机制建立前，**本交付没有当时独立保存的完整原始stdout/stderr和精确起止时间**；仅如实说明，不伪造日志。因此“全部核心CPU/参考命令原始证据齐备”成立，“每条早期探索命令都有完整原始字节回执”不成立。

39种非法门、23项判据破坏、非法target、超预算、访问重试、统计故障注入属于预期测试，不算未修复生产失败。提交实现说明中提及的冻结前Windows编辑纠错是执行者所述历史，本次没有对应运行证据，不把它冒称本次重现的科研脚本失败。

## 13. 正式阶段仍未知的内容和结束条件

本次网页执行没有连接项目Linux；没有原生BGE模型加载、正式execute、正式train/valid标签解析、test/owners访问、依赖安装或全仓回归。微型模型、缓存正文及手写标签是本次合法fixture，均在相关目录标明，不混同于缺席的正式文本/标签/缓存/权重。

仍待原顺序取得的项目CPU证据是quarter与logit各一次真实warm及一次真实加权更新，总计四次实际更新，加上不执行优化器更新的MSE独立梯度探针。`step28_er_weight_check.py:53–75,171–215`的设计会检查共有初始模型、warm后模型、Adam及当前/历史logit配对，并比较 `g_history_LOGIT = g_history_ER + g_0.5MSE`；**本次只审读设计，没有运行此原生入口**。该CPU检查额外的诊断forward不等于正式训练多一个历史forward；正式更新函数仍只有当前/历史两个群forward。

单GPU/单CPU、估6–8小时、硬12小时/16GiB是合同约束，16GiB是新产物空间限制，不把它误写成内存容量；本次没有实测新正式任务耗时、显存或实际产物体积。新代码不能覆盖仍执行low的旧字节；low何时释放资源不在本次观测范围。

从本次外审结论看，可以继续已确认的主审、必要原生CPU证据和随后原范围正式执行链；**不增加同范围重新审批，不自动替换ER，不以赢SEQ才算结束，不加点或换seed。** 等六个真实端点、18套新真实结果及45套既有复用完成后，才有本轮实际效果可作结果审查。

## 附录A：28个不同unittest的实际名目

以下逐项来自 `evidence/02_unittest/stderr.txt`，不是按测试类继承重复计数。前六个LogitWeightContracts为本次新增；其余22个为同文件既有受影响合同用例。

|编号|用例（原始名称）|实际结果|
|---:|---|---|
|1|`test_step28_er_weight_contracts.LogitWeightContracts.test_invalid_logit_reference_rejected_before_update` / `test_invalid_logit_reference_rejected_before_update`|OK|
|2|`test_step28_er_weight_contracts.LogitWeightContracts.test_logit_complete_saved_comparisons_reuse_45_sets` / `test_logit_complete_saved_comparisons_reuse_45_sets`|OK|
|3|`test_step28_er_weight_contracts.LogitWeightContracts.test_logit_real_continuation_checkpoint_and_unrefreshing_memory` / `test_logit_real_continuation_checkpoint_and_unrefreshing_memory`|OK|
|4|`test_step28_er_weight_contracts.LogitWeightContracts.test_matched_scope_and_gate_rejects_wrong_mse_and_target_origin` / `test_matched_scope_and_gate_rejects_wrong_mse_and_target_origin`|OK|
|5|`test_step28_er_weight_contracts.LogitWeightContracts.test_original_logit_exact_parity_and_two_forwards_one_clip` / `test_original_logit_exact_parity_and_two_forwards_one_clip`|OK|
|6|`test_step28_er_weight_contracts.LogitWeightContracts.test_quarter_supervision_and_half_mse_independent_gradients_adam` / `test_quarter_supervision_and_half_mse_independent_gradients_adam`|OK|
|7|`test_step28_er_weight_contracts.WeightContracts.test_candidate_views_and_paired_endpoint_arithmetic` / `test_candidate_views_and_paired_endpoint_arithmetic`|OK|
|8|`test_step28_er_weight_contracts.WeightContracts.test_collect_requires_every_point_and_preserves_36_sets` / `test_collect_requires_every_point_and_preserves_36_sets`|OK|
|9|`test_step28_er_weight_contracts.WeightContracts.test_complete_blind_gate_and_missing_endpoint` / `test_complete_blind_gate_and_missing_endpoint`|OK|
|10|`test_step28_er_weight_contracts.WeightContracts.test_confirmed_scope` / `test_confirmed_scope`|OK|
|11|`test_step28_er_weight_contracts.WeightContracts.test_gate_rejects_changed_start_and_calibration_domain` / `test_gate_rejects_changed_start_and_calibration_domain`|OK|
|12|`test_step28_er_weight_contracts.WeightContracts.test_gate_rejects_wrong_accumulated_total` / `test_gate_rejects_wrong_accumulated_total`|OK|
|13|`test_step28_er_weight_contracts.WeightContracts.test_gate_rejects_wrong_weight_and_unpaired_history` / `test_gate_rejects_wrong_weight_and_unpaired_history`|OK|
|14|`test_step28_er_weight_contracts.WeightContracts.test_low_checkpoint_uses_new_policy_and_weight` / `test_low_checkpoint_uses_new_policy_and_weight`|OK|
|15|`test_step28_er_weight_contracts.WeightContracts.test_low_endpoint_differences_use_quarter_and_seq` / `test_low_endpoint_differences_use_quarter_and_seq`|OK|
|16|`test_step28_er_weight_contracts.WeightContracts.test_low_full_collection_and_saved_reference_recovery` / `test_low_full_collection_and_saved_reference_recovery`|OK|
|17|`test_step28_er_weight_contracts.WeightContracts.test_low_real_tiny_stage_uses_paired_schedule_and_tenth_updates` / `test_low_real_tiny_stage_uses_paired_schedule_and_tenth_updates`|OK|
|18|`test_step28_er_weight_contracts.WeightContracts.test_low_scope_and_six_point_blind_gate` / `test_low_scope_and_six_point_blind_gate`|OK|
|19|`test_step28_er_weight_contracts.WeightContracts.test_low_selection_uses_quarter_independently_of_seq` / `test_low_selection_uses_quarter_independently_of_seq`|OK|
|20|`test_step28_er_weight_contracts.WeightContracts.test_only_history_weight_changes_forwards` / `test_only_history_weight_changes_forwards`|OK|
|21|`test_step28_er_weight_contracts.WeightContracts.test_original_23_guards_drive_selection_and_fallback` / `test_original_23_guards_drive_selection_and_fallback`|OK|
|22|`test_step28_er_weight_contracts.WeightContracts.test_original_er_exact_update_parity` / `test_original_er_exact_update_parity`|OK|
|23|`test_step28_er_weight_contracts.WeightContracts.test_real_tiny_checkpoint_replay_and_weight_metadata` / `test_real_tiny_checkpoint_replay_and_weight_metadata`|OK|
|24|`test_step28_er_weight_contracts.WeightContracts.test_resume_supply_cannot_revisit_first_or_skip` / `test_resume_supply_cannot_revisit_first_or_skip`|OK|
|25|`test_step28_er_weight_contracts.WeightContracts.test_stage_last_step_preserves_encoder_but_updates_head` / `test_stage_last_step_preserves_encoder_but_updates_head`|OK|
|26|`test_step28_er_weight_contracts.WeightContracts.test_tiny_full_restore_preserves_next_weighted_update` / `test_tiny_full_restore_preserves_next_weighted_update`|OK|
|27|`test_step28_er_weight_contracts.WeightContracts.test_unknown_coefficients_and_missing_history_rejected` / `test_unknown_coefficients_and_missing_history_rejected`|OK|
|28|`test_step28_er_weight_contracts.WeightContracts.test_weights_match_independent_derivatives_and_adam` / `test_weights_match_independent_derivatives_and_adam`|OK|


## 附录B：本轮freeze的34项来源，逐文件字节数及SHA

完整收到304成员及全部交付载荷另见JSON清单；本表专列当前有效34来源，旧31/28来源身份保留在原包，不覆写。

|当前冻结来源|字节数|SHA-256|
|---|---:|---|
|`docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md`|32430|`88c6eae5e2179f1671694ebeaa34258a9decab3f0baa70d23387ad5ba54e1485`|
|`docs/SELLER_ALIAS_ER_LOW.zh.md`|6234|`5dc9b64c456ef3d2bb08e0fcd0a847fe9cc1a764e7726aa585c345e268352bcb`|
|`docs/SELLER_ALIAS_ER_WEIGHT.zh.md`|9233|`d0f7a81a5a1c5f722fb32c50ec672f596f3b411a625631601f40583642725748`|
|`docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md`|6265|`432b453f07a60ac5422927d64b5f7892d536a63969a9721ca9d837af465b3f4d`|
|`reports/documentation/20261001/er_weight/decision.json`|1746|`da861037fc95389c088c19126d814742ffa9d1724415c39b22d6bb65f7ac1bc7`|
|`reports/documentation/20261002/er_low/decision.json`|1604|`c7ee4c8076cba6b375f50b0ae23a3bf1a3757771889a715cf0450177555686c6`|
|`reports/documentation/20261002/logit_weight/decision.json`|1851|`f97fc5df7a7abff48f3685c4f705ef9577aa188a96f940d93e901a69d20ca7cd`|
|`schema/step28_alias_ranking_policy.json`|3489|`b5b7e4fc628c389d9cf91664dcff64c28b5fc8e8295bc3d8a82a6e5cb20f9b01`|
|`schema/step28_bge_continual_policy.json`|5123|`28036c44903bb848d596c71ae987ce81beb2edc181fb846c010c4b70f6195534`|
|`schema/step28_chinese_base_policy.json`|8439|`add77dd222c8308eb013bd8226b1f6f51dc3a2403a533010322d27848c597f2a`|
|`schema/step28_er_low_policy.json`|5029|`201c7a766965df127803be89a9f3b7b1a5246053d95b2bcafa4dcc08cfa08c3d`|
|`schema/step28_er_weight_policy.json`|3732|`9eddf92646892d5890491e068ecbd4834463d0751de530a4701b7de8a98e4396`|
|`schema/step28_logit_weight_policy.json`|5059|`2aaf41724055ed3dff93ad65a8be2cc4e83c8f35d4515dace34bcdc44b081deb`|
|`scripts/run_step28_bge_continual_linux_20260930.sh`|1243|`5536ec24cecebc9e401c819f52b05aff4b976fcef530fffe42eaff605322c979`|
|`scripts/run_step28_er_check_linux_20261001.sh`|1133|`cdc161a0ab2f0efb1426f105578263e0dc81864e3aada6b99a97fcc2cfb92a59`|
|`scripts/run_step28_er_weight_linux_20261001.sh`|1418|`8682770c1590d714f1246f220fedbcf6317a2c1141f0bceb30f22d6bb4ff839d`|
|`scripts/step28_alias_calibration.py`|8818|`d7e79ca1c84c23a5d4e9534028d6a44833ee3ccaece7fb1b6dfbbd38430a7cca`|
|`scripts/step28_alias_ranking.py`|11658|`1b99709f55e1e4a6851bc7022410707e575fc898632351b4725e8db5435c3274`|
|`scripts/step28_bge_continual_check.py`|12126|`d919d2b26aee5b2e8676887fd5653e246bb20cb7cd65e28a5521df655c1e5e07`|
|`scripts/step28_bge_continual_evaluate.py`|13765|`55e9bf419801ec57686898cf78d8237962fc5e5549849a2e0eac3e66318353f1`|
|`scripts/step28_bge_continual_run.py`|34789|`bc425a88e502f9802aac81a0ac0977368606ed12c4e87e4bb2a6020ce9804db9`|
|`scripts/step28_bge_continual.py`|16341|`2af6bdf2001167aeafe9b701477e1c4785a146c7b8098498c73b6adf028eeaae`|
|`scripts/step28_chinese_base.py`|51352|`2b298459235d05d0934a0c822a14f8b91014e5d3797038b080eec5d9230225cc`|
|`scripts/step28_continual_expression_run.py`|26322|`3df878e3d78c07444cc0400b574adaf3a495b74fe193a14853706c7e9f67a760`|
|`scripts/step28_continual_population_data.py`|11587|`49cca1268b57422a46deefa8bc360423ccaeb8b5062750a39b540f0fb400e08d`|
|`scripts/step28_continual_population_evaluate.py`|20911|`398825da73b25b3d06420d7a75428451f38ff97f1239af02120d4921c70bca4a`|
|`scripts/step28_continual_population_run.py`|21310|`f0da5d51bb1291abe3417e5d695d8fc9134522d9fc7ced536417f4e484ab25e4`|
|`scripts/step28_continual_population.py`|9699|`f8f158fff648990ae575228d1b8dc112ac8c515ec5d6acc23a598120d770db20`|
|`scripts/step28_er_weight_check.py`|15227|`798edb10ebcf95efb63e82683e6e4e1d95594bb90a1a8ba93c64c51a5c5feb8f`|
|`scripts/step28_er_weight_evaluate.py`|11503|`05db784a1200f33f6449c6c29ec22152f05acc3329bbb07c15d05fc4a0de3bd1`|
|`scripts/step28_er_weight_run.py`|36238|`297574f91c44a63663aed7a77f247c71856bc1c213ca1c89d4d9fd3514e53a78`|
|`scripts/step28_er_weight.py`|13610|`7a112ae2f2bd2d56f9a631ae27524ee68c2e55c4ffc3762e58dc0dc46099c7e8`|
|`tests/test_step28_bge_continual_contracts.py`|31625|`de1066bb2fa535ab6c8d1a9079a00327d3b21e7e264b2ecc13d59f8e8246fa9a`|
|`tests/test_step28_er_weight_contracts.py`|43884|`de0af2b9dd54fcd792dd1a91daa90f0c1eb47ffa3fd14ae2aeab3ae38278ee83`|


## 附录C：六点盲门39种实际错误注入

以下全部为手写夹具上的预期拒绝；不是39次正式结果失败，不是39个额外unittest。24个非有限分数案例均同步更新了文件摘要，仍被实际数值门拒绝。

|编号|手写错误注入|实际拒绝原因|
|---:|---|---|
|1|`ABC_logit_quarter_stage2/calibration/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|2|`ABC_logit_quarter_stage2/development/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|3|`ABC_logit_quarter_stage2/stage-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|4|`ABC_logit_quarter_stage2/first-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|5|`ABC_logit_quarter_stage3/calibration/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|6|`ABC_logit_quarter_stage3/development/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|7|`ABC_logit_quarter_stage3/stage-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|8|`ABC_logit_quarter_stage3/first-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|9|`BCA_logit_quarter_stage2/calibration/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|10|`BCA_logit_quarter_stage2/development/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|11|`BCA_logit_quarter_stage2/stage-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|12|`BCA_logit_quarter_stage2/first-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|13|`BCA_logit_quarter_stage3/calibration/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|14|`BCA_logit_quarter_stage3/development/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|15|`BCA_logit_quarter_stage3/stage-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|16|`BCA_logit_quarter_stage3/first-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|17|`CAB_logit_quarter_stage2/calibration/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|18|`CAB_logit_quarter_stage2/development/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|19|`CAB_logit_quarter_stage2/stage-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|20|`CAB_logit_quarter_stage2/first-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|21|`CAB_logit_quarter_stage3/calibration/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|22|`CAB_logit_quarter_stage3/development/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|23|`CAB_logit_quarter_stage3/stage-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|24|`CAB_logit_quarter_stage3/first-cal/nonfinite_with_updated_hash`|Saved score/matrix schema differs|
|25|`MSE_wrong_0.125_even_if_total_recomputed`|Independent MSE weight or origin-stage model differs|
|26|`only_BCE_history_weighted_not_rank_hard`|Historical weight, total, or learning-rate log is wrong|
|27|`wrong_own_stage2_model_source`|Independent MSE weight or origin-stage model differs|
|28|`teacher_training_mode`|New LOGIT target generation differs|
|29|`new_target_list_missing`|New LOGIT target generation differs|
|30|`first_LOGIT_cache_replaced_by_ER`|Shared start evidence differs|
|31|`old_first_reference_summary_refreshed`|First-domain LOGIT targets were replaced or refreshed|
|32|`actual_old_survivor_values_refreshed_with_consistent_cache_hash`|First-domain LOGIT targets were replaced or refreshed|
|33|`stage3_consumes_first_cache_targets`|Training consumed different LOGIT targets from its stage cache|
|34|`unpaired_current_dropout_stream`|Training schedule is unpaired: current_dropout_stream|
|35|`Adam_reset`|Training schedule is unpaired: adam_step|
|36|`encoder_last_step_lr_nonzero`|Historical weight, total, or learning-rate log is wrong|
|37|`missing_sixth_endpoint`|Incomplete ER weight experiment|
|38|`old_study_policy_hash`|Incomplete ER weight experiment|
|39|`wrong_physical_update_total`|Incomplete ER weight experiment|


## 附录D：证据包使用及完整性

`original/logit_review.zip`是收到的原ZIP原样副本；`submission/`是其未修改304成员。`independent/`含全部独立源码、初版失败源、诊断和最终源；`evidence/`含所有15次记录命令的原stdout/stderr/command.json，以及矩阵、count、draws、checkpoint/缓存夹具、数值结果、故障恢复文件和新旧diff。`COMMAND_LEDGER.json/.csv`提供全命令机器可读索引，`COMMANDS.zh.md`提供命令及原流链接。

交付的 `FILE_MANIFEST.json`和`SHA256SUMS.txt`列出包内除这两个索引自身外的所有文件大小/SHA，避免自哈希递归；这两个索引的大小/SHA由外部交付回执记录。外部回执还记录完整证据ZIP的大小、SHA及解包后逐项核验结果。所有来源行号均以原 `submission/` 文件计；本报告引用本地独立证据，不依赖未提供的外部网页或远程运行。

复算程序记录的是本次真实环境和工作路径。带有固定 `/mnt/data/logit_audit` 的命令封装/末次原附件检查保留实际运行版本，不为“看起来可移植”静默改写；程序开头和命令清单足以定位工作目录。所有再执行建议仅限同样手写CPU范围，不包含原生BGE或正式execute授权扩展。
