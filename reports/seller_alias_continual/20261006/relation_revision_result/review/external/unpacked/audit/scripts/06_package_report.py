"""Create the full Chinese review and a self-contained, hash-inventoried delivery.
Uses only outputs of the independent audits; no model, training, or new statistics.
"""
from pathlib import Path
import json, hashlib, shutil, zipfile
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'audit/outputs'; IN=ROOT/'input'
DEST=ROOT/'delivery'; DEST.mkdir(exist_ok=True)
S=json.loads((OUT/'saved_statistics.json').read_text());E=json.loads((OUT/'execution_evidence.json').read_text());D=json.loads((OUT/'diagnostics_audit.json').read_text());C=json.loads((OUT/'report_crosscheck.json').read_text())
RR=IN/'reports/seller_alias_continual/20261006/relation_revision_result'
FNAME='RELATION_REVISION_RESULT.external_review.zh.md'
def ci(x): return f"[{x[0]:+.6f}, {x[1]:+.6f}]"
def effect_rows(items):
 lines=[]
 for ep,m,label in items:
  a=S['endpoints']['relation']['primary'][ep][m]['mean'];b=S['endpoints']['logit0.1']['primary'][ep][m]['mean'];d=S['delta'][ep][m]
  lines.append(f"|{label}|{a:.6f}|{b:.6f}|{d['mean']:+.6f}|{ci(d['conditional_95pct_interval'])}|")
 return '\n'.join(lines)
main=effect_rows([(ep,m,f'{ep} {label}') for m,label in [('map','MAP'),('average_precision','AP')] for ep in ['O','N','Z']]+[('final_all','map','final_all MAP')])
forget=effect_rows([(e,'map',e+' MAP') for e in ['F_first','F','G']])
prob=effect_rows([(e,m,e+' '+label) for e in ['O','N','Z'] for m,label in [('brier','Brier'),('log_loss','log-loss')]])
pooled=[]
for ep in ['O','N','Z']:
 for arm,label in [('relation','修订'),('logit0.1','LOGIT0.1')]:
  x=S['pooled_stage_cal'][arm][ep]
  pooled.append(f"|{ep}／{label}|{x['tp']}|{x['fp']}|{x['fn']}|{x['tn']}|{x['precision']:.6f}|{x['recall']:.6f}|{x['f1']:.6f}|{100*x['fpr']:.4f}%|")
local=effect_rows([('N','specificity','N specificity'),('Z','specificity','Z specificity')])
roles=[]
for o in ['ABC','BCA','CAB']:
 row=[]
 for role in ['raw','stage-cal','first-cal']:
  x=next(x for x in D['score_distributions_and_counts'] if x['point']==o+'_relation_revision_stage2' and x['domain']==o[1] and x['role']==role)
  row.append(f"{x['tp']}／{x['fp']}")
 roles.append(f"|{o}/{o[1]}|"+'|'.join(row)+'|')
qrows=[]
for o in ['ABC','BCA','CAB']:
 x=D['training'][o+'_relation_revision_stage2'];qrows.append(f"|{o}|{x['history_compressed']['first']:.6f}|{x['old_Q_first']:.6f}|{x['history_compressed']['mean']:.6f}|{x['weighted_R_mean']:.6f}|{x['history_compressed']['last24_mean']:.6f}|{x['gradient_norm']['median']:.3f}|")
report=r'''# 修订关系目标记忆：已完成固定点的独立结果外审

**审查对象：**20261006_135937，B＋Q＋0.1R；不是旧准入包，也不是再次签发启动许可。  
**审查依据：**用户本条上传的 `review_input.zip`、其冻结原件、按context解开的历史证据，以及本轮实际独立代码和输出。  
**裁决：在所提供且可核验的证据范围内，接收为有效的、配置层面的开发负结果；六项前瞻继续条件0/6，建议按原合同收尾。** 没有发现必须撤销本次完成资格或主要统计结论的SCIENTIFIC_BLOCKER；没有发现须重算、补训或更改源码才能接收的当前REPRODUCIBILITY_DEFECT。主执行者仍须独立处置，本报告不代替主审，不授权后续训练或清理权重。

本次审查不是以先前“可启动”裁决证明本次结果合格：重新核对了当前来源与完成/访问链、实际更新记录、保存矩阵、对照来源、统计与分析。现配置失败不改判旧配置失败；不能由两个配置的失败否定累计历史目标统计＋小缓存迁移的整个方法家族。

## 1. 证据、身份与实际能力边界

### 1.1 本次唯一原附件及嵌套历史

|文件|实测身份与完整性|
|---|---|
|本次 `review_input.zip`|11,461,459字节；SHA-256 `3df2d543301bfc4ad80e1d9bf33a7a3f04123d71b0068188308a4b7a34bfd9d3`；272文件、271载荷＋manifest；逐载荷大小/SHA、解压字节、CRC和路径检查通过|
|`history/relation_revision_pilot_external_evidence.zip`|5,958,634字节；SHA `c42e6e0232927ef7c4bc8decc5e9dbebc4889a036cb008ceb88e86a0c1346a6c`；209文件、208载荷通过|
|上一项中的 `input/relation_revision_pilot_review.zip`|5,427,862字节；SHA `992990d9c1b5f9c5053b6fc675509506ff4d66c5da22247d8c156e818ac2de84`；63条目＝60文件＋3目录；59载荷通过|
|再嵌套的 `history/relation_revision_review.zip`|5,257,725字节；SHA `480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5`；319条目＝305文件＋14目录；304载荷通过|

对应实际脚本及输出：`audit/scripts/01_identity.py`、`audit/outputs/identity.json`。旧快照中的未启动、未授权、待审、旧库存状态按其原时点保留；当前根manifest和本次合同决定本包身份与本次范围。

### 1.2 读取和执行范围

首先全文读取本轮 `review/context.zh.md` 和128行 `docs/SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md`。随后读取冻结pilot合同、policy、新runner与revision核心，追踪相关旧编排的当前供给、统计固化、保存恢复、评价、失败/恢复资格及其必要直接依赖；按context六组索引取得先前实际审查原文、K1和准入证据，未无差别重审未变数学或其他关闭路线。对二进制矩阵、分数和清单做程序化核对，不把“全文阅读”扩称为从未提供的权重、缓存或真值进行复现。

本网页工作区实际环境：Python **3.13.5**、NumPy **2.3.5**、Torch **2.10.0+cpu**。实际执行了五份独立审查脚本，并实际导入冻结新runner检查独立模块的全局绑定；没有调用其execute/train、load_model、score、标签loader或校准拟合。本轮没有重跑项目Linux CPU测试、BGE/GPU一步、正式训练或完整原生结果核验器。

特别是：九份模型和九份Memory本体不在包内。它们的Linux实查记录可以与端点、inventory、gate对应，但网页不能重新哈希这些本体、恢复模型/Adam/RNG或验证缓存正文。没有逐对标签，因此不能从真值重新生成MAP/AP、ROC、Brier、log-loss；这里独立复算的是已保存群指标的配对统计，另从保存counts重建其能决定的工作点指标。分数只用于变换、排序一致性和预测正数检查，未反推未开放标签。

这些边界是本次授权交付的预先范围，不应为得到更强验证而要求再次开放真值、读取私产或重训。可见模型自述为GPT-6 Astra Pro；无法据自述独立认证界面“6/Pro”的未公开后台标识。

### 1.3 独立复算覆盖

|独立检查|结果及边界|
|---|---|
|候选28套＋LOGIT所需27套|55组矩阵/计数，110个文件；另两份对照collection身份；共112文件身份核对|
|双方raw/stage-cal/first-cal/primary，七端点，22指标，主差值|1,386统计项，均值＋三顺序＋区间两端共8,316数值独立重算|
|更多群级/绝对统计|连同counts导出的19,800个指标值、绝对端点宏/合并统计，共31,476数值对照；最大绝对差 **4.44×10⁻¹⁶**|
|2592更新与随机供给|独立重建当前shuffle、历史抽样、Algorithm R、Adam步数及学习率，日志B/Q/R组合精确一致；4320群梯度呈现|
|盲分数|37组文件身份/结构；18个仿射变换逐值精确一致，排序与并列完全保持|
|图表|主报告22行数值表、全指标154行表；SVG18条轨迹54点与独立矩阵聚合匹配，并实际查看PNG|

脚本不调用生产endpoint_fields/summarize_field来产生期望；采用明确端点系数和bootstrap抽样频数矩阵独立实现。遵循同一冻结统计定义不是另选成功线。保存矩阵本身的语义仍依赖已审真实评价路径与原始执行证据，不能由矩阵自洽消除这个事实。

## 2. 完成资格、来源和真实计算路径

本节主要位置：冻结 `source/scripts/step28_relation_revision_run.py:18–72`；`step28_relation_revision.py:14–103`；`step28_relation_memory_run.py:186–385,478–556`；`job/execution.json`、`run/manifest.json`、`before_valid.json`、`access.json`、`completion.json`、`resource.json`及 `inventory.json`。独立证据为 `03_execution_evidence.py`、`execution_evidence.json`。

### 2.1 当前来源不是旧源码或手写夹具

31份来源逐文件与authorization、execution、run manifest、collected、evaluation以及所绑定CPU证据相符；也与上一轮实际审查的31份原字节一致。17份新原生科学来源和当前冻结文件相符，gate指向本轮正确的native/integration CPU回执。没有把旧BGE核验或K1声明冒充这一次正式结果。

实际导入检查十个编排函数的 `__globals__` 指向新独立workflow字典；`runner.method.update is revision.update`，旧prior模块没有被改成新核。当前新优化核实际仍为：

\[
B(C)+Q(v)+0.1R(H),\quad v=K^{-1}(Xp/378+0.001w).
\]

当前B先反传并释放图；历史同一活表示共同产生Q与R并一次反传；最后统一clip一次、AdamW一步。参考/统计固定，当前Y与w保持可微，Q中的经p及直接epsilon·w两条w路径均保留。R使用抽到的历史标签，不是教师间隔或LOGIT MSE。K1此前已通过，当前冻结目标未改；本轮不声称仅由更新日志重新证明所有参数梯度。

日志新增三分量和历史分量不是只存在字段：全部2592步独立核对了FP32当前B组合、Q＋FP32字面0.1R、总损失、encoder与head学习率，最大差均为0。首阶段Q/R全0，后两阶段1728步才有历史呈现。这个检查结合已审调用链支持实际执行身份；仅凭这些标量仍不能反推出分量梯度比例或因果作用。

### 2.2 供给、计数与标签开放

实际公开输入与Supply路径保持48fit/12当前校准/20valid每域。保存partition含144fit、36calibration、60development，240个群ID唯一且角色不重叠，development群/实际域顺序与collected相同。按群隔离是当前协议，不等于证明真实市场身份或未提供owners真值。

独立按冻结种子派生方式重建三顺序每阶段六轮当前shuffle：每48fit群呈现6次；重建每阶段288次历史有放回抽样及Algorithm R幸存成员。全部与更新记录一致，历史成员与当前阶段fit不交叉；seen/count首两次固化48/96，各阶段Adam步数288/576/864。108条console进度在单作业内单调推进，没有记录中的自动续跑或另一次训练。

第一次固化因old count=0不做运输；第二次先联合全部旧六群运输，再累加当前48fit各一次，之后Algorithm R与参考刷新。第三阶段无后继，按冻结合同不固化下一份记忆，因此最终Memory count/seen=96、stage=2并不表示少训第三阶段。

正常入口的实际順序仍为：完成2592更新、九端点、校准、完整保存恢复和全部盲分数 → blind_gate → before_valid → 消费valid一次 → 保存28套 → 统计 → 检查访问/资源后签发有效completion。before_valid为train1/valid0/heldout0/owners0，最终为train1/valid1/heldout0/owners0。一次尝试在loader前记账；没有把后处理恢复许可变成二次读标签。Audit和private_custody没有独立计数字段，不能从两个零计数声称已扫描整个服务器；依据的是当前实际输入路径、冻结禁止范围与提交记录未见访问。

LOGIT0.1使用已关闭结果的保存矩阵，其before_valid/access/completion原件也一致：train/valid各一次、heldout/owners0、当时新增1728更新。这里没有重训基线，也没有用本候选授权补做其他基线臂。

### 2.3 真实保存恢复的支持与不能复跑的部分

`checkpoint`先对完整12校准群及60valid群保存分数，拟合当前12群的单调校准，写Memory并从文件恢复；保存完整model/Adam，再真实load、校验加载后摘要、恢复RNG，重算校准与valid分数逐值一致；另存并恢复推理端点后删除已验证临时full。返回的恢复Memory用于下一阶段。它不是只设置 `full_restore_verified=True`。

九端点记录、完整/推理状态字节、Adam计数、恢复字段、校准ID和重放分数相互对应。九次校准均为当前12群、4536对、240正例，a>0且记录的NLL下降，首域校准映射正确沿用到first-cal。网页没有校准标签，故未重新拟合这些映射，也未核验拟合最优性；实际验证了被保存的映射如何作用于盲分数。

新候选各顺序从自己的BGE预训练＋新关系头起始，实际stage1也是本次的 `*_relation_revision_stage1`；只有对手首域来自 `*_shared`。candidate initial raw不替代首域训练完成后的指标。九模型与Memory本体的缺省附件是明确范围限制，不能称网页完整载模复现了九端点；当前证据与正常真实恢复路径相符，没有发现矛盾。

### 2.4 资源与停止/后处理边界

|记录|本次实际结果|合同限额/解释|
|---|---:|---|
|包装开始/结束|2026-10-06 14:05:45—17:55:53（UTC+8）|exit0；正式gate在开始前签发|
|总墙钟|13,808秒＝3小时50分8秒|24小时；不是从训练内核单独计时|
|CUDA allocated/reserved峰值|7.884 / 8.725GiB|reserved门28GiB，不与24GiB产物混写|
|RSS观察峰值|6.573GiB|64GiB|
|产物观察峰值|14.603GiB|24GiB，含相关console/wrapper及状态共存计量|
|阶段末完整Memory记录|484,442—504,581字节|1MiB；不是只报数值小计|
|九模型实查记录总字节|11,758,415,337|本体未上传、未在网页载入；当前未获清理许可|

`completion.json`资源数与`resource.json`及wrapper对应。首次回传inventory168载荷全部可重新哈希。无failure/recovery回执；来源中已修复的P1/P2/P3仍在实际运行版本：纯统计只能声明需要有效completion；只在完整矩阵且合格POSTPROCESS_IO_ONLY状态下继承同job剩余预算进行统计恢复，不重读标签或重训。本次是正常完成，并未通过恢复入口掩盖资源失败。

当前输出与Memory本体未完整提供，网页的物理字节核验不能覆盖服务器上所有瞬时驻留状态；这里接受的是已审计量口径的完整状态记录及运行证据，不作无条件系统资源证明。一次新原生探针与本次完整实际运行现在各有对应证据，仍不构成相对LOGIT的公平速度实验。旧新增1728更新与本次包含新首域2592更新不同，输入长度和额外核验也不同。

## 3. 保存结果、配对定义与六项判断

位置：冻结 `step28_relation_memory_run.py:400–458`，`step28_bge_continual_evaluate.py:14–39,91–138`；本次 `job/evaluation/collected.json`、`evaluation.json`，LOGIT两份collection及54个所需矩阵/counts。独立证据：`02_saved_statistics.py`、`saved_statistics.json`、`bootstrap_draws.npy`。

### 3.1 对手选择和角色

把冻结 `reference.linux_root` 映射到本包 `reference.local_root`，只读取本比较依赖的对照原件，不伪造路径或要求附上不相关历史臂。候选初始raw加9点三角色共28套；LOGIT自身3个shared首域和6个logit_tenth后续点，各三角色共27套。两份collection及下游文件均按原SHA/大小核对，三角色集合、60个group_ids、实际domains、22列顺序完全一致；不是仅看矩阵shape。

所有矩阵为60×22 FP64有限值；每群counts恰为20正、358负、总378。AP是群内378对的average precision；MAP是28查询AP的均值，不能混写。raw/stage-cal/first-cal的14个排序/曲线列逐值一致。primary取raw的14列和stage-cal其余8列，既不偷偷选最优校准角色，也不按结果换阈值。

完整22列：average_precision、trapezoidal_pr_auc、roc_auc、recall_at_fpr_1pct、brier、log_loss、precision、recall、f1、specificity、balanced_accuracy、mcc、map、mrr、recall_at_1/3/5/10、ndcg_at_1/3/5/10。

### 3.2 端点与配对bootstrap

令M[s,a]是某固定order第s阶段、该order第a个到达域的群宏指标。先按实际域保留20个群的配对，再聚合：

|端点|定义|
|---|---|
|O|(M[3,1]+M[3,2])/2|
|N|(M[2,2]+M[3,3])/2|
|Z|M[3,3]|
|F_first|M[1,1]−M[3,1]|
|F|((M[1,1]−M[3,1])+(M[2,2]−M[3,2]))/2|
|G|((M[2,2]−M[1,2])+(M[3,3]−M[2,3]))/2|
|final_all|(M[3,1]+M[3,2]+M[3,3])/3|

F_first/F/G对Brier和log-loss采用冻结的反向符号，使“损失增大对应遗忘、损失减小对应获得”。MAP的正ΔF意味着比对手更多遗忘，不能把所有正差都着色为改进。

独立抽样使用 `Generator(PCG64(20260930))`，5000次，在实际A/B/C每域20群内有放回抽取20；同一群抽样用于双方、全部阶段和三个固定顺序。用抽样频数乘端点系数，与生产逐索引求均值实现不同，结果仅有浮点求和差。分位数method=linear。

三个顺序不是三个独立种子；378边/28查询也不是bootstrap独立单元。O/N有120个群端点出现、Z有60个出现，仍只对应60个不同实际域群。区间条件于这次s0训练、三个固定顺序和已开发valid；不覆盖重训随机性、开发选择、不同时点搜索或真实市场分布，不是多指标同时95%区间。

### 3.3 主比较结果

差值始终为修订候选−LOGIT0.1。

|指标|修订|LOGIT0.1|差值|条件95%区间|
|---|---:|---:|---:|---|
__MAIN__

O MAP三顺序差为−0.120992、−0.041627、−0.096693；N/Z MAP也各顺序为负，不是单一路径独占总体结论。附录A提供七端点全部22项主指标、每顺序差与区间；完整四角色和绝对stage结果在独立JSON中。

六项前瞻规则为ΔO MAP>0，ΔN/Z MAP≥0，ΔO/N/Z AP≥0，全部false。它们按观察均值判定，区间不是悄然新增的通过条件；六个区间同时处于负侧是另外的、条件于既定抽样的证据。`automatic_followup=false`、有效completion的`worth_matched_replay=false`相符。旧Q-only配置不能替换主对手，也不能用描述性历史改动修改成功线。

## 4. 对“首域未改善、额外遗忘、新域获得”的独立判读

位置：主执行结果报告61–83行；候选/对手实际stage矩阵；独立`report_crosscheck.json`。

### 4.1 首阶段不足仍在，但不能唯一归因

|首阶段当前域MAP|修订|旧版关系记忆|LOGIT自身shared首域|
|---|---:|---:|---:|
|ABC/A|0.457547|0.459008|0.477153|
|BCA/B|0.474226|0.483205|0.493308|
|CAB/C|0.428168|0.441489|0.494656|

三点低于旧版和LOGIT，故“本固定点未改善首阶段观察能力”准确。此时还没有Q或R，历史项不可能解释这部分已出现的差距。与LOGIT相比，关系头、32维/tanh、Dropout和训练配置不完全匹配；本次不能识别哪个因素是唯一/主要原因，也不支持“B目标普遍无效”。与旧版的差只是预先未设新比较区间的历史描述，不把三个点值作为跨种子因果结论。

### 4.2 额外遗忘这次确有条件负向证据

|自身首域|修订阶段1→2→3|LOGIT阶段1→2→3|
|---|---|---|
|ABC/A|0.457547→0.385315→0.314347|0.477153→0.457921→0.465755|
|BCA/B|0.474226→0.417400→0.431364|0.493308→0.491393→0.463492|
|CAB/C|0.428168→0.397354→0.382737|0.494656→0.479123→0.476369|

|诊断|修订|LOGIT0.1|差值|条件95%区间|
|---|---:|---:|---:|---|
__FORGET__

ΔF_first和ΔF的MAP区间均大于0，因此可说本配置比基线额外遗忘更多；这不是只因终点O低便反推遗忘。旧版ΔF区间跨0仍是旧配置原结论，不应追改。BCA/B在最后阶段从0.417400回升至0.431364，不能写成每一步都下降；旧版CAB/C自身改善不能移植给新版CAB/C。

一个仅作点估计解释的恒等式是O＝两个旧域各自获得时均值−F。候选获得时均值0.440059，基线0.476723，相差−0.036664；额外遗忘+0.049774，恰有−0.036664−0.049774＝−0.086437的O差。这个算术拆分表明起点不足和后续下降两者共存，**不是为两个机制分配因果贡献百分比**，没有新增验收端点或区间。

### 4.3 新域有获得，不等于新域绝对水平好

G MAP候选自身区间为 **[0.044691, 0.075731]**，三顺序G均为正；说明确有阶段内新域获得。ΔG区间跨0，不能声称获得显著优于LOGIT，也不能据跨0证明等价或非劣。较低起点上的正获得与最终N/Z更低并不矛盾。

O/N/Z Recall@5分别0.484970/0.544940/0.545238，比基线少0.103125/0.059077/0.068155，区间均处负侧；MRR/NDCG等全部保留在附录，不用六项筛选代替所有用途相关指标。

## 5. 校准、固定0.5工作点与局部误报改善

位置：主执行报告55–57、105–120行；各点mapping/scores；counts；独立 `02_saved_statistics.py`、`04_diagnostics.py`。

### 5.1 概率预测误差与校准不是一回事

|指标（primary/stage-cal）|修订|LOGIT0.1|差值|条件95%区间|
|---|---:|---:|---:|---|
__PROB__

这些是**校准后的概率预测误差**，不是单独的校准误差或ECE，不能从Brier/log-loss直接分解唯一失效因素。候选自身校准也并非“没有效果”：O的raw Brier/log-loss约0.066411/0.244637，经stage-cal降至0.049139/0.197799；降低相对自身的概率误差，仍可能弱于基线并使固定0.5更保守。

18个保存变换严格等于 `raw.astype(float64)*a+b`，a>0，排序和并列逐值保持。不同角色差异只改变概率/工作点；不可能通过这类单调校准修复已观测MAP/AP差距。没有重拟合校准，也没有按valid选择替代阈值。

### 5.2 合并counts不是群宏F1

下表对预定端点的群出现先求和，再计算比例；O/N是120次群出现，Z是60次，不是新增独立样本。概率≥0.5等于校准logit≥0；不是概率阈值0。

|端点／方法|TP|FP|FN|TN|合并precision|recall|合并F1|FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
__POOLED__

例如O的候选群宏F1为0.055628，基线0.141583，差−0.085954；合并F1则是0.064591和0.145685。两种数值并不冲突。不能把群宏F1的条件区间附到合并F1上。每群正负数固定，使recall、specificity/FPR等相应均值与合并比率相容，但precision/F1一般不具有该等价性。

### 5.3 第二阶段“全判负”的范围要说完整

ABC/B、BCA/C、CAB/A的stage-cal当前域各20群、400正对、7160负对，均为TP=FP=0。已保存校准logit最大值分别为−0.561965、−0.013750、−0.295545，均小于0，直接支持无预测正例。分数的标准差分别约0.894、0.716、1.020，**不是常量输出，也不意味着MAP为0**。

ABC/CAB阶段2在全部60valid群均无正预测；BCA全60群检出2个，分别来自非当前域A/B，不能扩大“当前域全漏检”为三个顺序所有域都全漏检。

角色差异进一步确认这不是所有分数角色的同一事实：

|阶段2当前域|raw TP／FP|stage-cal TP／FP|first-cal TP／FP|
|---|---:|---:|---:|
__ROLES__

这些已有角色只是解释固定工作点的保守性，不授权选其中最有利的角色替换primary或重新设阈值。

### 5.4 低误报是实际局部正向，不能据总失败抹去

|指标|修订|LOGIT0.1|差值|条件95%区间|
|---|---:|---:|---:|---|
__LOCAL__

N/Z specificity的改善区间处正侧，对应FPR分别由约0.1816%降至0.0419%、0.2048%降至0.0838%。代价是recall由7.5%降至0.9167%、8.6667%降至1.8333%。N/Z中预测更少能带来低误报，并不能抵消候选检索和漏检恶化。O的specificity点值略差且区间跨0；不能一概写成所有域误报都改善。

七端点全部22指标和三顺序值均已复算。O/N/Z主指标中，N/Z specificity是总体方向有利的例外；三顺序局部亦有其他取舍，例如BCA的O recall略增，但这不推翻整体O recall负差。保留这些细节，不写“所有指标在每条路径上都更差”。

## 6. 阶段2第一步高Q：推断成立到什么程度

位置：主执行报告85–103行；冻结 `step28_relation_revision.py:54–69`；旧核心 `step28_relation_memory.py:158–173,225–269`；九份updates及原版对照updates。独立重算在 `diagnostics_audit.json`，手写例在 `04_diagnostics.py`。

### 6.1 时点和数值成立

|顺序|新Q首步|旧平方历史首步|新Q阶段均值|新加权0.1R均值|新Q末24步均值|新总梯度裁剪前范数中位数|
|---|---:|---:|---:|---:|---:|---:|
__QROWS__

第一阶段从空历史开始，第一次固化只有在当时表示下累计48fit的H/b/c/N并刷新参考，没有旧统计运输。进入第二阶段第一步后，当前B.backward与历史(Q+0.1R).backward之间没有optimizer.step；记录的Q是该步更新之前的值，而非第一步更新后的反馈。

在同表示、同确定性规范前向的理想条件Y=X下，令S=XXᵀ/378，则

\[
v=(S+\epsilon I)^{-1}(Sw+\epsilon w)=w.
\]

因此Q就是第一阶段末所累积的平方代理均值，N=48，尚不是经多次近似运输得到的历史目标。不能优先用“运输多次累积误差”解释这一初次高值。实际参考与权重本体未附，网页未逐个重算当时Y=X或48群真实平方损失；这里的等价是由已审代码/时点支持的条件数学解释，不是补造缺失本体实测。

### 6.2 可成立的结论

“B训练得到的状态在保留的平方Q下代价很高，因此应关注两种目标的协调性”是有代码、时点和日志支撑的诊断线索。H/b/c/N并不精确摘要新的BCE/query/hard训练目标。B可偏好更大的正负间隔或不同分数偏置；平方Q仍偏好±1分数与2间隔。这个解释没有改变既定算法，更不要求换回LOGIT约束。

随后Q均值下降同时旧域valid MAP下降，说明代理变小不等于旧域排序被保持。该关联增加了“代理与用途未充分一致”的疑虑，但没有分离B、Q、R、统计迁移、表示头和统一裁剪的因果贡献。

### 6.3 一个必要的反向检查：高Q不意味着排序已经错误

独立构造合法28账号、20正/358负、全查询均有正例的**手写标签群**。所有正分数相同、所有负分数相同，则完整top5-hard与查询归约可以直接算：

|手写分数|B|Q平方代理|R|MAP/AP|
|---|---:|---:|---:|---:|
|正+1、负−1|1.959991863|0|0.126928011|1 / 1|
|正+2、负−4|0.370972871|24.576719577|0.002475685|1 / 1|

B和R下降、排序仍完美，Q却从0升至24.58。另用有界32维规范特征X=Y与epsilon=.001手算，v−w最大误差5.68×10⁻¹³，支持同表示化简。它不是正式样本、BGE参数可达性或效果实验；不得据其数值拟合本次正式Q值。它证明的是：**损失尺度/目标偏好差异足以产生高Q，不需要先发生排序错误。**

所以高Q不能单独解释所有MAP不足，也不证明梯度一定相反。Q含常数/二次项；分量损失绝对值不是梯度范数。即使某一步Q大、0.1R小，也不能得到两者对encoder/head/w的范数比例、夹角或Adam实际更新占比。

### 6.4 明确不可推出的结论

- 没有测得正式分量梯度夹角，因此不能把“目标协调风险”改写成“实测梯度冲突”。
- 不能证明Q是额外遗忘的唯一/主要原因，或R无效；无相同配置的反事实轨迹。
- 不能把某个更大R系数或更小Q系数称为已验证修复，也不据本次失败自动批准搜索。
- 后两阶段1728步均clip前总范数>1，新中位数9.66—25.71、旧16.17—23.85。旧版同样频繁clip；大范数不等于梯度爆炸。统一clip和Adam的最终参数更新不能从标量损失直接还原。
- CAB第三阶段缓存A6/C0，使R没有C原始群的直接监督，但N=96统计仍包括C历史贡献；ABC/BCA旧域缓存仍存在也退化，所以缺域不是全部解释。运输奇异值仅说明有限矩阵的各向异性，不给缓存外误差上界。

保留单群训练映射与阶段末联合映射不一致、岭偏差、非线性运输近似、低秩投影、聚合排序取舍及未见卖家泛化未证等前轮限制。R改变部分缓存排序方向的压力，不自动补齐这些限制；本轮不重开已审机制或关闭路线。

## 7. 主执行报告、图表与原数值是否一致

独立核对主结果报告七行主比较、六行首域/轨迹、三行Q诊断、六行合并counts，共22行，全部满足各自展示精度。`analysis/metrics.zh.md`的154行九位小数表与独立统计最大舍入差4.99×10⁻¹⁰，没有需要更改的数值格。

实际解析 `analysis/stage_map.svg` 中18条轨迹（3顺序×3实际域×2方法）、54个点，按原轴的坐标变换还原MAP；与保存矩阵独立均值最大差7.13×10⁻¹⁰，属于SVG小数坐标精度。也实际查看PNG的图例、s0三顺序/开发valid标注、实虚线及轴范围；没有发现图表把首阶段换成shared、把负向轨迹画成改善或漏出界数据点。

原图是提交者根据保存统计绘制，网页没有为图做新训练或补真值。展示检查不证明MAP底层标签计算，但与本次可复算统计一致。

## 8. 分级意见、最小处置与收尾

|ID／性质|位置和证据|影响|最小必要处置|
|---|---|---|---|
|S0：SCIENTIFIC_BLOCKER|当前来源、执行、访问链、矩阵配对与分析均未发现足以撤销资格的具体矛盾|本配置可作为有效开发负结果接收|无新增训练或数据要求；主审独立确认收尾|
|R0：当前REPRODUCIBILITY_DEFECT|本次已承诺的来源、矩阵/计数、必要原件与独立脚本能够核对；未发现必须补做的交付缺陷|附带说明：模型/Memory/逐对标签未附限制底层复现强度，不把范围外缺省伪装为全面复跑|无需为了当前审查开放真值、上传权重或重新执行原生核验|
|N1：非阻断措辞建议|结果报告15行“剩余0”；wrapper实际13808秒，硬限86400秒；sync语境是剩余训练工作|容易误读成24小时预算已经耗尽，与正常完成记录混淆；不是当前资源超限或新可用预算授权|建议改为“剩余训练工作为0；本次作业已结束”，保留原冻结文件，以勘误/处置说明记录即可|
|N2：已正确限定的机制解释|报告95–101行，高Q、clip和后续MAP关联|是目标兼容性线索而非正式分量梯度或因果归因；当前措辞保留了限制|接收现有推断边界；后续引用不得删去限制，不要求现阶段补梯度/消融|
|O1：OUT_OF_SCOPE_OVERDESIGN|新训练/调参/种子/全量消融、扩缓存、通用故障平台、恢复group-meta或“LOGIT加约束”替换核心|会超出当前结果审查及用户限定，不能用作接收负结果的前提|本轮不执行、不自动提为必修；未来只在另有前瞻授权时讨论|

N1的算术上未用墙钟限额为72,592秒，但作业已完成，不能把它转移为新训练/重试许可。未发现必须改数值、重算区间、重拟合校准或修正图表的勘误。

本次五份审查脚本最终均exit0，但审查代码自身并非从未出错：第五脚本最初因未strip Markdown表格单元格空白而识别0行、触发断言；失败版本和原输出已保留，修复只影响审查解析，不改项目数据/统计/源码。早期交互查看schema的两次KeyError也单独记为网页检查探索错误。不能把这些写成项目Linux失败、触发重试或影响实验资格。详见 `audit/outputs/reviewer_notes.zh.md`。

## 9. 分别裁决

**科学/完成资格：**依据当前可核验来源和执行证据，接受本次正常完成；独立复算支持报告的六项0/6、概率/检索下降、更多MAP遗忘及固定工作点漏检。

**效果与开发判断：**相对前瞻主要对手LOGIT0.1，修订固定配置没有达到继续条件。与旧版的O/N/Z均值下降是描述性历史观察，不改为新主比较，不追改旧负结果，不抹去新域G获得、BCA回升和N/Z误报下降。

**机制/创新：**首阶段不足不由尚未启用的Q/R造成；阶段2初次高Q使B与平方代理的协调性值得关注，但没有正式梯度分量或反事实足以判Q独因、R无效、某个系数最佳。保留累计统计核心不等于其贡献或创新已成立；本次也不否定方法家族。

**后续权限：**建议按当前合同结束本配置并完成主审记录，不自动追加匹配重放、调参、改种子、重训基线、修改阈值或清理九模型/Memory。旧九权重的清理许可不适用于这次尚待主审收尾的九权重。

## 附录A：独立七端点×22项完整主指标

下表保留全部指标及分顺序差，不只呈现六筛选。完整raw/stage-cal/first-cal/primary结果、绝对stage结果、合并counts、抽样索引和所有精确浮点值位于 `audit/outputs/saved_statistics.json`、CSV和NPY。对于F_first/F，正差一般是更多遗忘而不是改善；对于G，正差表示更多阶段获得；Brier/log-loss诊断符号按第3.2节。

__APPENDIX__

## 附录B：独立材料索引与复现说明

- `audit/scripts/01_identity.py`：当前及三层历史ZIP身份/manifest，环境；输出identity。
- `02_saved_statistics.py`：矩阵、角色、七端点、5000次配对区间、counts和六规则；输出JSON/CSV/全表/抽样NPY。
- `03_execution_evidence.py`：当前31来源、实际模块绑定、完成访问链、供给/Algorithm R/Adam/LR/损失组合、资源及旧基线回执；不调用训练或模型。
- `04_diagnostics.py`：盲分数变换/预测数量、九点日志、旧版描述性对照、表与SVG、完全手写目标例。
- `05_report_crosscheck.py`：主报告展示舍入、局部有利指标和算术解释；不是新增验收或旧新版CI。
- `run_all.py`：只顺序运行前五份独立核验，记录stdout/退出码/实际墙钟；不调用项目CLI。
- `06_package_report.py`：仅以已生成输出组织报告与清单；不训练、不拟合。
- `review_input.zip`：本次用户原附件；`reviewed_sources/`提供本次31份原字节来源和主报告/context方便离线阅读；原图另附。

所有独立输入均来自原附件或上述新手写小例。通过 `01_identity.py`可按原路径解开完整历史；在ZIP根目录执行 `python audit/scripts/run_all.py`可重放这些检查，需要已有NumPy及CPU Torch环境。脚本不会安装依赖；没有相关环境时请明确保留未执行状态，而不是用假包或伪输出替代。`manifest.json`记录除自身外全部交付文件的字节与SHA，不包含其自身的循环哈希。
'''
app=(OUT/'primary_7x22.zh.md').read_text();app=app[app.find('## O'):].replace('\n## ','\n### ')
if app.startswith('## '):app='#'+app
for k,v in {'__MAIN__':main,'__FORGET__':forget,'__PROB__':prob,'__POOLED__':'\n'.join(pooled),'__ROLES__':'\n'.join(roles),'__LOCAL__':local,'__QROWS__':'\n'.join(qrows),'__APPENDIX__':app}.items():report=report.replace(k,v)
(DEST/FNAME).write_text(report)
# Copy only intended independent code and outputs, not imported bytecode or weights.
for sub in ['scripts','outputs']:
 target=DEST/'audit'/sub
 if target.exists():shutil.rmtree(target)
 shutil.copytree(ROOT/'audit'/sub,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
source_dest=DEST/'reviewed_sources'
if source_dest.exists():shutil.rmtree(source_dest)
shutil.copytree(RR/'source',source_dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(IN/'docs/SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md',source_dest/'RESULT.original.zh.md')
shutil.copy2(RR/'review/context.zh.md',source_dest/'CONTEXT.original.zh.md')
(DEST/'figures').mkdir(exist_ok=True)
for name in ['stage_map.png','stage_map.svg']:shutil.copy2(RR/'analysis'/name,DEST/'figures'/name)
shutil.copy2('/mnt/data/review_input.zip',DEST/'review_input.zip')
readme='''# 本轮独立结果外审交付

先读 `RELATION_REVISION_RESULT.external_review.zh.md`。本包面向20261006_135937已完成B＋Q＋0.1R结果，不是启动许可。

## 内容与范围
原附件review_input.zip保持原字节；包含其内所有必要历史。reviewed_sources是本次31冻结来源另加主报告/context的方便阅读副本；figures是提交图原字节。audit保存本轮实际使用的脚本、输出、审查侧失败及修正；未包含正式正文/真值/权重/Memory本体，也没有新增训练。

## 重放
在解压根目录、有既有Python/NumPy/CPU Torch环境时执行：

```bash
python audit/scripts/run_all.py
```

第一步解开review_input.zip到input并按context解开三层历史；后四步只做本地身份、数值和源码绑定审查。不安装包，不连接项目服务器，不调用BGE、训练、评分、标签解析或完整原生结果核验器。重放会更新audit/outputs中的运行输出；应先保留原交付以验证原manifest。时间与环境字段可随重放环境变化，数值断言保持冻结定义。需从项目提供的矩阵而非真值重算MAP/AP聚合。

`06_package_report.py`是本次打包脚本，默认以本次网页工作区为路径组织副本；不是重放统计所需入口。报告来源位置以原包RR/source及RR/job为准；RR=reports/seller_alias_continual/20261006/relation_revision_result。恢复后的input目录保持原项目相对路径，不修改冻结JSON中的Linux根目录。

manifest.json覆盖除其本身之外的所有文件，记录字节数与SHA-256。不得把“脚本通过/哈希吻合”替代报告对证据范围和机制归因的限制。
'''
(DEST/'README.zh.md').write_text(readme)
# all copied payloads and exact report verified before archive publication
records=[]
for p in sorted(DEST.rglob('*')):
 if p.is_file() and p.name!='manifest.json':
  b=p.read_bytes();records.append({'path':str(p.relative_to(DEST)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
manifest={'scope':'Independent completed-result review; manifest excludes itself.','files':records}
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
final=Path('/mnt/data'); zipname=final/'RELATION_REVISION_RESULT.audit_evidence.zip'
with zipfile.ZipFile(zipname,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(DEST.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(DEST)))
with zipfile.ZipFile(zipname) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(records)+1
 for r in records:
  b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
shutil.copy2(DEST/FNAME,final/FNAME)
result={'report':str(final/FNAME),'report_bytes':(final/FNAME).stat().st_size,'zip':str(zipname),'zip_bytes':zipname.stat().st_size,'zip_sha256':hashlib.sha256(zipname.read_bytes()).hexdigest(),'files':len(records)+1,'payloads':len(records),'payload_verification':True}
(ROOT/'delivery_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
