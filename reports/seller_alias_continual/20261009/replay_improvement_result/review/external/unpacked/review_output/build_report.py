"""Build the complete Chinese review from the already audited original evidence."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

OUT=Path(__file__).resolve().parent
INPUT=OUT.parent/'input'
def read(path):return json.loads(path.read_text())
e=read(INPUT/'result/job/evaluation/evaluation.json')
d=read(INPUT/'result/job/evaluation/diagnostics/overfitting.json')
man=read(INPUT/'result/job/run/manifest.json')
col=read(INPUT/'result/job/evaluation/collected.json')
scope=read(OUT/'READING_SCOPE.json')
history=read(OUT/'historical_matrix_drift.json')
arms=('LOGIT0.1','C','S','C_plus','S_strong')
orders=('ABC','BCA','CAB')
comparisons=('C_plus-S','C_plus-S_strong','C_plus-LOGIT0.1','C_plus-C','C-S','C-LOGIT0.1','S-LOGIT0.1')
eps=('O','N','Z','final_all','A2','F_first','F','G')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                     *['| '+' | '.join(map(str,row))+' |' for row in rows]])
def f(v,n=9):return f'{v:.{n}f}'
def est(v,n=6):
    lo,hi=v['conditional_95pct_interval']
    return f"{v['mean']:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]"
def delta(name,ep,m,role='primary'):return e['comparisons'][name]['delta'][role][ep][m]
def endpoint(a,ep,m='map',role='primary'):return e['endpoints'][a][role][ep][m]

passed_table=table(['比较','原23项通过数','本轮角色'],[
    (k,f"{sum(e['comparisons'][k]['checks'].values())}/23",'继续条件之一' if k in comparisons[:3] else '预定完整报告项') for k in comparisons])
map_table=table(['方法',*eps],[(a,*[f(endpoint(a,ep)['mean']) for ep in eps]) for a in arms])
map_delta_table=table(['C_plus 对手','O MAP差及原条件95%区间','N MAP差及原条件95%区间','Z MAP差及原条件95%区间'],[
    (b,*[est(delta('C_plus-'+b,ep,'map')) for ep in ('O','N','Z')]) for b in ('S','C','S_strong','LOGIT0.1')])
order_table=table(['顺序','C_plus−S：O','C_plus−S：N','C_plus−S：Z'],[
    (o,*[f(delta('C_plus-S',ep,'map')['per_order'][o]) for ep in ('O','N','Z')]) for o in orders])
prob_table=table(['端点','C_plus−LOGIT raw log_loss','C_plus−LOGIT stage-cal log_loss','C_plus−LOGIT first-cal log_loss'],[
    (ep,*[est(delta('C_plus-LOGIT0.1',ep,'log_loss',r)) for r in ('raw','stage-cal','first-cal')]) for ep in ('O','N','Z')])
class_table=table(['方法','O：群宏precision','O：群宏recall','O：群宏F1'],[
    (a,*[f(endpoint(a,'O',m)['mean']) for m in ('precision','recall','f1')]) for a in ('LOGIT0.1','S','C_plus')])
first_rows=[]
for o in orders:
    indices=np.flatnonzero(np.asarray(col['domains'])==o[0])
    vals=[]
    for p in (f'{o}_LOGIT0.1_stage1',o+'_shared'):
        file=INPUT/'result/job/evaluation'/col['points'][p]['raw']['matrix']['path']
        vals.append(float(np.load(file,allow_pickle=False)[indices,12].mean()))
    first_rows.append((o,o[0],*[f(v) for v in vals],f(endpoint('C_plus','F_first')['per_order'][o])))
first_table=table(['顺序','首个实际域','本轮LOGIT首域MAP','记录表共享首域MAP','C_plus 的 F_first'],first_rows)
diag=read(OUT/'diagnostics_audit.json')
labels=('有相应迹象','证据不足','未见明确迹象')
diag_count_table=table(['方法','fit：有迹象／证据不足／未见明确','cache：有迹象／证据不足／未见明确'],[
    (a,*['／'.join(str(diag['logical_counts'][a].get(role+'/'+label,0)) for label in labels) for role in ('fit','cache')]) for a in arms])
diag_examples=[]
for p,role,dom in [('CAB_C_plus_stage3','fit','B'),('BCA_C_plus_stage3','cache','B')]:
    v=d['points'][p][role][dom];i=2
    diag_examples.append((p+'/'+role+'/'+dom,str(v['fit_or_cache_groups'])+'/'+str(v['valid_groups']),
                          f(v['train_raw_curve'][0][i])+'→'+f(v['train_raw_curve'][5][i]),
                          f(v['valid_raw_curve'][0][i])+'→'+f(v['valid_raw_curve'][5][i])))
diag_example_table=table(['情境','训练侧群／valid群','raw log_loss：训练侧epoch1→6','raw log_loss：valid epoch1→6'],diag_examples)
diag_interval_rows=[]
for p,role,dom in [('CAB_C_plus_stage3','fit','B'),('BCA_C_plus_stage3','cache','B')]:
    v=d['points'][p][role][dom]
    for key,name in [('epoch1_to_6_train_benefit','训练改善'),('epoch1_to_6_valid_benefit','valid恶化'),('epoch1_to_6_gap_widening','差距扩大')]:
        value=v[key]['mean'][2];lo,hi=v[key]['conditional_95pct_interval'][2]
        if key=='epoch1_to_6_valid_benefit':value,lo,hi=-value,-hi,-lo
        diag_interval_rows.append((p+'/'+role+'/'+dom,name,est({'mean':value,'conditional_95pct_interval':[lo,hi]},9)))
diag_interval_table=table(['情境','方向','均值及原条件95%区间'],diag_interval_rows)
teacher=read(INPUT/'result/analysis/output/teacher_errors.json')
teacher_table=table(['方法','D0','D1','mean(e0×e1)','mean((e1−e0)^2)'],[
    (a,*['N/A' if row[k] is None else f(row[k]) for k in ('D0','D1','residual_cross_mean','residual_difference_mse')])
    for a in ('C','S','C_plus','S_strong','LOGIT0.1')
    for row in teacher['rows'] if row['point']==f'ABC_{a}_stage3' and row['epoch']==6])
history_table=table(['固定MAP端点','历史LOGIT','本轮LOGIT','本轮−历史'],[
    (ep,*[f(history['endpoint_descriptive_drift']['LOGIT0.1']['primary'][ep]['map'][k]) for k in ('old','new','new_minus_old')]) for ep in eps])
scope_table=table(['容器','成员','人工全文','人工节选','结构/搜索','其余未人工逐字阅读'],[
    (p,sum(v.values()),v.get('全文阅读',0),v.get('节选阅读',0),v.get('结构/字面搜索',0),v.get('未人工阅读',0)) for p,v in scope['summary'].items()])
machine=Counter(r['machine_scope'] for r in scope['files'] if r['container']=='P0')

identity={
    'review_date_Asia_Shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'requested_model':'GPT 6 Pro (explicit user request)',
    'observed_page_selector':{'observed':False,'changed':False,'evidence':'No current selector screenshot or UI observation was obtained; previous reviews do not prove this turn selector.'},
    'instance_identity_description':'GPT-6 Astra Pro',
    'identity_evidence_level':'Identity supplied to this conversation; not independent verification of service-side routing or exact model weights.',
    'reviewer':'Current instance only',
    'subagents':0,'parallel_reviewers':0,
    'execution_environment':'ChatGPT web review workspace, Python 3.12.14 / NumPy 2.3.5',
    'project_linux_access':False,'weights_loaded':False,'training':False,
    'formal_text_or_labels_read':False,
    'input':read(OUT/'input_identity.json'),
}
(OUT/'REVIEW_IDENTITY.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')

report="""# 2026-10-09 五臂记录重放完成结果：独立结果外审

审查对象：review_input(9).zip；当前实例独立完成，未委派辅助 agent、并行审查者或新会话。
审查范围：本轮新增执行证据、完成结果、共同过拟合诊断、LOGIT 历史复现差异和主执行者的结论。复用已关闭的合同／实现结论，只为本轮结果判读及具体新差异回查直接依赖。

## 一、结论与分级

支持将本轮作为已完成、可据其实际结果收尾的固定配置开发实验。C_plus 没有达到冻结的继续条件；相对 S 和 C 的新域 MAP 负结果成立于本轮原有条件统计口径。共同过拟合诊断的保存曲线、区间和判读一致。主执行者没有把局部 raw log_loss 过拟合直接写成全部 MAP 损失的原因，这个限定是正确的。[E1–E7]

确认一项 REPRODUCIBILITY_DEFECT：本轮 LOGIT0.1 未精确复现历史，且旧首域入口的数值运行设置没有在新实际入口同样落实、有效运行值没有完整保存。其影响必须保留，不能沿用旧“NO_OPEN_BLOCKERS”把本轮新反例重新表述为“完全复现”。但附件没有证明整个五臂实验无效，也没有证明四个记录表臂内部的负结果由该 LOGIT 差异产生。对本轮 LOGIT 的比较可以保留为“本轮实际重训系统条件下的比较”，不能等同历史系统已经被精确复现或 D1 的纯净独立效应。[E8–E10]

|分类|本次判断|影响范围与处置|
|---|---|---|
|SCIENTIFIC_BLOCKER|0 项已证实|未发现标签开放顺序、实际群配对、冻结指标／判据或结果汇总方面足以否定本轮结果的具体反例。这个结论只覆盖提供的保存证据及本次实际核对。|
|REPRODUCIBILITY_DEFECT：R1|1 项确认，限 LOGIT 历史精确复现与运行条件来源|0/27 历史角色矩阵复现、诊断前首步数值差异、旧新入口设置差异、有效值记录不足共同构成实质复现缺陷。以明确偏差记录和收窄主张完成本轮结果处置；不把根因标成已经解决。|
|未证实疑点|原因尚未隔离|不能断言 TF32 是唯一根因、不能断言全部由诊断插入导致，也不能据一个 gradient_norm 标量声称首步全部权重已发生哪种差异。|
|有限解释|校准、遗忘起点、小缓存、多重重叠判断、条件区间等|保留已经正确给出的边界，具体见第六至十节；不据此要求改指标或重做实验。|
|OUT_OF_SCOPE_OVERDESIGN|本轮不需要|新种子、系数搜索、选最好 epoch、另造数据／确认集、恢复框架、强制全量回归、上传权重／Memory／标签、为了翻转结果重训。|

C_plus 的性能失败是本轮实验答案，不是实验无效的证据。本轮应结束，不从 23/23 的单一对手比较推导自动继续。[E1、E4]

## 二、模型身份、输入身份与审查环境

模型请求、页面证据和身份自述分别记录：

1. 用户文字请求是“GPT 6 Pro”。
2. 本实例本轮没有读取或改变页面可见选择器，未取得可独立核对的本轮选择器截图或 UI 快照。旧外审关于选择器的记录不移植到本轮。
3. 本会话提供给当前实例的身份描述是“GPT-6 Astra Pro”。这属于当前实例身份信息，不是独立的服务端路由／权重鉴定；不伪称已经由页面确认了用户请求的精确产品标签。

完整身份记录见 REVIEW_IDENTITY.json。全部判断由当前实例作出，没有辅助 agent 或并行审查者。[E12]

本次实际核对的上传包为 47,913,772 字节、1,629 个成员，SHA-256：

5b410fa7b80fb9fd89a02555a898014f3877e53620dfc170a2ffcc02870da2d1

package_manifest.json 登记 1,628 个载荷、63,967,574 字节；逐项大小和 SHA 核对无差异、无缺件、无额外成员。外层未压缩总字节数为 64,512,166（包含该清单）。这些只证明输入身份，语义判断来自下述源码追踪和保存数据复算。[E2、E12]

本轮背景中的历史增量包为 9,907,776 字节，SHA-256 c5cba199e4c1a2ffb0aab47a476a26971a29306289ee306970488cfdc751940e；其中原结果包为 9,614,123 字节、335 成员，SHA-256 dbed856f06482e29b444ce45b5b44973083bbd28964516ae2cb5ccf5c8be3c38，与本轮 policy 登记相同。只读取本轮历史对照需要的矩阵、登记文件及直接合同，没有递归展开里面的旧实现审查输入包。[E8、E12]

所有新增计算均在本网页审查工作区执行：Python 3.12.14、NumPy 2.3.5。工作区运行在 Linux 容器这一事实，不等于访问了项目 Linux 服务器。没有项目连接、载模、GPU、训练、优化器更新、校准拟合或正式文本／标签读取；没有推断或恢复隐藏标签。脚本只用已保存分数、群指标、计数、身份、元数据和源码。重算结果单独保存，未替换原 evaluation.json、overfitting.json 或原统计。[E12]

## 三、完成、恢复、来源、开放顺序与预算

### 3.1 数量与共享首域

实际编排不是五臂各自独立训练全部首域。每个顺序的记录表家族有一个 shared 首域加四臂各两个后续阶段，即 9 个物理阶段；LOGIT 为其自己的 3 个阶段。三个顺序的数量为：

|量|记录表家族|LOGIT0.1|合计|
|---|---|---|---|
|物理阶段末状态|27|9|36|
|物理更新|7,776|2,592|10,368|
|梯度群呈现|14,688|4,320|19,008|
|物理 epoch 诊断点|162|54|216|
|逻辑阶段点|36|9|45|
|逻辑 epoch 点|216|54|270|

6 个物理首域阶段只呈现 current；30 个后续阶段呈现 current 和一个 history。因此更新为 36×288，梯度群呈现为 6×288＋30×576。三个记录表 shared 首域各有六个实际诊断点，在四臂中作为别名复用，增加 54 个逻辑 epoch 点。collected.logical_points 的 45 个映射与该规则完全一致；没有把别名当额外训练或独立样本。[E1–E3]

对全部 36 份更新文件核对了每阶段 288 行、逐步学习率、当前群完整六轮日程、history UID、实际目标分解、Adam 阶段末步数和非零有限 gradient_norm。按原命名种子独立重建了 36 条 current 日程、30 条 history draw 日程及 Algorithm R 存活成员；五臂对应阶段的群供给一致。全部保存目标分解相对记录 total 的最大绝对舍入差为 5.066394814789987×10^-8，在原代码自身的数值核对范围内；没有更改科研判据。[E3、E12]

当前 fit 144 群、calibration 36 群、valid 60 群身份互斥，共 240 个不同群 UID；每实际域分别是 48／12／20。每个 valid UID 在所有方法、顺序和阶段保持同一行位置。60 群并没有因三个顺序变成 180 个独立群。[E3、E4]

### 3.2 来源与完整恢复证据

当前 28 项 source 的真实大小／SHA 与 execution、run manifest、collected、evaluation、formal_gate、现行 CPU／GPU 资格记录一致，background/sources 亦对应。五臂 run.config 最终使用本轮 runtime，不能用被继承文件中的旧预算数字覆盖当前 72 小时。72 小时补充授权和冻结 shell／validate_gate 的 259200 秒配置相符。[E1–E3、E9]

36 个 point 均保存 full_restore_verified=true、阶段末 Adam 步数和完整状态／推理状态摘要。每个顺序四个记录表分支的 starts 在完整状态摘要、Memory 摘要、RNG 与 Adam 步数上相等，绑定同一 shared 首域。实际 checkpoint 先恢复完整 model/Adam，再在全 12 校准群和 60 valid 群重评分逐位核对；随后另存并恢复推理权重。不能把保存的布尔字段单独当证明；这里有实际调用顺序、分支摘要、分数文件和已关闭资格证据的相互支持。[E3、E9]

本网页没有权重和 Memory 正文，也按范围没有请求这些文件。因此“独立核对了恢复收据、调用路径与结果连接”是可支持的说法；“本网页重新完成 36 次真实模型恢复”则不成立。完整推理载荷的当前服务器存在性也不是这次远程读取核验的对象。[E12]

全部 216 份诊断记录的 epoch、stage_step、Adam 步数、方法／顺序／实际域、fit/cache/valid 身份及数组形状相符。36 个 epoch6 raw 分数数组与阶段末 raw 分数逐位一致；对应 epoch6 的三列 valid 群指标也与阶段末 raw 指标对应列逐位一致。72 份 stage-cal／first-cal 分数均是所存正斜率 affine 映射作用于对应 raw 分数的精确结果，三个保存角色的 14 个排序／曲线列均相等。[E3–E6]

### 3.3 开放次序

实际入口顺序是：完成全部训练 → blind_gate 检查所有阶段、配对、恢复、来源、预算和完整盲诊断 → 写 before_valid → 调用一次 development 标签解析 → 保存全部阶段矩阵／计数及全部 epoch valid 指标 → 运行统计 → 完成收据。before_valid 登记 train=1、valid=0；completion 与最终 access 均为 train=1、valid=1、heldout=0、owners=0，completion 绑定本轮 evaluation 与 overfitting 文件 SHA。包内当前 job 没有 failure.json。[E2、E3]

这支持“本作业在完整盲门后统一开放 valid 标签”的判断。这里的盲性指标签及据其计算的验证指标没有在训练期间用于选择；valid 文本的无标签前向本就属于冻结诊断，不能写成训练期完全没有接触 valid。上述保存材料不被抬高为对项目服务器所有进程的取证证明。[E1–E3]

### 3.4 资源和回传

wrapper 起止为 2026-10-08 15:25:57 至 2026-10-09 13:38:39（+08:00），exit_code=0，差 79,962 秒，即 22 小时 12 分 42 秒。内部 completion 计时 79,961.574720 秒，二者只是外层秒级时间戳与内部计时口径不同，并不冲突。468 条更新进度记录覆盖 36 个物理阶段，每阶段 13 个固定进度点。[E2、E12]

|项目|保存的实际值|当前上限|
|---|---|---|
|正式内部总时长|79,961.574720 秒|259,200 秒（72小时）|
|CUDA reserved 峰值|9,430,892,544 字节，约8.7832 GiB|28 GiB|
|进程 RSS 峰值|7,058,984,960 字节，约6.5742 GiB|64 GiB|
|输出峰值|53,538,384,484 字节，约49.86 GiB|64 GiB|

这是本作业记录的指标，不是整台服务器所有进程的峰值。63.51—63.80 小时为启动前最大形状成本与预留所构成的保守投影；实际完成约 22.21 小时既不违反预算，也不把该投影变成实测耗时或置信区间。现行 CPU／GPU 累计资格时间分别 108.682801／252.019093 秒，来源与72小时版本一致；没有把本网页计算计入项目资格账。[E2、E9、E12]

已保存的完整历史状态字节记录范围为 241,869—358,824，未见超过 1 MiB 的收据；该数包含所存摘要列明的完整序列化状态，不只教师向量。没有直接读取 Memory 正文，故不把这个收据范围夸大成重新测得了整个运行的每个瞬时状态。[E3、E12]

sync_manifest 的 1,591 份小文件、51,293,424 字节已在本网页按其登记逐项核对，无大小或 SHA 差异。72 份未随包提供的模型／Memory 载荷合计 46,988,038,388 字节，与 point 记录及排除清单一致。执行者关于保留这些载荷、未因负结果删除的说明可与清单相容；本网页没有代执行 SFTP、服务器进程检查或删除。[E2、E12]

## 四、指标角色、端点与原23项是否正确

### 4.1 角色和端点

22 列包含 4 个曲线指标、8 个概率／分类指标及 10 个检索指标；MAP 是群内逐查询 AP 的平均，再对群汇总，不是 378 个账号对的 AP。raw／stage-cal／first-cal 是三个实际保存角色；primary 是派生组合：[E4]

|角色|实际含义|
|---|---|
|raw|A0/eval 的原始 logits 及由其计算的指标|
|stage-cal|使用当前阶段该域固定12个校准群拟合的正斜率 affine 映射|
|first-cal|持续使用该路径首域冻结校准映射|
|primary|raw 的14个曲线／检索指标＋stage-cal 的8个概率／分类指标；不是第四个独立模型或映射|

设 m_t(j) 是阶段 t 在第 j 个到达域上的指标。O=[m3(1)+m3(2)]/2；N=[m2(2)+m3(3)]/2；Z=m3(3)；A2=m2(2)；final_all=[m3(1)+m3(2)+m3(3)]/3。A2 是第二到达域，不是实际字母域 A。F_first=b1(1)−b3(1)，F=[b1(1)−b3(1)+b2(2)−b3(2)]/2，G=[b2(2)−b1(2)+b3(3)−b2(3)]/2；其中 Brier／log_loss 的 b 取负，其余沿用指标方向。因此 F_first/F 越低越好、G 越高越好，绝对概率损失端点仍越低越好。[E4]

final_all=(2O+Z)/3、A2=2N−Z，也说明八端点不是八份独立证据。actual-domain fields 先将三个固定顺序的贡献对应到同一实际域、同一群行，再平均顺序。PCG64(20260930) 产生原 5000×3×20 群抽样；同一个实际群的不同方法／阶段／顺序／角色共同使用抽样。没有抽三个“独立种子”，也没有把账号对当独立 bootstrap 单元。[E4、E12]

### 4.2 独立复算范围

本网页从108个原60×22群矩阵（142,560个数值）重建五臂×四角色×八端点、全部七比较及候选stage-cal减对手raw的原统计。共9,680个指标汇总对象，含均值、三个逐顺序值和区间两端，共58,080个数值，全部与原 evaluation 逐项一致，最大差0。[E12]

另从108组群级混淆计数复算38,880个分类指标数值，以及432个 pooled／分实际域分类汇总块，均与保存结果一致；群宏均值与先加总计数后的 pooled 结果没有混用。两个大文本 all_metrics.txt、all_curves.txt 与原 JSON 字段按原格式完整一致。展示正确与原始统计正确分别核对，未拿展示文件替代数组审查。[E4–E7、E12]

### 4.3 原23项

原 comparison_checks 的组成是：O MAP 均值>0、O MAP 区间下界>0、O Recall@5均值>0、N MAP均值≥0、N Recall@5均值≥0；O/N各四项同角色 AP／ROC-AUC／Brier／log_loss 非退化，加各两项“候选校准后 Brier／log_loss 对对手 raw”的均值保护；Z再有MAP／Recall@5／AP／ROC-AUC／Brier／log_loss六项均值保护。总计23项，只有 O MAP 的一个门明确要求区间下界>0。23/23绝不是23个统计显著改善。[E4]

@@PASS_TABLE@@

继续条件只取前三组并且三组均须23/23，实际整体 false。全部161个布尔判据独立复算一致，逐项真值表见 CHECKS23.zh.txt。原5项验收和其历史false不被本轮23项追改。[E1、E4、E12]

C_plus−S 和 C_plus−C 各只通过四个跨角色 raw 对手损失保护项，其余19个同角色主条件失败。校准比对手未校准更好不足以判为方法胜出。[E4、E7]

C_plus−S_strong 失败五项：O MAP区间下界严格正、O Recall@5严格改善、N MAP非降、O AP非退化、Z MAP非退化。N MAP差仅约−0.000020259，但原均值零容忍保护就是失败，不能因为差值小或区间跨零事后放宽。[E4、E7]

C−S 失败七项：O MAP均值与区间、O Recall@5、O Brier、O log_loss、Z AP、Z Brier。它仍保持历史16/23的结果，不被新候选的比较替换。[E4、E8]

## 五、性能结果的正确解释

### 5.1 原 primary MAP

@@MAP_TABLE@@

### 5.2 C_plus 的主要配对差值

下表是原条件95%区间，差值均为候选减对手；不是本网页新增统计选择。[E4、E12]

@@MAP_DELTA_TABLE@@

相对 S，O MAP 尚不能确定方向；N、Z MAP 的原区间均在零以下。损失约0.6123和0.9344个MAP百分点（绝对指标差×100），不是0.61%／0.93%的相对变化。相对 C 也有同方向的新域损失。C_plus−S 的 final_all 差为 @@FINAL_S@@；对 C 为 @@FINAL_C@@，原区间也均负。这不是仅一个顺序翻转均值：[E4、E7]

@@ORDER_TABLE@@

N、Z 在三个顺序的观察差值都为负；O 则有顺序异质性，不能把平均区间跨零写成每域一致保护。应保留“本配置加项策略伴随新域学习代价”的结果，不能将旧域没有确定差异解释成新策略已实现更好保持。[E4]

相对 S_strong，O/N/Z MAP 的区间均跨零，主要排序收益没有建立；区间跨零也不是等效性证明。N 端点 AP差为 @@SS_AP@@，stage-cal Brier差为 @@SS_BRIER@@，stage-cal log_loss差为 @@SS_LL@@。这些是应保留的局部有利条件结果，但量级很小，不替代MAP主问题和五项失败，更不能推出D1普遍有效或完全无用。AP 本身仍是排序／曲线指标，这组证据应准确称为“局部 AP 与校准概率收益”。[E4]

相对本轮LOGIT，旧域O与Z的MAP区间为正；N虽观察均值为正并通过原均值门，区间跨零。MAP的G差为 @@LOGIT_G@@，不能从末期能力较高推导“新域增益也更大”。C与S同样对本轮LOGIT达到23/23；本轮没有显示C_plus刷新了记录表家族的主要观察表现。[E4、E7]

四个记录表臂与LOGIT在账号汇总架构／记录对表架构、A1监督、计算路径及历史表示上不同；本轮结果是完整系统比较。C_plus−S_strong 也只匹配teacher系数和，不匹配实际梯度范数、裁剪作用或随后轨迹。不能把跨架构全部收益归给D1，亦不能把同一瞬时损失的代数差当成全程相同student、相同新出生teacher的比较。[E1、E3、E9]

## 六、概率退化、分类取舍与遗忘获得起点

### 6.1 raw 和校准系统的结论不同

@@PROB_TABLE@@

raw log_loss 的三个差值均为正、原区间均为正，表明C_plus相对本轮LOGIT的未校准概率损失更差；stage-cal与first-cal三个端点均反向更好。first-cal说明该校准系统优势并非只依赖末阶段重拟合，但它仍不能变成“raw概率全面更好”。对S，C_plus的O/N/Z raw log_loss原区间也全部不利；stage-cal的N log_loss差为 @@S_N_LL@@，校准并没有抹去这个新域损失。[E4、E7]

评价器以 sigmoid(logit) 得到概率，将其截断至[10^-15,1−10^-15]后算未加权log_loss。它不等于包含BCE、query-rank、known-top5-hard和teacher项的训练总目标，也不能一般性等同无限幅BCEWithLogits。共同诊断用raw/A0/eval的同一指标，没有逐epoch校准或选择。[E5、E6]

### 6.2 群宏F1和低召回

C_plus−S 的 O 群宏F1差为 @@S_O_F1@@。旧域MAP区间跨零不能覆盖这个分类退化。下表采用primary中的stage-cal分类列，阈值固定为校准后概率0.5（校准后logit为0）：[E4、E7]

@@CLASS_TABLE@@

这些是群宏precision／recall／F1，不能拿宏precision的补数直接当作汇总误报比例；文件另有已核对的pooled计数口径。约8%—10%的旧域召回与部分排序／校准优势可以同时存在。现有指标不提供严格自动认定同控身份的部署保证，也不授权事后调阈值。[E4、E12]

### 6.3 遗忘量不能脱离各自获得点

@@FIRST_TABLE@@

C_plus 的平均F_first为−0.022359141，本轮LOGIT为+0.029540193；但CAB记录表首域的获得点低得多。先学得较弱、后来从较低起点提升，会改变遗忘量，不能直接解释成跨架构“抗遗忘机制更强”。C_plus在BCA仍有正的观察F_first；“三个顺序都没有遗忘”不成立。[E4]

四个记录表臂共享相同首域获得点，因此内部F_first比较没有这项首域起点差；综合F还含各臂第二阶段的第二域获得值，这些获得点不能一概视为相同。C_plus−S的F_first为 @@S_F_FIRST@@，仍未建立额外保持优势。最终绝对能力、相对获得点的保持以及新域G是不同问题，报告已将它们分开，这是正确的。[E4、E7]

## 七、共同过拟合诊断的统计与全部曲线

### 7.1 正确的抽样单位和判读

诊断固定比较阶段内epoch1→6，保留全部六点。当前fit为48群，与同实际域20个valid群比较；缓存按出生实际域拆分，与该旧域的20个valid群比较。fit/cache与valid身份互斥，分别独立抽各自的群；同一群集合跨epoch和方法共享命名种子及抽样。实际元数据也证实对应方法的群顺序相同，而不是仅在代码中省略arm名就假定配对成立。[E5、E12]

程序先统一效益方向：MAP/R5上升为改善，log_loss下降为改善。“有相应迹象”要求训练改善区间下界>0、valid效益变化区间上界<0、gap扩大区间下界>0三者同时成立；若仅观察值具备这些方向则为“证据不足”；否则为“未见明确迹象”。它不是只看gap为正，也没有新设统一gap阈值。[E5]

从含73,224个数值的保存诊断数组出发，独立复算全部76个非空情境的9,044个原曲线／变化／区间及群数数值，均与保存结果一致；228个判读完全一致。首域缓存N/A；CAB第三阶段五臂的C域缓存为空也均为N/A，没有补零或从档案补回已淘汰群。[E5–E7、E12]

36个物理fit情境与40个非空cache域情境，各三个指标，合计228个物理情境—指标判读。五臂逻辑展示各有fit27项、cache24项，共255行；其中共享首域多出27行逻辑别名，不是新的独立判断。[E5、E7]

@@DIAG_COUNTS@@

该表的计数与原summary完全一致。但228项本身也存在群、路径、指标重叠，不能据此执行“某方法过拟合次数更多”的方法级检验。全部15个“有相应迹象”清单另列于SUPPORTED_DIAGNOSTICS.zh.txt：LOGIT当前fit2项、cache9项，C_plus当前fit1项/cache1项，S_strong当前fit1项/cache1项；C/S为零。零项不证明没有过拟合。[E5、E7、E12]

### 7.2 C_plus 的两个明确 raw log_loss 情境

@@DIAG_EXAMPLES@@

@@DIAG_INTERVALS@@

CAB第三阶段B域的MAP虽有训练改善与valid观察下降，valid MAP区间跨零，原判读是“证据不足”，不能升级成MAP已有明确过拟合证据。BCA第三阶段缓存B域只有2个存活群，区间条件于这两个群的经验分布；它不能覆盖整个旧域缓存选择与重训不确定性，也不能替代全历史泛化估计。S_strong在这两个raw log_loss情境同样有迹象。[E5、E7]

六轮曲线并非要求逐轮单调，固定epoch1→6判据与局部曲线回落可以并存。不能根据第4轮较低的valid值改选第4轮。该诊断从已经完成48步后的epoch1开始，也不对最初48步或所有可能的中间行为给出“没有过拟合”的证明。[E5]

现有材料支持“局部训练—验证概率损失分离，与新域MAP损失并存”。没有验证中介因果链，也没有隔离teacher系数是唯一原因；所以不能把全部N/Z的MAP损失都归因于这些raw log_loss过拟合情境。对LOGIT的诊断计数同样仅指本轮实际重训路径，不能回填历史没有保存的首域曲线。[E5、E8]

## 八、D0／D1来源残差与泛化收益

180个有缓存的物理epoch情境保存了teacher误差；36个首域点没有缓存。D1使用独立固定命名的诊断A1分组，种子不含epoch和arm，以便同阶段／同群跨epoch和记录表方法比较；它并非每一步随机训练A1的损失日志。[E5、E7]

以下保留执行者给出的ABC第三阶段epoch6描述点，全部180点均有保存，未将该点升级为新主端点：

@@TEACHER_TABLE@@

在这个点，更小的来源输出残差没有对应为已建立的整体MAP优势。C_plus较S_strong有更小D1、更大D0，不能从中得出有用的独立参数梯度方向。跨臂第二阶段新保留群的出生teacher也可能随各自训练路径分化；残差是相对本臂相应冻结来源的拟合误差，不是对一个跨臂恒定、已验证正确“真函数”的泛化误差。[E3、E5、E9]

保存标量满足 mean((e1−e0)^2)=D0+D1−2mean(e0×e1)，最大算术差1.3877787807814457×10^-16。612个已存均值标量以明确逐项相加的相同次序核对均精确一致。最初本网页用自身运行环境的sum得到少量至多1.1102230246251565×10^-16的舍入差，另行保留该原始输出；没有改原表。它与本轮LOGIT训练轨迹差异不是同一问题。[E12]

两视图残差差MSE为正只说明输出残差不完全相同；残差交叉均值不是参数梯度夹角，较小D0/D1也不建立方法新颖性、一般历史保护或真实市场有效性。[E5、E7]

## 九、R1：LOGIT历史复现差异的证据、分类和最小处置

### 9.1 已确认的数值事实

本网页直接读取登记历史矩阵与本轮矩阵，以原group_ids／domains／metric_columns逐项配对复算。C为27/27、S为27/27，数值和矩阵数据字节均相同；LOGIT为0/27。每27项=三个顺序×三个阶段×三个保存角色，不是27个独立训练。primary是派生角色，不重复计入。[E8、E12]

跨所有群和指标的最大差1包含离散分类指标，不能叫“MAP差1”。本次对MAP列另核得到的最大群级绝对差为0.09530132565846866；这只是已定义MAP列的描述性差异范围，不是新显著性检验。C/S矩阵相同也只证明该输出层复现，不能据此宣称所有权重、Adam轨迹、每个epoch的内部状态都与历史逐位相同。[E8、E12]

下面为同一组既定MAP端点的旧／新均值与算术差，没有重做历史统计选择或构造新的置信区间：

@@HISTORY_TABLE@@

新LOGIT的O比历史低约0.003198，而N/Z分别高约0.002881／0.004454。不能把它叙述成“新LOGIT所有端点被系统削弱，因而制造了记录表优势”。同样，这张表也不能修复其历史精确复现资格：对C_plus相对LOGIT的O差影响方向与N/Z不同，现行冻结比较必须继续用本轮已登记结果，不替换为事后挑选的旧／新较有利端点。[E8、E12]

### 9.2 首步差异确实早于首次epoch诊断

旧manifest的初始模型摘要与新manifest的LOGIT初始化摘要同为：
e52523ec2f44dc7baa5dfad7ea595bbdc43f38aa3883114cdf62b5805dc6f2c7

旧新公开输入身份、分区登记、预训练归档身份一致；三个首域的全部288步current UID日程一致，current dropout命名种子的派生也一致。旧更新元数据由旧manifest绑定，旧.npy由该元数据绑定。实际首步保存值如下：[E8、E10、E12]

|顺序|历史第1步gradient_norm|本轮第1步gradient_norm|
|---|---|---|
|ABC|0.5013665556907654|0.5013661980628967|
|BCA|0.5003510117530823|0.5003511905670166|
|CAB|0.4996473193168640|0.4996475875377655|

三条首域路径在288步的每一步都至少有一个非计时标量不同。第1步的首个可见差异仅为gradient_norm；首次新epoch诊断在第48步之后。因此“全部差异都是逐epoch诊断插入造成的”与保存时间顺序矛盾。[E8、E12]

必须同时限定：这个证据只定位最早已记录的差异，不能仅靠一个范数标量断言全部参数梯度、首步权重或整条后续差异的具体因果链已经查明。旧新首域调用共同parent.update数学主体；旧路径首域arm为seq、新路径为logit，但首域history均为None，相关history分支不进入目标。旧入口还先做未训练模型评分，并在指定更新调用observe；这些可读的路径差别没有被逐项数值隔离，也不应随意指定其中一个为根因。[E3、E10]

### 9.3 来源差别与有效值缺口

旧真实首域入口step28_bge_continual_run.py的全文件SHA为
bc425a88e502f9802aac81a0ac0977368606ed12c4e87e4bb2a6020ce9804db9，
与旧manifest登记相符。execute第520—523行显式设置：

|数值运行项|旧实际入口|新实际execute／preflight|
|---|---|---|
|torch.use_deterministic_algorithms|True|没有对应调用|
|torch.backends.cuda.matmul.allow_tf32|False|没有对应赋值|
|torch.backends.cudnn.allow_tf32|False|没有对应赋值|
|torch.backends.cudnn.benchmark|False|没有对应赋值|

当前source内仍然存在这些文字，但它们位于旧run／execute函数。导入一个模块、复用它的load_model或update，不等于执行其旧run／execute。新execute只在此处设置线程数及GPU内存比例；所调用preflight核对环境、亲和性和资源，不设置上述四项。新shell也没有记录能代替这四项的有效运行值。实际路径的差别由源码直接支持。[E3、E10]

但是“没有显式重设”不等于已经观察到新进程四项值必然分别为False／True／True／True；库默认、环境或间接影响在本包没有逐项有效值收据。不能事后拿当前默认值补写当时实际值，更不能写“TF32已被证明是唯一根因”。旧入口的已知设置、新入口的设置缺失、旧新最终数值差异，是三层相互有关却不能互相替代的证据。[E8、E10]

已关闭的CPU完整capture中性与原生部件中性证明的是各自被测当前路径中插入诊断前后的一致性；它们没有证明新执行数值环境等同历史首域入口。因此该历史差异不被资格PASS或旧外审NO_OPEN_BLOCKERS消除。[E9、E10]

### 9.4 分类与影响范围

R1应正式登记为“历史LOGIT精确复现失败及数值运行条件来源不完整”的REPRODUCIBILITY_DEFECT，不只是无关格式问题。它直接影响以下主张：

— “本轮LOGIT完全复现了历史LOGIT”：不支持。
— “旧新差异只来自诊断插入／只来自TF32”：不支持。
— “对本轮LOGIT的23/23等于对精确复现的历史强对手通过”：不支持。
— “本轮各保存系统的原指标、原条件区间及观察比较”：可以保留，但必须写明基于本轮LOGIT重训实现。
— “C_plus对S／C的负结果和对S_strong未建立主要MAP收益”：不依赖历史LOGIT端点；R1不提供推翻这些内部比较的证据。

附件没有出现足以据此判整项实验无效的科研反例。这个结论不豁免R1，也不承诺将来明确相同数值条件后重训必然得到相同结果。[E4、E8–E10]

### 9.5 无需新训练的最小必要处置

本次即可完成的处置是：保留全部原结果和来源字节；把报告第八节“待结果外审处置”更新为明确的R1分类、证据和影响范围；附上本节旧／新既定端点差表、首步来源绑定及“四项有效值未记录／未知”；把所有LOGIT相关方法主张收窄为本轮重训系统比较。不能追填有效值、替换基线、重开标签或悄然修改冻结source。[E8、E10、E12]

可直接采用的收尾文字：

“本轮完成结果和原统计已审查，development_criteria_pass=false。LOGIT0.1历史精确复现未成立，登记为R1复现缺陷：C/S历史角色矩阵27/27相同，LOGIT0.1为0/27；首个已存数值差异早于首次epoch诊断，旧新入口的显式数值运行设置不同，新运行有效值未完整记录，原因未隔离。LOGIT相关比较仅指本轮实际重训系统；四个记录表臂内部比较和C_plus新域负结果保留。原始结果与历史结论不改，本轮结束。” 

该段处理的是结果发布口径与缺陷记录，不把“如实处置”写成“数值原因已经修复”。如果将来另获授权研究数值根因，应另行明确恢复旧设置及记录有效值的范围；本轮没有发起此类实现、实验或重训，也不以其为当前负结果收尾的前提。

## 十、推断边界是否充分

主报告已经保留了主要边界，审查接收这些限定，并在正文中补足以下容易误读的连接：

1. 单seed s0。三个循环顺序复用同一实际域群，不是三个训练seed、不是三个独立数据集，也不代表遍历了所有到达顺序。
2. bootstrap条件于本轮已训练系统、固定顺序、固定数据及已定校准。它不覆盖重新训练、生成新数据、运行数值条件、研究路径和重复开发valid选择的不确定性。
3. 本轮训练期盲存分数、训练后一次开valid是执行纪律；已经反复用于开发的valid不会因此成为新独立确认集。
4. 多指标、多端点、重叠情境和原23项有依赖，不能把通过数／迹象数用作独立试验次数、方法级总体显著性或创新性证明。不事后改原区间、多重性口径或均值保护门。
5. 28账号合成群，每查询固定27候选并保证1或2个正候选。MAP／Recall@k不直接证明全市场、没有正候选的检索或真实中文暗网卖家的有效性。合成受控结论不冒充真实市场证据。
6. 这次是C_plus固定配置未达已定改进目标的证据；不自动否定所有记录重放、重组teacher或整个方法家族。亦不据局部胜出重提已关闭的“创新性已经建立”。[E1、E4–E10]

不要求新增授权仪式、状态机、通用安全加固、整个历史的递归重审、额外种子或新的科研指标。这些不能修复一个主要是如实记录与限定主张即可处置的已完成结果问题。

## 十一、计算、失败与跳过事实

本次科学核对主脚本audit_saved.py的execution、metrics、diagnostics、historical四阶段均首轮exit0。计算只重建冻结日程、已存目标分解、原端点／原bootstrap／原23项、原epoch1→6诊断以及登记历史差异；未执行原正式runner主入口。每阶段原始stdout与带环境、耗时、输入身份的run JSON均保留。[E12]

网页辅助核对确有两次自身错误，不能隐藏，也不能嫁接为被审项目缺陷：

— audit_displays.py首次把pooled分类字段误写为pooled_all，KeyError，exit1。按实际JSON字段pooled修正后第二次exit0；首次源码、错误stdout和失败JSON原件保留。
— audit_provenance.py首次把sync_manifest.source_files的整数28误当来源列表作比较，AssertionError，exit1。改为核对计数28，并单独比较真实来源列表后第二次exit0；首次源码和错误stdout保留。

上述错误发生在本网页的保存结果读取辅助代码，没有修改任何实验来源／结果、读取新标签或改变主要复算输出。另有一次早期定位探查使用run/collected.json未命中，随后定位到真实evaluation/collected.json；不是项目缺件。[E12]

原执行者保存说明记载一轮汇总stdin因Windows中文编码产生Non-UTF-8 SyntaxError，后来UTF-8脚本成功；本包的summary和report保留了该事实。本网页没有把这个说明伪称为亲历该失败，也没有把它当作正式训练重试。更早CPU手写夹具失败已在关闭的实现外审中记录，本轮仅核当前资格来源和累计账，不递归重跑旧测试。[E7、E9]

本次有意跳过：所有模型与Memory正文、正式原始文本和标签、owners／Audit／private_custody／凭据；项目Linux连接、Torch／GPU原生测试、训练或重新校准；未变化旧实现的全量回归和未展开的深层历史包。没有请求这些受限资产。数组复算以原保存群指标为起点，因而不能声称重新由正式标签验证了每一个MAP／AP来源；本次没有出现要求违反边界重开标签的具体反例。[E12]

## 十二、逐文件实际阅读范围与交付

READING_SCOPE.json及READING_SCOPE.zh.txt分别列出三个容器共1,976个成员的实际人工范围、代码行、机器处理方式和身份检查强度。目录列出一个文件不意味着人工全文审查，字节哈希也不等于语义复核：

@@SCOPE_TABLE@@

P0是本轮包，P1是其中历史增量包，P2是其中原C/S结果包；没有继续展开P2中的旧实现输入ZIP。P0的1,119个数值数组在列明的保存数组检查／复算中被实际读取；大量剩余成员是收据、日志、源文件或已关闭背景。所有大展示表已机器逐字段核对，人工阅读聚焦本轮结果、关键代码、分级结论和明确异常。原大文件和未读历史没有为本次交付重复复制。[E12]

完整交付含：本正文、模型／输入身份、实际阅读台账、原23项真值表、全部支持诊断清单、必要计算源码、独立复算原始输出、失败源码／输出、复现说明、最终文件清单及SHA。ZIP自身SHA另列于外部DELIVERY_MANIFEST.json；包内文件SHA列于BUNDLE_MANIFEST.json和SHA256SUMS.txt，避免要求清单自我包含自己的哈希。计算重放仍须使用上述精确上传包，不需要项目服务器或权重／标签。

## 证据索引

E1 本轮合同与范围：P0/result/source/docs/SELLER_ALIAS_REPLAY_IMPROVEMENT.zh.md，全文1—35行；P0/result/source/schema/step28_replay_improvement_policy.json，全文1—21行；P0/background/budget_authorization.json、budget_disposition.zh.txt。预算补充取代旧48小时文字，其他范围不变。

E2 完成与资源：P0/result/job/completion.json、before_valid.json、access.json、resource.json、execution.json；P0/result/job.wrapper.txt、job.console.txt、sync_manifest.json；本次execution_audit.json、provenance_audit.json、display_audit.json。

E3 执行实际连接：P0/result/source/scripts/step28_replay_improvement_run.py，全文（update95—107、train_stage110—142、checkpoint145—207、train210—278、诊断门281—309、blind_gate312—406、collect409—427、finalize430—461、execute527—594）；P0/result/job/run/manifest.json、partition.json、points/*、updates/*、diagnostics/*。历史序列与Memory仅用允许的摘要和身份，不含正文。

E4 指标和统计：P0/result/job/evaluation/collected.json、evaluation.json、108个60×22矩阵和108组counts；P0/result/source/scripts/step28_bge_continual_evaluate.py 1—254；step28_record_replay_run.py 403—426；step28_continual_population_evaluate.py 1—101；step28_alias_calibration.py 61—94；继承bge policy的evaluation节。原5项未追改。

E5 共同诊断：P0/result/source/scripts/step28_replay_diagnostics.py 1—196；P0/result/job/evaluation/diagnostics/overfitting.json及216个valid群指标数组；run/diagnostics中的全部fit/cache/盲分数数组和元数据。

E6 诊断与阶段末连接：36个epoch6盲分数与stage-end raw分数、36个epoch6 valid指标与raw群矩阵相应三列；本次execution_audit.json及diagnostics_audit.json的检查记录。

E7 主执行者分析：P0/result/report.zh.txt全文；analysis/summarize_saved.py、summarize_teacher.py全文；output/summary.json、all_metrics.txt、all_curves.txt、curves.json、teacher_errors.json；analysis/console.txt、time.txt、output/execution.json。图像文件只核身份和生成代码来源，不冒称独立视觉审查。

E8 历史结果：P0/background/historical_result_input.zip → background/original_result_input.zip；其中result/job/evaluation/collected与C/S矩阵、reference/collected、reference/reference/collected及LOGIT矩阵；P0/result/analysis/output/historical/manifest.json、ABC/BCA/CAB_shared.json及.npy。选定历史成员及SHA见historical_inputs.json。

E9 关闭结论的适用范围：P0/background/implementation_review.zh.txt全文、implementation_primary.zh.txt全文、72小时补充授权／处置／qualification、当前CPU与GPU结果；只复用未变计算和既有覆盖，不以旧NO_OPEN_BLOCKERS豁免本轮差异。P2/result/source/docs/SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md和policy仅作直接继承边界。

E10 LOGIT直接来源：历史step28_bge_continual_run.py的520—523行及旧manifest绑定；当前同SHA副本的103—153、376—433、482—553；新run的95—107、527—594；step28_record_replay_verify.py的preflight23—41；record_replay.py的load_model39—40；chinese_base.py 110—131；bge_continual.py的首域update231—301；当前shell全文。AST搜索同时记录其他同名字句位于未调用的旧run函数。

E11 本轮分级与文字处置：本正文第一、九、十节；它们是本实例依据E1—E10作出的判断，不是主执行者原“通过”字段的机械转写。

E12 本次网页审查计算：audit_saved.py、audit_displays.py、audit_provenance.py、prepare_input.py、build_reading_scope.py；各audit JSON、recomputed_metrics.json、recomputed_diagnostics.json、historical_matrix_drift.json、historical_update_drift.json、stdout与run JSON、失败快照、REVIEW_IDENTITY.json及READING_SCOPE。文件身份以交付清单为准。

最终处置：支持本轮按负结果完成收尾，C_plus继续条件为false；明确保留R1历史复现缺陷及未隔离原因，对LOGIT主张限于本轮重训系统；不追改历史、不自动扩大范围或再训练。
"""
replacements={
    '@@PASS_TABLE@@':passed_table,'@@MAP_TABLE@@':map_table,'@@MAP_DELTA_TABLE@@':map_delta_table,
    '@@ORDER_TABLE@@':order_table,'@@PROB_TABLE@@':prob_table,'@@CLASS_TABLE@@':class_table,
    '@@FIRST_TABLE@@':first_table,'@@DIAG_COUNTS@@':diag_count_table,'@@DIAG_EXAMPLES@@':diag_example_table,
    '@@DIAG_INTERVALS@@':diag_interval_table,'@@TEACHER_TABLE@@':teacher_table,'@@HISTORY_TABLE@@':history_table,
    '@@SCOPE_TABLE@@':scope_table,
    '@@FINAL_S@@':est(delta('C_plus-S','final_all','map'),9),
    '@@FINAL_C@@':est(delta('C_plus-C','final_all','map'),9),
    '@@SS_AP@@':est(delta('C_plus-S_strong','N','average_precision'),9),
    '@@SS_BRIER@@':est(delta('C_plus-S_strong','N','brier'),9),
    '@@SS_LL@@':est(delta('C_plus-S_strong','N','log_loss'),9),
    '@@LOGIT_G@@':est(delta('C_plus-LOGIT0.1','G','map'),9),
    '@@S_N_LL@@':est(delta('C_plus-S','N','log_loss'),9),
    '@@S_O_F1@@':est(delta('C_plus-S','O','f1'),9),
    '@@S_F_FIRST@@':est(delta('C_plus-S','F_first','map'),9),
}
for k,v in replacements.items():report=report.replace(k,v)
assert '@@' not in report
(OUT/'REVIEW_REPORT.zh.txt').write_text(report,encoding='utf-8')

checks=list(e['comparisons']['C_plus-S']['checks'])
checks_text='原23项逐项真值表\n\n来自本轮原evaluation.json，并由独立脚本复算全部161个布尔值一致。PASS/FAIL沿用冻结条件，非显著性计数。\n\n'
checks_text+=table(['原判据字段',*comparisons],[(k,*['PASS' if e['comparisons'][c]['checks'][k] else 'FAIL' for c in comparisons]) for k in checks])
(OUT/'CHECKS23.zh.txt').write_text(checks_text+'\n')
support=[]
for p,roles in d['points'].items():
    for role,domains in roles.items():
        for dom,v in domains.items():
            if not isinstance(v,dict) or 'interpretation' not in v:continue
            for m,label in v['interpretation'].items():
                if label=='有相应迹象':
                    k=v['columns'].index(m)
                    row=[p,role,dom,m,v['fit_or_cache_groups'],v['valid_groups']]
                    for field in ('epoch1_to_6_train_benefit','epoch1_to_6_valid_benefit','epoch1_to_6_gap_widening'):
                        row.append(est({'mean':v[field]['mean'][k],'conditional_95pct_interval':v[field]['conditional_95pct_interval'][k]},9))
                    support.append(row)
(OUT/'SUPPORTED_DIAGNOSTICS.zh.txt').write_text('全部15个有相应迹象的物理情境—指标\n\n值为原保存字段；valid列为“效益变化”，负值为恶化。重叠情境不是独立的过拟合次数检验。\n\n'+table(['物理点','角色','实际域','指标','训练侧群数','valid群数','训练效益变化','valid效益变化','gap扩大'],support)+'\n')
assert len(support)==15

failures="""失败与跳过事实（本轮网页结果外审）

一、当前项目正式作业
随包completion及wrapper显示正常完成，exit_code=0；当前job未提供failure.json。网页核对没有发现正式失败或自动重训证据。不能将本网页辅助脚本错误算作项目训练失败。

二、本网页科学主核对
audit_saved.py execution / metrics / diagnostics / historical：四个阶段均首轮exit0；保留每次stdout和带环境／耗时的run JSON。
所有原端点、比较、区间、判据和诊断判读已核对。teacher均值初次求和差至多1.11e-16，明确逐项加法核对612个标量精确一致；两份原输出均保留。

三、本网页辅助脚本错误
1. display attempt01：KeyError: 'pooled_all'。审查者读取代码错用了字段名，真实字段为pooled。修改后attempt02 exit0。
   保留audit_displays_attempt01_failed.py、display_console_attempt01.txt、display_audit_attempt01_failed.json，以及修正后源码、console和display_audit.json。
2. provenance attempt01：AssertionError: sync_source_binding。审查者把source_files整数计数28当作完整列表比较。改为核计数，并独立对真实source列表，attempt02 exit0。
   保留audit_provenance_attempt01_failed.py、provenance_console_attempt01.txt，以及修正后源码、console和provenance_audit.json。
   首轮脚本未设置内部异常计时收据，故不补造其内部耗时／峰值RSS。
3. 早期只读路径探查run/collected.json未命中，实际为evaluation/collected.json；之后正确读取。该次不是程序异常或项目缺件。
4. 一次批量展示工具输出截断了旧实现审查正文的中段；已单独补读98—149行，完整人工阅读范围由合并记录给出。不把截断内容声称为未补读前已经读过。

四、原执行者保存的历史失败说明
原report/summary记载汇总stdin曾因Windows中文编码触发Non-UTF-8 SyntaxError，随后UTF-8文件执行成功。本网页阅读的是保存说明，不伪称亲历原失败。关闭的CPU资格旧夹具失败按既有记录保留，本轮未递归重跑。

五、有意跳过或未能独立证明的事项
按本轮范围不读权重／Memory正文、正式文本／标签、owners、Audit、private_custody或凭据；不连项目服务器，不运行Torch/GPU、训练或校准拟合。因而模型真实重载和原始标签到指标的整条链没有在本网页再执行；审查使用对应冻结调用和保存收据。
未隔离LOGIT历史数值差异的因果；未观察当前页面模型选择器；未独立验证服务端精确模型路由。未将这些缺口补写为通过。
未进行新指标／新阈值／新seed／新bootstrap选择；仅复算原固定规则。未展开更深层旧实现审查输入ZIP，未对提供图像做视觉审查（只核字节及生成代码来源）。
"""
(OUT/'FAILURES_AND_SKIPS.zh.txt').write_text(failures)
readme="""复现与交付说明

本ZIP保存本轮审查正文、阅读台账、必要计算源码和原始输出，未复制47,913,772字节输入包。
唯一输入：review_input(9).zip（用户标称原名review_input.zip），SHA256：
5b410fa7b80fb9fd89a02555a898014f3877e53620dfc170a2ffcc02870da2d1

一、环境与位置
已执行环境为网页审查工作区Python 3.12.14、NumPy 2.3.5。脚本只依赖标准库和NumPy，不导入项目模块、Torch，不需要网络、项目服务器、权重、Memory正文或原始标签。
解压审查ZIP后保持review_output目录结构。在其父目录执行下列命令：

python3 review_output/prepare_input.py /绝对路径/review_input(9).zip
python3 review_output/audit_saved.py execution
python3 review_output/audit_saved.py metrics
python3 review_output/audit_saved.py diagnostics
python3 review_output/audit_saved.py historical
python3 review_output/audit_displays.py
python3 review_output/audit_provenance.py

命令行输入路径若含括号／空格应由本地shell适当引用。prepare_input验证唯一输入身份，再将附件解到同级input；已存在的同名文件只验证相同字节，不改写。每个audit阶段原始stdout已提供，run JSON含实际输入文件、SHA及执行环境。重跑会产生自己的新运行时间和run记录，不应替换这里交付的原始运行证据。

二、核对与推断边界
科学数值核对从原保存矩阵／计数开始。只复算原PCG64(20260930)的5000次统计、固定角色／端点／23项、epoch1→6及原诊断种子。2e-12仅为复算浮点相等检查的机器容差，不是新增性能容忍或科学判据；原主统计及诊断大部分实际逐值完全相同，明列的teacher求和舍入差另行精确次序核对。
historical_matrix_drift中的旧／新端点为已定端点的描述性差值，没有新增置信区间或新的评价选择。
不运行带attempt01_failed的脚本来代替最终脚本：它们只作为真实失败证据保存。

三、阅读台账
READING_SCOPE分人工全文、节选、结构搜索、机器数组／JSON处理和仅身份核验。列出文件不代表全文读过。P0全部1,629成员身份核对；P1/P2只有明确列出的历史成员真正读取，更深层历史ZIP未展开。
read_evidence.py及reading_log.jsonl用于保留实际工具阅读范围；build_reading_scope.py据该记录及run输入记录汇总。build_report.py只是用已审查原字段排版本文和附表，没有产生新科学估计。

四、哈希与文件清单
BUNDLE_MANIFEST.json列包内其他有效载荷（不含其自身和SHA256SUMS.txt）；SHA256SUMS.txt列这些载荷及BUNDLE_MANIFEST.json，避免自我哈希。ZIP外的DELIVERY_MANIFEST.json记录ZIP本身及直接交付文件身份；最终聊天也给出ZIP SHA。
本交付不会覆盖原项目来源、原统计、历史结果或用户附件。
"""
(OUT/'REPRODUCE.zh.txt').write_text(readme)
print(json.dumps({'report_bytes':(OUT/'REVIEW_REPORT.zh.txt').stat().st_size,'report_characters':len(report),
                  'support_diagnostic_rows':len(support),'scope_members':scope['member_count'],
                  'machine_scope_counts_P0':dict(machine)},ensure_ascii=False,indent=2))
