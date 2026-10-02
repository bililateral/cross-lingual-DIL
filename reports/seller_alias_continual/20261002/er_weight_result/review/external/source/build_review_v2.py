"""Assemble the Chinese review from independently checked result files.
Writes presentation/evidence summaries only; never edits received project files.
"""
from __future__ import annotations
import ast,csv,datetime,hashlib,json,pathlib,shlex
E=pathlib.Path('/mnt/data/er_review_evidence');R=pathlib.Path('/mnt/data/er_review_input');D=E/'results/independent_v1'
def read(p):return json.loads(p.read_text())
ends=read(D/'independent_endpoints.json');comps=read(D/'independent_comparisons.json');summary=read(D/'summary.json');diag=read(D/'diagnostics.json')
with (D/'all_92_checks.csv').open() as f:decisions=list(csv.DictReader(f))
methods={'seq':'SEQ','er':'ER λ=1','half':'ER λ=0.5','quarter':'ER λ=0.25'}
comparison_order=['half_minus_er','quarter_minus_er','half_minus_seq','quarter_minus_seq'];comparison_names={'half_minus_er':'half−ER1','quarter_minus_er':'quarter−ER1','half_minus_seq':'half−SEQ','quarter_minus_seq':'quarter−SEQ'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def f(v,d=9,sign=False):return format(v,('+' if sign else '')+'.'+str(d)+'f')
def v(arm,ep,m):return ends[arm]['primary'][ep][m]['mean']
def delta(name,ep,m):return comps[name]['primary'][ep][m]
def interval(obj):return '['+', '.join(f(x,9,True) for x in obj['conditional_95pct_interval'])+']'
# All endpoint and difference values remain available without rounded Markdown.
endpoint_rows=[]
for arm,roles in ends.items():
 for role,eps in roles.items():
  for ep,metrics in eps.items():
   for m,x in metrics.items():endpoint_rows.append(dict(method=arm,role=role,endpoint=ep,metric=m,mean=x['mean'],ci_low=x['conditional_95pct_interval'][0],ci_high=x['conditional_95pct_interval'][1],**x['per_order']))
comparison_rows=[]
for name,obj in comps.items():
 for role in ('primary','against_raw_reference'):
  for ep,metrics in obj[role].items():
   for m,x in metrics.items():comparison_rows.append(dict(comparison=name,reference_role=role,endpoint=ep,metric=m,mean=x['mean'],ci_low=x['conditional_95pct_interval'][0],ci_high=x['conditional_95pct_interval'][1],**x['per_order']))
for name,rows in [('all_endpoint_summaries.csv',endpoint_rows),('all_comparison_summaries.csv',comparison_rows)]:
 with (E/'results'/name).open('w',newline='') as z:w=csv.DictWriter(z,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Human-readable line references to original source (derived copies, not replacements).
source_index=[]
for p in sorted(R.glob('scripts/*.py')):
 txt=p.read_text();nodes=ast.parse(txt);dest=E/'inspection/numbered_sources'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_text(''.join(f'{i:5d}  {line}\n' for i,line in enumerate(txt.splitlines(),1)))
 source_index.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),functions=[dict(name=n.name,start=n.lineno,end=n.end_lineno) for n in nodes.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]))
(E/'inspection/source_line_index.json').write_text(json.dumps(source_index,indent=2)+'\n')
sections=[]
sections.append('''# ER历史监督权重：2026-10-02实际结果外审

**审查对象：**收到的 `er_weight_result_review.zip`。本文件是新的结果外审，不是2026-10-01合同／实现审查的续签，也不把旧审查PASS作为此次结果正确性的证明。

**模型身份：**当前系统向我提供的身份是 **GPT-6 Astra Pro**。用户说明页面选择器显示Pro；我没有读取该页面的选择器或后台路由，不能独立认证“GPT 6 Pro”后端、内部变体或实际路由。以下结论以可重放的文件检查、源码与本次运行证据为依据，不以模型名称作为认证。

**总评：在本包允许核验的证据范围内，本轮结果有效性核验通过。** 未检出需要否决或重跑本轮结果的SCIENTIFIC_BLOCKER，未检出尚未修复、会阻止保存结果复算的项目REPRODUCIBILITY_DEFECT。通过不等于所有训练行为已经在网页环境重新执行，更不等于已证明优于SEQ或具备原创方法资格。

独立复算支持：half和quarter相对原ER1均为23/23；相对SEQ均为4/23；quarter依冻结规则入选。这个判断是“固定六群历史供给下，降低整个历史监督系数改善原ER基线”，不是“持续学习已经整体胜过普通续训”。本报告不授权任何额外训练、标签访问、服务器操作、阈值调整、权重删除或新配置。

## 1. 实际输入、证据来源与边界

原ZIP大小 **7,194,455字节**；SHA-256：

```text
5de5711fbe6d20107e8607e8c4aec85a2ac47293e845af9c8c6ece1358631db0
```

365个唯一成员，解压总字节13,647,064，CRC检查通过。`source_inventory.json`之外364个载荷逐文件大小和SHA均吻合。新作业回传清单的288文件、11,609,562字节均在实际路径找到并匹配；28份冻结科研来源的实际字节与各层清单一致。没有为了审计通过改写收到的任何源文件、矩阵、训练记录或原报告。原ZIP的逐字节副本保存在证据包 `original/er_weight_result_review.zip`，最终封包时再次核对。

本报告使用以下路径别名，均相对于原ZIP根目录：

```text
J = reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job
B = reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job
S = reports/seller_alias_continual/20261002/er_weight_result
```

新job确实位于J，不在项目回传显示别名 `S/returned/job` 下；审计按实际成员路径读取。旧基线的5个冻结记录、3个shared点、6个ER训练记录，以及J/evaluation/reference中的45套旧指标／计数足以完成这轮保存结果复算；没有要求补发完整旧job、历史模型或缓存正文。

最先阅读的两个文档为 `docs/SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md` 和冻结 `docs/SELLER_ALIAS_ER_WEIGHT.zh.md`。合同头部“未启动”属于2026-10-01准备快照，不拿它否定有新完成记录的2026-10-02结果。

### 1.1 三种证据强度不能混称

**本网页实际文件／实际计算：**ZIP及清单、来源文件、保存数组和JSON／CSV、手工参考、独立bootstrap与选择、项目审计的本机复跑，均在当前容器实际检查。文件SHA只能证明本次收到的文件与包内清单一致，不能单凭清单证明原记录所描述的每个历史行为。

**项目Linux运行记录支持：**训练起止、GPU与软件环境、解析访问账、完整盲门、原生checkpoint恢复、实际梯度／参数观察、现场权重哈希与进程观测。其多份记录、执行路径与本次读取的产物一致，但不是本网页重新连接服务器的独立现场观测。

**未重做且本轮明确不要求重做：**原生BGE加载／前向／训练，Adam张量和GPU RNG原件读取，正式标签到AP/MAP的计算，缓存正文重读，现场模型实物哈希，服务器进程状态核验。没有读取正式文本、标签文件、owners、私有资产、凭据或封存真值；没有连接项目服务器。手工参考中的标签和三参数玩具模型均为确定的审计夹具，不是正式数据或新科学实验。

对应证据：`results/intake.json`、`results/zip_members.txt`、`results/independent_v1/verified_input_files.json`、原包 `source_inventory.json`、S/return_inventory.json、S/observation.json；原始源码带行号的派生副本在 `inspection/numbered_sources/`。
''')
sections.append('''## 2. 从研究问题到实际执行：逐项核对

用户要回答的是：相同有限历史供给下，λ=0.5／0.25相比ER1是否更合适，同时如实对照SEQ的旧分布留出卖家候选排序与基础识别。不同阶段到来的是不同卖家群体；不是同一控制者跨阶段追踪。输入仅屏蔽标题／描述；ID在本次审计中只用于核对身份、日程和群对应，不作为模型特征。

下表“未检出”表示相应风险核查没有发现实际违约，不表示该风险类别不存在，也不把每项通过包装成一个发现。
''')
requirements=[
('输入完整性','原ZIP／source_inventory；J与B实际路径；独立审计inventory/returned/sources','365成员；364载荷、288回传、28来源匹配','无偏差；保留原ZIP与SHA','REPRODUCIBILITY_DEFECT：未检出；外部来源真实性不由SHA独立证明'),
('完成量与退出','J/started.txt、finished.txt、completion.json、resource_usage.log、train.log','3456更新、6912群呈现、12端点、48盲分数；exit 0','无偏差；不重启或追加训练','SCIENTIFIC_BLOCKER：未检出；实际训练行为由运行记录支持'),
('完整盲门先于valid','er_weight_run.py:261–358、437–445；J/before_valid.json与access.json','before_valid为train1/valid0；完成账为train1/valid1/heldout0/owners0','无偏差；不在网页重走需私有文件的blind_gate','SCIENTIFIC_BLOCKER：未检出；不声称独立操作系统访问审计'),
('首状态复用','J/run/manifest restored_starts；B/shared记录；bge_continual_run.py:230–241','六条路径来自三个shared完整状态，Adam从288继续；首映射与缓存来源一致','无偏差；旧shared字节记录保留','SCIENTIFIC_BLOCKER：未检出；不读取原生完整checkpoint张量'),
('日程与缓存配对','12新与6旧updates JSON；data.py:202–258；supplemental_v2','独立ID算法重建18记录，共10,368个current/history ID位置吻合','无偏差；不修改六群供给','SCIENTIFIC_BLOCKER：未检出；缓存正文不在网页核验范围'),
('dropout流','er_weight_run.py:43–59；新旧current_dropout_stream；派生种子CSV','冻结种子及实际记录的current流一致；历史流公式相同','无偏差；明确种子重建不是GPU mask重演','SCIENTIFIC_BLOCKER：未检出；逐元素GPU mask未观察'),
('整个历史目标λ','er_weight.py:101–155；288×14更新数组；手工解析梯度','BCE+rank+0.5 hard整体乘λ；当前项不缩放；无两项和归一化','无偏差；不把加权日志变小当收益','SCIENTIFIC_BLOCKER：未检出'),
('单次裁剪／连续Adam','同上；core.make_optimizer与restore_state；手工Adam式','两次backward后一次clip和step；Adam末步576/864；玩具连续状态对照成立','无偏差；第288局部步encoder LR=0无需修复','SCIENTIFIC_BLOCKER：未检出；原生优化器张量不在包内'),
('实际梯度与参数观察','J/run/updates；96个模块-观察步记录；每步gradient_norm','四观察步1/29/30/288均有记录；3456步合并范数>1','无偏差；不据裁剪推出梯度冲突','SCIENTIFIC_BLOCKER：未检出；没有current/history夹角证据'),
('当前域校准','J/run/maps、points、scores；run.py:339–355','12当前域校准群／4536对／240正对；a>0；来源、模型、首映射身份一致','无偏差；不重新拟合或挑角色','SCIENTIFIC_BLOCKER：未检出；拟合标签与原优化过程未重做'),
('新旧81套与列顺序','collected.json及reference；独立矩阵／计数审计','36+45套均60×22；60群20/域；列与身份相同','无偏差；存储自洽不是标签真值重验','REPRODUCIBILITY_DEFECT：未检出'),
('计数／概率／排序区别','population_evaluate.py:21–100；手工AP／检索参考','AP≠梯形PR-AUC；query MAP≠pair AP；阈值0.5 recall≠R@K','无偏差；固定阈值和raw/stage-cal角色不改','SCIENTIFIC_BLOCKER：未检出；无标签AP/MAP来源不可从数组再独立识别'),
('七端点／配对区间','独立直接公式及PCG64；independent_v1','O/N/Z/F_first/F/G/final_all、5000同域同群配对抽样全部复现','无偏差；保留条件区间限制','SCIENTIFIC_BLOCKER：未检出'),
('原23项及候选顺序','合同:35–43；evaluate.py:16–26；all_92_checks.csv','四比较23/23、23/23、4/23、4/23；quarter按O MAP入选','无偏差；不把赢SEQ设为停止标准','SCIENTIFIC_BLOCKER：未检出'),
('报告数值与解读','结果报告全文；supplemental_v2/report_numeric_checks.csv','335数值含264完整核心指标按显示精度相符；关键跨0／负区间解释正确','不必修写原结果；使用限定结论','REPRODUCIBILITY_DEFECT：未检出；叙述性现场行为仍仅有项目记录'),
('Linux保存结果审计','两审计源、S/audit_execution及stdout/stderr；本机原样复跑','672,510与7.11e−15复现；主独立参考并非复用其端点函数','无偏差；解释最大差统计口径','REPRODUCIBILITY_DEFECT：未检出；不是数十万独立实验'),
('两候选／单种子／LOGIT','冻结合同:19、43；结果报告第6节','已开发valid、同60群三顺序、单s0；LOGIT监督仍λ1','保留界限；不新增匹配λ LOGIT训练','SCIENTIFIC_BLOCKER：未检出；不能推断创新或真实市场效果'),
]
sections.append(table(['需求','源码／实际证据','核验结果','偏差与最小处置','分类与未验证范围'],requirements))
sections.append('''
## 3. 干预语义、连续状态与有限历史供给

### 3.1 优化的确是完整历史组合目标

实际计算为：

```text
L_base(g) = mean_378_pairs BCE(g) + mean_28_queries rank(g)
            + 0.5 × mean_28_queries hard(g)
L_step = L_base(current) + λ × L_base(history)
g = ∇L_base(current) + λ × ∇L_base(history)
g_clipped = g × min(1, 1 / (||g||₂ + 1e−6))
AdamW.step(g_clipped)  # 每步一次，沿用既有矩估计和步数
```

`rank`为每查询对27个候选的log-sum-exp减正例logit均值；`hard`沿用每查询已知负例中top5与其正例的softplus排序惩罚。源码没有把λ只乘BCE、只乘hard或只乘标量日志，也没有除以1+λ。当前／历史分别前向与反向，历史梯度实际带λ，合并后只调用一次梯度裁剪与优化器更新。（`step28_chinese_base.py:165–191`；`step28_alias_ranking.py:89–129`；`step28_er_weight.py:101–155`。）

我没有仅凭日志分解或执行者测试采信：手工参考独立写出378对解析导数，逐项比较BCE、rank、hard，检查378维梯度及4坐标中心差分；再用两层合计3参数玩具encoder/head，给定非零Adam一阶／二阶矩与步数288，独立计算第289步偏差校正、梯度裁剪及AdamW权重衰减。只检查已冻结λ=1、0.5、0.25，λ=1与旧ER函数的参数和梯度逐位相同。引用的实际函数体通过AST单独提取，没有导入训练入口、加载BGE或接触正式样本。

手算与实际函数体的参数更新最大差为0；378维解析梯度最大差2.78e−17；有限差分最大差1.68e−11。玩具标量日志存在约1.0–1.6e−7的float32输出舍入，梯度仍与独立公式一致。这里的小模型更新是审计参考，不计入正式3456更新，也没有保存玩具原生权重。证据：`source/hand_objective_reference_v3.py`、`results/hand_reference_v3/hand_reference.json`、对应完整运行日志。

### 3.2 连续Adam与局部学习率重启不是矛盾

每条新路径从原shared的Adam步数288开始，第二阶段完成为576，第三阶段完成为864。完整restore包括model、optimizer和RNG；加载后的optimizer状态有摘要核对，而不是仅恢复模型后新建零矩Adam。阶段内encoder峰值学习率1e−5，前29步warmup，后续衰减，局部第288步为0；head恒为1e−3。恢复逻辑与记录相符，但网页没有从原生checkpoint读取实际矩张量，不能将源码＋运行记录说成网页张量认证。（`step28_continual_population.py:182–213`；`step28_bge_continual_run.py:58–73、230–241`；`step28_bge_continual.py:50–53、222–228`。）

新12份记录中的1、29、30、288共4个观察步，每步encoder/head各一项，共96个模块-观察步记录；全部记录合并梯度有限且非零。第288步head参数变化、encoder参数不变，与encoder学习率0完全一致，不是“编码器没训练”的缺陷。3456步保存的合并梯度范数都大于1，说明该裁剪确实普遍介入优化轨迹；**这不说明两个梯度方向相反，也不证明梯度冲突**。没有单独梯度向量或夹角，不能越界归因。

### 3.3 同六群供给得到独立ID参考支持

除了新旧日志逐项相等，我依据冻结的schedule_seed=20260918、memory_seed=20260930和拟合群ID，以独立Python Random／Algorithm R代码重建当前六轮排列、保留六群及逐步有放回历史抽样。18份新旧阶段日志合计10,368个current/history位置全部相同；生成了1728个配对步的current/history派生dropout种子行。派生种子只核对合同与源码规定的流，并未重新产生GPU dropout mask。

第三阶段实际缓存构成为：ABC留A四群、B两群；BCA留B两群、C四群；CAB留A六群，最早C域已无缓存群。half、quarter与原ER均一致。原缓存源SHA必须一致；新分支序列化总字节不必相同，因为arm、阶段等附属元数据也计入预算。这不意味着暗中更换供给。全部记录的历史及附属序列化体量均低于1MiB；网页未读取其正文，因此只将大小、成员、来源及复原记录作证据。

由此可回答整个ER系统对历史监督系数的敏感性，但不能把结果唯一归因为缓存过拟合、历史域退出、梯度冲突或校准。λ还会改变合并梯度方向、裁剪比例和Adam历史；“只改一个超参数”不意味着只有一个优化机制发生变化。当前群各出现6次，历史缓存群平均48次，是固定曝光结构，不等同于每群恰好48次。（证据：`results/supplemental_v2/id_schedule_checks.json`、派生种子CSV；`results/independent_v1/diagnostics.json`。）
''')
sections.append('''## 4. 真实完成、盲门与访问账

项目记录的运行起止为2026-10-01 15:08:21至2026-10-02 01:47:50，Asia/Shanghai。时间戳差38,369秒，与GNU 10:39:29一致；程序内部38,367.119733秒是不同计时边界，不应强求完全相等。GNU exit 0、最大RSS7,092,740KiB；程序最大观察产物18,289,684,475字节，不是连续采样的精确高水位。新续训量2×3×2×288=3456，每更新current＋history两群即6912呈现。

`train.log`可解析出144个进度事件：12阶段各12次，每24更新记录一次，时间单调，逻辑步数与对应阶段相符，末状态为完成。本包正式作业stdout和stderr合并至该文件，不存在独立 `J/stderr.log`；其中方法重命名FutureWarning不构成运行失败。项目保存结果审计则另有独立stdout/stderr文件，后者为空。

盲门在12个端点、48分数数组、模型／缓存／起点身份和12训练记录全部检查后返回，随后写before_valid（valid仍0），再进入一次valid解析，之后36套新矩阵与计数先全部落盘，再做比较。`collected.json`包含群ID与实际域顺序；这些并非另一个独立group_ids.json文件。上述顺序由入口源码和哈希绑定记录共同支持。（`step28_er_weight_run.py:261–358、361–385、437–453`。）

网页实际核对了可见哈希、尺寸、成员和计数，不重新执行原blind_gate，因为该函数还要求读取有意未提供的原生模型、缓存实物。由此不能把“记录中的盲门完成”写成“网页再次通过全部原生盲门”。访问账属于入口的解析次数记录，不是操作系统对所有潜在文件访问的全局取证。

S/observation.json的保存时刻是2026-10-02 10:08:32.430915（Asia/Shanghai）；它包含退出状态、空GPU结果和返回码1、仅表头的进程检查结果。原报告10:06:50首次查看、四个原PID均不存在的描述属于执行者现场叙述；该JSON没有提供被查询的完整PID／命令清单供网页重演。我不将它升级为独立现场核验。这一点不推翻已保存的完成产物。

12份原生权重共15,677,731,710字节，其observation文件记录与12个端点的模型路径／字节／SHA一致。网页核对的是**这些记录之间的一致性**，没有对Linux的15.68GB文件实物再次哈希。没有理由要求传输它们来完成本轮保存结果审查。
''')
sections.append('''## 5. 指标身份与手工参考

### 5.1 新36套＋旧45套，不能误算成81组独立样本

新12端点×raw/stage-cal/first-cal三角色=36套；旧15端点（3shared＋6SEQ＋6ER）×三角色=45套。每套60×22的float64矩阵及60组计数；总矩阵单元106,920。实际域A/B/C各20群，群序、列序和分区一致，旧45套逐项绑定原collected记录。四方法各三阶段、三顺序、三角色、三实际域、22指标构成7128行CSV，其中shared点被多方法引用，不是重复新训练。

列顺序为：

```text
average_precision, trapezoidal_pr_auc, roc_auc, recall_at_fpr_1pct,
brier, log_loss, precision, recall, f1, specificity,
balanced_accuracy, mcc, map, mrr,
recall_at_1, recall_at_3, recall_at_5, recall_at_10,
ndcg_at_1, ndcg_at_3, ndcg_at_5, ndcg_at_10
```

每群28账号、378个无向对；保存计数满足tp+fn=20、fp+tn=358。根据计数独立反算precision、recall、F1、specificity、balanced accuracy、MCC；总29,160个比率数值相符。新分数数组在logit≥0时的预测正数与tp+fp相符。没有标签无法独立拆解某一个预测究竟是真正还是假正；计数自洽不能冒称计数已从真值再次验证。

`average_precision`是378对上的非梯形AP，`map`是28查询AP的平均，查询在另27账号中排序、每查询1或2正例；二者不是同一终点。`trapezoidal_pr_auc`按PR折线面积单列。手工例子y=[1,0,1,0]、分数[4,3,2,1]：AP=5/6，梯形PR-AUC=19/24，ROC-AUC=3/4；分数改成并列[2,2,1,1]：AP=1/2，梯形面积=5/8，ROC-AUC=1/2。实际指标函数通过这些不同值的参考，而不是以一个常数样本同时“验证”二者。另有4账号检索手算：MAP/MRR=2/3、R@1=1/2、NDCG@3=3/4，验证query级定义及排除自身。（`step28_continual_population_evaluate.py:21–100`；网页hand_reference。）

### 5.2 校准身份与固定工作点

raw分数为排序来源，stage-cal是当前域校准后的主概率与固定0.5工作点，first-cal只诊断。12个映射均为正斜率，来源分数哈希、当前实际域、12群／4536对／240正对、模型摘要与点记录吻合。对12端点×两种映射×60×378=544,320个变换值重新计算 `a×raw+b`，与保存数组逐位相同；稳定排序及并列关系亦相同。校准后的14个曲线／检索列与raw共45,360个值逐位相同。正斜率保持排序不等于保持分类工作点。

固定0.5对应校准logit≥0，不能与检索Recall@5混淆；`recall_at_fpr_1pct`是原定义的1% FPR曲线指标，既不是本轮新的分类阈值，也不等于另行历史报告的0.1%自动判定门。网页没有重拟合映射、重读校准标签或从新valid曲线选阈值。

**没有从真值重算AP/MAP。** 所有正式AP、MAP、曲线与检索基础数值仍来自被授权的那一次正式收集。本次独立复算的是它们的身份／排列、代数约束、端点聚合、区间和决策；手工夹具只检验函数语义，不代替正式真值链路。
''')
sections.append('''## 6. 七端点、实际域分层配对5000抽样与独立实现

设一个到达顺序为(a,b,c)，`R[t,d]`为阶段t后在实际域d的20群指标均值。先在每个顺序内定义，再对ABC/BCA/CAB三个顺序等权平均：

```text
O       = (R[3,a] + R[3,b]) / 2
N       = (R[2,b] + R[3,c]) / 2
Z       = R[3,c]
F_first = R[1,a] - R[3,a]
F       = ((R[1,a]-R[3,a]) + (R[2,b]-R[3,b])) / 2
G       = ((R[2,b]-R[1,b]) + (R[3,c]-R[2,c])) / 2
final_all = (R[3,A] + R[3,B] + R[3,C]) / 3
```

对Brier/log_loss，F_first、F、G三个“差值型”端点的符号反转，使遗忘越大越坏、学习越大越好。O/N/Z/final_all的损失保持原方向，越低越好。F不是任意阶段最大值减最终值，N不是最终两个新域平均；不能换一种看起来更直观的公式再宣称原实现错误。

本网页主独立源码没有导入生产端点函数、生产bootstrap函数、生产审计TERMS或生产比较判据。实现路径是：每个实际域矩阵先取样并求20群均值，再按以上直接公式构造三个顺序端点，最后配对求差与顺序平均。生产／项目审计主要通过60行字段加权或bincount矩阵乘法形成统计量；两种计算顺序不同，结果仅有浮点末位差异。

抽样使用 `Generator(PCG64(20260930)).integers(0,20,size=(5000,3,20))`。每次在A、B、C各自20群内有放回抽20群；相同抽样索引用于所有方法、阶段、角色和顺序，以保留同一群被反复测量的配对相关性。实际域是A/B/C，不是“到达第一／第二／第三域”。生成的300,000个整数与原draws全部相同。三顺序共享60群，不能看作180个独立群或三独立训练种子。

点估计用原20群，不用bootstrap平均代替。差值区间直接来自每次配对差，不能把两条边际区间相减或对候选／参照独立抽样。5000个差值排序后，用h=(n−1)q、相邻值线性插值取2.5%和97.5%；例如0…4999的端点为124.975和4874.025。另检查同一方法减自身时全部配对结果为0。

手工域／阶段矩阵 `[[10,20,30],[11,22,33],[8,19,36]]`（行是阶段，列A/B/C）得ABC七端点 `[13.5,29,36,2,2.5,2.5,21]`；BCA为 `[27.5,20.5,8,1,−1,0,21]`；CAB为 `[22,15,19,−6,−1.5,−1,21]`，并另外检查损失符号。这覆盖了到达顺序转换，而不是只做全常数手算。

主独立复算：14,784个端点数值的最大差5.551115123125783e−16，4,752个比较数值的最大差3.7470027081099033e−16；全部四比较和92个布尔条件一致。保存了完整5000次配对差分样本、draws、端点与比较JSON，并另导出全精度CSV。实际命令和完整输出见 `logs/independent_v1/`；源码 `source/independent_result_audit.py`。

这些区间条件于已训练模型、固定数据生成与分区、固定顺序／映射，并不覆盖重新训练、数据生成、多种子或两候选筛选带来的不确定性。没有多重比较校正的独立确认性保证，均值保护也不是总体非劣效检验。
''')
sections.append('## 7. 独立复算结果与全部23项\n\n### 7.1 核心端点\n')
sections.append(table(['方法','O MAP','O R@5','N MAP','N R@5','Z MAP','Z R@5','F_first MAP↓'],[[methods[a],*[f(v(a,ep,m)) for ep,m in [('O','map'),('O','recall_at_5'),('N','map'),('N','recall_at_5'),('Z','map'),('Z','recall_at_5'),('F_first','map')]]] for a in methods]))
sections.append('\n四比较以“候选−参照”为方向；下表MAP/R@5越大越好，F_first越小越好。\n')
sections.append(table(['比较','ΔO MAP','O MAP条件95%区间','ΔO R@5','ΔN MAP','ΔZ MAP','通过项'],[[comparison_names[n],f(delta(n,'O','map')['mean'],9,True),interval(delta(n,'O','map')),*[f(delta(n,ep,m)['mean'],9,True) for ep,m in [('O','recall_at_5'),('N','map'),('Z','map')]],f"{comps[n]['passed']}/23"] for n in comparison_order]))
sections.append('''
### 7.2 原23项逐项结果及失败原因

数值为对应差值或O MAP区间下界；括号给出实际判定。排序列取raw；概率列除标注“raw参照”外，比较主角色stage-cal。下表不采用任何事后容差，条件中的严格大于与非严格大于等于均保持原样。
''')
labels=['O MAP均值 > 0','O MAP区间下界 > 0','O R@5均值 > 0','N MAP均值 ≥ 0','N R@5均值 ≥ 0','O AP均值 ≥ 0','O ROC-AUC均值 ≥ 0','O Brier均值 ≤ 0','O log_loss均值 ≤ 0','O Brier − raw参照 ≤ 0','O log_loss − raw参照 ≤ 0','N AP均值 ≥ 0','N ROC-AUC均值 ≥ 0','N Brier均值 ≤ 0','N log_loss均值 ≤ 0','N Brier − raw参照 ≤ 0','N log_loss − raw参照 ≤ 0','Z MAP均值 ≥ 0','Z R@5均值 ≥ 0','Z AP均值 ≥ 0','Z ROC-AUC均值 ≥ 0','Z Brier均值 ≤ 0','Z log_loss均值 ≤ 0']
keys=[x['check'] for x in decisions if x['comparison']=='half_minus_er'];dlookup={(x['comparison'],x['check']):x for x in decisions};rows=[]
assert len(labels)==len(keys)==23
for i,(label,key) in enumerate(zip(labels,keys),1):
 row=[i,label]
 for n in comparison_order:
  x=dlookup[n,key];row.append(f(float(x['value']),9,True)+'（'+('通过' if x['passed']=='True' else '失败')+'）')
 rows.append(row)
sections.append(table(['项','原条件',*[comparison_names[n] for n in comparison_order]],rows))
sections.append('''
half/quarter对ER1没有失败项；对SEQ的19个失败项完全相同，即除第10、11、16、17项外全部失败。四个通过项仅是O/N的Brier、log_loss相比**未校准raw参照**更小；这不是四个指标胜过同样stage-cal的SEQ。第2项区间条件失败与第1项均值条件失败是两个冻结条件，并非两次独立统计发现。

**23项不是23个显著性检验。** 只有O MAP规定区间下界>0，其他大部分为观察均值约束。half−ER1的ΔO R@5均值+0.004761905，但区间[−0.008631, +0.018155]跨0，仍按原均值>0条件通过。不能据“23/23”写成23项都已显著改善，也不能把原来不要求区间的条件临时加严。

### 7.3 quarter−SEQ的关键区间必须保留
''')
sections.append(table(['端点／指标','差值','条件95%区间','可说与不可说'],[[ep+' / '+m,f(delta('quarter_minus_seq',ep,m)['mean'],9,True),interval(delta('quarter_minus_seq',ep,m)),text] for ep,m,text in [('O','map','区间跨0；不能称显著更差，也不能称等效或已非劣效'),('O','recall_at_5','区间完全负；旧域检索R@5仍较差'),('N','map','区间完全负；新域刚学完MAP仍较差'),('Z','map','区间完全负；最终最新域MAP仍较差'),('F_first','map','区间跨0；微小负均值不证明更抗遗忘')]]))
sections.append('''
### 7.4 选择顺序与局部表现

候选先须相对ER1全部23项通过；之后按O MAP、O R@5、较大λ依次打破精确平局。两候选均合格，quarter的O MAP=0.411997391高于half的0.388076929，故已在第一排序键决出，不需要启用R@5或较大λ平局规则。若无人通过，原规则才回退ER1；对SEQ是否通过不是资格条件或继续训练触发器。源码只建立局部逻辑视图复用旧端点函数，不修改旧manifest、方法全局常量或历史产物名。（`step28_er_weight_evaluate.py:16–69`；冻结合同:39–43。）
''')
sections.append(table(['ΔO MAP','ABC','BCA','CAB'],[[comparison_names[n],*[f(delta(n,'O','map')['per_order'][o],9,True) for o in ('ABC','BCA','CAB')]] for n in comparison_order]))
sections.append('''
half的ABC顺序O MAP为负，进一步拆实际旧域，A为−0.000322629、B为−0.002727014；这正说明总体23项保护不是每顺序／每实际域保护。quarter相对ER1的三个顺序O MAP均为正，本次保存数组的六个最终旧域MAP差值也均为正；这是本次观察事实，不是“23项通过”在定义上保证的性质。quarter对SEQ仅BCA顺序的O MAP为正。

完整792个最终实际域×顺序×指标差值保存在 `results/supplemental_v2/all_final_domain_differences.csv`，不另筛有利格子作为新成功标准。
''')
sections.append('## 8. 基础识别、完整指标与遗忘／学习端点\n\n### 8.1 固定0.5工作点不能被候选检索指标替代\n')
sections.append(table(['方法','端点','固定0.5 Recall','固定0.5 F1','Specificity','检索R@5'],[[methods[a],ep,f(v(a,ep,'recall')),f(v(a,ep,'f1')),f(v(a,ep,'specificity')),f(v(a,ep,'recall_at_5'))] for a in ('seq','er','quarter') for ep in ('O','N','Z')]))
sections.append('''
quarter较ER1不再全部零正召回，但O/N/Z固定0.5 recall仅约2.25%／2.08%／0.92%，仍不能描述为自动判定能力已经充足。其specificity相对ER1的1.0略有下降，说明23项不是对22个全部指标的全面不退化保证。冻结基础保护主要覆盖AP、ROC-AUC、Brier、log_loss等指定项，不可改写为所有分类量或每域都受保护。

这些都是先按群计算再按端点定义平均的宏指标；pooled precision/recall必须先合并tp/fp/fn/tn再计算，不能把群precision平均当pooled。独立审计另重算2592个pooled计数／比率数值，与保存JSON一致。候选检索仍是实际用途，固定工作点低召回不自动否决候选排序用途；本轮也没有确定新的人工复核预算K或宣告新的自动判定成功门。

### 8.2 O/N/Z全部22指标

下表从独立端点JSON生成，不抄录执行者排版表。原报告对应264个值全部在其九位小数显示精度内吻合。
''')
metrics=list(ends['seq']['primary']['O'])
for ep in ('O','N','Z'):
 sections.append('\n#### '+ep+'\n')
 sections.append(table(['指标',*[methods[a] for a in methods]],[[m,*[f(v(a,ep,m)) for a in methods]] for m in metrics]))
sections.append('\n### 8.3 其余端点的MAP与R@5\n\nF_first/F越小越好，G越大越好；final_all为最终三实际域平均。四方法、四角色、七端点、22指标的全精度均值／每顺序值／区间共2464行已另存 `results/all_endpoint_summaries.csv`，不是只检查以下摘录。\n')
sections.append(table(['端点','指标',*[methods[a] for a in methods]],[[ep,m,*[f(v(a,ep,m)) for a in methods]] for ep in ('F_first','F','G','final_all') for m in ('map','recall_at_5')]))
sections.append('''
## 9. 项目Linux“672,510个数值”审计的正确性与覆盖

检查了原包 `scripts/step28_er_weight_audit.py` 与 `scripts/step28_bge_continual_audit.py`。它们以JSON／NPY／CSV读取保存结果，不导入训练模块；新入口复用审计辅助数值比较、混淆计数和独立TERMS，不拿模型重新推理。端点TERMS与合同公式相符，实际域bincount权重保留配对，同一draws贯穿全部比较；线性分位数和loss端点符号正确；23项及资格/平局顺序正确。其常数手工bootstrap例子本身覆盖有限，所以本网页补充了非恒定阶段／域手算、直接取样路线与逐条件复算。

项目记录的两源文件SHA：

```text
step28_er_weight_audit.py
29f488e841098ba9d458113ff2deee45575a9c9438d52c57428ca901b5fcd777
step28_bge_continual_audit.py
2fff73cdc0e26843a10dfd35388ade1b31f9ca890d889c4c31f13b98efc1b193
```

网页核对源文件字节与以上SHA，核对S/audit_stdout.log解析结果等于S/audit/audit.json，stderr为空，audit_execution退出0。其项目命令CPU47、Python3.10环境，UTC 02:17:23.038370开始、02:17:23.654617结束。网页在允许CPU0用收到的原审计源码原样复跑，独立输出位置不在原输入目录；同样得到672,510、最大差7.105427357601002e−15、四比较和选择一致。这个复跑是辅助交叉核验，不是主独立参考，更不是在项目Linux再次执行。

### 9.1 最大差的统计口径需要准确

项目的7.11e−15是其Audit.equal／array跟踪到的数值比较最大差，不是“所有核验、所有日志代数、所有模型误差均≤7.11e−15”。其损失分解另以 `np.allclose(rtol=3e−6, atol=3e−6)` 检查，不计入同一最大差累计。网页独立以更直接绝对差检查得到：6912个current/history损失分解最大差3.2782554626464844e−7，仍低于3e−6，符合float32分项日志舍入；加权历史、合并总损失与学习率等保存等式逐项相等。

因此原报告“672,510个数值最大差7.11e−15”按其跟踪口径属实；本外审明确该口径，不把它外推到未跟踪量，也不把这处较大但合容差的损失舍入误差误判为科学错误。

网页主独立源码记录1,325,341次数值比较，其中包含整数draws、重复角色、身份/尺寸以及自身差分，不是1,325,341个独立观测或显著性检验。最有解释意义的覆盖是：81套矩阵，全部60群22列；544,320仿射值；29,160计数比率；14,784端点摘要；4,752比较摘要；92条件；7128阶段行；以及12新训练记录和6旧配对记录。分项误差与次数完整保存在summary和numeric_comparisons.csv，不以一个大计数替代正确性论证。
''')
sections.append('''## 10. 对完整报告论断的审查及研究结论

`docs/SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md`的核心表、四比较、每顺序O MAP表、O/N/Z三套完整22指标表与关键文字区间共335数值已自动按显示精度逐项核对；最大显示差4.999110616576985e−7来自六位小数舍入，九位小数表按各自精度通过。运行数字、访问账、源文件数量、回传／权重记录另与原证据核对。没有发现需要更改原数值或结论方向的表述。

可保留的结论是：**在本次固定六群历史供给、单s0和已开发valid上，half、quarter均改善原ER；quarter按预定规则作为此次ER配置选择结果。** half虽未入选，仍有有效相对正结果；ER1对SEQ的负结果和新配置对SEQ的不足均继续保留，不因不合预期宣告实验无效。

不能升级的结论包括：quarter全面胜SEQ；O MAP跨0即已等效；F_first微小下降即抗遗忘显著改善；23项通过即每指标／每域／每顺序显著改善；调λ就是原创方法；0.25在连续范围内最优；六群缓存或梯度冲突已经得到因果机制证明；中文合成结果代表真实市场身份匹配效果。原报告已经说明主要边界，本审查不建议为了“修绿”改原报告门槛。

原LOGIT的历史监督仍λ=1、额外logit MSE系数0.5。它不是在quarter监督λ=0.25上加MSE的匹配消融。因此旧LOGIT与新quarter之差不能全部归因于logit项。本包原LOGIT仅作为旧背景记录，网页未对LOGIT重新生成指标、重训或声称其全面超过SEQ。

两个预定候选仍属于已开发valid筛选，不能因为候选预先写明就把筛选后的条件区间变成独立确认试验；单s0的三顺序不是三个训练种子。有效负结果不等于研究不可继续，基线开发有效也不等于未来方法创新成立。本次合同已经达到结束条件，不自动加λ、换缓存、改阈值、开新种子或使用新最终留出。
''')
sections.append('''## 11. 分类、最小处置及未验证事项

### 11.1 SCIENTIFIC_BLOCKER

**未检出当前结果的科学阻断项。** 在所给证据范围，干预、配对供给、保存评价与冻结决策相符；四比较和选择由独立参考复现。不得因“没有赢SEQ”判实验无效，也不得要求新的科学实验才能接受这轮既定比较。

保留的不可验证范围是正式标签到AP/MAP、原生权重和Adam/RNG张量、GPU mask、现场进程以及日志未覆盖的访问行为；这些是有意限定的证据边界，不虚构为已完成，也不把缺少私有载荷本身归为应阻断本包的缺陷。

### 11.2 REPRODUCIBILITY_DEFECT

**项目收到的科学来源及保存结果未检出尚未修复的复算缺陷。** 原审计CPU47在网页不可用时替换为允许CPU0，并明确环境，不冒称原环境；实际job路径按包内路径使用，不将回传别名误判为漏包。

本网页自己的审计辅助程序出现两处可复现缺陷，已只修订审计参考并保留失败源／日志／diff，未修改项目28份来源或原输出：

1. `hand_reference_v1`：全部数值断言之后，JSON写出时np.bool_不是可序列化Python bool，退出1。v2仅将标志显式转bool并另存新结果目录；退出0。v3仅新增可移植的project/output/cpu命令行参数，数值算法不变；再次退出0。
2. `supplemental_v1`：补充实际域明细时误找独立group_ids.json，路径不存在，退出1；此前日程和335报告数值检查已执行且保存，但不据此把失败程序标为完整通过。v2改从现有collected.json的group_ids字段读取，并显式断言新旧／partition群序与列序一致；退出0。未改任何科学量、容差或预期结果。

另有探索性查看误把正式stderr当独立文件、误取point.training键，两次非数值审计失败通过独立标注的“重现运行”保留完整日志；正确来源分别是合并train.log与manifest.training／run/updates。还有一次探索性打印的SyntaxError在取数前发生，原命令／错误文本单独保留。这些不归咎于项目科学实现。原探索调用未使用计时包装，未伪造其起止时刻；重现运行的时刻只代表重现。

Files接口对ZIP解析返回HTTP400，不代表空文件；转用已实际挂载ZIP读取，完成CRC、清单和内容审查。该工具回退也在失败记录中说明。

### 11.3 OUT_OF_SCOPE_OVERDESIGN

为本轮验收追加λ、重训SEQ/ER/LOGIT、要求赢SEQ才停、扩种子／数据／阈值、加载大权重或重读正式标签、索取缓存正文／密钥、增加隔离与系统加固等，均不是本轮必要修复。没有执行这些动作，也不把它们列为“必须自行执行”的补充实验。

### 11.4 最小处置

保留原所有输入和输出、原ER1/SEQ结果及half有效结果；在本次开发配置记录中采用quarter，明确其对SEQ仍不通过。将本REVIEW和证据包作为10月2日结果外审归档即可；不需要为本审查修写28份冻结来源、改门槛或重算正式标签指标。原生权重保留／删除服从既有授权，本网页没有执行文件清理。未来匹配监督强度的LOGIT比较或创新方法需要新的研究合同，但那不是此次结果有效性的前提，也未由此报告发起。
''')
sections.append('## 12. 本网页实际命令、环境和日志\n\n所有下列时间均为UTC；项目报告的Asia/Shanghai为UTC+8。当前容器Python3.13.5、NumPy2.3.5；玩具参考Torch2.10.0+cpu。允许CPU为0–4，实际数值进程绑定CPU0，OMP/MKL/OpenBLAS均1线程。项目训练则是其记录中的Python3.10.19、Torch2.9.1+cu130、CUDA13.0、RTX5090、CPU47；不混淆两者。\n')
logrows=[]
for p in sorted((E/'logs').glob('*/execution.json'),key=lambda p:read(p)['started_at_utc']):
 x=read(p);logrows.append([p.parent.name,x['started_at_utc'],x['finished_at_utc'],str(x['exit_code']),f(x['elapsed_seconds'],6)])
sections.append(table(['运行','UTC开始','UTC结束','退出码','秒'],logrows))
sections.append('''
另于UTC 02:49:15.543201至02:49:27.035418，在全新 `/mnt/data/er_review_portable_smoke` 目录实际执行了可移植复现入口，四个子进程均退出0。独立端点、比较、92条件CSV和draws与首次运行逐字节相同；手工参考与补充审计结论相同。四子进程完整日志另存 `logs/portable_child_*`，不是只测试入口或声称用户以后自行运行。随后最终质量核对再次确认原365成员未改。

每个目录均保留 `execution.json`（完整argv、cwd、环境、起止、退出码）、原样完整 `stdout.log` 和 `stderr.log`。失败输出未覆盖，修订diff位于results；项目Linux日志仍在原ZIP的S中，不与网页日志混放。

主要命令（以下已实际执行；完整绝对路径以日志为准）：

```bash
python source/independent_result_audit.py \
  --project /mnt/data/er_review_input \
  --zip /mnt/data/er_weight_result_review.zip \
  --output /mnt/data/er_review_evidence/results/independent_v1 --cpu 0

python -B scripts/step28_er_weight_audit.py \
  --project /mnt/data/er_review_input \
  --job /mnt/data/er_review_input/reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job \
  --baseline /mnt/data/er_review_input/reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job \
  --inventory /mnt/data/er_review_input/reports/seller_alias_continual/20261002/er_weight_result/return_inventory.json \
  --output /mnt/data/er_review_evidence/results/supplied_audit_cpu0 --cpu 0

python source/hand_objective_reference_v3.py \
  --project /mnt/data/er_review_input \
  --output /mnt/data/er_review_evidence/results/hand_reference_v3 --cpu 0

python source/supplemental_result_audit_v2.py \
  --project /mnt/data/er_review_input \
  --independent /mnt/data/er_review_evidence/results/independent_v1 \
  --output /mnt/data/er_review_evidence/results/supplemental_v2 --cpu 0
```

这里用相对source/是为排版展示；原审计的cwd为解压项目根，其余参考的绝对路径在execution.json中。无需连接项目服务器即可从原ZIP复核保存结果。证据包的可移植入口 `source/reproduce_saved_review.py` 仅解压收到的ZIP到新工作目录并执行同一保存结果审计与确定性手工参考；不发起网络访问、依赖安装或训练任务。提供入口是复现材料，不要求用户另开实验来补齐本次结论。

## 13. 证据包索引

`original/`保留本次原ZIP；`source/`保留全部独立源码、v1失败版本、v2/v3修订和复现入口；`logs/`保留每次正式审计／失败重现的完整进程输出；`results/`保留draws、5000配对差分样本、独立端点／比较／92条件、手工参考、日程、全部指标CSV及修订diff；`inspection/`保留带行号源码、探索失败说明和来源索引。

最终 `SHA256SUMS.txt` 与 `FILE_MANIFEST.json` 给出本证据包所有载荷的大小／SHA；它们不包含自身，防止递归自哈希。外层证据ZIP的大小／SHA单列在下载旁的交付清单。`results/final_input_integrity.json`核对封包前原ZIP、保留副本及原365成员仍与收到时一致。报告只引用包内源码／证据和本次推导，未调用外部资料替代本包证据。

**最终判定：本轮结果审查通过，quarter的预定ER配置选择成立；不成立的是“整体优于SEQ／已证明抗遗忘优势／调权构成原创方法”这些更强结论。保留真实正负结果与上述验证边界，按本轮合同结束，不新增实验。**
''')
text='\n\n'.join(sections).strip()+'\n';(E/'REVIEW.zh.md').write_text(text);pathlib.Path('/mnt/data/REVIEW.zh.md').write_text(text)
print(json.dumps(dict(report_bytes=len(text.encode()),report_lines=len(text.splitlines()),endpoint_csv_rows=len(endpoint_rows),comparison_csv_rows=len(comparison_rows),source_numbered_files=len(source_index)),ensure_ascii=False,indent=2))
