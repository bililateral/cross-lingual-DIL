"""Assemble the Chinese review from independently produced saved-output analyses."""
from __future__ import annotations
import json,pathlib,shlex
B=pathlib.Path('/mnt/data/bge_review_evidence'); O=B/'outputs'
def load(n): return json.loads((O/n).read_text())
S=load('independent_endpoints.json'); D=load('descriptive_logit_minus_seq.json'); C=load('independent_comparisons.json')
G=load('report_crosscheck_v2/report_23_conditions.json'); F=load('report_crosscheck_v2/report_first_learning.json'); W=load('report_crosscheck_v2/report_fixed_workpoints.json'); MEM=load('independent_memory_diagnostics.json')
I=load('inventory_verification.json'); checks=load('independent_result_audit.json'); tr=load('independent_training_evidence.json')
parts=[]
def text(s): parts.append(s.strip()+'\n')
def table(headers,rows):
 text('| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows))
def f(x):return f'{x:.6f}'
def signed(x):return f'{x:+.6f}'
def v(a,e,m):return S[a]['primary'][e][m]['mean']
text('''# 首轮有限历史中文BGE持续学习：独立结果外审

审查日期：2026-10-01。审查者自述模型为 **GPT-6 Astra Pro**；后端具体部署标识未向我独立暴露，不能将用户指定的“GPT 6 Pro”写成已核实的后端身份。

审查对象：`bge_continual_result_review.zip`，不是前轮实现包。本报告只依据本次附件、其中保存的历史证据以及本次在当前容器实际执行的无标签核验。没有读取项目正式标签、owners、heldout，没有加载项目大模型权重，没有训练、重拟合、选择新阈值或连接项目服务器。公开群ID仅用于对齐、分区和随机过程重建，不用于推断账号身份或标签。

## 0. 三层结论必须分开

**第一层：接受为已完成、在提交证据边界内有效的首轮诊断结果。** 原始载荷、冻结来源、正式运行记录、保存分数/矩阵/计数以及独立端点复算没有发现足以推翻本轮结果的实质矛盾。这个接受不是“已重新从正式标签计算所有指标”，也不是“已重新加载全部原生模型验证训练”。后两项本轮未执行，包内材料也不支持这样声称。

**第二层：按原冻结验收，SEQ遗忘门通过；ER−SEQ仅4/23，不通过；LOGIT−ER为23/23，通过。** 后续多种子要求没有完成，不能因为三个到达顺序都跑完而改写为多种子完成。没有新增LOGIT−SEQ正式验收，没有引入旧MAP +0.01线，没有修改端点、阈值或成功标准。

**第三层：尚不能称为已得到满足“保旧且学新、概率保护”的、整体优于普通续训SEQ的持续学习方法。** LOGIT是对ER退化的有效修复，但相对SEQ的旧域Recall@5、新域与最终最新域表现仍弱，最终全域MAP和Recall@5也下降。LOGIT对SEQ并非“所有指标都差”：旧域MAP、AP/AUC和部分概率指标略好；准确结论是存在明显取舍，而不是整体成功。已知SEQ/ER/LOGIT的这轮诊断不构成方法创新或投稿达标证明。

本轮发现 **1项低严重度REPRODUCIBILITY_DEFECT：正文引用了未入包的`observation.json`**，只影响附带进程/GPU状态主张的证据导航，不影响正式退出记录和主结果复算。下面列出的科学阻断是对更强结论的限制，并不伪装为当前实现错误；现有结果正文已基本主动披露这些限制。

## 1. 附件完整性、原始证据与来源链
''')
table(['核查对象','独立结果'],[
 ['上传ZIP字节数',I['archive_bytes']],['上传ZIP SHA-256','`'+I['archive_sha256']+'`'],['ZIP成员','496：495个载荷 + source_inventory.json'],['载荷字节数 / 总解压字节数',f"{I['payload_bytes']} / {I['uncompressed_total']}"],['成员集合、CRC、全部载荷大小/SHA','全部匹配；未发现额外或缺失成员'],['正式源码链','18份冻结来源在授权、启动、原生核验、正式manifest和本次附件中一致'],['完成快照回传','326份文件，共16,241,505字节；逐份大小/SHA一致'],['本次结束前复查','全部495载荷及原ZIP保持原字节；解压目录没有新增文件'],['清单创建时刻','2026-10-01T02:07:09.177184+00:00']])
text('''这些SHA核对证明本次所读附件内部、声明清单与保存来源之间的一致性，不是对远端机器过去每一次行为的外部认证。没有为本轮要求额外签名、系统跟踪或网络仪式。

Windows报告中的`returned/job`在包内按照原Linux相对作业目录映射。已用`source_inventory.json`验证这一映射；不能把映射后的目录位置当作缺失结果，也不能用提交后的状态更新替换冻结作业字节。本次审查者自己的第一版文档链接诊断曾把该目录链接误报，第二版只修正目录映射逻辑，保留了第一版源码、输出和日志。

### 1.1 引用位置与可复核索引

下表中的行号均指原附件UTF-8文本的1起始行号，不是本报告的行号。`reviewed_sources/`保留24份关键源码/合同/历史审查的逐字节副本；全部原件在证据包的`input/bge_continual_result_review.zip`内。完整路径与SHA见`outputs/reviewed_source_index.json`。
''')
table(['代号','原包路径 / 含义'],[
 ['C','docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md（冻结合同）'],['B','scripts/step28_chinese_base.py（BGE加载、聚合、标签/评分原语）'],['T','scripts/step28_bge_continual.py（Supply、Memory、LR、目标与更新）'],['R','scripts/step28_bge_continual_run.py（阶段、共享分支、盲门、采集、执行）'],['V','scripts/step28_bge_continual_evaluate.py（端点、bootstrap、验收、7326行表）'],['M','scripts/step28_continual_population_evaluate.py（22指标原语）'],['H','scripts/step28_alias_ranking.py（困难排序项）'],['K','scripts/step28_alias_calibration.py（正仿射校准）'],['P','scripts/step28_continual_population.py（对称头、Adam、状态保存/恢复）'],['D','scripts/step28_continual_population_data.py（账号/对标签对齐、Algorithm R）'],['X','scripts/step28_continual_expression_run.py（公开输入装载及监督附着）'],['U','scripts/step28_continual_population_run.py（资源和保存预算辅助函数）'],['Q','scripts/step28_bge_continual_audit.py（提交方新增无标签结果审计）'],['N','scripts/step28_bge_continual_check.py（项目原生CPU核验）'],['RESULT','docs/SELLER_ALIAS_BGE_RESULT.zh.md（待审结果正文）'],['J','reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job/'],['CPU','reports/seller_alias_continual/20260930/bge_continual_cpu/20260930_142608/job/'],['A','reports/seller_alias_continual/20261001/bge_continual_result/'],['HIST','reports/documentation/20260930/continual_plan/implementation/review/']])
text('''### 1.2 四种证据层级不能互相替代
''')
table(['层级','实际材料','能够支持 / 不能支持'],[
 ['历史网页实现核验','HIST/response_visible.zh.md；external/REVIEW.zh.md完整F1—F12；独立源码、diff、成功/失败日志','支持当时手工/微型路径；不是本次正式数据结果，更不是原生BGE实际288步'],
 ['项目原生CPU核验','CPU/stdout.log、stderr.log、contracts.json、seq/er/logit.json、audit.json、起止/退出/资源日志','原生BGE确实执行6个手工更新，22项合同测试；部分计数由夹具1→288，再执行一步到289，不能冒充真实288步'],
 ['正式实际运行','J/train.log、run/manifest.json、21点记录及更新数组、完整盲门/监督账、completion、起止/退出/GNU资源日志','支持实际6048更新、9504呈现、首域共享和保存恢复的正式记录；大权重未入包，当前未独立重新加载'],
 ['保存结果审计','A/audit为执行者审计；本次独立scripts/02、03、05为新参考，另补跑未改的Q','支持无标签输出的哈希、计数、仿射、端点、区间和验收复算；不支持从正式标签重新计算AP/MAP等']])
text('''## 2. 合同、真实代码及运行记录逐项对照

### 2.1 研究对象、监督边界与输入对齐

任务是不同卖家群体按域到达后，对旧分布留出卖家的中文合成马甲候选排序/基础识别保持；不是同一控制者跨阶段开新号的真实市场追踪。每群28账号，378个无向对，20个同控正对；每个查询在其余27账号中排序，有1或2个正候选。合同的数据构造包含经营内容与表达变化，不能仅凭此轮结果宣称真实市场效果或把全部变化单因果归于表达。

每个实际域的60个训练侧群按照公开、模型无关的哈希规则分为48拟合/12当前校准；另有20个valid群。独立按公开ID重建了分区，核对总240个群及训练/校准/valid角色互斥，三个顺序复用同一划分。这里的ID计算仅用于分区与账目，不恢复身份。

模型输入路径只把屏蔽标题、描述送入编码器。公开群、账号、域及商品ID用于分组、排序、样本对齐和校验，不进入神经网络特征。账号/商品顺序固定，监督表按明确的账号对键附着到相同上三角边序，不依赖CSV行号恰好一致。BGE标题/描述分别求均值和总体标准差，拼出4096维账号表示；对称关系输入由绝对差与逐元素积构成，再经8192→128→1关系头。没有把域标签隐式加入头输入的活跃路径。

公开文本一次预检、离线数据预装与“学习器逻辑上只用当前拟合群/允许的记忆”并不矛盾；合同不是声称操作系统级恶意隔离。正式控制流先全部盲推理与保存，再通过完整门，最后一次性读取valid监督、采集评价。记录中门前train=1、valid=0；完成时train=1、valid=1、heldout=0、owners=0。它们是来源绑定的正常程序执行证据，不是外部系统调用跟踪。本次没有把当前容器的公开元数据读取记成项目标签读取。

证据：C:L21—43；D:L65—116；X:L58—128；B:L110—191、194—230；R:L76—107、244—373、482—548；本次`independent_training_evidence.json`中的分区重建、门前后账目与源绑定结果。未从包外取得正式标签，不能独立重建实际20个正对的位置。

### 2.2 初态、首域共享、Adam与阶段局部学习率

每个顺序从原始预训练BGE和相同s0随机头新建。没有复用已经联合学习过三域的历史中文模型当首域起点。三个顺序初始模型/分数摘要一致；各自首域实际训练288步后，保存模型、Adam及RNG，再为SEQ/ER/LOGIT建立新实例并恢复该同一状态。frozen直接固定首域模型与首映射，不新增后续训练。

阶段局部编码器学习率在第s步为：s≤29时`1e-5*s/29`；其后为`1e-5*(288-s)/259`。第288步编码器学习率恰为0，头仍是0.001；不能因为这一步编码器参数不变就报告冻结或漏训练。Adam计数连续288→576→864，阶段间没有重置优化器动量。本次逐一复算21个288行数组的全部学习率、计数、当前域序列及六轮呈现；观察点1/29/30/288的梯度和变化范围与该语义一致。

真实首域共享是正式记录支持的事实，不是用原生CPU“计数夹具”冒充。另一方面，当前没有24份大模型/完整状态载荷，不能把正式记录中的`restore_verified`改写成“我本次实际重新加载并比对了全部张量”。

证据：T:L50—62；P:L145—210；R:L103—241、376—479；J/run/points、J/run/manifest.json；本次21点状态、学习率和观察点重建。

### 2.3 真实目标、归约、配对及计算量

令每个群的监督目标为`L_group = BCE + 查询排序项 + 0.5×top-5困难负例项`。困难负例只能从该群已知负对选择；选择索引不反传，所选分数仍参与梯度。正例先按查询做规定的平均，不把有两个正候选的查询当两个独立查询任意加权。

SEQ只用当前群目标。ER在一次更新中分别计算当前群和一个历史群的群目标，二者相加，不除以2。LOGIT在同样当前/历史监督之上，加`0.5×mean_378[(z_student_train−z_reference_eval)^2]`。当前与历史分别前向/反向，合并梯度后只做一次范数裁剪和一次Adam更新。LOGIT的MSE复用本次历史学生前向，参考detach；不是额外训练教师，也不是校准概率MSE。

ER/LOGIT同缓存成员、同历史抽取、同当前顺序；每步使用配对训练种子，当前前向与ER/LOGIT历史前向的随机状态契约一致。分支参数随训练不同，因此“配对”不意味着所有后续梯度或logit相等。相比SEQ，ER/LOGIT多了历史群前后向和更大的组合目标；本比较不是等计算量、等总目标尺度的纯记忆信息消融。正式合同与结果正文已经披露，不作为事后实现缺陷。

独立核验6048行的记录目标组合：记录总目标与`当前+历史+0.5MSE`完全一致；分项由float32记录后转double求和的最大差为3.427267074584961e-7，按预先独立脚本中的1e-6容差核查，不据结果放宽阈值。SEQ历史为0，非LOGIT的MSE为0。由21×288得到6048次实际更新；其中12个ER/LOGIT后续点各288次历史呈现，共3456，合计9504次群梯度呈现。

本次没有对正式张量重新自动微分。梯度正确性所依赖的历史手工参考、真实原生CPU梯度探针与正式运行记录各自单列，不能用一张损失表替代正式梯度重算。记录的MSE host enqueue时间也不是准确独立GPU核耗时。

证据：C:L51—60、80—102；H:L89—137；T:L231—301；R:L103—153；本次`independent_update_diagnostics.*`及`independent_training_evidence.json`。

### 2.4 历史选择、淘汰、预算与LOGIT参考

按Algorithm R对唯一拟合群顺序处理，首域48群、再到96群；不是把每轮重复呈现重新投入水库。最多保留6个完整拟合群；阶段内成员固定，每更新抽一个历史群。淘汰之后的旧群没有在活跃学习路径重新读回。ER和LOGIT选择独立于分数/模型，选择种子和抽取均已用另一份参考从公开ID及记录状态重建。

LOGIT的新入库参考取自相应域学习完毕的eval raw logits；幸存群的参考不被下一阶段结束模型刷新。重建核对了参考原始域/阶段及幸存SHA；正式参考的378个数值正文没有回传，不能声称本次又计算了它们。完整状态文件属于实验保存/分支/评价产物，不是让学习器任意调用历史教师的免费记忆；源码中没有相应学习器访问路径。

**预算证据分两层：**21个实际点的附属JSON元数据可在本包直接规范序列化复测，为21,065—21,181字节；36个可见完整记忆摘要报告191,740—260,623字节，均小于1MiB，其SHA和附属状态变化关系一致。12个完整记忆正文含正式文本/标签，被明确排除在包外，因此完整正文总字节只能核对来源绑定的正式摘要，不能宣称独立重新测量。该260,623是本包36个可见摘要的最大值，不冒充所有时刻的独立实测峰值。

证据：C:L64—74、86—102；T:L109—219；D:L202—258；R:L90—100、156—227、376—479；本次`independent_memory_diagnostics.json`及21次附属状态字节复测。
''')
table(['训练顺序/阶段','历史成员域构成','288次抽取的域分布','单历史群呈现次数'],[[f"{x['order']} / {x['training_stage']}",', '.join(f'{k}{n}' for k,n in x['domain_counts'].items()),', '.join(f'{k}:{n}' for k,n in x['draws_by_domain'].items()),f"{x['minimum_draws_per_group']}—{x['maximum_draws_per_group']}"] for x in MEM])
text('''CAB第三阶段表中未列出的首域C即C0。当前48个群各出现6次，历史6群平均48次，相当于每群平均8倍重复，但不等于每个群的梯度贡献恰好8倍。

### 2.5 校准角色、模型绑定与排序不变

每个真实训练终点只使用当前域12个校准群，共4536对/240个正对，拟合`p=sigmoid(a*z+b)`，a为正。目标是未加权NLL，不是优化0.5召回、Recall@5或valid曲线。参数边界、优化选项、停止记录、最后迭代a/b、投影梯度记录及模型/分数SHA都与冻结实现一致。21次拟合记录都报告成功，末次NLL不大于初始值；本次没有正式校准标签，未重新证明其损失、梯度或最优性。

主角色的14个曲线/排序列取raw，Brier/log_loss与6个固定阈值分类列取stage-cal。first-cal仅用于固定首映射诊断，不为选出有利结果而切换。frozen固定首域模型与首映射。

正斜率在实数意义上保持排序，还必须确认保存浮点数组没有改变并列。本次直接检查原raw到stage-cal/first-cal的双精度仿射数组，逐群全部378条边的完整排序与并列关系，以及28个查询各27候选的完整顺序；不是只看一个相关系数或只看Top-5。2520个群/变换检查、70,560个查询检查通过，14列保存曲线/排序数值也逐项完全相同；两个变换合计观察到72处相邻并列关系，都保留。

**概率保护在本合同中评价的是“模型＋当前域映射”的结果。** 排序不变不意味着概率不变，raw输出损失改善也不等于相对同样stage-cal的SEQ不退化。旧域NLL/Brier变动不能被解释为纯粹的表示遗忘，也不能用概率改好掩盖固定工作点退化。

证据：C:L106—112；K全文尤其L75—91及拟合函数；R:L156—227；V:L91—122；本次`independent_calibration_records.*`、`independent_result_audit.json`。

### 2.6 正式运行、保存顺序与资源证据

原始开始/结束为北京时间2026-09-30 14:40:21至2026-10-01 06:06:13，退出码0。GNU elapsed为15:25:51，两个秒级时间戳相差15:25:52，程序内部55,549.224442秒；三者计时边界不同，不是需修复的秒级矛盾。

正式21点为3个共享首域点加18个后续分支点；保留21个推理权重与3个共享完整状态，共24份大载荷。85个盲分数数组包括初始、各点开发/校准raw与两个仿射角色。一次valid采集后写64个60×22矩阵及64套群计数，再执行统计。这里64=初始raw 1份＋21点×3角色，不是只保存64份中的有利结果。全部盲门与完成账目一致，原train.log有252个24步观察事件；只有已知接口改名FutureWarning，无正式训练异常退出。

正式环境记录Python3.10.19、Torch2.9.1+cu130、CPU0、GPU0 RTX5090。程序最大观察产物41,787,374,219字节，小于64GiB产物上限；GNU最大RSS7,084,352KiB是内存口径，不与产物上限混用。CPU核验的256MiB也是证据产物限制，不能把约10GiB原生模型RSS误判为违反该项。

保存、恢复验证与额外恢复故障注入必须分开：实际正式记录声明21点原生保存恢复验证完成；历史手工测试里33个伪权重/分数损坏门测试及4个统计故障保全测试没有在本轮正式输出上再次注入。不能写成“本轮重新运行全部故障测试”。

后续`connection_status.json`中的SSH reset/timeout属于完成/回传后的访问问题；原回传记录中的过早解包失败也不是科研运行失败。当前Linux清理是否完成、GPU进程是否仍为空不在本次独立已验证范围内。

证据：R:L244—373、482—548；V:L177—254；U:L38—74；J原始起止、退出、GNU资源、train.log、complete/before_valid/manifest；A/return_inventory.json、return_sync.json、connection_status.json。

## 3. 回读既有F1—F12：闭环，而不是重开

已回读实际可见回复`HIST/response_visible.zh.md`、完整`HIST/external/REVIEW.zh.md`、主审处置、数值修正及必要独立源码、版本差异和成功/失败原始日志。上一轮“原生待做”“正式待做”是当时的时点陈述；本次用后来保存的实际记录补足相应层级，不把已关闭问题再列成新缺陷。
''')
closure=[
 ['F1','阶段可见性与旧域留出泛化','历史手工门＋本次活跃路径阅读、公开分区独立重建和正式监督账','维持接受；非操作系统隔离保证'],
 ['F2','目标、归约、梯度','历史独立目标/梯度；项目原生6更新探针；本次6048行目标记录','维持接受；本次未重做正式张量反传'],
 ['F3','LR、完整Adam/RNG分支','历史微型真实288步并恢复下一步；原生计数夹具；正式21点和288/576/864记录','原生/正式证据已补入；当前不重载大权重'],
 ['F4','六群、1MiB、淘汰/旧参考','本次独立Algorithm R/抽取/幸存参考SHA；实测21个附属JSON','接受所示证据；完整记忆正文仍未独立复测'],
 ['F5','三角色及当前校准','本次21个拟合记录、全部仿射与排序/并列、固定首映射绑定','维持接受；P1旧示例修正已关闭'],
 ['F6','完整盲门和一次解析','本次实际21点、85数组、门前valid=0、后valid=1','正式记录补足；不把旧故障注入说成本次重做'],
 ['F7','先保存64矩阵/计数','本次64+64齐全、摘要匹配、7326行复算；旧四处故障保全','维持接受；未破坏正式结果做注入'],
 ['F8','O/N/Z/F/G与域映射','本次直接R(t,j)独立算2464端点指标情况','接受，损失方向和最终全域恒等式一致'],
 ['F9','两项23条件及多种子','本次46项逐项重算；SEQ遗忘门重算','ER4/23、LOGIT23/23；未来多种子仍未评价'],
 ['F10','22指标和条件区间','历史6×22手工指标；本次计数6指标、域配对5000次、全部表','接受已存统计；不声称正式标签级AP/MAP重算'],
 ['F11','来源、预算、原生边界','18来源和326原输出；项目CPU原始日志；正式6048/9504与资源记录','维持接受，网页CPU与项目原生两层分开'],
 ['F12','历史与科学定位','旧基础test并未重新封存；当前不主张创新；新留出待新合同','维持原边界，不将这轮诊断包装为新方法']]
table(['项','原问题','本轮证据','处置'],closure)
text('''### 3.1 已关闭的主审修正

P1是历史外审说明文字的解析例数值错配，不是项目损失实现错误。当前示例`p=20/378`，从`p±0.04`改为`p±0.01`，保存的正确log_loss增量为0.010836263997970538，Brier增量0.0009；约0.01154属于前一份不同概率分箱的例子。原错误回复、修正来源和字节都已保留，本次没有把该旧例重新包装成结果缺陷。

P2是历史日志覆盖说明过宽：首个`verify_submission`只有源码/stdout/stderr/exit，没有命令/时间JSON；其后16项有。主审已经明确更正，不能编造丢失原始元数据，也不需要为这个旧遗漏重新做不变实验。

历史独立参考v1为6通过/2失败/2错误，v2为9通过/1失败，v3为10通过；额外指标初稿API调用形状错误，修订后6×22=132指标通过。原因包括softplus运行时阈值语义、float32 BCE分项的两次舍入、手工预算对象接口、校准函数名以及1维/2维指标接口。原稿、诊断、diff和stderr都在原包，不是执行者隐藏失败后只交最终PASS。Windows主审GBK解码失败修为显式UTF-8也不是正式训练失败。

本次没有重新执行这些历史手工测试，故它们仍然是历史实际日志支持的证据；后续项目CPU的22项测试和6原生更新是另一次真实运行。

## 4. 本次独立无标签数值核验

主参考`scripts/02_independent_results_v1.py`**不导入项目模块或提交方审计脚本**。先用保存矩阵重新计算实际域均值，再直接写出R(t,j)端点式并在每次抽样后组合，不采用项目/提交方的系数场矩阵公式；因此不是简单再调用同一评价函数看是否返回同一JSON。另有独立训练账参考`scripts/03_independent_training_evidence_v1.py`。
''')
table(['实际执行检查','覆盖与结果'],[
 ['原始64指标矩阵','每份60×22，float64、有限；群/域顺序和角色齐全'],['混淆计数和阈值预测','3840个群/角色：整数、非负，正数20/负数358，score≥0预测数=TP+FP'],['由计数重算六分类指标','precision、recall、F1、specificity、balanced accuracy、MCC，共23,040值'],['raw及仿射分数','85数组、模型/映射/当前校准绑定；全部仿射数组精确一致'],['完整排序/并列','2520群/变换及70,560个查询顺序；并列保持'],['实际域bootstrap','PCG64(20260930)，(5000,3,20)，独立再生与原抽样完全相同'],['端点与区间','4方法×4角色×7端点×22指标=2464情况；均值/各顺序/条件区间'],['预定验收','两项23条件共46项＋SEQ遗忘门全部逐项一致'],['完整阶段表','7326行，每行身份、角色、端点、数值与来源别名核对'],['数值比较总数','1,043,476；最大绝对差5.551115123125783e-16'],['结果正文交叉核对','148个表格数值一致；六位小数最大舍入差4.920983641e-7'],['独立源码首次运行','主数值参考和训练账参考均首次运行通过，未根据失配改公式']])
text('''六分类指标中的precision/F1/MCC零分母按冻结原语定义为0。检验`TP+FP=预测正数`不等于从分数确认TP和FP分别是哪一些样本；真实标签未提供。AP、MAP、MRR、Recall@K、NDCG@K、PR/ROC曲线、Brier和log_loss在本次参考中是保存矩阵的输入值，而不是从正式真值独立重新求出的输出。对应算法原语做了静态阅读，历史手工/原生证据另列。

### 4.1 端点、符号与抽样单位

对一个到达顺序，R(t,j)表示学完阶段t后、第j个到达域上的群宏指标。实际域A/B/C与到达位置j严格区分。

- `O=(R(3,1)+R(3,2))/2`：最终旧域。
- `N=(R(2,2)+R(3,3))/2`：新域刚学完。
- `Z=R(3,3)`：最终最新域，防止被N平均掩盖。
- `F_first=R(1,1)−R(3,1)`；`F=[R(1,1)−R(3,1)+R(2,2)−R(3,2)]/2`。
- `G=[R(2,2)−R(1,2)+R(3,3)−R(2,3)]/2`。
- `final_all=[R(3,1)+R(3,2)+R(3,3)]/3=(2O+Z)/3`。

Brier/log_loss在F_first/F/G中翻转符号，使遗忘正值仍表示变差、学习增益正值仍表示改善；在普通“候选−参考”的差值表中不翻转，二者≤0才是概率损失保护。本次逐项核对，不用高低方向混写。

对实际A/B/C各20个valid群有放回抽样5000次，每次抽到的同一群同时用于全部方法、阶段和顺序；随后对三顺序平均。独立分位数按排序后位置`(5000−1)q`做线性插值，q为0.025/0.975。三顺序不是180个独立群，也不是三个独立训练种子；没有在bootstrap内重训模型或重新拟合校准器。

这些区间条件于此次数据划分、开发valid、训练种子、三条路径、模型及映射，不包括训练随机性和数据构造选择不确定性。23条是一个预定合取规则，不是23次独立实验、23个独立显著性证据或同时置信保证。点估计“非下降”条件不能改写为总体非劣效成立。

证据：C:L118—160；V:L91—175；本次`independent_bootstrap_draws.npy`、`independent_endpoints.*`及`independent_23_conditions.csv`。

## 5. 首域先学会、随后遗忘，同时新域学习：门通过
''')
table(['顺序','初始MAP','首域后MAP','第二阶段首域MAP','最终首域MAP','首域遗忘','新域G'],[[r['order'],f(r['initial_map']),f(r['first_map']),f(r['stage2_first_domain_map']),f(r['stage3_first_domain_map']),f(r['F_first_map']),f(r['G_map'])] for r in F])
ff=S['seq']['primary']['F_first']['map'];gg=S['seq']['primary']['G']['map']
text(f'''首域MAP平均遗忘 **{f(ff['mean'])}**，条件95% **[{f(ff['conditional_95pct_interval'][0])}, {f(ff['conditional_95pct_interval'][1])}]**。平均新域G为 **{f(gg['mean'])}**，条件95% [{f(gg['conditional_95pct_interval'][0])}, {f(gg['conditional_95pct_interval'][1])}]。

首域Recall@5初始→学完分别为ABC 0.176786→0.602679、BCA 0.147321→0.648214、CAB 0.153571→0.621429。因而“首域本来没有学会，所以谈不上遗忘”与本次记录不符。三条完整路径均有正的首域遗忘和正的新域G，超过冻结门所需的至少两条；平均遗忘及其区间下界也为正，三项检查均通过。

CAB首域在第二至第三阶段从0.433074回升到0.441522，所以只支持首域最终相对刚学完下降，而非每次转移单调下降；路径共存不等于直接测量出某一次更新的梯度冲突。

## 6. 两项正式比较：全部23条逐项结果

本表统一写“候选−参考”；AP/AUC/MAP/Recall越大越好，Brier和log_loss越小越好。第2行是ΔO MAP条件区间的下界，不是又一个均值。第10/11/16/17行专门比较候选stage-cal对参考raw，其余概率比较是双方各自stage-cal。阈值按冻结合同保留为0。
''')
labels=['O MAP','O MAP区间下界','O Recall@5','N MAP','N Recall@5','O AP','O ROC-AUC','O Brier','O log_loss','O Brier vs参考raw','O log_loss vs参考raw','N AP','N ROC-AUC','N Brier','N log_loss','N Brier vs参考raw','N log_loss vs参考raw','Z MAP','Z Recall@5','Z AP','Z ROC-AUC','Z Brier','Z log_loss']
table(['#','条件','方向','ER−SEQ','通过？','LOGIT−ER','通过？'],[[g['index'],labels[g['index']-1],g['operator'],signed(g['er_minus_seq']),'是' if g['er_pass'] else '否',signed(g['logit_minus_er']),'是' if g['logit_pass'] else '否'] for g in G])
text('''**ER−SEQ仅第10、11、16、17条通过，其余19条全部失败。** 这4条只说明ER的校准输出比SEQ的未校准raw参考在指定概率损失上好，不能写成“ER比同样校准后的SEQ好”。ER的ΔO MAP为−0.039873，条件95% [−0.054070, −0.025440]。

**LOGIT−ER全部23条通过。** ΔO MAP为+0.046887，条件95% [+0.038783, +0.054931]，三个顺序的O MAP也均高于ER。该成功是对这个预定对照的成功，不会因ER本身很差而变成无效；但也不会自动升格为优于SEQ或多种子稳定成功。

完整精度、每项bool与来源字段保存在`independent_23_conditions.csv`及`independent_comparisons.json`。本轮未增加LOGIT−SEQ的条件区间或新验收。

## 7. 是否满足用户“保旧且学新、概率保护”的整体目标

### 7.1 原定O/N/Z排序端点的绝对值
''')
table(['方法','O MAP','O Recall@5','N MAP','N Recall@5','Z MAP','Z Recall@5'],[[a.upper() if a!='frozen' else a]+[f(v(a,e,m)) for e in ('O','N','Z') for m in ('map','recall_at_5')] for a in ('frozen','seq','er','logit')])
text('''frozen旧域表现较好但新域较弱，正说明只保护旧域不够；不能把不继续训练作为用户“保旧且学新”要求已经解决的证据。LOGIT相比ER在各主排序端点均改善，但相对SEQ有以下描述性差异。
''')
table(['端点','ΔMAP：LOGIT−SEQ','ΔRecall@5：LOGIT−SEQ','解释'],[[e,signed(D[e]['map']),signed(D[e]['recall_at_5']),label] for e,label in [('O','旧域MAP略好，旧域R@5更差'),('N','新域刚学完的两项均更差'),('Z','最终最新域两项均更差'),('final_all','最终全域两项均更差')]])
text('''LOGIT−SEQ的O MAP按ABC/BCA/CAB分别为+0.020049、+0.002061、−0.001071。因此也不能写成“所有顺序都超过SEQ”。这里不检验事后新门，只用绝对表现阻止对已有23/23比较对象的偷换。

### 7.2 AP/AUC与概率保护
''')
table(['方法 / 端点','AP','ROC-AUC','Brier↓','log_loss↓'],[[a.upper()+'/'+e]+[f(v(a,e,m)) for m in ('average_precision','roc_auc','brier','log_loss')] for e in ('O','N','Z') for a in ('seq','er','logit')])
text('''LOGIT的旧域AP、AUC、Brier和log_loss略好于SEQ，不能将用户摘要中的“只有旧MAP略好”机械扩展成所有指标上的字面判断；真正关键的是，它没有同时满足旧域Recall@5、新域及最终最新域的保护。N和Z上的上述四项均比SEQ差。最终全域stage-cal Brier为SEQ 0.045078、LOGIT 0.045273，log_loss为0.181313、0.181812，也不支持整体概率优势。

此外，当前域校准使O log_loss从raw降到stage-cal：SEQ 0.225775→0.184187，ER 0.275197→0.192677，LOGIT 0.235852→0.182699。这是保存概率损失的观察，不是证明每个域“概率完全校准”，也不保证可用自动判定工作点。first-cal作为诊断保留，不因哪项好看而换成主结果。

### 7.3 固定0.5工作点必须保留

在保存的正仿射logit上，`p≥0.5`等价于`a*z+b≥0`。ER三个最终模型在全部60群上的最大校准logit依次为−0.251455、−0.352682、−0.527280，对应最大概率约0.437465、0.412732、0.371151。**三者都没有任何正预测**；每个模型的计数均为TP=0、FP=0、FN=1200、TN=21480。
''')
table(['方法','三个最终模型的正预测数','群宏固定0.5召回','群宏precision','合并计数precision','Recall@5'],[[w['method'].upper(),'/'.join(str(z['positive_predictions']) for z in w['records']),f(w['group_macro_recall']),f(w['group_macro_precision']),f(w['pooled_precision']),f(w['recall_at_5'])] for w in W])
text('''合并计数precision一列只是对同60群上三次模型评价的描述性账目，绝不能当180个独立群推断。由于每群恰20个正对，群宏召回与该描述性合并召回恰相同；precision没有这一恒等性。

ER的specificity=1、balanced accuracy=0.5、MCC=0来自全负预测，不能掩盖召回0。LOGIT固定阈值召回也只有0.030278，低于SEQ的0.064444。候选Recall@5分别是0.497619和0.545040等另一种排序度量，不是0.5阈值召回；`recall_at_fpr_1pct`又是第三种曲线统计，不能混用，不能把某个低阈值结果当作已批准部署阈值。

上述观察不宣布0.5是唯一合理部署阈值，也不偷偷新增0.5召回成功门；它只是忠实保留原工作点结果和用户基础识别要求。当前不重选阈值、不重新拟合、不读取标签。

### 7.4 最终全域完整22指标
''')
metrics=list(S['seq']['primary']['final_all'])
table(['指标','frozen','SEQ','ER','LOGIT'],[[m]+[f(v(a,'final_all',m)) for a in ('frozen','seq','er','logit')] for m in metrics])
text('''AP与梯形PR-AUC分开；Recall@K是每查询找回正候选比例，不是Hit@K；上述分类列是群宏，不是把所有计数混合后再求值。LOGIT的最终全域recall_at_fpr_1pct略高于SEQ，specificity也高，不能因此否定更低MAP/Recall@5及固定召回的取舍；同样不能声称LOGIT相对SEQ“22项全败”。

## 8. 缓存、重复和梯度裁剪：已有线索与因果边界

### 8.1 缓存失域属实，但不是ER退化的充分单因果解释

独立重建CAB第三阶段为A6/C0，首域C确实完全失去历史样本；ABC为A4/B2，BCA为B2/C4。Algorithm R无最低域配额，因此该现象在冻结随机方案内，不是当前实现漏保留。不能事后重选水库种子、顺序或成员来让本轮胜出。

ABC和BCA也有ER旧域MAP退化，故CAB失域无法单独解释全部负结果。ER/LOGIT同成员同抽取而结果差异大，也说明“六群是什么”之外，训练目标/优化过程确实值得研究，但现有输出仍不足以隔离具体机制。

### 8.2 缓存拟合与留出泛化分离：相容，不是因果证明

历史群每阶段34—63次、当前群各6次；ER最后48步历史组合损失约0.396663—0.579454，当前约1.723963—2.873204。低历史损失和旧域留出退化与缓存过拟合相容，但群的难度、目标分布、标签熵、参考年龄和目标尺度都可能不同。不能仅比较两个平均训练损失就证明过拟合是唯一原因，也不能把本轮命名为已验证的“重放过拟合机制”。

### 8.3 本次新增诊断：后两阶段每一步都发生全局裁剪

从保存的更新数组独立统计，阈值为1.0时，全部6048次更新中6030次裁剪；后两阶段的5184次全部超过阈值。该统计来自未修改原数组，见`report_crosscheck_v2/report_crosschecks.json`与`independent_update_diagnostics.*`。

这使“组合梯度如何影响实际更新方向”成为有根据的候选机制，但不是现成因果结论。全局裁剪仅按同一系数缩放当步总梯度；总范数无法恢复当前、历史和MSE各分量的方向、相互抵消或Adam预条件后的更新。不能由全部裁剪直接推断编码器无效、历史梯度一定压制当前梯度、或调大clip一定改善。阶段最后一步编码器LR=0也是合同设计，不是裁剪造成。

这项补充值得随结果附注，但结果正文未列裁剪统计不构成撤销现有诊断的缺陷。正式配对的LOGIT−ER改善仍然有效，其具体原因尚未隔离。

## 9. 发现与处置分类

下表将“阻止更强科学主张”和“当前交付错误”分开。没有为了给出新缺陷而把已披露的负结果标成代码bug。
''')
findings=[
 ['S1','SCIENTIFIC_BLOCKER（更强结论）','“LOGIT整体优于SEQ，保旧且学新已解决”','RESULT:L79—105；本报告§7、descriptive_logit_minus_seq.json：O R@5−0.004315，N MAP−0.025321，Z MAP−0.028013','只保留“LOGIT−ER预定验收通过”；明确SEQ是必要绝对参照。不改本轮门，不补事后LOGIT−SEQ正式验收。当前正文已这样披露。'],
 ['S2','SCIENTIFIC_BLOCKER（工作点/概率泛化）','“候选检索用途可以不报告固定阈值退化”；“校准损失下降即基础识别已保护”','RESULT:L117—125；三ER最终max cal logit<0、预测正数0；N/Z概率损失LOGIT仍劣于SEQ','保留raw排序、stage-cal损失、固定0.5计数及first-cal诊断的不同角色。只限定主张，不重选阈值/标签。'],
 ['S3','SCIENTIFIC_BLOCKER（因果）','“CAB丢失C导致所有ER退化”；“低缓存损失已证明过拟合”；“裁剪证明梯度冲突”','RESULT:L111—115；独立memory/update诊断；ABC/BCA也退化，合成总范数无方向信息','改为相容线索/候选机制，提出新合同下的区分性对照。当前正文已保留主要因果限定。'],
 ['S4','SCIENTIFIC_BLOCKER（外推与创新）','“三个顺序等于多种子”；“23/23证明总体非劣效/新方法”','C:L118—160；RESULT:L59—61、156—173；同60群，s0，开发valid，已知方法','明确条件区间、预定合取规则、无多种子/独立最终留出/创新保证；当前诊断无需为此重训。'],
 ['R1','REPRODUCIBILITY_DEFECT（低严重度）','RESULT引用的额外进程/GPU观测可直接随包核验','RESULT:L22、L30引用A/observation.json，但495载荷无该路径；A/current_status.json只有提交状态摘要，非完整原始进程查询输出','在后续追加说明中把链接改到确实随包的状态记录，并将PID/GPU为空限定为执行者当时报告，或移除无原始支撑的附带主张。保留原18来源及原输出，不要求再连服务器。'],
 ['O1','OUT_OF_SCOPE_OVERDESIGN','以缺少新GPU运行、原生大权重重载、新标签重算、多种子或新阈值为由拒收本轮诊断','用户授权仅包内无标签小结果；C冻结范围；包明确排除权重/标签','本轮注明未验证即可，不执行/要求越界工作。新方法、多种子或新独立留出均需新合同。'],
 ['O2','OUT_OF_SCOPE_OVERDESIGN','重开P1/P2、把returned/job映射当缺失、要求额外签名/系统/网络仪式','HIST/disposition.json、primary_numeric_correction.json；source_inventory.json映射','沿用已关闭修正和明示映射；只修真实缺失observation链接，不扩大成来源危机。']]
table(['编号','分类','受影响主张','文件/行/实际证据','最小充分修正'],findings)
text('''R1不阻断结果有效性：科学运行结束有原始`finished.txt`、`exit_status.txt`、GNU资源记录和completion等直接证据；缺的是附带“两个进程不存在/GPU为空”的观测记录，不是训练退出或矩阵。没有发现需改正式科学源码、重跑训练、重读标签或变更冻结结论的实质可复现性缺陷。本次审查者自己的失败详见§11，不能算成项目缺陷。

## 10. 下一步建议：只提出新合同中的可辨别科学问题

本节没有执行任何新实验，不构成授权；也不保证改善、原创性或CCF B/C发表。已知重放、分数蒸馏、配额和梯度诊断本身不能直接包装成创新。

### 10.1 区分“记忆覆盖不足”和“同群重复过强”

假设A：六群对旧分布代表性不足/发生旧域全丢。假设B：即使覆盖旧域，过强重复及目标权重仍损害留出泛化。可在新合同中设计小型预定对照：同6群/同字节上限的Algorithm R与保留最低旧域覆盖的规则；另一个独立因素改变历史监督权重或重复强度。保留相同当前呈现、初始化和预定随机种子，把成员覆盖和监督强度分开解释。

最低覆盖方案只能从当时允许的缓存和当前域群构造，不能从已经淘汰的旧全库读回；域元数据不进入模型。改变重复次数会改变计算量，需预先说明以群梯度呈现、优化器更新或总计算哪一项配平，不能宣称自动同时等价。若强度下降在不改变成员时就缓解ER退化，与仅覆盖规则有效的结论不同。不要因本次CAB不利而事后重选一个“更均匀”的随机水库当对照。

### 10.2 区分“目标信息有用”和“组合优化几何改变”

本轮ER采用当前+历史之和，后两阶段全步裁剪。新合同可以在严格配对数据下采样记录当前/历史/MSE各自梯度范数、余弦、合并前后方向以及Adam实际参数更新；同时预定比较固定目标权重/归一化方式，而不根据valid结果逐次加大或缩小保持系数。

需要控制额外诊断反向的计算量与内存，不把诊断当免费。若无梯度方向记录，单看低历史损失或总范数依然不能区分“冲突”和“同向但比例不合适”。若目标归一化有效，也只能说明当前约束下的一种机制，不足以认定通用抗遗忘创新。

### 10.3 区分“raw分数坐标保持”和“候选关系保持”

现有MSE同时约束分数位置、尺度和对间相对差，而排序对正仿射变换不敏感。可在新合同中，对同一缓存/参考年龄/存储预算对比raw-logit MSE与明确去中心化或候选间相对间隔的约束；同时保留独立概率保护，而不是假设排序保持就保护概率。所有附加参考、邻接结构或统计量都计入1MiB，不能让方法把旧数据偷偷转移成不计费软状态。

还应区分“学生train/参考eval”的随机噪声正则效应与真实历史关系保持。可设计匹配前向模式或参考噪声的控制，但增加的前向、教师副本或目标刷新必须事前纳入预算；幸存群刷新只能使用仍允许的群。若只改变dropout/参考年龄就取得相同收益，原来“关系保持机制”的解释就需要缩小。

### 10.4 后续验收不要只围绕击败ER

新候选至少同时报告SEQ绝对基线与ER/LOGIT配对对照，保留O/N/Z、候选预算K、基础识别/概率和固定工作点口径。用户的复核预算尚未据本轮确定，不能事后以另一个K证明本轮成功。多种子与新的独立最终留出可在新合同明确，但不是撤销当前有效负结果的先决条件；已用于开发或旧基础test的群不能重新命名为未见独立test。

真正有价值的下一步是一个能排除替代解释、同时改善保旧和学新的机制与证据组合，而不是把本轮23/23换个名字叫新算法。

## 11. 本次实际执行、失败与修订账

以下耗时为本次容器命令壁钟，和正式训练15小时耗时无关；内部脚本计时另保存在输出JSON。实质核验均保存了源码、精确命令、cwd、UTC起止、退出码、原始stdout/stderr与其SHA。交互式文件阅读/目录查看没有逐次生成命令计时回执，不能声称所有浏览动作都有该回执。
''')
logs=[]
for path in sorted((B/'logs').glob('*.json')):
 j=json.loads(path.read_text());logs.append((path.stem,j))
desc={'00_inventory':'清单初稿，缩进错误，核验未完成','01_revise_inventory':'只修审查者缩进，保留原稿','02_inventory_v2':'全部ZIP及SHA、环境核验','03_catalog':'只读结构/字段清点','04_independent_results_v1':'独立无标签数值参考（首次通过）','05_independent_training_v1':'独立训练/记忆证据参考（首次通过）','06_supplied_auditor_reproduction':'补跑提交方审计，误用CPU24，亲和设置失败','07_supplied_auditor_correct_cpu':'只改CPU参数为实际允许的0，源码不改','08_report_crosschecks_v1':'148数值一致；目录映射误报保留','09_revise_report_links':'只修审查者链接诊断目录映射','10_report_crosschecks_v2':'最终正文交叉核对；真实未入包观测链接1项','11_preserve_and_reverify':'复查495原载荷未改、保存原包/24关键文本副本'}
table(['运行记录','工作','退出码','壁钟秒'],[[n,desc.get(n,n),j['exit_code'],f"{j['elapsed_seconds']:.6f}"] for n,j in logs])
text('''独立主数值参考内部耗时3.162199秒，最大RSS638,956KiB；训练账参考内部0.056761秒。提交方未改审计Q在本次有效重跑内部约0.785秒，其1,061,316数值比较的最大差约7.1054e-15；它只是补充可复跑性证据，不替代本次先完成的另一实现。

当前环境Python3.13.5、NumPy2.3.5、SciPy1.17.0、pandas2.2.3、Torch2.10.0+cpu、scikit-learn1.8.0，Linux x86_64。没有transformers/sentence-transformers，没有安装任何新依赖；实际允许CPU为0—4。环境发现记录在`outputs/environment.json`，各命令固定OMP/BLAS/MKL线程为1。

### 11.1 本次两次实际非零退出与一次诊断逻辑修订

清单脚本初稿`00_inventory.py`发生IndentationError，退出1，保留源码与原始stderr；修订脚本只处理缩进，另生成`00_inventory_v2.py`，不覆盖第一稿。

补跑Q时审查者传入当前容器不允许的`--cpu 24`，`sched_setaffinity`抛OSError(22)，退出1，发生在实质审计和输出创建之前。`04_supplied_auditor_retry.py`读取允许CPU集合后选择0，保留原失败，未修改项目Q；Q原源码SHA为`2fff73cdc0e26843a10dfd35388ade1b31f9ca890d889c4c31f13b98efc1b193`。不能把审查者错误CPU参数报告成项目审计算法失败。

正文交叉核对v1已退出0、148个数值全部一致，但其自制链接逻辑只查文件级映射，对`returned/job`目录误报。v2通过清单中的子树映射修复该导航诊断，保留v1在outputs根目录；最终输出在`outputs/report_crosscheck_v2/`。数值逻辑、训练源码、正式输出和判据没有改变。v2剩余`observation.json`是真实缺失，而不是同类映射误报。

报告/打包生成的后续日志另收在logs；机器清单与`COMMANDS.md`比本节截至来源复查的表更完整。没有隐藏数值参考失败：两份主要独立参考实际均第一版通过。

### 11.2 明确未运行 / 无法由此包独立核实

本次未从正式标签重新算AP/MAP/MRR/Recall@K/NDCG@K、曲线、Brier/log_loss；未从预测数组恢复标签。未加载正式模型或进行原生BGE推理、反传、checkpoint张量比较。未独立重测12个含正式文本/标签的完整记忆正文或重算参考logit数值。未重拟合当前校准器或验证正式标签上的最优性。未重新执行历史手工故障注入、多种子、heldout或新阈值选择。未独立核验远端进程/GPU清理现状或作业外的全部文件读操作。

这些明确边界和现有正证据应一起保留：缺少标签级独立重算不等于本轮结果被否证，但必须限制“我实际验证过什么”的措辞。

## 12. 交付与可复核内容

`REVIEW.zh.md`为本完整报告；`COMMANDS.md`列实际精确命令及回执；`FAILURES_AND_REVISIONS.zh.md`单列本次失败/修订和历史失败层级；`scripts/`为实际执行源码；`logs/`为原始stdout/stderr/exit/命令元数据；`outputs/`为独立矩阵汇总、2464端点CSV、46条条件CSV、bootstrap抽样、训练/记忆/校准/工作点诊断、148项正文核对及输入清单；`reviewed_sources/`为关键原件副本；`input/`保留原始上传ZIP。`MANIFEST.json`和`SHA256SUMS.txt`覆盖交付内容，外部打包回执记录最终ZIP的大小/SHA、退出与耗时，避免要求ZIP自包含自身哈希的循环。

**最终处置：保留并接受这轮诊断结果，原样报告SEQ遗忘门通过、ER失败、LOGIT对ER通过；拒绝把它提升为已经整体优于SEQ、概率/工作点全面保护或已有实质创新。唯一必须补正的当前交付点是缺失观测链接及其附带主张的证据口径；任何新方法、新对照、新种子或新留出都另立合同。**
''')
report='\n'.join(parts)
(B/'REVIEW.zh.md').write_text(report,encoding='utf-8')
pathlib.Path('/mnt/data/bge_continual_result_independent_review.zh.md').write_text(report,encoding='utf-8')
failures='''# 失败与修订记录\n\n完整解释见REVIEW.zh.md第3、11节。以下均保留原稿和原始stdout/stderr，未用最终成功摘要覆盖。\n\n| 阶段 | 记录 | 失败/修订 | 责任与影响 |\n|---|---|---|---|\n|本次|logs/00_inventory.*、scripts/00_inventory.py|IndentationError，退出1；01修订为v2后02通过|审查者脚本缩进，未产生不完整核验结论|\n|本次|logs/06_supplied_auditor_reproduction.*|CPU24不在当前允许集合，OSError22，退出1|审查者参数错误，不是项目算法错误|\n|本次|logs/07_supplied_auditor_correct_cpu.*|实际允许CPU0；未修改Q源码；退出0|只修执行参数|\n|本次|logs/08_report_crosschecks_v1.*|退出0且148数值通过；目录映射诊断误报|审查者导航逻辑；第一版全部输出保留|\n|本次|logs/09_revise_report_links.*、10_report_crosschecks_v2.*|仅加入清单目录子树映射；最终仍真实缺1个observation链接|不改任何数值或正式源码|\n|历史网页|原包HIST/external/logs/independent_v1/v2/v3.*|6/2/2→9/1/0→10/0/0|已关闭历史参考实现修正；本次未重跑|\n|历史网页|原包HIST/external/logs/independent_metrics*.|初稿接口形状不对，修正后132手工指标通过|历史证据，不是正式标签重算|\n|历史主审|原包HIST/disposition.json、primary_numeric_correction.json|P1数值示例来源错配、P2首项命令时间JSON缺失、GBK解码修复|已披露关闭，不重开、不编造原始记录|\n|项目回传|原包A/return_sync.json、connection_status.json|过早解包、完成后SSH访问失败|不是正式训练异常；清理状态未独立验证|\n\n主数值参考scripts/02_independent_results_v1.py和训练证据参考scripts/03_independent_training_evidence_v1.py均首次实际执行通过，没有失败后调整终点、浮点容差或统计公式。\n'''
(B/'FAILURES_AND_REVISIONS.zh.md').write_text(failures,encoding='utf-8')
readme='''# 本次结果独立外审证据\n\n首先阅读REVIEW.zh.md。此包不包含额外项目标签、模型权重或完整历史文本；input内是用户本次上传ZIP的逐字节副本。没有任何新增训练或项目连接。\n\n主要独立参考为scripts/02_independent_results_v1.py和03_independent_training_evidence_v1.py，均不导入项目模块。05_report_crosschecks_v2.py是最终正文/工作点诊断；v1原稿及输出保留。Q的补充重跑不替代独立参考。\n\n输出最终版本：outputs/report_crosscheck_v2；根目录同名report_*保留第一版诊断，不应误用其已更正的目录链接误报。\n\n实际命令保留绝对路径/mnt/data/bge_input与/mnt/data/bge_review_evidence，与本次运行环境一致。COMMANDS.md及logs/*.json给出精确参数、cwd和依赖环境。它们是执行证据，不是要求用户在本地重跑；复现需先按相同路径解压input原ZIP，防止覆盖已有文件。原始失败也列入COMMANDS，不能把失败命令当推荐操作。\n\nMANIFEST.json给各文件大小/SHA；SHA256SUMS.txt覆盖全部载荷及MANIFEST。清单自身不自哈希。最终打包receipt在ZIP外，记录ZIP大小/SHA、退出和耗时。\n'''
(B/'README.zh.md').write_text(readme,encoding='utf-8')
print(json.dumps({'report_path':str(B/'REVIEW.zh.md'),'report_characters':len(report),'report_utf8_bytes':len(report.encode()),'report_lines':len(report.splitlines()),'new_acceptance_rules':0,'formal_labels_read':0,'files_written':['REVIEW.zh.md','FAILURES_AND_REVISIONS.zh.md','README.zh.md','/mnt/data/bge_continual_result_independent_review.zh.md']},ensure_ascii=False,indent=2))
