# vendorlink_beta_quarter：训练前定向独立审查报告

审查日期：2026-11-13（UTC+8）。

**模型身份说明：**本会话可自述的身份是 **GPT-6 Astra Pro**。我不能独立验证用户界面选择器及后台路由，不将模型自述写成后台身份验证结果。

## 一、审查结论

**有条件接受本次科学合同、β实际接线、联合响应判据和保存对照复用方案；当前代码不建议原样部署一次性等待。先闭合两项具体的进程生命周期复现缺陷，再由主执行者完成正式门绑定与独立处置，即可按现有授权部署唯一等待。**

本轮判断分为四层：

|层次|结论|
|---|---|
|科研有效性阻断 `SCIENTIFIC_BLOCKER`|未发现需要改变β、训练目标、评价数学或联合判据的具体阻断。|
|部署前复现缺陷 `REPRODUCIBILITY_DEFECT`|发现 **BQ-R01、BQ-R02**，均在新增身份读取/等待生命周期中，需最小修复及对应CPU补证。|
|部署前既定收尾|正式`gate.json`尚未形成；须绑定修复后源码、本轮主审处置、核验、等待依赖与正确新账。准备输入不能只翻转两个布尔值后使用。|
|后续执行|上述事项闭合后，可部署一次性等待；只有中文整批技术完成、监管收尾、相关进程释放与资源准入均满足，才自动训练。无需为本报告重新索取已给出的科学执行授权。|

两项缺陷都没有证明β数学错误、科学门被放宽或已有结果受污染。BQ-R01可能把正常进程释放误判为不可自动重试的控制器失败；BQ-R02会把已经没有生产者的缺失收尾留在等待状态。**它们需要修复，但不需要追加候选、seed、epoch、正式数据、新标签或三路GPU资格。**

本报告完成的是包内静态审查：解包、大小/SHA核验、JSON与文本检查、已保存数字的简单算术和源码追踪。没有访问服务器、正式数据或凭据，没有导入执行项目实现、运行模型/测试、重算预测指标或统计重采样，也没有部署等待或启动训练。

## 二、附件身份与引用约定

### 2.1 实际逐份核验

|项目|核验结果|
|---|---|
|收到的附件名|`input(3).zip`|
|实际大小|**204,057字节**，与请求一致|
|实际SHA256|`07f68544bb6ad3b85cf5fff72a127fb104ac3eedf348eeffac46ea8f343d5c53`|
|成员数|**49：48份载荷＋1份manifest**|
|ZIP完整性|CRC检查通过；无重复成员、缺项或未列载荷|
|逐载荷核验|**48/48的实际大小与SHA256均匹配清单**|
|载荷总大小|485,554字节，不含manifest|
|manifest实际SHA256|`9e3bb4cf3517c885aea28d007cecc24234a1f3e7a6773e15e487a1e7ef6fd5b6`|

上述是对收到字节的实际核验，不是转述清单自称通过。

### 2.2 包内路径缩写

|缩写|完整包内根路径|
|---|---|
|`B/`|`reports/documentation/20261113/vendorlink_beta_quarter/`|
|`S/`|`B/workspace/scripts/`|
|`Q/`|`B/qualification/`|
|`N/`|`reports/documentation/20261113/vendorlink_next_review/`|
|`D/`|`reports/documentation/20261112/vendorlink_deterministic/`|
|`R/`|`D/result/`|
|`P/`|`reports/documentation/20261110/vendorlink_redesign/fit_reference/`|
|`M/`|`P/mechanism/`|

下文引用这些路径、函数名及必要的单行定位，均可在附件中复核。引用旧审查原文表示承接其已闭合范围，不表示本轮重新读取了旧审查提及的全部私有载荷。

## 三、当前授权、研究问题与必须保留的负结果

用户本次已明确选择唯一有限探针：β＝**0.191859540449699**、seed51/52/53、完整AlphaBay→Dream→Agora、每域40轮、最多3-worker、独立累计42小时；必要审查/测试闭合后，等待中文H独立消融整批结束且资源满足，再自动执行。此前“优先结案”是方向建议；历史材料中的“尚未授权”是当时时点。两者都不能覆盖本次明确决定。〔B/scope.zh.md §1、§4；B/review/request.zh.md；N/primary_disposition.zh.md §2〕

本次回答的是：在保持完整H文本、多记录结构和其他训练条件的前提下，降低出生评分保持系数，能否恢复Dream学习、改善两个旧域最终能力并保护Agora。它不是寻找最优β，也不能改变已经完成的同协议SEQ轨迹。〔B/scope.zh.md §1—3〕

需要承接的主要背景如下，能力为百分数，F为百分点：

|方法|旧域初学Q|旧域最终O|遗忘F|Dream净学习GB|Agora最终R33|
|---|---:|---:|---:|---:|---:|
|H_SEQ|87.4806|86.3475|1.1331|4.1047|90.7556|
|H_ER|87.7106|86.6435|1.0672|4.5648|91.3448|
|旧随机Full|87.1445|85.6785|1.4660|3.4324|89.2698|
|确定性Full，即本次ref|86.7375|86.1603|0.5773|2.6185|89.4908|

这些是包内既有结果，不是本次β探针的新结果。确定性Full对V_SET有外部优势，但对SEQ/ER的旧末态仍较弱，Dream获得不足，Agora终态也较弱。因此较小F不能单独解释为有益抗遗忘；原明显遗忘、系统保持和组件贡献失败均保留。〔R/report.zh.md §2—5；R/tables.zh.md；R/comparison_results.json〕

本轮另以简单算术核对了`comparison_results.json`中七个方法/读取器条目的189格主宏F1及21条记录的Q/O/F/GB/GC关系，未见矛盾。没有重新执行项目判定函数或重算全部13指标。原P的5pp学习门失败不被后续4pp补充追认；SEQ平均F＝1.1331pp也不会因新Full改变。〔P/scope.zh.md §3；M/scope.zh.md §2；R/review/primary_disposition.zh.md〕

任务仍是已登记观察账号的互斥新文本识别：546/157/875账号，每账号12fit、4gallery、2dev、2confirm，主评价为Q2对同市场全部FIT12，各端点重编码。它不等于未见账号泛化、自然时间流或有真值的跨账号同控识别；三个开发seed也不是独立确认。〔B/scope.zh.md 第9行；N/background.zh.md §3；S/step35_vendorlink_account_evaluate.py::score〕

## 四、需要最小修复的两项发现

### BQ-R01：正常进程退出时，新增内核身份读取可能误触发整个等待或调度失败

**分类：`REPRODUCIBILITY_DEFECT`；部署前必修。**

**具体源码：**

- `S/step30_evolution_control.py::kernel_identity`，第21行开始，直接读取`/proc/{pid}/stat`。
- `S/step35_vendorlink_beta_wait.py::dependency_observation`，第29行开始，先创建`psutil.Process`，再调用上述函数；异常分支只捕获`psutil.NoSuchProcess`。
- `S/step30_evolution_control.py::resource_observation`，第54行开始，对具名承诺进程采取同样的异常处理。
- `S/step35_vendorlink_beta_wait.py::wait_then_run`，第110行开始，将漏出的异常记为失败并退出；一次性等待目录阻止自动重新开始。
- `S/step30_evolution_control.py::Runner.run_many`，第339行开始，控制器异常进入本批活动任务清理。

**触发路径：**中文监管进程，或已经登记的外部资源承诺进程，在`psutil.Process(pid)`构造成功以后、读取`/proc/{pid}/stat`以前正常退出并被其父进程回收。原始文件读取会抛出`FileNotFoundError`，并非`psutil.NoSuchProcess`。于是程序跳过本来已有的“进程已退出”处理。

在等待器中，即使中文整批及监管都已合法完成，也可能被写成不可自动重试的等待失败。在资源观察中，若正式门登记了恰好退出的外部承诺进程，该异常可能传入Runner，连带终止已经启动的健康任务。

**限定：**这是源码可确定的异常类型缺口和可能的竞态路径；本轮没有证据表明它已经在服务器发生。也不应泛称每个自有worker正常退出都会触发：直接子进程通常由同一父进程`poll`回收并从活动表移除，回收前还可能保留为zombie。中文监管和具名外部进程已足以构成此发现。

**最小修法：**仅在读取该PID的`stat`文件时，将明确的文件消失归一为调用方已经处理的`psutil.NoSuchProcess`。例如以下局部处理思路：

```python
try:
    stat_text = Path(f'/proc/{pid}/stat').read_text()
except FileNotFoundError as error:
    raise psutil.NoSuchProcess(pid) from error

fields = stat_text.rsplit(') ', 1)[1].split()
# 后续保留原start_ticks解析及boot_id读取。
```

不要宽泛吞掉所有`OSError`，也不要把权限错误、格式错误、`boot_id`不可读或活进程身份不符变成正常退出。该修法不放松身份核验，也不授权任何实验重试。

**最小补证：**用手写CPU元数据夹具模拟“Process对象已取得、随后该PID的stat消失”，检查中文完整闭合时可继续按完成回执判断；尚未闭合时仍拒绝；外部承诺已退出时记为`exited`而不产生控制器全局异常。无需BERT、GPU或正式数据。

### BQ-R02：中文主批次已完成但监管已退出、缺最终收尾回执时，等待器仍继续空等

**分类：`REPRODUCIBILITY_DEFECT`；与BQ-R01一并作部署前最小闭合。**

**具体源码：**`S/step35_vendorlink_beta_wait.py::dependency_observation`已计算`live`。在中文主`completion.json`不存在时，第38行开始会正确判断监管是否仍活着；但第51行的“监管`state/completion.json`不存在”分支直接返回`WAITING_CHINESE_SUPERVISOR_CLOSURE`，没有再次检查`live`。

这不是没有实际生产者依据的假设。包内`B/review/dependency_sources/step28_h_ablation_pause.py::takeover`第297行先等待`controller.execute()`返回，第299行才写监管完成回执；`step28_h_ablation_control.py::execute`第202行先写主批次完成回执。因此存在“主回执已有、监管回执尚未写”的明确窗口。

**触发路径：**监管在这个窗口被SIGKILL或崩溃，来不及留下failure文件。之后监管已经退出、成为zombie或PID身份改变，等待器仍反复返回可等待状态，累积活动检查费用，直到辅助预算门触发。

**影响限定：**它不会在缺少监管完成回执时提前开训，也不会污染训练结果；问题是把应明确终止的依赖状态误保留为等待状态，与合同要求的失败/身份变化留证终止不符。〔B/scope.zh.md 第36行〕

**最小修法：**监管完成回执缺失时，只有`live=True`才继续等；`live=False`则抛出明确的“中文监管未完整收尾即退出或身份改变”错误，由现有等待失败记录处理，并保持不自动重试。示意：

```python
if not (state / 'completion.json').exists():
    if not live:
        raise RuntimeError('Chinese supervisor exited before final closure')
    return dict(ready=False, status='WAITING_CHINESE_SUPERVISOR_CLOSURE')
```

**最小补证：**使用相同的合法主批次完成夹具，分别覆盖“监管仍活且缺收尾→等待”“监管已退出且缺收尾→明确失败”“两份合法完成回执齐备且进程释放→可以继续”。这可与BQ-R01的定向CPU批次一起完成；不需要建立恢复平台、补造中文回执或修改中文作业。

## 五、β实际接线与完整H保留情况

### 5.1 配置修改进入了真实训练路径

`beta_tasks()`从`deterministic_tasks()`派生，只改变任务ID和`settings.beta`；仍是三个固定seed的`H_full`、ABC、40轮和确定性出生分支。`execute()`检查真实模型的`AccountSettings`与任务设置相等，然后在B/C每次更新抽一个历史群并调用`update_account()`。〔S/step35_vendorlink_account_policy.py::beta_tasks、deterministic_tasks；S/step35_vendorlink_account_run.py::execute〕

实际目标可概括为：

\[
L=L_{\mathrm{current},H}+\gamma L_{\mathrm{CE}}
+\alpha L_{\mathrm{history},H}
+\beta\,\mathrm{MSE}\!\left(H(K_{\mathrm{student}}),H(K_{\mathrm{birth}})\right),
\]

其中γ＝1，α＝0.07049361507739538，β＝0.191859540449699。确定性Full的随机历史监督分支按`H_ER`目标执行，避免原随机出生MSE重复进入；独立`backward_birth()`再加入一次`beta*mse`，不再乘α。教师detach，学生经`proxy→pair_backward→z.backward`传到pair头和encoder。〔S/step35_vendorlink_account_methods.py::backward_h、backward_birth；S/step29_vendorlink_methods.py::h_objective〕

各分支累积后共同裁剪；主optimizer一次更新，辅助optimizer按γ规则另更新一次，二者参数不重叠。不能把“一次主更新”误写成全系统只有一个optimizer对象。〔S/step35_vendorlink_account_methods.py::update_account〕

### 5.2 其余结构、随机流和教师规则保留

标题/描述双通道、多记录pair MLP及H聚合、自然/重组监督、RMS辅助CE、分词与学习率日程继续沿原路径。当前与历史监督仍用train随机分支；确定性上下文只作用于出生分支，覆盖其前向、分块反传及检查点反算，并恢复随机状态和dropout属性。〔S/step35_vendorlink_account_methods.py::deterministic_student；S/step29_vendorlink_methods.py::Model、h_pool；S/step35_vendorlink_account_run.py::make_model〕

Memory仍为六群Algorithm R、1MiB、每次一个历史群；只有前两阶段结束后入库，新存活群才产生出生教师，已有教师不刷新。〔S/step29_vendorlink_methods.py::Memory.retain、draw；S/step35_vendorlink_account_run.py::execute〕

当前分组公式与冻结账号数对应：A/B/C每轮204/60/327个当前群，40轮为8160/2400/13080次更新；每条23640次，后两域历史消费15480次，三条合计70920/46440。验收函数核实际更新、曝光、历史消费及确定性分支计数，不只检查完成标签。〔S/step29_vendorlink_data.py::group_chunks；S/step35_vendorlink_account_run.py::groups_for；S/step35_vendorlink_account_control.py::validate_training〕

### 5.3 “四分之一”的解释没有越界

CPU03保存证据为固定状态、同一教师下出生损失比0.25，梯度相对于四分之一参照的最大绝对差0。它支持该分支的系数接线，不推出总梯度、共同裁剪后梯度、AdamW净更新或全程效果都缩为四分之一。〔Q/verify.py::test_beta_actual_gradient_and_execute；Q/cpu03/receipt.json〕

β改变B阶段参数后，也可能改变B结束时新出生的教师；所以完整ABC比较识别的是这一较弱β策略的总效应。合同已经明确该边界，没有将所有阶段教师张量冒称为固定相同。〔B/scope.zh.md 第13行；S/step29_vendorlink_methods.py::Memory.retain〕

## 六、联合响应门与原强门的独立性

### 6.1 新响应门逐项对应合同

主指标为宏F1，ref指11月12日确定性Full，所有差值使用未舍入值。`beta_comparison()`的实现对应如下：

|部分|固定要求|代码判断|
|---|---|---|
|Dream学习|平均GB≥4pp，每seed非负|`gain('GB')`|
|Dream初学保护|R22对SEQ、ER分别平均≥−1pp、每seed≥−3pp|分别调用`lower`|
|Dream响应方向|至少2/3 seed的R22严格高于ref|使用`>0`计数|
|旧域总体|平均ΔO>0，至少2/3 seed正，最差≥−1pp|均值、正向数及最小值同时检查|
|两个旧域|平均R31、平均R32分别不低于ref|分别检查，不能互相补偿|
|Agora终态|平均R33不低于ref，至少2/3 seed非负|使用`>=0`计数|
|Agora学习|平均GC≥4pp，每seed非负|`gain('GC')`|
|Agora相对保护|R33对SEQ、ER、V_CLS、V_SET分别平均≥−2pp、每seed≥−5pp|四个对照均检查|
|原质量条件|原绝对能力、初学、新域及外部旧域保护|接回`initial_protection/new_protection/external`|

原`initial_protection`确实包含R11/R22/R33均值≥0.70、每seed≥0.60，因而绝对能力条件没有遗漏。`joint_response`不要求F小于ref；合成测试也覆盖了“新F更高但响应通过”。这与恢复初学Q时F可能上升相容。〔S/step35_vendorlink_account_policy.py::beta_comparison、decide；Q/verify.py::test_original_results_and_response_boundaries〕

### 6.2 原失败门另报，没有被响应门覆盖

`beta_comparison()`先调用原比较逻辑，将完整结果保留在`original_gates`。原`decide()`仍分别计算：SEQ明显遗忘均值3.5pp及种子条件；Full−SEQ的O/F各3pp及稳定性；Full−ER的O/F各2pp及相应保护；绝对能力、初学、新域和外部比较。`old_score_increment_conditions`继续独立输出。〔S/step35_vendorlink_account_policy.py::deterministic_comparison、decide〕

因此，**新响应通过可以与原强门失败同时成立**。只恢复到ER附近时，可以报告减轻额外约束代价，不能宣布出生保持具有额外价值。无论本次响应好坏，返回值均关闭confirm、application和automatic_followup。〔B/scope.zh.md §3；S/step35_vendorlink_account_policy.py::beta_comparison；S/step35_vendorlink_account_control.py::pipeline〕

### 6.3 R11/R12与完整指标的解释边界

代码单列同seed的R11/R12差异，任何非零差异都会置`interpretation_requires_identity_review=True`。这要求结果解释前核对来源与未改路径，不能归因于尚未发生的β干预。该自动flag检查的是主宏F1；不能把它说成自动完成了全部13指标的预干预一致性审查。完整指标已由评分分析器保存，结果审查可据此继续核对。〔S/step35_vendorlink_account_policy.py::beta_comparison；S/step35_vendorlink_account_evaluate.py::analyze〕

必须同时报告Q、O、带符号F、两个旧域各自遗忘、GB/GC和全部13指标。因F＝Q−O，若Δ表示新值减对照，则“遗忘减少＝ΔO−ΔQ”；这是恒等式，不是两项独立因果贡献。每query只有一个正确账号时，MAP＝MRR，Top1＝micro-F1＝Recall@1＝NDCG@1，不能当成多份独立胜利。〔S/step35_vendorlink_account_policy.py::summarize；S/step35_vendorlink_account_evaluate.py::metrics〕

## 七、保存对照身份、完整评分与一次开放

### 7.1 对照复用具有实际检查入口

`deterministic_baseline()`逐个绑定原P/M的gate、batch、裁决/完成、budget和results，检查固定任务、seed、数据manifest、预训练引用及原裁决。`beta_reference()`再检查最新确定性Full的协议、相同P/M来源、数据/预训练、三任务及完整比较结果。`permit()`实际调用这些函数；CPU核验也真实调用了保存对照读取，而非只用合成结果替代。〔S/step35_vendorlink_account_control.py::permit、deterministic_baseline、beta_reference；Q/verify.py第88行开始〕

相关旧权重已按此前许可删除，但已保存指标仍可在同协议下复用。当前新训练从共同预训练冷启动，新评分只加载本批新端点，没有通过已删旧模型续训或重新评分的路径。〔N/background.zh.md §6；S/step35_vendorlink_account_run.py::make_model、restore〕

**网页核验的边界：**本包包含已开放比较数字和旧审查原文；原P/M及确定性运行的部分原gate、batch、原评分清单、数据与预训练载荷仅有路径/大小/SHA引用，没有全部随包提供。我确认的是绑定逻辑、包内数字和CPU03记载的Linux来源检查；没有独立读取服务器上每份被引用载荷，也没有重新核验已删权重或原预测矩阵。现有材料未出现需要索取这些载荷才能解决的具体矛盾，不将其变成补件或全历史重验要求。

### 7.2 全批完整后评分、评分完整后一次分析

实际顺序为：冻结三任务→全部训练成功→写`training_complete.json`→串行完成三个评分任务→写`scoring_complete.json`→单CPU分析一次。`run_many()`只要仍有失败或缺项，就抛错关闭下游。〔S/step35_vendorlink_account_control.py::campaign.run_phase；S/step30_evolution_control.py::Runner.run_many〕

训练验收检查任务、identity、stage0/1/2/3端点及必要计数。评分前核整批训练清单；分析前核整批评分清单。评价器要求九格R01/R11/R12/R21/R22/R23/R31/R32/R33、全部FIT12矩阵与对应角色来源完整，拒绝重复分析目录，并写`opening='whole_batch_once'`。本批正常完成后应有27格、351个新指标值。〔S/step35_vendorlink_account_control.py::worker、validate_training；S/step35_vendorlink_account_evaluate.py::validate_scores、analyze〕

这里“盲评分”是评分阶段不先计算和开放科学指标。评分器仍须检查查询与档案账号映射和文本互斥关系，不能说程序对账号排列完全不知情。训练worker只读当前fit和存活Memory；评分role固定为dev，本β路径不进入confirm或新增应用输出。〔S/step35_vendorlink_account_control.py::worker；S/step35_vendorlink_account_evaluate.py::score〕

## 八、三槽资源、故障处置与42小时预算

### 8.1 CPU分配和GPU/RAM准入

`Runner.command()`已改为从父进程允许核心中取最后`2*workers`个，再按槽分成两个核；三槽要求至少六个允许核心。CPU03真实子进程使用了`[0,1]`、`[2,3]`、`[4,5]`。正式等待启动命令必须保留至少六核亲和性，不能复用CPU01的两核包装。〔S/step30_evolution_control.py::Runner.command；Q/cpu03/receipt.json〕

资源逻辑对本批未体现GPU承诺扣除`max(0, reserve−NVML已观察占用)`，对已登记外部GPU承诺也做身份核对与扣账；本批RAM按每槽20GiB减已观察RSS的待体现额扣除，实际RSS上限16GiB。新身份采用start_ticks/boot_id，create_time只作旧格式兼容。〔S/step30_evolution_control.py::resource_observation；S/step35_vendorlink_account_policy.py::BETA_LIMITS〕

**5GiB是逐槽准入预留，不是整PID显存硬帽，也没有allocator硬配额。**最多三槽不保证始终能同时派出三条，资源不足时可以逐槽等待。旧单路采样峰值和CPU03都不能保证三路长期峰值或13小时内一定完成。〔B/scope.zh.md §4；S/step35_vendorlink_account_control.py::configure_compute；D/qualification/gpu01/receipt.json〕

磁盘准入保留12GiB本批输出帽、启动/派发空闲64GiB、运行空闲32GiB和每个活跃保存的2GiB预留。运行阶段持续检查RSS、磁盘与生命周期；没有发现把磁盘占用误当显存占用的接线。〔S/step30_evolution_control.py::resource_observation、Runner.run_many；S/step35_vendorlink_account_io.py::Check〕

### 8.2 条件性现场前提：外部RAM承诺

现资源函数对RAM待体现额只遍历本批`owned`；`other_gpu_commitments`进入的是GPU扣账，没有对应外部RAM预留字段。故不能笼统声称“所有已核实外部GPU/RAM未来承诺均已扣除”。〔S/step30_evolution_control.py::resource_observation，第31行与第53行〕

当前准备门的外部承诺列表为空，本包没有证明现场存在某个未体现的外部RAM承诺。**这不列为无条件第三项阻断。**正式部署时应确认实际有无此类承诺；若存在，就把已核实的RAM预留纳入`max(0, reserved_ram−RSS)`扣账，或等该承诺释放后准入。若不存在，记录该现场前提即可。无需建立全局锁或通用资源平台。

### 8.3 预算数值及累计语义

|分桶/操作|固定上限或预留|判断|
|---|---:|---|
|训练桶|144000秒＝40小时|独立新账|
|每条完整训练|46800秒＝13小时|三条合计140400秒＝39小时|
|GPU评分桶|3600秒＝1小时|三个评分各900秒，最坏合计2700秒|
|辅助桶|3600秒＝1小时|含CPU核验、部署、活动监控、分析和收尾|
|分析|最多600秒|启动时再按实际辅助余量收紧|
|收尾|30秒|Ledger初始化另计，不应遗漏|

Runner按已发生费用、活动任务完整预约和新任务完整预约准入；并行任务生命周期分别计费，逐次向上取整，失败不退费，不跨桶借额。分析余量不足则拒绝，不保证凑足42小时就一定完成。〔S/step35_vendorlink_account_control.py::make_jobs、campaign；S/step30_evolution_control.py::Runner.run_many；S/step30_evolution_io.py::Ledger〕

三次CPU已发生费用为21秒；加实现报告登记的保守部署/核对60秒，正式门辅助起账应至少81秒，且再加此后新发生的必要工作。`cpu_test_seconds`至少承接21秒；Ledger另加30秒收尾。本轮新GPU核验费用为0，复用旧GPU资格不应把旧10秒再次当成本批新增消耗。〔B/implementation.zh.md §2；Q/cpu01/receipt.json；Q/cpu02/receipt.json；Q/cpu03/receipt.json〕

旧三条训练25小时38分49秒可作量级参照，不能据此保证三路工期等于旧工期的三分之一；共享同GPU的争用仍受13小时/条和分桶硬帽约束。中文结束日期与本批开训时点在本包中未知。〔B/scope.zh.md 第42行〕

### 8.4 失败不重试的主路径成立

真实CUDA OOM会留下类别，失败槽退役，健康槽可继续当前阶段尚未派发的独立任务。非OOM失败停止新派发并排空已运行健康任务；共享来源/数据有效性或磁盘故障停止相关本批。启动尝试被消费后不回到待派列表，不自动缩token/batch或替换配置。缺必要训练不评分，缺必要评分不开放整批结果。〔S/step30_evolution_control.py::Runner.run_many；S/step35_vendorlink_account_control.py::main、campaign〕

这些是静态接线结论，并不等于CPU03新增实测过每一种OOM、共享故障和清理分支；未变资格继续按既有范围承接。

## 九、中文完成双条件、唯一等待与首更自删

除第四节两项缺陷外，正常路径的关键条件有实际接线：

|要求|实际实现及判断|
|---|---|
|中文整批完成|检查固定完成状态、63新增/9复用、原sources及接管request/updated_sources；与包内实际生产者字段一致。|
|监管最终收尾|额外要求`CONTROLLED_CAMPAIGN_COMPLETE`，不只看训练暂空或PAUSED。|
|无失败且释放进程|检查主目录/监管目录failure；扫描同owner、含中文workspace或精确request路径的相关进程。|
|不读取中文科学效果|判断使用运行元数据、来源身份及进程状态；科学负结果本身不阻塞。|
|中文与资源联合条件|先核中文，再检查本批资源；进入campaign时再次核中文，Runner逐槽再次做资源准入。|
|唯一等待|拒绝已有campaign；`wait.mkdir()`排他占用，另持文件锁；直接campaign CLI拒绝β协议绕过入口。|
|等待计费|每120秒观察，活动片段累计后取整，sleep不计费，费用带入同一辅助桶；失败留证且不自动重来。|
|首更自删|读取真实更新日志首行，要求`optimizer_updates==1`且`total_updates>=1`，校验绑定后只删本批`workspace/scripts/once_wait.py`，保存自删回执并继续控制器。|

证据：`S/step35_vendorlink_beta_wait.py::dependency_observation、wait_then_run`；`S/step35_vendorlink_account_control.py::campaign、main`；`S/step30_evolution_control.py::Runner.retire_waiter`；`B/review/dependency_sources/step28_h_ablation_control.py::execute`及`step28_h_ablation_pause.py::takeover`。

首更日志是在`update_account()`实际执行优化器步骤并返回后才写入，不是仅“模型加载成功”便删除入口。正式门应将一次性wrapper放在独立`waiter`绑定中，长期生产sources绑定14份运行源码；不能把计划删除的wrapper作为每个后续worker必须重新核验存在的常驻源码。当前源码已经使用这一分离结构。〔S/step35_vendorlink_account_methods.py::update_account；S/step35_vendorlink_account_run.py::execute；S/step35_vendorlink_account_control.py::permit〕

本包`preflight.json`是13:27:14的历史快照，当时中文为8完成、3活动、52待派，不是本轮网页取得的实时状态，也不证明现在已经结束。部署后应由绑定的等待逻辑重新判断，不能用快照代替放行条件。〔B/preflight.json〕

## 十、CPU01/02/03与旧GPU证据的准确范围

### 10.1 原回执和版本关系

|核验|实际结果|完整生命周期|向上计费|
|---|---|---:|---:|
|CPU01|5项中4通过、1 ERROR，exit1|10.693278秒|11秒|
|CPU02|5项中4通过、1 ERROR，exit1|4.426962秒|5秒|
|CPU03|5 PASS，0 failure/0 error，exit0|4.727358秒|5秒|
|合计|前两次失败保留|—|**21秒**|

三份receipt与各自result/stderr相容。CPU01/02回执内的source_manifest与`deployment_before_cpu.json`一致；CPU03与`deployment_cpu03.json`一致。CPU03的17项绑定全部匹配当前包字节；前两次只有`qualification/verify.py`不同，生产源码、入口和scope没有随夹具修正而变化。〔Q/cpu01—cpu03；B/deployment_before_cpu.json；B/deployment_cpu03.json〕

CPU01包装命令确实只给0、1两核，与三槽六核门矛盾。CPU02已经改为0—5；“15秒夹具与生产15秒关闭余量冲突”的解释与实现记录、当前修订说明和Runner相容。但两次stderr只打印最终`Incomplete phase`，没有随包提供当次dispatch细因及旧verify全文，不能声称本外审独立重放了CPU02当时的具体失败。无需因此重跑旧失败，保留原件、恢复身份和费用即可。〔B/implementation.zh.md §2；Q/cpu01/stderr.txt；Q/cpu02/stderr.txt；Q/verify.py第180行〕

### 10.2 CPU03的真实覆盖与限制

|测试内容|现有证据实际支持|不能扩大成的结论|
|---|---|---|
|β损失及梯度|小BERT真实出生分支计算；损失比0.25，梯度四分之一参照最大差0，encoder梯度非零|正式效果正向、总AdamW更新比例四分之一|
|微型训练接线|真实`execute()`完成ABC9更新、6历史消费和4端点，缺计数拒绝|三条正式40轮已经执行或必定完整|
|联合门与原结果|实际保存对照读取；合成通过、部分失败、缺seed拒绝、F升高仍可响应通过|逐一穷举全部边界或真实OOM到评分阻断的完整联调|
|中文完成判断|手写完成/监管/failure文件下调用真实函数|真实生产者运行、真实进程释放与退出竞态均已实测；放行用例把`process_iter`置空|
|联合等待与唯一入口|先中文后资源、两次sleep调用、campaign只调用一次、重复目录拒绝|真实`once_wait→permit→campaign`完整联调；dependency/resource/campaign均被mock|
|三槽分核|三个真实CPU子进程、六个互不重叠核心、39小时任务配置|三路GPU资格；主要NVML与准入被mock|
|GPU待体现承诺|独立夹具中9GiB余量减5GiB待体现预留后不足下一槽5GiB，正确拒绝|外部RAM、三路长期占用、所有进程退出竞态已覆盖|

以上依据为`Q/verify.py`及`Q/cpu03/receipt.json`。现有测试没有覆盖BQ-R01/BQ-R02，这就是本次要求定向补证的原因；其他未穷举路径不自动变成必修测试清单。

### 10.3 旧GPU资格可以复用，但要保留身份边界

旧确定性GPU回执记录：单路真实BERT、44记录、双通道256token、875辅助类、SDPA/检查点、一次真实更新、出生分支encoder/pair非零梯度、模式恢复。当前methods/run/evaluate等相关未变源码SHA与准备门保留的旧来源一致，因此按未变范围复用有依据。〔D/qualification/gpu01/receipt.json；D/implementation.zh.md；B/input_gate.json.sources〕

准确表述应是：**本轮新增的是CPU增量证据，GPU资格复用旧单路证据；本轮没有做三路GPU资格，也没有证明未来三路峰值或40轮稳定性。**没有具体改变科学计算图的剩余缺口，不要求补一轮三路GPU资格。

## 十一、正式门定稿与最小闭合后的部署意见

### 11.1 准备输入不是正式可执行门

`input_gate.json`明确`approved=false/review_closed=false`，仍保留旧确定性运行的sources路径、scope、CPU、授权/免审措辞、`source_git`和费用`gpu=10/auxiliary=35/cpu_test_seconds=5`，且尚无本次wrapper的正式绑定。请求和实现报告已说明它只是准备输入。因此这不构成“用户未授权”的发现，也不能只改两个布尔值后投入使用；`permit()`的当前源码路径集合校验本身会拒绝这种用法。〔B/input_gate.json；B/review/request.zh.md；S/step35_vendorlink_account_control.py::permit〕

### 11.2 可执行的最小闭合范围

1. **局部修正BQ-R01、BQ-R02。**只改变新增身份读取和监管终态识别，不改变β、科学判据、任务、token/batch或故障重试政策。
2. **完成对应的有界CPU补证并留费。**保留CPU01/02/03原件和21秒，新增费用记本批辅助及CPU子账；不重跑整套GPU资格、旧训练或全部科学测试。
3. **生成真正的本批正式门。**绑定修复后的14份生产源码、一次性wrapper、当前合同、本轮独立主审处置、CPU03及定向补证、旧GPU复用来源、旧P/M/确定性保存对照和中文依赖身份；更新当前授权依据与实际新账。
4. **落实现场前提。**父进程至少六核；确认未体现外部承诺的实际状态，尤其不能把仅有GPU扣账说成外部RAM也已处理；当前门与部署字节一致，等待/campaign未被重复创建。
5. **只部署一个等待进程。**正常中文完整闭合且资源满足后自动训练；依赖失败、身份变化或本批技术失败则按合同留证停止，不自行重试、续跑或开新队列。

这些是当前已批准任务内的实施闭合，**不要求用户再次选择科学方案或再次授权同一探针**。主执行者仍须独立核实修复和实际部署，网页报告不代替现场来源、进程和资源验证。

### 11.3 不列为本轮必修的事项

不要求追加β网格、β＝0重复训练、新seed、改变epoch/顺序、重建已删权重、打开confirm、新标签或全历史复验；不要求全局资源锁、通用恢复平台、失败自动恢复机制；不因CPU模拟覆盖有限就默认加三路GPU资格。完整13指标、R11/R12来源调查和原门分项，是结果阶段按既定合同解释证据的要求，不是授权继续搜索的入口。

**最终意见：本次有限β探针的科研合同和科学接线可接受；两项具体复现缺陷闭合、正式门与费用绑定完成并经主执行者独立处置后，可以部署唯一等待。当前原样代码尚不建议部署。**

## 附录：实际阅读范围与未核验内容

### A. 48份载荷的核验与阅读分组

|组别|具体材料|载荷数|
|---|---|---:|
|本轮合同、请求与准备元数据|B/scope、implementation、review/request；input_gate、preflight、两份deployment|7|
|本轮CPU核验|Q/verify.py；cpu01/02/03各receipt、result、stderr|10|
|本轮生产部署|六份step35_account源码；step29_data/methods/run；step28_formal_queue；step30_policy/io/control；beta_wait；once_wait|15|
|中文完成生产者|review/dependency_sources下control与pause两份源码|2|
|方向背景及闭合审查|N/background、primary_disposition、external_report|3|
|原确定性合同和资格|D/scope、implementation；原CPU/GPU receipt|4|
|最新保存结果与结果审查|R/report、tables、comparison_results；review/primary_disposition、external_report|5|
|原P/M合同|P/scope、M/scope|2|
|合计|上述全部载荷逐份大小/SHA匹配；manifest另计|**48**|

完整阅读了本包14份Markdown，包括两轮完整外审原文及主审处置；读取全部本轮CPU原回执、结果和stderr以及相关JSON绑定。源码审查覆盖本轮修改及其实际训练、评价、来源校验、预算、等待、资源、故障和中文完成生产者调用链。未变共享源码的无关历史入口按原资格复用，不将本次称为全仓或全部历史实现重新审查。

### B. 明确没有重新核验的内容

未读取正式商品正文、账号名单/映射、标签载荷、预测矩阵、Memory、模型权重、confirm及服务器实时状态；未读取未随包提供的旧P/M全部原始载荷、旧测试全文或原逐步训练日志。没有将本地包内核验冒称为项目Linux原生执行，也没有用旧审查原文替代本轮新增等待缺陷的独立判断。

本报告所有“通过”均限定到明示证据范围；BQ-R01/BQ-R02为静态发现和待执行者闭合项，不是声称已代为修复或测试。
