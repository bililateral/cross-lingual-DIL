# 确定性出生评分 Full：定向科研结果审查

审查日期：2026-11-13（UTC+8）。可自述模型身份：GPT-6 Astra Pro；本报告不声称已独立验证界面选择器或后台路由身份。

**审查结论：接受本轮为完整、来源对应且数值一致的固定三 seed 开发比较。未发现使结果无效的阻断，也未发现必须修复后才能接受结果的可复核缺陷；本轮无必要修复。**

**科学结论仍为未达到目标。** 确定性候选对旧 Full 有有限的终态改善，但未建立相对同配置 ER/SEQ 的保持优势，Dream 初学保护退步，Agora 终态三个 seed 均低于 ER/SEQ。保存的 `STOP_NO_H_ADVANTAGE`、`old_score_increment=false` 与确认未开放均正确。运行完整、结果有效、科学门失败，可以同时成立。

下文所有判断以本附件为依据；操作限于解压、字节/SHA核验、静态阅读和已保存聚合数字的简单算术，没有执行项目实现、载模、重算标签或预测，也没有重采样。

## 一、附件身份、授权时点与证据范围

### 1. 实际核包结果

|项目|独立核验结果|
|---|---|
|页面文件名|`input(1).zip`|
|实际大小|206,417 字节，与请求一致|
|实际 SHA256|`bb0ef9887e3d9d762e56ca0a6413236569708d7de934de627776e8969cc24f55`|
|ZIP成员|56：55份材料＋`MANIFEST.json`|
|载荷总大小|784,575 字节，与清单一致|
|逐文件核验|55份材料的大小、SHA256全部匹配|
|成员集合与归档完整性|无重复成员、无未列载荷、无缺项；ZIP CRC检查通过|

完整阅读了指定的 `reports/documentation/20261112/vendorlink_deterministic/result/review/request.zh.md`，并按其范围追核。包内“待许可”是准备时记录，已被用户本轮对具体包的明确批准覆盖，不构成阻断。

为便于原样回收，以下引用均指**包内路径**，使用三个明确缩写：

- **D/**＝`reports/documentation/20261112/vendorlink_deterministic/`。
- **R/**＝`D/result/`。
- **B/**＝`reports/documentation/20261110/vendorlink_redesign/fit_reference/`。

### 2. 实现关闭的依据应准确表述

`D/gate.json` 的 `approved=true`、`review_closed=true` 与 `review_basis=explicit_20261112_user_external_review_waiver_and_independent_primary_closure` 一致：本轮训练前是**用户明确免除方案/实现外审，完成必要核验及执行者主审后开训**。这不是“已经通过网页独立实现外审”。`D/implementation.zh.md` 对此写得清楚，本次没有理由重开未变资格。

### 3. 本次能独立核实到哪一层

新源码、新训练/评分完成回执、保存指标及汇总都在附件内，可以核对其实际字节和相互对应。

旧 P/M 的完整原始 `gate/batch/budget/results/metrics` 文件、正式 `updates.jsonl`、模型和预测矩阵没有随包提供。相关路径、大小、SHA及Linux核对记录在 `D/gate.json.prior_P/prior_M`、`R/summary.json.metric_sources`、`R/history_summary.json[*].source`、`R/check.json` 中保留。因此，本报告可以确认这些引用、保存旧数值和当前比较之间没有发现矛盾，**不能声称重新读取或验证了未附载荷的内容**。

这是明确的证据边界。现有保存材料足以判断本轮所提出的结果与解释，未出现必须打开原始载荷才能解决的具体矛盾，故不将其列为补件或重算要求。

## 二、三 seed 训练、评分与结果来源对应

### 1. 冻结来源与完整任务身份一致

附件13份生产源码的实际大小/SHA均匹配 `D/gate.json.sources` 和 `D/final_sources.json.sources`；两者源码集合一致，`source_git/git` 均为 `c902ada1`。本轮scope、implementation、原FIT12 scope及CPU/GPU回执的附件字节，也与gate中的相应绑定一致。

任务对应不只检查名称。我核对了以下位置中的任务、seed、配置和身份：

`R/batch.json.tasks`、`training_complete.json.tasks`、`scoring_complete.json.tasks`、`analysis_access.json.tasks`、`check.json.tasks[*].task`，以及全部 `source_receipts/training_complete_s*.json` 和 `scoring_complete_s*.json` 的 `task/identity/training_identity`。

三条均为固定 `H_CE_RMS/H_full`，`epochs=40`、`stages=3`、AlphaBay→Dream→Agora，`deterministic_history=true`；α、β、RMS、六群及1 MiB设定一致。

|seed|任务ID|当前更新|历史群消费|确定性历史消费|评分格|
|---|---|---:|---:|---:|---:|
|51|`M_H_full_det_s51`|23,640|15,480|15,480|9|
|52|`M_H_full_det_s52`|23,640|15,480|15,480|9|
|53|`M_H_full_det_s53`|23,640|15,480|15,480|9|
|合计|三条完整轨迹|70,920|46,440|46,440|27|

六份训练/评分完成回执的实际SHA与整批 `completions` 引用、`check.tasks[*].training_completion/scoring_completion` 一致。六份包装回执均为 `success=true`、`released=true`、`error=null`、`failure_kind=null`。没有发现缺seed、不同配置拼接或旧Full替代新Full的证据。

### 2. 评分范围与完整后开放相符

每条评分回执均为：

- `status=BLIND_SCORES_COMPLETE`、`role=dev`。
- `reference_role=fit`、`reference_records=12`。
- `cells=[R01,R11,R12,R21,R22,R23,R31,R32,R33]`。
- `files` 恰好九个 `H_full:Rxx` 键，`sources` 为三市场各自fit/dev的六份绑定。

`scripts/step35_vendorlink_account_control.py` 的 `campaign.run_phase()`、`worker()`，以及 `step35_vendorlink_account_evaluate.py` 的 `validate_scores()/analyze()`，对应“全部训练完整→串行评分→全部评分完整→一次分析”的实际路径。`R/analysis_access.json.opening=whole_batch_once` 与之相符。

`R/check.json.times` 保存的时间顺序为：2026-11-13 **04:37:38**训练完整，**04:48:42**评分完整，**04:48:43**结果及整批完成，均为UTC+8。`campaign_dispatch.json` 为COMPLETE，无失败、待派、退槽或预算停止；09:53的 `observation.json` 记录控制器不存在、匹配进程为空。

现场观察中的 `scientific_decisions.M.status=NOT_READ` 表示观察器没有解析科学裁决；它同时记录分析产物存在。不能据此认定科学结果未产生，实际裁决在 `R/completion.json.candidate_M`。

### 3. 端点核验的限定

三份训练完成回执均有stage0/1/2/3，共12份端点记录，其中9份为训练后端点、3份为初始化端点。其路径、大小、保存SHA与 `R/check.json.models` 一致，所记大小合计3,918,621,201字节。

但 `verification=metadata_only_saved_SHA_retained` 明确表示此次现场检查沿用保存SHA、核对元数据。本外审没有重新验证权重内容，也没有重新数原始训练日志行或验证预测数组。以上来源链支持本轮保存结果的对应，不应扩写为对全部Linux大载荷的重新验真。

## 三、主宏F1、全部13指标与统计一致性

### 1. 新旧Full的字段命名没有混用

|位置|新确定性候选|旧Full|
|---|---|---|
|`R/metrics.json`|`M_H_full_det_s51:H_full:Rxx`等任务键|不在该新指标文件中|
|`R/results.json`|`H_full`下三个seed|不在该新结果文件中|
|`R/comparison_results.json`|`candidate.H_full`|`baseline.H_full`|
|`R/summary.json`、`detail.json`|`H_full_det`|`H_full`|

因此，原始新结果仍叫 `H_full` 不表示它覆盖了旧Full；比较文件和统计文件已经明确区分。

### 2. 实际核对结果

新 `metrics.json` 有 **27×13＝351** 个指标值，逐项与 `detail.H_full_det.Rxx.指标.values` 对应。新 `results.H_full` 与 `comparison_results.candidate.H_full` 一致；各R格确实取对应指标的 `macro_F1`。

全部统计材料覆盖七个方法/读取器条目、九格、三个seed，即 **189格、2,457个逐seed指标值**。这些值均有限且在[0,1]，其 `mean/sd/min/max` 与保存的 `values` 一致；21条宏F1行的派生量及五组配对差、配对SD和正向seed数均一致。算术核对仅有通常的浮点末位差，未发现实质偏差。

这里对2,457值的结论是**附件内保存值及统计的完整一致性**；其中旧P/M原始指标源文件的逐字节核验，依赖已附Linux来源记录，不能冒称本外审重新读取了那两份未附文件。

### 3. 指标含义与依赖关系正确

当前13项为：

`macro_F1`、`Top1`、`micro_F1`、`MRR`、`MAP`、`Recall@1/3/5/10`、`NDCG@1/3/5/10`。

`step35_vendorlink_account_evaluate.py.metrics()` 按每账号类F1求平均得到macro-F1；`analyze()` 写入主结果的是 `m['macro_F1']`。主门没有误用MAP，也没有把macro-F1与整体正确率互换。

全部189格均符合：

- `MAP=MRR`。
- `Top1=micro_F1=Recall@1=NDCG@1`。

这是每个query只有一个正确账号时的指标关系，不能把这些重复读数当成独立证据。三seed样本SD描述本批训练随机性的离散，不是置信区间，也不能据此宣布总体显著优越或严格劣效。

`V_CLS_ads` 已纳入数值核对，但它是逐条文本读取，查询单位与Q2聚合账号读取不同；它不构成新增独立训练的外部方法。

## 四、宏F1主结果、配对差与稳定性

### 1. 三seed主量

以下能力单位为%，遗忘F为百分点（pp），均为均值±样本SD。依据为 `R/summary.json.summary[方法]` 及 `R/comparison_results.json`。

|方法/读取器|旧域初学Q|旧域最终O|遗忘F|Agora最终R33|
|---|---:|---:|---:|---:|
|H_SEQ|87.4806±0.4107|86.3475±0.2848|1.1331±0.5185|90.7556±0.3854|
|H_ER|87.7106±0.4861|86.6435±1.0187|1.0672±0.8282|91.3448±0.1684|
|旧H_full|87.1445±0.8042|85.6785±0.7137|1.4660±0.1873|89.2698±0.6446|
|确定性H_full|86.7375±0.2746|86.1603±0.6740|0.5773±0.8827|89.4908±0.2602|
|V_SET|83.3007±1.2251|80.0810±0.3369|3.2197±1.1495|87.4743±0.7458|
|V_CLS|84.8775±1.0865|40.0488±0.7589|44.8287±1.0571|86.3860±0.4563|

相对旧Full，确定性候选O提高0.4818pp，Agora提高0.2210pp。这是保存终态上的真实差值，不能把全部改善称为“小F制造的假象”。

相对ER/SEQ，它的O仍分别低0.4832/0.1872pp，Agora低1.8540/1.2648pp。因此，“对旧Full局部改善，但未建立同配置ER/SEQ优势”的主结论成立，且不只是差一点达到预定幅度门。

### 2. 相同seed配对

下表O、Agora为“新候选−对照”；遗忘减少为“对照F−新候选F”，正值有利。SD是**配对差本身**的样本SD。证据：`R/summary.json.pairs`、`R/completion.json.comparisons`。

|对照|O差（pp）|F减少（pp）|Agora差（pp）|正向seed数：O／F减少／Agora|
|---|---:|---:|---:|---|
|旧Full|+0.4818±0.7245|+0.8888±1.0683|+0.2210±0.3954|2/3／2/3／2/3|
|ER|−0.4832±1.5020|+0.4899±1.6968|−1.8540±0.4280|1/3／1/3／0/3|
|SEQ|−0.1872±0.9082|+0.5559±1.3719|−1.2648±0.5805|1/3／2/3／0/3|
|V_SET|+6.0793±0.5152|+2.6424±1.0410|+2.0165±0.5108|3/3／3/3／3/3|
|V_CLS|+46.1115±1.3031|+44.2515±1.1582|+3.1048±0.4758|3/3／3/3／3/3|

对两个VendorLink读取器的优势在本批三个seed中方向一致，应保留。它不能替代Full−ER的出生评分增量判断。

`step35_vendorlink_account_evaluate.py.classification()` 使用当前encoder搭配旧市场分类头；`encode()/rectangular()` 则用当前encoder重编码query与FIT12参考来计算V_SET。两者来自同一VendorLink训练，不能当作两个独立训练方法。V_CLS的巨大旧域下降也不能直接归结为已经证明的某个唯一机制。

### 3. seed53确实影响均值，稳定性没有被证明

证据：`R/summary.json.rows.H_full_det`、`pairs.H_full/H_ER.rows`。下表均为pp。

|seed|新Full F|对旧Full O差|对旧Full F减少|对ER O差|对ER F减少|
|---|---:|---:|---:|---:|---:|
|51|1.3310|+0.4857|−0.0451|−1.2620|−0.7843|
|52|0.7945|−0.2446|+0.6578|−1.4358|−0.1620|
|53|−0.3938|+1.2044|+2.0536|+1.2483|+2.4160|

这揭示了三种不同情况：

- seed51对旧Full终态更好，却没有更少遗忘。
- seed52对旧Full遗忘更少，却终态更差。
- seed53同时带来较大O改善和F减少，是相对ER平均F改善的主要来源；对ER的另外两个seed在O和F减少上均不利。

seed53的负F具体来自 `F_A=−0.7875pp`、`F_B=0`，即AlphaBay终态高于初学、Dream终态等于初学。它不能证明两域都获得正向迁移，更不能代表普遍不遗忘。

另外，新Full的F样本SD为0.8827pp，高于旧Full的0.1873pp。虽然其Agora等部分量的SD下降，也不能概括成整体更稳定。原报告已披露seed53和不利seed，没有选择性删除；以上是对稳定性边界的进一步明确。

## 五、Q/O/F恒等式与Dream初学解释

### 1. 公式正确，分解不构成因果归因

`step35_vendorlink_account_policy.py.summarize()` 定义：

`Q=(R11+R22)/2`，`O=(R31+R32)/2`，`F=Q−O`；

`N=(R22+R33)/2`，`Z=R33`，`GB=R22−R12`，`GC=R33−R23`。

`summary.rows` 另列 `F_A=R11−R31`、`F_B=R22−R32`，且 `F=(F_A+F_B)/2`。全部21行均一致。

若Δ表示“新候选−对照”，则：

**遗忘减少＝ΔO−ΔQ。**

|对照|ΔQ（pp）|ΔO（pp）|F减少（pp）|
|---|---:|---:|---:|
|旧Full|−0.4069|+0.4818|+0.8888|
|ER|−0.9731|−0.4832|+0.4899|
|SEQ|−0.7431|−0.1872|+0.5559|

对ER/SEQ，Q下降幅度大于O下降幅度，因此即使终态更弱，F仍会更小。**这说明F不能单独承载“旧能力保护更好”的结论。**

对旧Full，则同时存在较弱Q和较好O。报告明确将上述关系写成代数分解，没有把0.4069pp、0.4818pp当作两种独立原因的因果贡献；这一处理正确。O/F也不能被当成两份独立显著性证据。

### 2. 较弱初学具体发生在Dream

全部H方法的 `R11` 和 `R12` **所有13指标**逐seed一致。这支持第一域评价起点及Dream学前读数对齐；它不等于本外审检查了相应权重字节相同。

以下宏F1均值来自 `summary.summary[方法].R12/R22/GB/R32`：

|方法|Dream学前|Dream初学|Dream净增量（pp）|Dream最终|
|---|---:|---:|---:|---:|
|SEQ|87.9689%|92.0736%|4.1047|90.3751%|
|ER|87.9689%|92.5336%|4.5648|90.7289%|
|旧Full|87.9689%|91.4013%|3.4324|89.4551%|
|确定性Full|87.9689%|90.5874%|2.6185|89.7381%|

新候选Dream初学对SEQ逐seed低 **0.4246/1.6985/2.3355pp**，对ER低 **0.9554/2.5478/2.3355pp**，三个seed均不利。其Dream净增量为 **3.1847/2.9724/1.6985pp**，平均2.6185pp，未达4pp。

因此“小遗忘同时伴随较弱Dream获得”有直接的逐seed支持。不过对旧Full，Dream初学是seed51提高、seed52/53下降，不能误写为对旧Full也三seed全降。

Agora净增量8.7846pp通过4pp要求，但它是相对该方法自身的R23学前模型计算，不能替代最终R33与ER/SEQ的比较。不同方法的Agora学前状态不同，增量通过与终态较弱并不矛盾。

## 六、原科学门：哪些失败、哪些通过

依据为 `B/scope.zh.md`、`B/mechanism/scope.zh.md`、`D/scope.zh.md`、`step35_vendorlink_account_policy.py.decide()`，以及 `R/summary.json.conditions`、`completion.json.original_M/candidate_M`。判断使用未舍入值。

|条件|独立核对结论|
|---|---|
|当前4pp规则下的SEQ学习条件|通过；`candidate_M.checks.learning=true`。SEQ的GB均值4.1047pp，且相对V_CLS及绝对能力条件满足。不能据此改写原P的5pp失败记录。|
|明显遗忘前提|失败。SEQ F逐seed为0.5635/1.2581/1.5778pp，均值1.1331pp；0/3达到3pp，均值不到3.5pp。新Full无法改变这项基线事实。|
|Full−SEQ保持|失败。O差−0.1872pp、F减少0.5559pp都不到3pp；O只有1/3 seed正向。其余seed不反向超过1pp的下限实际上通过：O最差−0.9798pp，F减少最差−0.7675pp。|
|Full初学保护|失败点是Dream对SEQ的逐域均值差−1.4862pp，越过−1pp。每seed−3pp下限、绝对能力70%/60%、对V_CLS的2pp/5pp要求均通过。|
|Full新学习保护|失败点是Dream GB均值2.6185pp＜4pp。Agora GC、两域增量每seed非负、B/C绝对能力及对SEQ/V_CLS/V_SET的2pp/5pp相对容差均通过。|
|外部竞争力|通过。对V_SET/V_CLS的O分别提高6.0793/46.1115pp，均3/3 seed正向，满足2pp及方向保护。|
|Full−ER保持增量|自身失败。O均值更低，F减少不到2pp，两者均仅1/3 seed正向；不是只因系统门false被连带判false。|
|Full−ER初学保护|失败。Dream均值−1.9462pp越过−1pp；逐seed−3pp下限通过。|
|Full−ER新域相对容差|通过。Dream−1.9462pp、Agora−1.8540pp均未越过均值−2pp，每seed未越过−5pp。它与更严格的初学−1pp条件失败不矛盾。|
|总体及组件状态|`STOP_NO_H_ADVANTAGE`、`old_score_increment=false`、`old_score_increment_conditions=false`，均正确。|

这里有两项尤其需要保留：

第一，**对SEQ/ER的A/B初学1pp/3pp保护是逐域判断**，不能把A/B先平均成Q后用Q差替代；也不能与对V_CLS的2pp/5pp相近要求混用。

第二，原Full的系统初学保护为true，新候选变为false；原、新Full的新学习组合保护都为false。同时，新候选相对ER的2pp/5pp新域容差由旧Full的不满足变为满足。这是局部改善与另一项保护退步同时发生，不能只报其中一侧。

原P按5pp/3.5pp规则停止的记录保持；后续用户把新增学习要求改为4pp并明确允许M，是后续开发安排。当前 `learning=true` 不会追溯改变原P裁决。

## 七、全部13指标中的正向与负向证据

### 1. 新候选最终三域完整指标表

以下为均值±样本SD，单位%。来源：`R/detail.json.H_full_det.R31/R32/R33`，已与新 `metrics.json` 的351值核对。

|指标|AlphaBay R31|Dream R32|Agora R33|
|---|---:|---:|---:|
|macro_F1|82.5824±1.1570|89.7381±0.3243|89.4908±0.2602|
|Top1|85.4701±0.9218|91.7197±0.6369|91.6571±0.1143|
|micro_F1|85.4701±0.9218|91.7197±0.6369|91.6571±0.1143|
|MRR|88.0584±0.7917|93.8763±0.5302|93.3091±0.1169|
|MAP|88.0584±0.7917|93.8763±0.5302|93.3091±0.1169|
|Recall@1|85.4701±0.9218|91.7197±0.6369|91.6571±0.1143|
|NDCG@1|85.4701±0.9218|91.7197±0.6369|91.6571±0.1143|
|Recall@3|89.2552±0.5887|95.3291±0.9729|94.0571±0.4121|
|NDCG@3|87.6824±0.7524|93.8302±0.7881|93.0766±0.2325|
|Recall@5|90.9035±0.7402|95.9660±0.3677|95.2762±0.2379|
|NDCG@5|88.3575±0.8105|94.0952±0.5647|93.5799±0.1079|
|Recall@10|92.9792±0.1057|98.3015±0.3677|96.4952±0.0660|
|NDCG@10|89.0460±0.6210|94.8708±0.3142|93.9835±0.0905|

### 2. 辅助指标不会改变主裁决，但不应被概括为全部同向

以下均为 `detail.json` 中对应方法、终态R格和指标的均值差，单位pp。

**对旧Full：** 三域MAP均提高，AlphaBay最终13项指标均值全部提高。但Dream的Recall@3、Recall@5、NDCG@5分别下降0.2123、1.0616、0.2595pp；Agora的Recall@3、NDCG@3、Recall@5、NDCG@5分别下降0.2667、0.0839、0.1143、0.0216pp。因此“局部改善”准确，“三域所有指标全面改善”不成立。

**对ER/SEQ：** Agora最终全部13项指标均值都较低，说明新域末态代价不限于宏F1。但AlphaBay宏F1相对ER/SEQ分别高0.0244/0.2625pp，而MAP分别低0.2039/0.1843pp；Dream Recall@10比ER、SEQ均高0.8493pp。故也不能写成“每个域、每个指标都输”。

**对V_SET：** 三域宏F1与MAP均较高，但Dream Recall@5和Recall@10均低1.2739pp，Recall@3均值持平。外部竞争力门基于旧域O的宏F1，通过该门不代表所有检索截断指标均优。

原报告已经提示指标异向，未用单个辅助读数挽回主门失败。以上是值得保留的非阻断补充，没有发现能推翻其核心裁决而被遗漏的重要正向证据。

## 八、确定性分支的实际作用与机制解释边界

### 1. 实际代码支持“只改变出生MSE学生随机模式”

为解释结果中的操作是否真实，定向追踪了 `step35_vendorlink_account_methods.py`：

1. `deterministic_student()` 临时把encoder普通Dropout及BERT SDPA的dropout设零，用 `fork_rng` 和 `finally` 恢复随机状态及属性；保留训练/激活检查点路径。
2. `backward_h()` 在确定性历史任务中按ER目标计算随机历史监督，移除该分支原出生MSE；当前H监督与RMS辅助CE、历史自然/重组监督仍保持原随机模式。
3. `backward_birth()` 对同一历史群计算自然账号池分数，教师detach，目标为 `beta*mse`，没有再乘alpha；通过 `pair_backward()` 和 `z.backward()` 传播，确定性上下文覆盖检查点反算。
4. `update_account()` 累积各分支梯度后只共同裁剪一次、主optimizer更新一次，辅助optimizer沿原规则。

`step29_vendorlink_methods.py` 的pair头为Linear/ReLU/Linear；`teacher()/Memory.retain()` 对应eval出生教师和既有教师不刷新。`step35_vendorlink_account_run.py.execute()` 则在历史步骤检查确定性分支标记并累计 `deterministic_history_groups`。

这些与本轮冻结单一干预相符，没有发现重复计入MSE、学生断梯度或把整个训练长期切为eval的具体矛盾。本节仅解释结果所依赖的实际操作，不重新展开已关闭的实现资格审查。

### 2. 46,440次执行不等于46,440次全部非零

`R/history_summary.json["51"/"52"/"53"]` 分别记录 `updates=23640`、`birth_updates=15480`。每seed的阶段细项为：

|阶段|出生MSE/出生分支记录梯度的count|首步|正值步数|
|---|---:|---|---:|
|Dream，`stages["2"]`|2,400|两者均0|2,399|
|Agora，`stages["3"]`|13,080|两者均正|13,080|

三seed合计46,440次确定性分支执行，其中保存的出生MSE及其记录表示梯度有 **46,437次为正**；另外三次是各seed的Dream首步。

首步零值仅属于**出生分支**。`stages["2"].history_supervised.first` 三seed均大于0，不能扩写为整个历史监督或全部历史梯度为零。原报告此句宜加上“出生分支”以避免歧义。

六个seed×后续阶段的保存均值也符合：

`history_objective = alpha × history_supervised + birth_weighted`；

`birth_weighted ≈ beta × mse`。

前者最大残差约1.78×10⁻¹¹，后者约1.13×10⁻⁸，与浮点计算尺度相符，没有出现β误乘α或双重计入的数字迹象。

### 3. 正式调用与资格回执共同排除“空开关”，不证明性能收益

`D/qualification/cpu01/receipt.json` 记录独立梯度参考最大差0、旧路径/β0更新最大差0，以及手写实际入口9次更新、6次确定性历史消费。

`gpu01/receipt.json` 记录真实BERT/SDPA、激活检查点、`deterministic_eval_max_abs=0`、`modes_restored=true`、`actual_updates=1`；隔离出生分支的encoder/pair梯度范数分别约0.001260662/0.080058560，并记录实际共同更新后的参数变化。

结合正式历史汇总，足以支持：

> 确定性出生评分分支在本轮实际执行，目标和反传连接有效，正式步骤产生了非零出生目标及记录表示梯度。

它不单独量化每次正式Adam更新中出生项的净贡献，因为还有其他目标、共同裁剪和优化器状态；更不证明这些梯度改善了dev旧域保持能力。报告中的“产生保持作用”“保持确有作用”建议收紧为上述表述，属于非阻断措辞修正。

### 4. 不能把新旧训练差当成纯dropout噪声分解

`B/mechanism/diagnosis/report.zh.md` 中的模式诊断是在固定参数、固定记忆群上比较随机/eval学生。当前新旧Full之后走过不同参数轨迹，整阶段损失均值的差不再是同参数下的噪声量。

旧Full的Dream历史目标均值为0.6421/0.8411/0.6992，新候选为0.6294/0.7816/0.7379，第三seed甚至更高。它支持“关闭该分支随机性后仍有明显保持约束，性能问题未解决”，不支持“已精确消除了多少有害噪声”，也不证明dropout完全无害或是唯一有害原因。

旧报告的历史记录梯度包含监督和保持，本轮表格的出生分支梯度仅含出生项，不能直接作同口径梯度比例比较。当前也没有梯度夹角或各候选因素的隔离证据。固定β、绝对评分目标、Memory覆盖及与当前学习的相容性仍只是可能解释，不能据此宣布某个新β或新机制必然有效。

## 九、费用、适用范围与结论强度

### 1. 费用闭合，无预算矛盾

`R/formal_budget.json.events` 的逐任务向上取整与 `used_seconds` 一致；结合 `D/gate.json.prior_seconds`、30秒收尾和 `R/budget.json` 的结果另账，可得到：

|桶|最终累计秒数|授权上限|
|---|---:|---:|
|训练|92,329|144,000|
|GPU评分及必要核验|675|3,600|
|辅助，含结果统计/保管|1,418|3,600|
|总计|94,422＝26时13分42秒|42小时，仍分别受各桶上限约束|

正式控制器原账为94,389秒，结果统计3秒和准备/保管30秒另计，未覆盖原账或重复相加。

新旧Full三条训练实际生命周期之和相比较，增加约23.8%，算术正确。但共享排程不同，这只说明本批实际成本变化，不能视为严格的纯算法时间开销基准。显存/RSS的约3.49/2.69 GiB是采样峰值，原报告没有把它们误称为瞬时绝对峰值或人为硬帽。

### 2. 当前证据支持的任务与推广范围

任务是546/157/875个已登记观察账号、互斥Q2商品文本对FIT12参考的账号识别；市场已知、候选池和ABC顺序固定，当前端点重新编码参考与query。训练监督身份并非本轮评价中的未见账号。

因此，本轮可以讨论这个固定开发协议下的终态能力、遗忘及方法差；不能转写为真实跨账号同控识别准确率、未见账号泛化、严格自然时间流效果或不同市场顺序下的普遍优势。

已见dev、经历过前序开发选择、三个固定seed和单一候选，使结果适合有限开发判断。三个seed不是三份独立数据集；样本SD不是置信区间；没有confirm结果。共享训练硬件也限制了纯计算成本归因。这些限制已经足以界定当前结论，不要求通过追加seed、bootstrap、重训或改变门来“补成成功”。

`R/completion.json` 的 `confirmation=false`、`application=false`、`automatic_followup=false` 与当前实际入口一致。旧P/M负结果及费用保留；旧M曾有无真值应用输出，不意味着本轮新候选也生成了该输出。

## 十、分级发现与处置

|类别|审查意见|本轮处置|
|---|---|---|
|使本轮结果无效的真实阻断|未发现来源错配、缺seed、缺评分格、指标/公式错误或错误裁决。|接受为完整有效的开发比较。|
|必要可复核缺陷|未发现需要修复、补件或重新计算才能接受结果的缺陷。|**无必要修复。**|
|原科学目标未通过|明显遗忘前提、Full−SEQ保持、初学/新学习保护及Full−ER增量均有真实失败项。|保留 `STOP_NO_H_ADVANTAGE`；当前证据不支持科学成功主张或进入confirm。|
|非阻断措辞C01|`R/report.zh.md` §5“产生保持作用/保持确有作用”容易被理解为已证明有益的抗遗忘效果。|建议改为“保持分支实际执行并产生目标与梯度；性能收益另由结果判断”。|
|非阻断措辞C02|同节“首个历史步的MSE和记录表示梯度均为0”应限定出生分支。|建议明确“出生MSE及其分支记录表示梯度为0”；历史监督本身非零。|
|非阻断说明C03|“全部13指标核对”不等于全部指标同向改善，也不等于对未附旧指标原文件重新验SHA。|保留本文所列辅助指标反例、样本SD性质和源文件边界。|
|后续研究建议|β、绝对评分目标、记忆覆盖和学习相容性尚未分离；原场景明显遗忘不足也未解决。|仅列为以后另定研究问题时的可能方向，不构成本轮必修或新实验许可。|

这些措辞调整不改变任何保存数字、原门、负结果或当前裁决。现有材料足以完成本轮审查，不需要为本报告再启动科研执行。

## 十一、实际阅读与核对范围

本次对55份载荷全部核验了大小和SHA，但不将“核过字节身份”混称为对全部历史源码的重新审查。实际内容核对范围如下：

|范围|具体材料与阅读重点|
|---|---|
|当前请求与主报告|完整阅读 `R/review/request.zh.md`、`R/report.zh.md`、`R/tables.zh.md`。|
|当前合同与实现背景|完整阅读 `D/scope.zh.md`、`D/implementation.zh.md`；核对 `gate.json`、`final_sources.json` 的来源、身份、授权依据及边界。|
|原合同与相关历史|阅读 `B/scope.zh.md`、`B/mechanism/scope.zh.md`、`B/mechanism/result/report.zh.md`、`B/mechanism/diagnosis/report.zh.md`，仅使用与本次母体、门和机制解释直接相关的部分。|
|指标与派生统计|解析核对 `R/metrics.json`、`results.json`、`comparison_results.json`、`summary.json`、`detail.json`、`completion.json`；覆盖新351值、全部2457保存值及相应统计、公式、配对和门。|
|执行与开放链|核对 `R/check.json`、`observation.json`、`campaign_dispatch.json`、`batch.json`、`training_complete.json`、`scoring_complete.json`、`analysis_access.json`、`analysis_receipt.json` 和 `source_receipts/` 全12份材料。|
|操作与资源证据|核对 `R/history_summary.json`、`formal_budget.json`、`budget.json`，以及 `D/qualification/cpu01/receipt.json`、`gpu01/receipt.json`。|
|实际代码|定向追踪六份 `step35_vendorlink_account_{io,policy,methods,run,evaluate,control}.py` 的绑定、训练目标、实际调用、评分/指标和科学裁决；定位 `step29_vendorlink_methods.py` 的评分/教师/Memory及相关运行连接。其余未变依赖核对冻结字节身份，复用原审查，不冒称全量重审。|

本报告没有重新验证未提供的正文、账号映射、标签、预测矩阵、Memory、权重或旧P/M原始文件内容，也没有访问包外私有路径。对当前结果的接受，限定于上述明示范围。

