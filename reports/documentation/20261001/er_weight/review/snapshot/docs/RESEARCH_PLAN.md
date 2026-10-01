# 科研计划与后续顺序

## 当前阶段：ER权重比较已确认，本地实现与外审准备

当前新增阶段（2026-10-01）：用户已完整确认ER历史监督权重比较，并授权“本地修改→网页外审／主审→直接Linux核验及正式运行”。新增λ=0.5、0.25，复用原λ=1/SEQ及三个首阶段完整状态；3456新更新、6912群梯度呈现、12新终点，预计11—14小时，强制24小时/32GiB，train/valid各一次新许可、test/owners不读。相对原ER原23项全过者按O MAP择优，无合格者保留原ER并结束比较。合同、代码、16项手工用例和静态核查已准备，实际网页审查和Linux运行尚未开始；正式启动不再重复申请。见[完整合同](SELLER_ALIAS_ER_WEIGHT.zh.md)、[用户确认](../reports/documentation/20261001/er_weight/decision.json)、[实现审读与未验证项](../reports/documentation/20261001/er_weight/implementation.zh.md)。下文首轮已完成结果及其许可是历史阶段，不与本轮新增许可混算。

截至2026-10-01，**首轮有限历史持续学习训练、唯一valid评价、完整回传和本地独立核查均已完成；实际结果外审全文、独立源码与日志附件已亲自核对，结果有效性通过；ER负结果与LOGIT对ER通过均保留，整体抗遗忘目标尚未达成。** 作业从9月30日14:40:21运行至10月1日06:06:13（Asia/Shanghai），退出0、6048次更新、9504次群梯度呈现、21个终点及64份指标矩阵完整。实际总耗时15小时25分51秒，剩余0；原预计18—26小时／10月1日08:40—16:40完成，实际提前，36小时／64GiB是强制上限。09:24:59核对作业进程已退出。[本轮完整报告](SELLER_ALIAS_BGE_RESULT.zh.md)、[当前状态](../reports/seller_alias_continual/20261001/bge_continual_result/current_status.json)。

外审138成员证据包、136载荷／137校验条目及实际输入495来源匹配；主审核对17,160个既有JSON数值、49项布尔判定，最大差5.55e−16。已修正漏入包的观测链接证据口径，正式18来源和原结果未改；新增裁剪线索保留因果限制。[结果外审与逐项主审](../reports/seller_alias_continual/20261001/bge_continual_result/review/report.zh.md)。

本轮困难排序BGE、s0、ABC／BCA／CAB、冻结／SEQ／ER／LOGIT、6群且全部历史及附属状态≤1MiB及O／N／Z判据未改。train=1、valid=1、heldout／owners=0；326份小结果16,241,505字节回传匹配，18份冻结来源未变。Linux保存结果独立核查1,061,316数值、最大差7.11e−15通过，新标签与模型加载均0。[原始评价](../reports/seller_alias_continual/20261001/bge_continual_result/returned/job/evaluation/evaluation.json)、[独立核查](../reports/seller_alias_continual/20261001/bge_continual_result/audit/audit.json)。

SEQ首域MAP遗忘0.063393、条件95%[0.041809,0.085423]，三个顺序同时有新域学习，预定遗忘观察门通过。ER−SEQ仅4/23通过，旧域MAP下降0.039873；LOGIT−ER为23/23通过，旧域MAP增加0.046887。但LOGIT相对SEQ旧域Recall@5及新域、最终最新域性能仍下降，不能称整体优于普通续训或抗遗忘已解决。单种子／已开发valid结论不等于创新或论文最终验收。历史18权重例外保持；本轮必要共享状态／对照仍保留Linux，未回传大权重或历史文本标签。

本轮先观察困难排序BGE的新群体学习和旧域遗忘，再诊断随机重放及重放加历史分数保持。每顺序从原预训练BGE及同一s0随机头开始，不能借用已见三域的联合权重。

### 已完成的准备

- 首轮具体方法、6群／1MiB供给、配对预算、O／N／Z判据及6048更新已确认。
- 合同、policy、代码、实际网页外审与主审、用户交付批准、Linux CPU核验及正式启动均已完成。
- 正式6048更新、一次valid评价、完整回传和本地独立核查已完成；实际结果外审与主审也已完成，下一步明确新方法合同。

### 当前及随后工作

首轮结果外审与主审已完成。当前先完成页首已确认的ER有限权重比较：网页实际审查和主审通过后直接Linux核验及正式运行；新方法设计可以并行准备，创新方法、多种子和最终独立留出仍须独立合同。LOGIT保留为已知方法适配基线；原结果和判据保持，不重复基础test。

两端文件清理已完成：Windows删除58份原有冗余文件112,594,084字节，Linux删除1份中转包18,784字节；必要科研记录、原始外审证据、18份历史权重和活动训练状态保留。9月30日清理时Git／LFS未改；10月1日先补26项本地归档及1项结果外审收尾，随后按用户纠正全部推送至原远端科研分支，27项提交的目标2f293eac已与实际远端一致。阶段提交与推送纪律已补明确，详见[远端补交记录](../reports/maintenance/20261001/git_push/report.zh.md)；未清理远端LFS。[清理完成回执](../reports/maintenance/20260930/project_cleanup/completion.json)、[存储与恢复](STORAGE.zh.md)。

## 已完成基础验收及其历史边界

2026-09-30基础验收闭环：**固定s0独立test的实际结果外审及主审已完成，结果有效性通过；原合同仍为8/9，唯一失败T1（ΔMAP≥0.01）。** ΔMAP=+0.007484203，条件95%区间[0.000050331,0.014610678]，ΔRecall@5=+0.014434524；原定AP/AUC及Brier/log_loss观察值保护通过。用户获知结果后明确“可以，这个test结果可以了，有提升就行”，接受这一实际增益作为后续研究基础；保留事后决定的性质和原机器`passed=false`。这不等于各域、所有分类工作点或总体非劣效均得到证明。详见[完整结果](SELLER_ALIAS_TEST_RESULT.zh.md)及[用户原话](../reports/seller_alias_continual/20260928/test_result/user_acceptance.json)。

实际回复`1c91151a-333d-44ad-832c-aa67239abb74`、完整F1—F10、独立源码和全部原始运行／修订日志已亲自审读；118成员外审证据包完整归档。外部116份实证文件／117条校验记录、259份提交来源和34份当前冻结科学来源匹配；外审实际收到的ZIP大小／SHA与用户授权包一致。网页独立数值核验444,635项、最大差9.99e−16；主执行者核对1,138个既有JSON数值和全部判定，两个重放CSV与原Linux字节一致。已修正两个约53秒区间的报告标签：包含评分后文件／参数核对，非纯前向；原数值、源码、失败及审查参考修正均保留，不重跑。历史失败包网页覆盖17/18，本地实物18/18均已匹配。上述科学主审核对未新连Linux、启动科研运行或读取正式文本／标签／模型；随后文档与小型证据归档同步另记。[逐项主审及限制](../reports/seller_alias_continual/20260928/test_result/review/disposition.json)。

正式运行仍为9月28日18:30:09—18:32:14 CST，125.20秒退出0、剩余0（原估10—30分钟，上限1小时／4GiB）；29份正式结果及9份原审计证据回传一致。heldout本阶段一次许可已消费，train/valid/owners0，不重复test或在其上调参；18份Linux权重继续按用户例外保留。**基础验收之后的首轮持续学习方案、比较、预算和评价合同现已确认并正式执行。** 当前进度见页首；不将联合三域权重当首域初态，现有120群不再当最终未开发留出，后续新最终留出另立合同。

## 历史执行记录

以下保留各时点的准备、失败及等待状态；“当前／下一步／未授权”均指原记录时点。现行状态以页首带观测时间的持续学习记录为准，历史许可不能移作新许可。

最新下一步（2026-09-28）：**test入口及Linux手工核验已完成，完成实际网页实现外审与主审后直接执行固定s0独立test。** 用户已明确“核查test实现完毕后就开始test吧”，同一具体范围不再申请启动。校准17项valid与实际结果外审保持通过。[授权](../reports/documentation/20260928/test_plan/authorization.json)、[实现核验](SELLER_ALIAS_TEST_IMPLEMENTATION.zh.md)。

用户已选择现有120群全部用于本次基础模型验收；持续学习最终留出以后另建、确认并封存，本次没有切分或生成数据。[test合同](SELLER_ALIAS_TEST.zh.md)固定A/C s0 E6及已有映射、九项同时通过条件、5000次分域整群bootstrap。最终CPU15项通过，34份科学来源一致；实现包经明确授权后16:15:33 CST提交网页Pro、16:16:01确认正在回应，待实际外审及主审。正式test尚未启动、heldout解析0。正式程序及小型核对预计10—30分钟，强制上限1小时／4GiB，单GPU单CPU；不包含外审等待，没有实际完成时刻。独立验收和结果有效性审查通过前不设计持续学习。

以下校准提交状态为外审结束前历史，当前状态以以上两段及test合同为准。

最新下一步（2026-09-28）：**完成概率校准结果外审与主审，再准备固定s0的独立合成test验收合同。** 已确认的六映射／12参数研究正式完成，14:08:51 CST退出0，实际1.40秒、剩余0（原估1—5分钟，上限1小时／256MiB）；17项valid开发条件全部通过。C的原排序收益保留，概率损失相对同样校准的A和原始A均满足保护；26份科学来源未改，train／valid各一次许可已消费，test／owners未读，18权重保留。新结果包已准备；自动审批以此前具体授权仅覆盖实现包为由拒绝上传，用户随后明确允许本结果包，14:39:51 CST已发送，Pro网页确认正在回应；必要收尾后暂停等待。实现审查不替代结果审查。test仍须具体合同、入口审查及单独阶段授权；在独立验收前不开始持续学习设计。[完整结果与校准解释](SELLER_ALIAS_CALIBRATION_RESULT.zh.md)、[已确认合同](SELLER_ALIAS_CALIBRATION.zh.md)、[用户回答](../reports/documentation/20260928/calibration_plan/decision.json)。下方九模型结果为校准前历史，不回写其9过4败。

更新日期：2026-09-28。本轮三配置九模型训练、7776更新及一次valid已完成，02:15:28 CST退出0，实际14小时25分、剩余0（原估计14—18小时，24小时上限）。C−A的MAP／R5及AP／AUC改善，但Brier／log_loss均值与固定s0退化，13项通过9项，完整验收未通过。本地保存结果独立核查、实际GPT6 Pro结果外审与主执行者逐项处置均已完成，结果有效性通过，完整验收失败保持；外审没有确认必须修复的新代码或数值缺陷。用户最新明确保留本轮全部18份权重作为后续候选，仅留Linux，未执行删除；C保留候选身份。见[结果报告](SELLER_ALIAS_RANKING_RESULT.zh.md)及[用户保留决定](../reports/seller_alias_continual/20260928/ranking_result/weight_retention.json)。原判据和监督边界保持，test与持续学习设计继续暂缓，尚未确认新训练或概率校准方案。

## 已确认工作的状态

| 工作 | 当前状态 |
|---|---|
| 旧三模型／敏感性评价与外审 | 已完成、原结论保持 |
| 中文模型Windows下载、Linux归档、本地副本清理 | 已完成 |
| 四配置合同、真实CPU更新、实际GPT6 Pro实现审查 | 已完成 |
| 四配置六轮训练、3456更新、八模型恢复／评分 | 20:18:48退出0、完成 |
| 完整门后valid一次正式评价 | 20:26:21退出0、完成 |
| 小型结果回传、15202数值及2424盲计数核查 | 已完成、无新标签读取 |
| 网页结果外审与候选检索用途补充 | 实际GPT6 Pro完整回复已审读，独立脚本Linux复核及处置完成 |
| Git LFS | 本地prune释放0；用户已取消后续清理，草稿已删，远端未改 |

完整分析见[中文结果](SELLER_ALIAS_CHINESE_RESULT.zh.md)。D为预定主候选；D−A、D−LaBSE的排序改善与原严格自动判定门失败同时成立。

## 已确认的下一步：保留D基线，先做针对性改进

2026-09-24用户明确要求：**把D保留为可用基线，再做一轮有针对性的改进；等这轮实验真正提升模型的马甲候选排序能力后，再开始设计持续学习方案。** 主执行者判断该顺序合理，现作为正式下一步；不再并行推进持续学习方案设计。[用户决定与范围](../reports/documentation/20260924/next_plan/decision.json)。

1. **保留D固定结果与配置基线。** 保留完整结果、数据边界、训练配置与负面指标。2026-09-26用户明确将原D权重纳入全部训练载荷清理，此项覆盖此前“保留第6轮模型”的存储要求；后续如需新推理／重建，另行确认具体范围与监督许可。D已有候选排序能力的研究结论不因删除权重改判，不追认为通过原自动判定门。
2. **保留C排序收益及已完成的概率校准结果。** 原九模型四项概率保护失败保持历史身份；另行确认并执行A/C三种子E6的六个正斜率／截距拟合，17项开发条件全部通过，实际结果外审及主审已闭环。当前按第3项准备固定s0独立验收。全部18权重继续保留。[校准结果](SELLER_ALIAS_CALIBRATION_RESULT.zh.md)、[实现证据](SELLER_ALIAS_CALIBRATION_IMPLEMENTATION.zh.md)、[原九模型结果](SELLER_ALIAS_RANKING_RESULT.zh.md)。
3. **先核实真实提升，再设计持续学习。** 用户于2026-09-25确认：先在valid开发比较，满足预先约定判据后，再做一次独立合成test验收。test须在具体合同和入口审查完成、另获阶段授权后读取。完成预定验收、本地主审和适用外审且达到提升判据后，才进入持续学习方案设计。若未达到，保留负结果、分析原因并重新讨论改进方向，持续学习设计继续暂缓。[本次回答与范围](../reports/documentation/20260925/retrieval_plan/decision.json)。

“先改进再持续学习”是用户确认的推进顺序，不表示D没有基础能力，也不恢复原0.1%／50%自动判定门作为通用资格线。本轮主要指标、阈值、轮次、历史结果均不追溯改变。未来持续学习起点须按到达顺序建立，已联合见过三域的权重不能冒充首域模型。

旧中文阶段与本轮加权汇总阶段的train／valid许可均各自消费完成。加权汇总作业于2026-09-25 12:57:45—20:35:20 CST完成，4320更新及一次valid，实际约7小时37分，剩余0；13项改进条件中8项未通过，保留有效负结果，test未授权，持续学习设计暂缓。实际GPT6 Pro结果回复、附件、主审处置及独立脚本Linux重放均完成；原自动审批拒绝、新授权和等待暂停保留为历史，不把训练前外审当作本次结果外审。Linux全部训练权重依用户要求已删除，D仅保留配置与结果。未修改冻结计算、成功线或18份正式来源。下一步讨论不能自动重开本轮、追加监督或追认成功。接手入口为[交接](AI_RESEARCH_HANDOFF.zh.md)、[纪律](RESEARCH_DISCIPLINE.zh.md)及[本轮结果](SELLER_ALIAS_POOLING_RESULT.zh.md)。

## 训练期间的需求讨论

2026-09-27用户明确：最终交付为投稿论文，目标CCF B或C类期刊／会议，新方法须有一定创新；先完成当前中文合成受控任务的持续学习主线，再决定是否扩大候选库或纳入无同控候选查询。用户进一步确认：**研究建立在可用历史训练数据较少的背景之下**，后续围绕该前提研究旧分布能力保持与新群体学习。此背景已确定；具体历史数据可用量、可用方式及比较预算在后续合同中量化，具体刊会和期限尚未指定。[讨论与回答原文](RESEARCH_DISCUSSION.zh.md)保留问题、实际回复、生效范围与结果报告框架，并标出首次test前须明确的数据用途问题。未变更活动合同、13项判据或监督授权；基础排序改善不自动构成论文方法创新，持续学习设计仍在基础模型独立验收之后。

### 直接相关文献与论文定位

按用户要求完成原始文献核查，随后取得网页GPT6 Pro完整回复与来源附件并独立处置。[文献定位](SELLER_ALIAS_LITERATURE.zh.md)已补入LUAR、DiCGRL、ERD和VeriDark，并明确CREAM的共享覆盖训练选择、L²R无兼容版本与vanilla的区别，以及QDC训练蒸馏的关联。现有工作比初稿覆盖更直接：多记录文本、未见身份评价、共享对象和关系保持均不能单独认证创新。结论仍是问题值得继续，但尚无可以作算法差异判断的完整新机制；不能据此宣称方法创新或录用资格。已处理提交呈现中的链接格式缺陷，保留原件与稳定文献目录。[主审处置](../reports/seller_alias_papers/20260927/continual_position/disposition.json)。潜在贡献只是待证问题，不冻结新的持续学习方案，不改变活动实验与先基础模型验收的顺序。

## 历史原文（以下均为当时状态）

> 以下保留整理前正文、历史结论和来源散列。旧“当前／下一步／待恢复”只指其原记录时点，不是本阶段执行指令。

2026-09-15入口校正：本文下方为历史计划或进展，旧“当前／下一步”仅指原记录阶段，不能作为现行执行授权。现行主线是有限历史记忆下的中文暗网卖家马甲持续识别；第一路联合诊断已完成，第二路随机ER已于9月15日12:38正常结束并完成一次valid评价，本地三项缓解判据失败，实际结果外审及本地复核已完成，有效负基线闭环。当前状态、授权及最新实测时间以[科研交接](AI_RESEARCH_HANDOFF.zh.md)为准，执行合同见[随机ER方案](SELLER_ALIAS_MITIGATION_PLAN.zh.md)，大型载荷归档位置见[存储与恢复](STORAGE.zh.md)。历史结论和哈希原义不变。

版本：`2026-05-14`

本文档是当前项目的正式总纲。目标不是记录灵感，而是固定研究边界、实验顺序、数据纪律和每一步的验收条件，确保后续工作可复现、可审计、可写入论文方法部分。

## 当前执行状态（2026-05-14）

当前 active boundary 是 `2026-04-23` English valid/test top-up refreeze。该边界在 English item-level seller-facing direct identifier 扩充和 validation split 修复之后，继续补强英文 valid/test。Step 7、Step 9、Step 11 已经在 Linux runtime 上按该边界重跑并同步；Step 11 已于 `2026-04-24` 完成 manifest-only 清理和当前 cluster-level audit。`2026-05-13` 本地全项目复核未发现更新的 Step 3 / Step 4 / Step 5 / Step 7 / Step 9 / Step 11 artifacts，因此当前结论仍以这条边界为准。

`2026-05-14` 已新增两个待重跑的方法分支：Step 9 的 `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` 训练期正样本表征 mixup，以及 Step 11 的 `relation_reliability_filter` 低可靠关系边过滤。它们借鉴 RABot 的 minority augmentation / spurious-edge filtering 思路，但按本项目任务重写为 seller-pair verification 和 candidate-cluster triage 控制分支。当前结果表在 Linux 重跑前仍以 `2026-04-24` 同步 artifacts 为准。

当前主线结论：

- Step 5 active supervision:
  - English `en_content_train_pool`: `1321 reviewed / 734 supervision`
  - English split: `train / valid / test = 401 / 152 / 181`
  - English split labels: `train = 116 positive / 285 negative`, `valid = 42 positive / 110 negative`, `test = 51 positive / 130 negative`
  - Chinese `zh_target_strict`: `1016 reviewed / 522 supervision`
  - Chinese split: `train / valid / test = 335 / 81 / 106`
  - Chinese split labels: `train = 61 positive / 274 negative`, `valid = 14 positive / 67 negative`, `test = 21 positive / 85 negative`
- Step 7 clean zero-shot reference:
  - current summary: `reports/step7_training_summary.json`
  - `core_zero_shot_default`: ROC-AUC `0.588235`, AP `0.448547`, balanced accuracy `0.562465`
  - `core_zero_shot_bge_m3`: ROC-AUC `0.601681`, AP `0.448761`, balanced accuracy `0.562465`
  - best clean Step 7 ROC-AUC ablation: `core_zero_shot_default_no_structural`, ROC-AUC `0.623529`, AP `0.287652`, balanced accuracy `0.572269`
  - operational identifier control: `identifier_augmented_default`, ROC-AUC `0.606443`, AP `0.418989`, balanced accuracy `0.619888`
  - status: current-boundary Step 7 rerun synchronized; small-validation guard is resolved, but several LightGBM fusion models collapse to shallow one-tree solutions
- Step 9 clean few-shot promoted line:
  - `core_few_shot_multilingual_e5_large_lr_l2 / 50pct`
  - status: current-boundary Step 9 rerun synchronized; this line repairs the collapsed Step 7 fusion baseline but only modestly exceeds raw E5 semantic ranking
  - seed metrics:
    - `20260320`: ROC-AUC `0.819048`, AP `0.540482`, balanced accuracy `0.589356`
    - `20260321`: ROC-AUC `0.824650`, AP `0.541473`, balanced accuracy `0.583473`
    - `20260322`: ROC-AUC `0.811765`, AP `0.534180`, balanced accuracy `0.589356`
- Step 11 clean discovery family:
  - policy updated to `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct`
  - graph threshold override `0.56`
  - current-boundary Step 11 graph outputs synchronized and manifest-retained
- Step 11 conservative anchor/control:
  - `core_zero_shot_bge_m3`
  - no graph threshold override; resolves to pairwise selected threshold `0.483444`
  - current-boundary graph outputs synchronized and manifest-retained
- Step 11 operational identifier controls:
  - `identifier_augmented_few_shot_default_lr_l2 / 100pct / seeds 20260320 / 20260321 / 20260322`
- Step 11 manifest and audit:
  - manifest: `reports/step11_current_manifest_20260424.json`
  - current retained Step 11 summaries: `13`
  - stale/unreferenced `reports/step11_*` files deleted: `200`
  - current audit: `reports/step11_cluster_level_audit.current_20260424.csv/json`
  - `2026-05-13` recheck: root `reports/` still has exactly `13` current Step 11 clustering summaries; the cluster-level audit remains explicit-summary based, not `reports/` glob based

当前关键纪律：

- clean scientific claim uses the transfer-safe LR/L2 few-shot line, not identifier-augmented controls.
- identifier-augmented results are operational controls only.
- current-boundary Step 7 / Step 9 / Step 11 are synchronized and technically valid for current-boundary analysis.
- BGE-M3 must not be assumed to be the current strongest clean zero-shot baseline; Step 9 must compare against the refreshed clean Step 7 baselines.
- Step 9 policy has been expanded on `2026-04-22` so the next rerun covers clean default, BGE, no-structural clean ablation, embedding-only / plus-reranker semantic controls, and identifier-augmented operational controls.
- Step 9 policy has been expanded again on `2026-05-14` with a training-only positive-pair mixup branch; synthetic rows are not Step 5 labels and must not enter `zh_valid` / `zh_test`.
- Step 11 graph filtering has been expanded on `2026-05-14` with deterministic relation reliability scoring before reciprocal-top-k/shared-neighbor pruning; this is a spurious-edge control, not cluster ground truth.
- Step 12 robustness audit has optional mixup model specs and paired comparisons; before the new Step 9 artifacts exist, these specs are skipped instead of breaking the current audit.
- The current cluster-level audit has separated direct identifier/contact cores from template/topic expansions for the thirteen manifest-retained Step 11 summaries.
- LR/L2 graph expansion is proof-level evidence only for audited identifier/contact cores, not for whole low-threshold components.
- stale root Step 11 outputs and old report snapshot directories were removed during cleanup; current Step 11 audits must use `reports/step11_current_manifest_20260424.json` and explicit `--summary` inputs, not `reports/` globbing.

## 历史执行状态（2026-04-20）

### Step 5 v3 queue source correction

`2026-04-20` 重新审计发现，第一版 Step 5 v3 targeted review policy 仍硬编码指向旧的 overfit BGE few-shot graph：

- `reports/step11_core_few_shot_bge_m3_ratio_10pct_seed_20260320_clustering_summary.json`
- primary graph threshold `0.457259`

该图来自一个小样本 `10pct` BGE few-shot run，曾在小测试集上出现可疑 perfect score，并产生过宽的 threshold-pass graph。因此它不再作为后续 targeted review 的可信来源。

当前已完成修正：

- `schema/step5_independent_targeted_review_policy.json` 已改为读取 calibrated BGE graph：
  - `reports/step11_core_calibrated_bge_m3_clustering_summary.json`
  - `reports/step11_core_calibrated_bge_m3_zh_target_strict_scored_pairs.csv`
  - `reports/step11_core_calibrated_bge_m3_zh_target_strict_clusters.threshold_0500000.csv`
- 新 corrected queue 已生成：
  - `reports/step5_zh_target_strict_independent_targeted_review_queue.step11_calibrated_bge.csv`
  - `reports/step5_zh_target_strict_independent_targeted_rereview_queue.step11_calibrated_bge.csv`
  - `reports/step5_independent_targeted_review_queue_summary.calibrated_bge.json`
- corrected queue 结果：
  - net-new review rows: `0`
  - rereview rows: `37`
  - missing retained pairs: `0`

纪律要求：

- 不再使用旧 `step11_bge_v3` queue 做新的 cleanup 决策
- active freeze 暂不自动改动
- 只有 corrected 37-row rereview queue 被显式复核后，才允许再切下一版 freeze

## 历史执行状态（2026-04-16）

当前项目已经完成三件必须区分的动作：

1. 当前 Step 5 v2 boundary、fresh Step 7 / Step 9 / Step 11 结果、当前 docs，已冻结成独立里程碑快照：
   - `reports/step5_v2_milestone_snapshot_20260416`
   - `reports/step5_milestone_snapshot_summary.json`
2. Step 5 v3 targeted rereview queue 已生成：
   - `schema/step5_independent_targeted_review_policy.json`
   - `scripts/step5_build_independent_targeted_review_queue.py`
   - `reports/step5_zh_target_strict_independent_targeted_review_queue.step11_bge.csv`
   - `reports/step5_zh_target_strict_independent_targeted_rereview_queue.step11_bge.csv`
   - `reports/step5_independent_targeted_review_queue_summary.json`
3. Step 5 v3 targeted cleanup 与 refreshed freeze 已完成：
   - `schema/step5_independent_targeted_cleanup_policy.json`
   - `scripts/step5_apply_targeted_rereview_policy.py`
   - `reports/step5_independent_targeted_cleanup_summary.json`
   - `reports/step5_frozen_silver_summary.json`

当前 v3 queue 的实际状态：

- net-new review rows：`0`
- rereview rows：`34`
- requested but not retained：`3`

因此当前状态不是“准备做 v3”，而是：

- `34` 条高风险边的 v3 rereview 已完成
- 已有足够 justified 的变更切出新 freeze
- 当前 active training boundary 已切换为 v3 cleaned freeze

同时，`2026-04-16` 的代码审计已经确认并修复了两类运行器问题，但结果文件尚未刷新：

- Step 9 现在会显式断言 sampled `zh_train` 与固定 `zh_valid` / `zh_test` 的 seller overlap 必须为 `0`
- Step 11 不再把 `0.8 / 0.9` 当成所有 scorer 都适用的硬 sensitivity views；当绝对阈值高于当前 scorer 的 observed score ceiling 时，会自动回填分位数阈值
- Step 11 的 Step 9 few-shot family-best 选择不再只看单个 seed 的 `zh_test` 峰值，而会优先考虑 ratio 级别的稳定性

当前真正剩下的阻塞不是 Step 5，而是本地运行环境缺少 `lightgbm`。

- Step 7 在本地 Windows 环境会直接报错：
  - `lightgbm is required for Step 7 fusion training`
- 因此当前还不能把旧的 Step 7 / Step 9 / Step 11 结果继续当作 active boundary 下游结论

因此，在进入任何新的主线结论前，还需要在 Linux 或带 `lightgbm` 的环境中按顺序重跑：

- Step 7
- Step 9 few-shot
- Step 9 calibration
- Step 11 dynamic trio

## 1. 研究目标

本项目研究的是：

- 在英文市场学习“马甲判别规律”
- 将这些规律迁移到中文市场
- 在中文市场内部识别疑似马甲账户

本项目明确不做：

- 英文账号与中文账号的一对一直接配对
- 未经人工复核的全自动中文银标签构造
- 在数据边界未审计完成前直接训练模型

## 2. 当前核心研究问题

本项目围绕四个问题展开：

1. 英文市场中，账号内容、风格、结构和标识符能否支持马甲识别
2. 在不直接使用中文监督标签的情况下，英文学到的规律能否零样本迁移到中文市场
3. 少量中文标注能否显著提升目标域识别性能
4. 在全量中文市场中，基于 pairwise 打分和图聚类，能否得到有意义的疑似马甲簇

## 3. 数据分层与正式定义

### 3.1 EN-Gold

用途：

- 严格 benchmark
- benchmark 泄漏审计
- 不作为英文内容主训练池

数据：

- `tijkc3xx.sql`
- `3z669jwe.sql`
- `suspected_sockpuppet_strong.csv`
- `suspected_sockpuppet_weak.csv`
- `suspected_imposter_rows.csv`

当前约束：

- `tijkc3xx.sql` 主要是 vendor registry
- `3z669jwe.sql` 是强身份线索辅助证据
- benchmark 相关 alias 不能回流到英文内容训练池

### 3.2 EN-Content

用途：

- 英文 seller-profile 构造
- 英文银标签候选构造
- 英文监督训练
- 领域适配预训练语料

数据：

- `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`
- `market_item.xlsx` 的非目标中文市场部分

当前约束：

- 与 `EN-Gold` alias 重叠的内容卖家必须排除
- 非目标英文池中带 CJK 的行必须排除

### 3.3 ZH-Target Strict

用途：

- 中文严格目标域评测
- 中文银标签候选构造
- 中文少样本适配
- 中文全量图聚类

数据：

- `market_item.xlsx` 中 `market in {"中文暗网交易市场", "茶马古道"}`

当前约束：

- 只有这两个 market 属于严格中文目标域
- 其他 market 就算文本出现中文，也不得自动视为中文目标域

### 3.4 ZH-Target Aux

用途：

- 中文辅助语料
- 中文 seller-profile 补充
- 领域适配预训练补充语料

数据：

- `products_data.csv`

当前约束：

- 该文件缺乏明确 market 字段
- 当前只能作为辅助域，不进入严格中文 benchmark

## 4. 总体技术路线

正式主方案维持为三段式：

1. Embedding backbone
   默认：`gte-multilingual-base`
   强对照：`BGE-M3`
   经典基线：`multilingual-e5-large`
   跨语言对齐基线：`LaBSE`、`paraphrase-multilingual-mpnet-base-v2`
2. Text reranker
   默认：`gte-multilingual-reranker-base`
   备选：`bge-reranker-v2-m3`
3. Final fusion
   默认：`LightGBM`

分工：

- backbone 负责跨语言表征
- reranker 负责账号对文本精排
- LightGBM 融合语义、风格、结构、标识符和规则特征

## 5. 当前已完成步骤

### Step 1. 统一 schema 与数据审计

目标：

- 把原始文件统一到一个规范输入口径
- 明确 source mapping
- 明确不可直接合并的字段和主键策略

已产出：

- `schema/source_schema_map.json`
- `docs/STEP1_SCHEMA.md`
- `scripts/step1_schema_audit.py`
- `reports/step1_schema_audit.json`

已验证结论：

- 所有 source mapping 均能对上原始字段
- `market_item.xlsx` 中只有两个中文 market 可进入严格目标域
- `products_data.csv` 市场归属未定，必须独立对待

验收条件：

- 字段映射无缺失
- 数据源角色定义明确
- 所有 hard rules 固化到 schema

### Step 2. 严格切分与泄漏隔离

目标：

- 将 `EN-Gold`、`EN-Content`、`ZH-Target Strict`、`ZH-Target Aux`、`Excluded` 正式分开
- 把英文训练池与英文 benchmark 的 alias overlap 清到 0，并把 benchmark-linked aux fingerprint overlap 清到 0

已产出：

- `schema/step2_split_policy.json`
- `docs/STEP2_SPLIT_AND_LEAKAGE.md`
- `scripts/step2_build_split_manifests.py`
- `reports/step2_split_summary.json`
- `reports/step2_en_gold_benchmark_manifest.csv`
- `reports/step2_en_gold_alias_exclusion_list.csv`
- `reports/step2_en_gold_contact_exclusion_list.csv`
- `reports/step2_content_item_manifest.csv`
- `reports/step2_content_seller_manifest.csv`
- `reports/step2_aux_pgp_evidence_manifest.csv`

当前结果：

- 官方英文 benchmark alias 排除表：`24944`
- 强身份闭包 alias 排除表：`24985`
- 强身份闭包 fingerprint：`19507`
- 英文内容原始 item：`459669`
- 英文内容保留 item：`311019`
- 严格中文目标域 item：`17556`
- 中文辅助域 item：`2104`
- 后验 `EN-Content` vs `EN-Gold` alias overlap：`0`
- 后验 `EN-Content` vs benchmark-linked aux fingerprint overlap：`0`

验收条件：

- `EN-Content` 与 `EN-Gold` alias overlap 必须为 `0`
- `EN-Content` 与 benchmark-linked aux fingerprint overlap 必须为 `0`
- `ZH-Target Strict` 只能来自两个指定中文 market
- `products_data.csv` 保留在辅助域

### Step 3. 账号级画像构造

目标：

- 把 item-level 数据变成 seller-level 画像
- 为银标签构造、pairwise 输入和后续模型提供统一账户输入单位

已产出：

- `schema/step3_seller_profile_schema.json`
- `docs/STEP3_SELLER_PROFILE.md`
- `scripts/step3_build_seller_profiles.py`
- `reports/step3_seller_profile_summary.json`
- `reports/step3_seller_profiles.en_content_train_pool.jsonl`
- `reports/step3_seller_profiles.zh_target_strict.jsonl`
- `reports/step3_seller_profiles.zh_target_aux.jsonl`
- `reports/step3_item_identity_signals.en_content_train_pool.csv`
- `reports/step3_item_identity_signals.zh_target_strict.csv`
- `reports/step3_item_identity_signals.zh_target_aux.csv`

当前结果：

- 总 seller profile：`13292`
- 英文训练 seller：`7522`
- 中文严格目标域 seller：`5097`
- 中文辅助域 seller：`673`
- 所有 profile 均生成了非空 `profile_text`
- 大多数 seller 已生成 long-tail signature 字段：
  - 英文训练 `signature_description_segments`：`7232`
  - 中文严格目标域 `signature_description_segments`：`4773`
- seller 数与 item 数已和 Step 2 对齐

验收条件：

- seller 数按 bucket 必须与 Step 2 完全一致
- item 数按 bucket 必须与 Step 2 完全一致
- 每个 seller profile 的 `profile_text` 必须非空

### Step 4. 英文与中文银标签候选清单构造

目标：

- 在 seller-profile 基础上构造候选账号对
- 不直接输出最终标签，而是输出“待人工复核候选”

已产出：

- `schema/step4_silver_candidate_schema.json`
- `docs/STEP4_SILVER_CANDIDATES.md`
- `scripts/step4_build_silver_candidates.py`
- `reports/step4_candidate_summary.json`
- `reports/step4_en_silver_candidate_pairs.csv`
- `reports/step4_zh_target_strict_silver_candidate_pairs.csv`
- `reports/step4_zh_target_aux_silver_candidate_pairs.csv`

当前结果：

- 英文候选对：`6623`
- 中文严格目标域候选对：`3793`
- 中文辅助域候选对：`580`
- 英文高优先级候选：`3126`
- 中文严格目标域高优先级候选：`678`
- 所有候选均保留了 rule hits 与人工复核字段
- `same_alias_identity_continuity` 与 `sockpuppet_primary` 已明确区分

方法说明：

- 英文候选：
  - 共享联系方式
  - 共享长描述克隆
  - 共享内容型标题克隆
  - `aux PGP` 指纹别名补充证据
  - seller profile 稀疏近邻召回（词项 + bigram + signature fields）
- 中文候选：
  - 共享联系方式
  - 跨 seller 文案克隆
  - seller profile 稀疏近邻召回（字级 2-3 gram + Latin token + signature fields）
  - 结构支持分数辅助排序

验收条件：

- 候选表必须包含“入选原因”
- 必须保留人工复核状态字段
- 不能把候选自动当成最终真标签
- Step 4 原始 review queue 不能直接替代 Step 5 的分层平衡复核队列
- 当前活动复核入口以 Step 5 平衡复核队列为准；Step 4 raw review queue 属于可再生中间产物，不作为保留报告文件

## 6. 下一步与后续完整科研流程

### Step 5. 人工复核与银标签冻结

目标：

- 把候选对表转成可实验使用的银标签集
- 在冻结银标签时保留证据多样性，避免后续监督学习退化为 identifier shortcut

已产出准备文件：

- `schema/step5_review_policy.json`
- `docs/STEP5_REVIEW_AND_FREEZE.md`
- `scripts/step5_build_review_strata.py`
- `reports/step5_review_strata_summary.json`
- `reports/step5_en_balanced_review_queue.csv`
- `reports/step5_zh_target_strict_balanced_review_queue.csv`
- `reports/step5_zh_target_aux_balanced_review_queue.csv`

当前准备结果：

- 英文平衡复核队列：`6623`
- 中文严格目标域平衡复核队列：`3793`
- 中文辅助域平衡复核队列：`580`
- 可优先形成非 identifier 主导正样本的候选数：
  - 英文：`5210`
  - 中文严格目标域：`3765`
  - 中文辅助域：`580`

方法：

- 每个候选对先被分入固定 `review_stratum`：
  - `identifier_plus_text`
  - `text_clone_primary`
  - `semantic_structural`
  - `identifier_primary`
  - `semantic_only`
  - `same_alias_continuity`
- Step 5 复核必须按分层平衡队列推进，不能只按全局 rank 从高到低确认
- 每个候选对至少打上：
  - `positive`
  - `negative`
  - `uncertain`
- 不确定样本不得混入监督训练正负样本
- `same_alias_continuity` 只保留做审计与 continuity 分析，不进入主 sockpuppet 监督池

输出：

- 英文冻结银标签集
- 中文冻结银标签集
- 复核日志
- 带 `review_stratum` 的标签审计表

验收条件：

- 所有训练用银标签都有复核状态
- 每条冻结银标签都保留 `review_stratum`
- 当存在足够非 identifier 候选时，主监督正样本中非 identifier 支持样本占比不低于 `30%`
- 固定训练/验证/测试拆分

### Step 6. 领域适配预训练

目标：

- 让 backbone 适应暗网语料和跨域风格

语料：

- 英文内容 seller 文本
- 中文目标域 seller 文本

策略：

- 只使用无标签 seller/profile 文本
- 不引入 `EN-Gold` 的监督信号
- 作为增强实验，不替代无 DAPT 基线

输出：

- DAPT 后的 backbone checkpoint

验收条件：

- 与未适配 backbone 做严格对照
- 保持训练日志和参数可复现

### Step 7. 英文源域 seller-pair verification 模型训练

Step 7 在整篇方案中的角色不是一个孤立的“英文监督训练”环节，而是整条跨语言路线的源域训练步骤：基于 Step 3 seller profiles、Step 4/5 冻结后的 pair supervision，以及 Step 7 transfer-safe pair features 与 multilingual semantic scores，在英文内容训练池上训练 seller-pair 判别器，并以中文严格目标域 zero-shot 评测作为首要跨语言外部检验。

#### 7.1 Step 7 的目标与角色

Step 7 的正式任务是：

- 在 `en_content_train_pool` 上训练 seller-pair verification function
- 学习能够跨语言迁移的 seller-pair relation modeling 规律
- 为 Step 8 的 `zh_target_strict` zero-shot transfer 提供源域 baseline

这里的训练对象是 seller 对，而不是：

- item 分类
- 市场分类
- 单账号表征学习
- 英文账号与中文账号的一对一直接配对

换言之，Step 7 学到的是“给定两个 seller profile，判断其是否属于同一控制者”的 pairwise verification function；后续迁移到中文时，迁移的也是这一判别函数，而不是英汉账号之间的一一映射关系。

#### 7.2 输入数据与训练池定义

Step 7 的输入严格来自三部分。

1. Step 3 seller-level profiles  
   - `reports/step3_seller_profiles.en_content_train_pool.jsonl`
   - `reports/step3_seller_profiles.zh_target_strict.jsonl`
   - `reports/step3_seller_profiles.zh_target_aux.jsonl`
   - 其中 `profile_text` 是多语言语义编码与 reranker 的统一文本输入

2. Step 5 冻结后的 pair-level supervision  
   - `reports/step5_en_frozen_silver_labels.csv`
   - `reports/step5_zh_target_strict_frozen_silver_labels.csv`
   - 英文冻结标签提供 Step 7 的源域 `positive / negative` 监督
   - 中文严格目标域冻结标签不参与源域主训练，但为 Step 8 的固定 zero-shot 外部检验容器和 Step 9 的 few-shot 适配容器提供边界清楚的 pair labels / review labels

3. Step 7 pair-feature tables  
   - `reports/step7_pair_feature_preview.en_content_train_pool.csv`
   - `reports/step7_pair_feature_preview.zh_target_strict.csv`
   - `reports/step7_pair_feature_preview.zh_target_aux.csv`
   - 以及后续 semantic-enriched pair tables
   - 其中 relation features 来自 Step 3 seller profile 汇总字段与 Step 4 candidate-level 汇总字段，而不是直接把原始 item records 喂给训练器

训练池与目标池定义保持固定：

- 源域训练池：`en_content_train_pool`
- 严格 zero-shot 目标池：`zh_target_strict`
- 中文辅助池：`zh_target_aux`

其中 `zh_target_aux` 可以作为辅助语料、seller profile 补充和语义特征构造对象，但它不替代 `zh_target_strict`，也不进入严格中文 zero-shot benchmark。

当前已产出准备文件：

- `schema/step7_transfer_safe_pair_feature_schema.json`
- `schema/step7_semantic_model_policy.json`
- `schema/step7_training_policy.json`
- `docs/STEP7_TRANSFER_SAFE_PAIR_FEATURES.md`
- `scripts/step7_build_pair_feature_preview.py`
- `scripts/step7_download_models.py`
- `scripts/step7_build_semantic_pair_features.py`
- `scripts/step7_train_baseline_models.py`
- `scripts/step7_run_default_pipeline.py`
- `reports/step7_pair_feature_preview_summary.json`
- `reports/step7_pair_feature_preview.en_content_train_pool.csv`
- `reports/step7_pair_feature_preview.zh_target_strict.csv`
- `reports/step7_pair_feature_preview.zh_target_aux.csv`

当前预览结果：

- 英文 pair feature preview：`6623`
- 中文严格目标域 pair feature preview：`3793`
- 中文辅助域 pair feature preview：`580`
- 英文 `core_transfer_eligible`：`5553`
- 中文严格目标域 `core_transfer_eligible`：`3793`
- 中文辅助域 `core_transfer_eligible`：`580`

#### 7.3 Transfer-safe seller-pair feature construction

Step 7 的核心不是学习 seller 单侧的绝对原始属性，而是学习 seller-pair 的关系特征。当前 transfer-safe pair features 可分为三层。

第一层：relation-level structural features

- `same_market_raw_bool`
- `same_source_dataset_bool`
- `profile_category_jaccard`
- `shared_title_count_capped`
- `shared_description_count_capped`
- `shared_category_count_capped`

第二层：market-relative transfer-safe style/scale gaps

- `item_count_percentile_gap_abs`
- `price_median_percentile_gap_abs`
- `title_length_median_percentile_gap_abs`
- `description_length_median_percentile_gap_abs`
- `digit_ratio_mean_percentile_gap_abs`
- `punct_ratio_mean_percentile_gap_abs`
- `repeated_title_share_percentile_gap_abs`
- `repeated_description_share_percentile_gap_abs`
- `max_category_share_percentile_gap_abs`

这些 gap 特征都必须先在 `source_market_raw` 内做相对统计归一化，再构造成 pairwise 差异，避免 raw market-scale 数值直接主导跨语言分类器。

#### 7.4 Multilingual semantic feature extraction over seller profiles

Step 7 以 seller `profile_text` 为统一文本载体，针对 seller 对额外构造 multilingual semantic scores，包括：

- multilingual embedding cosine
- multilingual reranker score

其作用不是替代 relation features，而是为 seller-pair verification 提供跨语言语义对齐信号。当前实现支持多个 multilingual backbone / reranker 组合对照，包括：

- GTE multilingual
- BGE-M3
- `multilingual-e5-large`
- LaBSE
- `paraphrase-multilingual-mpnet-base-v2`

默认主结果采用 `default_gte` 组合，其余 backbone 仅作为稳健性比较与消融，不改变主线的数据边界与特征政策。

#### 7.5 Core / Identifier-augmented / EN-only 三类实验视图

Step 7 必须显式区分三类实验视图。

1. `core_zero_shot_default`
   - 只使用 `core_zero_shot_ready_now` + 默认 multilingual semantic feature set
   - 它是跨语言 zero-shot 的主结果模型
   - `zero_shot_safe = true`
   - 所有核心结论优先基于它报告

2. `identifier_augmented_default`
   - 在 core 视图之上加入：
     - `has_shared_contact_exact`
     - `has_shared_pgp_fingerprint`
     - `shared_contact_count_capped`
     - `shared_pgp_fingerprint_count_capped`
   - 它只能作为增强模型或消融模型
   - `zero_shot_safe = true`
   - 它不能被写成 zero-shot 主结论的唯一支撑

3. `en_only_ablation_default`
   - 在 core 视图之上加入 `uppercase_ratio_mean_percentile_gap_abs`
   - 它是英文侧边界对照
   - `zero_shot_safe = false`
   - 它不属于 zero-shot-safe 主模型，不能作为中文迁移主结果

贯穿上述三类视图的硬规则必须在 Step 7 正文中固定下来：

- 原始价格、原始长度、原始 uppercase 比例不能直接进入 zero-shot 主分类器
- 稀疏 `lexical_similarity` 只是候选召回证据，不是跨语言主判别分数；除非后续单独标准化并独立验证，否则不进入默认主分类视图
- identifier 特征不能构成 core transfer model 的基础；如进入增强模型训练，正样本中应对 `30%` 到 `50%` 的 identifier 信号做 masking/dropout
- `same_alias_identity_continuity` 对不能混入主 sockpuppet 监督池
- `EN-Gold` 只做严格 benchmark，不回流英文内容训练池

#### 7.6 LightGBM pairwise fusion and threshold selection

Step 7 当前实现采用 `LightGBM` fusion 作为 seller-pair binary classifier，而不是端到端神经网络直接输出最终标签。

当前训练协议固定为：

- 分类标签：`positive / negative`
- 训练池：`en_content_train_pool`
- 数据切分：`train / valid / test`
- 阈值选择：在 `valid` split 上按 `balanced_accuracy` 选择 threshold

在这一设定下，backbone 和 reranker 负责提供多语言语义分数，最终的二分类判别仍由 LightGBM 在 relation-level structural features、market-relative style/scale gaps 和 multilingual semantic scores 上完成融合。

#### 7.7 Step 7 输出及其在 Step 8 / Step 9 中的使用方式

Step 7 的直接输出包括：

- `core_zero_shot_default` 及其 backbone 对照模型
- `identifier_augmented_default`
- `en_only_ablation_default`
- 对应的验证集 / 测试集结果、预测文件与模型文件

但 Step 7 本身不是最终研究结论，而是英文源域 seller-pair verification baseline。

其在后续步骤中的作用必须写清楚：

- Step 8 使用该 baseline，特别是 `core_zero_shot_default`，在 `zh_target_strict` 上做 zero-shot transfer
- Step 9 在固定 `zh_target_strict` 测试集不变的前提下，评估少量中文监督对这一 baseline 的 few-shot adaptation 提升

因此，Step 7 是整个跨语言方案的 source-domain training step，而不是一个单独闭环的最终结论步骤。

验收条件：

- 训练集、验证集、测试集边界清楚
- 不发生 benchmark 泄漏
- core zero-shot 主模型只使用 transfer-safe 特征视图与默认 multilingual semantic feature set
- identifier-augmented 结果必须与 core 结果分开报告
- `en_only_ablation_default` 不得被当作中文 zero-shot 主结果

### Step 8. 中文零样本迁移

目标：

- 不使用中文训练标签，直接测试英文训练模型在中文目标域上的迁移能力

输入：

- Step 7 训练出的 `core_zero_shot_model`
- 中文银标签测试集

输出：

- 中文零样本评估结果
- 可选辅助结果：`identifier_augmented_model` 仅作补充或消融，不作为主结论

验收条件：

- 不允许在此阶段使用中文训练标签
- 固定测试集，不反复调参污染
- 主结果不得依赖 Step 4 的 raw sparse lexical similarity 阈值
- 主结果不得依赖英文域特有的 raw uppercase / raw length / raw price 特征

### Step 9. 中文少样本适配

目标：

- 评估少量中文标注对目标域识别性能的提升
- 在不污染 pure few-shot 基线的前提下，评估冻结强 zero-shot scorer 后的中文目标域 calibration

设置：

- 分为两个独立方法分支：
  - pure few-shot retraining
  - frozen-score calibration
- pure few-shot 保持：
  - 中文银标签训练子集比例：`10% / 20% / 50%`
  - 固定中文测试集
- calibration 分支保持：
  - 冻结 Step 7 scorer
  - `zh_train` 拟合校准器
  - `zh_valid` 仅做 threshold selection
  - `zh_test` 作为固定评测集

输出：

- pure few-shot 对照实验结果
- calibration 分支结果与 artifact
- `positive_pair_mixup` 训练期少数类正样本表征增强 artifact 和 `synthetic_train_only` 记录

验收条件：

- 测试集固定不变
- 每个比例至少跑统一随机种子和统一报告格式
- calibration 分支必须与 pure few-shot 分开记账，不覆盖既有 `step9_few_shot_summary.json`
- `positive_pair_mixup` 只能使用 sampled `zh_train` 中 `usable_for_core_transfer = 1` 且 `core_transfer_eligible = 1` 的 positive，不得使用 uncertain、closure-derived audit-only positive，不得写入 Step 5 frozen labels，不得进入 `zh_valid` / `zh_test`
- `synthetic_train_only` 只能解释为 training-only minority regularization，不能解释为新增标注数据

### Step 10. 模型与特征消融

目标：

- 明确性能来源

当前状态（历史归档快照，`2026-03-26`）：

- backbone / control-view 子集已完成并已同步审查
- 当前已固定的比较结果包括：
  - `core_zero_shot_default`
  - `core_zero_shot_bge_m3`
  - `core_zero_shot_multilingual_e5_large`
  - `core_zero_shot_labse`
  - `core_zero_shot_paraphrase_multilingual_mpnet`
  - `identifier_augmented_default`
  - `en_only_ablation_default`
- 协议默认 backbone 视图在该归档快照中保持为 `core_zero_shot_default`
- 但该页面本身已经不是当前主线结论来源；修复后的当前结果应以 `docs/PROJECT_PROGRESS.md` 为准
- 当前已编码的 feature-view ablation 已完成，而不是继续追加 backbone rerun
- 当前已编码并可直接运行的 feature-view ablation 包括：
  - `core_zero_shot_default_no_reranker`
  - `core_zero_shot_default_reranker_only`
  - `core_zero_shot_default_no_semantics`
  - `core_zero_shot_default_no_style_gap`
  - `core_zero_shot_default_no_structural`
- raw-vs-relative 对照已完成：
  - `core_zero_shot_default_raw_style_gap_control` 已按固定 Step 7 协议重跑
  - raw absolute gap 控制在英文与中文评测上都弱于 market-relative 主线
- 因此 market-relative 特征仍保持为当前协议默认主线，而不是回退到 raw absolute gap 控制

比较维度：

- `multilingual-e5-large`
- `BGE-M3`
- `gte-multilingual-base`
- `LaBSE`
- `paraphrase-multilingual-mpnet-base-v2`
- 有无 reranker
- 有无风格特征
- 有无结构特征
- 有无标识符特征
- `core_zero_shot_model` vs `identifier_augmented_model`
- raw absolute 特征 vs market-relative 特征

输出：

- 消融表
- backbone 对照表

验收条件：

- 只改变单个变量
- 统一数据切分和评测集

### Step 11. 中文候选子图聚类

目标：

- 在中文目标市场的候选召回子图上输出可追溯的疑似马甲簇

方法：

- 用 Step 7、Step 9 run-specific scorer，或 Step 9 calibration scorer 对 seller 对打分
- 基于阈值或边权建图
- 对 threshold-pass 边计算 `relation_reliability_score`，在 reciprocal-top-k 和 shared-neighbor pruning 之前过滤低可靠关系边
- reliability score 明确区分 semantic similarity 和 identity reliability：直接 seller-facing contact/PGP、稀有文本闭合、结构支持、风格一致性加分；boilerplate/template、semantic-topic-only 边扣分
- 做连通分量或社区发现

输出：

- 中文 seller 候选子图
- 疑似马甲簇列表
- 每条 scored pair 的 `relation_reliability_score`、`relation_reliability_components`、`edge_score_final`

验收条件：

- 阈值策略明确
- 聚类输出可以追溯到 pairwise 分数
- relation reliability 只能作为候选边过滤和误差分析控制，不能作为新的监督标签来源
- 必须明确当前覆盖的是 candidate-recalled subgraph，而不是 `5097` seller 的全对全穷举图

### Step 12. 统计分析与论文结果汇总

目标：

- 形成可进入论文的实验报告

指标：

- 召回阶段：`Recall@10`、`Recall@50`、`MRR`
- Pairwise 阶段：`PR-AUC`、`F1`、`Precision`、`Recall`
- 聚类阶段：`Pairwise F1`、`B-cubed F1`
- 统计：`95% bootstrap CI`

输出：

- 主结果表
- 消融表
- 迁移表
- 误差分析

验收条件：

- 每个主实验都有均值和区间
- 所有图表都能回溯到固定版本数据

当前执行状态（`2026-05-13`）：

- 已新增 Step 12 固定测试集统计稳健性审计：
  - policy: `schema/step12_statistical_robustness_policy.json`
  - runner: `scripts/step12_statistical_robustness_audit.py`
  - summary: `reports/step12_statistical_robustness_zh_test_20260513.json`
  - metrics: `reports/step12_statistical_robustness_model_metrics_20260513.csv`
  - paired comparisons: `reports/step12_statistical_robustness_paired_comparisons_20260513.csv`
- 审计保持固定 `zh_test = 106`，其中 `21` positive / `85` negative；不合并 `zh_train`、`zh_valid`、`zh_test`
- grouped bootstrap 的采样单位为 Step 5 `split_component_id`，当前 fixed test 中有 `39` 个 component，最大 component size 为 `14`
- 当前 clean E5 LR/L2 few-shot seed-mean 对 raw E5 的点估计略高，但不支持强显著表述：
  - ROC-AUC diff `+0.012325`，95% grouped CI `[-0.108240, 0.147650]`
  - AP diff `+0.019920`，95% grouped CI `[-0.251152, 0.326280]`
- 因此论文叙事应写为：Step 9 LR/L2 修复了 collapsed Step 7 fusion baseline，并提供有用的 graph-triage scorer；但相对 raw semantic E5/LaBSE/BGE 的提升在当前 fixed `zh_test` 上仍是 modest and uncertainty-bounded，而不是统计上稳健击败 raw semantic baseline。

## 7. 正式实验与步骤映射

### Exp-1 英文真标签严格基线

对应步骤：

- Step 1
- Step 2
- Step 7 的 benchmark 分支

模型：

- 弱结构特征 + LightGBM

目的：

- 建立无泄漏 benchmark

### Exp-2 英文内容主模型

对应步骤：

- Step 3
- Step 3 item-level identity/contact extraction check
- Step 4
- Step 5
- Step 7

模型：

- backbone + reranker + LightGBM
- 其中 `core_zero_shot_model` 为主结果，`identifier_augmented_model` 为增强/消融结果

目的：

- 训练真正依赖内容的马甲模型

### Exp-3 中文零样本迁移

对应步骤：

- Step 8

目的：

- 验证跨语言迁移是否成立

### Exp-4 中文少样本适配

对应步骤：

- Step 9

目的：

- 验证少量目标域标注的收益

### Exp-5 模型与特征消融

对应步骤：

- Step 10

目的：

- 明确性能来源与必要组件

### Exp-6 中文候选子图聚类

对应步骤：

- Step 11

目的：

- 输出中文市场疑似马甲簇

## 8. 贯穿全流程的硬性纪律

- 不允许 `EN-Gold` 回流英文内容训练池
- 不允许将所有含中文字符的行自动当成中文目标域
- 不允许未人工复核的银标签直接当最终真标签
- 不允许在测试集上反复试错后回写训练规则
- 不允许边做实验边改变 seller-profile schema
- 不允许混淆“辅助域”和“严格目标域”
- 不允许把 `same_alias_identity_continuity` 混入主 sockpuppet 监督池
- 不允许把 raw market-scale 价格、raw 长度、raw uppercase ratio 直接喂给零样本主分类器
- 不允许把 raw sparse lexical similarity 当作默认跨语言主分类特征
- 不允许只复核 identifier-heavy 候选后就冻结主监督标签集

## 9. 当前项目状态

当前已完成：

- Step 1
- Step 2
- Step 3
- Step 4
- Step 5 v2 targeted cleanup 与新的 freeze
- Step 5 v2 里程碑快照
- Step 5 v3 targeted cleanup 与新的 freeze
- Step 5 Chinese boundary expansion / positive-anchor expansion 与新的 freeze
- Step 5 English source-domain expansion / top-up 与新的 freeze
- Step 5 English item-level direct-identifier expansion 与新的 freeze
- Step 10 归档快照保留
- Step 9 / Step 11 运行器审计加固
- Step 7 在 `2026-04-21` previous freeze 上的 Linux rerun
- Step 9 few-shot 在 `2026-04-21` previous freeze 上的 Linux rerun
- Step 9 calibration 在 `2026-04-21` previous freeze 上的 Linux rerun
- Step 11 六个 previous-boundary 目标 summary 的 rerun
- Step 11 cluster-level audit on exactly the six previous-boundary summaries
- Step 5 label-stratified validation repair on `2026-04-22`
- Step 7 在 `2026-04-22` label-stratified active freeze 上的 Linux rerun

当前未完成：

- Step 9 / Step 11 在 `2026-04-22` label-stratified active freeze 上的 Linux rerun
- upstream raw/OCR/source-field acquisition if stronger proof-level same-controller claims are required
- fresh Step 11 current-summary manifest after the next current-boundary rerun

当前主线边界：

- `en_content_train_pool`
  - `945 reviewed / 476 supervision`
  - `train / valid / test = 280 / 77 / 119`
  - `train = 105 positive / 175 negative`
  - `valid = 30 positive / 47 negative`
  - `test = 44 positive / 75 negative`
- `zh_target_strict`
  - `1016 reviewed / 522 supervision`
  - `train / valid / test = 335 / 81 / 106`
  - `train = 61 positive / 274 negative`
  - `valid = 14 positive / 67 negative`
  - `test = 21 positive / 85 negative`

当前最需要区分的是：

- 上述 Step 5 边界已经是 active boundary
- 当前 Step 7 已经是 active boundary 结果
- 当前 Step 9 已经是 active boundary 结果
- 当前 Step 11 policy 已更新为 E5/LabSE/BGE residual candidate set，graph outputs 已按该 policy 重跑并同步
- `reports/` 里的 Step 11 输出已经按 `reports/step11_current_manifest_20260424.json` 清理；当前 audit 只认 manifest-retained summaries 和它们的 `output_paths`

当前 Step 7 主结论：

- `core_zero_shot_bge_m3` 不再是当前最强 clean zero-shot baseline
  - `roc_auc = 0.601681`
  - `average_precision = 0.448761`
  - `balanced_accuracy = 0.562465`
- `core_zero_shot_default_no_structural` 是当前 best clean ROC-AUC ablation
  - `roc_auc = 0.623529`
  - `average_precision = 0.287652`
  - `balanced_accuracy = 0.572269`
- `identifier_augmented_default` 是 operational/direct-identifier control
  - `roc_auc = 0.606443`
  - `average_precision = 0.418989`
  - `balanced_accuracy = 0.619888`

当前边界 Step 9 few-shot 主结论：

- clean few-shot graph-triage 主线切换为 `core_few_shot_multilingual_e5_large_lr_l2 / 50pct`
- 三个 seed 均明显超过 collapsed Step 7 fusion baseline：
  - seed `20260320`: ROC-AUC `0.819048`, AP `0.540482`, balanced accuracy `0.589356`
  - seed `20260321`: ROC-AUC `0.824650`, AP `0.541473`, balanced accuracy `0.583473`
  - seed `20260322`: ROC-AUC `0.811765`, AP `0.534180`, balanced accuracy `0.589356`
- 该结果支持 Step 9 修复 Step 7 fusion collapse，但相对 raw E5 semantic baseline 的提升较小，论文叙事必须保留 raw semantic 对照

当前边界 Step 9 calibration 主结论：

- calibration 不是 discovery mainline
- fixed `0.5` threshold 下三组 calibration 在 `zh_test` 均预测 `0` 个 positive
- `core_calibrated_default`: ROC-AUC `0.588235`, AP `0.448547`, balanced accuracy `0.500000`
- `core_calibrated_bge_m3`: ROC-AUC `0.601681`, AP `0.448761`, balanced accuracy `0.500000`
- `identifier_augmented_calibrated_default`: ROC-AUC `0.606443`, AP `0.418989`, balanced accuracy `0.500000`

这意味着：

- current-boundary clean few-shot graph-triage 主线是 E5-LR/L2 50pct
- BGE-M3 residual 100pct 和 LaBSE-LR/L2 100pct 是 clean controls
- calibration 是 sensitivity/control，不是主线

当前 Step 11 主结论：

- archive fallback 已经彻底关闭
- policy 已改为当前候选集：E5-LR/L2 50pct、BGE-M3 residual 100pct、LaBSE-LR/L2 100pct、zero-shot BGE anchor、identifier operational 100pct
- Step 11 graph outputs 已按当前 policy 重跑并在 `2026-04-24` 完成 manifest cleanup
- authoritative current manifest: `reports/step11_current_manifest_20260424.json`
- reports/ 中只保留 manifest 引用的当前 Step 11 输出；不要 glob 历史 `reports/step11_*` 文件

当前最重要的科研解释：

- clean LR/L2/residual few-shot 在当前边界已经显示 graph-triage 潜力，但 Step 11 只能生成候选簇，不能生成真值
- 当前 cluster-level audit 已确认 graph expansion 的证据解释：只有 direct identifier/contact core 能支持 same-controller claim
- LR/L2 主图中的 top clusters 仍需要按：
  - template-copy clique
  - mixed-evidence clique
  - topic/buyer clique
  这样的业务语义来审，而不是直接写成同一控制者
- direct same-controller claim still requires direct identifier/contact cores
- active audit output:
  - `reports/step11_cluster_level_audit.current_20260424.csv`
  - `reports/step11_cluster_level_audit.current_20260424.json`
- active audit decisions:
  - `same_controller_high_confidence = 7`
  - `same_controller_core_with_possible_expansion = 1`
  - `partial_anchor = 6`
  - `template_clone_not_controller = 66`
  - `semantic_topic_not_controller = 60`
  - `uncertain = 0`
- strict direct-identity recheck supersedes the mechanical same-controller counts for paper claims:
  - previous-boundary strict direct recheck artifacts were removed from `reports/` during cleanup and remain historical text evidence only
  - current-boundary paper claims should rely on `reports/step11_cluster_level_audit.current_20260424.csv/json`
  - no whole Step 11 cluster should be claimed as a same-controller ring unless the current audit marks a direct identifier/contact anchored core
- the stable LR/L2 candidate `121394 || 435064 || 95895` is downgraded, not promoted, because its evidence is reviewed-uncertain external URL/product-context overlap

当前 paper-targeted Step 5 expansion check:

- policy: `schema/step5_paper_targeted_expansion_policy.json`
- queue: `reports/step5_zh_target_strict_paper_targeted_expansion_queue.20260422.csv`
- summary: `reports/step5_paper_targeted_expansion_queue_summary.20260422.json`
- conservative review: `reports/step5_paper_targeted_expansion_codex_review_summary.20260422.json`
- selected rows: `20`
- reviewed labels: `20 uncertain`, `0 positive`, `0 negative`
- unreviewed non-URL shared direct-contact pairs found: `0`

这意味着：

- 使用 Step 11 反向挖候选是可行的，但只能作为 triage
- 当前不能把新队列应用到 Step 5 supervision，因为没有新增高置信 positive/negative
- 该检查之后已经上移到原始 item-level identity/contact extraction；结果见下一节

当前 Step 3 item-level identity extraction check:

- parser/schema:
  - `scripts/step3_build_seller_profiles.py`
  - `schema/step3_seller_profile_schema.json`
- new evidence files:
  - `reports/step3_item_identity_signals.en_content_train_pool.csv`
  - `reports/step3_item_identity_signals.zh_target_strict.csv`
  - `reports/step3_item_identity_signals.zh_target_aux.csv`
- Step 5 queue route:
  - `schema/step5_item_identity_expansion_policy.json`
  - `scripts/step5_build_item_identity_expansion_queue.py`
  - `reports/step5_zh_target_strict_item_identity_expansion_queue.20260422.csv`
  - `reports/step5_item_identity_expansion_queue_summary.20260422.json`
  - `reports/step5_item_identity_expansion_codex_review_summary.20260422.json`
- Step 3 rerun acceptance checks still pass:
  - seller counts match Step 2
  - item counts match Step 2
  - all profile text is non-empty
- Chinese strict extraction:
  - item-level identity signals: `3,785`
  - direct-identity-eligible signals: `1,477`
  - shared seller-facing direct token groups: `39`
  - candidate rows surviving frozen/reviewed exclusion: `0`
  - skipped shared-token pairs: `43 frozen_pair`

这意味着：

- raw item-level parser 升级已经完成并保留上下文证据
- 当前中文 strict 原始文本中的共享 seller-facing direct-token pair 已经被现有 Step 5 frozen/reviewed boundary 覆盖
- 中文 item-level 检查本身不触发 Step 5 freeze，也不触发 Step 7/9/11 重跑
- 若论文还需要更多 proof-level positives，下一步不是再扩大 Step 11 图，而是补充新的 raw/OCR/source fields 或外部可核验证据

当前 English item-level identity expansion:

- policy: `schema/step5_en_item_identity_expansion_policy.json`
- queue: `reports/step5_en_item_identity_expansion_queue.20260422.csv`
- queue summary: `reports/step5_en_item_identity_expansion_queue_summary.20260422.json`
- review summary: `reports/step5_en_item_identity_expansion_codex_review_summary.20260422.json`
- apply summary: `reports/step5_en_item_identity_expansion_apply_summary.20260422.json`
- selected rows: `36`
- reviewed labels: `35 positive / 1 negative`
- Step 4 candidates appended: `22`
- existing Step 4 candidates updated: `14`
- final English supervision: `476`
- final English primary positive supervision: `179`
- seller/alias split overlap: `0`

这意味着：

- 英文侧已经更接近“大源域、小目标域”的迁移叙事，不再只是略高于最低样本阈值
- 新增样本来自 seller-facing direct identifiers，而不是随机 easy negatives
- 该边界已经被 2026-04-23 English valid/test top-up supersede

当前 English valid/test top-up:

- direct policy: `schema/step5_en_item_identity_expansion_valid_test_topup_policy.json`
- direct queue: `reports/step5_en_item_identity_expansion_valid_test_topup_queue.20260423.csv`
- direct review summary: `reports/step5_en_item_identity_expansion_valid_test_topup_codex_review_summary.20260423.json`
- direct apply summary: `reports/step5_en_item_identity_expansion_valid_test_topup_apply_summary.20260423.json`
- direct selected rows: `46`
- direct labels: `30 positive / 16 negative`
- Step 4 candidates appended: `38`
- source top-up policy: `schema/step5_en_source_expansion_valid_test_topup_policy.json`
- source top-up queue: `reports/step5_en_source_expansion_valid_test_topup_queue.20260423.csv`
- source top-up review summary: `reports/step5_en_source_expansion_valid_test_topup_codex_review_summary.20260423.json`
- source top-up apply summary: `reports/step5_en_source_expansion_valid_test_topup_apply_summary.20260423.json`
- source top-up selected rows: `330`
- source top-up labels: `212 negative / 118 uncertain / 0 positive`
- active English supervision: `734`
- active English split:
  - train `401 = 116 positive / 285 negative`
  - valid `152 = 42 positive / 110 negative`
  - test `181 = 51 positive / 130 negative`
- active Chinese strict split is unchanged:
  - train `335 = 61 positive / 274 negative`
  - valid `81 = 14 positive / 67 negative`
  - test `106 = 21 positive / 85 negative`
- seller/alias split overlap: `0`
- Step 5 coverage checks: pass

这意味着：

- 英文 valid/test 已经被抬到更稳规模，Step 7 不应再触发 `valid_row_count <= 100` 的 small-validation guard
- 本轮没有盲扩中文；中文 proof-level positive 稀缺结论不变
- Step 7 和 Step 9 已在该边界上重跑并同步；Step 11 仍需重跑后才能更新当前 graph/cluster 结论

上一轮 Step 5 label-stratified refreeze 修复：

- root cause: English valid split had `27/27` positives from identifier strata and `53/53` negatives from `semantic_structural`, making Step 7 validation shortcut-separable
- fix: `scripts/step5_freeze_silver_labels.py` now balances seller-component split assignment by `review_label x review_stratum`
- policy: `schema/step5_freeze_policy.json` now blocks English freezes where train/valid/test miss available hard positive or hard negative strata
- then-active English split:
  - train `280 = 105 positive / 175 negative`
  - valid `77 = 30 positive / 47 negative`
  - test `119 = 44 positive / 75 negative`
- active Chinese strict split:
  - train `335 = 61 positive / 274 negative`
  - valid `81 = 14 positive / 67 negative`
  - test `106 = 21 positive / 85 negative`
- this boundary has now been superseded by the 2026-04-23 English valid/test top-up

## 10. 当前推荐执行顺序

从现在开始，建议按以下顺序继续：

1. 固定当前 `2026-04-23` English valid/test top-up Step 5 active freeze，先不要继续扩 supervision boundary。
2. 当前 Step 7 和 Step 9 已经在该边界上同步；Step 9 的 clean graph-triage candidate 更新为 `core_few_shot_multilingual_e5_large_lr_l2 / 50pct`，并保留 `core_few_shot_bge_m3_residual_lr / 100pct`、`core_few_shot_labse_lr_l2 / 100pct` 作为 clean controls。
3. 同步 `2026-05-14` 的 Step 9 / Step 11 code 和 policy 到 Linux，先重跑 `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup`，评价它是否稳定改善 AUC/AP 和 Step 12 paired comparison。
4. 再重跑 Step 11 当前候选集，使 `relation_reliability_filter` 进入 scored-pair 和 cluster summaries。
5. Step 11 当前候选集包括：E5-LR/L2 50pct、E5-LR/L2 positive-pair mixup、BGE-M3 residual 100pct、LaBSE-LR/L2 100pct、`core_zero_shot_bge_m3` conservative anchor、`identifier_augmented_few_shot_default_lr_l2 / 100pct` operational control。
6. Step 11 同步回来后，用显式 current-summary manifest 做 cluster audit；不能恢复全目录 glob。
7. 在新 Step 11 audit 前，strict direct recheck CSV 只能作为上一边界 identity-claim 参考。
8. 不要应用 `reports/step5_zh_target_strict_paper_targeted_expansion_queue.20260422.csv` 到 Step 5 supervision；本轮全是 uncertain。
9. 在论文叙事中分开三条线：clean E5/LabSE/BGE residual few-shot graph triage、zero-shot BGE anchor、identifier-augmented operational control；同时报告 raw E5/LaBSE/BGE semantic baseline，避免夸大 few-shot 相对最强 raw semantic baseline 的提升。
10. 若继续追中文 proof-level positives，优先获取/导入新的 raw/OCR/source fields 或外部可核验证据；当前中文 Step 3 item-level text extraction 已未发现新增未审 direct-token pair。
