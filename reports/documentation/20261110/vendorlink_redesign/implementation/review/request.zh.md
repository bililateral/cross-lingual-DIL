# VendorLink账号监督CE适配：定向实现审查请求

本次是已审方案的实际实现审查。用户最新明确“实现完成就提交外审”，请按当前代码和原生证据独立审查，不重复设计已经关闭的旧路线。请求网页GPT 6 Pro；请分别说明可见选择与自身身份，不推定不可验证的后台模型。

## 背景与科学任务

目标：保留完整H的文本、多记录身份关联结构，在暗网卖家应用中先学好、出现具有实际幅度的顺序遗忘，再检验Full的回放和出生评分保持是否让旧域最终更好，同时新域仍学得好，并与未专门缓解遗忘的VendorLink分类/集合方法比较。

当前共同任务沿原论文账号监督层级。AlphaBay→Dream→Agora，546/157/875个账号；每账号固定12 fit/4 gallery/2 dev query/2 confirm query，复用既有资格与已建角色。H用Q2/G4多记录评分；V_CLS用当前encoder和各旧市场保留分类头，V_SET用同一V encoder集合评分；每域同一全候选池。跨账号发现仅无真值候选应用输出，不宣称真实马甲同控准确率。当前实现核验只用手写输入，没有消费真实fit/gallery/dev/confirm载荷。

旧20261109 P六条已结束：H_SEQ三个种子遗忘均值4.3657pp，但AlphaBay/Dream初学较V_CLS分别低10.7518/7.8556pp；原学习门及5pp门失败，Full/ER/C未启动。用户后设新目标是遗忘3.5pp、初学相对V_CLS均值低不超过2pp/逐seed低不超过5pp。原失败、原5pp门及71450秒费用保持，不追改成通过。旧权重和专用脚本已按明确许可清理；只从Git恢复必要函数用于新入口，不恢复旧临时OOM控制器。

## 已审且固定的设计

完整方案为包中`reports/documentation/20261110/vendorlink_redesign/proposal.zh.md`。本会话原方案外审报告已在包内保存，主审见`review/primary_disposition.zh.md`；已闭合R01原表示必须除√2、R02辅助AdamW精确定义。用户针对完整时限预约不足要求修正，总预算80→88h（training72→80，GPU4、aux4不变），不增任务/候选/seed或改变约63—78h预计实耗。原外审上沿D6＋P19＋4×46/6＋末两M各10h预约=75h40m，80h训练桶覆盖；超出投影仍停，不保证所有最坏时限组合完成。

唯一方法增量：H1原BERT-base-cased768，title/description各256，sum-last4 masked mean、每通道归一化、拼接/√2得1536维。当前市场K类CE在同一表示上按记录均值加权γ，γ仅0/1；原H记录对BCE/rank/hard、自然/重组、多记录评分、回放和出生教师保持。CE仅用于当前域，辅助头不参与最终评分，不把CE本身称新抗遗忘机制。

辅助Linear每域重建，初始化隔离原Torch随机流；辅助AdamW固定lr3e-4、betas(.9,.999)、eps1e-8、weight_decay.01对W/b均适用、amsgrad=False。沿原阶段学习率调度。原encoder/pair优化状态跨域保留，CE与H梯度合并后共同裁剪一次，再各自optimizer更新；encoder不可被两个optimizer重复更新。

D=H_base/H_CE/V0/V1×AlphaBay/Dream八个冷启动，seed41/40epoch。V以A/B V_CLS均值选，差≤.001优先V0；H_CE两域均≥.70且相对入选V每域最多低2pp，失败正常STOP_ADAPTATION。D严门可能误停，不等于否定所有CE。

P=H_CE_SEQ与V×51/52/53六条ABC；先同时检查H初学绝对能力、相对V2pp/5pp、新域增量与3.5pp遗忘/跨seed方向。M=同配置ER/Full×同三seed六条；同时检查Full初学/新域保护、Full−SEQ的O提升及F改善、Full最终旧域对两V的优势；C不新训，复用12端点的一次独立文本确认。组件Full−ER需M/C同时过门，不用系统PASS替代组件证明。完整数值以方案及policy为准。O/F改善相关，不是两个独立统计证据。

新运行最多2worker，H＋H或H＋V，最多一个V。用户明确取消人为allocator配额及固定GiB整PID终止；H5GiB/V17GiB是派发预留，仍查物理资源/已承诺占用。单worker真实OOM退出并退休，健康worker继续当前阶段独立任务，失败不自动重跑，无健康worker不自动补位，缺项不开放整批效果/下游；非OOM、共享故障按各自边界停止。正式启动仍未获本轮授权；`gate.json`保持approved=false，当前外审未回收所以review_closed=false。

## 当前实现及原生证据

当前13份运行源码：六个`step35_vendorlink_account_*`模块，复用step29 data/methods/run、step28_formal_queue、step30 policy/io/control。`step30...Runner`只增加默认关闭的continue_after_oom及按job预留；新入口开启，未覆盖Linux中文活动workspace或旧冻结源码。

请从实际入口追踪：policy.tasks→control.make_jobs/worker→run.execute→methods.backward_h/update_account→evaluate.score/analyze→policy.decide/pipeline。特别检查D Dream是否真正冷训/评价Dream；旧`DOMAINS[:stages]`和旧单A按max(V_CLS,V_SET)选参已替换。所有当前/旧域、确认开放及完整性条件须进入实际接线。

Linux既有py310原生证据：CPU01四项通过（Git cc8dd742）；唯一GPU01通过（Git1b3caa2c）；主审发现CPU回执格式/旧permit不一致，修正后只运行CPU02一个受影响入口检查（Gitf7e1f51f）。两次CPU合计5项零失败/错误，分别4.6817/2.1886秒；GPU17.5477秒，没有重复GPU。CPU01四项和CPU02单项的源文件从各自Git精确恢复到包的`evidence_sources/`，其来源清单保留实测字节身份；运行目录一次性脚本已按纪律退役，正式入口不依赖它们。

CPU证据包括独立普通autograd＋手写CE参照、γ0原H更新逐位等价、γ1改变encoder梯度、辅助重建与原optimizer保留、完整状态恢复、手写Dream实际冷训与Q2/G4评分、门失败/缺seed、80h预约、实际CPU子进程经真实Runner的OOM退休/健康续领/失败不重入/下游阻断；CPU02实际permit正反例及main的CUDA OOM/普通错误分类。故障注入子进程是CPU模拟，不冒称真的耗尽GPU。

GPU证据：真实BERT最大44记录×两通道256token、946记录对、875类，4次手写更新覆盖γ0/γ1与保存恢复下一步；原pair/history生效，CE实际改变encoder更新，恢复后下一步模型和日志逐位一致。整PID采样峰值3.973GiB，RAM3.488GiB；未用GPU硬cap，不冒称双路长训实测。原V科学路径未改，复用成功训练证据，不另做V资格。

当前本轮费用training0/GPU18/aux128秒，总146秒；aux包括CPU8及Linux核对准备收尾保守120秒，非全部测得科研运行时间；未来campaign另沿Ledger记一次30秒尾部预留。旧费用不借用。当前gate、部署/回传/清理证据在implementation目录；没有正式campaign或监听。

## 请优先回答

1. 当前CE是否真正与原H共享表示并进入encoder更新？γ0是否维持对照，历史CE、优化器参数交叠、裁剪、域切换、RNG或恢复是否有会改变科研解释的漏接？
2. D/P/M/C的真实入口、数据可见性、V选参、seed、比较门及Full/ER组件结论是否与合同一致？有无形式存在但不会生效的关键操作？
3. 用户OOM规则是否经真实入口落实，失败类型是否恰当隔离，预算/资源预留/无GPU硬cap是否存在会导致当前任务错误开放、违规重试或错误终止的具体问题？
4. 现有旧证据＋本次CPU/GPU增量是否足够；若不足，只指出最小受影响证据，不要求无依据地重复完整资格、双路GPU、所有方法/种子或科学实验。
5. 列出可定位的科学阻断、复现缺口、合理后续建议及超范围建议；每个必修请指文件/函数、具体可达触发与影响、最小修法。不要把工程核验通过写成已经追平V或验证抗遗忘，也不要新增科学选择/训练预算来“实现预期结果”。

请给明确结论：实现可在正式授权后进入D，还是存在必须先闭合的具体问题。以当前13运行文件为准，不把evidence_sources的历史测试源当活动入口。方案定义已关闭，只有发现直接矛盾或真实新缺口才回查，不重开无关历史。包内没有真实账号、正文、标签、逐查询预测、权重、Memory或凭据；无需索取这些材料来审本次增量。
