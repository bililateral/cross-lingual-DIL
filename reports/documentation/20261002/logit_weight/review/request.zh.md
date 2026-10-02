请使用网页GPT 6 Pro实际审查附件中的匹配ER0.25的SEQ／ER／LOGIT合同及最小实现，实际打开ZIP并核对source_inventory.json及本轮logit_weight/freeze.json。说明可见模型身份；没有独立后端标识则明确不能认证，不把自述当证明。

用户已在decision.json逐字确认完整方案：先让独立λ=0.1正式开始（已完成），再着手本次；本次复用SEQ与ER0.25，只新增LOGIT＝L_current＋0.25 L_history＋0.5 MSE，MSE不再乘0.25。从原三首域model/Adam/RNG/LOGIT缓存/首映射续训，s0、ABC/BCA/CAB、1728更新、3456群梯度呈现、6终点，18新+45既有指标/计数。主比较LOGIT−匹配ER沿原23项，完整报告ER−SEQ、LOGIT−SEQ，结果好坏均结束；不自动替换ER、不调λ/MSE/seed。单GPU＋单CPU估6—8h，硬12h/16GiB；新train/valid各一次，不读test/owners。外审及主审后直接必要CPU与正式，等0.1释放资源后部署，不重复申请同范围启动。

请从用户问题和公式独立追踪实际计算，而非只确认executor policy、代码与测试一致。重点完整历史BCE/rank/0.5hard乘0.25且MSE单独0.5、历史train forward共用图/无额外forward、target无梯度、累加后一次clip/Adam；首域原LOGIT目标与共同状态绑定；新分支第二域末eval只产生新保留群target，旧幸存target不刷新，旧LOGIT1第二域target不复用；6群/1MiB及当前/历史群、dropout/Adam配对；新study贯穿全部入口；六端点24分数及一次valid门；三组严格配对、18套新结果先保存后统计、45套既有结果不重读标签。23项、22指标（AP≠梯形PR-AUC）、O/N/Z/F/G与5000实际域整群条件bootstrap保持，不将三顺序当三种子。

本次只改原四ER模块、同一测试文件、两个Bash，新增policy/合同/用户决定；implementation.diff为相对本次修改前646ed9e2的实际差异。原首轮18来源不变。原low/weight freeze是历史身份，当前34来源以本轮freeze为准；Linux正在运行的low仍用旧31字节，本地新代码尚未同步覆盖。请读附带直接相关历史外审全文及处置，旧已关闭问题不重开；原权重、原始文本/标签/缓存正文不在包内。

请在网页可用Linux CPU环境实际执行 `PYTHONPATH=scripts:tests python -B -m unittest test_step28_er_weight_contracts -v`（当前28个不同用例），两个CLI的--help、两个Bash的bash -n，并作必要独立参考。新六例包括原LOGIT1更新逐值对齐、独立三分量梯度/裁剪/Adam及末步LR、无效参考拒绝、完整六点门错误系数/来源/刷新拒绝、实际微型288步与checkpoint/缓存，以及18新+45复用保存后故障恢复。请独立核对新的三组端点/判据，关注相对ER正但相对SEQ仍可能负，不承诺效果。没有安装依赖或模型的必要，不要求全仓回归。

Windows仅编辑/源码审读、JSON及哈希/差异检查；未运行科研脚本、AST或unittest。本阶段网页/项目CPU/正式未执行，本次结果未知。可使用包内已有小型指标/计数和手写输入；不要连接项目Linux、运行原生BGE入口或execute、读正式文本/标签/缓存、加载正式模型、读test/owners、安装环境。项目原生CPU四次真实手写更新及MSE独立梯度探针将在本地主审后另取得，不把未运行说成通过，也不因遵守先后顺序而要求提前跑正式。

请对实际问题区分SCIENTIFIC_BLOCKER、REPRODUCIBILITY_DEFECT、OUT_OF_SCOPE_OVERDESIGN，说明受影响主张、代码位置、独立证据、最小修复及未验证范围。不要增加系统/网络加固、Linux Git、状态机、密钥仪式、成功线或重复审批。返回完整中文REVIEW、所有独立源码、实际命令/环境/起止/退出码、原stdout/stderr、用例/数值证据、全部失败及修订、逐文件大小/SHA清单和完整下载ZIP；保留收到的原提交ZIP身份，不能静默修改原项目。一次给出完整证据，不只回复摘要或“无阻断”。
