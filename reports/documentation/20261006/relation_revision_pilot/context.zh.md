# 新目标原生准入与正式接入外审上下文

2026-10-06，Asia/Shanghai。本次只审新增正式接入、合同、原生证据及上一轮K1关闭；未变数学和已关闭旧流程复用。提供完整历史材料是供具体问题回查，不要求重审全部历史或制造新实验。

## 当前授权与交付

用户要求“把新目标的 BGE/GPU 原生一步和正式效果也验证一下，通过后开始正式训练”。主执行者说明效果必须训练后才可验证，提出原生一步＋接入核验→一次固定候选训练→完整评价。用户对具体预算、一次访问、固定判据答复“按上述范围推进”。完整范围见根目录 docs/SELLER_ALIAS_RELATION_REVISION_PILOT.zh.md。两项手写核验已执行且通过；正式训练尚未启动，正式文本和标签尚未读取，正式gate尚未签发，外审／主审通过是实际前提。

新许可：原生GPU0＋单CPU600秒、28GiB reserved／64GiB RSS／128MiB证据；手写接入单CPU600秒／2GiB RSS／16MiB证据。一次正式固定B+Q+0.1R、d32、epsilon=.001、s0三顺序2592更新九端点，24小时／24GiB；train/valid各一次尝试，完整盲门后开放valid，不碰test/owners。对LOGIT0.1六项观察继续条件不变，无调参、失败自动重试或额外效果预试验。

## 研究背景和应保留的判断

目标是在六完整历史群及全部附属状态≤1MiB下持续学习新卖家群，保持旧分布未见卖家的排序与基础识别。当前为中文合成受控表达数据；28账号、378边、20正358负，每查询27候选且有正。每域48 fit／12 calibration／20已开发valid。三顺序是同一s0，不是三种子或新独立确认集。

旧关系目标固定点有效但对LOGIT0.1六项继续条件0/6，O/N/Z MAP观察差均负且区间低于0；首阶段已弱、CAB局部正结果保留，不能把差异唯一归因于历史迁移。旧九权重已清理、LOGIT0.1末端权重也已清理；本次仅复用已开放基线指标矩阵，不加载它们。group-meta关闭，risk负结果不改。

本次核心保持累计H/b/c/N平方统计及少量缓存估计迁移，目标变更已经上一轮审查。Q仍只精确摘要固定表示时旧平方代理，不是新B/R充分统计。Q对大正确间隔的回拉、Q/R冲突、运输与岭误差、缓存外泛化均未消除；增加R不证明MAP保证。R及三项相加本身不是创新，累计信息超出缓存的独立贡献尚未建立。本轮不扩大消融。

## 本次实际变化及复用路径

新增六文件：新合同、schema/step28_relation_revision_policy.json、scripts/step28_relation_revision_run.py、scripts/step28_relation_revision_admission.py、scripts/run_step28_relation_revision_linux_20261006.sh、tests/test_step28_relation_revision_run.py。生产修订核心和旧正式runner均未改。

新runner用importlib把旧runner装入独立模块，明确绑定method.update=revision.update，其余模型／Memory／reference复用。POLICY、端点名、来源闭包及validate_gate换成本轮；旧模块本身不被改写。执行实际仍经过旧train→checkpoint→blind_gate→collect→finalize→complete，恢复只保留旧已审保存矩阵收尾路径。请核对函数全局解析、调用实际新优化核、来源准入和端点／对手映射，不能仅据组件存在判通过。通用内部状态及比较臂名relation继续复用，端点实际名relation_revision、policy study和源码身份区分本轮；不混淆旧结果。

新gate要求APPROVED_RELATION_REVISION_PILOT、精确31份来源、job/预算/监督次数、外审主审NO_OPEN_BLOCKERS及新native/cpu模式与来源。核验部署实际33文件包含手写测试直接依赖；31份正式来源闭包清单另见CPU结果source_files，两个数职责不同。此前旧GPU原生26秒证据只作历史，未移用。

CPU两用例中，第一复用旧完整手写编排夹具，但替换其实际run模块并计数新optimization_step：2592次当前、1728次历史；检查B分解及Q+.1R、真实微型保存扰动恢复、九端点／28指标套、盲门及旧收尾恢复边界。人工基线只证明评价接线，不证明真实效果。第二以明确fixture_only gate检查身份门分支，不充当真实GPU证据。

GPU手写当前和不同合法历史标签各448段×256tokens，既有BGE、真实新优化核、首次Adam分配。Q、R、合项分别对首层query.weight、关系头权重及评分w求探针梯度；三个探针不是全参数梯度一致性证明。后者复用上一轮已关闭K1全参数裁剪前比较和四故障识别。这里额外三次VJP计入核验预算，正式训练无该诊断。统计为count1二次内核夹具，不冒称完整训练首域；原生仅手写输入、不保存权重。

## 本次真实Linux证据

独立目录 reports/documentation/20261006/relation_revision_pilot/workspace，既有py310／CPU0；仅模型软链接，无正式数据／基线入口。核验前GPU0无计算作业、可用32063MiB，CPU0空闲；两项各一次，无失败／错误／跳过／重试。

CPU开始13:14:28，2用例62.563秒，外层64.35秒、exit0；采样RSS768090112B，time最大RSS863004KiB。GPU开始13:17:53，外层35.20秒、exit0，内部更新含诊断24.306837秒；reserved18589155328B（约17.31GiB）、allocated18316646400B，采样RSS2820386816B、time最大RSS2840876KiB。不同RSS口径分别保留。

参考／活特征最大差0，三类参数实际改变，clip/step各一次，Adam step1；首层前向钩子784次，包含诊断和检查点重算。FP32参数396、梯度394、Adam moments788；不是所有参数都有梯度的宣称，沿用原dtype检查。Q/R/合项的三探针梯度有限非零，详细原值、dtype及全部来源见 evidence/reports/native/result.json。核验状态PASS_HANDWRITTEN_ONLY，不能写作效果通过。

回传原ZIP10916字节，SHA256 e17eb3073b24af5b203f940828a2813c684e7cc741c5b74fe744dc747f18e917；9载荷大小/SHA匹配；CPU31正式来源与Windows一致。原始console、wrapper、preflight及manifest均随包。

## 完整上下文入口

1. 根目录当前合同／policy／全部正式来源和新测试，以根manifest为本次身份。
2. reports/documentation/20261006/relation_revision_pilot/evidence：本次原生CPU/GPU、资源和来源原件。
3. reports/documentation/20261006/relation_revision：上轮完整外审原文、主审、K1源码及Linux原证据，K1已另授权完成一次2.36秒；漏R、断R、错误系数、误用当前标签均识别。生产核心未改。
4. history/relation_revision_review.zip：上一轮完整305文件上下文原包，5257725B，SHA480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5；内含原设计、实现、P1/P2/P3、旧结果、基线、纪律／用户决定及各轮原始外审证据。解压后按其context导航，需要核对具体未变函数时回查。旧待授权、待执行和未原生文字按制作时快照，不取代本页。
5. 根目录当前纪律、交接、讨论供权限和历史定位；最新本轮授权以上文和新合同为准。上传包不包含SSH凭据、正式原始文本标签、缓存或训练权重。

请只提出会影响本轮有效执行和必要交付的具体缺口，明确证据、触发条件和最小处置；不要把尚未证明效果／创新、未做额外消融或泛化系统加固列为当前实施错误。主执行者独立负责裁决和必要复核。
