# 固定关系目标记忆结果：资源、故障账与基线资格专项

2026-10-06。审查包：`relation_memory_result_review`。已完整阅读本次 `review/context.zh.md` 与 `request.zh.md`，检查本次直接运行记录、冻结正式入口及必要准入依赖；另由子审阅读context第2组基线合同、旧结果审查和直接资格证据。没有执行正式运行、PyTorch、BGE、GPU、原始数据或标签读取，没有检索新文献。

**专项结论：未发现阻断本次结果关闭的资源／故障账／基线资格错误。** 现有材料支持这一次固定配置在约定Linux资源内完整完成；该结论不等于创新或效果成立，也不等于未来任意输入／环境的运行保证。先前入口审查的P1/P2/P3不得在已经修复并完成的这次结果中继续当作未关闭阻断。

## 1. 正式资源与耗时：直接凭据一致

以下路径以 `reports/seller_alias_continual/20261006/relation_result/` 为前缀。

| 核查项 | 直接凭据 | 独立核对与判断 |
|---|---|---|
| 正式墙钟 | `job.wrapper.txt:1–4`：2026-10-05 15:06:32+08:00至18:57:22+08:00，exit0 | 两个带时区时间戳差恰为13,850秒，即3小时50分50秒。 |
| 程序内计时 | `job/completion.json:1` → budget.elapsed_seconds=13849.124231643276；`job/resource.json:1`=13849.125963777304 | 与包装差小于1秒，低于86,400秒；resource略晚于completion符合完成后持久化顺序。 |
| CUDA峰值 | 同上：allocated=8,465,105,408；reserved=9,367,977,984字节 | 分别7.883744与8.724609GiB；reserved低于28GiB。 |
| RSS观察峰值 | 同上：6,900,240,384字节 | 6.426350GiB，低于64GiB。是程序采样观察值，报告没有冒称OS绝对峰值。 |
| 产物观察峰值 | 同上：15,679,024,129字节 | 14.602229GiB，低于24GiB；冻结入口包含job及控制台、wrapper，并在full+infer共存删除前强制计量。 |
| 更新进度 | `job.console.txt`108条JSON记录 | 精确覆盖ABC/BCA/CAB各3阶段，每24步一次直到288；所有时间和四类峰值单调，完成快照不低于任一进度记录。 |
| 完成资格 | `job/completion.json:1`：COMPLETE_RELATION_FIXED_POINT、2592、worth_matched_replay=false | `evaluation.json`自身仅为STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION；有效性来自受预算约束的completion，符合修后合同。 |
| 盲门及访问 | `job/before_valid.json:1`、`job/access.json:1`、completion | 盲门9点且valid0，结束train1/valid1/heldout0/owners0；before_valid引用的manifest大小/SHA在网页实算一致。 |
| 故障／恢复 | 本次原始inventory及提交的job记录 | 未见failure或recovery路径，完整日志、exit0、completion相互支持本次正常完成，不能将旧恢复入口缺口当成本次发生过的失败。 |

**性质：实际结果的资源可行性证据。影响：可以把这一次从“最大形状一步准入”推进到“固定正式流程完成”。最小处理：保留当前观察峰值和适用范围措辞，无需追加运行。**

正式13,850秒覆盖该Bash启动后的训练、固化、评分、保存恢复、校准与评价；它不包括此前26秒原生准入、CPU手写核验、部署准备、翌日回传和1.022秒独立结果核验，也不应被说成整个研究工作的全部投入。主报告第13、36、108行已将这些范围分开，没有发现夸大。

## 2. 冻结入口与准入修复相符

以冻结 `source/scripts/step28_relation_memory_run.py` 为准：

- **预算及快照，第73–140行：** 从包装开始时间补入重库导入和准入消耗；记录累计墙钟、RSS/CUDA和产物峰值；snapshot不再调用预算断言；控制台路径使用job.name字符串追加；只容忍本次`run/work`正常消失临时文件的FileNotFoundError。
- **失败停止，第143–183行：** watchdog的记账置于try/finally，finally退出；普通异常和可处理TERM保存资源及失败类别，POSTPROCESS_IO_ONLY还需资源检查通过。无需把本次未触发的每个异常路径重新全部实测。
- **检查点，第219–245行：** 完整及推理状态写盘前分别检查reserve；实际恢复、完整盲分数重放通过后，第244行强制计量重叠峰值，才删除已核验临时完整状态。
- **完成资格，第454–495行：** 纯统计输出不授予有效完成；`complete`在统计后检查预算再生成completion，并在尾部检查失败时删掉completion。
- **受限恢复，第498–527、565–569行：** 独立CLI现在绑定原gate/source/job，只接受POSTPROCESS_IO_ONLY；继承原预算和wrapper更晚的耗时，限定单核与线程；恢复在进行中会替换旧失败状态，不调用训练或标签解析。这次材料没有该恢复被触发的证据。
- **正式资源门，第530–556行及Bash第8–16行：** GPU0、单核/线程、GPU与主机资源门、磁盘26GiB、28GiB allocator、硬timeout均实际保留。直接启动观察`reports/documentation/20261005/relation_memory/execution.json`记录Python亲和[0]，与Bash和原生资源门一致。

纯标准库核验了23个冻结源文件的实际字节和SHA，且gate、execution、manifest、collected、evaluation、deployment及修后CPU证据的来源集合完全相等。核心源码SHA仍为`4d2fc05c945470a0e22fd563eed93e6b346904a7c1138cb0029406f09676b044`，与原生证据相符；正式run源码为`213119e1205ed7cc765cc44f8a6fb8c3822198da25a28ca99a7d81b29a0bc094`，即10月5日修后版本，不是旧存在P1/P2/P3的版本。

**性质：身份与实际代码路径核对，不是用哈希替代算法判断。影响：可以复用原生核心准入与已关闭的最小入口修复。最小处理：不重开已关闭问题、不追加GPU探针。**

## 3. 权重、缓存与单步资源的证据边界

九个points均记录真实完整状态恢复、临时状态核验后删除；推理权重记录与inventory逐项大小/SHA一致，合计11,758,382,199字节。九个Memory记录与inventory描述符一致，阶段末范围484,440—504,583字节。**这些实际载荷没有附在外审包；网页只交叉比对其记录，没有重新计算权重/缓存载荷的哈希，更没有重载BGE。** Linux现时核验的范围据inventory、sync和独立核验原件引用，不能冒称网页完成了同一载荷检查。

原生最大形状两群含额外历史VJP为16.639秒更新、26秒包装、15.555GiB reserved；正式任务的输入集合、实际平均形状和诊断工作不同。原来12—18小时只是估计，实测3小时50分50秒与该估计不同并不自成矛盾，也不能据较低正式峰值推断漏反传。主报告第13行已说明手写最大形状不代表正式群平均成本。未获得正文来独立计算每群token分布，因此不将时间差定量归因于文本长度。

**性质：测量范围与不可在网页重做部分的明确边界，不是本次比较的复现阻断。最小处理：保留边界，不要求新读取真值、下载权重或重跑原生。**

## 4. LOGIT0.1资格与公平时间比较

基线子审详细核对见`baseline/baseline_qualification_checks.json`及其标准库脚本。旧结果审查主审`reports/seller_alias_continual/20261004/result_review/report.zh.md:7,42,44`已关闭；原外审全文第7–11、310–316行未发现科学阻断或影响交付的复现缺陷。直接LOGIT completion为COMPLETE、1728新增物理更新、train1/valid1/heldout0/owners0，before_valid为完整盲门且valid0，exit0。它明确`all_guards_against_seq_pass=false`，没有把强基线当作全旧判据通过。

两份基线合同描述符与实际execution记录的大小/SHA一致；候选和基线collected的有序group_ids、domains、22个metric_columns完全相同。LOGIT_LOW合同第11–13行允许从共同首域model/Adam/RNG恢复；候选因头、损失、Dropout变化从预训练自训首域。当前比较因此是合法的完整配置效果比较，而非单一记忆机制因果对照。旧基线权重经批准删除不阻断保存矩阵比较，也不能伪称仍可恢复旧末端模型。

LOGIT0.1旧任务由`started.txt`16:01:03至`finished.txt`21:23:48，2026-10-03，共**19,365秒=5小时22分45秒**；resource_usage第5行一致，退出0。旧job只新执行后两阶段1728更新、3456群梯度呈现，首域早已训练并恢复。候选本次为2592更新、4320群呈现且自训首域；两者头、损失、Dropout、CPU核（旧47、本次0）、实际运行时点及保存/评价工作范围也不同。

**性质：基线资格成立，但效率因果归因不成立。影响：可以陈述各自实际job耗时，不能直接形成算法加速比或等算力优势，更不能归因于关系记忆本身。最小处理：主报告第13、108行现有“不宣称公平效率贡献”正确，无需为接纳本负结果追加效率实验。**

## 5. 本网页实际执行的检查

- `check_resources.py`：仅Python标准库读取JSON、文本及所附源文件，核对23来源、108条进度、资源上界、完成/访问/盲门以及记录间身份。
- 输出`resource_checks.json`：PASS_SUBMITTED_RESOURCE_RECEIPT_CROSSCHECKS。
- `baseline/check_baseline_qualification.py`：纯标准库核对必要基线元数据、合同描述符及记录，输出`baseline_qualification_checks.json`。

网页运行没有导入生产训练/评价模块、Torch或模型，没有连接项目Linux、读取原始正文/逐对标签或执行正式流程。上述PASS仅属于这些已说明的提交材料交叉检查，不冒充原生训练证据。
