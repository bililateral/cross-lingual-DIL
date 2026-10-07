# 当前固定点的来源、执行和资格证据独立复核

本子审查只对附件中的已完成结果、保存更新日志、冻结来源及历史原审查证据进行检查。当前原始来源未修改；没有导入项目模块，没有执行项目 `execute`、`native`、手写训练测试或历史数学脚本，没有载入模型或Memory本体，没有读取正式正文或逐对真值，也没有访问项目服务器或联网。本地附件运算不称为项目Linux复跑。

## 结论

在本次任务范围内，**未发现已证实的SCIENTIFIC_BLOCKER，未发现当前必须修复的REPRODUCIBILITY_DEFECT**。保存证据支持把该点作为按固定合同完成的配置级开发结果继续审查；性能负向与程序有效性分开。不能把文件一致、日志总式正确或实现旧外审当成效果合格证明。

`independent_provenance_audit.py`实际一次运行成功，退出0，0.3374秒；`additional_record_checks.py`实际一次运行成功，退出0。前者9631个逐项断言主要是哈希、逐步编号和账目检查，不能解读为9631个独立科研实验或正确性证明。全部源码、真实stdout/stderr与JSON输出均保留。

## 一、来源身份与附件范围

外层`manifest.json`是252项唯一载荷清单，252项文件大小/SHA均由本脚本实际重新计算且相同。根层253成员口径包括manifest自身。本子审查另核对历史实现原包为13,204,125字节、SHA256 `545ec223fcf2b7408568310a19589c8fc203a721ab8a015e48c9d8435f3a589a`，87成员及86项manifest载荷全部相符。

当前23项正式source逐文件与`source_inventory.json`一致；其顺序、path、bytes、SHA同时与`gate.json/source_files`、`job/execution.json/sources`、`job/run/manifest.json/source_files`、修后CPU原件和新native原件严格相同。`job/execution.json/gate`绑定实际gate字节。源码冻结副本中的合同11390字节、SHA `c5c2eff899e823055b82c13b71068c9b6eb72fff0f5d90df2945c46478ce9575`，policy3029字节、SHA `69032e8b1936ce9a67e5f857e5014254f4e804b9eb1002ecf717036372ef13da`，是实际执行身份。

冻结policy的两个LOGIT0.1 collections在`reference.local_root`下重新核对为原字节。对手文件的服务器路径和本包路径是显式映射，不把本包未保留服务器绝对目录当作路径错误。CPU回执亦由gate里的部署路径显式对应附件`repair/cpu_20261006_231000/result/result.json`，bytes/SHA相同。

需要正确描述171记录：`inventory.json/files`登记171个远端非权重结果，其中本包实际提供162个载荷，另外9个是明确不上传的Memory文件记录。已对162个实际载荷逐项SHA重算；9个Memory只有大小/SHA与端点记录交叉核对，没有声称读到本体。另9个权重的大小/SHA/最终路径/uid在保管表，合计11,758,264,524字节；本地没有这些模型本体，不能声称重新计算其权重SHA。

30项`startup_inventory.json`与现有文件比较只有`job.wrapper.txt`正常增长；启动清单中的37字节恰为最终wrapper的原始started_at前缀，前缀SHA精确匹配。最终`progress.json`保留RUNNING是最后一次增量快照；除正式结束加入的状态、逻辑更新和呈现计数外，与最终manifest完全相同。两者均不是当前缺陷。

## 二、固定合同、访问、完成和资源

实际冻结合同保持完整baseline头、共享首域、d129/.001、FP16规范前向与FP32直通，目标`B_current + 0.1 B_history + 0.5 D`，s0及ABC/BCA/CAB；不增加候选。本轮固定五条件与24小时/24GiB/28GiB reserved/64GiB RSS一致，gate未替换判据或预算。

正式wrapper：2026-10-06 23:19:15至10-07 02:45:15，时差精确12,360秒、退出0。`completion.json`为`COMPLETE_FUNCTION_MEMORY_FIXED_POINT`，开发条件false；`resource.json`与completion各峰值精确一致。二者elapsed差0.00183168秒为最后刷新差，远小于wrapper启动至退出包围的时间。资源保存峰值reserved9,405,726,720字节、RSS7,387,922,432字节、产物观察峰值15,682,921,370字节，均低于冻结上限。

`before_valid.json`绑定最终run/manifest原字节，9点完整盲门PASS，access为train1/valid0/heldout0/owners0；最终access与completion均train1/valid1/heldout0/owners0。实际runner的`execute`读取顺序为train→完整blind_gate→before_valid→parse_once(development)→collect→complete；未发现边界与记录矛盾。这里核对的是代码与原件，未用它宣称外审直接观察过每次系统调用。

正式console含72个每24步进度行，顺序恰为ABC/BCA/CAB各第二、三阶段的24—288步，elapsed严格单调。没有供应正式failure账；补充CPU的已知测试夹具失败与新结果分析脚本SyntaxError单独记录，不移记为正式训练失败。

## 三、九端点与1728更新的独立核对

9个points和9个updates均与manifest大小/SHA相同，端点中实际保存的mapping、calibration/raw/stage-cal/first-cal分数引用全部匹配实际文件。points的模型、Memory记录与同步保管清单相同。所有point均记录full_restore_verified，Adam端点为288/576/864；首域update序列为空且记录shared模型摘要相同，后两阶段各288次新增更新，三顺序合计1728。逻辑更新2592与群梯度呈现3456是另外两个已冻结口径。

使用独立stdlib代码按已冻结seed派生公式重新建立当前六遍日程、Algorithm R和历史抽样流；不导入项目schedule或Memory类。全部1728条当前/历史群UID逐位相同，当前fit与历史成员域归属合法，calibration只取当前域12群。每步阶段/编号/Adam序号、encoder分段学习率、head学习率.001均精确相同。总式重构误差最大0；B三个已保存分量与FP32总和的差只在浮点舍入范围内，未据Python double加法把FP32舍入误认错误。

所有1728条gradient_norm均大于clip=1，范围6.720738410949707—36.95793151855469。这结合已冻结更新路径中的统一clip说明正式每步触发裁剪。保存日志未分离B与D梯度，不支持用D标量/总梯度范数比推断正则梯度弱，也不支持据此调系数。

累计N由48到96，第三阶段维持96且retention为null，与无后继不固化规则相同。每份Memory完整字节记录904,979—925,109，均小于1,048,576。第二次固化后缓存独立复算为ABC 4A+2B、BCA 2B+4C、CAB 6A；CAB不存在C原文回放，但其累计count仍96。仅凭成员集合不能宣称C历史统计删除，更不能将全局失败定因于缺C缓存。

三条第二阶段首步D分别−6.315935428978668e−15、−3.157967714489334e−15、−6.315935428978668e−15，实际是保存的微小浮点抵消量。第三阶段首步D非零不违背旧多教师/运输目标；没有本次正式每步D-only梯度日志。stage1没有新更新，其梯度与D训练均值不可用；本脚本用null而非0表示不适用，避免将原核验器的空序列0占位解释成首域实测。

## 四、R1/F1已关闭与原审查历史

实际完整阅读原`FUNCTION_MEMORY_IMPLEMENTATION.external_review.zh.md`；其中原裁决是“最小修复后进入已授权的一次BGE原生完整一步”，并非那时已正式通过。阅读了原`independent_checks.py`中的实际F1与R1反例及其真实JSON输出：旧F1把真实CPU回执同时放两槽被接受；旧R1 write抛OSError时exit调用为空。原审查者自身清单顺序比较与跨CPU浮点比较器修正的原文、输出也保留，并没有改写为项目失败。

将历史原输入与本轮23正式来源逐文件比较：只有`step28_function_memory_run.py`、`step28_function_memory_verify.py`和`test_step28_function_memory_run.py`三文件改变。核心`step28_function_memory.py`、函数数学测试、合同、policy及未变依赖原字节一致。当前F1实际增加mode=cpu/gpu和native kind检查；当前R1实际用finally执行退出；其余差异为原生模式/环境观察和对应两个测试及CLI措辞。完整差异保存在`historical_to_formal.diff.txt`。

修后CPU result真实mode=cpu，9/9、0失败/错误/跳过，测试5.728秒、外层7.36秒、最大RSS851,428KiB，与本轮23来源相同。包括watchdog write成功/失败两分支与F1正确/错误类型检查。23:03的4通过/1 error属于夹具漏policy，失败原件按用户明确指令删除；失败简要、9,774字节删除事实、一次受限重试授权均保留。累计CPU6.31+4.41+7.36=18.08秒，不清零。

新native原件mode=gpu、kind=`native_handwritten_first_optimizer_step`，外层27.58秒。两输入各448段、每段256token；规范reference/live差0；首次Adam一步，393有状态参数；参数395/非空梯度393/Adam moments786均FP32。D-only三个探针有限非零，三类参数均改变，checkpoint首层调用560；97项历史有效概率为0，退出后与进入前精确相同。reserved16,819,159,040字节及RSS均在预算内。这支持当前新核接通和最大形状准入，不等于本外审重跑GPU，也不是完整BGE独立全参数梯度一致性。

正式gate明确保存用户免审原话，并限定为本轮R1/F1修复和native/接入准入，明确不是结果审查免除。原生gate还绑定实际原实现外审报告SHA。没有伪称增量外审已经发生；当前结果审查仍按冻结合同第7节完成。

## 五、范围限制与分类

### SCIENTIFIC_BLOCKER

本子范围发现0项。没有证据需要撤销当前结果或重训。

### 当前REPRODUCIBILITY_DEFECT

本子范围发现0项。R1/F1是已关闭历史缺陷；旧清单顺序/浮点比较器、本次结果分析脚本首次SyntaxError、本包路径适配都不是正式训练失败。

### 非阻断说明

应在完整外审清楚写出“252外层载荷、23来源、162已附非权重结果可直接哈希；9模型/9Memory仅记录核对”，以及“shared完整状态和实际重载由冻结代码及原运行记录支持，本审没有本体重放”。171不是当前ZIP所附171个可直接哈希非权重文件。没有需要追加数据或重跑的要求。

09:45:59无活动进程的表述来自主执行者报告；本包提供启动观察和最终wrapper/完成/同步清单，外审没有实时查看进程。wrapper已经足以支持已结束的运行事实，无需再取服务器状态。尚未写回本次审查submission链接也不影响冻结结果身份。

### OUT_OF_SCOPE_OVERDESIGN

不建议新增签名/权限/守卫平台、重复9项CPU或native、索取Memory/权重/标签/正文、全量回归、更多阈值/种子/系数或缓存改造。当前已充分核对来源与完成记录；结果能否超过LOGIT0.1由保存矩阵和预定五条件决定，与这些额外流程无关。

## 实际阅读与执行清单

先完整阅读context和主结果报告；完整阅读冻结函数合同/policy、AGENTS及科研纪律适用1—6节、最新交接段落；完整阅读function_memory实现主审处置和原实现外审报告。实际解析当前source_inventory、startup_inventory、gate、authorization、execution、before_valid、completion、resource、access、run/manifest、partition、9points、9updates、CPU/native result、native_gate、CPU失败简要/重试/删除、核验语法日志及成功输出，选择性静态阅读正常train/checkpoint/blind_gate/execute、模式/更新/容器、seed/schedule、R1/F1修复和对应测试。原实现包其他载荷作身份核对，不冒称每份都全文审阅；只检索了深层ZIP目录定位旧shared原件，没有执行历史脚本或据深层手写fixture充当baseline正式记录。

实际可重放命令：

```text
python independent_provenance_audit.py --input <解包后的review_input目录> --out <输出目录>
python additional_record_checks.py --input <解包后的review_input目录> --out <输出目录>
```

两脚本只需Python标准库；前者独立输出逐文件阅读/哈希范围及逐项检查，后者检查历史阶段的wrapper/预算/快照正常演变。原包不修改。
