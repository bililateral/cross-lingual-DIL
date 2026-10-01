# 旧分数保持基线：实现与核验

2026-09-15旧分数保持基线已完成：distillation_execution/20260915_164430/job于18:00:02退出0，1620更新、9次完整恢复评分、3首域模型/全Adam/全分数精确重建、6缓存检查及25目标群前向通过。实际正式训练3710.917494秒，完整1小时15分31秒；32份2355700字节小文件回传匹配，3模型5660853090字节留Linux且新鲜哈希匹配。Windows distillation_evaluation/20260915_181330在完整门后valid一次评价1.755147秒完成；本阶段train/valid均已消费，test/owners未读。独立14130数最大差2.22e-16及540盲预测计数匹配，不新增标签访问。相对SEQ首域AP保持H=+0.053668406，条件95%[0.032935981,0.075454319]；新域AP差N=−0.059000452，区间[−0.072663850,−0.045636162]；同顺序3/3，因新域能力门失败，整体仍未通过。相对随机ER保持增量+0.073268416成立，不能替代三门；保留固定阈值recall/F1相对ER下降。属于旧域保持有效的局部进展，不是自有方法或真实市场结论。完整结果见docs/SELLER_ALIAS_DISTILLATION_RESULT.zh.md；实际GPT 6 Pro结果外审b2af6353-cadb-4e07-a855-997d0ac1e136及主执行者复核完成：无科研阻断，R3缓存口径已区分快照205128与训练完整状态205146字节，不改正式结果。外审17986数值检查与本地主审5764数回比（最大1.11e-15）及18首批目标盲向量SHA核对分开记录；未重读标签或重跑。新域能力门仍失败，不自动调参、重训或开放test。 下方先前待启动/启动中表述均为历史。

2026-09-15，用户明确要求“那先实现这个基线吧”。依据[具体方案](SELLER_ALIAS_REPLAY_DISTILLATION.zh.md)，实现随机完整群ER加阶段入库logit保持。是已有蒸馏思想的基线适配，不是原版DER++或自有关系选择器。Linux新阶段未恢复；train/valid正式解析、模型加载、GPU更新均为零。

## 文件与实现语义

- [独立runner](../scripts/step28_continual_distillation.py)：新训练、缓存目标与完整门、Windows评价；旧12份ER来源保持原字节。
- [政策](../schema/step28_continual_distillation_policy.json)：系数0.5、原6群/1MiB、1620更新、三门与原SEQ/ER文件摘要固定。
- [Linux入口](../scripts/run_step28_distillation_linux_20260915.sh)：单可见GPU、单CPU线程、离线py310，3小时外层上限与完整时间/退出记录；目前没有执行。
- [针对性测试](../tests/test_step28_continual_distillation_contracts.py)：手工群和手工标签，不使用正式监督。

旧BCE权重和两支日程保留。历史支一次前向同时计算mean BCE与0.5×mean MSE，一次反传；与当前BCE反传一起统一裁剪、一次AdamW。生产不提供系数搜索入口。系数0仅作为手工验证选项，检查与旧ER精确一致；非零系数用独立损失和Adam动量作参考。

每域入库选择后，只对最终保留的本域新群评分。目标来自当时本分支已恢复的模型，eval/inference_mode，保存CPU/CUDA随机状态并恢复原模块模式。目标为378维小端float32，以base64写入JSON，附账号对顺序摘要、生成点及模型摘要。9072字节只是6组原始数组大小；base64、全部元数据、原文、标签和两个随机流均按完整序列化包计入1MiB。不能只按裸数组计费。

入库期间逐群检查含存活目标的状态，淘汰群的目标同步移除；存活旧群目标不刷新。新增目标逐一加入时继续检查。缓存实际写盘、完整恢复后作为下一阶段供数，阶段中目标/Memory正文固定。三域结束不再生成目标。只读公开ID重建得到实际新增评分次数ABC/BCA/CAB为9/9/7，共25，低于预先规定36上限；不是增加25次训练更新。

首域仍精确重建原180步，模型/完整Adam/原valid全分数匹配并实际恢复后进入181步；不使用推理模型加空Adam。9点评分仍完整双遍，3最终模型只保留Linux。原SEQ和ER评价复用保存矩阵，新增结果与两者分别比较；相对ER的字段明确写random_er，不错误沿用sequential名称。

Windows在任何新valid解析前检查新9点/6缓存回执、目标时点/顺序/不刷新/评分次数/字节、日程/Adam/更新、来源和分数文件，并检查原ER及SEQ保存证据。只哈希缓存原字节，不解码train监督。新增valid只解析一次后形成全部22指标、逐群混淆计数和配对区间。超过ER的增量条件与相对SEQ三门分别报告；前者不替代后者。micro指标可由保存计数汇总，不另读标签。

## 主执行者亲自复核

已核读新runner及直接依赖：原replay的完整训练/缓存/门/比较，expression的公开输入、二值标签对齐和日程，population的LaBSE加载、商品重数池化、上三角边序、梯度和状态摘要，population_run的checkpoint预留/完整恢复/双遍评分，population_evaluate的AP与梯形PR-AUC、阈值、检索与汇总。新代码没有调用原population全分区Archive或旧生产训练入口。

本轮回读原ER实现外审8e6b1e2a-d9bd-4a9c-9d92-d335ef1af29d及结果外审633ea43b-1acb-4e5c-b447-604994b8d1bf完整原文；缓存内外诊断外审099d9968-0d27-4d8d-8a11-ca1ad9c5f5e0沿用已亲自阅读核验的原文及闭环记录。没有将历史外审算作新实现通过。

延续已关闭R1的正确报告范围：Budget只覆盖run目录的观测，外层日志、最终回执和完整进程时间需在结果收尾合并核验。新增模型与完整状态共存要记录，最终占用和离散观测不能叫精确磁盘峰值。无新监控系统、额外GPU smoke或旧阶段重跑。

本地修正了新增评价中复用旧比较函数时，ER参照字段会被标成SEQ的问题；现已按比较角色重命名，并加入手工结果检查。该问题在正式执行前发现，未产生科研结果。没有修改冻结旧来源或重读其监督。

## 验证与缺口

首轮新增合同集：7通过、3个实际Torch用例因本地无Torch跳过。受影响范围扩展到旧replay、expression的PilotContracts及population的MetricContracts，共39项，35通过、4跳过、0失败/错误；第四个跳过是旧ER的实际Torch用例。日志和准确执行范围见[测试记录](../reports/seller_alias_continual/20260915/distillation_implementation/tests.json)。新增Torch断言及结果角色检查的最终核验另见同目录final_checks.json，不以旧计数代替最终证据。

新Python AST和CLI --help通过；本地Git Bash对新入口bash -n通过，不等于Linux执行通过。只读公开输入、原ER完整门和10份原保存指标映射通过；实际当前文件验证12份冻结ER来源相同，未解析正式标签、加载模型或连接服务器，见[参照核验](../reports/seller_alias_continual/20260915/distillation_implementation/local_reference_check.json)。

三个新真实Torch用例覆盖：手算MSE/梯度和无梯度目标；零系数原ER模型/完整Adam精确一致、非零目标独立参考、保存恢复后继续更新；真实CPU玩具模型目标评分、模式/RNG保持及跨域淘汰/目标不刷新。本地缺少Torch，暂不声称这些已实跑通过。外审请求在其CPU环境执行；恢复Linux阶段后，正式标签解析前仍须在py310执行这三项且零跳过。玩具测试不冒充LaBSE/CUDA通过；正式六段首更新还要记录两支编码器/头的真实梯度和更新。

送审时用户消息489119c4-b28b-4f6a-9832-da49a1f4a1a2，148份文件、ZIP 2779184字节，页面当时显示Pro思考中；该等待现已结束，实际回复model_slug=gpt-6-pro。当前处置见下文和reports/seller_alias_continual/20260915/distillation_implementation/review.json；未通过必要核验前不能启动依赖训练。后续新Linux阶段仍先报告并暂停。

本次送审已确认网页收到并开始思考后，删除上传副本2779184字节及本轮两份浏览器临时文件8318字节，共2787502字节；三个路径已核实不存在，见同目录cleanup.json。清单、提交原文、测试与必要科研证据保留，无一次性辅助脚本遗留。本次清理不改变此前其他阶段临时副本的历史状态。

## 外审发现、最小修复及独立复核

实际外审全文已读取并保存review_reply.json，原提交源码/测试分别保存在submitted_runner.py、submitted_tests.py作为缺陷与修复证据，不是待清理辅助脚本。外审认为方法身份和科研配置可保留，但提交版不能放行。

D1：observed_update的布尔参数observe被内部同名hook函数覆盖。生产非首更新传False时，before为空，但后续函数对象恒真，导致optimizer.step已经执行后访问before['encoder']抛KeyError。主执行者逐句核对确认；旧真实测试两步都用默认True，180步日程测试mock整个更新，因此遗漏该路径。若其他前置门通过，正式ABC第二域第二步（总第182更新）会触发；尚未正式运行，这不是方法负结果。

修复仅将内部函数及lambda调用改为observe_gradient；源码与原提交的差异已程序化确认只有这两处改名。既有真实CPU更新用例改为第一步True、第二步False，保留系数0/0.5独立参数与Adam动量参考、真实保存恢复；关闭观察时检查观察值为0，不把未观测支当作实测非零。三个Torch用例数量保持不变。没有改模型、损失、数据、日程或科研判据。

D2：原148文件包确实遗漏schema/step28_continual_population_policy.json，导致外审三个旧population测试FileNotFoundError；本地原文件存在，非新生产依赖错误。该原字节政策已纳入review_followup_files.json，原148清单及3错误外审记录保持不变；当前浏览器服务断开，尚未补发，不声称原包已完整或外审三项已通过。

修后本地运行新合同集及这三个旧测试，13执行、10通过、3项真实Torch因缺依赖跳过、0失败/错误，见correction_tests.log及review_verification.json。19份其余送审脚本/政策/测试原字节未变。外审自述：原新增Torch3通过，但未覆盖D1；其独立修复副本的新Torch3通过。两者不追认为本地实际执行，也不追认为原提交版已正确。

外审另自述两项CPU补充、2376项保存矩阵比较；主执行者按原文核对其公式、范围和现有源代码/历史证据，没有取得或执行它的补充脚本，不登记为本地新实测。下载本轮核验包时download.saveAs报页面/上下文/浏览器关闭，随后MCP Transport closed；未取得ZIP或补丁文件。修复依据全文明确的两处改名和亲自代码复核，不冒称应用了下载补丁。

针对反复断连已只读核查Codex本机日志：2026-09-15 14:14:21、15:19:52、16:33:31均有Node未处理异步TargetClosedError；最新随后出现MCP输入流终止。确认服务异常退出后连接失效；浏览器最初为何关闭、具体哪个promise拒绝和alpha版本是否相关仍未确定，不归因网络或Linux。证据在browser_failure.json。后续避免原样盲重试下载及把延长超时当修复；没有修改全局MCP配置或终止进程。

下一Linux阶段尚未恢复：先同步最小修复与必要输入、只读检查资源，使用原py310执行修后的3项CPU用例，必须零跳过通过才可进入正式train解析和1620更新。沿用1空闲GPU/1CPU线程，完整预计75—110分钟，上限3小时/16GiB；valid只能在全部完成门后Windows一次解析，test/owners不读。若CPU门失败则停止，不消耗正式train、不自动续跑。按报告/暂停纪律等待用户恢复该阶段。

用户随后明确“启动linux吧”，已恢复上述Linux阶段；16:41:35只读资源检查满足，先同步修后代码及必要输入。后续正式启动记录在distillation_deployment及新distillation_execution目录。用户另要求下次网页下载时重点排查重复断连原因，不原样盲重试；不改变科研外审纪律。

## 本次Linux启动与回传

2026-09-15旧分数保持基线已在Linux后台启动：distillation_execution/20260915_164430/job，16:44:30 CST，启动PID15077、训练Python PID15090。修后3项真实CPU张量用例全部通过、0跳过，1.319秒；16:47:31快照首顺序ABC_shared已90更新，GPU占用12902MiB，未见异常或完成退出。该阶段train一次解析已消费，valid/test/owners未解析；总1620更新配置不变，完整预计75—110分钟，上限3小时/16GiB。102份20645584字节部署文件全部大小/SHA匹配，30复制72复用；6份启动证据回传匹配。后台进程脱离SSH会话，退出连接不停止训练。实际总训练时长及全部方法验证须待完成后报告，当前不能声称缓解有效。见[启动快照](../reports/seller_alias_continual/20260915/distillation_execution/20260915_164430/startup_status.json)及同目录startup_transfer.json。下方待启动文字保留为历史，不代表当前状态。下次Playwright下载继续核对下载事件、浏览器关闭、未处理异常和MCP退出的先后关系；原始关闭原因未确定，不原样盲重试。

## 浏览器根因排查补记

2026-09-15网页下载根因排查：已用独立临时配置定位到Chrome恢复History与shared_proto_db中同一已完成下载记录时的崩溃触发条件；History id12和对应DownloadDB记录构成最小复现，副本删去该缓存记录则正常，扩展保持启用且无已安装IDM。Chrome原生0xc0000005先于保存失败，MCP未捕获自动saveAs拒绝又导致Node退出。系统IDM注册存在不等于MCP启用，当前不归因IDM；新下载ID冲突假设被实测id29否定，不声称数据库物理损坏或已定位C++故障函数。按用户“先把根因定位”要求，临时永久禁扩展配置已撤回，原下载数据库未手工修复；当前服务仅保留会话级诊断错误捕获。实际外审证据包65565字节、30成员、ZIP CRC通过，已保存external_evidence.zip；尚未审读/执行脚本，不将下载成功等同科研复核完成。原失败记录保留，训练不受影响。详见reports/seller_alias_continual/20260915/distillation_implementation/browser_root_cause.json。

此前清理被自动审批以blocked by policy拒绝、0删除的记录保留在browser_diagnostic_cleanup.json。用户再次明确要求清理后，13项临时文件及目录已全部删除，共16802925字节（约16.02 MiB），逐项确认不存在；外审证据ZIP散列未变，原浏览器配置未修改。见[清理完成回执](../reports/seller_alias_continual/20260915/distillation_implementation/browser_cleanup_completion.json)。

## 浏览器修复与验收收尾

2026-09-15浏览器修复已验收：个人工具playwright-mcp.cjs在每次启动MCP专用Chrome前，用LevelDB原生接口清除与History匹配的已完成/已取消自动化下载缓存；History只读，逐下载saveAs失败被捕获。两次独立MCP、取消后同MCP两次重开浏览器、已安装启动器及当前原生MCP的真实外审附件下载均通过；65565字节ZIP与原包SHA一致。离线核验1预期缓存删除、164其余键值及History/Cookies字节不变。一次性缓存清理、仅删History及禁扩展方案均未通过持续验收并已撤回。用户确认archive-check.zip被IDM接管：HTTP测试在禁扩展时仍被接管，HTTPS及实际外审下载已走浏览器；不能再以扩展列表为空排除IDM的其他接管方式，也不据此把原生崩溃唯一归因IDM。未改IDM全局设置；本地缓存规避不是Chrome上游C++修复。临时脚本/日志/证书/下载副本共60文件725213字节已清理，个人修复运行依赖6739387字节及回滚备份保留。详见reports/seller_alias_continual/20260915/distillation_implementation/browser_fix.json及browser_fix_cleanup.json；旧诊断文字是修复前历史。本轮浏览器工作未连接Linux或改科研代码/标签。

修复固定使用已核查的MCP 0.0.81、Playwright 1.64.0-alpha-2026-09-14及Chrome 153.0.8010.37；启动器复用现有npm缓存中的MCP依赖，新增classic-level 3.0.0在个人工具目录，不属于科研脚本。配置或浏览器版本更换、依赖缓存被清除时需核对该工具集成。完整原配置及浏览器下载状态备份在C:/Users/35734/.codex/playwright-recovery/20260915_174003；恢复浏览器状态须先关闭MCP Chrome。外审证据包已取得且传输核验通过，尚不代表已审读/运行其中科研核验脚本。
