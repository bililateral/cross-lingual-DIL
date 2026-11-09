# 账号监督实现外审：主执行者处置

2026-11-09。用户已批准本精确包外审，并明确“允许该包外审，闭合后按已批准范围运行”。本处置承接 D/P/M/C、4单域＋最多12完整轨迹、最多2 worker、累计60小时及全部科学门；不新增候选、数据、种子或预算。

**结论：五项运行控制缺口已按原合同局部修复并完成必要 CPU 闭合，E01历史用途来源已补明。主审闭合，允许承接既有授权启动唯一正式控制器。** 修后源码没有再次送网页外审；这是科研纪律第6节的已审范围内主审闭合，不冒称外审直接通过了修后版本。

## 外审原件与适用范围

[完整原文](report.zh.md)由本地 Playwright MCP 在原审查会话下载：<https://chatgpt.com/c/6af15075-65f8-83ea-a02d-ce335292df9f>。文件30618字节，SHA256 `e8824a850ba39dcd4d8372a1ed62ffb03d35604d136ad0990a21e97c465161bd`。输入为用户批准的41项/113173字节ZIP，SHA256 `62159832796833361c5d76bb8b633a27035ef9a4025bcab29f4975698f3d40ae`，仅提交一次。界面GPT-6、Pro；审查者自述GPT-6 Astra Pro，未独立核验后台身份。

外审判定送审精确版本有五项必修；没有发现足以否定完整H目标、当前流、Memory/出生教师、旧V头、全池双读取器和主要科学数值门的确定错误。后者不是效果保证。主执行者阅读完整原文，逐项核对真实代码/差分及Linux回执，未把外审小逻辑检查当作项目Linux运行。

## 必修项的实际处置

| 项 | 具体修正与证据 | 主审判断 |
|---|---|---|
| R01 原子保存竞态 | `Check`复用既有`directory_bytes`，仅忽略枚举后文件消失，原磁盘阈值不变；CPU夹具在stat时抛FileNotFoundError，正常通过 | 关闭；不需要新增双路训练 |
| R02 启动异常后继续派发 | 本批halt策略在启动异常分支立即置停止派发；实际Runner以Popen抛错，第二槽启动调用次数为0 | 关闭；CPU03已通过的健康已派任务排空证据复用，旧默认策略不改 |
| R03 共享失败未上报 | 明确`SharedDiskLimit`；worker在来源许可检查之前先按部署位置约束输出目录，来源核对ValueError/OSError与worker共享磁盘失败写共享标记；父控看到标记即停止本批相关活动 | 关闭；CPU验证两类标记及真实小子进程相关停止/无第三次派发。普通OOM/RSS/生命周期RuntimeError没有一概升级为共享错误 |
| R04 64GiB启动门 | 新LIMITS绑定64GiB派发门，实际Runner资源判断消费此字段；40GiB拒绝、64GiB允许，其余资源足够 | 关闭；这是落实已批准门，不是增加资源要求；旧调用未提供该字段时保持原行为 |
| R05 辅助硬帽 | 持续判断计入已结费用、未结父活动、活动aux作业完整承诺与60秒清理余量；不足时终止本批拥有的子进程。贴边分析额外留下60秒父活动空间，避免一启动就取消。阶段间小段父活动单独计费，不包住子作业或等待 | 关闭；CPU分别验证辅助余额不足及活动辅助承诺保护，相关进程释放、账本不超帽；另验证阶段判断在计时段内、阶段执行在计时段外 |

唯一增量核验：[cpu_review_repairs01/receipt.json](../cpu_review_repairs01/receipt.json)、[console.txt](../cpu_review_repairs01/console.txt)。Linux既有py310，7项通过；完整进程2.472306568874046秒，计3秒aux；16份实际部署源码/测试在启动前大小与SHA均匹配。本次无模型加载、无真实数据解析、无新增GPU作业。运行源码的最终身份见正式许可；测试中资源数值和短子进程属于手写夹具，不改正式预算。

## E01：历史用途U的独立主审核对

这项是外审包的来源展示不足，未证明发生泄漏。主执行者复用了资格前实际核对的`preparation_config.json.evidence_sources`，并再次只做七份既有证据文件的大小/SHA核对，没有重读真实正文或重跑资格。下表列具体来源身份；正式许可绑定本处置。

- 原正式训练：旧`formal/gate.json.fit_sources`正是资格已核对的三份fit文件；`step29_vendorlink_formal.load_fit`仅从对应市场绑定读取fit，并核查分区。全部旧方法/顺序/种子均由这四条固定文本/账号覆盖。原heldout评价来自账号互斥的heldout，不扩充这些旧fit账号的U。
- 原资格：`step29_vendorlink_export.py`从每市场fit中固定取前三个满群，`step29_vendorlink_native.py`使用manifest的qualification_samples；本次唯一真实资格已逐记录证实三份样本属于8208条U。手写压力文本不是市场源数据。
- ABC：正式完成原件为`STOP_PRECHECK_NO_VENDOR_DREAM_FORGETTING`、training=0，无候选训练。实际`step29_english_formal.py`评分入口只消费`dev_sources`；`step29_english_control.py`前置失败直接返回。其已绑定三份dev文件与下述全池输入相同；旧fit与dev按原账号键确定性划分互斥。旧方案只提供范围背景，不独自代替完成证据。
- 两次全池：两份`inputs.json.dev_sources`均绑定同三份旧dev：AlphaBay 557002字节/SHA `bfa24e43737bf296ea92eee45faee30574a9832d0d74a88deb1a173b0f184dc9`；Dream 199666字节/SHA `ba45f52519e72b0e51209a1f30735d0ce3af4ffbc0769c7a03ef53196d9c54af`；Agora 467927字节/SHA `64d6fc761f0c74f5f86dd3d3a1de84125a4bf65fd81bc1f667ae8635e72a9381`。核对输入元数据，不打开其qrels或预测。
- 数据盘点/关系证据核查不是模型训练、评分或根据模型效果选配置；关系候选已按用户要求删除，本轮没有恢复。既有离线全局精确清洗使用未来文本、跨账号近重复未穷尽等边界保留。

因此，新留出角色仍以旧fit账号全部模型已用文本U为污染排除起点。不能写成网页外审已独立核验全部历史来源，也不能写成新账号独立确认；C是同账号、同训练模型的未使用文本确认。

## 旧证据复用、费用与剩余限制

CPU02的四项通过、CPU03的两个针对性补测及GPU02沿各自证据边界复用；本轮修正未改变模型、梯度、初始化、当前流/Memory、数据角色、评分公式和判据。CPU03实际走dev评分，未跑完整C控制器，C的相同角色分支/完整矩阵及门按静态追踪结论保留。最终控制器来源检查和本轮调度修复分别由实际关闭入口证据、此次无模型分支核验支持，不称旧GPU执行过这些控制器改动。

原CPU01/02失败、GPU01 OOM及原费用保留。新增3秒后准备账为aux302/GPU18/训练0，总320秒，CPU子账260秒含在aux内。正式Ledger另加一次30秒尾部费，初始350秒；后续真实阶段与监控费用继续入同一60小时账，不归零、不借其他实验余额。GPU02峰值仅是手写短测，不保证长轨迹峰值或工期；正式14GiB/PID、13GiB allocator和原RAM/磁盘限制仍执行。

按合同正常停止P/M/C，不要求实现预期结果。全批结果开放后，利用已有保存矩阵补齐均值、标准差、范围、配对差和逐域代价展示；它是结果交付工作，不另加启动核验或统计成功门。真实跨账号候选仍无同控真值，不报告真实马甲准确率。最终效果及结果外审尚未发生。

## E01来源身份

以下均为已有文件的机械身份记录，不含真实账号/正文/封存标签。路径相对Windows项目根；Linux已有七份证据副本位于本轮workspace/evidence下，资格配置绑定了其绝对路径。

| 来源 | 字节 | SHA256 |
|---|---:|---|
| reports/documentation/20261030/vendorlink/formal/gate.json | 26708 | b0f8efb8e00a158192b1e40ec4fdea06b6176e927875bf3b0a6d5e869a4dc098 |
| reports/documentation/20261108/vendorlink_fullpool/inputs.json | 115350 | 9106ede7c0a55a5a57a284ba5f6650159ae57e5439a6a99e168bf6af51345090 |
| reports/documentation/20261108/vendorlink_fullpool/history/inputs.json | 113625 | dad4a12f1c352a5586e8e35b942f3b4cabec7aa5dc216dbcf7e393dc1319712f |
| reports/documentation/20261104/english_abc/plan.zh.txt | 16022 | 8a948e53f70b7f95e38be9629cdb28fb7e7731d17ca506be3a8e90e7c4f44bbc |
| scripts/step29_vendorlink_export.py | 4642 | 818d13006ef4a5b26df3cf23d84ba1d67b62cbe91e6a9aeee90ffa695f49ac17 |
| scripts/step29_vendorlink_formal.py | 10143 | 3f9f0d394405600a5fac5891b98ca7bcc71dba4ccf51e22b2e3ced7bb9ff3d84 |
| scripts/step29_vendorlink_native.py | 17831 | 103844c8e162f21089f94fd99f47e100b736c04d0dfc158a3bb41a67be8a20ce |
| reports/seller_alias_continual/20261107/history_result/source_receipts/abc_formal/completion.json | 2567 | 1f8c0fdb116c17524a5a07997038f7b332075630df8a965542595f5032824c7c |
| reports/documentation/20261105/english_abc_formal_prepare/formal_gate.draft.json | 231623 | 620676cfc1ce6bc9ab99a70ca03418a8f24a1d32e64bbfea5bd6eb5ca4345a0d |
| scripts/step29_english_formal.py | 18709 | 69411ca4b63fb63431d280ed8fa30179ce4835e9ea7b2ba823b4abe48216a067 |
| scripts/step29_english_control.py | 19652 | eae2c423226455b0719e10d6a8c67209557900b205eff5bc7a3fd32f24afe00c |

增量核验原回执身份：cpu_review_repairs01/receipt.json，4627字节，SHA256 `1f79bcba38d7d12ec1dc334bb50152b4f0abd5e223d45eb4a343a9bc14ac741a。
