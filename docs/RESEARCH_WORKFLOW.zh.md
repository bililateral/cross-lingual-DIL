# 科研工作流工具与最短接手路径

本页说明辅助工具用法，不增加实验门或授权。常设边界见[科研纪律](RESEARCH_DISCIPLINE.zh.md)，当前有效状态只看[交接当前区](AI_RESEARCH_HANDOFF.zh.md#当前工作)，科学设定由[设计入口](CURRENT_EXPERIMENT_DESIGN.md#当前合同入口)定位。

## 接手与更新

1. 先核对分支、工作区和最新用户决定，读取交接`## 当前工作`至`## 近期执行与审查历史`之间的内容。按研究线小节定位；不要先全文读取讨论、存储或历史快照。
2. 从设计入口找到当前方案及补充、实际入口和必要既有审查。冻结材料的旧“待批准”按其时点解释，不覆盖后续有出处的决定。
3. 涉及运行时，用下面的一次性只读观察核对现场，再更新交接对应项。观测不重启、补位或控制作业，不开放未完整阶段效果。
4. 作业或审查状态变化，更新交接对应研究线和原件链接。有新合同、待办、用户决定或保管变化时，再更新各自负责的入口。普通进度观测归作业回执和交接。

## Linux只读观察

使用已核对身份的 `scripts/research_observe.py` 和 Linux 既有 py310。普通作业只传 `--root`，沿用根目录的 `launch.json` 和 `gate.json`。迁移过的作业同时传入交接具名的有效门和身份回执。下例使用 11 月 17 日 β 的历史路径说明字段对应关系。

```bash
research_root=/home/yongpeng/cross-lingual/reports/documentation
/home/yongpeng/miniconda3/envs/py310/bin/python -B \
  /path/to/verified/scripts/research_observe.py \
  --root "$research_root/20261113/vendorlink_beta_quarter" \
  --gate "$research_root/20261113/vendorlink_beta_quarter/gate_after_postprocess_resume.json" \
  --identity-receipt "$research_root/20261116/h_ablation_result/recovery/state/beta_handover.json"
```

工具先核对回执绑定的门路径（具名模式）、大小和 SHA，再解析同一份已核字节并定位作业。进程身份兼容顶层启动回执和嵌套的 `identity`，逐项核对 PID、属主、完整命令、cwd 以及创建时间或 `start_ticks/boot_id`。若回执给出两种启动标识，两者都要吻合。遇到 `matches=false/null`，结合观测时点区分历史进程、已退出和读取受限。

训练进度只从 `run/workload.json` 的 `current_updates` 取得，按完整训练轮或完成时刷新。缺文件时 `counters=null`；工具不打开含样本与训练明细的 `updates.jsonl`。缺调度回执时 `dispatch_record_exists=false`，失败数、活动槽及控制器错误状态为 `null`。回执实际记录空失败列表时，失败数才为零。

输出为 stdout JSON。观测按顺序读取，任务切换时各项可能取自不同时刻。包装验收、完成文件存在和科学判定分别报告，`NOT_READ` 表示效果内容未读。按当轮权限保管新的具名快照；遇不可解析的元数据或其他作业布局，按实际入口核对。

预算三项分开：`settled_seconds`是账本已结算；`active_lifecycle_estimate_seconds`是身份匹配的活动worker暂估；`active_full_commitment_estimate_seconds`由同机创建时间/生命周期deadline推算完整承诺。后两项互相重叠，不能相加；预约不是已耗。原始预约未单独持久化，因此承诺明确标为估计，不冒充权威准入账。监管尚未结账费用不估算，CPU子账不重复加。工具从不写账，也不以这些显示值代替原Ledger准入。

## Windows具名清单与SFTP核验

`scripts/research_files.ps1`只处理文件管理和身份。每次提供具体相对路径，不能用整目录扫描代替已批准范围。Manifest/Verify可机械核对既有证据，不表示获得数据、标签或外发许可。

```powershell
# 先按批准范围列明文件；stdout可保存为本轮清单。
./scripts/research_files.ps1 -Action Manifest -Root . `
  -Files @('scripts/research_observe.py','tests/test_research_observe.py')

./scripts/research_files.ps1 -Action Verify -Root . -Manifest ./manifest.json

# 新文件写入本次获准的既有隔离目录；方向必填。
./scripts/research_files.ps1 -Action Sync -Root . -Manifest ./manifest.json `
  -HostName yongpeng@10.201.109.111 -RemoteRoot /explicit/isolated/directory -Direction Push
```

Push 先核本地清单，再用系统 SSH 调用 Linux 既有 py310 执行随工具提供的具名文件核对逻辑。目标和父目录由本轮明确指定。已有同内容直接跳过写入，已有异内容默认拒绝整批。只有本次获准更新的具名可变文件，才在调用中另给 `-ExpectedRemoteManifest <旧身份清单>`，列明对应路径和已核实的旧大小、SHA。旧身份不符或应存在的旧目标已缺失都会停止。冻结文件沿用原保管约定。

有变化的内容先用 SFTP 传入目标同目录下本次私有暂存目录。发布前再次核对整批目标和暂存字节，再逐文件发布；新目标以独占方式创建。工具之间以父目录锁协调发布，其他写入工具须与本次具名替换错开。最终逐个 SFTP 读回并校验，成功回执中 `actions` 与清单逐项对应，为 `create`、`replace` 或 `skip`。中断可能留下已发布的部分文件或本次暂存，先核对现场；同内容可安全重试。

Pull 以已取得的远端身份清单为准，全批核对后才安装本地文件，同名异内容会停止。认证由 OpenSSH 处理。一次 Push 的预检、上传、发布、读回使用独立客户端连接，口令认证可能分别提示。本地读回与远端暂存均按本次具名路径收尾。

## 外审收尾复用

沿用已有`submission/recovery/primary_disposition/final_sync`材料，不新增审批文件或第二份状态系统。通过指定Playwright回收原文后，由主执行者阅读、逐项裁决、完成必要修复并记录证据；模型请求、可见选择器、自述身份分别保留。

```powershell
./scripts/research_files.ps1 -Action ReviewRecord -Root ./review `
  -Files @('external/report.zh.md','completion_page.yml') `
  -ReviewUrl 'https://chatgpt.com/c/已核实会话' `
  -Disposition primary_disposition.zh.md `
  -BrowserOutcome '已实际观察的关闭结果'
```

ReviewRecord只生成具名原文/完成页及主审文件的身份，要求明确主审位置和实际页面结果；不能自动判定外审通过、科学达标、清理完成或新实验许可。按需要将其字段合入原有recovery，而非重复套多份回执。原输入和提交保持冻结；临时材料清理仍先核实际依赖及精确恢复，已有同步清单可直接复用。

## 后续接口修改

新辅助工具明确区分`read_object(path)`的已解析对象与文件绑定/路径；传输只接受显式`path/bytes/sha256`清单，缺字段、错类型、重复路径或SHA不符即拒绝。训练产物完成记录、包装验收结果与科学门分别显示，不用一个success替代三层结论。活动科学来源不为统一字段热改；将来确有接口变更时只补受影响的类型/错误传播/收尾核验，既有合同和有效证据继续适用。

常规验证在 Linux 既有 py310 运行 `tests/test_research_observe.py`，在 Windows 运行 `tests/research_files.tests.ps1` 做文件机械检查。具名 Linux 验证已获准时，可给后者传 `-SyncTestHost 用户@主机`，用随机临时目录内的虚构文本验证实际 OpenSSH/SFTP 入口并清理。
