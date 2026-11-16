# 科研工作流工具与最短接手路径

本页说明辅助工具用法，不增加实验门或授权。常设边界见[科研纪律](RESEARCH_DISCIPLINE.zh.md)，当前有效状态只看[交接当前区](AI_RESEARCH_HANDOFF.zh.md#当前工作)，科学设定由[设计入口](CURRENT_EXPERIMENT_DESIGN.md#当前合同入口)定位。

## 接手与更新

1. 先核对分支、工作区和最新用户决定，读取交接`## 当前工作`至`## 近期执行与审查历史`之间的内容。按研究线小节定位；不要先全文读取讨论、存储或历史快照。
2. 从设计入口找到当前方案及补充、实际入口和必要既有审查。冻结材料的旧“待批准”按其时点解释，不覆盖后续有出处的决定。
3. 涉及运行时，用下面的一次性只读观察核对现场，再更新交接对应项。观测不重启、补位或控制作业，不开放未完整阶段效果。
4. 启动、阶段结束、失败、授权改变或审查结案时核对受影响入口。普通更新数变化只更新必要观察和交接；设计入口不复制PID、门开关或网页生成状态，计划不复制进度。

## Linux只读观察

使用现有py310执行已同步且核对身份的`scripts/research_observe.py`，显式给出含`launch.json`和`gate.json`的执行根。例如：

```bash
/home/yongpeng/miniconda3/envs/py310/bin/python -B \
  /path/to/verified/scripts/research_observe.py \
  --root /home/yongpeng/cross-lingual/reports/documentation/20261110/vendorlink_redesign/implementation
```

输出仅为stdout JSON，需要保管时重定向到本轮具名新证据文件，不覆盖旧回执。工具不导入科研实现，不载模、不读效果值/标签/正文/分数；只读取允许的启动、gate、dispatch、账本、更新计数和训练包装元数据。效果仅报告文件存在与否，`NOT_READ`不是通过或失败。进程同时核对PID、属主、创建时间、完整命令和cwd；`matches=false/null`须调查，不能仅凭进程存在判断仍是原作业。命令成功只表示观察完成，不表示训练健康或科学门通过。

发生过依赖身份迁移或exec接续的作业，根launch.json可能只保存历史进程，先按交接找到当前迁移回执和gate再核对身份。本工具当前要求created型身份及固定launch/gate布局；只有start_ticks/boot_id或迁移绑定不兼容时，采用与实际字段对应的一次性只读核对，不为通过工具而改活动门、补建假launch或重启监听。进程身份不匹配先区分历史绑定与真实故障，不能仅凭旧PID不存在判定当前作业已停。

观察是依次读取的现场快照；任务切换时可能存在短暂不一致。JSON若正在写入而不可解析，工具拒绝给出成功快照；按具体缺项再观察，不据此重启作业。当前工具适配现有launch/gate/campaign目录形式；遇其他格式应明确报不支持，不能猜字段。

预算三项分开：`settled_seconds`是账本已结算；`active_lifecycle_estimate_seconds`是身份匹配的活动worker暂估；`active_full_commitment_estimate_seconds`由同机创建时间/生命周期deadline推算完整承诺。后两项互相重叠，不能相加；预约不是已耗。原始预约未单独持久化，因此承诺明确标为估计，不冒充权威准入账。监管尚未结账费用不估算，CPU子账不重复加。工具从不写账，也不以这些显示值代替原Ledger准入。

## Windows具名清单与SFTP核验

`scripts/research_files.ps1`只处理文件管理和身份。每次提供具体相对路径，不能用整目录扫描代替已批准范围。Manifest/Verify可机械核对既有证据，不表示获得数据、标签或外发许可。

```powershell
# 先按批准范围列明文件；stdout可保存为本轮清单。
./scripts/research_files.ps1 -Action Manifest -Root . `
  -Files @('scripts/research_observe.py','tests/test_research_observe.py')

./scripts/research_files.ps1 -Action Verify -Root . -Manifest ./manifest.json

# 远端目录须已存在且确属本轮；Push会写具名目标，不能指向活动冻结来源。
./scripts/research_files.ps1 -Action Sync -Root . -Manifest ./manifest.json `
  -HostName yongpeng@10.201.109.111 -RemoteRoot /explicit/isolated/directory -Direction Push
```

Sync使用系统OpenSSH/SFTP，认证由其终端提示完成，工具没有密码参数或凭据字段。Push前核对本地身份，上传后逐份get读回再比较大小/SHA；Pull须提供远端已取得的身份清单，全批校验后才安装本地文件，不覆盖同路径异内容。失败不出成功回执；Push失败可能已有部分文件到达远端，应先核对现场，不能声称自动回滚。工具不在远端建目录/删除，不递归传输；私有目录和常见模型/原始数组等载荷被排除，其他名称仍须人工核对权限。临时批处理及读回副本只在本次唯一临时目录，finally核对路径后清理。

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

常规验证：Linux现有py310运行`tests/test_research_observe.py`，Windows运行`tests/research_files.tests.ps1`做文件机械检查。它们是保留的普通工具回归，不是一次性科研资格，也不自动授权任何模型、标签或GPU作业。
