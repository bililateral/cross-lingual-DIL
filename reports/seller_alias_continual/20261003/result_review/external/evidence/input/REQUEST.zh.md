# ER0.1与匹配LOGIT0.25：正式结果联合外审请求

请对两项已获准且已结束的开发实验作实际独立结果审查。只审查结果、支持这些结果的实际执行和最小审计实现；不重审无变化历史、不增加实验或成功线。请实际读取附件、检查保存数组，给出明确判定和必要修正；不要仅根据本方PASS或哈希数量裁定。用户要求网页GPT 6 Pro；请如实区分可见模型、自述与无法核实的后台身份。

## 本次问题与范围

1. ER0.1：完整历史基础损失乘0.1，主比较0.1−0.25，完整报告0.1−SEQ；原23项对0.25全过才选0.1，结果好坏均结束。
2. 匹配LOGIT0.25：L当前＋0.25 L历史＋0.5 MSE，MSE不再乘0.25；主比较LOGIT−ER0.25，另完整报告ER0.25−SEQ和LOGIT−SEQ。不换成ER0.1对手，不新增训练或调参。
3. 复用原首域完整model/Adam/RNG/缓存/首映射、s0、ABC/BCA/CAB；两项各1728更新、6终点、18新指标集。有限历史6群/1MiB，当前域校准12群，60 valid群；各train/valid一次，完整盲门前valid0，test/owners0。实际完成均约5小时22分钟，未超各12小时/16GiB。
4. 本地主张：ER0.1对0.25为23/23、对SEQ为10/23；LOGIT对ER0.25为23/23、对SEQ16/23；旧ER0.25对SEQ4/23保持不变。请独立检查，而非预设正确。

请重点检查：实际来源31/34项分离；共享起点与配对更新/随机流；系数和更新日志；LOGIT目标来源与旧目标不刷新；保存分数、校准和计数；完整18+81、18+45的复用身份；O/N/Z/F_first/F/G/final_all全部22指标、5000次按实际域配对整群区间；五组23项及开发选择；报告是否把局部收益、负结果、跨零区间、未达原验收、方法创新和外推边界区分正确。不要把区间跨零改判观察值不退化，亦不要把负结果当作运行无效。

## 附件结构与实际复算

`project/`按实际Linux相对路径布置已授权的小证据。根scripts/schema中的冻结实现是ER0.1实际31来源；LOGIT的34来源在`project/reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/`。两份正式来源原样保留，不能用一个版本冒充另一作业。

新原作业在以下相对路径：

- ER：`reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job`
- LOGIT：`reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job`
- 首轮基线：`reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job`
- 旧ER权重参考：`reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job`

原始回传清单与本地新审计在`reports/seller_alias_continual/20261003/{er_low,logit_weight}_result/`。报告在Windows仓库所用`*_result/job/...`链接，包内对应上面真实Linux job路径，不重复存一份完整job；`return_inventory.json`逐项给出Linux `path`和Windows `returned_path`的对应。原始绝对命令/工作目录是项目执行证据，不要求网页环境存在项目主机。

本次最小变化只有保存结果审计脚本增加low/logit分支、实际来源根和LOGIT目标检查；原训练、policy和实际部署未改。`scripts/step28_er_weight_audit.py`及未变`step28_bge_continual_audit.py`只需NumPy，不加载Torch或正式标签/模型。可在网页Linux解包后，以当前已有Python设置单CPU/线程，读取`os.sched_getaffinity(0)`选一个允许CPU，将它传给`--cpu`；不要硬用项目CPU47。命令格式：

```text
python -B scripts/step28_er_weight_audit.py --study low --project . --source-root . --job reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job --baseline reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job --inventory reports/seller_alias_continual/20261003/er_low_result/return_inventory.json --output <新的外审输出目录> --cpu <允许CPU>

python -B scripts/step28_er_weight_audit.py --study logit --project . --source-root reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace --job reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job --baseline reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job --inventory reports/seller_alias_continual/20261003/logit_weight_result/return_inventory.json --output <另一个新的外审输出目录> --cpu <允许CPU>
```

仅重跑本方脚本不是独立结论；请按需要用不同公式/索引/分位数实现复核关键端点、比较、保护和原始计数，指出未独立重算的范围。运行必须限保存小结果与必要手写例，不加载模型、不访问服务器、不安装新模型、不读取或索要正式标签/缓存正文、不重训、不访问test/owners。

## 历史与取证边界

包含两次实现审查原文及主审、首轮与ER权重结果的原审查原文/主审、必要合同与CPU证据。正文中的旧“未运行/下一步”是各自形成时的快照，不推翻后续完成证据。未把旧审查的完整大型手写证据ZIP再次嵌套；若需要这些包外材料，应说明具体缺口，不声称已读全部历史附件。

Linux原生权重本次流式哈希与点记录匹配，但包不含权重，不能在网页再次载模验证。形式上完成/恢复记录须结合此前已审实现、原生CPU及正式日志判断；本方不声称独立重做原训练。真值依赖AP/MAP来自原一次正式收集，不以已有盲分数推测标签再计算。独立审计已有数值最大差7.11e−15仅针对其计数器跟踪量；float32基础损失分解另有约3.4e−7误差，不能混作全栈精度。

技术读取、结果开放和自适应选择是不同权限：本次允许重读已经开放保存结果，禁止借审查增加正式数据解析或后续候选。已开发valid、单s0、三个顺序、条件bootstrap不能支持独立最终确认、三种训练种子、连续λ最优、方法创新或真实市场效果。原23項只是本次冻结验收，不作为通用科研定理。

## 交付

请输出完整中文审查原文、裁定及具体证据定位，分清SCIENTIFIC_BLOCKER、REPRODUCIBILITY_DEFECT、未验证范围和OUT_OF_SCOPE_OVERDESIGN。仅提出影响当前科学结论/交付的必要修正，不新增系统加固、审批体系、训练或标签访问。

如实际执行，请保存独立源码、精确命令、环境、原stdout/stderr、起止时间/退出码和数值结果；失败与修订原件保留，不把预期注入当生产失败或重复执行当新增用例。提供可下载的`result_review_evidence.zip`，带成员清单/大小/SHA；附原提交ZIP及收到成员的完整性核对，勿改原提交字节。无需复制项目大权重、模型或未提供历史包。主要执行者将自行读取关键原件并处置，外审不替代项目原生证据和授权。
