# Evolution 预实验实现

2026-11-07。用户在确认净学习修法、保留双SEQ遗忘门及其余安排后，明确“确认好方案后就开始具体的实现吧，考虑清楚各种边界，不要出错”。本轮据此实施[现行方案](proposal.zh.md)§7列明的实际接线和Linux py310手写CPU核验（累计不超过1200秒，计入后续辅助费用），准备实现审查。正式数据构建/划分、人工样本查看、预训练载模、GPU训练/评分、自动等待与推送仍关闭；本轮只用随机微型模型和手写输入。具体新审查包外发许可另核，不沿用旧方案包许可。

## 实际入口与对应行为

| 文件 | 实际用途 |
|---|---|
| `scripts/step30_evolution_data.py` | 保管端四表构建；调用原audit.scan与followup近重复原语；记录同一原始行，保留大小写；固定增广匹配、160/64/96分区、4条/账号、8账号群；盲文本与qrels分开，固定fit语言抽样。正式调用仍需新的数据许可。 |
| `scripts/step30_evolution_run.py` | 同一cased工厂供训练/教师/恢复/评分；H复用原真实VJP和目标；V_ER联合类别、两次分支反传、一次clip/step；单域或三域正常完成；每任务一个起点小head引用，阶段末各一个encoder/head推理端点，无每轮latest/optimizer/Memory整包。 |
| `scripts/step30_evolution_policy.py` | 有限A候选和固定种子；B双基线门、C/D各比较和质量保护；修正后的同域同对照净学习双路径；固定5000次配对群bootstrap及内外部支持分层。 |
| `scripts/step30_evolution_evaluate.py` | 所有任务完整后九格盲评分；全部分数有限且ID/形状完整后才记一次qrels开放尝试；十指标与逐query保存。 |
| `scripts/step30_evolution_io.py` | 来源核验、不同角色访问记录、1200秒CPU子预算、48小时三桶、真实资源/保存回调；失败费用不退回。 |
| `scripts/step30_evolution_control.py` | 独立目录单worker；A→B→C→D逐门派发；失败不再派发，科学门false正常结束；无监听/自动等待/重试，正式CLI必须具备明确许可和已关闭实现审查。 |

旧 `step29_vendorlink_data.py`、`step29_vendorlink_methods.py` 及 `step29_vendorlink_run.set_rates` 只复用不改。旧run.execute/save_state和旧英文小写工厂不进入本轮执行。中文活动部署及原英文失败/结果不变。新系统名为H0_SEQ/H_SEQ/H_ER/H_full/V_SEQ/V_ER；内部Settings保留VendorLink_SEQ只表示复用的分类/余弦计算家族，不表示数据来自旧ABC。每个任务和端点都绑定Evolution时期、方案、来源与数据清单。

CPU证据只回答接线/边界是否按要求工作，不能证明Evolution上有遗忘、适配或H优势。当前实现尚待必要CPU核验和外审，不能称已获正式运行资格。
