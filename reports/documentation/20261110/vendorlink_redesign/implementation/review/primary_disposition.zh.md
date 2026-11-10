# 账号CE实现外审：独立主审处置

2026-11-10。完整回收并逐节阅读[外审原文](external/report.zh.md)，36462字节，SHA256 `93685fa3997a300f86623c742aa1899ee9b85363d707a135e7226678e6e18020`。对应40成员、143125字节送审包及原提交身份不改。本轮审查只覆盖已确定实现、必要原生核验及工程准入，不是正式效果外审。

## 1. 裁决与有效授权

外审未发现必须修复的科学阻断或可复现性缺陷，不要求追加CPU/GPU资格。主执行者结合实际入口、源码版本差分、原生回执及本轮Linux只读来源/资源观测，接受当前实现可以进入D的结论；没有新增代码修复或科研测试。A01/A02按下表闭合，A03承接既有结果报告义务。

用户最新明确：“外审已结束，仔细审查后，修改必要的问题，没问题了就正式开始实验。汇报预期实验耗时”，并提醒完成必须事项。这是当前完整方案的条件性正式启动授权，取代旧暂停；外审依据提交时gate=false而写的“待另获授权”属于当时状态，不恢复为重复确认。主审闭合后按[现行scope补充](../scope.zh.md)和来源绑定gate启动D，原条件链及失败规则不变。实际启动与更新由后续启动回执证明，本主审不预写运行成功。

## 2. 意见处置

| 项 | 独立核查及处置 | 状态 |
|---|---|---|
| A01 版本与执行覆盖 | CPU01实际调用历史run.execute，GPU01调用当前make_model/groups_for及实际methods更新，不能写成当前完整循环又由GPU重测。已从Git实际取得下附差分：run从CPU01到GPU01仅一行注释变化，无计算、数据访问或控制流变化；当前methods与CPU01相同。control仅CPU回执字段接线变化，由CPU02受影响入口正反例覆盖。旧完整循环证据可在未变范围复用，不加测试。 | 已关闭 |
| A02 C端点数量措辞 | 当前实现报告改为“12条既有轨迹各自stage0—3的完整端点集合”，与pipeline/score一致。C不新增训练，完整确认矩阵不能理解为仅12个最终模型。送审请求和原ZIP字节冻结，不回写历史。 | 已关闭 |
| A03 完整结果报告 | 阶段完整且按门开放后，报告逐域、逐seed、同seed配对差、均值、样本SD及范围；学习、遗忘、旧域末态、新域保护和外部比较分别解释，Full−ER组件需M与C同时通过。不能只摘均值或PASS。属于现有交付要求，不是启动前新增实验。 | 承接后续结果分析 |

源码身份并不等于执行覆盖。GPU01只证明其4次手写真实BERT更新及确定下一步恢复，3.973GiB为单路采样峰值；没有混合双路长训、真实显存耗尽或完整40epoch效果证据。CPU调度使用真实Runner/子进程但OOM与20GiB观察为模拟，不伪称真实GPU OOM。未变V、角色资格与原H目标证据按原范围复用。

## 3. 精确历史差分

以下由Windows Git原blob差分取得，属于文件身份机械核对，无科研执行。

```diff
diff --git a/scripts/step35_vendorlink_account_run.py b/scripts/step35_vendorlink_account_run.py
index b7f51659..e79fdc23 100644
--- a/scripts/step35_vendorlink_account_run.py
+++ b/scripts/step35_vendorlink_account_run.py
@@ -57,7 +57,7 @@ def make_model(binding, task, device='cuda:0'):
     if task['method'] != 'VendorLink_SEQ':
         # Recompute transformer activations during backward with the original
         # dropout RNG restored. Same inputs/objective; avoids retaining all
-        # 88 full-length channel graphs in the approved 13 GiB allocation.
+        # 88 full-length channel graphs. No artificial GPU allocation cap.
         encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
     tokenizer = BertTokenizer.from_pretrained(binding['directory'], local_files_only=True, do_lower_case=True)
     if not tokenizer.do_lower_case:
```

```diff
diff --git a/scripts/step35_vendorlink_account_control.py b/scripts/step35_vendorlink_account_control.py
index bd06fe92..233ec868 100644
--- a/scripts/step35_vendorlink_account_control.py
+++ b/scripts/step35_vendorlink_account_control.py
@@ -64,7 +64,9 @@ def permit(path):
     root = Path(g['experiment_root']).resolve()
     if any(not Path(g[k]).resolve().is_relative_to(root) for k in ('workspace', 'campaign')):
         raise ValueError('New outputs must stay inside this experiment')
-    if read(g['gpu']['path'])['status'] != 'PASS' or read(g['cpu']['path'])['exit_code'] != 0:
+    cpu = read(g['cpu']['path'])
+    if (read(g['gpu']['path'])['status'] != 'PASS' or cpu['status'] != 'PASS'
+            or cpu['tests'] < 1 or cpu['errors'] or cpu['failures']):
         raise ValueError('Actual necessary CPU/GPU evidence must pass')
     manifest = read(g['manifest']['path'])
     if manifest['status'] != 'QUALIFIED' or any(manifest['counts'][d]['eligible_accounts'] < 128 for d in DOMAINS):
```

当前13份运行源码保持送审字节不变；不恢复已退役测试或旧OOM包装器，不重跑资格、不新增候选/seed/顺序、不改任何科学门。

## 4. 准入、预算与保管

实际控制器继续使用D8→条件P6→条件M6→条件C及允许的无真值应用输出。D检查H_CE学好，P同时检查学好与明显遗忘，M/C分别检查Full系统和组件；任何完整性缺项禁止效果开放及下一阶段。共同12/4/2/2角色、H多记录评分、V_CLS/V_SET两读取器、种子与公平开发机会保持。

最多2槽，H＋H/H＋V且最多一V；不设人为allocator或整PID显存硬cap，5/17GiB只作实际派发预留。OOM退休失败槽，健康槽续领当前阶段独立任务，不自动重试或补退休槽。其他失败按原分级停止。

88累计任务小时分桶80训练/4GPU/4辅助，禁止互借；继承training0、GPU18、aux128秒，CPU8已包含在aux中。正式Ledger另登记一次30秒收尾费，之后按实际生命周期计费，旧71450秒及旧失败不清零。原预计全链63—78累计任务小时、双路34—46小时工期仍是规划，需据实际速度更新；不因短测通过宣称科学效果或工期保证。

完成前保留审查原文、主审、源码/核验/费用/来源和原提交身份；无用上传ZIP及本轮浏览器重复副本在精确Git恢复核对后清理。数据、完整状态及正式端点留Linux，活动13源码和共享BERT保护，中文暂停不动。交接唯一对应项、计划、决定及存储原位更新，本地提交不推送。
