# 实际执行命令与回执

这些是本次真实执行历史，包含已失败命令；不是要求用户本地执行。所有核验仅使用附件保存的小结果，正式标签读取为0。

## 00_inventory

cwd: `/mnt/data`；退出码：1；壁钟：0.589757340 秒。

UTC 2026-10-01T02:13:10.145382+00:00 → 2026-10-01T02:13:10.735105+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/00_inventory.py
```

原始回执：`logs/00_inventory.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 01_revise_inventory

cwd: `/mnt/data`；退出码：0；壁钟：0.539433144 秒。

UTC 2026-10-01T02:13:23.164737+00:00 → 2026-10-01T02:13:23.704137+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/revise_inventory_v2.py
```

原始回执：`logs/01_revise_inventory.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 02_inventory_v2

cwd: `/mnt/data`；退出码：0；壁钟：0.880342291 秒。

UTC 2026-10-01T02:13:24.359743+00:00 → 2026-10-01T02:13:25.240052+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/00_inventory_v2.py
```

原始回执：`logs/02_inventory_v2.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 03_catalog

cwd: `/mnt/data`；退出码：0；壁钟：0.551632464 秒。

UTC 2026-10-01T02:14:33.572665+00:00 → 2026-10-01T02:14:34.124266+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/01_catalog.py
```

原始回执：`logs/03_catalog.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 04_independent_results_v1

cwd: `/mnt/data`；退出码：0；壁钟：3.797314005 秒。

UTC 2026-10-01T02:16:58.449919+00:00 → 2026-10-01T02:17:02.247199+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/02_independent_results_v1.py
```

原始回执：`logs/04_independent_results_v1.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 05_independent_training_v1

cwd: `/mnt/data`；退出码：0；壁钟：0.705158076 秒。

UTC 2026-10-01T02:22:19.120823+00:00 → 2026-10-01T02:22:19.825939+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/03_independent_training_evidence_v1.py
```

原始回执：`logs/05_independent_training_v1.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 06_supplied_auditor_reproduction

cwd: `/mnt/data`；退出码：1；壁钟：0.545244396 秒。

UTC 2026-10-01T02:22:55.122264+00:00 → 2026-10-01T02:22:55.667474+00:00

```sh
python -B /mnt/data/bge_input/scripts/step28_bge_continual_audit.py --project /mnt/data/bge_input --job /mnt/data/bge_input/reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job --inventory /mnt/data/bge_input/reports/seller_alias_continual/20261001/bge_continual_result/return_inventory.json --output /mnt/data/bge_review_evidence/outputs/supplied_auditor_reproduction --cpu 24
```

原始回执：`logs/06_supplied_auditor_reproduction.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 07_supplied_auditor_correct_cpu

cwd: `/mnt/data`；退出码：0；壁钟：1.886527153 秒。

UTC 2026-10-01T02:23:05.936534+00:00 → 2026-10-01T02:23:07.823028+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/04_supplied_auditor_retry.py
```

原始回执：`logs/07_supplied_auditor_correct_cpu.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 08_report_crosschecks_v1

cwd: `/mnt/data`；退出码：0；壁钟：0.595285750 秒。

UTC 2026-10-01T02:26:17.937478+00:00 → 2026-10-01T02:26:18.532733+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/05_report_crosschecks_v1.py
```

原始回执：`logs/08_report_crosschecks_v1.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 09_revise_report_links

cwd: `/mnt/data`；退出码：0；壁钟：0.566863275 秒。

UTC 2026-10-01T02:26:43.576252+00:00 → 2026-10-01T02:26:44.143083+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/revise_report_links_v2.py
```

原始回执：`logs/09_revise_report_links.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 10_report_crosschecks_v2

cwd: `/mnt/data`；退出码：0；壁钟：0.558327415 秒。

UTC 2026-10-01T02:26:44.754328+00:00 → 2026-10-01T02:26:45.312616+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/05_report_crosschecks_v2.py
```

原始回执：`logs/10_report_crosschecks_v2.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 11_preserve_and_reverify

cwd: `/mnt/data`；退出码：0；壁钟：0.597383919 秒。

UTC 2026-10-01T02:31:48.704138+00:00 → 2026-10-01T02:31:49.301488+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/06_preserve_and_reverify.py
```

原始回执：`logs/11_preserve_and_reverify.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 12_write_report

cwd: `/mnt/data`；退出码：0；壁钟：0.560746968 秒。

UTC 2026-10-01T02:36:40.228936+00:00 → 2026-10-01T02:36:40.789651+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/07_write_report.py
```

原始回执：`logs/12_write_report.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 13_delivery_checks

cwd: `/mnt/data`；退出码：0；壁钟：0.565975054 秒。

UTC 2026-10-01T02:38:14.868693+00:00 → 2026-10-01T02:38:15.434635+00:00

```sh
python /mnt/data/bge_review_evidence/scripts/08_delivery_checks.py
```

原始回执：`logs/13_delivery_checks.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。

## 最终打包

`python /mnt/data/bge_review_evidence/scripts/run_recorded.py 14_package -- python /mnt/data/bge_review_evidence/scripts/package_evidence.py`

最终ZIP不能自包含自身最终哈希及打包结束日志；该机械打包步骤的精确命令、原始stdout/stderr、退出码、耗时与ZIP哈希均收在外部交付receipt。所有此前完成的核验/修订/交付检查命令（00—13）已完整包含在本ZIP内。
