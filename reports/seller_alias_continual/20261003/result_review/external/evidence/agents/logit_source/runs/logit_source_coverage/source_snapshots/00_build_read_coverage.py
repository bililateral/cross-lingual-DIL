#!/usr/bin/env python3
"""Record actual human source-reading coverage, separately from byte checks."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "input/project"
WORKSPACE = PROJECT / "reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace"
MANIFEST = WORKSPACE / "reports/job/run/manifest.json"
ER_MANIFEST = PROJECT / "reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job/run/manifest.json"
manifest = json.loads(MANIFEST.read_text())
er_manifest = json.loads(ER_MANIFEST.read_text())
er_rows = {r["path"]: r for r in er_manifest["source_files"]}
own_complete = {
    "docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md": "合同全文：固定0.25历史/.5MSE、恢复、当前校准、先盲门后评价、18+45集、固定比较与范围",
    "reports/documentation/20261002/logit_weight/decision.json": "完整确认问题、用户答复、预算与顺序的形成时快照",
    "schema/step28_logit_weight_policy.json": "完整175行：固定LOGIT主比较、baseline钉住、统计、权限与预算",
    "scripts/run_step28_er_check_linux_20261001.sh": "完整34行：study透传、CPU/时间/线程/原流；实际隔离入口另按原证据",
    "scripts/run_step28_er_weight_linux_20261001.sh": "完整40行：study透传、12h上限、原流；实际隔离入口另按原证据",
    "scripts/step28_er_weight.py": "完整237行：contract/source/paired baseline/ContinuationSupply/目标与唯一clip-step",
    "scripts/step28_er_weight_run.py": "完整551行：恢复、两群更新、六端点、记忆新目标、盲门、收集与正式入口",
    "scripts/step28_er_weight_evaluate.py": "完整188行：18+45复用、固定3比较、无自动selection、角色/计数/统计路由",
    "scripts/step28_er_weight_check.py": "完整232行：手写原生两臂、两probe梯度reference、counter fixture边界及原生证据产出",
    "tests/test_step28_er_weight_contracts.py": "完整687行：22既有/6新增测试内容、显式假文件夹具/微型真更新区别、关键系数和恢复边界",
    "scripts/step28_bge_continual.py": "完整301行：Memory target生命周期/精确序列化、Supply/current、LR、原目标更新",
    "scripts/step28_alias_ranking.py": "完整222行：hard top5和raw梯度、score委派；其他历史分支未被当作本次训练入口",
    "scripts/step28_continual_population.py": "完整210行：对称head、AdamW两组、state_dict全模型/optimizer保存恢复",
    "scripts/step28_continual_population_data.py": "完整258行：Group/配对索引、Memory Algorithm R/种子与schedule；旧Archive未作为本次实际输入路径",
}
partial = {
    "scripts/step28_bge_continual_run.py": [[1,245],[350,500]],
    "scripts/step28_chinese_base.py": [[200,315]],
    "tests/test_step28_bge_continual_contracts.py": [[1,190]],
}
records = []
for source in manifest["source_files"]:
    name = source["path"]
    payload = (WORKSPACE / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source["sha256"]
    same_er = name in er_rows and er_rows[name] == source
    own = name in own_complete
    assert own or same_er, name
    records.append({**source, "absolute_path": str(WORKSPACE / name),
        "full_read_by_logit_source_agent": own,
        "reader": "logit_source" if own else "er_source (shared exact-byte source)",
        "full_read_description": own_complete.get(name, "ER agent's full reading is reused after exact source-record equality; not claimed as a second independent full reading."),
        "partial_additional_read_by_logit_source": partial.get(name, []),
        "er_source_manifest_record_identical": same_er,
        "er_path_if_identical": str(PROJECT / name) if same_er else None})
additional = [
    PROJECT.parent / "REQUEST.zh.md", PROJECT / "AGENTS.md", PROJECT / "docs/RESEARCH_DISCIPLINE.zh.md",
    PROJECT / "reports/documentation/20261002/logit_weight/review/external/REVIEW.zh.md",
    PROJECT / "reports/documentation/20261002/logit_weight/review/report.zh.md",
    PROJECT / "docs/SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md",
    WORKSPACE.parent / "cpu/stdout.log", WORKSPACE.parent / "cpu/stderr.log",
    WORKSPACE / "reports/job/resource_usage.log", WORKSPACE / "reports/job/started.txt",
    WORKSPACE / "reports/job/finished.txt",
]
more = []
for p in additional:
    payload = p.read_bytes()
    more.append({"absolute_path": str(p), "bytes": len(payload),
                 "sha256": hashlib.sha256(payload).hexdigest(),
                 "full_read_by_logit_source_agent": True})
logit_rows = {r["path"]:r for r in manifest["source_files"]}
result = {"scope": "Actual source text inspection by the LOGIT agent plus explicitly identified exact-byte ER agent reuse, not hash checking alone.",
    "logit_source_records": records, "additional_full_reads": more,
    "changed_from_er31": [name for name in logit_rows if name in er_rows and logit_rows[name] != er_rows[name]],
    "added_to_er31": sorted(set(logit_rows) - set(er_rows)),
    "unchanged_from_er31": sorted(name for name in er_rows if name in logit_rows and logit_rows[name] == er_rows[name]),
    "own_full_logit_sources": sum(r["full_read_by_logit_source_agent"] for r in records),
    "shared_er_full_source_reuse": sum(not r["full_read_by_logit_source_agent"] for r in records),
    "file_data_checks": "read_accesses.json and source_correspondence.json in results_v2 document independent program reads; numerical vectors were processed by the reviewer program, not manually read as every scalar.",
    "unprovided": ["docs/AI_RESEARCH_HANDOFF.zh.md", "six old ER0.25 update-log bodies cited by old manifest", "formal target vectors/checkpoints/labels; forbidden and not requested", "full historical large external evidence ZIPs; no claim of reading them"]}
(HERE / "read_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(HERE / "v1_to_v2.diff").write_text("".join(difflib.unified_diff(
    (HERE / "independent_logit_reference.py").read_text().splitlines(keepends=True),
    (HERE / "independent_logit_reference_v2.py").read_text().splitlines(keepends=True),
    fromfile="independent_logit_reference.py (failed original)",
    tofile="independent_logit_reference_v2.py (explicit missing-old-log boundary)")))
print(json.dumps({k: result[k] for k in ("own_full_logit_sources", "shared_er_full_source_reuse", "changed_from_er31", "added_to_er31")}, ensure_ascii=False, indent=2))
