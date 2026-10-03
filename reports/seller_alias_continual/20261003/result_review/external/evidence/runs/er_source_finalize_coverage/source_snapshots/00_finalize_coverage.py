#!/usr/bin/env python3
"""Materialize the ER source-review reading log after semantic full-file reading.

This script inventories prior reads; hashing here does not perform or replace
semantic review. All file paths are supplied result-package paths or this review's
own artifacts. It imports no project code and opens no formal input bodies.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "input/project"
ER = "reports/seller_alias_continual/20261002/er_low_execution/20261002_124018"
manifest_path = f"{ER}/job/run/manifest.json"
manifest = json.loads((PROJECT / manifest_path).read_bytes())

subjects = {
    "docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md": "继承的首轮困难排序BGE合同、数据和预算、完整评价、23项验收及外推边界。",
    "docs/SELLER_ALIAS_ER_LOW.zh.md": "本轮唯一0.1干预、完整历史目标、共享恢复、0.25主对手与SEQ独立判定、已开发valid边界。",
    "docs/SELLER_ALIAS_ER_WEIGHT.zh.md": "前轮0.5/0.25合同、缓存和流匹配、历史结果复用和开发选择来源。",
    "reports/documentation/20261001/er_weight/decision.json": "前轮授权与已开发选择事实，不能追认为原预先计划。",
    "reports/documentation/20261002/er_low/decision.json": "0.1单点决定和运行授权的具体范围。",
    "schema/step28_alias_ranking_policy.json": "困难项top5、0.5系数及继承策略。",
    "schema/step28_bge_continual_policy.json": "s0、ABC/BCA/CAB、每站288、共享起点、6群1MiB与bootstrap种子。",
    "schema/step28_chinese_base_policy.json": "BGE输入维度、池化、BCE与排序、优化器、微批、分区和评价策略。",
    "schema/step28_er_low_policy.json": "唯一tenth候选、quarter主对手、预算、来源冻结和已有结果链接。",
    "schema/step28_er_weight_policy.json": "原half/quarter候选参数化，与本轮共享方法规则。",
    "scripts/run_step28_bge_continual_linux_20260930.sh": "首轮入口、py310、线程/CUDA环境和超时约束。",
    "scripts/run_step28_er_check_linux_20261001.sh": "必要手写CPU入口、时间/磁盘边界和原stdout/stderr留存。",
    "scripts/run_step28_er_weight_linux_20261001.sh": "低权重正式入口选择、12小时超时及原生环境。",
    "scripts/step28_alias_calibration.py": "未加权Bernoulli NLL及梯度、正仿射、L-BFGS-B边界与收敛判据、排序并列保持。",
    "scripts/step28_alias_ranking.py": "已知负例top5选择、索引detach与分数梯度保留、28查询全正例平均。",
    "scripts/step28_bge_continual.py": "种子分流、阶段Supply、AlgorithmR的Memory序列化与历史抽样、共享方法。",
    "scripts/step28_bge_continual_check.py": "手写群与原生层级probe、恢复/优化器检查的实施与边界。",
    "scripts/step28_bge_continual_evaluate.py": "实际域索引、O/N/Z/F_first/F/G/final_all公式、22列、5000配对整群和原23项。",
    "scripts/step28_bge_continual_run.py": "原首站保存/恢复、RNG、一次标签许可账本、预算、分支/评价盲门。",
    "scripts/step28_chinese_base.py": "从48/12分区、屏蔽title/description到BGE活图、mean/std池化、378无序边损失、score/persistence/evaluation整链。",
    "scripts/step28_continual_expression_run.py": "实际被调用的public_inputs及labels资格入口；其余旧运行代码也全文读以区分活动路径。",
    "scripts/step28_continual_population.py": "对称边头、AdamW参数组、模型/优化器真实张量摘要及torch保存加载。",
    "scripts/step28_continual_population_data.py": "Group结构、pairs排序、label对齐、AlgorithmR；旧Archive含其他数据入口但本輪无活动调用。",
    "scripts/step28_continual_population_evaluate.py": "22指标定义、AP/梯形PR-AUC、阈值计数、排序并列和宏/池化区别。",
    "scripts/step28_continual_population_run.py": "canonical/hash/json/模型档案与保存公共依赖；旧独立main未调用。",
    "scripts/step28_er_weight.py": "31来源、旧记录与共享起点身份、.1完整历史反向、单次总裁剪与Adam更新。",
    "scripts/step28_er_weight_check.py": "quarter/tenth同起点手写原生分支、只两次实际更新及Adam计数rebase、probe记录。",
    "scripts/step28_er_weight_evaluate.py": "18新增加81复用、99集合对齐，主差tenth-quarter及SEQ，选择不换对手。",
    "scripts/step28_er_weight_run.py": "实际当前/历史流、六续训阶段、首状态恢复、正仿射校准、原生盲分数、统一valid门与保存。",
    "tests/test_step28_bge_continual_contracts.py": "原生/微型合同案例的实际断言和覆盖，不把测试数量当科学结论。",
    "tests/test_step28_er_weight_contracts.py": "参数化低权重、梯度缩放、计数与数据门、重放/失效案例的具体断言。",
}
assert set(subjects) == {r["path"] for r in manifest["source_files"]}

def record(path, subject, origin="project"):
    file_path = PROJECT / path if origin == "project" else PROJECT.parent / path
    raw = file_path.read_bytes()
    return {
        "path": path,
        "path_root": origin,
        "absolute_input_path": str(file_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(raw.decode("utf-8").splitlines()),
        "full_text_semantic_read": True,
        "reading_mode": "full text was read and reasoned about in the review; inventory created afterward",
        "subjects_reviewed": subject,
    }

sources = []
for expected in manifest["source_files"]:
    rec = record(expected["path"], subjects[expected["path"]])
    assert (rec["bytes"], rec["sha256"]) == (expected["bytes"], expected["sha256"])
    rec["frozen_manifest_record_matches"] = True
    sources.append(rec)

additional = [
    record("REQUEST.zh.md", "先读完整请求，确认只审两个完成的开发点、来源复用与禁止访问边界。", "input"),
    record("AGENTS.md", "项目科学工作纪律及当前请求的优先级。"),
    record("docs/RESEARCH_DISCIPLINE.zh.md", "负结果、预算、一次监督、授权与证据分类纪律。"),
    record("docs/SELLER_ALIAS_ER_LOW_RESULT.zh.md", "结果宣称、对SEQ局部退化与开发选择边界；数值总裁定由独立统计审查合并。"),
    record("reports/documentation/20261002/er_low/review/external/REVIEW.zh.md", "历史外审完整叙述、原失败修订与环境；其旧evidence.zip未在本包，未冒称读过。"),
    record("reports/documentation/20261002/er_low/review/report.zh.md", "主执行方对前次外审的处置和原生核验/启动补记。"),
    record(manifest_path, "实际完整manifest文字与独立字段解析；正式输入正文及权重只显示摘要，未打开。"),
    record(f"{ER}/job/execution.json", "原生py310/CUDA/GPU、31来源、授权和CPU审计身份。"),
    record(f"{ER}/job/train.log", "72个每24更新进度事件、六段完成及末尾COMPLETE，FutureWarning不等同失败。"),
    record(f"{ER}/cpu/stdout.log", "原生CPU真实quarter/tenth顺序、实际两更新分支和耗时。"),
    record(f"{ER}/cpu/stderr.log", "原生测试断言结果与FutureWarning原文。"),
    record(f"{ER}/cpu/resource_usage.log", "原生CPU6:04.25墙时、资源和退出码0。"),
]

independent_result = json.loads((HERE / "independent_er_source_result.json").read_bytes())
already_full = {r["path"] for r in sources + additional if r["path_root"] == "project"}
structured = []
for path, rec in independent_result["input_files"].items():
    if path in already_full:
        continue
    structured.append({
        "path": path,
        **rec,
        "full_text_semantic_read": False,
        "reading_mode": "full saved file parsed, relevant fields independently checked by independent_er_source_v2.py",
        "verification_result": "independent_er_source_result.json; all original bytes hashed by reader",
    })

payload = {
    "schema_version": 1,
    "reviewer": "ER source-review subtask within the current GPT-6 Astra Pro context",
    "identity_boundary": "Does not authenticate a separate webpage GPT 6 Pro model session.",
    "scope": "ER0.1 frozen31 and shared directly called scientific dependencies, plus necessary saved metadata/handwrittenCPU evidence",
    "manifest": manifest_path,
    "frozen_full_read_count": len(sources),
    "full_read_sources": sources,
    "additional_full_read": additional,
    "saved_structured_evidence_reads": structured,
    "shared_content_rule": "LOGIT reviewer may map a byte-identical SHA to this full reading; differing source versions require its own full read and are not replaced by this record.",
    "limits": [
        "No formal text, formal label, cache body, model weight, test/owners or server access.",
        "Reading/hashing metadata does not prove unavailable model tensors or input contents.",
        "A parsed saved file is explicitly distinguished from a semantic full-source read.",
        "No third-party torch/transformers/scipy library source was supplied or audited; native version identity was read and this webpage used independent NumPy only.",
    ],
}
out = HERE / "read_coverage.json"
with out.open("x", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
for path in (out, HERE / "notes.md"):
    raw = path.read_bytes()
    print(json.dumps({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, ensure_ascii=False))
print(json.dumps({"frozen_sources_fully_read": len(sources), "additional_full_text_files": len(additional), "structured_saved_evidence_files": len(structured)}, ensure_ascii=False))
