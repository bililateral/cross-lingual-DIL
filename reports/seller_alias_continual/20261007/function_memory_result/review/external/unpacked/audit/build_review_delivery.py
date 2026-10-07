#!/usr/bin/env python3
"""Package completed read-only audit, original input and full Chinese review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-work", type=Path, required=True)
    ap.add_argument("--original-zip", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    work, dest = args.audit_work.resolve(), args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    bundle = dest / "function_memory_result_external_review"
    bundle.mkdir(parents=True, exist_ok=True)
    for group in ("provenance", "metrics", "mechanism", "root_counts"):
        target = bundle / "audit" / group
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted((work / group).iterdir()):
            if source.is_file():
                shutil.copyfile(source, target / source.name)
    shutil.copyfile(args.original_zip, bundle / "original_input.zip")
    shutil.copyfile(Path(__file__), bundle / "audit/build_review_delivery.py")
    shutil.copyfile(work / "review_main.zh.txt", bundle / "audit/review_main.zh.txt")

    stats = json.loads((work / "metrics/all_recomputed_statistics.json").read_text())
    matrix_result = json.loads((work / "metrics/result.json").read_text())
    counts = json.loads((work / "root_counts/result.json").read_text())
    mechanism = json.loads((work / "mechanism/mechanism_checks.json").read_text())
    provenance = json.loads((work / "provenance/provenance_result.json").read_text())
    assert matrix_result["status"] == counts["status"] == "PASS"
    assert provenance["status"] == "PASS_WITH_DECLARED_EVIDENCE_LIMITS"
    assert matrix_result["statistics"]["numeric_values"] == 9504
    assert matrix_result["passed"] == 0 and not counts["failures"]
    cols = ['average_precision', 'trapezoidal_pr_auc', 'roc_auc', 'recall_at_fpr_1pct',
            'brier', 'log_loss', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy',
            'mcc', 'map', 'mrr', 'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
            'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10']
    text = (work / "review_main.zh.txt").read_text(encoding="utf-8")
    for ep in ("O", "A2", "N", "Z", "final_all", "F_first", "F", "G"):
        text += f"\n\n## 附表：{ep} 全部22指标（primary）\n\n"
        text += "|指标|LOGIT0.1|本方法|本方法减基线|差值条件95%区间|\n|---|---:|---:|---:|---|\n"
        for metric in cols:
            b = stats["endpoints"]["logit0.1"]["primary"][ep][metric]["mean"]
            c = stats["endpoints"]["function_memory"]["primary"][ep][metric]["mean"]
            d = stats["delta"][ep][metric]
            lo, hi = d["conditional_95pct_interval"]
            text += f"|{metric}|{b:.9f}|{c:.9f}|{d['mean']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|\n"
    text += "\n\n## 附表：六训练段保存标量独立汇总\n\n"
    text += "|顺序/阶段|首步D|D均值|D最大|当前B均值|0.1历史B均值|0.5D均值|裁剪步数|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for name, x in mechanism["segments"].items():
        text += (f"|{name}|{x['D_first']:.12g}|{x['D_mean']:.9f}|{x['D_max']:.9f}|"
                 f"{x['current_B_mean']:.9f}|{x['weighted_history_B_mean']:.9f}|{x['weighted_D_mean']:.9f}|{x['norm_over_1']}|\n")
    text += "\n首域没有本轮新增更新，不把空序列的0占位当作训练标量实测。以上D及梯度范数均为保存标量核对，未重算模型梯度。\n"
    text += "\n最终结论：本次固定配置的有效开发负结果可以接收；原五条件0/5，相对LOGIT0.1未达改进目标。\n"
    report_name = "FUNCTION_MEMORY_RESULT.external_review.zh.txt"
    (bundle / report_name).write_text(text, encoding="utf-8")
    shutil.copyfile(bundle / report_name, dest / report_name)
    package_identity = {"original_filename": "review_input.zip", **identity(args.original_zip)}
    with zipfile.ZipFile(args.original_zip) as z:
        package_identity.update(members=len(z.infolist()), manifest_payloads=252,
                                crc_bad_member=z.testzip())
    assert package_identity["bytes"] == 18320431
    assert package_identity["sha256"] == "a0c8ae95371cbf1ff0667171f6e0909b56509ddadda9601263093b57d1d7a72d"
    (bundle / "audit/package_identity.json").write_text(json.dumps(package_identity, indent=2) + "\n")
    review_scope = {
        "review_type": "Completed fixed-point result review, not launch approval",
        "review_date": "2026-10-07", "model_identity_provided_to_reviewer": "GPT-6 Astra Pro",
        "browser_selector_independently_verified": False,
        "independent_source_scripts": ["audit/provenance/independent_provenance_audit.py",
            "audit/provenance/additional_record_checks.py", "audit/metrics/independent_metrics_audit.py",
            "audit/mechanism/check_mechanism_logs.py", "audit/root_counts/audit_saved_counts.py"],
        "actual_environment": matrix_result["environment"], "formal_training_executed": False,
        "project_native_executed": False, "historical_tests_rerun": False,
        "model_memory_labels_text_or_credentials_loaded": False,
        "network_research_or_project_server_access": False,
        "saved_scores_read": "Current 28 development score sets for affine and positive-count checks; no labels reconstructed",
        "statistic_entries": 1584, "statistic_numbers": 9504,
        "max_statistic_difference": matrix_result["statistics"]["max_absolute_error"],
        "group_count_classification_numbers": 19800,
        "current_blockers": 0, "current_reproducibility_defects": 0,
        "criteria_passed": 0, "criteria_total": 5,
        "reading_scope": "See main report section 10 and per-audit input/reading manifests; identity hashing is distinct from semantic reading",
        "reviewer_inspection_corrections": [
            "A section-title probe used '## 7.' while actual header is '## 7'; read actual header after rg lookup.",
            "A read probe requested review/submission.json, which is referenced but absent in input; not a required frozen-result artifact.",
            "A source-reading command first used an incorrect audit script basename; corrected to the actual listed file.",
            "These were reviewer read/inspection issues, not project training failures; the five independent audit scripts completed successfully."
        ],
    }
    (bundle / "audit/review_scope.json").write_text(json.dumps(review_scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = '''# 审查交付说明

主原文：FUNCTION_MEMORY_RESULT.external_review.zh.txt，含完整中文裁决、证据边界、五条件、八端点全部22指标和标量附表。
original_input.zip 是用户提交的 review_input.zip 原字节副本；完整嵌套历史仍在其中，未修改。
audit/ 内是本次五份实际独立脚本、真实stdout/stderr、机器输出和阅读范围。旧历史脚本未在本轮重新执行。
manifest.json 登记本包除manifest自身外的每个文件字节数/SHA，既有子清单按原字节保留。

## 复算

将本包和original_input.zip分别解压。以下命令只读取已保存记录、分数、群指标和群计数；不导入项目模块，不安装环境，不运行execute/native，不拟合或载入模型，不读正文/逐对标签/Memory。
使用已有Python与NumPy，先创建一个新的输出目录。将INPUT_ROOT改为original_input.zip的解压根，OUTPUT_ROOT改为新的结果目录。

python audit/provenance/independent_provenance_audit.py --input INPUT_ROOT --out OUTPUT_ROOT/provenance
python audit/provenance/additional_record_checks.py --input INPUT_ROOT --out OUTPUT_ROOT/provenance
python audit/metrics/independent_metrics_audit.py --input-root INPUT_ROOT --output-dir OUTPUT_ROOT/metrics
python audit/mechanism/check_mechanism_logs.py --root INPUT_ROOT --output OUTPUT_ROOT/mechanism
python audit/root_counts/audit_saved_counts.py INPUT_ROOT OUTPUT_ROOT/root_counts/result.json

前一条provenance脚本创建输出目录，后一条补充脚本沿用该目录。不要使用python -O，以免关闭审查断言。
运行时间、环境字符串和阅读清单中的绝对路径随复算环境变化；应核对数值、固定来源、通过状态及记录范围，不要求整个输出JSON的字节恒等。
本轮数值容差1e-12；原实际输出最大统计差4.44e-16，分类/仿射/共享矩阵差0。

审查者在读取标题/路径时的纠正在audit/review_scope.json如实说明；没有改动项目来源，也不计为正式训练失败。
'''
    (bundle / "README.zh.txt").write_text(readme, encoding="utf-8")
    manifest = {"schema": "completed_result_review_payload_manifest", "review_date": "2026-10-07",
                "model_identity": "GPT-6 Astra Pro", "files": []}
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p != bundle / "manifest.json":
            manifest["files"].append({"path": p.relative_to(bundle).as_posix(), **identity(p)})
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = dest / "function_memory_result_external_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(bundle).as_posix(),
                        compress_type=zipfile.ZIP_STORED if p.name == "original_input.zip" else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert len(z.infolist()) == len(manifest["files"]) + 1
        for rec in manifest["files"]:
            raw = z.read(rec["path"])
            assert len(raw) == rec["bytes"] and hashlib.sha256(raw).hexdigest() == rec["sha256"]
        assert z.read(report_name) == (dest / report_name).read_bytes()
        assert z.read("original_input.zip") == args.original_zip.read_bytes()
    result = {"report": {"path": str(dest / report_name), **identity(dest / report_name)},
              "zip": {"path": str(zip_path), **identity(zip_path), "members": len(manifest["files"]) + 1,
                      "payloads": len(manifest["files"])}, "all_payload_hashes_verified": True,
              "report_same_in_zip": True, "original_input_preserved_exactly": True}
    (dest / "delivery_identity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
