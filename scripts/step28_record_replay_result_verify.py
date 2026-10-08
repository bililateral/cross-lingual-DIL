"""Audit completed record-replay evidence on Linux using saved arrays only.

No training imports, model loading, text/label parsing, or Memory deserialization.
The original job is read-only; an independent output directory is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import numpy as np


ORDERS = ("ABC", "BCA", "CAB")
ROLES = ("raw", "stage-cal", "first-cal", "primary")
ENDPOINTS = ("O", "N", "Z", "F_first", "F", "G", "final_all", "A2")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path, record: dict) -> Path:
    path = root / record["path"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    assert path.stat().st_size == record["bytes"], str(path)
    assert digest.hexdigest() == record["sha256"], str(path)
    return path


def seed(value: int, *parts: object) -> int:
    payload = (json.dumps([value, *parts], separators=(",", ":")) + "\n").encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63-1)


def point_name(order: str, arm: str, stage: int) -> str:
    return order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"


def classification(counts: dict) -> dict:
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    denominator = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** .5
    return {"precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn),
            "f1": 2*tp/(2*tp+fp+fn), "specificity": tn/(tn+fp),
            "balanced_accuracy": .5*(tp/(tp+fn)+tn/(tn+fp)),
            "mcc": (tp*tn-fp*fn)/denominator if denominator else 0.}


def main(job: Path, workspace: Path, output: Path) -> None:
    started = time.monotonic()
    assert not output.exists(), "New output directory required"
    completion = read(job / "completion.json")
    assert completion["status"] == "COMPLETE_RECORD_REPLAY_DEVELOPMENT"
    assert not (job / "failure.json").exists()
    assert completion["access"] == read(job / "access.json") == dict(train=1, valid=1, heldout=0, owners=0)
    before = read(job / "before_valid.json")
    assert before["status"] == "PASS_COMPLETE_BLIND_GATE" and before["access"] == dict(train=1, valid=0, heldout=0, owners=0)
    manifest = read(verify(job, before["manifest"]))
    saved = read(verify(job, completion["evaluation"]))
    collected = read(verify(job / "evaluation", saved["collected"]))
    assert collected["status"] == "ALL_46_METRIC_COUNT_SETS_SAVED"
    assert sum(len(v) for v in collected["points"].values()) == 46
    assert completion["physical_updates"] == manifest["physical_updates"] == 4320
    assert manifest["gradient_group_presentations"] == 7776
    expected_points = {point_name(o, a, s) for o in ORDERS for a in ("C", "S") for s in (1, 2, 3)}
    assert set(manifest["points"]) == set(manifest["training"]) == expected_points
    assert set(collected["points"]) == expected_points | {"initial"}
    for key, limit in (("elapsed_seconds", 86400), ("peak_output_bytes", 32*2**30),
                       ("peak_cuda_reserved_bytes", 28*2**30), ("peak_rss_bytes", 64*2**30)):
        assert completion["budget"][key] <= limit
    execution = read(job / "execution.json")
    assert execution["sources"] == manifest["source_files"] == collected["source_files"] == saved["source_files"]
    for record in execution["sources"]:
        verify(workspace, record)
    verify(workspace, execution["gate"])
    partition = read(verify(job / "run", manifest["partition"]))
    uid_domain = {r["group_uid"]: r["domain"] for r in partition["fit"]}
    logs, diagnostics = {}, {}
    for order in ORDERS:
        assert manifest["starts"][order+"_C"] == manifest["starts"][order+"_S"]
        for arm in ("C", "S"):
            members, seen, origins, previous = [], 0, {}, None
            retention_rng = random.Random(seed(20260930, order, "retention"))
            for stage in (1, 2, 3):
                name = point_name(order, arm, stage)
                point = manifest["points"][name]
                assert read(job / "run/points" / (name+".json")) == point
                tr = read(verify(job / "run", manifest["training"][name]))
                fit = sorted(uid for uid, d in uid_domain.items() if d == order[stage-1])
                rng = random.Random(seed(20260918, order, stage, "current"))
                schedule = []
                for _ in range(6):
                    shuffled = fit.copy()
                    rng.shuffle(shuffled)
                    schedule.extend(shuffled)
                assert tr["current_ids"] == schedule and len(tr["rows"]) == tr["updates"] == 288
                assert point["full_restore_verified"] and tr["adam_step"] == point["adam_step"] == stage*288
                draws_rng = random.Random(seed(20260930, order, stage, "history_draws"))
                history = [members[draws_rng.randrange(6)] for _ in range(288)] if stage > 1 else []
                assert tr["history_ids"] == history
                if stage > 1:
                    for k in ("members", "seen", "origins", "tables"):
                        assert tr["memory_before"][k] == tr["memory_after"][k] == previous[k]
                    assert tr["memory_before"]["count"] == 0 and tr["memory_after"]["count"] == 288
                    assert tr["memory_before"]["stage"] == tr["memory_after"]["stage"] == stage
                for i, row in enumerate(tr["rows"], 1):
                    assert row["step"] == i and row["current_uid"] == schedule[i-1]
                    assert row["history_uid"] == (history[i-1] if history else None)
                    total = row["current"]["supervised"]
                    for role in ("current", "history") if stage > 1 else ("current",):
                        assert all(np.isfinite(v) for v in row[role].values())
                        assert row[role]["record_gradient_norm"] >= 0
                    if stage > 1:
                        h = row["history"]
                        d = (h["mse0"]+h["mse1"])/2 if arm == "C" else h["mse0"]
                        np.testing.assert_allclose(h["distillation"], d, rtol=2e-6, atol=2e-6)
                        total += .1*h["supervised"]+.5*d
                    np.testing.assert_allclose(row["total"], total, rtol=2e-6, atol=2e-6)
                    lr = 1e-5*(i/29 if i <= 29 else (288-i)/259)
                    assert np.isclose(row["encoder_lr"], lr, rtol=1e-12, atol=1e-16) and row["head_lr"] == .001
                    assert np.isfinite(row["gradient_norm"]) and row["gradient_norm"] >= 0
                if stage < 3:
                    for uid in fit:
                        seen += 1
                        index = len(members) if len(members) < 6 else retention_rng.randrange(seen)
                        if index < 6:
                            if index == len(members):
                                members.append(uid)
                            else:
                                origins.pop(members[index])
                                members[index] = uid
                            origins[uid] = stage
                summary = point["memory_summary"]
                assert summary["members"] == members and summary["seen"] == seen and summary["origins"] == origins
                assert summary["bytes"] == point["memory"]["bytes"] <= 2**20
                assert summary["sha256"] == point["memory"]["sha256"] and set(summary["tables"]) == set(members)
                verify(job / "run", point["memory"])  # File identity only; no deserialization.
                if previous:
                    for uid in set(previous["tables"]) & set(summary["tables"]):
                        assert previous["tables"][uid] == summary["tables"][uid]
                mapping = read(verify(job / "run", point["mapping"]))
                assert mapping["status"] == "PASS_CALIBRATION_FIT" and mapping["a"] > 0
                assert point["calibration_ids"] == [r["group_uid"] for r in partition["calibration"] if r["domain"] == order[stage-1]]
                scores = {r: np.load(verify(job / "run", rec), allow_pickle=False) for r, rec in point["scores"].items()}
                for role, score in scores.items():
                    assert score.shape == ((12, 378) if role == "calibration" else (60, 378)) and np.isfinite(score).all()
                for role, key in (("stage-cal", "stage"), ("first-cal", "first")):
                    np.testing.assert_array_equal(scores[role], scores["raw"].astype(np.float64)*point["maps"][key]["a"]+point["maps"][key]["b"])
                logs[name] = tr
                diagnostics[name] = {"memory_bytes": summary["bytes"], "members_by_domain": dict(Counter(uid_domain[u] for u in members)),
                    "training_seconds": tr["training_seconds"], "clipped_steps": sum(r["gradient_norm"] > 1 for r in tr["rows"]),
                    "calibration": {k: mapping[k] for k in ("a", "b", "final_nll", "raw_brier", "calibrated_brier")},
                    "mean_current_loss": float(np.mean([r["current"]["supervised"] for r in tr["rows"]])),
                    "mean_history_loss": float(np.mean([r["history"]["supervised"] for r in tr["rows"]])) if stage > 1 else None,
                    "mean_distillation": float(np.mean([r["history"]["distillation"] for r in tr["rows"]])) if stage > 1 else None}
                previous = summary
        for stage in (2, 3):
            for key in ("current_ids", "history_ids"):
                assert logs[point_name(order, "C", stage)][key] == logs[point_name(order, "S", stage)][key]
    assert sum(len(t["rows"]) for t in logs.values()) == 4320
    baseline = workspace / saved["reference"]["linux_root"]
    basecols = {name: read(verify(baseline, {"path": name, **rec})) for name, rec in saved["reference"]["collections"].items()}
    for col in basecols.values():
        for key in ("group_ids", "domains", "metric_columns"):
            assert col[key] == collected[key]
    columns = collected["metric_columns"]
    assert len(set(collected["group_ids"])) == 60 and len(columns) == 22
    rows = {d: np.flatnonzero(np.array(collected["domains"]) == d) for d in "ABC"}
    assert all(len(r) == 20 for r in rows.values())
    rank = [i for i, k in enumerate(columns) if k not in ("brier", "log_loss", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")]
    losses = [columns.index(k) for k in ("brier", "log_loss")]
    arrays, counts = {}, {}
    checked_sets = set()
    for arm in ("C", "S", "LOGIT0.1"):
        arrays[arm], counts[arm] = {}, {}
        for order in ORDERS:
            for stage in (1, 2, 3):
                if arm in ("C", "S"):
                    root, col, name = job / "evaluation", collected, point_name(order, arm, stage)
                elif stage == 1:
                    root, col, name = baseline / "reference", basecols["reference/collected.json"], order+"_shared"
                else:
                    root, col, name = baseline, basecols["collected.json"], f"{order}_logit_tenth_stage{stage}"
                a, c = {}, {}
                for role in ROLES[:3]:
                    rec = col["points"][name][role]
                    a[role] = np.load(verify(root, rec["matrix"]), allow_pickle=False)
                    assert a[role].shape == (60, 22) and a[role].dtype == np.float64 and np.isfinite(a[role]).all()
                    c[role] = read(verify(root, rec["counts"]))
                    assert len(c[role]) == 60
                    for i, row in enumerate(c[role]):
                        assert all(type(row[k]) is int and row[k] >= 0 for k in ("tp", "fp", "fn", "tn"))
                        assert row["tp"]+row["fn"] == 20 and row["fp"]+row["tn"] == 358
                        for metric, value in classification(row).items():
                            np.testing.assert_allclose(a[role][i, columns.index(metric)], value, rtol=0, atol=1e-12)
                    checked_sets.add(str(root / rec["matrix"]["path"]))
                for role in ROLES[1:3]:
                    np.testing.assert_array_equal(a[role][:, rank], a["raw"][:, rank])
                a["primary"] = a["stage-cal"].copy()
                a["primary"][:, rank] = a["raw"][:, rank]
                arrays[arm][order, stage], counts[arm][order, stage] = a, c
    initial = collected["points"]["initial"]["raw"]
    initial_matrix = np.load(verify(job / "evaluation", initial["matrix"]), allow_pickle=False)
    initial_counts = read(verify(job / "evaluation", initial["counts"]))
    assert initial_matrix.shape == (60, 22) and len(initial_counts) == 60
    for i, row in enumerate(initial_counts):
        assert row["tp"]+row["fn"] == 20 and row["fp"]+row["tn"] == 358
        for metric, value in classification(row).items():
            np.testing.assert_allclose(initial_matrix[i, columns.index(metric)], value, rtol=0, atol=1e-12)
    def field(arm: str, role: str, ep: str) -> np.ndarray:
        out = np.zeros((3, 60, 22))
        for oi, order in enumerate(ORDERS):
            def cell(stage: int, arrival: int) -> np.ndarray:
                result = np.zeros((60, 22))
                ids = rows[order[arrival-1]]
                result[ids] = arrays[arm][order, stage][role][ids]
                return result
            r11, r12, r22, r23, r31, r32, r33 = (cell(s, a) for s, a in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)))
            value = {"O": (r31+r32)/2, "N": (r22+r33)/2, "Z": r33, "F_first": r11-r31,
                     "F": (r11-r31+r22-r32)/2, "G": (r22-r12+r33-r23)/2, "final_all": (r31+r32+r33)/3, "A2": r22}[ep]
            if ep in ("F_first", "F", "G"):
                value[:, losses] *= -1
            out[oi] = value
        return out
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, (5000, 3, 20))
    differences = []
    def stats(f: np.ndarray, expected: dict) -> dict:
        per_order = f.sum(1)/20
        means = per_order.mean(0)
        averaged = f.mean(0)
        boot = sum(averaged[rows[d]][draws[:, di]].mean(1) for di, d in enumerate("ABC"))
        ci = np.quantile(boot, [.025, .975], axis=0, method="linear")
        result = {}
        for k, name in enumerate(columns):
            actual = np.r_[means[k], ci[:, k], per_order[:, k]]
            target = np.r_[expected[name]["mean"], expected[name]["conditional_95pct_interval"], [expected[name]["per_order"][o] for o in ORDERS]]
            np.testing.assert_allclose(actual, target, rtol=0, atol=1e-12)
            differences.extend(np.abs(actual-target).tolist())
            result[name] = {"mean": float(means[k]), "conditional_95pct_interval": ci[:, k].tolist()}
        return result
    for arm in arrays:
        for role in ROLES:
            for ep in ENDPOINTS:
                stats(field(arm, role, ep), saved["endpoints"][arm][role][ep])
    checks = {}
    for other in ("S", "LOGIT0.1"):
        primary = {}
        for role in ROLES:
            for ep in ENDPOINTS:
                recomputed = stats(field("C", role, ep)-field(other, role, ep), saved["comparisons"]["C-"+other]["delta"][role][ep])
                if role == "primary":
                    primary[ep] = recomputed
        lower = lambda ep, metric: primary[ep][metric]["conditional_95pct_interval"][0]
        checks["C-"+other] = {"O_MAP_lower_positive": lower("O", "map") > 0,
            "final_all_MAP_mean_positive": primary["final_all"]["map"]["mean"] > 0,
            "A2_MAP_lower_ge_minus_point01": lower("A2", "map") >= -.01,
            "Z_MAP_lower_ge_minus_point01": lower("Z", "map") >= -.01,
            "O_AP_lower_ge_minus_point01": lower("O", "average_precision") >= -.01}
        assert checks["C-"+other] == saved["comparisons"]["C-"+other]["checks"]
        assert all(checks["C-"+other].values()) == saved["comparisons"]["C-"+other]["pass"]
    assert all(all(v.values()) for v in checks.values()) == saved["development_criteria_pass"] == completion["development_criteria_pass"]
    fixed, paths = {}, {}
    for arm in arrays:
        fixed[arm], paths[arm] = {}, {}
        for ep, pairs in {"O": ((3, 1), (3, 2)), "N": ((2, 2), (3, 3)), "Z": ((3, 3),)}.items():
            total = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
            for order in ORDERS:
                for s, a in pairs:
                    for i in rows[order[a-1]]:
                        for k in total:
                            total[k] += counts[arm][order, s]["stage-cal"][i][k]
            fixed[arm][ep] = {**total, **classification(total), "fpr": total["fp"]/(total["fp"]+total["tn"])}
        for order in ORDERS:
            paths[arm][order] = {str(s): {d: {metric: float(arrays[arm][order, s]["primary"][rows[d], columns.index(metric)].mean()) for metric in ("map", "average_precision", "brier", "log_loss")} for d in "ABC"} for s in (1, 2, 3)}
    output.mkdir(parents=True)
    result = {"status": "PASS_SAVED_RESULT_INDEPENDENT_VERIFICATION", "seconds": time.monotonic()-started,
        "scope": "Saved matrices/counts, endpoints, paired bootstrap, schedules, recorded restores and identities; no model load, labels/text or Memory deserialization",
        "source_files": len(execution["sources"]), "current_metric_count_sets": 46,
        "metric_sets_including_baseline": len(checked_sets)+1, "statistic_numbers_checked": len(differences),
        "maximum_statistic_absolute_error": max(differences), "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "checks": checks, "training_points": diagnostics, "fixed_half": fixed, "stage_metrics": paths,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["记录作用表完整结果表；所有数字来自已核对的保存评价。", "绝对值为均值；差值为均值 [配对条件95%下界, 上界]。F/F_first为遗忘，G为获得，Brier/log_loss方向按合同统一。", "三顺序不是三个种子；逐项条件区间不是联合95%保证。", ""]
    for role in ROLES:
        for ep in ENDPOINTS:
            lines.extend([f"{role} / {ep}", "metric | C | S | LOGIT0.1 | C-S mean [CI] | C-LOGIT0.1 mean [CI]"])
            for metric in columns:
                absolute = [saved["endpoints"][a][role][ep][metric]["mean"] for a in ("C", "S", "LOGIT0.1")]
                delta = [saved["comparisons"]["C-"+a]["delta"][role][ep][metric] for a in ("S", "LOGIT0.1")]
                lines.append(metric+" | "+" | ".join(f"{x:.9f}" for x in absolute)+" | "+" | ".join(f"{r['mean']:+.9f} [{r['conditional_95pct_interval'][0]:+.9f}, {r['conditional_95pct_interval'][1]:+.9f}]" for r in delta))
            lines.append("")
    (output / "tables.zh.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "seconds", "statistic_numbers_checked", "maximum_statistic_absolute_error", "checks")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.job, args.workspace, args.output)
