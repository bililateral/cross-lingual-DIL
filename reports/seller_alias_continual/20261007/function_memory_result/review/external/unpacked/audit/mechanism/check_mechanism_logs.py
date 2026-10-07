"""Independent, read-only audit of published scalar logs and memory summaries.

Only stdlib/NumPy and saved evidence are used. No project code is imported, no
models/Memory payloads/text/pair labels are opened, and no model is fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import time
import zipfile

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    reads: dict[str, dict] = {}
    base = root / "reports/seller_alias_continual/20261007/function_memory_result"

    def payload(path: Path) -> bytes:
        assert path.is_relative_to(root), path
        assert path.suffix not in {".pt", ".bin", ".safetensors"}, path
        raw = path.read_bytes()
        reads[str(path.relative_to(root))] = {
            "path": str(path.relative_to(root)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return raw

    def read(path: Path):
        return json.loads(payload(path))

    def array(path: Path, record: dict) -> np.ndarray:
        raw = payload(path)
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        val = np.load(io.BytesIO(raw), allow_pickle=False)
        assert val.shape == (60, 22) and val.dtype == np.float64
        assert np.isfinite(val).all()
        return val

    def seed_for(seed: int, *parts) -> int:
        encoded = (json.dumps([seed, *parts], ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode()
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)

    policy = read(base / "source/schema/step28_function_memory_policy.json")
    bge = read(base / "source/schema/step28_bge_continual_policy.json")
    assert policy["loss"] == dict(coordinate="fp16_forward_fp32_straight_through",
                                  current=1, epsilon=.001, function=.5,
                                  trace=3, dimension=129, history=.1)
    partition = read(base / "job/run/partition.json")
    manifest = read(base / "job/run/manifest.json")
    fit_domain = {r["group_uid"]: r["domain"] for r in partition["fit"]}
    assert len(fit_domain) == 144
    assert Counter(fit_domain.values()) == dict(A=48, B=48, C=48)
    assert Counter(r["domain"] for r in partition["calibration"]) == dict(A=12, B=12, C=12)
    assert Counter(r["domain"] for r in partition["development"]) == dict(A=20, B=20, C=20)
    report = {"scope": "saved_scalar_logs_and_metric_summaries_only",
              "project_code_imported": False, "model_or_memory_loaded": False,
              "pair_labels_or_text_read": False, "training_or_native_executed": False,
              "python": platform.python_version(), "numpy": np.__version__,
              "segments": {}, "points": {}, "mechanism_claims": {}}
    total_updates = 0
    all_norms, all_d = [], []
    scalar_formula_max_error = 0.
    supervised_formula_max_error = 0.
    for order in ("ABC", "BCA", "CAB"):
        keep: list[str] = []
        seen = 0
        retention_rng = random.Random(seed_for(bge["memory_seed"], order, "retention"))
        for stage in (1, 2, 3):
            name = f"{order}_function_memory_stage{stage}"
            tr = read(base / f"job/run/updates/{name}.json")
            point = read(base / f"job/run/points/{name}.json")
            fit = sorted(uid for uid, d in fit_domain.items() if d == order[stage - 1])
            expected_current: list[str] = []
            if stage > 1:
                current_rng = random.Random(seed_for(bge["schedule_seed"], order, stage, "current"))
                for _ in range(6):
                    epoch = fit.copy()
                    current_rng.shuffle(epoch)
                    expected_current.extend(epoch)
            assert tr["current_ids"] == expected_current
            assert len(tr["updates"]) == (0 if stage == 1 else 288)
            assert tr["order"] == order and tr["stage"] == stage
            assert tr["adam_step"] == point["adam_step"] == stage * 288
            history_rng = random.Random(seed_for(bge["memory_seed"], order, stage, "history_draws"))
            expected_history = [keep[history_rng.randrange(6)] for _ in expected_current]
            assert tr["history_ids"] == expected_history
            assert not set(tr["current_ids"]) & set(tr["history_ids"])
            rows = tr["updates"]
            for i, r in enumerate(rows, 1):
                assert np.isfinite(list(r.values())).all()
                assert (r["step"], r["stage"], r["adam_step"]) == (i, stage, (stage - 1) * 288 + i)
                expected_lr = 1e-5 * (i / 29 if i <= 29 else (288 - i) / 259)
                assert r["encoder_lr"] == expected_lr and r["head_lr"] == .001
                error = abs(r["total"] - (r["current_total"] + .1 * r["history_total"] + .5 * r["function"]))
                assert error <= 1e-12
                scalar_formula_max_error = max(scalar_formula_max_error, error)
                for role in ("current", "history"):
                    # B is evaluated in FP32: keep the original scalar-add order.
                    expected_b = np.float32(np.float32(r[role + "_bce"]) + np.float32(r[role + "_rank"]))
                    expected_b = np.float32(expected_b + np.float32(.5) * np.float32(r[role + "_hard"]))
                    error_b = abs(float(expected_b) - r[role + "_total"])
                    assert error_b == 0.
                    supervised_formula_max_error = max(supervised_formula_max_error, error_b)
            if stage == 1:
                start = manifest["restored_starts"][order]
                assert start["adam_step"] == 288 and start["first_scores_replayed_exactly"]
                assert start["model_state_sha256"] == point["model_state_sha256"]
            else:
                norms = np.array([r["gradient_norm"] for r in rows])
                ds = np.array([r["function"] for r in rows])
                all_norms.extend(norms.tolist())
                all_d.extend(ds.tolist())
                report["segments"][name] = {
                    "updates": len(rows), "D_first": ds[0], "D_mean": float(ds.mean()),
                    "D_min": float(ds.min()), "D_max": float(ds.max()),
                    "weighted_D_mean": float(.5 * ds.mean()),
                    "current_B_mean": float(np.mean([r["current_total"] for r in rows])),
                    "history_B_mean": float(np.mean([r["history_total"] for r in rows])),
                    "weighted_history_B_mean": float(.1 * np.mean([r["history_total"] for r in rows])),
                    "preclip_norm_min": float(norms.min()), "preclip_norm_max": float(norms.max()),
                    "preclip_norm_mean": float(norms.mean()), "norm_over_1": int((norms > 1).sum()),
                    "history_draws_by_domain": dict(Counter(fit_domain[uid] for uid in tr["history_ids"])),
                }
            previous_keep, previous_seen = keep.copy(), seen
            if stage < 3:
                for uid in fit:
                    seen += 1
                    if len(keep) < 6:
                        keep.append(uid)
                    else:
                        index = retention_rng.randrange(seen)
                        if index < 6:
                            keep[index] = uid
            assert point["memory_summary"] == dict(members=keep, count=seen, seen=seen, stage=min(stage, 2))
            assert point["memory_bytes"] == point["memory"]["bytes"] <= 1048576
            if stage < 3:
                ret = point["retention"]
                assert ret["old_count"] == previous_seen and ret["new_count"] == seen == ret["reservoir_seen"]
                assert ret["old_members"] == previous_keep and ret["new_members"] == keep
                assert ret["h_min_eigenvalue"] >= -ret["h_psd_tolerance"]
                assert ret["serialized_bytes"] == point["memory_bytes"]
                if stage == 2:
                    sv = np.array(ret["transport_singular_values"])
                    assert sv.shape == (129,) and np.isfinite(sv).all() and (sv > 0).all()
            else:
                assert point["retention"] is None
            report["points"][name] = {
                "N": seen, "reservoir_seen": seen, "memory_stage": min(stage, 2),
                "memory_bytes": point["memory_bytes"],
                "cache_by_domain": {d: sum(fit_domain[uid] == d for uid in keep) for d in "ABC"},
                "transport_singular_min": float(sv.min()) if stage == 2 else None,
                "transport_singular_max": float(sv.max()) if stage == 2 else None,
                "full_restore_verified_record": point["full_restore_verified"],
                "reloaded_model_not_independently_executed": True,
            }
            total_updates += len(rows)
    assert total_updates == manifest["physical_updates"] == 1728
    assert manifest["gradient_group_presentations"] == 3456 and manifest["logical_updates"] == 2592
    assert len(all_norms) == 1728 and min(all_norms) > 1
    report["totals"] = {
        "new_updates": total_updates, "empty_shared_update_logs": 3,
        "norms_over_1": sum(n > 1 for n in all_norms),
        "preclip_norm_range": [min(all_norms), max(all_norms)],
        "D_range": [min(all_d), max(all_d)],
        "scalar_total_max_error": scalar_formula_max_error,
        "FP32_supervised_total_max_error": supervised_formula_max_error,
        "current_schedule_history_draws_and_algorithm_R_exact": True,
    }

    # Independent per-domain point means: use actual group/domain IDs, never
    # contiguous row assumptions or repeated orders as extra independent data.
    candidate_col = read(base / "job/evaluation/collected.json")
    ref_root = root / policy["reference"]["local_root"]
    ref_col = read(ref_root / "collected.json")
    shared_col = read(ref_root / "reference/collected.json")
    for col in (ref_col, shared_col):
        for k in ("group_ids", "domains", "metric_columns"):
            assert col[k] == candidate_col[k]
    columns = candidate_col["metric_columns"]
    metrics = ("map", "average_precision")
    domain_rows = {d: np.array([i for i, x in enumerate(candidate_col["domains"]) if x == d]) for d in "ABC"}
    point_means = {}
    stage_means = {"candidate": {}, "reference": {}}
    first_equal = {}
    for order in ("ABC", "BCA", "CAB"):
        for stage in (1, 2, 3):
            name = f"{order}_function_memory_stage{stage}"
            rec = candidate_col["points"][name]["raw"]["matrix"]
            cand = array(base / "job/evaluation" / rec["path"], rec)
            if stage == 1:
                rec = shared_col["points"][order + "_shared"]["raw"]["matrix"]
                ref = array(ref_root / "reference" / rec["path"], rec)
                assert np.array_equal(cand, ref)
                first_equal[order] = True
            else:
                rec = ref_col["points"][f"{order}_logit_tenth_stage{stage}"]["raw"]["matrix"]
                ref = array(ref_root / rec["path"], rec)
            stage_means["candidate"][order, stage] = cand
            stage_means["reference"][order, stage] = ref
            for d in "ABC":
                point_means[f"{order}_stage{stage}_{d}"] = {
                    m: {"candidate": float(cand[domain_rows[d], columns.index(m)].mean()),
                        "reference": float(ref[domain_rows[d], columns.index(m)].mean()),
                        "delta": float((cand-ref)[domain_rows[d], columns.index(m)].mean())}
                    for m in metrics
                }
    report["saved_raw_all22_first_stage_matrices_equal"] = first_equal
    report["stage_domain_MAP_AP"] = point_means
    decomposition = {}
    for order in ("ABC", "BCA", "CAB"):
        di = domain_rows[order[0]]
        ni = domain_rows[order[1]]
        zi = domain_rows[order[2]]
        j = columns.index("map")
        vals = {}
        for arm in ("candidate", "reference"):
            p1 = stage_means[arm][order, 1][:, j]
            p2 = stage_means[arm][order, 2][:, j]
            p3 = stage_means[arm][order, 3][:, j]
            vals[arm] = {
                "A2": float(p2[ni].mean()),
                "O": float((p3[di].mean() + p3[ni].mean())/2),
                "Z": float(p3[zi].mean()), "final_all": float(p3.mean()),
                "F_first": float((p1[di] - p3[di]).mean()),
                "F": float(((p1[di] - p3[di]).mean() + (p2[ni] - p3[ni]).mean())/2),
                "G": float(((p2[ni] - p1[ni]).mean() + (p3[zi] - p2[zi]).mean())/2),
            }
        vals["delta"] = {k: vals["candidate"][k] - vals["reference"][k] for k in vals["candidate"]}
        decomposition[order] = vals
    report["MAP_decomposition_by_order"] = decomposition
    report["MAP_decomposition_mean"] = {
        arm: {k: float(np.mean([decomposition[o][arm][k] for o in decomposition])) for k in decomposition["ABC"][arm]}
        for arm in ("candidate", "reference", "delta")
    }

    native_rel = "reports/documentation/20261006/function_memory/native_20261006_231500/result/result.json"
    native_record = read(root / native_rel)
    native = native_record["native"]
    assert native_record["status"] == "PASS_HANDWRITTEN_ONLY" and native_record["mode"] == "gpu"
    assert native["kind"] == "native_handwritten_first_optimizer_step" and not native["official_data"]
    assert native["reference_max_difference"] == 0
    assert native["input_shapes"] == [dict(texts=448, minimum=256, maximum=256)] * 2
    assert all(np.isfinite(native["history_only_gradient_norms"])) and min(native["history_only_gradient_norms"]) > 0
    assert native["dropout_before"] == native["dropout_after"]
    assert len(native["dropout_in_history"]) == 97 and not any(native["dropout_in_history"].values())
    report["read_previous_native_record_not_rerun"] = {
        "input_shapes": native["input_shapes"], "D_perturbed": native["optimizer"]["function"],
        "history_only_gradient_norms": native["history_only_gradient_norms"],
        "dropout_observed_values": len(native["dropout_in_history"]),
        "dropout_restored_exact": True, "reference_coordinate_max_difference": 0,
        "adam_parameters_with_state": native["adam_parameters_with_state"], "dtypes": native["dtypes"],
    }

    # Read nested primary historical evidence, but do not execute historical tests.
    hist_path = root / "history/function_memory_implementation_external_evidence.zip"
    hist_raw = payload(hist_path)
    history_reads = []
    with zipfile.ZipFile(io.BytesIO(hist_raw)) as archive:
        for name in ("FUNCTION_MEMORY_IMPLEMENTATION.external_review.zh.md",
                     "audit/scripts/independent_checks.py", "audit/scripts/check_lifecycle.py",
                     "audit/outputs/independent_checks.json", "audit/outputs/lifecycle_checks.json"):
            raw = archive.read(name)
            history_reads.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        historical_core = archive.read("input/scripts/step28_function_memory.py")
        current_core = payload(base / "source/scripts/step28_function_memory.py")
        assert current_core == historical_core
    report["historical_evidence"] = {
        "current_core_equal_previous_implementation_external_review_input": True,
        "archive_sha256": hashlib.sha256(hist_raw).hexdigest(),
        "selected_nested_members_read_without_execution": history_reads,
        "prior_reruns_executed_this_audit": 0,
    }
    report["mechanism_claims"] = {
        "initial_stage2_D_near_zero": True,
        "D_scalar_does_not_measure_gradient_size": "Formal per-term gradient norms/angles absent; no inference of relative gradient strength.",
        "all_steps_clip": "Verified preclip norm > 1 for all 1728; not in itself evidence of a defect.",
        "CAB_final_buffer": "6 A and 0 C raw groups, cumulative N=96; source retains old transported H/b and unchanged c.",
        "CAB_and_ABC_second_stage": "Both already lose new-domain MAP before second consolidation affects stage3 training.",
        "transport_spectrum": "Read singular values establish nonidentity only; unseen-domain approximation error is not measured.",
        "history_points": "Different historical intervention combinations; descriptive comparisons do not identify causal improvement.",
        "novelty": "Retaining cumulative function statistics with six-group transport is a method description, not a novelty proof.",
    }
    report["input_files"] = [reads[k] for k in sorted(reads)]
    report["elapsed_seconds"] = time.perf_counter() - begun
    report["status"] = "PASS_SAVED_EVIDENCE_CHECKS_NO_PROJECT_EXECUTION"
    target = out / "mechanism_checks.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "input_count": len(reads),
                      "totals": report["totals"], "MAP_decomposition_mean": report["MAP_decomposition_mean"],
                      "output": str(target), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
