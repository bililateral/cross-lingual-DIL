"""Finalize and independently audit saved sensitivity matrices; no label parsing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import step28_sensitivity_evaluate as adapter

data, frozen = adapter.data, adapter.frozen


def reconcile(actual: np.ndarray, expected: np.ndarray, columns: list[str]) -> dict:
    delta = np.abs(actual - expected)
    allowed = np.array([name in ("brier", "log_loss") for name in columns])
    assert np.array_equal(actual[:, ~allowed], expected[:, ~allowed])
    ulps = delta[:, allowed] / np.spacing(np.abs(expected[:, allowed]))
    assert np.isfinite(ulps).all() and np.all(ulps <= 4), float(ulps.max())
    return {"different_values": int(np.count_nonzero(delta)),
            "maximum_absolute_difference": float(delta.max()),
            "maximum_ulps_brier_log_loss": float(ulps.max()),
            "other_20_columns_exact": True}


def independent_contrasts(candidate: dict, reference: dict, column: int) -> dict:
    names = ("H", "N", "G_candidate", "G_reference", "F_candidate", "F_reference",
             "second_candidate_change", "second_reference_change", "final_candidate_minus_reference")
    differences = {name: {domain: [] for domain in "ABC"} for name in names}
    orders = {}

    def group(source: dict, key: str, domain: str) -> np.ndarray:
        index = "ABC".index(domain)
        return source[key][index * 20:index * 20 + 20, column]

    for order in ("ABC", "BCA", "CAB"):
        first, second, third = order
        shared = group(reference, order + "_shared", first)
        final_candidate = group(candidate, order + "_distill_stage3", first)
        final_reference = group(reference, order + "_stage3", first)
        local = {"H": final_candidate - final_reference,
                 "F_candidate": shared - final_candidate, "F_reference": shared - final_reference}
        for name, value in local.items():
            differences[name][first].append(value)
        n, gain, reference_gain = [], [], []
        for stage, domain in ((2, second), (3, third)):
            ca = group(candidate, f"{order}_distill_stage{stage}", domain)
            cb = group(candidate, order + ("_shared" if stage == 2 else "_distill_stage2"), domain)
            ra = group(reference, f"{order}_stage{stage}", domain)
            rb = group(reference, order + ("_shared" if stage == 2 else "_stage2"), domain)
            for name, value in (("N", ca - ra), ("G_candidate", ca - cb), ("G_reference", ra - rb)):
                differences[name][domain].append(value)
            n.append(float((ca - ra).mean()))
            gain.append(float((ca - cb).mean()))
            reference_gain.append(float((ra - rb).mean()))
        for name, source, ending, previous in (
                ("second_candidate_change", candidate, "_distill_stage3", "_distill_stage2"),
                ("second_reference_change", reference, "_stage3", "_stage2")):
            value = group(source, order + ending, second) - group(source, order + previous, second)
            differences[name][second].append(value)
            local[name] = value
        for domain in "ABC":
            differences["final_candidate_minus_reference"][domain].append(
                group(candidate, order + "_distill_stage3", domain) - group(reference, order + "_stage3", domain))
        orders[order] = {**{name: float(v.mean()) for name, v in local.items()},
                         "N": float(np.mean(n)), "G_candidate": float(np.mean(gain)),
                         "G_reference": float(np.mean(reference_gain)), "N_transitions": n,
                         "G_candidate_transitions": gain, "G_reference_transitions": reference_gain}
    return {"differences": {name: np.stack([np.mean(v[domain], axis=0) for domain in "ABC"])
                             for name, v in differences.items()}, "by_order": orders}


def finalize(run: Path, collection: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("Use a new result directory")
    config, _, _, scores, references, _, checks = adapter.prepare(run)
    result = data.read_json(collection / "metrics.json")
    assert result["status"] == "METRICS_COLLECTED_SHARED_RECONCILIATION_REQUIRED"
    assert result["config"] == config and result["score_manifest_sha256"] == data.sha256(run / "manifest.json")
    columns = result["columns"]
    draws = np.random.default_rng(20260910).integers(0, 20, size=(5000, 3, 20))
    reconciliation, blind_counts, compared, max_error = {}, 0, 0, 0.

    def check(actual, expected) -> None:
        nonlocal compared, max_error
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.abs(a - b).max())
        assert error <= 1e-12, error
        compared += a.size
        max_error = max(max_error, error)

    summary = {}
    for arm, value in result["arms"].items():
        matrices = {}
        for point, record in value["points"].items():
            matrix = np.load(data.verify(collection / record["file"]["path"], record["file"]), allow_pickle=False)
            assert matrix.shape == (60, 22) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
            matrices[point] = matrix
            for domain_index, domain in enumerate("ABC"):
                check(matrix[domain_index*20:domain_index*20+20].mean(0), [record["by_domain"][domain][c] for c in columns])
            if point.endswith("_shared"):
                reconciliation[arm + "/" + point] = reconcile(matrix, references["sequential"][point], columns)
            counts = np.array([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in record["confusion"]])
            assert np.all(counts[:, [0, 2]].sum(1) == 20) and np.all(counts[:, [1, 3]].sum(1) == 358)
            assert np.array_equal((scores[arm][point].astype(float) >= 0).sum(1), counts[:, :2].sum(1))
            blind_counts += 60
        value["comparisons"] = {role: frozen.compare(matrices, reference, draws, config, role)
                                for role, reference in references.items()}
        for role, reference in references.items():
            saved = value["comparisons"][role]
            for column, metric in enumerate(columns):
                independent = independent_contrasts(matrices, reference, column)
                for name, differences in independent["differences"].items():
                    observed = saved["metrics"][metric]["contrasts"][name]
                    samples = np.mean([differences[i][draws[:, i]].mean(1) for i in range(3)], axis=0)
                    check(observed["mean"], differences.mean())
                    check(observed["conditional_95pct_interval"], np.percentile(samples, [2.5, 97.5]))
                for order, numbers in independent["by_order"].items():
                    for name, expected in numbers.items():
                        check(saved["metrics"][metric]["by_order"][order][name], expected)
            if role == "sequential":
                ap = saved["metrics"]["average_precision"]
                h, n = ap["contrasts"]["H"], ap["contrasts"]["N"]
                same = [o for o, v in ap["by_order"].items() if v["H"] > 0 and v["G_candidate"] >= .02]
                gates = {"retention": h["mean"] >= .01 and h["conditional_95pct_interval"][0] > 0,
                         "new_capability": n["conditional_95pct_interval"][0] >= -.01,
                         "same_order_learning_and_retention": len(same) >= 2}
                assert saved["gate"] == {"checks": gates, "same_orders": same, "passed": all(gates.values())}
        first = np.concatenate([matrices[o + "_distill_stage3"][20*i:20*i+20] for i, o in enumerate(("ABC", "BCA", "CAB"))])
        first_counts = np.array([[r[k] for k in ("tp", "fp", "fn", "tn")]
                                for i, o in enumerate(("ABC", "BCA", "CAB"))
                                for r in value["points"][o + "_distill_stage3"]["confusion"][20*i:20*i+20]]).sum(0)
        summary[arm] = {"first_domain_final_metrics": dict(zip(columns, first.mean(0).tolist())),
                        "first_domain_final_counts": first_counts.tolist(),
                        "gate": value["comparisons"]["sequential"]["gate"],
                        "ap_comparisons": {role: v["metrics"]["average_precision"]["contrasts"] for role, v in value["comparisons"].items()}}
    result.update(status="BOTH_SENSITIVITIES_VALID_EVALUATED_INTERPRETATION_REQUIRED",
                  interpretation="Finite two-point sensitivity on reused valid; conditional unadjusted intervals. No optimum, unique cause, automatic winner or further search.",
                  metric_file_base=collection.relative_to(data.ROOT).as_posix(),
                  parse_history={"failed_actual_development_parses": 1, "explicitly_authorized_recovery_actual_parses": 1,
                                 "finalization_actual_parses": 0, "sensitivity_total_actual_parses": 2})
    audit = {"status": "PASS_SAVED_METRIC_RECONCILIATION_AND_INDEPENDENT_CONTRASTS",
             "collection": data.record(collection / "metrics.json", data.ROOT),
             "reconciliation": reconciliation, "numeric_values_compared": compared,
             "maximum_absolute_difference": max_error, "blind_prediction_count_checks": blind_counts,
             "label_parses": 0, "model_loading": 0,
             "limitations": "No label-based recomputation of saved group metrics; intervals conditional on fixed orders, seeds and reused development groups. Four-ULP bound applies only to Brier/log-loss; all other shared metrics exact.",
             "checks": checks, "summary": summary}
    destination.mkdir(parents=True)
    data.write_json(destination / "evaluation.json", result)
    data.write_json(destination / "audit.json", audit)
    lines = ["# 敏感性：最终首域60群等权完整指标", "", "分类阈值为logit≥0；AP与梯形PR-AUC分别报告。", "", "|指标|减弱重放 α=.5 β=.5|减弱蒸馏 α=1 β=.25|", "|---|---:|---:|"]
    for column in columns:
        lines.append("|" + column + "|" + "|".join(f"{summary[arm]['first_domain_final_metrics'][column]:.9f}" for arm in frozen.ARMS) + "|")
    (destination / "metrics.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.run.resolve(), args.collection.resolve(), args.out.resolve())
    print(data.json_bytes({k: result[k] for k in ("status", "numeric_values_compared", "maximum_absolute_difference", "blind_prediction_count_checks")}).decode())


if __name__ == "__main__":
    main()
