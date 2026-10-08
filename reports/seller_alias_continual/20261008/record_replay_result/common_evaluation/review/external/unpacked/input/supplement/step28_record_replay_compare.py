"""Apply the frozen LOGIT 23 checks to opened C/S/LOGIT saved results on Linux.

Only reads registered metric matrices, metadata, and two frozen pure functions
from the original result-review ZIP. No training imports or raw data access.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import time
import zipfile

import numpy as np


ORDERS = ("ABC", "BCA", "CAB")
ARMS = ("C", "S", "LOGIT0.1")
ROLES = ("raw", "stage-cal", "first-cal", "primary")
PAIRS = (("C", "LOGIT0.1"), ("S", "LOGIT0.1"), ("C", "S"))
COLUMNS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
           "brier", "log_loss", "precision", "recall", "f1", "specificity",
           "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
           "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10")
RANK_COLUMNS = [i for i, name in enumerate(COLUMNS) if name not in
                ("brier", "log_loss", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")]
TERMS = {
    "O": ((3, 1, .5), (3, 2, .5)), "N": ((2, 2, .5), (3, 3, .5)), "Z": ((3, 3, 1.),),
    "F_first": ((1, 1, 1.), (3, 1, -1.)),
    "F": ((1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)),
    "G": ((2, 2, .5), (1, 2, -.5), (3, 3, .5), (2, 3, -.5)),
    "final_all": ((3, 1, 1 / 3), (3, 2, 1 / 3), (3, 3, 1 / 3)), "A2": ((2, 2, 1.),),
}
INPUT_SHA = "dbed856f06482e29b444ce45b5b44973083bbd28964516ae2cb5ccf5c8be3c38"
CHECK_SOURCE_SHA = "55e9bf419801ec57686898cf78d8237962fc5e5549849a2e0eac3e66318353f1"


def main(input_zip: Path, output: Path) -> None:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    assert not output.exists(), "Require a new output directory"
    assert hashlib.sha256(input_zip.read_bytes()).hexdigest() == INPUT_SHA
    archive = zipfile.ZipFile(input_zip)
    reads = {}

    def read(name: str, record: dict | None = None) -> bytes:
        blob = archive.read(name)
        identity = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        if record is not None:
            assert all(identity[k] == record[k] for k in identity), name
        reads[name] = identity
        return blob

    def read_json(name: str, record: dict | None = None) -> dict:
        return json.loads(read(name, record))

    completion = read_json("result/job/completion.json")
    saved = read_json("result/job/evaluation/evaluation.json", completion["evaluation"])
    current = read_json("result/job/evaluation/collected.json", saved["collected"])
    policy = read_json("result/source/schema/step28_record_replay_policy.json")
    refs = {path: read_json("reference/" + path, identity)
            for path, identity in policy["reference"]["collections"].items()}
    assert current["metric_columns"] == list(COLUMNS)
    assert len(current["group_ids"]) == len(set(current["group_ids"])) == 60
    for ref in refs.values():
        assert all(ref[k] == current[k] for k in ("group_ids", "domains", "metric_columns"))
    rows = {domain: np.flatnonzero(np.array(current["domains"]) == domain) for domain in "ABC"}
    assert all(len(indices) == 20 for indices in rows.values())

    source = read("result/source/scripts/step28_bge_continual_evaluate.py")
    assert hashlib.sha256(source).hexdigest() == CHECK_SOURCE_SHA
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in ("comparison_checks", "ranking_guards")]
    assert len(functions) == 2
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "frozen_23_checks", "exec"), namespace)

    cache, arrays = {}, {}
    for arm in ARMS:
        for order in ORDERS:
            for stage in (1, 2, 3):
                folder, registry = "result/job/evaluation", current
                name = order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"
                if arm == "LOGIT0.1":
                    folder = "reference/reference" if stage == 1 else "reference"
                    registry = refs["reference/collected.json" if stage == 1 else "collected.json"]
                    name = order + "_shared" if stage == 1 else f"{order}_logit_tenth_stage{stage}"
                if (folder, name) not in cache:
                    values = {}
                    for role in ROLES[:3]:
                        record = registry["points"][name][role]["matrix"]
                        matrix = np.load(io.BytesIO(read(folder + "/" + record["path"], record)), allow_pickle=False)
                        assert matrix.shape == (60, 22) and matrix.dtype == np.dtype("<f8")
                        assert np.isfinite(matrix).all()
                        values[role] = matrix
                    for role in ROLES[1:3]:
                        assert np.array_equal(values[role][:, RANK_COLUMNS], values["raw"][:, RANK_COLUMNS])
                    values["primary"] = values["stage-cal"].copy()
                    values["primary"][:, RANK_COLUMNS] = values["raw"][:, RANK_COLUMNS]
                    cache[folder, name] = values
                arrays[arm, order, stage] = cache[folder, name]
    assert len(cache) == 24  # 15 current physical stages plus 9 LOGIT stages; 72 matrices.

    draw = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 3, 20))
    weights = np.stack([(draw == i).sum(axis=2) for i in range(20)], axis=2) / 20
    maximum_error, checked_numbers = 0., 0

    def summarize(field: np.ndarray, expected: dict | None = None) -> dict:
        nonlocal maximum_error, checked_numbers
        per_order = field.mean(2).sum(1)
        mean = field.mean(0).mean(1).sum(0)
        samples = np.einsum("bdg,dgm->bm", weights, field.mean(0), optimize=False)
        interval = np.quantile(samples, [.025, .975], axis=0, method="linear")
        result = {name: {"mean": float(mean[i]), "conditional_95pct_interval": interval[:, i].tolist(),
                         "per_order": dict(zip(ORDERS, per_order[:, i].tolist()))}
                  for i, name in enumerate(COLUMNS)}
        if expected is not None:
            for name in COLUMNS:
                a, b = result[name], expected[name]
                actual = [a["mean"], *a["conditional_95pct_interval"], *a["per_order"].values()]
                target = [b["mean"], *b["conditional_95pct_interval"], *[b["per_order"][o] for o in ORDERS]]
                error = float(np.max(np.abs(np.array(actual) - target)))
                maximum_error = max(maximum_error, error)
                checked_numbers += 6
                assert error <= 1e-12
            return expected  # Preserve original published floating-point statistics exactly.
        return result

    fields, results = {}, {}
    for arm in ARMS:
        results[arm] = {}
        for role in ROLES:
            results[arm][role] = {}
            for endpoint, terms in TERMS.items():
                field = np.zeros((3, 3, 20, 22))
                for oi, order in enumerate(ORDERS):
                    for stage, arrival, coefficient in terms:
                        domain = order[arrival - 1]
                        field[oi, "ABC".index(domain)] += coefficient * arrays[arm, order, stage][role][rows[domain]]
                if endpoint in ("F_first", "F", "G"):
                    field[..., [4, 5]] *= -1
                fields[arm, role, endpoint] = field
                results[arm][role][endpoint] = summarize(field, saved["endpoints"][arm][role][endpoint])
    decisions, check_rows = {}, []
    for candidate, reference in PAIRS:
        label = candidate + "-" + reference
        results[label] = {}
        for role in ROLES:
            results[label][role] = {}
            for endpoint in TERMS:
                previous = saved["comparisons"].get(label, {}).get("delta", {}).get(role, {}).get(endpoint)
                results[label][role][endpoint] = summarize(
                    fields[candidate, role, endpoint] - fields[reference, role, endpoint], previous)
        delta = results[label]["primary"]
        versus_raw = {ep: {name: {"mean": float((fields[candidate, "primary", ep]
                      - fields[reference, "raw", ep]).mean(0).mean(1).sum(0)[COLUMNS.index(name)])}
                      for name in ("brier", "log_loss")} for ep in ("O", "N")}
        decision = namespace["comparison_checks"](delta, versus_raw)
        assert len(decision["checks"]) == 23
        decisions[label] = decision
        definitions = [
            ("old_map_improves", "O", "map", "mean", ">"),
            ("old_map_interval_above_zero", "O", "map", "ci_lower", ">"),
            ("old_recall5_improves", "O", "recall_at_5", "mean", ">"),
            ("new_map_non_decrease", "N", "map", "mean", ">="),
            ("new_recall5_non_decrease", "N", "recall_at_5", "mean", ">="),
        ]
        for ep in ("O", "N"):
            for name in ("average_precision", "roc_auc", "brier", "log_loss"):
                definitions.append((f"{ep}_{name}_non_degradation", ep, name, "mean",
                                    "<=" if name in ("brier", "log_loss") else ">="))
            for name in ("brier", "log_loss"):
                definitions.append((f"{ep}_{name}_against_raw_reference", ep, name, "versus_raw_mean", "<="))
        for name in ("map", "recall_at_5", "average_precision", "roc_auc", "brier", "log_loss"):
            definitions.append((f"Z_{name}_non_degradation", "Z", name, "mean",
                                "<=" if name in ("brier", "log_loss") else ">="))
        for key, ep, name, statistic, operator in definitions:
            value = (versus_raw[ep][name]["mean"] if statistic == "versus_raw_mean" else
                     delta[ep][name]["conditional_95pct_interval"][0] if statistic == "ci_lower" else
                     delta[ep][name]["mean"])
            passes = value > 0 if operator == ">" else value >= 0 if operator == ">=" else value <= 0
            assert passes == decision["checks"][key]  # Independently decoded check-to-statistic mapping.
            check_rows.append(dict(comparison=label, check=key, endpoint=ep, metric=name,
                                   statistic=statistic, value=value, operator=operator, threshold=0, passed=passes))
    assert checked_numbers == 21120
    payload = {
        "status": "COMPLETE_POSTHOC_COMMON_EVALUATION_PENDING_EXTERNAL_REVIEW",
        "started_at_utc": started_at, "elapsed_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "input_zip_sha256": INPUT_SHA, "legacy_check_source_sha256": CHECK_SOURCE_SHA,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap": {"engine": "PCG64", "seed": 20260930, "replicates": 5000,
                      "unit": "Paired whole groups within actual domains; shared across fixed orders and arms",
                      "interval": "Marginal conditional 95% linear percentile; excludes retraining and selection uncertainty"},
        "replayed_statistics": {"numbers": checked_numbers, "maximum_absolute_error": maximum_error},
        "original_five_checks": {k: {"checks": v["checks"], "pass": v["pass"]} for k, v in saved["comparisons"].items()},
        "original_development_criteria_pass": saved["development_criteria_pass"],
        "common_23_checks": decisions, "check_details": check_rows, "results": results, "read_identities": reads,
        "boundary": "Authorized post-result supplement; not replacement of the frozen five checks, independent confirmation, or overfitting test. No model, labels, raw text, Memory, GPU, fitting, or selection.",
    }
    lines = ["统一22指标：C、S、LOGIT0.1；23项为事后同口径补充，原5项不改。",
             "primary=raw排序/曲线14项＋stage-cal概率2项/分类6项。A2为保留的额外端点。",
             "差值均为左臂减右臂；括号为固定模型/顺序下群bootstrap条件95%区间。",
             "完整三顺序均值、各臂区间及23项数值见statistics.json。"]
    for role in ROLES:
        for ep in TERMS:
            lines += ["", f"{role} / {ep}", "metric | C | S | LOGIT0.1 | C-LOGIT0.1 [CI] | S-LOGIT0.1 [CI] | C-S [CI]"]
            for name in COLUMNS:
                values = [f"{results[a][role][ep][name]['mean']:.9f}" for a in ARMS]
                for a, b in PAIRS:
                    stat = results[a + "-" + b][role][ep][name]
                    lo, hi = stat["conditional_95pct_interval"]
                    values.append(f"{stat['mean']:+.9f} [{lo:+.9f}, {hi:+.9f}]")
                lines.append(" | ".join([name, *values]))
    lines += ["", "原23项逐条值（阈值均为0；判定使用完整精度，未用复算容差放宽门槛）", "comparison | check | value | operator | passed"]
    lines += [f"{r['comparison']} | {r['check']} | {r['value']:+.12g} | {r['operator']} 0 | {r['passed']}" for r in check_rows]
    output.mkdir(parents=True)
    (output / "statistics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "tables.zh.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "elapsed_seconds", "environment", "replayed_statistics", "common_23_checks")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.input_zip, arguments.output)
