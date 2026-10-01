"""One authorized fixeds0 A/C test: blind inference, one heldout parse, full capture."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time
from typing import Any, Callable

import numpy as np

import step28_alias_test as method

data, base, metrics = method.data, method.base, method.metrics
core = base.core
POLICY = data.ROOT / "schema/step28_alias_test_policy.json"
COLLECTED = "ALL_FOUR_TEST_MATRICES_SAVED_BEFORE_STATISTICS"
COMPLETE = "FIXED_PAIR_TEST_EVALUATED_NO_AUTOMATIC_RETRY"


def contract() -> dict:
    p = data.read_json(POLICY)
    expected = []
    for key, comparison, metric, statistic, operator, bound in method.CRITERIA:
        expected.append({"id": key, "comparison": comparison.split("_minus_"),
                         "metric": metric, "statistic": "conditional_95pct_interval_lower"
                         if statistic == "lower" else "point_difference",
                         "operator": operator, "bound": bound})
    if (p["criteria"] != expected or p["columns"] != list(metrics.COLUMNS)
            or p["roles"] != list(method.ROLES) or p["fixed_seed"] != "s0"
            or p["statistics"]["bootstrap_seed"] != 20260928
            or p["statistics"]["bootstrap_replicates"] != 5000
            or p["data"]["groups"] != 120 or p["data"]["groups_per_domain"] != 40
            or p["data"]["split"] != "heldout" or not p["all_nine_required"]
            or p["acceptance_tolerance"] != 0 or p["acceptance_rounding"]
            or p["access"]["label_parses"] != {"train": 0, "development": 0, "heldout": 1, "owners": 0}
            or p["access"]["automatic_retry"] or p["inference"]["optimizer_updates"] != 0
            or p["inference"]["calibration_fits"] != 0):
        raise ValueError("Confirmed independent test contract differs")
    return p


def sources() -> list[dict]:
    p = contract()
    paths = {row["path"] for row in p["sources"]["inherited_scientific_sources"]}
    paths.update(("scripts/step28_alias_test.py", "scripts/step28_alias_test_run.py",
                  "scripts/step28_alias_test_audit.py", "tests/test_step28_alias_test_contracts.py",
                  "scripts/run_step28_alias_test_linux_20260928.sh",
                  "schema/step28_alias_test_policy.json", "docs/SELLER_ALIAS_TEST.zh.md",
                  "reports/documentation/20260928/test_plan/authorization.json"))
    for row in p["sources"]["inherited_scientific_sources"]:
        data.verify(data.ROOT / row["path"], row)
    return [data.record(data.ROOT / path, data.ROOT) for path in sorted(paths)]


def checked_json(record: dict, root: Path = data.ROOT) -> dict:
    return data.read_json(data.verify(root / record["path"], record))


def allowed_input(p: dict, relative: str, root: Path = data.ROOT) -> Path:
    if relative not in ("groups.csv", "heldout/items.jsonl", "heldout/supervision/pairs.csv"):
        raise ValueError("Only the three authorized test inputs are accessible")
    rec = p["data"]["inputs"][relative]
    expected = Path(p["data"]["root"]) / relative
    if rec["path"] != expected.as_posix():
        raise ValueError("Test input identity/path mismatch")
    return data.verify(root / rec["path"], rec)


def public_inputs(p: dict, root: Path = data.ROOT) -> tuple[list, list[dict]]:
    manifest = checked_json(p["data"]["manifest"], root)
    validation = checked_json(p["data"]["validation"], root)
    if (manifest["study"] != p["data"]["study"] or manifest["root_seed"] != p["data"]["root_seed"]
            or validation["status"] != "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION"):
        raise ValueError("Frozen generation identity/qualification differs")
    for name, rec in p["data"]["inputs"].items():
        if {key: rec[key] for key in ("bytes", "sha256")} != manifest["files"][name]:
            raise ValueError("Input identity differs from generation manifest")
    with allowed_input(p, "groups.csv", root).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["domain", "split", "group_uid", "group_index", "accounts", "items"]:
            raise ValueError("Group metadata schema differs")
        metadata = list(reader)
    if len(metadata) != 360 or len({r["group_uid"] for r in metadata}) != 360:
        raise ValueError("Public split inventory differs")
    selected = sorted((r for r in metadata if r["split"] == "heldout"),
                      key=lambda r: (r["domain"], int(r["group_index"])))
    method.domain_rows([r["domain"] for r in selected])
    ids = {r["group_uid"] for r in selected}
    collected, items_seen = defaultdict(lambda: defaultdict(list)), set()
    with allowed_input(p, "heldout/items.jsonl", root).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (set(row) != {"group_uid", "seller_uid", "item_uid", "title", "description"}
                    or not all(isinstance(v, str) and v for v in row.values())
                    or row["group_uid"] not in ids or row["item_uid"] in items_seen):
                raise ValueError("Heldout text schema/identity differs")
            items_seen.add(row["item_uid"])
            collected[row["group_uid"]][row["seller_uid"]].append(
                (row["item_uid"], row["title"], row["description"]))
    if set(collected) != ids:
        raise ValueError("Missing test group")
    groups, sellers_seen = [], set()
    for rec in selected:
        seller_rows = collected[rec["group_uid"]]
        sellers = tuple(sorted(seller_rows))
        group = data.Group(rec["group_uid"], sellers,
                           tuple(tuple(sorted(seller_rows[s])) for s in sellers))
        group.validate()
        if (sellers_seen.intersection(sellers) or int(rec["accounts"]) != 28
                or int(rec["items"]) != sum(map(len, group.items))):
            raise ValueError("Test accounts reused or metadata size mismatch")
        sellers_seen.update(sellers)
        groups.append(group)
    return groups, selected


def historical_models(p: dict) -> tuple[dict, dict]:
    for name in ("calibration_completion", "calibration_evaluation", "calibration_result_disposition"):
        result = checked_json(p["sources"][name])
        if name == "calibration_result_disposition":
            if result["status"] != "PASS_CALIBRATION_RESULT_REVIEW":
                raise ValueError("Calibration result review not closed")
        elif not result["acceptance"]["passed"]:
            raise ValueError("Calibration development acceptance not passed")
    manifest = checked_json(p["sources"]["ranking_manifest"])
    run_root = (data.ROOT / p["sources"]["ranking_manifest"]["path"]).parent
    partition = checked_json(manifest["partition"], run_root)
    info = {}
    for name, rec in p["models"].items():
        model_manifest = checked_json(rec["source_manifest"])
        point = model_manifest["points"]["6"]
        fitted = checked_json(rec["calibration"])
        threshold = checked_json(rec["old_threshold_source"])
        if (name not in ("A", "C") or rec["run_id"] != ("s0_d" if name == "A" else "s0_hard")
                or model_manifest["run_id"] != rec["run_id"] or point["run_id"] != rec["run_id"]
                or point["epoch"] != rec["epoch"] or rec["epoch"] != 6
                or point["model"]["sha256"] != rec["payload"]["sha256"]
                or point["model"]["bytes"] != rec["payload"]["bytes"]
                or point["model"]["state_sha256"] != rec["payload"]["payload_state_sha256"]
                or point["model_state_sha256"] != rec["payload"]["model_parameters_sha256"]
                or fitted["origin"]["model"] != point["model"]
                or fitted["origin"]["model_state_sha256"] != point["model_state_sha256"]
                or fitted["a"] != rec["a"] or fitted["b"] != rec["b"]
                or threshold["model_state_sha256"] != point["model_state_sha256"]):
            raise ValueError("Fixed A/C model/map/threshold identity differs")
        method.calibration.parameters(fitted)
        c = point["metadata"]["reference_config"]
        if (c != method.calibration.ranking.reference_config(method.calibration.ranking.contract(), "s0")
                or c["data_manifest_sha256"] != p["data"]["manifest"]["sha256"]
                or c["models"]["split_rank"] != p["inference"]["pretrained_archive"]):
            raise ValueError("Inference configuration differs from the actual checkpoint")
        info[name] = {"point": point, "config": c, "map": fitted,
                      "threshold": float(threshold["threshold"])}
    if set(info) != {"A", "C"}:
        raise ValueError("Both prespecified models required")
    return info, partition


def array(path: Path, record: dict, dtype: Any) -> np.ndarray:
    values = np.load(data.verify(path, record), allow_pickle=False)
    if values.shape != (120, 378) or values.dtype != dtype or not np.isfinite(values).all():
        raise ValueError("Incomplete/nonfinite score array")
    return values


def save_scores(out: Path, name: str, raw: np.ndarray, mapping: dict, origin: dict) -> dict:
    if name not in ("A", "C") or raw.shape != (120, 378) or raw.dtype != np.float32:
        raise ValueError("Complete fixed-role float32 inference required")
    transformed = method.calibration.transform(raw, mapping)
    order = method.calibration.preserve_order(raw, transformed)
    files = {}
    for role, values in ((name + "_raw", raw), (name + "_cal", transformed)):
        path = out / (role + "_scores.npy")
        if path.exists():
            raise FileExistsError(path)
        np.save(path, values, allow_pickle=False)
        files[role] = data.record(path, out)
        if not np.array_equal(array(path, files[role], values.dtype), values):
            raise ValueError("Score file reload differs")
    return {"files": files, "origin": origin, "order": order}


def restore_scores(out: Path, blind: dict, p: dict, info: dict) -> dict:
    if set(blind["models"]) != {"A", "C"} or blind["policy_sha256"] != data.sha256(POLICY):
        raise ValueError("Both fixed model scores required before supervision")
    values = {}
    for name in ("A", "C"):
        record = blind["models"][name]
        if record["origin"] != {"model": p["models"][name]["payload"],
                                "map": p["models"][name]["calibration"]}:
            raise ValueError("Saved score model/map binding differs")
        if set(record["files"]) != {name + "_raw", name + "_cal"}:
            raise ValueError("Missing original or calibrated score role")
        for suffix, dtype in (("raw", np.float32), ("cal", np.float64)):
            role = name + "_" + suffix
            rec = record["files"][role]
            if rec["path"] != role + "_scores.npy":
                raise ValueError("Score role/path mismatch")
            values[role] = array(out / rec["path"], rec, dtype)
        expected = method.calibration.transform(values[name + "_raw"], info[name]["map"])
        if not np.array_equal(expected, values[name + "_cal"]):
            raise ValueError("Map replay differs")
        method.calibration.preserve_order(values[name + "_raw"], values[name + "_cal"])
    return values


def parse_once(out: Path, p: dict, groups: list, blind: dict, info: dict,
               root: Path = data.ROOT) -> tuple[np.ndarray, dict]:
    if data.read_json(out / "blind.json") != blind:
        raise ValueError("Saved complete blind collection differs from active collection")
    restore_scores(out, blind, p, info)
    if blind["group_ids"] != [g.uid for g in groups]:
        raise ValueError("Blind scores and query groups are not aligned")
    access = out / "heldout_access.json"
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "parse_attempts": 1,
              "train": 0, "development": 0, "heldout": 1, "owners": 0,
              "blind": data.record(out / "blind.json", out)}
    with access.open("xb") as stream:
        stream.write(data.json_bytes(record))
    rows = defaultdict(list)
    with allowed_input(p, "heldout/supervision/pairs.csv", root).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["group_uid", "seller_uid_left", "seller_uid_right", "label"]:
            raise ValueError("Test supervision schema differs")
        for row in reader:
            rows[row["group_uid"]].append(row)
    if set(rows) != {g.uid for g in groups}:
        raise ValueError("Test supervision missing/extra groups")
    truth = np.asarray([data.align_labels(g, rows[g.uid]) for g in groups], dtype=np.uint8)
    if truth.shape != (120, 378) or not np.all(truth.sum(1) == 20):
        raise ValueError("Test supervision size/positive totals differ")
    left, right = np.triu_indices(28, 1)
    degree = np.zeros((120, 28), dtype=np.int64)
    for edge, (i, j) in enumerate(zip(left, right, strict=True)):
        degree[:, i] += truth[:, edge]
        degree[:, j] += truth[:, edge]
    if not np.isin(degree, [1, 2]).all():
        raise ValueError("Test query positive counts differ")
    alignment = {"group_ids": [g.uid for g in groups], "pairs": int(truth.size),
                 "positive_pairs": int(truth.sum()), "queries": int(degree.size),
                 "sellers": {g.uid: list(g.sellers) for g in groups},
                 "labels_saved": False}
    return truth, alignment


def collect(out: Path, truth: np.ndarray, scores: dict, blind: dict, info: dict,
            source_files: list[dict]) -> dict:
    if set(scores) != set(method.ROLES) or truth.shape != (120, 378):
        raise ValueError("Four complete score roles and shared test truth required")
    destination = out / "evaluation"
    destination.mkdir()
    result = {"status": COLLECTED, "source_files": source_files,
              "policy_sha256": data.sha256(POLICY), "columns": list(metrics.COLUMNS),
              "group_ids": blind["group_ids"], "domains": blind["domains"],
              "blind": data.record(out / "blind.json", out), "points": {}, "automatic": {}}
    for role in method.ROLES:
        matrix, counts = metrics.group_metrics(truth, scores[role])
        path = destination / (role + "_metrics.npy")
        np.save(path, matrix, allow_pickle=False)
        result["points"][role] = {"file": data.record(path, destination), "counts": counts}
    for name in ("A", "C"):
        threshold = info[name]["threshold"]
        mapping = info[name]["map"]
        result["automatic"][name] = {
            "original_threshold": threshold, "mapped_threshold": mapping["a"] * threshold + mapping["b"],
            "decision": "compare_original_logits_with_original_threshold_no_refit",
            "counts": base.binary_counts(truth, scores[name + "_raw"], threshold).tolist()}
    data.write_json(destination / "collected.json", result)
    return result


def finalize(out: Path, source_files: list[dict]) -> dict:
    destination = out / "evaluation"
    if (destination / "evaluation.json").exists():
        raise FileExistsError("Published test evaluation cannot be overwritten")
    saved = data.read_json(destination / "collected.json")
    if (saved["status"] != COLLECTED or saved["source_files"] != source_files
            or saved["policy_sha256"] != data.sha256(POLICY)
            or saved["columns"] != list(metrics.COLUMNS) or set(saved["points"]) != set(method.ROLES)):
        raise ValueError("Complete collection and matching scientific sources required")
    if len(saved["group_ids"]) != 120 or len(set(saved["group_ids"])) != 120:
        raise ValueError("Test group collection differs")
    method.domain_rows(saved["domains"])
    matrices = {}
    for role, rec in saved["points"].items():
        if rec["file"]["path"] != role + "_metrics.npy":
            raise ValueError("Metric role/path mismatch")
        matrix = np.load(data.verify(destination / rec["file"]["path"], rec["file"]), allow_pickle=False)
        if matrix.shape != (120, 22) or matrix.dtype != np.float64 or not np.isfinite(matrix).all():
            raise ValueError("Incomplete metric matrix")
        matrices[role] = matrix
        rec["mean"] = dict(zip(metrics.COLUMNS, matrix.mean(0).tolist(), strict=True))
        rec["by_domain"] = {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(saved["domains"]) == d].mean(0).tolist(), strict=True)) for d in "ABC"}
        rec["fixed_classification"] = base.fixed_classification(rec["counts"], saved["domains"])
    ranking_columns = [metrics.COLUMNS.index(k) for k in (*metrics.CURVE_KEYS, *metrics.RETRIEVAL_KEYS)]
    for name in ("A", "C"):
        if not np.array_equal(matrices[name + "_raw"][:, ranking_columns], matrices[name + "_cal"][:, ranking_columns]):
            raise ValueError("Calibrated test ranking/curve metric changed")
    draws = method.bootstrap_draws()
    draw_path = destination / "bootstrap_draws.npy"
    if draw_path.exists():
        if not np.array_equal(np.load(draw_path, allow_pickle=False), draws):
            raise ValueError("Saved bootstrap draws differ")
    else:
        np.save(draw_path, draws, allow_pickle=False)
    result = {**saved, **method.summarize(matrices, saved["domains"], draws), "status": COMPLETE,
              "bootstrap": data.record(draw_path, destination),
              "statistics": contract()["statistics"], "training_updates": 0, "calibration_fits": 0}
    for rec in result["automatic"].values():
        rec["report"] = base.automatic_report(np.asarray(rec["counts"]), saved["domains"], draws)
    data.write_json(destination / "evaluation.json", result)
    return result


class Budget:
    def __init__(self, out: Path, seconds: int, output_bytes: int):
        self.out, self.seconds, self.output_bytes = out, seconds, output_bytes
        self.started, self.maximum_observed_bytes = time.monotonic(), 0

    def check(self) -> None:
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        self.maximum_observed_bytes = max(self.maximum_observed_bytes, size)
        if time.monotonic() - self.started > self.seconds or size > self.output_bytes:
            raise RuntimeError("Authorized test budget exhausted")


def inference_one(p: dict, name: str, info: dict, groups: list, check: Callable) -> tuple[np.ndarray, dict]:
    import torch
    rec, saved = p["models"][name], info[name]
    path = data.verify(data.ROOT / rec["payload"]["path"], rec["payload"])
    model = base.load_model(saved["config"], "split_rank")
    metadata = core.restore_state(path, model, None, rec["payload"]["payload_state_sha256"])
    if metadata != saved["point"]["metadata"] or core.state_digest(model.state_dict()) != rec["payload"]["model_parameters_sha256"]:
        raise ValueError("Actual restored test model differs")
    started = time.monotonic()
    values = base.score(model, groups, saved["config"], "split_rank", check)
    actual = {"model_file": data.record(path, data.ROOT), "parameters_sha256": core.state_digest(model.state_dict()),
              "score_seconds": time.monotonic() - started, "groups": len(groups), "training_updates": 0}
    if actual["parameters_sha256"] != rec["payload"]["model_parameters_sha256"]:
        raise ValueError("Inference changed model parameters")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return values, actual


def execute(out: Path, authorization_path: Path) -> dict:
    import torch
    if platform.system() != "Linux" or len(os.sched_getaffinity(0)) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("Existing Linux environment,oneCPU and one visible GPU required")
    p, snapshot = contract(), sources()
    authorization = data.read_json(authorization_path)
    if (not p["formal_execution_authorized"] or authorization["status"] != "READY_AUTHORIZED_FIXED_PAIR_TEST"
            or authorization["source_files"] != snapshot or authorization["policy_sha256"] != data.sha256(POLICY)
            or authorization["job"] != out.resolve().relative_to(data.ROOT).as_posix()
            or authorization["label_parses"] != p["access"]["label_parses"]):
        raise ValueError("Matching user-authorized and source-bound test readiness required")
    for name, status in (("cpu_evidence", "PASS_TEST_HANDMADE_AUDIT"),
                         ("review_disposition", "PASS_TEST_IMPLEMENTATION_REVIEW")):
        evidence = checked_json(authorization[name])
        if evidence["status"] != status or evidence["source_files"] != snapshot:
            raise ValueError("Actual current verification/review evidence missing")
    free_gpu, _ = torch.cuda.mem_get_info()
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    free_host = int(meminfo["MemAvailable"].split()[0]) * 1024
    if (free_gpu < p["runtime"]["minimum_free_gpu_bytes"]
            or free_host < p["runtime"]["minimum_free_host_bytes"]
            or shutil.disk_usage(data.ROOT).free < p["runtime"]["minimum_free_output_bytes"]):
        raise RuntimeError("Wait for shared resources; do not displace other jobs")
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, p["runtime"]["maximum_seconds"], p["runtime"]["maximum_output_bytes"])
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    try:
        info, partition = historical_models(p)
        groups, metadata = public_inputs(p)
        old_ids = {r["group_uid"] for role in ("fit", "calibration", "development") for r in partition[role]}
        if old_ids.intersection(g.uid for g in groups):
            raise ValueError("Test groups overlap old fit/calibration/valid")
        archive = core.model_files(base.model_config(info["A"]["config"], "split_rank"))
        expected = p["inference"]["pretrained_archive"]
        if any(archive[key] != expected[key] for key in ("content_sha256", "file_count", "total_size_bytes")):
            raise ValueError("Existing encoder/tokenizer archive changed")
        data.write_json(out / "preparation.json", {"source_files": snapshot, "authorization": data.record(authorization_path.resolve(), data.ROOT),
                        "policy": p, "group_metadata": metadata, "pretrained_archive": archive,
                        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                        "torch": torch.__version__, "cuda": torch.version.cuda,
                                        "gpu": torch.cuda.get_device_name(), "cpu_affinity": sorted(os.sched_getaffinity(0))}})
        blind = {"policy_sha256": data.sha256(POLICY), "group_ids": [g.uid for g in groups],
                 "domains": [r["domain"] for r in metadata], "models": {}, "actual_inference": {}}
        for name in ("A", "C"):
            budget.check()
            values, actual = inference_one(p, name, info, groups, budget.check)
            blind["models"][name] = save_scores(out, name, values, info[name]["map"],
                {"model": p["models"][name]["payload"], "map": p["models"][name]["calibration"]})
            blind["actual_inference"][name] = actual
            data.write_json(out / "blind_progress.json", blind)
            print(data.json_bytes({"event": "model_scored", "model": name, "groups": 120,
                                   "elapsed_seconds": time.monotonic() - budget.started}).decode(), flush=True)
        data.write_json(out / "blind.json", blind)
        scores = restore_scores(out, blind, p, info)
        if sources() != snapshot:
            raise ValueError("Source changed before heldout supervision")
        budget.check()
        truth, alignment = parse_once(out, p, groups, blind, info)
        data.write_json(out / "alignment.json", alignment)
        collect(out, truth, scores, blind, info, snapshot)
        del truth
        result = finalize(out, snapshot)
        budget.check()
        completion = {"status": COMPLETE, "source_files": snapshot, "acceptance": result["acceptance"],
                      "label_parses": p["access"]["label_parses"], "training_updates": 0, "calibration_fits": 0,
                      "seconds": time.monotonic() - budget.started, "peak_observed_output_bytes": budget.maximum_observed_bytes,
                      "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "evaluation": data.record(out / "evaluation/evaluation.json", out)}
        data.write_json(out / "completion.json", completion)
        return completion
    except Exception as error:
        data.write_json(out / "failure.json", {"status": "TEST_FAILED_NO_AUTOMATIC_RETRY", "type": type(error).__name__,
                        "message": str(error), "heldout_parse_attempts": int((out / "heldout_access.json").exists())})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "finalize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.mode == "execute" and args.authorization is None:
        parser.error("execute requires the actual reviewed readiness receipt")
    result = execute(args.out.resolve(), args.authorization.resolve()) if args.mode == "execute" else finalize(args.out.resolve(), sources())
    print(data.json_bytes({"status": result["status"], "acceptance": result["acceptance"]}).decode())


if __name__ == "__main__":
    main()
