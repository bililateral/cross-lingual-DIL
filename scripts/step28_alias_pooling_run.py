"""Scoped Linux execution and valid-only evaluation of paired item pooling.

No test/owners loader, automatic rerun, model selection or early stopping.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import step28_alias_pooling as method

base, data, core, metrics = method.base, method.data, method.core, method.metrics
COMPLETE = "COMPLETE_POOLING_4320_VALID_BLIND"


def sources() -> list[dict]:
    paths = {row["path"] for row in base.sources()}
    paths.update(("scripts/step28_alias_pooling.py", "scripts/step28_alias_pooling_run.py",
                  "scripts/step28_alias_pooling_audit.py", "schema/step28_alias_pooling_policy.json",
                  "tests/test_step28_alias_pooling_contracts.py",
                  "scripts/run_step28_alias_pooling_linux_20260925.sh",
                  "docs/SELLER_ALIAS_POOLING.zh.md",
                  "reports/documentation/20260925/retrieval_plan/authorization.json"))
    return [data.record(data.ROOT / path, data.ROOT) for path in sorted(paths)]


def historical_reference(policy: dict, partition: dict | None = None, *, verify_models: bool = False) -> dict:
    history = policy["historical_d"]
    pinned = {r["path"]: r for r in history["files"]}
    for path, record in pinned.items():
        data.verify(data.ROOT / path, record)
    root = data.ROOT / history["run_root"]
    old = data.read_json(root / "manifest.json")
    completion = data.read_json(root / "completion.json")
    part = data.read_json(root / "partition.json")
    arm = data.read_json(root / "split_rank/manifest.json")
    calibrated = data.read_json(root / "split_rank/calibration.json")
    if (old["status"] != base.COMPLETE or old["config"] != base.contract()
            or old["source_files"] != base.sources()
            or completion["manifest_sha256"] != data.sha256(root / "manifest.json")
            or old["arms"]["split_rank"]["manifest"]["sha256"] != data.sha256(root / "split_rank/manifest.json")
            or arm["updates"] != 864 or arm["group_schedule_sha256"] != history["group_schedule_sha256"]
            or arm["dropout_stream"] != history["dropout_stream"]
            or arm["preflight"]["initial_model_state_sha256"] != history["expected_initial_common_state_sha256"]
            or (partition is not None and part != partition)):
        raise ValueError("Historical D does not match its frozen comparison role")
    if (arm["calibration"]["sha256"] != data.sha256(root / "split_rank/calibration.json")
            or calibrated["model_state_sha256"] != arm["points"]["6"]["model_state_sha256"]
            or calibrated["score_sha256"] != arm["points"]["6"]["scores"]["calibration"]["sha256"]):
        raise ValueError("Historical D calibration binding differs")
    arrays, old_metrics, model_records = {}, {}, []
    for epoch in ("3", "6"):
        point = arm["points"][epoch]
        if point["model"] != history["model_records_from_frozen_manifest"]["epoch" + epoch]:
            raise ValueError("Historical model record differs")
        record = point["scores"]["development"]
        if record["path"] != f"scores/epoch{epoch}_development.npy":
            raise ValueError("Historical score role differs")
        arrays[epoch] = base.load_array(root / "split_rank" / record["path"], record, (60, 378), np.float32)
        old_path = history["evaluation_root"] + f"/split_rank/epoch{epoch}_metrics.npy"
        old_metrics[epoch] = base.load_array(data.ROOT / old_path, pinned[old_path], (60, 22), np.float64)
        if verify_models:
            path = data.verify(root / "split_rank" / point["model"]["path"], point["model"])
            model_records.append(data.record(path, data.ROOT))
    return {"points": arrays, "old_metrics": old_metrics, "threshold": calibrated["threshold"],
            "partition": part, "manifest": arm, "model_files_verified": model_records,
            "old_collected": data.read_json(data.ROOT / history["evaluation_root"] / "collected.json")}


def checkpoint(model: Any, optimizer: Any, policy: dict, c: dict, run_id: str,
               epoch: int, groups: dict, out: Path, budget: Any) -> dict:
    started = time.monotonic()
    before = {role: method.score(model, groups[role], c, budget.check) for role in base.ROLES}
    metadata = {"run_id": run_id, "epoch": epoch, "policy_sha256": method.POLICY_SHA256, "reference_config": c}
    temporary = out / "work" / f"epoch{epoch}.pt"
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(temporary, model, optimizer, metadata)
    if core.restore_state(temporary, model, optimizer, full["state_sha256"]) != metadata:
        raise ValueError("Full state metadata changed on reload")
    result = {"run_id": run_id, "epoch": epoch, "full_model_and_adam_reloaded": True,
              "checkpoint": full, "model_state_sha256": core.state_digest(model.state_dict()), "scores": {}}
    for role in base.ROLES:
        after = method.score(model, groups[role], c, budget.check)
        if not np.array_equal(before[role], after):
            raise ValueError("Actual checkpoint restore changed complete blind scores")
        path = out / "scores" / f"epoch{epoch}_{role}.npy"
        np.save(path, before[role], allow_pickle=False)
        result["scores"][role] = data.record(path, out)
    path = out / "models" / f"epoch{epoch}.pt"
    budget.check(base.persistence.checkpoint_reserve(model, None))
    inference = core.save_state(path, model, None, metadata)
    core.restore_state(path, model, None, inference["state_sha256"])
    if core.state_digest(model.state_dict()) != result["model_state_sha256"]:
        raise ValueError("Retained inference state differs")
    result["model"] = {**inference, "path": path.relative_to(out).as_posix(), "actual_reload_verified": True}
    budget.check(1)
    base.persistence.remove_work_file(temporary, out)
    result["train_metrics"] = {}
    for role in ("fit", "calibration"):
        truth = np.asarray([g.labels for g in groups[role]], dtype=np.uint8)
        matrix, counts = metrics.group_metrics(truth, before[role])
        path = out / "scores" / f"epoch{epoch}_{role}_metrics.npy"
        np.save(path, matrix, allow_pickle=False)
        result["train_metrics"][role] = {"file": data.record(path, out), "counts": counts}
    result["seconds"] = time.monotonic() - started
    return result


def preflight(policy: dict, run_id: str, groups: dict, budget: Any, *, tokenize: bool) -> dict:
    import torch

    seed_id, arm = run_id.split("_")
    c = method.reference_config(policy, seed_id)
    model = method.load_model(policy, seed_id, arm == "weighted")
    maximum, records = 0, 0
    if tokenize:
        for split in ("train", "development"):
            for group in groups[split]:
                budget.check()
                texts = base.record_texts(group, "separate_moments")
                lengths = [len(row) for row in model.encoder.tokenizer(texts, padding=False, truncation=False)["input_ids"]]
                maximum, records = max(maximum, max(lengths)), records + len(lengths)
                if maximum > c["input"]["token_budget"]:
                    raise ValueError("Public text exceeds frozen token budget")
    prediction = method.score(model, groups["train"][:1], c, budget.check)
    result = {"common_state_sha256": method.common_digest(model),
              "full_state_sha256": core.state_digest(model.state_dict()),
              "public_tokenization_performed": tokenize, "records_checked": records, "max_tokens": maximum,
              "actual_forward_shape": list(prediction.shape), "label_reads": 0, "updates": 0}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_one(out: Path, run_id: str, policy: dict, groups: dict, partition: dict,
              budget: Any, before: dict) -> dict:
    import torch

    seed_id, arm = run_id.split("_")
    c = method.reference_config(policy, seed_id)
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    initial_allocated = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device)
    out.mkdir()
    for name in ("models", "scores", "work"):
        (out / name).mkdir()
    model = method.load_model(policy, seed_id, arm == "weighted")
    if (core.state_digest(model.state_dict()) != before["full_state_sha256"]
            or method.common_digest(model) != before["common_state_sha256"]):
        raise ValueError("Formal initialization differs from public preflight")
    optimizer = method.make_optimizer(model, c, policy)
    rows, stream = base.schedule(groups["fit"], c)
    result = {"run_id": run_id, "reference_config": c, "preflight": before,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "group_schedule_sha256": hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest(),
              "dropout_stream": stream, "fit_group_ids": [g.uid for g in groups["fit"]],
              "calibration_group_ids": [g.uid for g in groups["calibration"]],
              "label_parses": 0, "training": [], "points": {}}
    for start, stop, epoch in ((0, 432, 3), (432, 864, 6)):
        segment_started = time.monotonic()
        losses, observations = [], {}
        for step in range(start, stop):
            observed = step in (0, 1, 432)
            log = method.update(model, optimizer, rows[step], c, data.seed_for(stream, step, "dropout"),
                                observe=observed, check=budget.check)
            losses.append([log[key] for key in ("bce", "rank", "total")])
            if observed:
                observations[str(step + 1)] = log["modules"]
            if step in (start, stop - 1) and (not optimizer.state or any(
                    float(state["step"]) != step + 1 for state in optimizer.state.values())):
                raise ValueError("Adam step continuity differs")
            if (step + 1) % 24 == 0:
                print(data.json_bytes({"event": "updates", "run_id": run_id, "completed": step + 1,
                                       **budget.state()}).decode(), flush=True)
        result["training"].append({"start": start, "stop": stop, "updates": stop - start,
                                   "seconds": time.monotonic() - segment_started, "observations": observations,
                                   "mean_losses_by_epoch": dict(zip(("bce", "rank", "total"),
                                       np.asarray(losses).reshape(3, 144, 3).mean(1).T.tolist()))})
        point = checkpoint(model, optimizer, policy, c, run_id, epoch, groups, out, budget)
        result["points"][str(epoch)] = point
        if epoch == 6:
            scores = np.load(out / point["scores"]["calibration"]["path"], allow_pickle=False)
            truth = np.asarray([g.labels for g in groups["calibration"]], dtype=np.uint8)
            calibrated = base.calibrate(truth, scores, [r["domain"] for r in partition["calibration"]])
            calibrated.update(epoch=6, model_state_sha256=point["model_state_sha256"],
                              score_sha256=point["scores"]["calibration"]["sha256"])
            data.write_json(out / "calibration.json", calibrated)
            result["calibration"] = data.record(out / "calibration.json", out)
        data.write_json(out / "progress.json", result)
    result["updates"] = 864
    result["formal_training_seconds"] = sum(s["seconds"] for s in result["training"])
    torch.cuda.synchronize(device)
    result["resources"] = {"arm_seconds_before_cleanup": time.monotonic() - started,
                           "gpu_name": torch.cuda.get_device_name(device),
                           "initial_allocated_bytes": initial_allocated, "initial_reserved_bytes": initial_reserved,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                           "scope": "CUDA allocator from before model load through calibration, includes baseline; excludes preflight and cleanup, not whole-device usage."}
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    data.write_json(out / "manifest.json", result)
    return result


def cpu_evidence(path: Path) -> dict:
    evidence = data.read_json(path)
    if (evidence["status"] != "PASS_POOLING_NATIVE_AND_CONTRACTS"
            or evidence["policy_sha256"] != method.POLICY_SHA256
            or evidence["method_sha256"] != data.sha256(Path(method.__file__))
            or evidence["formal_label_reads"] != 0 or evidence["native_handmade_updates"] != 4
            or evidence["contracts"]["failed"] != 0 or evidence["contracts"]["skipped"] != 0):
        raise ValueError("Missing actual native verification for the current model computation")
    return data.record(path, data.ROOT)


def train(out: Path, policy: dict, budget: Any, audit: Path) -> dict:
    import torch

    out.mkdir()
    snapshot = sources()
    result = {"status": "RUNNING", "policy": policy, "source_files": snapshot, "runs": {},
              "native_verification": cpu_evidence(audit),
              "label_parses": {"train": 0, "development": 0, "heldout": 0, "owners": 0},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(out / "startup.json", result)
    try:
        c = method.reference_config(policy, "s0")
        groups, metadata, checked = base.public.public_inputs(c)
        selected, partition = base.partition(groups, metadata, c)
        old = historical_reference(policy, partition)
        result["inputs"] = checked
        data.write_json(out / "partition.json", partition)
        result["partition"] = data.record(out / "partition.json", out)
        archive = core.model_files(base.model_config(c, "split_rank"))
        if any(archive[key] != c["models"]["split_rank"][key]
               for key in ("file_count", "total_size_bytes", "content_sha256")):
            raise ValueError("Actual pretrained archive differs")
        result["pretrained_archive"] = archive
        before = {run_id: preflight(policy, run_id, groups, budget, tokenize=i == 0)
                  for i, run_id in enumerate(method.NEW_RUNS)}
        if before["s0_weighted"]["common_state_sha256"] != policy["historical_d"]["expected_initial_common_state_sha256"]:
            raise ValueError("s0 candidate does not start at original D common weights")
        for seed_id in ("s1", "s2"):
            if before[seed_id + "_d"]["common_state_sha256"] != before[seed_id + "_weighted"]["common_state_sha256"]:
                raise ValueError("Paired common initialization differs")
        rows, stream = base.schedule(selected["fit"], c)
        if (hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest() != old["manifest"]["group_schedule_sha256"]
                or stream != old["manifest"]["dropout_stream"]):
            raise ValueError("s0 schedule does not match historical D")
        result["preflights"] = before
        data.write_json(out / "preflight.json", result)
        data.write_json(out / "access.json", {"train_parse_attempts": 1, "development": 0, "heldout": 0, "owners": 0})
        result["label_parses"]["train"] = 1
        groups["train"] = base.public.attach_labels(groups["train"], c, "train")
        selected, labelled_partition = base.partition(groups, metadata, c)
        if labelled_partition != partition:
            raise ValueError("Labels changed public partition")
        for run_id in method.NEW_RUNS:
            arm = train_one(out / run_id, run_id, policy, selected, partition, budget, before[run_id])
            result["runs"][run_id] = {"manifest": data.record(out / run_id / "manifest.json", out),
                                      "updates": arm["updates"], "formal_training_seconds": arm["formal_training_seconds"]}
            data.write_json(out / "progress.json", result)
        if sources() != snapshot or sum(r["updates"] for r in result["runs"].values()) != 4320:
            raise ValueError("Sources or formal budget changed")
        result.update(status=COMPLETE, physical_updates=4320, budget=budget.state())
        data.write_json(out / "manifest.json", result)
        data.write_json(out / "completion.json", {"status": COMPLETE, "manifest_sha256": data.sha256(out / "manifest.json")})
        return result
    except Exception as error:
        data.write_json(out / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(error),
                                               "label_parses": result["label_parses"]})
        raise


def validate_run(out: Path, policy: dict, groups: dict, metadata: list, checked: dict) -> tuple[dict, dict, dict]:
    result = data.read_json(out / "manifest.json")
    complete = data.read_json(out / "completion.json")
    if ((out / "failure.json").exists() or result["status"] != COMPLETE
            or complete["status"] != COMPLETE or complete["manifest_sha256"] != data.sha256(out / "manifest.json")
            or result["source_files"] != sources() or result["policy"] != policy
            or result["inputs"] != checked or set(result["runs"]) != set(method.NEW_RUNS)
            or result["physical_updates"] != 4320
            or result["label_parses"] != {"train": 1, "development": 0, "heldout": 0, "owners": 0}):
        raise ValueError("Formal run is incomplete or no longer matches frozen inputs")
    data.verify(data.ROOT / result["native_verification"]["path"], result["native_verification"])
    selected, partition = base.partition(groups, metadata, method.reference_config(policy, "s0"))
    if data.read_json(data.verify(out / result["partition"]["path"], result["partition"])) != partition:
        raise ValueError("Saved public partition differs")
    arrays, model_files = {}, []
    old = historical_reference(policy, partition, verify_models=True)
    arrays["s0_d"] = {"points": old["points"], "threshold": old["threshold"]}
    model_files.extend(old["model_files_verified"])
    for run_id in method.NEW_RUNS:
        record = result["runs"][run_id]
        if record["manifest"]["path"] != f"{run_id}/manifest.json" or record["updates"] != 864:
            raise ValueError("Run identity differs")
        arm = data.read_json(data.verify(out / record["manifest"]["path"], record["manifest"]))
        seed_id, kind = run_id.split("_")
        c = method.reference_config(policy, seed_id)
        schedule, stream = base.schedule(selected["fit"], c)
        if (arm["run_id"] != run_id or arm["reference_config"] != c or arm["updates"] != 864
                or arm["label_parses"] != 0 or arm["dropout_stream"] != stream
                or arm["group_schedule_sha256"] != hashlib.sha256(data.json_bytes([g.uid for g in schedule])).hexdigest()
                or arm["fit_group_ids"] != [g.uid for g in selected["fit"]]
                or arm["calibration_group_ids"] != [g.uid for g in selected["calibration"]]
                or arm["preflight"] != result["preflights"][run_id]
                or set(arm["points"]) != {"3", "6"} or len(arm["training"]) != 2):
            raise ValueError("Training scope or schedule mismatch")
        expected_modules = {"encoder", "head", "aggregation"} if kind == "weighted" else {"encoder", "head"}
        for segment, start in zip(arm["training"], (0, 432), strict=True):
            if (segment["start"] != start or segment["stop"] != start + 432 or segment["updates"] != 432
                    or set(segment["observations"]) != ({"1", "2"} if start == 0 else {"433"})):
                raise ValueError("Training update evidence differs")
            for observation in segment["observations"].values():
                if set(observation) != expected_modules or any(
                        not item["finite_nonzero_gradient"] or not item["parameters_changed"] for item in observation.values()):
                    raise ValueError("Missing actual module learning evidence")
            losses = segment["mean_losses_by_epoch"]
            if (any(np.asarray(losses[key]).shape != (3,) or not np.isfinite(losses[key]).all()
                    for key in ("bce", "rank", "total"))
                    or not np.allclose(np.asarray(losses["bce"]) + losses["rank"], losses["total"], rtol=1e-6, atol=1e-7)):
                raise ValueError("Saved loss does not perform the approved objective")
        points = {}
        for epoch in ("3", "6"):
            point = arm["points"][epoch]
            if (point["run_id"] != run_id or point["epoch"] != int(epoch)
                    or not point["full_model_and_adam_reloaded"] or not point["model"]["actual_reload_verified"]
                    or point["model"]["path"] != f"models/epoch{epoch}.pt"):
                raise ValueError("Checkpoint identity or actual restore evidence differs")
            for role in base.ROLES:
                rec = point["scores"][role]
                if rec["path"] != f"scores/epoch{epoch}_{role}.npy":
                    raise ValueError("Score role or epoch differs")
                scores = base.load_array(out / run_id / rec["path"], rec, (base.ROLE_SIZES[role], 378), np.float32)
                if role == "development":
                    points[epoch] = scores
            for role in ("fit", "calibration"):
                rec = point["train_metrics"][role]["file"]
                if rec["path"] != f"scores/epoch{epoch}_{role}_metrics.npy":
                    raise ValueError("Training metric role differs")
                base.load_array(out / run_id / rec["path"], rec, (base.ROLE_SIZES[role], 22), np.float64)
            actual = data.verify(out / run_id / point["model"]["path"], point["model"])
            model_files.append(data.record(actual, data.ROOT))
        calibrated = data.read_json(data.verify(out / run_id / arm["calibration"]["path"], arm["calibration"]))
        p6 = arm["points"]["6"]
        if (arm["calibration"]["path"] != "calibration.json" or calibrated["source_role"] != "train_calibration_only"
                or calibrated["epoch"] != 6 or calibrated["model_state_sha256"] != p6["model_state_sha256"]
                or calibrated["score_sha256"] != p6["scores"]["calibration"]["sha256"]
                or not np.isfinite(calibrated["threshold"])
                or calibrated["threshold"] != max(b["threshold"] for b in calibrated["bounds"].values())):
            raise ValueError("Calibration model/score binding differs")
        count = np.asarray(calibrated["counts_by_group"])
        if (count.shape != (36, 4) or count.dtype.kind not in "iu" or np.any(count < 0)
                or not np.all(count[:, 0] + count[:, 2] == 20)
                or not np.all(count[:, 1] + count[:, 3] == 358)):
            raise ValueError("Calibration counts differ")
        for domain in "ABC":
            bound = calibrated["bounds"][domain]
            mask = np.asarray([r["domain"] for r in partition["calibration"]]) == domain
            if (count[mask].sum(0).tolist() != calibrated["counts_by_domain"][domain]
                    or bound["negative_pairs"] != 4296 or bound["allowed_false_positives"] != 4
                    or bound["threshold"] != float(np.nextafter(np.float64(bound["next_negative_logit"]), np.inf))
                    or calibrated["counts_by_domain"][domain][1] > 4):
                raise ValueError("Calibration rule differs")
        arrays[run_id] = {"points": points, "threshold": calibrated["threshold"]}
    result["fresh_model_files_verified_before_valid"] = model_files
    return arrays, partition, result


def collect(truth: np.ndarray, arrays: dict, partition: dict, destination: Path,
            policy: dict, provenance: dict) -> dict:
    """Finish and save every metric/count before any bootstrap or old-result comparison."""
    if set(arrays) != set(method.ALL_RUNS) or truth.shape != (60, 378):
        raise ValueError("Need all paired runs and complete development supervision")
    domains = [r["domain"] for r in partition["development"]]
    result = {"status": "ALL_TWELVE_METRIC_MATRICES_COLLECTED_BEFORE_UNCERTAINTY",
              "policy": policy, "columns": list(metrics.COLUMNS), "runs": {}, "domains": domains,
              "group_ids": [r["group_uid"] for r in partition["development"]],
              "provenance": provenance,
              "label_parses": {"train": 0, "development": 1, "heldout": 0, "owners": 0}}
    for run_id in method.ALL_RUNS:
        (destination / run_id).mkdir()
        points = {}
        for epoch in ("3", "6"):
            scores = arrays[run_id]["points"][epoch]
            matrix, counts = metrics.group_metrics(truth, scores)
            path = destination / run_id / f"epoch{epoch}_metrics.npy"
            np.save(path, matrix, allow_pickle=False)
            points[epoch] = {"file": data.record(path, destination), "counts_at_logit_zero": counts,
                             "fixed_classification": base.fixed_classification(counts, domains),
                             "mean": dict(zip(metrics.COLUMNS, matrix.mean(0).tolist())),
                             "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(domains) == d].mean(0).tolist())) for d in "ABC"}}
        count = base.binary_counts(truth, arrays[run_id]["points"]["6"], arrays[run_id]["threshold"])
        result["runs"][run_id] = {"points": points, "automatic_classification": {
            "threshold": arrays[run_id]["threshold"], "counts_by_group": count.tolist()}}
    data.write_json(destination / "collected.json", result)
    return result


def finalize(destination: Path) -> dict:
    """Saved evidence only: this operation has no formal-input or label parser."""
    if (destination / "evaluation.json").exists():
        raise FileExistsError("Do not overwrite a published evaluation")
    result = data.read_json(destination / "collected.json")
    policy = method.contract()
    if (result["status"] != "ALL_TWELVE_METRIC_MATRICES_COLLECTED_BEFORE_UNCERTAINTY"
            or result["policy"] != policy or result["columns"] != list(metrics.COLUMNS)
            or set(result["runs"]) != set(method.ALL_RUNS)):
        raise ValueError("Incomplete saved evaluation collection")
    matrices = {}
    for run_id in method.ALL_RUNS:
        for epoch in ("3", "6"):
            record = result["runs"][run_id]["points"][epoch]["file"]
            if record["path"] != f"{run_id}/epoch{epoch}_metrics.npy":
                raise ValueError("Collected matrix role differs")
            value = base.load_array(destination / record["path"], record, (60, 22), np.float64)
            matrices[run_id, epoch] = value
    # Save the observed discrepancy before considering any assertion. Historical
    # matrices are results, not a fresh supervision parse.
    pinned = {r["path"]: r for r in policy["historical_d"]["files"]}
    old_collected_path = policy["historical_d"]["evaluation_root"] + "/collected.json"
    old_collected = data.read_json(data.verify(data.ROOT / old_collected_path, pinned[old_collected_path]))
    alignment = {"old_group_ids_match": old_collected["group_ids"] == result["group_ids"],
                 "old_domains_match": old_collected["domains"] == result["domains"], "points": {}}
    for epoch in ("3", "6"):
        path = policy["historical_d"]["evaluation_root"] + f"/split_rank/epoch{epoch}_metrics.npy"
        old = base.load_array(data.ROOT / path, pinned[path], (60, 22), np.float64)
        delta = matrices["s0_d", epoch] - old
        new_point = result["runs"]["s0_d"]["points"][epoch]
        old_point = old_collected["arms"]["split_rank"]["points"][epoch]
        alignment["points"][epoch] = {"max_absolute_difference": float(np.abs(delta).max()),
            "difference_by_group_metric": delta.tolist(),
            "counts_match": new_point["counts_at_logit_zero"] == old_point["counts_at_logit_zero"]}
    data.write_json(destination / "historical_alignment.json", alignment)
    if (not alignment["old_group_ids_match"] or not alignment["old_domains_match"]
            or any(p["max_absolute_difference"] > 1e-12 or not p["counts_match"] for p in alignment["points"].values())):
        raise ValueError("Historical D evaluation differs; all evidence is saved, investigate without rereading labels")
    domains = result["domains"]
    evaluation = policy["evaluation"]
    deltas = np.stack([matrices[seed + "_weighted", "6"] - matrices[seed + "_d", "6"] for seed in method.SEED_IDS])
    summary = method.paired_summary(deltas, domains, evaluation)
    result["paired_primary"] = summary
    result["acceptance"] = method.acceptance(summary, evaluation)
    draws = np.random.default_rng(evaluation["valid_bootstrap_seed"]).integers(
        0, 20, size=(evaluation["bootstrap_replicates"], 3, 20))
    result["per_seed_comparisons"] = {}
    for run_id in method.ALL_RUNS:
        automatic = result["runs"][run_id]["automatic_classification"]
        result["runs"][run_id]["automatic_classification"] = {
            **base.automatic_report(np.asarray(automatic["counts_by_group"]), domains, draws), **automatic}
    for seed in method.SEED_IDS:
        weighted, reference = seed + "_weighted", seed + "_d"
        result["per_seed_comparisons"][seed] = {
            "metrics": base.metric_comparison(matrices[weighted, "6"], matrices[reference, "6"], domains, draws),
            "automatic": base.automatic_comparison(
                np.asarray(result["runs"][weighted]["automatic_classification"]["counts_by_group"]),
                np.asarray(result["runs"][reference]["automatic_classification"]["counts_by_group"]), domains, draws)}
    result["status"] = "POOLING_VALID_EVALUATED_TEST_REMAINS_UNAUTHORIZED"
    result["interpretation"] = "MAP improvement plus nondegradation guards; fixed seeds and repeatedly developed synthetic valid only. No test access, no winner/seed selection, no claim of population noninferiority or continual improvement."
    data.write_json(destination / "evaluation.json", result)
    return result


def evaluate(out: Path, destination: Path, policy: dict, budget: Any) -> dict:
    if destination.exists():
        raise FileExistsError("Use a new evaluation directory; do not reparse labels")
    c = method.reference_config(policy, "s0")
    groups, metadata, checked = base.public.public_inputs(c)
    arrays, partition, run = validate_run(out, policy, groups, metadata, checked)
    budget.check()
    destination.mkdir()
    data.write_json(destination / "preparse_verification.json", {
        "status": "FIVE_RUNS_COMPLETE_TWELVE_MODEL_FILES_FRESHLY_VERIFIED",
        "files": run["fresh_model_files_verified_before_valid"], "sources": sources(),
        "manifest_sha256": data.sha256(out / "manifest.json")})
    data.write_json(destination / "access.json", {"development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    try:
        labelled = base.public.attach_labels(groups["development"], c, "development")
        truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
        collect(truth, arrays, partition, destination, policy,
                {"training_manifest": data.record(out / "manifest.json", data.ROOT),
                 "preparse_verification": data.record(destination / "preparse_verification.json", destination)})
        del truth, labelled
        result = finalize(destination)
        budget.check()
        return result
    except Exception as error:
        data.write_json(destination / "failure.json", {"status": "EVALUATION_FAILED_NO_LABEL_RETRY",
            "error": str(error), "complete_collection_saved": (destination / "collected.json").exists()})
        raise


def execute(job: Path, audit: Path) -> dict:
    import torch

    policy = method.contract()
    runtime = policy["runtime"]
    if (platform.system() != "Linux" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Requires existing Linux py310 and one eligible visible GPU")
    available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
                     if row.startswith("MemAvailable:"))
    if (torch.cuda.mem_get_info()[0] < runtime["minimum_free_gpu_bytes"]
            or available < runtime["minimum_free_host_bytes"]
            or shutil.disk_usage(data.ROOT).free < runtime["maximum_output_bytes"]):
        raise RuntimeError("Insufficient shared resources; wait")
    if (not job.is_relative_to((data.ROOT / "reports").resolve())
            or (job / "run").exists() or (job / "evaluation").exists()):
        raise ValueError("A new project reports job is required")
    job.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    budget = base.persistence.Budget(job, {"runtime": runtime})
    train(job / "run", policy, budget, audit)
    evaluation = evaluate(job / "run", job / "evaluation", policy, budget)
    result = {"status": "COMPLETE_POOLING_TRAIN_AND_VALID", "acceptance": evaluation["acceptance"],
              "budget": budget.state(), "formal_updates": 4320,
              "label_parses": {"train": 1, "development": 1, "heldout": 0, "owners": 0}}
    data.write_json(job / "completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("execute", "finalize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Research scripts execute only in the existing Linux py310 environment")
    if args.action == "execute":
        if args.audit is None:
            parser.error("execute requires --audit")
        result = execute(args.out.resolve(), args.audit.resolve())
    else:
        result = finalize(args.out.resolve())
    print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())


if __name__ == "__main__":
    main()
