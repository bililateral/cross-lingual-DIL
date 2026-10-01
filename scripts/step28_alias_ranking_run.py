"""Nine paired runs and one gated valid evaluation; no test/owners execution.

Formal execute requires the user's separate readiness-stage authorization.
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

import step28_alias_ranking as method

base, data, core, metrics = method.base, method.data, method.core, method.metrics
COMPLETE = "COMPLETE_RANKING_7776_VALID_BLIND"
COLLECTED = "ALL_EIGHTEEN_VALID_MATRICES_SAVED_BEFORE_STATISTICS"
OBSERVED_STEPS = (1, 2, 87, 88, 433)
UPDATE_COLUMNS = (*method.LOSS_NAMES, "encoder_lr", "head_lr")


def sources() -> list[dict]:
    paths = {row["path"] for row in base.sources()}
    paths.update(("scripts/step28_alias_ranking.py", "scripts/step28_alias_ranking_run.py",
                  "scripts/step28_alias_ranking_audit.py", "tests/test_step28_alias_ranking_contracts.py",
                  "schema/step28_alias_ranking_policy.json", "docs/SELLER_ALIAS_RANKING.zh.md",
                  "scripts/run_step28_alias_ranking_linux_20260927.sh",
                  "reports/documentation/20260927/ranking_plan/decision.json"))
    return [data.record(data.ROOT / path, data.ROOT) for path in sorted(paths)]


def verify_authorization(path: Path, policy: dict) -> dict:
    record = data.read_json(path)
    if (record.get("status") != "AUTHORIZED_RANKING_TRAIN_AND_VALID"
            or record.get("policy_sha256") != method.POLICY_SHA256
            or record.get("runs") != list(method.RUNS)
            or record.get("physical_updates") != 7776
            or record.get("train_parse_attempts") != 1 or record.get("valid_parse_attempts") != 1
            or record.get("test_access") is not False or record.get("owners_access") is not False
            or record.get("runtime") != policy["runtime"]
            or record.get("source_files") != sources()):
        raise ValueError("Formal training requires the matching user-authorized readiness scope")
    return data.record(path, data.ROOT)


def cpu_evidence(path: Path) -> dict:
    evidence = data.read_json(path)
    if (evidence["status"] != "PASS_RANKING_NATIVE_AND_CONTRACTS"
            or evidence["policy_sha256"] != method.POLICY_SHA256
            or evidence["method_sha256"] != data.sha256(Path(method.__file__))
            or evidence["source_files"] != sources()
            or evidence["native_handmade_updates"] != 4
            or evidence["formal_label_reads"] != 0 or evidence["formal_input_reads"] != 0
            or evidence["contracts"]["failed"] != 0 or evidence["contracts"]["skipped"] != 0
            or evidence["original_d_wrapper_exact"] is not True):
        raise ValueError("Missing current CPU contracts and actual native training verification")
    for record in evidence["native_reports"].values():
        data.verify(path.parent / record["path"], record)
    return data.record(path, data.ROOT)


def checkpoint(model: Any, optimizer: Any, policy: dict, config: dict, run_id: str,
               epoch: int, groups: dict, out: Path, budget: Any) -> dict:
    started = time.monotonic()
    _, arm = method.split_run(run_id)
    completed = epoch * 144
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    expected = [method.encoder_lr(policy, arm, completed), policy["encoder_schedule"]["head_lr"]]
    if rates != expected or any(float(v["step"]) != completed for v in optimizer.state.values()):
        raise ValueError("Checkpoint optimizer step or learning rate differs")
    before = {role: method.score(model, groups[role], config, budget.check) for role in base.ROLES}
    metadata = {"run_id": run_id, "epoch": epoch, "completed_updates": completed,
                "policy_sha256": method.POLICY_SHA256, "reference_config": config,
                "optimizer_lrs": rates,
                "next_encoder_lr": method.encoder_lr(policy, arm, completed + 1) if completed < 864 else None}
    temporary = out / "work" / f"epoch{epoch}.pt"
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(temporary, model, optimizer, metadata)
    if core.restore_state(temporary, model, optimizer, full["state_sha256"]) != metadata:
        raise ValueError("Actual model/Adam restore changed metadata")
    if [float(g["lr"]) for g in optimizer.param_groups] != rates:
        raise ValueError("Actual restore changed optimizer rates")
    result = {"run_id": run_id, "epoch": epoch, "metadata": metadata,
              "full_model_and_adam_reloaded": True, "checkpoint": full,
              "model_state_sha256": core.state_digest(model.state_dict()), "scores": {}}
    for role in base.ROLES:
        after = method.score(model, groups[role], config, budget.check)
        if not np.array_equal(before[role], after):
            raise ValueError("Actual restored model changed complete blind scores")
        path = out / "scores" / f"epoch{epoch}_{role}.npy"
        np.save(path, before[role], allow_pickle=False)
        result["scores"][role] = data.record(path, out)
    path = out / "models" / f"epoch{epoch}.pt"
    budget.check(base.persistence.checkpoint_reserve(model, None))
    inference = core.save_state(path, model, None, metadata)
    if core.restore_state(path, model, None, inference["state_sha256"]) != metadata:
        raise ValueError("Inference metadata changed")
    if core.state_digest(model.state_dict()) != result["model_state_sha256"]:
        raise ValueError("Inference model state differs")
    result["model"] = {**inference, "path": path.relative_to(out).as_posix(), "actual_reload_verified": True}
    budget.check(1)
    base.persistence.remove_work_file(temporary, out)
    result["train_metrics"] = {}
    for role in ("fit", "calibration"):
        truth = np.asarray([group.labels for group in groups[role]], dtype=np.uint8)
        matrix, counts = metrics.group_metrics(truth, before[role])
        path = out / "scores" / f"epoch{epoch}_{role}_metrics.npy"
        np.save(path, matrix, allow_pickle=False)
        result["train_metrics"][role] = {"file": data.record(path, out), "counts": counts}
    result["seconds"] = time.monotonic() - started
    return result


def preflight(policy: dict, run_id: str, groups: dict, budget: Any, *, tokenize: bool) -> dict:
    import torch

    seed, _ = method.split_run(run_id)
    config = method.reference_config(policy, seed)
    model = method.load_model(policy, seed)
    maximum, records = 0, 0
    if tokenize:
        for split in ("train", "development"):
            for group in groups[split]:
                budget.check()
                texts = base.record_texts(group, "separate_moments")
                lengths = [len(row) for row in model.encoder.tokenizer(
                    texts, padding=False, truncation=False)["input_ids"]]
                maximum, records = max(maximum, max(lengths)), records + len(lengths)
                if maximum > config["input"]["token_budget"]:
                    raise ValueError("Public text exceeds fixed token budget")
    scores = method.score(model, groups["train"][:1], config, budget.check)
    result = {"initial_state_sha256": core.state_digest(model.state_dict()),
              "public_tokenization_performed": tokenize, "records_checked": records,
              "max_tokens": maximum, "actual_forward_shape": list(scores.shape),
              "label_reads": 0, "updates": 0}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_one(out: Path, run_id: str, policy: dict, groups: dict, partition: dict,
              budget: Any, before: dict) -> dict:
    import torch

    seed, arm = method.split_run(run_id)
    config = method.reference_config(policy, seed)
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    initial_allocated = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device)
    out.mkdir()
    for name in ("models", "scores", "work"):
        (out / name).mkdir()
    model = method.load_model(policy, seed)
    if core.state_digest(model.state_dict()) != before["initial_state_sha256"]:
        raise ValueError("Formal initial state differs from public preflight")
    optimizer = core.make_optimizer(model, config)
    schedule, stream = base.schedule(groups["fit"], config)
    result = {"run_id": run_id, "reference_config": config, "preflight": before,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "group_schedule_sha256": hashlib.sha256(data.json_bytes([g.uid for g in schedule])).hexdigest(),
              "dropout_stream": stream, "fit_group_ids": [g.uid for g in groups["fit"]],
              "calibration_group_ids": [g.uid for g in groups["calibration"]],
              "label_parses": 0, "training": [], "points": {}, "update_columns": list(UPDATE_COLUMNS)}
    history = []
    for start, stop, epoch in ((0, 432, 3), (432, 864, 6)):
        segment_started = time.monotonic()
        observations = {}
        for index in range(start, stop):
            step = index + 1
            observed = step in OBSERVED_STEPS
            log = method.update(model, optimizer, schedule[index], config, policy, arm, step,
                                data.seed_for(stream, index, "dropout"), observe=observed, check=budget.check)
            history.append([log[key] for key in UPDATE_COLUMNS])
            if observed:
                observations[str(step)] = log["modules"]
            if not optimizer.state or any(float(value["step"]) != step for value in optimizer.state.values()):
                raise ValueError("Actual Adam continuity differs")
            if step % 24 == 0:
                print(data.json_bytes({"event": "updates", "run_id": run_id, "completed": step,
                                       "encoder_lr": log["encoder_lr"], **budget.state()}).decode(), flush=True)
        segment = np.asarray(history[start:stop], dtype=np.float64)
        result["training"].append({"start": start, "stop": stop, "updates": stop - start,
                                   "seconds": time.monotonic() - segment_started,
                                   "observations": observations,
                                   "mean_losses_by_epoch": dict(zip(method.LOSS_NAMES,
                                       segment[:, :4].reshape(3, 144, 4).mean(1).T.tolist()))})
        np.save(out / "updates.npy", np.asarray(history, dtype=np.float64), allow_pickle=False)
        result["update_log"] = data.record(out / "updates.npy", out)
        point = checkpoint(model, optimizer, policy, config, run_id, epoch, groups, out, budget)
        result["points"][str(epoch)] = point
        if epoch == 6:
            scores = np.load(out / point["scores"]["calibration"]["path"], allow_pickle=False)
            truth = np.asarray([group.labels for group in groups["calibration"]], dtype=np.uint8)
            calibrated = base.calibrate(truth, scores, [r["domain"] for r in partition["calibration"]])
            calibrated.update(epoch=6, model_state_sha256=point["model_state_sha256"],
                              score_sha256=point["scores"]["calibration"]["sha256"])
            data.write_json(out / "calibration.json", calibrated)
            result["calibration"] = data.record(out / "calibration.json", out)
        data.write_json(out / "progress.json", result)
    result["updates"] = 864
    result["encoder_positive_lr_updates"] = sum(row[4] > 0 for row in history)
    result["formal_training_seconds"] = sum(s["seconds"] for s in result["training"])
    torch.cuda.synchronize(device)
    result["resources"] = {"arm_seconds_before_cleanup": time.monotonic() - started,
                           "gpu_name": torch.cuda.get_device_name(device),
                           "initial_allocated_bytes": initial_allocated, "initial_reserved_bytes": initial_reserved,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                           "scope": "CUDA allocator for this arm, not whole-card memory or other processes."}
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    data.write_json(out / "manifest.json", result)
    return result


def train(out: Path, policy: dict, budget: Any, audit: Path, authorization: dict) -> dict:
    import torch

    out.mkdir()
    snapshot = sources()
    result = {"status": "RUNNING", "policy": policy, "source_files": snapshot, "runs": {},
              "authorization": authorization, "native_verification": cpu_evidence(audit),
              "label_parses": {"train": 0, "development": 0, "heldout": 0, "owners": 0},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(out / "startup.json", result)
    try:
        config = method.reference_config(policy, "s0")
        groups, metadata, checked = base.public.public_inputs(config)
        selected, partition = base.partition(groups, metadata, config)
        result["inputs"] = checked
        data.write_json(out / "partition.json", partition)
        result["partition"] = data.record(out / "partition.json", out)
        archive = core.model_files(base.model_config(config, "split_rank"))
        if any(archive[key] != config["models"]["split_rank"][key]
               for key in ("file_count", "total_size_bytes", "content_sha256")):
            raise ValueError("Actual pretrained archive differs")
        result["pretrained_archive"] = archive
        before = {run: preflight(policy, run, groups, budget, tokenize=i == 0)
                  for i, run in enumerate(method.RUNS)}
        for seed in method.SEEDS:
            if len({before[f"{seed}_{arm}"]["initial_state_sha256"] for arm in method.ARMS}) != 1:
                raise ValueError("Paired initial states differ")
        result["preflights"] = before
        data.write_json(out / "preflight.json", result)
        data.write_json(out / "access.json", {"train_parse_attempts": 1, "development": 0, "heldout": 0, "owners": 0})
        result["label_parses"]["train"] = 1
        groups["train"] = base.public.attach_labels(groups["train"], config, "train")
        selected, labelled_partition = base.partition(groups, metadata, config)
        if labelled_partition != partition:
            raise ValueError("Public partition changed after label alignment")
        for run_id in method.RUNS:
            arm = train_one(out / run_id, run_id, policy, selected, partition, budget, before[run_id])
            result["runs"][run_id] = {"manifest": data.record(out / run_id / "manifest.json", out),
                                      "updates": arm["updates"], "formal_training_seconds": arm["formal_training_seconds"]}
            data.write_json(out / "progress.json", result)
        if sources() != snapshot or sum(r["updates"] for r in result["runs"].values()) != 7776:
            raise ValueError("Frozen sources or total update budget changed")
        result.update(status=COMPLETE, physical_updates=7776, budget=budget.state())
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
    if ((out / "failure.json").exists() or result["status"] != COMPLETE or complete["status"] != COMPLETE
            or complete["manifest_sha256"] != data.sha256(out / "manifest.json")
            or result["source_files"] != sources() or result["policy"] != policy or result["inputs"] != checked
            or set(result["runs"]) != set(method.RUNS) or result["physical_updates"] != 7776
            or result["label_parses"] != {"train": 1, "development": 0, "heldout": 0, "owners": 0}):
        raise ValueError("Incomplete or mismatched frozen formal run")
    data.verify(data.ROOT / result["native_verification"]["path"], result["native_verification"])
    data.verify(data.ROOT / result["authorization"]["path"], result["authorization"])
    selected, partition = base.partition(groups, metadata, method.reference_config(policy, "s0"))
    if data.read_json(data.verify(out / result["partition"]["path"], result["partition"])) != partition:
        raise ValueError("Public partition differs")
    for seed in method.SEEDS:
        if len({result["preflights"][f"{seed}_{arm}"]["initial_state_sha256"] for arm in method.ARMS}) != 1:
            raise ValueError("Paired initialization differs")
    arrays, model_files = {}, []
    for run_id in method.RUNS:
        record = result["runs"][run_id]
        if record["manifest"]["path"] != f"{run_id}/manifest.json" or record["updates"] != 864:
            raise ValueError("Run identity differs")
        arm = data.read_json(data.verify(out / record["manifest"]["path"], record["manifest"]))
        seed, kind = method.split_run(run_id)
        config = method.reference_config(policy, seed)
        schedule, stream = base.schedule(selected["fit"], config)
        if (arm["run_id"] != run_id or arm["reference_config"] != config or arm["updates"] != 864
                or arm["label_parses"] != 0 or arm["dropout_stream"] != stream
                or arm["group_schedule_sha256"] != hashlib.sha256(data.json_bytes([g.uid for g in schedule])).hexdigest()
                or arm["fit_group_ids"] != [g.uid for g in selected["fit"]]
                or arm["calibration_group_ids"] != [g.uid for g in selected["calibration"]]
                or arm["preflight"] != result["preflights"][run_id]
                or set(arm["points"]) != {"3", "6"} or len(arm["training"]) != 2
                or arm["update_columns"] != list(UPDATE_COLUMNS) or arm["update_log"]["path"] != "updates.npy"):
            raise ValueError("Training identity, schedule or update evidence differs")
        updates = base.load_array(out / run_id / "updates.npy", arm["update_log"], (864, 6), np.float64)
        expected_lrs = np.asarray([method.encoder_lr(policy, kind, step) for step in range(1, 865)])
        weight = policy["hard_ranking"]["weight"] if kind == "hard" else 0.
        if (not np.array_equal(updates[:, 4], expected_lrs)
                or not np.all(updates[:, 5] == policy["encoder_schedule"]["head_lr"])
                or arm["encoder_positive_lr_updates"] != int((expected_lrs > 0).sum())
                or np.any(updates[:, :4] < 0) or (kind != "hard" and np.any(updates[:, 2] != 0))
                or not np.allclose(updates[:, 0] + updates[:, 1] + weight * updates[:, 2], updates[:, 3],
                                   rtol=1e-6, atol=1e-7)):
            raise ValueError("Actual learning rates or objective differ from the approved computation")
        for segment, start in zip(arm["training"], (0, 432), strict=True):
            expected_obs = {str(step) for step in OBSERVED_STEPS if start < step <= start + 432}
            if (segment["start"] != start or segment["stop"] != start + 432 or segment["updates"] != 432
                    or set(segment["observations"]) != expected_obs):
                raise ValueError("Observed update scope differs")
            for observation in segment["observations"].values():
                if set(observation) != {"encoder", "head"} or any(
                        not row["finite_nonzero_gradient"] or not row["parameters_changed"] for row in observation.values()):
                    raise ValueError("Actual module learning evidence is missing")
            means = updates[start:start + 432, :4].reshape(3, 144, 4).mean(1)
            for i, name in enumerate(method.LOSS_NAMES):
                if not np.array_equal(means[:, i], np.asarray(segment["mean_losses_by_epoch"][name])):
                    raise ValueError("Epoch summary differs from actual per-update evidence")
        points = {}
        for epoch in ("3", "6"):
            point = arm["points"][epoch]
            step = int(epoch) * 144
            metadata_expected = {"run_id": run_id, "epoch": int(epoch), "completed_updates": step,
                                 "policy_sha256": method.POLICY_SHA256, "reference_config": config,
                                 "optimizer_lrs": [method.encoder_lr(policy, kind, step), policy["encoder_schedule"]["head_lr"]],
                                 "next_encoder_lr": method.encoder_lr(policy, kind, step + 1) if step < 864 else None}
            if (point["run_id"] != run_id or point["epoch"] != int(epoch) or point["metadata"] != metadata_expected
                    or not point["full_model_and_adam_reloaded"] or not point["model"]["actual_reload_verified"]
                    or point["model"]["path"] != f"models/epoch{epoch}.pt"):
                raise ValueError("Model checkpoint identity or actual restore differs")
            for role in base.ROLES:
                rec = point["scores"][role]
                if rec["path"] != f"scores/epoch{epoch}_{role}.npy":
                    raise ValueError("Blind score role differs")
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
            raise ValueError("Calibration binding differs")
        count = np.asarray(calibrated["counts_by_group"])
        if (count.shape != (36, 4) or count.dtype.kind not in "iu" or np.any(count < 0)
                or not np.all(count[:, 0] + count[:, 2] == 20) or not np.all(count[:, 1] + count[:, 3] == 358)):
            raise ValueError("Calibration confusion counts differ")
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
    if set(arrays) != set(method.RUNS) or truth.shape != (60, 378):
        raise ValueError("Complete nine runs and valid supervision required")
    domains = [r["domain"] for r in partition["development"]]
    result = {"status": COLLECTED, "policy": policy, "columns": list(metrics.COLUMNS), "runs": {},
              "domains": domains, "group_ids": [r["group_uid"] for r in partition["development"]],
              "provenance": provenance, "label_parses": {"train": 0, "development": 1, "heldout": 0, "owners": 0}}
    for run_id in method.RUNS:
        (destination / run_id).mkdir()
        points = {}
        for epoch in ("3", "6"):
            matrix, counts = metrics.group_metrics(truth, arrays[run_id]["points"][epoch])
            path = destination / run_id / f"epoch{epoch}_metrics.npy"
            np.save(path, matrix, allow_pickle=False)
            points[epoch] = {"file": data.record(path, destination), "counts_at_logit_zero": counts,
                             "fixed_classification": base.fixed_classification(counts, domains),
                             "mean": dict(zip(metrics.COLUMNS, matrix.mean(0).tolist())),
                             "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(domains) == d].mean(0).tolist()))
                                           for d in "ABC"}}
        counts = base.binary_counts(truth, arrays[run_id]["points"]["6"], arrays[run_id]["threshold"])
        result["runs"][run_id] = {"points": points, "automatic_classification": {
            "threshold": arrays[run_id]["threshold"], "counts_by_group": counts.tolist()}}
    data.write_json(destination / "collected.json", result)
    return result


def finalize(destination: Path) -> dict:
    """Only saved evidence; no formal data/model/label loading or role selection."""
    if (destination / "evaluation.json").exists():
        raise FileExistsError("Published evaluation must not be overwritten")
    result = data.read_json(destination / "collected.json")
    policy = method.contract()
    if (result["status"] != COLLECTED or result["policy"] != policy
            or result["columns"] != list(metrics.COLUMNS) or set(result["runs"]) != set(method.RUNS)):
        raise ValueError("Incomplete or mismatched collected evaluation")
    matrices = {}
    for run_id in method.RUNS:
        for epoch in ("3", "6"):
            record = result["runs"][run_id]["points"][epoch]["file"]
            if record["path"] != f"{run_id}/epoch{epoch}_metrics.npy":
                raise ValueError("Saved matrix role differs")
            matrices[run_id, epoch] = base.load_array(destination / record["path"], record, (60, 22), np.float64)
    domains, evaluation = result["domains"], policy["evaluation"]
    result["comparisons"] = {}
    for candidate, reference in [policy["primary_comparison"], *policy["auxiliary_comparisons"]]:
        deltas = np.stack([matrices[f"{seed}_{candidate}", "6"] - matrices[f"{seed}_{reference}", "6"]
                           for seed in method.SEEDS])
        result["comparisons"][f"{candidate}_minus_{reference}"] = method.paired_summary(deltas, domains, evaluation)
    result["acceptance"] = method.acceptance(result["comparisons"]["hard_minus_d"], evaluation)
    draws = np.random.default_rng(evaluation["valid_bootstrap_seed"]).integers(
        0, 20, size=(evaluation["bootstrap_replicates"], 3, 20))
    for run_id in method.RUNS:
        automatic = result["runs"][run_id]["automatic_classification"]
        result["runs"][run_id]["automatic_classification"] = {
            **base.automatic_report(np.asarray(automatic["counts_by_group"]), domains, draws), **automatic}
    result["per_seed_comparisons"] = {}
    for candidate, reference in [policy["primary_comparison"], *policy["auxiliary_comparisons"]]:
        name = f"{candidate}_minus_{reference}"
        result["per_seed_comparisons"][name] = {}
        for seed in method.SEEDS:
            cand, ref = f"{seed}_{candidate}", f"{seed}_{reference}"
            result["per_seed_comparisons"][name][seed] = {
                "metrics": base.metric_comparison(matrices[cand, "6"], matrices[ref, "6"], domains, draws),
                "automatic": base.automatic_comparison(
                    np.asarray(result["runs"][cand]["automatic_classification"]["counts_by_group"]),
                    np.asarray(result["runs"][ref]["automatic_classification"]["counts_by_group"]), domains, draws)}
    result["status"] = "RANKING_VALID_EVALUATED_TEST_UNAUTHORIZED"
    result["interpretation"] = "Only hard minus d E6 is primary. Auxiliary comparisons are unadjusted descriptive evidence; no selection of arm, seed or epoch; synthetic repeatedly developed valid only."
    data.write_json(destination / "evaluation.json", result)
    return result


def evaluate(out: Path, destination: Path, policy: dict, budget: Any) -> dict:
    if destination.exists():
        raise FileExistsError("New evaluation directory required; no label reparse")
    config = method.reference_config(policy, "s0")
    groups, metadata, checked = base.public.public_inputs(config)
    arrays, partition, run = validate_run(out, policy, groups, metadata, checked)
    budget.check()
    destination.mkdir()
    data.write_json(destination / "preparse_verification.json", {
        "status": "NINE_RUNS_COMPLETE_EIGHTEEN_WEIGHTS_FRESHLY_VERIFIED",
        "files": run["fresh_model_files_verified_before_valid"], "sources": sources(),
        "manifest_sha256": data.sha256(out / "manifest.json")})
    data.write_json(destination / "access.json", {"development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    try:
        labelled = base.public.attach_labels(groups["development"], config, "development")
        truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
        collect(truth, arrays, partition, destination, policy, {
            "training_manifest": data.record(out / "manifest.json", data.ROOT),
            "preparse_verification": data.record(destination / "preparse_verification.json", destination)})
        del truth, labelled
        result = finalize(destination)
        budget.check()
        return result
    except Exception as error:
        data.write_json(destination / "failure.json", {"status": "EVALUATION_FAILED_NO_LABEL_RETRY",
            "error": str(error), "complete_collection_saved": (destination / "collected.json").exists()})
        raise


def execute(job: Path, audit: Path, authorization: Path) -> dict:
    import torch

    policy = method.contract()
    approved = verify_authorization(authorization, policy)
    runtime = policy["runtime"]
    if (platform.system() != "Linux" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Existing Linux py310 and one eligible visible GPU required")
    available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
                     if row.startswith("MemAvailable:"))
    if (torch.cuda.mem_get_info()[0] < runtime["minimum_free_gpu_bytes"]
            or available < runtime["minimum_free_host_bytes"]
            or shutil.disk_usage(data.ROOT).free < runtime["maximum_output_bytes"]):
        raise RuntimeError("Insufficient shared resources; wait")
    if (not job.is_relative_to((data.ROOT / "reports").resolve())
            or (job / "run").exists() or (job / "evaluation").exists()):
        raise ValueError("A new reports job directory is required")
    job.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    budget = base.persistence.Budget(job, {"runtime": runtime})
    train(job / "run", policy, budget, audit, approved)
    evaluation = evaluate(job / "run", job / "evaluation", policy, budget)
    result = {"status": "COMPLETE_RANKING_TRAIN_AND_VALID", "acceptance": evaluation["acceptance"],
              "budget": budget.state(), "formal_updates": 7776,
              "label_parses": {"train": 1, "development": 1, "heldout": 0, "owners": 0}}
    data.write_json(job / "completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("execute", "finalize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Research scripts run only in existing Linux py310")
    if args.action == "execute":
        if args.audit is None or args.authorization is None:
            parser.error("execute requires current --audit and separate --authorization")
        result = execute(args.out.resolve(), args.audit.resolve(), args.authorization.resolve())
    else:
        result = finalize(args.out.resolve())
    print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())


if __name__ == "__main__":
    main()
