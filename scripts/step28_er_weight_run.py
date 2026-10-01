"""ER weight continuations with exact historical starts and one gated valid collection."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np

import step28_er_weight as method
import step28_er_weight_evaluate as evaluation

parent, prior = method.parent, method.prior
base, data, core = method.base, method.data, method.core
save_array, rng_state, restore_rng = prior.save_array, prior.rng_state, prior.restore_rng
COMPLETE = "COMPLETE_3456_ER_WEIGHT_UPDATES_VALID_BLIND"


def old_point(reference: dict, name: str) -> dict:
    root = reference["job"] / "run"
    return data.read_json(data.verify(root / reference["manifest"]["points"][name]["path"],
                                      reference["manifest"]["points"][name]))


def old_training(reference: dict, order: str, stage: int) -> dict:
    rec = reference["manifest"]["training"][f"{order}_er_stage{stage}"]
    return data.read_json(data.verify(reference["job"] / "run" / rec["path"], rec))


def train_stage(model: Any, optimizer: Any, current: list, memory: Any, c: dict,
                order: str, stage: int, arm: str, root: Path, reference_log: dict,
                budget: Any) -> dict:
    import torch

    started = time.monotonic()
    old_policy = parent.contract()
    sequence, stream = parent.schedule(current, old_policy, order, stage)
    if [g.uid for g in sequence] != reference_log["current_ids"]:
        raise ValueError("Current schedule differs from original ER")
    memory.begin_stage(stage)
    updates, history_ids, observations = [], [], {}
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for index, group in enumerate(sequence):
        history, target = memory.draw()
        if target is not None or history.uid != reference_log["history_ids"][index]:
            raise ValueError("ER history is unpaired or contains logits")
        history_ids.append(history.uid)
        record = method.update(
            model, optimizer, group, history, c, method.ARMS[arm], stage, index + 1,
            data.seed_for(stream, index, "dropout"),
            data.seed_for(old_policy["memory_seed"], order, stage, index, "history_dropout"),
            observe=index + 1 in (1, 29, 30, 288), check=budget.check)
        updates.append([record[key] for key in method.STEP_COLUMNS])
        if record["modules"]:
            observations[str(index + 1)] = record["modules"]
        if (index + 1) % 24 == 0:
            print(data.json_bytes({"event": "updates", "order": order, "arm": arm,
                                   "stage": stage, "stage_updates": index + 1,
                                   "logical_updates": parent.adam_step(optimizer),
                                   **budget.state()}).decode(), flush=True)
    summary = memory.summary()
    if (summary["draw_count"] != 288 or summary["members"] != reference_log["memory_after_training"]["members"]
            or history_ids != reference_log["history_ids"]):
        raise ValueError("Historical supply differs after training")
    name = method.point_name(order, arm, stage)
    result = {"name": name, "order": order, "arm": arm, "stage": stage,
              "history_weight": method.ARMS[arm], "updates": 288,
              "adam_step": parent.adam_step(optimizer), "actual_domain": order[stage - 1],
              "current_ids": [g.uid for g in sequence], "history_ids": history_ids,
              "current_dropout_stream": stream, "memory_after_training": summary,
              "observations": observations, "update_columns": list(method.STEP_COLUMNS),
              "update_file": save_array(root / "updates" / (name + ".npy"),
                                        np.asarray(updates, dtype=np.float64), root)}
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
        result["cuda_allocator"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    result["training_seconds"] = time.monotonic() - started
    data.write_json(root / "updates" / (name + ".json"), result)
    return result


def checkpoint(root: Path, name: str, model: Any, optimizer: Any, c: dict,
               order: str, stage: int, current_cal: list, valid: list,
               first_map: dict, budget: Any, weight: float) -> dict:
    """Real full state restore and complete score replay at every endpoint."""
    started = time.monotonic()
    arm = next((arm for arm, value in method.ARMS.items() if value == weight), None)
    if (arm is None or name != method.point_name(order, arm, stage) or first_map is None
            or len(current_cal) != 12 or len(valid) != 60 or parent.adam_step(optimizer) != stage * 288):
        raise ValueError("Checkpoint is not a complete stage with current-only calibration")
    state = rng_state()
    metadata = {"name": name, "order": order, "stage": stage, "seed": "s0",
                "completed_updates": stage * 288, "policy_sha256": method.POLICY_SHA256,
                "history_weight": weight, "parent_policy_sha256": parent.POLICY_SHA256,
                "rng": state, "config": c}
    scores = {"calibration": parent.ranking.score(model, current_cal, c, budget.check),
              "development": parent.ranking.score(model, valid, c, budget.check)}
    full_path = root / "work" / (name + ".pt")
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(full_path, model, optimizer, metadata)
    if core.restore_state(full_path, model, optimizer, full["state_sha256"]) != metadata:
        raise ValueError("Full model/Adam metadata differs")
    restore_rng(state)
    for role, groups in (("calibration", current_cal), ("development", valid)):
        if not np.array_equal(scores[role], parent.ranking.score(model, groups, c, budget.check)):
            raise ValueError("Actual full restore changed complete scores")
    model_digest = core.state_digest(model.state_dict())
    path = root / "models" / (name + ".pt")
    budget.check(base.persistence.checkpoint_reserve(model, None))
    inference = core.save_state(path, model, None, metadata)
    core.restore_state(path, model, None, inference["state_sha256"])
    if core.state_digest(model.state_dict()) != model_digest:
        raise ValueError("Inference state restore differs")
    score_records = {role: save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
                     for role, values in scores.items()}
    truth = np.asarray([group.labels for group in current_cal], dtype=np.uint8)
    mapping = parent.calibration.fit(scores["calibration"], truth, role="calibration", check=budget.check)
    mapping.update(name=name, actual_domain=order[stage - 1],
                   calibration_group_ids=[g.uid for g in current_cal], model_state_sha256=model_digest,
                   score_source=score_records["calibration"])
    map_path = root / "maps" / (name + ".json")
    data.write_json(map_path, mapping)  # Preserve actual solver failure before refusing continuation.
    if mapping["status"] != "PASS_CALIBRATION_FIT":
        raise RuntimeError("Stage calibration failed; no alternative fit or retry")
    restored_map = data.read_json(map_path)
    if restored_map != mapping:
        raise ValueError("Serialized mapping differs")
    first = first_map
    for role, use_map in (("stage-cal", mapping), ("first-cal", first)):
        values = parent.calibration.transform(scores["development"], use_map)
        parent.calibration.preserve_order(scores["development"], values)
        score_records[role] = save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
    # RNG is part of the shared branch state; checkpoint/evaluation cannot advance it.
    restore_rng(state)
    auxiliary = {"checkpoint_metadata": metadata,
                 "first_map": dict(zip(("a", "b"), parent.calibration.parameters(first))),
                 "stage_map": dict(zip(("a", "b"), parent.calibration.parameters(mapping))),
                 "history_weight": weight, "er_weight_policy_sha256": method.POLICY_SHA256}
    auxiliary_bytes = len(data.json_bytes(auxiliary))
    if auxiliary_bytes > 1048576:
        raise ValueError("Common phase/RNG/map metadata alone exceeds history allowance")
    result = {"name": name, "order": order, "stage": stage, "actual_domain": order[stage - 1],
              "completed_updates": stage * 288, "full_model_adam_and_rng_restore_verified": True,
              "history_weight": weight, "policy_sha256": method.POLICY_SHA256,
              "model_state_sha256": model_digest,
              "model": {**inference, "path": path.relative_to(root).as_posix()},
              "full_checkpoint": {**full, "path": full_path.relative_to(root).as_posix()},
              "full_checkpoint_retained": False,
              "scores": score_records, "mapping": data.record(map_path, root),
              "learner_auxiliary": auxiliary, "learner_auxiliary_serialized_bytes": auxiliary_bytes,
              "first_map_parameters": dict(zip(("a", "b"), parent.calibration.parameters(first)))}
    # Verified intermediate file only; all inference endpoints remain active research evidence.
    base.persistence.remove_work_file(full_path, root)
    result["intermediate_deleted_bytes"] = full["bytes"]
    result["checkpoint_calibration_score_seconds"] = time.monotonic() - started
    data.write_json(root / "points" / (name + ".json"), result)
    return result


def train(job: Path, p: dict, budget: Any) -> tuple[dict, dict, list, dict]:
    import torch

    reference = method.baseline(p)
    old_root = reference["job"] / "run"
    root = job / "run"
    root.mkdir()
    for name in ("models", "work", "scores", "maps", "points", "memory", "updates"):
        (root / name).mkdir()
    c = method.config(p)
    groups, metadata, checked = base.public.public_inputs(c)
    _, partition = base.partition(groups, metadata, c)
    if partition != reference["partition"] or checked != reference["manifest"]["public_inputs"]:
        raise ValueError("Inputs/partition differ from the paired original experiment")
    data.write_json(root / "partition.json", partition)
    archive = core.model_files(base.model_config(c, "split_rank"))
    if any(archive[k] != c["models"]["split_rank"][k]
           for k in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Pretrained archive differs")
    # Verify all required original payload identities before consuming new supervision.
    for order in method.ORDERS:
        shared = old_point(reference, order + "_shared")
        data.verify(old_root / shared["full_checkpoint"]["path"], shared["full_checkpoint"])
        rec = reference["manifest"]["memories"][order + "_er_after1"]["file"]
        data.verify(old_root / rec["path"], rec)
    groups["train"] = prior.parse_once(job, groups["train"], c, "train")
    selected, after = base.partition(groups, metadata, c)
    if after != partition:
        raise ValueError("Partition changed after training-label alignment")
    supply = method.ContinuationSupply(selected, partition)
    manifest = {"status": "RUNNING", "source_files": method.sources(), "policy_sha256": method.POLICY_SHA256,
                "baseline_records": p["baseline"]["records"], "public_inputs": checked,
                "pretrained_archive": archive, "partition": data.record(root / "partition.json", root),
                "points": {}, "training": {}, "memories": {}, "restored_starts": {},
                "physical_updates": 0, "gradient_group_presentations": 0}
    data.write_json(root / "startup.json", manifest)
    for order in method.ORDERS:
        shared = old_point(reference, order + "_shared")
        rec = reference["manifest"]["memories"][order + "_er_after1"]["file"]
        memory_payload = data.verify(old_root / rec["path"], rec).read_bytes()
        first_scores = prior.array(old_root, shared["scores"]["development"], (60, 378), np.float32)
        for arm, weight in method.ARMS.items():
            path = order + "_" + arm
            model, optimizer = prior.restore_branch(old_root, shared, c)
            state = rng_state()
            if not np.array_equal(first_scores, parent.ranking.score(model, groups["development"], c, budget.check)):
                raise ValueError("Restored first-state full blind scores differ from original runtime")
            restore_rng(state)
            memory = parent.Memory.from_bytes(memory_payload)
            if (memory.with_logits or memory.order != order or memory.reservoir.seen != 48
                    or memory.draw_stage != 0 or memory.draw_count != 0
                    or memory.first_map != shared["first_map_parameters"]):
                raise ValueError("Original ER cache/first-map restore differs")
            memory.auxiliary = {**shared["learner_auxiliary"], "history_weight": weight,
                                "er_weight_policy_sha256": method.POLICY_SHA256}
            memory.to_bytes()
            supply.resume(path, order, shared)
            manifest["restored_starts"][path] = {
                "full_checkpoint": shared["full_checkpoint"], "memory_source": rec,
                "first_scores_replayed_exactly": True, "model_state_sha256": shared["model_state_sha256"],
                "adam_step": parent.adam_step(optimizer), "first_map": memory.first_map,
                "memory_summary": memory.summary()}
            for stage in (2, 3):
                current, current_cal = supply.current(path, order, stage)
                name = method.point_name(order, arm, stage)
                train_stage(model, optimizer, current, memory, c, order, stage, arm,
                            root, old_training(reference, order, stage), budget)
                point = checkpoint(root, name, model, optimizer, c, order, stage,
                                   current_cal, groups["development"], memory.first_map, budget, weight)
                memory.auxiliary = point["learner_auxiliary"]
                data.write_json(root / "memory" / (name + "_budget.json"), memory.summary())
                if stage == 2:
                    memory.retain(current, 2, None)
                    expected = reference["manifest"]["memories"][f"{order}_er_stage2"]
                    if memory.summary()["members"] != expected["members"]:
                        raise ValueError("Stage-end reservoir members differ from original ER")
                    manifest["memories"][name] = prior.save_memory(root, name, memory)
                manifest["points"][name] = data.record(root / "points" / (name + ".json"), root)
                manifest["training"][name] = data.record(root / "updates" / (name + ".json"), root)
                manifest["physical_updates"] += 288
                manifest["gradient_group_presentations"] += 576
                data.write_json(root / "progress.json", manifest)
                del current, current_cal
            del model, optimizer, memory
            gc.collect()
            torch.cuda.empty_cache()
    if method.sources() != manifest["source_files"]:
        raise ValueError("Scientific sources changed")
    manifest.update(status=COMPLETE, budget=budget.state())
    data.write_json(root / "manifest.json", manifest)
    return manifest, partition, groups["development"], reference


def blind_gate(root: Path, manifest: dict, reference: dict) -> dict:
    """Reject incomplete, unpaired, or incorrectly weighted updates before valid labels."""
    if (manifest["status"] != COMPLETE or manifest["source_files"] != method.sources()
            or manifest["policy_sha256"] != method.POLICY_SHA256
            or manifest["physical_updates"] != 3456 or manifest["gradient_group_presentations"] != 6912
            or set(manifest["points"]) != set(method.expected_points())
            or set(manifest["training"]) != set(method.expected_points())
            or set(manifest["memories"]) != {method.point_name(o, a, 2) for o in method.ORDERS for a in method.ARMS}
            or set(manifest["restored_starts"]) != {o + "_" + a for o in method.ORDERS for a in method.ARMS}):
        raise ValueError("Incomplete ER weight experiment")
    partition = data.read_json(data.verify(root / manifest["partition"]["path"], manifest["partition"]))
    if partition != reference["partition"]:
        raise ValueError("Blind partition changed")
    result = {}
    for order in method.ORDERS:
        for arm, weight in method.ARMS.items():
            start = manifest["restored_starts"][order + "_" + arm]
            shared = old_point(reference, order + "_shared")
            initial_memory = reference["manifest"]["memories"][order + "_er_after1"]
            if (start["full_checkpoint"] != shared["full_checkpoint"] or start["adam_step"] != 288
                    or not start["first_scores_replayed_exactly"]
                    or start["model_state_sha256"] != shared["model_state_sha256"]
                    or start["first_map"] != shared["first_map_parameters"]
                    or start["memory_source"] != initial_memory["file"]
                    or start["memory_summary"]["members"] != initial_memory["members"]
                    or start["memory_summary"]["with_logits"]
                    or start["memory_summary"]["seen"] != 48):
                raise ValueError("Shared start evidence differs")
            retained = manifest["memories"][method.point_name(order, arm, 2)]
            data.verify(root / retained["file"]["path"], retained["file"])
            if (retained["members"] != reference["manifest"]["memories"][f"{order}_er_stage2"]["members"]
                    or retained["seen"] != 96 or retained["with_logits"]
                    or retained["serialized_bytes"] != retained["file"]["bytes"]
                    or retained["serialized_bytes"] > 1048576):
                raise ValueError("Saved next-stage cache is incomplete or unpaired")
            for stage in (2, 3):
                name = method.point_name(order, arm, stage)
                point_rec, log_rec = manifest["points"][name], manifest["training"][name]
                point = data.read_json(data.verify(root / point_rec["path"], point_rec))
                log = data.read_json(data.verify(root / log_rec["path"], log_rec))
                if (point["name"] != name or point["order"] != order or point["stage"] != stage
                        or point["actual_domain"] != order[stage - 1] or point["completed_updates"] != stage * 288
                        or point["history_weight"] != weight or not point["full_model_adam_and_rng_restore_verified"]
                        or point["policy_sha256"] != method.POLICY_SHA256):
                    raise ValueError("Native endpoint identity or actual restoration differs")
                data.verify(root / point["model"]["path"], point["model"])
                original_log = old_training(reference, order, stage)
                for field in ("current_ids", "history_ids", "current_dropout_stream", "adam_step", "updates"):
                    if log[field] != original_log[field]:
                        raise ValueError("Training schedule is unpaired: " + field)
                if (log["update_columns"] != list(method.STEP_COLUMNS)
                        or log["history_weight"] != weight
                        or log["memory_after_training"]["members"] != original_log["memory_after_training"]["members"]
                        or log["memory_after_training"]["serialized_bytes"] > 1048576):
                    raise ValueError("Replay configuration or memory differs")
                values = prior.array(root, log["update_file"], (288, len(method.STEP_COLUMNS)), np.float64)
                columns = {key: values[:, i] for i, key in enumerate(method.STEP_COLUMNS)}
                for role in ("current", "history"):
                    if not np.allclose(columns[role + "_total"], columns[role + "_bce"]
                                       + columns[role + "_rank"] + .5 * columns[role + "_hard"],
                                       rtol=3e-6, atol=3e-6):
                        raise ValueError("Base objective decomposition differs")
                if (not np.array_equal(columns["history_weight"], np.full(288, weight))
                        or not np.array_equal(columns["weighted_history_total"], weight * columns["history_total"])
                        or not np.array_equal(columns["total"], columns["current_total"] + columns["weighted_history_total"])
                        or not np.array_equal(columns["encoder_lr"], [parent.stage_lr(i) for i in range(1, 289)])
                        or not np.all(columns["head_lr"] == .001)):
                    raise ValueError("Historical weight, total, or learning-rate log is wrong")
                if set(log["observations"]) != {"1", "29", "30", "288"}:
                    raise ValueError("Missing real gradient/update observations")
                for step, modules in log["observations"].items():
                    for module in ("encoder", "head"):
                        if (not modules[module]["finite_nonzero_combined_gradient"]
                                or modules[module]["parameters_changed"] != (module == "head" or step != "288")):
                            raise ValueError("Missing real encoder/head update")
                scores = point["scores"]
                raw = prior.array(root, scores["development"], (60, 378), np.float32)
                prior.array(root, scores["calibration"], (12, 378), np.float32)
                mapping = data.read_json(data.verify(root / point["mapping"]["path"], point["mapping"]))
                expected_cal = [row["group_uid"] for row in partition["calibration"] if row["domain"] == order[stage - 1]]
                if (mapping["calibration_group_ids"] != expected_cal
                        or mapping["actual_domain"] != order[stage - 1]
                        or mapping["group_count"] != 12 or mapping["pair_count"] != 4536
                        or mapping["positive_count"] != 240
                        or mapping["score_source"] != scores["calibration"]
                        or mapping["model_state_sha256"] != point["model_state_sha256"]
                        or mapping["status"] != "PASS_CALIBRATION_FIT"
                        or point["first_map_parameters"] != start["first_map"]):
                    raise ValueError("Current-only calibration or first map differs")
                roles = {"raw": raw}
                for role, use_map in (("stage-cal", mapping), ("first-cal", start["first_map"])):
                    transformed = prior.array(root, scores[role], (60, 378), np.float64)
                    if not np.array_equal(transformed, parent.calibration.transform(raw, use_map)):
                        raise ValueError("Saved calibration transform differs")
                    parent.calibration.preserve_order(raw, transformed)
                    roles[role] = transformed
                result[name] = roles
    return result


def collect(root: Path, scores: dict, labelled: list, partition: dict, source_files: list) -> dict:
    root.mkdir()
    if set(scores) != set(method.expected_points()):
        raise ValueError("Collect all new points together")
    if [g.uid for g in labelled] != [row["group_uid"] for row in partition["development"]]:
        raise ValueError("Valid truth and blind group identities differ")
    truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    collected = {"status": "COLLECTING", "points": {}, "source_files": source_files,
                 "policy_sha256": method.POLICY_SHA256, "metric_columns": list(method.metrics.COLUMNS),
                 "group_ids": [g.uid for g in labelled],
                 "domains": [row["domain"] for row in partition["development"]]}
    for name in method.expected_points():
        if set(scores[name]) != set(method.ROLES):
            raise ValueError("Missing output role")
        entries = {}
        for role in method.ROLES:
            matrix, counts = method.metrics.group_metrics(truth, scores[name][role])
            rec = save_array(root / f"{name}_{role}.npy", matrix, root)
            counts_path = root / f"{name}_{role}_counts.json"
            data.write_json(counts_path, counts)
            entries[role] = {"matrix": rec, "counts": data.record(counts_path, root)}
        collected["points"][name] = entries
    collected["status"] = "ALL_36_ER_WEIGHT_MATRICES_SAVED_BEFORE_COMPARISONS"
    data.write_json(root / "collected.json", collected)
    return collected


def execute(job: Path, audit_path: Path, authorization_path: Path) -> dict:
    import torch

    p, source_files = method.contract(), method.sources()
    auth = data.read_json(authorization_path)
    audit = data.read_json(audit_path)
    if (platform.system() != "Linux" or auth.get("status") != "AUTHORIZED_ER_WEIGHT_AFTER_REVIEW"
            or auth.get("policy_sha256") != method.POLICY_SHA256 or auth.get("source_files") != source_files
            or auth.get("job") != job.relative_to(data.ROOT).as_posix()
            or auth.get("runtime") != p["runtime"] or auth.get("supervision") != p["supervision"]
            or auth.get("review_and_primary_passed") is not True):
        raise ValueError("Matching reviewed formal authorization is required")
    if (audit.get("status") != "PASS_ER_WEIGHT_HANDMADE_CPU" or audit.get("source_files") != source_files
            or audit.get("contracts", {}).get("failed") != 0 or audit.get("contracts", {}).get("skipped") != 0
            or audit.get("formal_inputs") is not False or audit.get("formal_labels") is not False
            or set(audit.get("native", {})) != set(method.ARMS)
            or audit.get("native_history_gradient_scaling_verified") is not True):
        raise ValueError("Necessary current handmade/native CPU verification is missing")
    for record in audit["native"].values():
        data.verify(audit_path.parent / record["path"], record)
    if (not job.is_relative_to((data.ROOT / "reports").resolve())
            or any((job / n).exists() for n in ("run", "evaluation", "access.json"))):
        raise ValueError("Use a new reports job; formal entry cannot resume")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One eligible visible GPU required")
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    if environment != p["baseline"]["environment"]:
        raise RuntimeError("Paired runtime differs; do not silently retrain controls")
    available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
                     if row.startswith("MemAvailable:"))
    if (torch.cuda.mem_get_info()[0] < p["runtime"]["minimum_free_gpu_bytes"]
            or available < p["runtime"]["minimum_free_host_bytes"]
            or shutil.disk_usage(data.ROOT).free < p["runtime"]["maximum_output_bytes"]):
        raise RuntimeError("Wait for resources without affecting other users")
    job.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    budget = base.persistence.Budget(job, {"runtime": p["runtime"]})
    data.write_json(job / "access.json", {"train": 0, "valid": 0, "heldout": 0, "owners": 0})
    data.write_json(job / "execution.json", {**environment, "source_files": source_files,
                                            "authorization": data.record(authorization_path, data.ROOT),
                                            "audit": data.record(audit_path, data.ROOT),
                                            "cpu_affinity": sorted(os.sched_getaffinity(0))})
    try:
        manifest, partition, valid, reference = train(job, p, budget)
        scores = blind_gate(job / "run", manifest, reference)
        data.write_json(job / "before_valid.json", {"status": "PASS_ER_WEIGHT_COMPLETE_BLIND_GATE",
                                                    "manifest": data.record(job / "run/manifest.json", job),
                                                    "label_parses": data.read_json(job / "access.json")})
        labelled = prior.parse_once(job, valid, method.config(p), "development")
        collect(job / "evaluation", scores, labelled, partition, source_files)
        del labelled, valid, scores
        result = evaluation.finalize(job / "evaluation", reference["job"] / "evaluation")
        budget.check(1)
        if method.sources() != source_files:
            raise ValueError("Sources changed before completion")
        completion = {"status": result["status"], "physical_updates": 3456,
                      "label_parses": data.read_json(job / "access.json"), "budget": budget.state(),
                      "selection": result["selection"],
                      "evaluation": data.record(job / "evaluation/evaluation.json", job)}
        data.write_json(job / "completion.json", completion)
        return completion
    except Exception as error:
        data.write_json(job / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(error),
                                               "label_parses": data.read_json(job / "access.json"),
                                               "recovery": "After all 36 matrices exist, finalize uses saved metrics only"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("execute", "finalize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Research scripts run only on Linux py310")
    if args.action == "execute":
        if args.audit is None or args.authorization is None:
            parser.error("execute needs --audit and --authorization")
        result = execute(args.out.resolve(), args.audit.resolve(), args.authorization.resolve())
    else:
        p = method.contract()
        result = evaluation.finalize(args.out.resolve(), data.ROOT / p["baseline"]["linux_job"] / "evaluation")
    print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())


if __name__ == "__main__":
    main()
