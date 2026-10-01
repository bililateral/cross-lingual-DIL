"""Three sequential orders, shared first states, restricted replay and gated valid.

Only an explicit matching formal authorization and native evidence enable execute.
Preparing this file does not authorize Linux, supervision or training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable

import numpy as np

import step28_bge_continual as method
import step28_bge_continual_evaluate as evaluation

base, data, core = method.base, method.data, method.core
COMPLETE = "COMPLETE_6048_UPDATES_21_ENDPOINTS_VALID_BLIND"


def sources() -> list[dict]:
    paths = [
        "scripts/step28_bge_continual.py", "scripts/step28_bge_continual_run.py",
        "scripts/step28_bge_continual_evaluate.py", "scripts/step28_bge_continual_check.py",
        "tests/test_step28_bge_continual_contracts.py", "schema/step28_bge_continual_policy.json",
        "docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md", "scripts/run_step28_bge_continual_linux_20260930.sh",
        "scripts/step28_alias_ranking.py", "scripts/step28_alias_calibration.py",
        "scripts/step28_chinese_base.py", "schema/step28_alias_ranking_policy.json",
        "schema/step28_chinese_base_policy.json", "scripts/step28_continual_population.py",
        "scripts/step28_continual_population_data.py", "scripts/step28_continual_population_evaluate.py",
        "scripts/step28_continual_population_run.py", "scripts/step28_continual_expression_run.py",
    ]
    return [data.record(data.ROOT / path, data.ROOT) for path in sorted(paths)]


def array(root: Path, record: dict, shape: tuple, dtype: Any) -> np.ndarray:
    result = np.load(data.verify(root / record["path"], record), allow_pickle=False)
    if result.shape != shape or result.dtype != dtype or not np.isfinite(result).all():
        raise ValueError("Saved score/matrix schema differs")
    return result


def save_array(path: Path, values: np.ndarray, root: Path) -> dict:
    if path.exists():
        raise FileExistsError(path)
    np.save(path, values, allow_pickle=False)
    if not np.array_equal(np.load(path, allow_pickle=False), values):
        raise ValueError("Array roundtrip changed values")
    return data.record(path, root)


def rng_state() -> dict:
    import torch
    return {"cpu": torch.get_rng_state().tolist(),
            "cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    import torch
    torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG count differs")
        torch.cuda.set_rng_state_all([torch.tensor(row, dtype=torch.uint8) for row in state["cuda"]])
    if rng_state() != state:
        raise ValueError("RNG restore changed state")


def parse_once(job: Path, groups: list, c: dict, split: str,
               loader: Callable = base.public.attach_labels) -> list:
    if split not in ("train", "development"):
        raise ValueError("Forbidden supervision split")
    path = job / "access.json"
    access = data.read_json(path)
    key = "train" if split == "train" else "valid"
    if access[key] != 0:
        raise ValueError("This stage's supervision parse has already been attempted")
    access[key] = 1
    data.write_json(path, access)  # Attempt recorded before opening any label CSV.
    return loader(groups, c, split)


def save_memory(root: Path, name: str, memory: method.Memory) -> dict:
    path = root / "memory" / (name + ".json")
    if path.exists():
        raise FileExistsError(path)
    payload = memory.to_bytes()
    path.write_bytes(payload)
    restored = method.Memory.from_bytes(path.read_bytes())
    if restored.to_bytes() != payload:
        raise ValueError("Memory restoration changed payload")
    return {"file": data.record(path, root), **memory.summary(),
            "custody": "Linux-only training payload; contains formal texts and labels, exclude from review/small-result sync"}


def train_stage(model: Any, optimizer: Any, current: list, memory: method.Memory | None,
                c: dict, p: dict, order: str, stage: int, arm: str,
                root: Path, budget: Any) -> dict:
    import torch

    started = time.monotonic()
    sequence, stream = method.schedule(current, p, order, stage)
    if memory is not None:
        memory.begin_stage(stage)
    history_ids, updates, observations = [], [], {}
    torch.cuda.synchronize() if next(model.parameters()).is_cuda else None
    if next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()
    for index, group in enumerate(sequence):
        history, reference = memory.draw() if memory is not None else (None, None)
        if history is not None:
            history_ids.append(history.uid)
        record = method.update(
            model, optimizer, group, history, reference, c, arm, stage, index + 1,
            data.seed_for(stream, index, "dropout"),
            data.seed_for(p["memory_seed"], order, stage, index, "history_dropout"),
            observe=index + 1 in (1, 29, 30, 288), check=budget.check)
        updates.append([record[key] for key in method.STEP_COLUMNS])
        if record["modules"]:
            observations[str(index + 1)] = record["modules"]
        if (index + 1) % 24 == 0:
            print(data.json_bytes({"event": "updates", "order": order, "method": arm,
                                   "stage": stage, "stage_updates": index + 1,
                                   "logical_updates": method.adam_step(optimizer),
                                   **budget.state()}).decode(), flush=True)
    if memory is not None and memory.draw_count != 288:
        raise ValueError("Incomplete historical presentation count")
    name = order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"
    result = {"order": order, "method": "shared" if stage == 1 else arm, "stage": stage,
              "actual_domain": order[stage - 1], "seed": "s0", "updates": 288,
              "encoder_positive_lr_updates": 287, "adam_step": method.adam_step(optimizer),
              "current_ids": [g.uid for g in sequence], "history_ids": history_ids,
              "current_schedule_sha256": hashlib.sha256(data.json_bytes([g.uid for g in sequence])).hexdigest(),
              "current_dropout_stream": stream, "update_columns": list(method.STEP_COLUMNS),
              "scalar_timing_scope": "Host construction/enqueue time only; backward is joint with historical supervision, not separately isolated GPU compute",
              "observations": observations, "memory_after_training": memory.summary() if memory else None,
              "update_file": save_array(root / "updates" / (name + ".npy"),
                                        np.asarray(updates, dtype=np.float64), root)}
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
        result["cuda_allocator"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                     "scope": "This stage training interval; not whole-card or other-process use"}
    result["training_seconds"] = time.monotonic() - started
    data.write_json(root / "updates" / (name + ".json"), result)
    return result


def checkpoint(root: Path, name: str, model: Any, optimizer: Any, c: dict,
               order: str, stage: int, current_cal: list, valid: list,
               first_map: dict | None, budget: Any) -> dict:
    """Real full state restore and complete score replay at every endpoint."""
    started = time.monotonic()
    if len(current_cal) != 12 or len(valid) != 60 or method.adam_step(optimizer) != stage * 288:
        raise ValueError("Checkpoint is not a complete stage with current-only calibration")
    state = rng_state()
    metadata = {"name": name, "order": order, "stage": stage, "seed": "s0",
                "completed_updates": stage * 288, "policy_sha256": method.POLICY_SHA256,
                "rng": state, "config": c}
    scores = {"calibration": method.ranking.score(model, current_cal, c, budget.check),
              "development": method.ranking.score(model, valid, c, budget.check)}
    full_path = root / ("branches" if stage == 1 else "work") / (name + ".pt")
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(full_path, model, optimizer, metadata)
    if core.restore_state(full_path, model, optimizer, full["state_sha256"]) != metadata:
        raise ValueError("Full model/Adam metadata differs")
    restore_rng(state)
    for role, groups in (("calibration", current_cal), ("development", valid)):
        if not np.array_equal(scores[role], method.ranking.score(model, groups, c, budget.check)):
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
    mapping = method.calibration.fit(scores["calibration"], truth, role="calibration", check=budget.check)
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
    first = first_map if first_map is not None else mapping
    for role, use_map in (("stage-cal", mapping), ("first-cal", first)):
        values = method.calibration.transform(scores["development"], use_map)
        method.calibration.preserve_order(scores["development"], values)
        score_records[role] = save_array(root / "scores" / (name + "_" + role + ".npy"), values, root)
    # RNG is part of the shared branch state; checkpoint/evaluation cannot advance it.
    restore_rng(state)
    auxiliary = {"checkpoint_metadata": metadata,
                 "first_map": dict(zip(("a", "b"), method.calibration.parameters(first))),
                 "stage_map": dict(zip(("a", "b"), method.calibration.parameters(mapping)))}
    auxiliary_bytes = len(data.json_bytes(auxiliary))
    if auxiliary_bytes > 1048576:
        raise ValueError("Common phase/RNG/map metadata alone exceeds history allowance")
    result = {"name": name, "order": order, "stage": stage, "actual_domain": order[stage - 1],
              "completed_updates": stage * 288, "full_model_adam_and_rng_restore_verified": True,
              "model_state_sha256": model_digest,
              "model": {**inference, "path": path.relative_to(root).as_posix()},
              "full_checkpoint": {**full, "path": full_path.relative_to(root).as_posix()},
              "full_checkpoint_retained": stage == 1,
              "scores": score_records, "mapping": data.record(map_path, root),
              "learner_auxiliary": auxiliary, "learner_auxiliary_serialized_bytes": auxiliary_bytes,
              "first_map_parameters": dict(zip(("a", "b"), method.calibration.parameters(first)))}
    if stage != 1:
        # Verified intermediate file only; all inference endpoints remain active research evidence.
        base.persistence.remove_work_file(full_path, root)
        result["intermediate_deleted_bytes"] = full["bytes"]
    result["checkpoint_calibration_score_seconds"] = time.monotonic() - started
    data.write_json(root / "points" / (name + ".json"), result)
    return result


def restore_branch(root: Path, point: dict, c: dict) -> tuple[Any, Any]:
    model = base.load_model(c, "split_rank")
    optimizer = core.make_optimizer(model, c)
    record = point["full_checkpoint"]
    path = data.verify(root / record["path"], record)
    metadata = core.restore_state(path, model, optimizer, record["state_sha256"])
    if metadata["completed_updates"] != 288 or method.adam_step(optimizer) != 288:
        raise ValueError("Branch did not restore the shared full first stage")
    restore_rng(metadata["rng"])
    if core.state_digest(model.state_dict()) != point["model_state_sha256"]:
        raise ValueError("Shared branch model mismatch")
    return model, optimizer


def blind_gate(root: Path, manifest: dict, p: dict) -> dict:
    """Inspect actual complete weights and score files before any valid parsing."""
    if (manifest["status"] != COMPLETE or manifest["source_files"] != sources()
            or manifest["physical_updates"] != 6048 or manifest["gradient_group_presentations"] != 9504
            or set(manifest["points"]) != set(evaluation.expected_points())):
        raise ValueError("Incomplete or changed pilot before valid")
    initial = array(root, manifest["initial"]["scores"], (60, 378), np.float32)
    partition = data.read_json(data.verify(root / manifest["partition"]["path"], manifest["partition"]))
    fitting = {r["group_uid"]: r["domain"] for r in partition["fit"]}
    points = {}
    schedules, memories, first_maps = {}, {}, {}
    for name in evaluation.expected_points():
        ref = manifest["points"][name]
        point = data.read_json(data.verify(root / ref["path"], ref))
        stage = point["stage"]
        expected_stage = 1 if name.endswith("_shared") else int(name[-1])
        if (point["name"] != name or stage != expected_stage or point["order"] != name[:3]
                or point["actual_domain"] != name[stage - 1]
                or point["completed_updates"] != stage * 288
                or point["full_model_adam_and_rng_restore_verified"] is not True):
            raise ValueError("Stage identity/reload record differs")
        data.verify(root / point["model"]["path"], point["model"])
        if stage == 1:
            data.verify(root / point["full_checkpoint"]["path"], point["full_checkpoint"])
        raw = array(root, point["scores"]["development"], (60, 378), np.float32)
        array(root, point["scores"]["calibration"], (12, 378), np.float32)
        mapping = data.read_json(data.verify(root / point["mapping"]["path"], point["mapping"]))
        if (mapping["status"] != "PASS_CALIBRATION_FIT" or mapping["group_count"] != 12
                or mapping["pair_count"] != 4536 or mapping["positive_count"] != 240
                or mapping["actual_domain"] != point["actual_domain"]
                or mapping["model_state_sha256"] != point["model_state_sha256"]
                or mapping["score_source"] != point["scores"]["calibration"]
                or mapping["calibration_group_ids"] != [r["group_uid"] for r in partition["calibration"]
                                                       if r["domain"] == point["actual_domain"]]):
            raise ValueError("Current-domain calibration identity differs")
        if stage == 1:
            first_maps[point["order"]] = dict(zip(("a", "b"), method.calibration.parameters(mapping)))
        if point["first_map_parameters"] != first_maps[point["order"]]:
            raise ValueError("Fixed first-stage diagnostic map changed")
        restored = {"raw": raw}
        for role, use_map in (("stage-cal", mapping), ("first-cal", point["first_map_parameters"])):
            values = array(root, point["scores"][role], (60, 378), np.float64)
            if not np.array_equal(values, method.calibration.transform(raw, use_map)):
                raise ValueError("Saved map/score pairing differs")
            method.calibration.preserve_order(raw, values)
            restored[role] = values
        log_record = manifest["training"][name]
        log = data.read_json(data.verify(root / log_record["path"], log_record))
        values = array(root, log["update_file"], (288, len(method.STEP_COLUMNS)), np.float64)
        expected_arm = "shared" if stage == 1 else name.split("_")[1]
        if (log["updates"] != 288 or log["adam_step"] != stage * 288
                or log["order"] != point["order"] or log["stage"] != stage
                or log["method"] != expected_arm or log["actual_domain"] != point["actual_domain"]
                or log["update_columns"] != list(method.STEP_COLUMNS)
                or len(log["current_ids"]) != 288 or len(set(log["current_ids"])) != 48
                or set(log["current_ids"]) != {uid for uid, d in fitting.items() if d == point["actual_domain"]}
                or any(log["current_ids"].count(uid) != 6 for uid in set(log["current_ids"]))):
            raise ValueError("Current presentation budget differs")
        columns = dict(zip(method.STEP_COLUMNS, values.T))
        if (not np.array_equal(columns["encoder_lr"], [method.stage_lr(s) for s in range(1, 289)])
                or not np.all(columns["head_lr"] == .001)
                or not np.allclose(columns["current_total"], columns["current_bce"] + columns["current_rank"]
                                   + .5 * columns["current_hard"], atol=2e-6, rtol=1e-6)
                or not np.allclose(columns["history_total"], columns["history_bce"] + columns["history_rank"]
                                   + .5 * columns["history_hard"], atol=2e-6, rtol=1e-6)
                or not np.allclose(columns["total"], columns["current_total"] + columns["history_total"]
                                   + .5 * columns["logit_mse"], atol=1e-12, rtol=0)):
            raise ValueError("Learning-rate or loss decomposition differs")
        key = (point["order"], stage)
        current_tuple = tuple(log["current_ids"])
        if key in schedules and schedules[key] != current_tuple:
            raise ValueError("Current sequences are not paired")
        schedules[key] = current_tuple
        if log["method"] in ("er", "logit"):
            if len(log["history_ids"]) != 288 or set(log["history_ids"]) & set(log["current_ids"]):
                raise ValueError("Historical budget or separation differs")
            memory_key = (point["order"] + "_" + expected_arm + "_after1" if stage == 2
                          else point["order"] + "_" + expected_arm + "_stage2")
            memory_record = manifest["memories"][memory_key]
            payload = data.verify(root / memory_record["file"]["path"], memory_record["file"]).read_bytes()
            memory = method.Memory.from_bytes(payload)
            if (memory.first_map != first_maps[point["order"]]
                    or memory.with_logits != (expected_arm == "logit")
                    or memory.order != point["order"] or memory.seed != p["memory_seed"]
                    or any(fitting[g.uid] not in point["order"][:stage - 1] for g in memory.reservoir.groups)
                    or any(fitting[uid] != point["order"][origin - 1]
                           for uid, origin in memory.reference_origins.items())):
                raise ValueError("Historical origin or reference contract differs")
            memory.begin_stage(stage)
            replayed = [memory.draw()[0].uid for _ in range(288)]
            if replayed != log["history_ids"] or memory.summary() != log["memory_after_training"]:
                raise ValueError("Actual saved memory/draw state differs from training")
            if expected_arm == "er" and np.any(columns["logit_mse"] != 0):
                raise ValueError("ER included an unauthorized logit penalty")
            value = (log["memory_after_training"]["members"], log["history_ids"])
            if key in memories and memories[key] != value:
                raise ValueError("ER and LOGIT history members/draws differ")
            memories[key] = value
        elif log["history_ids"] or np.any(values[:, 4:9] != 0):
            raise ValueError("SEQ/shared first stage used history")
        points[name] = restored
    return {"initial": initial, "points": points}


def collect(root: Path, scores: dict, labelled: list, partition: dict, source_files: list) -> dict:
    """Write every matrix/count before bootstrap, comparison or acceptance."""
    root.mkdir()
    truth = np.asarray([group.labels for group in labelled], dtype=np.uint8)
    if [g.uid for g in labelled] != [r["group_uid"] for r in partition["development"]]:
        raise ValueError("Valid truth/score group order differs")
    result = {"status": "COLLECTING", "points": {}, "source_files": source_files,
              "group_ids": [g.uid for g in labelled],
              "domains": [r["domain"] for r in partition["development"]],
              "metric_columns": list(method.metrics.COLUMNS)}
    for name in ["initial", *evaluation.expected_points()]:
        entries = {"raw": scores["initial"]} if name == "initial" else scores["points"][name]
        stored = {}
        for role, values in entries.items():
            matrix, counts = method.metrics.group_metrics(truth, values)
            matrix_record = save_array(root / f"{name}_{role}.npy", matrix, root)
            count_path = root / f"{name}_{role}_counts.json"
            data.write_json(count_path, counts)
            stored[role] = {"matrix": matrix_record, "counts": data.record(count_path, root)}
        if name == "initial":
            result["initial"] = stored["raw"]
        else:
            result["points"][name] = stored
    result["status"] = "ALL_PILOT_MATRICES_SAVED_BEFORE_COMPARISONS"
    data.write_json(root / "collected.json", result)
    return result


def train(job: Path, p: dict, budget: Any) -> tuple[dict, dict, list]:
    import torch

    root = job / "run"
    root.mkdir()
    for name in ("models", "branches", "work", "scores", "maps", "points", "memory", "updates"):
        (root / name).mkdir()
    c = method.config(p)
    groups, metadata, checked = base.public.public_inputs(c)
    selected, partition = base.partition(groups, metadata, c)
    archive = core.model_files(base.model_config(c, "split_rank"))
    if any(archive[k] != c["models"]["split_rank"][k]
           for k in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Actual local pretrained BGE archive differs")
    data.write_json(root / "partition.json", partition)
    manifest = {"status": "RUNNING", "source_files": sources(), "points": {}, "training": {},
                "partition": data.record(root / "partition.json", root), "public_inputs": checked,
                "pretrained_archive": archive, "physical_updates": 0, "gradient_group_presentations": 0,
                "memories": {}, "initial": {}, "policy_sha256": method.POLICY_SHA256}
    data.write_json(root / "startup.json", manifest)
    model = base.load_model(c, "split_rank")
    initial_digest = core.state_digest(model.state_dict())
    # Public preflight checks lengths only; no truncation or feedback to training.
    maximum = 0
    for split in ("train", "development"):
        for group in groups[split]:
            budget.check()
            lengths = [len(row) for row in model.encoder.tokenizer(
                base.record_texts(group, "separate_moments"), padding=False, truncation=False)["input_ids"]]
            maximum = max(maximum, max(lengths))
            if maximum > c["input"]["token_budget"]:
                raise ValueError("Formal input exceeds frozen token budget")
    initial_scores = method.ranking.score(model, groups["development"], c, budget.check)
    manifest["initial"] = {"model_state_sha256": initial_digest, "maximum_tokens": maximum,
                           "scores": save_array(root / "scores" / "initial_raw.npy", initial_scores, root)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    groups["train"] = parse_once(job, groups["train"], c, "train")
    selected, labelled_partition = base.partition(groups, metadata, c)
    if labelled_partition != partition:
        raise ValueError("Partition changed after train alignment")
    supply = method.Supply(selected, partition)
    for order in method.ORDERS:
        model = base.load_model(c, "split_rank")
        if core.state_digest(model.state_dict()) != initial_digest:
            raise ValueError("Order initialization differs")
        optimizer = core.make_optimizer(model, c)
        shared_path = order + "_shared"
        current, current_cal = supply.current(shared_path, order, 1)
        train_stage(model, optimizer, current, None, c, p, order, 1, "seq", root, budget)
        shared = checkpoint(root, shared_path, model, optimizer, c, order, 1,
                            current_cal, groups["development"], None, budget)
        first_map = shared["first_map_parameters"]
        initial_memory = {}
        for arm in ("er", "logit"):
            memory = method.Memory(order, p["memory_seed"], arm == "logit", first_map)
            memory.auxiliary = shared["learner_auxiliary"]
            tick = time.monotonic()
            memory.retain(current, 1, lambda rows: method.ranking.score(model, rows, c, budget.check))
            retention_seconds = time.monotonic() - tick
            initial_memory[arm] = memory.to_bytes()
            manifest["memories"][order + "_" + arm + "_after1"] = save_memory(root, order + "_" + arm + "_after1", memory)
            manifest["memories"][order + "_" + arm + "_after1"]["retention_and_reference_seconds"] = retention_seconds
        if method.Memory.from_bytes(initial_memory["er"]).summary()["members"] != method.Memory.from_bytes(initial_memory["logit"]).summary()["members"]:
            raise ValueError("Shared first memory selection differs")
        del memory, current, current_cal, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        for arm in method.UPDATED:
            path = order + "_" + arm
            supply.branch_after_first(path, shared_path)
            model, optimizer = restore_branch(root, shared, c)
            memory = method.Memory.from_bytes(initial_memory[arm]) if arm != "seq" else None
            for stage in (2, 3):
                current, current_cal = supply.current(path, order, stage)
                train_stage(model, optimizer, current, memory, c, p, order, stage, arm, root, budget)
                name = f"{order}_{arm}_stage{stage}"
                point = checkpoint(root, name, model, optimizer, c, order, stage, current_cal,
                                   groups["development"], first_map, budget)
                if memory is not None:
                    memory.auxiliary = point["learner_auxiliary"]
                    data.write_json(root / "memory" / (name + "_budget.json"), memory.summary())
                if memory is not None and stage == 2:
                    tick = time.monotonic()
                    memory.retain(current, 2, lambda rows: method.ranking.score(model, rows, c, budget.check))
                    retention_seconds = time.monotonic() - tick
                    manifest["memories"][name] = save_memory(root, name, memory)
                    manifest["memories"][name]["retention_and_reference_seconds"] = retention_seconds
                del current, current_cal
            del model, optimizer, memory
            gc.collect()
            torch.cuda.empty_cache()
        for name in [n for n in evaluation.expected_points() if n.startswith(order)]:
            manifest["points"][name] = data.record(root / "points" / (name + ".json"), root)
            manifest["training"][name] = data.record(root / "updates" / (name + ".json"), root)
        manifest["physical_updates"] = len(manifest["points"]) * 288
        manifest["gradient_group_presentations"] = manifest["physical_updates"] + len(manifest["points"]) // 7 * 1152
        data.write_json(root / "progress.json", manifest)
    if sources() != manifest["source_files"]:
        raise ValueError("Scientific sources changed during formal run")
    manifest.update(status=COMPLETE, budget=budget.state())
    data.write_json(root / "manifest.json", manifest)
    return manifest, partition, groups["development"]


def execute(job: Path, audit_path: Path, authorization_path: Path) -> dict:
    import torch

    if platform.system() != "Linux":
        raise RuntimeError("Use existing Linux py310 only")
    p = method.contract()
    authorization = data.read_json(authorization_path)
    if (authorization.get("status") != "AUTHORIZED_BGE_CONTINUAL_TRAIN_VALID"
            or authorization.get("policy_sha256") != method.POLICY_SHA256
            or authorization.get("source_files") != sources()
            or authorization.get("job") != job.relative_to(data.ROOT).as_posix()
            or authorization.get("physical_updates") != 6048
            or authorization.get("train_parse_attempts") != 1 or authorization.get("valid_parse_attempts") != 1
            or authorization.get("test_access") is not False or authorization.get("owners_access") is not False
            or authorization.get("runtime") != p["runtime"]):
        raise ValueError("A separate matching formal-stage authorization is required")
    audit = data.read_json(audit_path)
    if (audit.get("status") != "PASS_BGE_CONTINUAL_HANDMADE_CPU"
            or audit.get("source_files") != sources() or audit.get("contracts", {}).get("failed") != 0
            or audit.get("contracts", {}).get("skipped") != 0
            or audit.get("formal_inputs") is not False or audit.get("formal_labels") is not False
            or set(audit.get("native", {})) != set(method.UPDATED)):
        raise ValueError("Current handmade CPU/native verification is required")
    for record in audit["native"].values():
        data.verify(audit_path.parent / record["path"], record)
    if not job.is_relative_to((data.ROOT / "reports").resolve()) or any((job / n).exists() for n in ("run", "evaluation", "access.json")):
        raise ValueError("Use a new reports job; no automatic resumption")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One eligible visible GPU required")
    available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
                     if row.startswith("MemAvailable:"))
    if (torch.cuda.mem_get_info()[0] < p["runtime"]["minimum_free_gpu_bytes"]
            or available < p["runtime"]["minimum_free_host_bytes"]
            or shutil.disk_usage(data.ROOT).free < p["runtime"]["maximum_output_bytes"]):
        raise RuntimeError("Insufficient shared resources; wait without affecting other users")
    job.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    budget = base.persistence.Budget(job, {"runtime": p["runtime"]})
    data.write_json(job / "access.json", {"train": 0, "valid": 0, "heldout": 0, "owners": 0})
    data.write_json(job / "execution.json", {"authorization": data.record(authorization_path, data.ROOT),
                                            "audit": data.record(audit_path, data.ROOT),
                                            "source_files": sources(), "python": platform.python_version(),
                                            "torch": torch.__version__, "cuda": torch.version.cuda,
                                            "gpu": torch.cuda.get_device_name(), "cpu_affinity": sorted(os.sched_getaffinity(0))})
    try:
        manifest, partition, valid_groups = train(job, p, budget)
        scores = blind_gate(job / "run", manifest, p)
        data.write_json(job / "before_valid.json", {"status": "PASS_COMPLETE_BLIND_GATE",
                                                    "manifest": data.record(job / "run/manifest.json", job),
                                                    "points": 21, "supervision": data.read_json(job / "access.json")})
        labelled = parse_once(job, valid_groups, method.config(p), "development")
        collect(job / "evaluation", scores, labelled, partition, manifest["source_files"])
        del labelled, scores, valid_groups
        result = evaluation.finalize(job / "evaluation")
        budget.check(1)
        if sources() != manifest["source_files"]:
            raise ValueError("Scientific sources changed during evaluation")
        completion = {"status": result["status"], "physical_updates": 6048,
                      "label_parses": data.read_json(job / "access.json"), "budget": budget.state(),
                      "evaluation": data.record(job / "evaluation/evaluation.json", job)}
        data.write_json(job / "completion.json", completion)
        return completion
    except Exception as error:
        data.write_json(job / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(error),
                                               "label_parses": data.read_json(job / "access.json"),
                                               "recovery": "After complete collection, finalize reads saved evidence only"})
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
            parser.error("execute requires --audit and --authorization")
        result = execute(args.out.resolve(), args.audit.resolve(), args.authorization.resolve())
    else:
        result = evaluation.finalize(args.out.resolve())
    print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())


if __name__ == "__main__":
    main()
