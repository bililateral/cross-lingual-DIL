"""Source-bound C/S pilot: 4,320 updates, complete blind gate, one valid opening."""
from __future__ import annotations

import argparse
import ast
import gc
import os
from pathlib import Path
import random
import shutil
import signal
import threading
import time
import traceback

import numpy as np
import psutil

import step28_record_replay as method
import step28_bge_continual_run as previous
import step28_bge_continual_evaluate as evaluation

data, base, core, parent = method.data, method.base, method.core, method.parent


def sources() -> list[dict]:
    paths = {data.ROOT / name for name in (
        "scripts/step28_record_replay.py", "scripts/step28_record_replay_run.py",
        "scripts/step28_record_replay_verify.py", "tests/test_step28_record_replay.py")}
    todo = list(paths)
    while todo:
        path = todo.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and name.startswith("step28_"):
                    child = data.ROOT / "scripts" / (name + ".py")
                    if child not in paths:
                        paths.add(child)
                        todo.append(child)
    paths.update(data.ROOT / "schema" / name for name in (
        "step28_record_replay_policy.json", "step28_bge_continual_policy.json",
        "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update(data.ROOT / name for name in (
        "docs/SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md",
        "scripts/run_step28_record_replay_linux_20261007.sh"))
    return [data.record(path, data.ROOT) for path in sorted(paths)]


def point_name(order: str, arm: str, stage: int) -> str:
    return order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"


def expected_points() -> list[str]:
    return [name for order in parent.ORDERS for name in
            [order+"_shared", *(point_name(order, arm, stage) for arm in ("C", "S") for stage in (2, 3))]]


class Budget:
    def __init__(self, root: Path):
        self.root, self.limits = root, method.policy()["runtime"]
        self.started = time.monotonic() - max(0., time.time()-float(os.environ.get("RECORD_REPLAY_STARTED_EPOCH", time.time())))
        self.lock = threading.RLock()
        self.peak_bytes = self.peak_rss = self.peak_reserved = 0
        self.last_disk = -float("inf")
        self.phase = "startup"

    def snapshot(self) -> dict:
        return {"elapsed_seconds": time.monotonic()-self.started, "peak_output_bytes": self.peak_bytes,
                "peak_rss_bytes": self.peak_rss, "peak_cuda_reserved_bytes": self.peak_reserved,
                "phase": self.phase, "limits": self.limits}

    def check(self, reserve: int = 0) -> None:
        import torch
        with self.lock:
            elapsed = time.monotonic()-self.started
            self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
            if torch.cuda.is_initialized():
                self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
            projected_bytes = self.peak_bytes
            if reserve or elapsed-self.last_disk >= 10:
                used = 0
                for path in [*self.root.rglob("*"), self.root.with_suffix(".console.txt"), self.root.with_suffix(".wrapper.txt")]:
                    try:
                        if path.is_file():
                            used += path.stat().st_size
                    except FileNotFoundError:
                        if path.parent != self.root/"run/work":
                            raise
                self.peak_bytes = max(self.peak_bytes, used)
                projected_bytes = max(self.peak_bytes, used+reserve)
                self.last_disk = elapsed
                if shutil.disk_usage(self.root).free < reserve:
                    raise RuntimeError("Insufficient disk")
                data.write_json(self.root/"resource.json", self.snapshot())
            if (elapsed >= self.limits["maximum_gpu_stage_seconds"]
                    or projected_bytes > self.limits["maximum_output_bytes"]
                    or self.peak_rss > self.limits["maximum_rss_bytes"]
                    or self.peak_reserved > self.limits["maximum_cuda_reserved_bytes"]):
                raise RuntimeError("Approved cumulative budget exceeded")


def train_stage(model, optimizer, current: list, memory, c: dict, arm: str,
                order: str, stage: int, root: Path, budget) -> dict:
    if parent.adam_step(optimizer) != (stage-1)*288:
        raise ValueError("Adam continuity at stage start differs")
    if memory is not None:
        memory.begin_stage(stage)
        before = memory.summary()
    else:
        before = None
    schedule, _ = parent.schedule(current, parent.contract(), order, stage)
    records, histories = [], []
    name = point_name(order, arm, stage)
    budget.phase = name
    started = time.monotonic()
    for step, group in enumerate(schedule, 1):
        history, target = memory.draw() if memory is not None else (None, None)
        row = method.update(model, optimizer, group, history, target, c, arm, order, stage, step, budget.check)
        row.update(step=step, current_uid=group.uid, history_uid=history.uid if history else None)
        records.append(row)
        if history:
            histories.append(history.uid)
        if step == 1 or step % 24 == 0:
            print(data.json_bytes({"event": "updates", "point": name, "completed": step,
                "adam_step": parent.adam_step(optimizer), "stage_elapsed_seconds": time.monotonic()-started,
                **budget.snapshot()}).decode(), flush=True)
    after = memory.summary() if memory is not None else None
    if memory is not None and any(before[k] != after[k] for k in ("members", "seen", "origins", "tables")):
        raise ValueError("Historical membership/source changed within stage")
    log = {"order": order, "arm": "shared" if stage == 1 else arm, "stage": stage,
           "updates": len(records), "adam_step": parent.adam_step(optimizer),
           "current_ids": [g.uid for g in schedule], "history_ids": histories,
           "memory_before": before, "memory_after": after, "rows": records,
           "training_seconds": time.monotonic()-started}
    path = root/"updates"/(name+".json")
    data.write_json(path, log)
    return data.record(path, root)


def checkpoint(root: Path, name: str, model, optimizer, memory, current: list,
               calibration: list, valid: list, c: dict, stage: int, budget) -> tuple[dict, method.Memory]:
    if len(calibration) != 12 or len(valid) != 60 or parent.adam_step(optimizer) != stage*288:
        raise ValueError("Incomplete stage endpoint")
    rng = previous.rng_state()
    scores = {"calibration": method.score(model, calibration, c, budget.check),
              "raw": method.score(model, valid, c, budget.check)}
    mapping = parent.calibration.fit(scores["calibration"], np.asarray([g.labels for g in calibration], np.uint8),
                                     role="calibration", check=budget.check)
    if mapping["status"] != "PASS_CALIBRATION_FIT":
        raise ValueError("Calibration failed; no alternative fit")
    current_map = dict(zip(("a", "b"), parent.calibration.parameters(mapping)))
    first_map = memory.maps.get("first", current_map)
    memory.maps = {"first": first_map, "stage": current_map, "torch_rng": rng,
                   "completed_stage": stage, "adam_step": stage*288}
    retention = memory.retain(current, stage, lambda g: method.reference(model, g, c, budget.check)) if stage < 3 else None
    previous.restore_rng(rng)
    mp = root/"memory"/(name+".bin")
    mp.write_bytes(memory.to_bytes())
    restored = method.Memory.from_bytes(mp.read_bytes())
    meta = {"name": name, "stage": stage, "rng": rng, "memory": data.record(mp, root),
            "policy_sha256": data.sha256(method.POLICY)}
    full_path = root/"work"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(full_path, model, optimizer, meta)
    if core.restore_state(full_path, model, optimizer, full["state_sha256"]) != meta:
        raise ValueError("Full checkpoint metadata differs")
    previous.restore_rng(restored.maps["torch_rng"])
    for role, groups in (("calibration", calibration), ("raw", valid)):
        if not np.array_equal(scores[role], method.score(model, groups, c, budget.check)):
            raise ValueError("Complete blind scores differ after full restoration")
    for role, m in (("stage-cal", current_map), ("first-cal", first_map)):
        scores[role] = parent.calibration.transform(scores["raw"], m)
        parent.calibration.preserve_order(scores["raw"], scores[role])
    mapping_path = root/"maps"/(name+".json")
    data.write_json(mapping_path, mapping)
    budget.check(base.persistence.checkpoint_reserve(model, None))
    inference_path = root/"models"/(name+".pt")
    inference = core.save_state(inference_path, model, None, meta)
    if core.restore_state(inference_path, model, None, inference["state_sha256"]) != meta:
        raise ValueError("Inference checkpoint differs")
    previous.restore_rng(rng)
    result = {"stage": stage, "order": memory.order, "adam_step": parent.adam_step(optimizer),
        "calibration_ids": [g.uid for g in calibration], "maps": {"first": first_map, "stage": current_map},
        "mapping": data.record(mapping_path, root), "memory": data.record(mp, root),
        "memory_summary": restored.summary(), "retention": retention, "full_restore_verified": True,
        "full_state": {**full, "path": full_path.relative_to(root).as_posix()},
        "model": {**inference, "path": inference_path.relative_to(root).as_posix()},
        "scores": {role: previous.save_array(root/"scores"/(name+"_"+role+".npy"), value, root)
                   for role, value in scores.items()}}
    budget.check(1)
    if stage != 1:
        base.persistence.remove_work_file(full_path, root)
    data.write_json(root/"points"/(name+".json"), result)
    return result, restored


def train(job: Path, budget) -> tuple[dict, dict, list]:
    import torch
    root = job/"run"
    root.mkdir()
    for name in ("models", "work", "memory", "points", "scores", "maps", "updates"):
        (root/name).mkdir()
    c = method.config()
    groups, metadata, checked = base.public.public_inputs(c)
    _, partition = base.partition(groups, metadata, c)
    archive = core.model_files(base.model_config(c, "split_rank"))
    if any(archive[k] != c["models"]["split_rank"][k] for k in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Pinned pretrained BGE differs")
    data.write_json(root/"partition.json", partition)
    manifest = {"status": "RUNNING", "source_files": sources(), "public_inputs": checked,
                "partition": data.record(root/"partition.json", root), "pretrained_archive": archive,
                "points": {}, "training": {}, "starts": {}, "physical_updates": 0,
                "gradient_group_presentations": 0}
    data.write_json(root/"startup.json", manifest)
    model = method.load_model(c)
    initial_digest = core.state_digest(model.state_dict())
    maximum = 0
    for split in ("train", "development"):
        for group in groups[split]:
            budget.check()
            lengths = [len(row) for row in model.encoder.tokenizer(base.record_texts(group, "separate_moments"),
                       padding=False, truncation=False)["input_ids"]]
            maximum = max(maximum, max(lengths))
            if maximum > 256:
                raise ValueError("Formal token budget exceeded")
    initial = method.score(model, groups["development"], c, budget.check)
    manifest["initial"] = {"model_state_sha256": initial_digest, "maximum_tokens": maximum,
        "scores": previous.save_array(root/"scores/initial_raw.npy", initial, root)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    groups["train"] = previous.parse_once(job, groups["train"], c, "train")
    selected, labelled_partition = base.partition(groups, metadata, c)
    if labelled_partition != partition:
        raise ValueError("Partition changed after supervision alignment")
    supply = parent.Supply(selected, partition)
    for order in parent.ORDERS:
        model = method.load_model(c)
        if core.state_digest(model.state_dict()) != initial_digest:
            raise ValueError("Order initialization differs")
        optimizer = core.make_optimizer(model, c)
        shared_name = order+"_shared"
        current, cal = supply.current(shared_name, order, 1)
        manifest["training"][shared_name] = train_stage(model, optimizer, current, None, c, "C", order, 1, root, budget)
        shared, memory = checkpoint(root, shared_name, model, optimizer, method.Memory(order), current, cal,
                                    groups["development"], c, 1, budget)
        manifest["points"][shared_name] = shared
        manifest["physical_updates"] += 288
        manifest["gradient_group_presentations"] += 288
        del model, optimizer, current, cal, memory
        gc.collect()
        torch.cuda.empty_cache()
        for arm in ("C", "S"):
            path = order+"_"+arm
            supply.branch_after_first(path, shared_name)
            model = method.load_model(c)
            optimizer = core.make_optimizer(model, c)
            full = shared["full_state"]
            restored_meta = core.restore_state(data.verify(root/full["path"], full), model, optimizer, full["state_sha256"])
            memory = method.Memory.from_bytes(data.verify(root/shared["memory"]["path"], shared["memory"]).read_bytes())
            previous.restore_rng(memory.maps["torch_rng"])
            if restored_meta["memory"] != shared["memory"] or parent.adam_step(optimizer) != 288:
                raise ValueError("Shared branch restoration differs")
            manifest["starts"][path] = {"full_state_sha256": full["state_sha256"], "memory": memory.summary(),
                                         "rng": previous.rng_state(), "adam_step": 288}
            for stage in (2, 3):
                current, cal = supply.current(path, order, stage)
                name = point_name(order, arm, stage)
                manifest["training"][name] = train_stage(model, optimizer, current, memory, c, arm, order, stage, root, budget)
                point, memory = checkpoint(root, name, model, optimizer, memory, current, cal,
                                           groups["development"], c, stage, budget)
                manifest["points"][name] = point
                manifest["physical_updates"] += 288
                manifest["gradient_group_presentations"] += 576
                data.write_json(root/"progress.json", manifest)
            del model, optimizer, current, cal, memory
            gc.collect()
            torch.cuda.empty_cache()
        base.persistence.remove_work_file(root/shared["full_state"]["path"], root)
    manifest["status"] = "ALL_4320_UPDATES_15_STATES_VALID_BLIND"
    data.write_json(root/"manifest.json", manifest)
    return manifest, partition, groups["development"]


def blind_gate(root: Path, manifest: dict) -> tuple[dict, dict]:
    if (manifest["status"] != "ALL_4320_UPDATES_15_STATES_VALID_BLIND" or manifest["source_files"] != sources()
            or manifest["physical_updates"] != 4320 or manifest["gradient_group_presentations"] != 7776
            or set(manifest["points"]) != set(expected_points()) or set(manifest["training"]) != set(expected_points())):
        raise ValueError("Incomplete source-bound experiment")
    partition = data.read_json(data.verify(root/manifest["partition"]["path"], manifest["partition"]))
    fit = {r["group_uid"]: r["domain"] for r in partition["fit"]}
    scores = {"initial": previous.array(root, manifest["initial"]["scores"], (60, 378), np.float32), "points": {}}
    sequences = {}
    for order in parent.ORDERS:
        if manifest["starts"][order+"_C"] != manifest["starts"][order+"_S"]:
            raise ValueError("C/S full first-stage states differ")
    for name in expected_points():
        point = manifest["points"][name]
        order, stage = point["order"], point["stage"]
        arm = "C" if stage == 1 else name.split("_")[1]
        log = data.read_json(data.verify(root/manifest["training"][name]["path"], manifest["training"][name]))
        ids = sorted(uid for uid, domain in fit.items() if domain == order[stage-1])
        rng = random.Random(data.seed_for(20260918, order, stage, "current"))
        expected_ids = []
        for _ in range(6):
            epoch = ids.copy()
            rng.shuffle(epoch)
            expected_ids.extend(epoch)
        if (log["updates"] != 288 or log["adam_step"] != 288*stage or log["current_ids"] != expected_ids
                or len(log["rows"]) != 288 or not point["full_restore_verified"] or point["adam_step"] != 288*stage):
            raise ValueError("Stage updates, schedule or restoration differs")
        saved_memory = method.Memory.from_bytes(data.verify(root/point["memory"]["path"], point["memory"]).read_bytes())
        if saved_memory.summary() != point["memory_summary"]:
            raise ValueError("Saved memory summary differs")
        # Independently reconstruct Algorithm R membership from arrived fit IDs only.
        retention_rng = random.Random(data.seed_for(20260930, order, "retention"))
        retained, seen = [], 0
        for arrival in order[:min(stage, 2)]:
            for uid in sorted(uid for uid, d in fit.items() if d == arrival):
                seen += 1
                index = len(retained) if len(retained) < 6 else retention_rng.randrange(seen)
                if index < 6:
                    if index == len(retained):
                        retained.append(uid)
                    else:
                        retained[index] = uid
        if retained != point["memory_summary"]["members"] or seen != point["memory_summary"]["seen"]:
            raise ValueError("Reservoir membership differs from independent Algorithm R")
        if any(fit[uid] != order[birth-1] for uid, birth in saved_memory.origins.items()):
            raise ValueError("Source birth domain differs")
        if stage > 1:
            prior_name = point_name(order, arm, stage-1)
            prior_record = manifest["points"][prior_name]["memory"]
            mem = method.Memory.from_bytes(data.verify(root/prior_record["path"], prior_record).read_bytes())
            mem.begin_stage(stage)
            if mem.summary() != log["memory_before"]:
                raise ValueError("History did not begin from prior bounded state")
            drawn = [mem.draw()[0].uid for _ in range(288)]
            if drawn != log["history_ids"] or mem.summary() != log["memory_after"]:
                raise ValueError("History draw replay differs")
            for uid in mem.tables.keys() & saved_memory.tables.keys():
                if not np.array_equal(mem.tables[uid], saved_memory.tables[uid]):
                    raise ValueError("Surviving teacher table changed at boundary")
            key = (order, stage)
            pair = (log["current_ids"], drawn, log["memory_before"]["members"])
            if key in sequences and sequences[key] != pair:
                raise ValueError("C/S sampling unpaired")
            sequences[key] = pair
        elif log["history_ids"] or log["memory_before"] is not None:
            raise ValueError("First stage used history")
        for step, row in enumerate(log["rows"], 1):
            if row["step"] != step or row["encoder_lr"] != parent.stage_lr(step) or row["head_lr"] != .001:
                raise ValueError("Learning rate or local step differs")
            total = row["current"]["supervised"]
            if stage > 1:
                h = row["history"]
                d = (h["mse0"]+h["mse1"])/2 if arm == "C" else h["mse0"]
                if not np.isclose(h["distillation"], d, rtol=2e-6, atol=2e-6):
                    raise ValueError("C/S teacher weighting differs")
                total += .1*h["supervised"]+.5*d
            if not np.isfinite(row["total"]) or not np.isclose(total, row["total"], rtol=2e-6, atol=2e-6):
                raise ValueError("Objective decomposition differs")
        cal_ids = [r["group_uid"] for r in partition["calibration"] if r["domain"] == order[stage-1]]
        if point["calibration_ids"] != cal_ids:
            raise ValueError("Calibration used wrong domain/groups")
        data.verify(root/point["model"]["path"], point["model"])
        mapping = data.read_json(data.verify(root/point["mapping"]["path"], point["mapping"]))
        if mapping["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Calibration did not pass")
        previous.array(root, point["scores"]["calibration"], (12, 378), np.float32)
        roles = {role: previous.array(root, point["scores"][role], (60, 378), np.float32 if role == "raw" else np.float64)
                 for role in parent.ROLES}
        for role, key in (("stage-cal", "stage"), ("first-cal", "first")):
            if not np.array_equal(roles[role], parent.calibration.transform(roles["raw"], point["maps"][key])):
                raise ValueError("Saved output role differs")
        scores["points"][name] = roles
    return scores, partition


def collect(root: Path, scores: dict, labelled: list, partition: dict) -> dict:
    root.mkdir()
    if [g.uid for g in labelled] != [r["group_uid"] for r in partition["development"]]:
        raise ValueError("Valid identity alignment differs")
    truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    result = {"status": "COLLECTING", "source_files": sources(), "metric_columns": list(parent.metrics.COLUMNS),
              "group_ids": [g.uid for g in labelled], "domains": [r["domain"] for r in partition["development"]], "points": {}}
    for name in ["initial", *expected_points()]:
        entries = {"raw": scores["initial"]} if name == "initial" else scores["points"][name]
        result["points"][name] = {}
        for role, values in entries.items():
            matrix, counts = parent.metrics.group_metrics(truth, values)
            cp = root/(name+"_"+role+"_counts.json")
            data.write_json(cp, counts)
            result["points"][name][role] = {"matrix": previous.save_array(root/(name+"_"+role+".npy"), matrix, root),
                                           "counts": data.record(cp, root)}
    result["status"] = "ALL_46_METRIC_COUNT_SETS_SAVED"
    data.write_json(root/"collected.json", result)
    return result


def read_roles(root: Path, roles: dict) -> dict:
    values = {}
    for role in parent.ROLES:
        values[role] = evaluation.read_matrix(root, roles[role]["matrix"])
        evaluation.read_counts(root, roles[role]["counts"])
    for role in parent.ROLES[1:]:
        if not np.array_equal(values["raw"][:, evaluation.RANK_COLUMNS], values[role][:, evaluation.RANK_COLUMNS]):
            raise ValueError("Positive affine map altered ranking metrics")
    return values


def fields(arrays: dict, domains: list, role: str, endpoint: str) -> np.ndarray:
    if endpoint != "A2":
        return evaluation.endpoint_fields(arrays, domains, "er", role, endpoint)
    result = np.zeros((3, 3, 20, 22), np.float64)
    rows = evaluation.domain_rows(domains)
    for i, order in enumerate(parent.ORDERS):
        record = arrays[evaluation.point_name(order, "er", 2)]
        values = record["stage-cal"].copy() if role == "primary" else record[role]
        if role == "primary":
            values[:, evaluation.RANK_COLUMNS] = record["raw"][:, evaluation.RANK_COLUMNS]
        d = "ABC".index(order[1])
        result[i, d] = values[rows[d]]
    return result


def finalize(root: Path) -> dict:
    p = method.policy()
    col = data.read_json(root/"collected.json")
    if (col["status"] != "ALL_46_METRIC_COUNT_SETS_SAVED" or col["source_files"] != sources()
            or set(col["points"]) != {"initial", *expected_points()} or col["metric_columns"] != list(parent.metrics.COLUMNS)):
        raise ValueError("Incomplete saved collection")
    baseline = data.ROOT/p["reference"]["linux_root"]
    refs = {name: data.read_json(data.verify(baseline/name, rec)) for name, rec in p["reference"]["collections"].items()}
    for ref in refs.values():
        if any(ref[k] != col[k] for k in ("group_ids", "domains", "metric_columns")):
            raise ValueError("Baseline metric pairing differs")
    arrays = {arm: {} for arm in ("C", "S", "LOGIT0.1")}
    for order in parent.ORDERS:
        for stage in (1, 2, 3):
            logical = evaluation.point_name(order, "er", stage)
            for arm in ("C", "S"):
                arrays[arm][logical] = read_roles(root, col["points"][point_name(order, arm, stage)])
            ref_root, collection, key = ((baseline/"reference", refs["reference/collected.json"], order+"_shared")
                if stage == 1 else (baseline, refs["collected.json"], f"{order}_logit_tenth_stage{stage}"))
            arrays["LOGIT0.1"][logical] = read_roles(ref_root, collection["points"][key])
    draws = evaluation.bootstrap_draws()
    endpoints = (*evaluation.ENDPOINTS, "A2")
    fs = {arm: {role: {ep: fields(a, col["domains"], role, ep) for ep in endpoints}
                for role in (*parent.ROLES, "primary")} for arm, a in arrays.items()}
    result = {"endpoints": {arm: {role: {ep: evaluation.summarize_field(v, draws) for ep, v in eps.items()}
                                  for role, eps in roles.items()} for arm, roles in fs.items()}, "comparisons": {}}
    for other in ("S", "LOGIT0.1"):
        deltas = {role: {ep: evaluation.summarize_field(fs["C"][role][ep]-fs[other][role][ep], draws)
                         for ep in endpoints} for role in (*parent.ROLES, "primary")}
        delta = deltas["primary"]
        lower = lambda ep, metric: delta[ep][metric]["conditional_95pct_interval"][0]
        checks = {"O_MAP_lower_positive": lower("O", "map") > 0,
                  "final_all_MAP_mean_positive": delta["final_all"]["map"]["mean"] > 0,
                  "A2_MAP_lower_ge_minus_point01": lower("A2", "map") >= -.01,
                  "Z_MAP_lower_ge_minus_point01": lower("Z", "map") >= -.01,
                  "O_AP_lower_ge_minus_point01": lower("O", "average_precision") >= -.01}
        result["comparisons"]["C-"+other] = {"delta": deltas, "checks": checks, "pass": all(checks.values())}
    result["development_criteria_pass"] = all(r["pass"] for r in result["comparisons"].values())
    result["absolute_stage_results"] = {}
    rows = evaluation.domain_rows(col["domains"])
    for name, roles in col["points"].items():
        result["absolute_stage_results"][name] = {}
        for role, rec in roles.items():
            matrix = evaluation.read_matrix(root, rec["matrix"])
            counts = evaluation.read_counts(root, rec["counts"])
            result["absolute_stage_results"][name][role] = {
                "macro_all": dict(zip(parent.metrics.COLUMNS, matrix.mean(0).tolist())),
                "macro_by_domain": {d: dict(zip(parent.metrics.COLUMNS, matrix[r].mean(0).tolist())) for d, r in zip("ABC", rows)},
                "fixed_half_classification": base.fixed_classification(counts, col["domains"])}
    result.update(status="STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION", source_files=sources(),
                  reference=p["reference"], collected=data.record(root/"collected.json", root), automatic_followup=False)
    evaluation.write_once(root/"evaluation.json", data.json_bytes(result))
    return result


def validate_gate(job: Path, gate_path: Path) -> dict:
    p = method.policy()
    gate = data.read_json(gate_path)
    if (gate.get("status") != "APPROVED_RECORD_REPLAY_PILOT" or gate.get("source_files") != sources()
            or gate.get("job") != job.relative_to(data.ROOT).as_posix() or gate.get("runtime") != p["runtime"]
            or gate.get("supervision") != p["supervision"] or gate.get("review_disposition") != "NO_OPEN_BLOCKERS"):
        raise ValueError("Source-bound reviewed execution gate differs")
    for mode in ("cpu", "gpu"):
        rec = gate[mode]
        evidence = data.read_json(data.verify(data.ROOT/rec["path"], rec))
        if evidence.get("mode") != mode or evidence.get("status") != "PASS_HANDWRITTEN_ONLY" or evidence.get("source_files") != sources():
            raise ValueError("Required matching verification evidence missing")
        if mode == "gpu":
            native = evidence.get("native", {})
            if native.get("kind") != "native_record_replay_first_optimizer_step":
                raise ValueError("Native actual first update evidence missing")
            estimate = native.get("shape_upper_projection_seconds", float("inf"))
            if (not np.isfinite(estimate) or estimate <= 0
                    or estimate > p["runtime"]["maximum_gpu_stage_seconds"]
                    or native.get("within_formal_24h_projection") is not True):
                raise ValueError("Native update passed but formal time projection is not admissible")
    return p


def execute(job: Path, gate_path: Path) -> dict:
    import torch
    from step28_record_replay_verify import preflight
    p = validate_gate(job, gate_path)
    if job.exists() or not job.is_relative_to(data.ROOT/"reports"):
        raise ValueError("New independent output required")
    resources = preflight()
    if shutil.disk_usage(data.ROOT).free < p["runtime"]["maximum_output_bytes"]:
        raise ValueError("Insufficient free output space")
    for name, rec in p["reference"]["collections"].items():
        data.verify(data.ROOT/p["reference"]["linux_root"]/name, rec)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    job.mkdir(parents=True)
    budget = Budget(job)
    data.write_json(job/"access.json", dict(train=0, valid=0, heldout=0, owners=0))
    data.write_json(job/"execution.json", {"gate": data.record(gate_path, data.ROOT), "sources": sources(),
                                        "resources": resources, "numerics": numerics})
    stopped = threading.Event()
    def watchdog():
        while not stopped.wait(1):
            try:
                budget.check()
            except BaseException as exc:
                try:
                    data.write_json(job/"failure.json", {"status": "BUDGET_STOP_NO_RETRY", "error": str(exc), "budget": budget.snapshot()})
                finally:
                    os._exit(2)
    def terminate(signum, frame):
        raise RuntimeError("TERM: preserve failed run; no automatic retry")
    signal.signal(signal.SIGTERM, terminate)
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    try:
        manifest, partition, valid = train(job, budget)
        budget.phase = "blind_gate"
        scores, verified_partition = blind_gate(job/"run", manifest)
        if verified_partition != partition:
            raise ValueError("Partition changed")
        data.write_json(job/"before_valid.json", {"status": "PASS_COMPLETE_BLIND_GATE", "points": 15,
            "manifest": data.record(job/"run/manifest.json", job), "access": data.read_json(job/"access.json")})
        budget.check()
        budget.phase = "collect"
        labelled = previous.parse_once(job, valid, method.config(), "development")
        collect(job/"evaluation", scores, labelled, partition)
        del labelled, valid, scores
        budget.phase = "statistics"
        result = finalize(job/"evaluation")
        budget.check(16384)
        access = data.read_json(job/"access.json")
        if access != dict(train=1, valid=1, heldout=0, owners=0):
            raise ValueError("Completion access ledger differs")
        completion = {"status": "COMPLETE_RECORD_REPLAY_DEVELOPMENT", "physical_updates": 4320,
                      "access": access, "budget": budget.snapshot(), "development_criteria_pass": result["development_criteria_pass"],
                      "evaluation": data.record(job/"evaluation/evaluation.json", job)}
        with budget.lock:
            data.write_json(job/"completion.json", completion)
            try:
                budget.check(1)
            except BaseException:
                (job/"completion.json").unlink(missing_ok=True)
                raise
        return completion
    except BaseException:
        data.write_json(job/"failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "traceback": traceback.format_exc(), "budget": budget.snapshot()})
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Existing Linux py310 only")
    execute(args.out.resolve(), args.gate.resolve())
