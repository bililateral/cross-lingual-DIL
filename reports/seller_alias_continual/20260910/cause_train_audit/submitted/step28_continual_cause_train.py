"""Paired training intervention, exact original replay, gated Windows valid audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np

import step28_continual_cause_probe as probe
import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics
import step28_continual_population_run as previous

POLICY = data.ROOT / "schema/step28_continual_cause_train_policy.json"
COMPLETE = "COMPLETE_PAIRED_TRAINING_ORIGINAL_EXACT_VALID_LABELS_UNREAD"
BRANCHES = ("original", "reassigned")
POINTS = ("shared", "original_stage2", "original_stage3", "reassigned_stage2", "reassigned_stage3")


def sources() -> list[dict]:
    paths = [Path(__file__), POLICY, probe.CATALOG,
             data.ROOT / "tests/test_step28_continual_cause_train_contracts.py"]
    paths += [Path(m.__file__) for m in (probe, core, data, metrics, previous)]
    paths.append(data.POLICY_PATH)
    return [data.record(p, data.ROOT) for p in paths]


def contract() -> dict:
    config = data.read_json(POLICY)
    if (config["orders"] != list(probe.ORDERS) or config["branches"] != list(BRANCHES)
            or config["epochs_per_stage"] != 3 or config["physical_updates"] != 2700
            or config["heldout_access"] or config["owners_access"]
            or config["intervention_seed"] != 20260910):
        raise ValueError("Training-side contract differs")
    return config


def public_inputs(original: dict) -> tuple[dict[str, list[data.Group]], list[dict], dict]:
    """Read public train/valid only; never instantiate the old all-split Archive."""
    config = original["config"]
    root = data.ROOT / config["data_root"]
    for file, key in (("manifest.json", "data_manifest_sha256"), ("validation.json", "data_validation_sha256")):
        if data.sha256(root / file) != config[key]:
            raise ValueError("Pinned data qualification differs")
    dm = data.read_json(root / "manifest.json")
    if (root / "sources/step28_continual_catalog.json").read_bytes() != probe.CATALOG.read_bytes():
        raise ValueError("Generation catalog differs")
    checked = {}

    def path(relative: str) -> Path:
        checked[relative] = dm["files"][relative]
        return data.verify(root / relative, checked[relative])

    with path("groups.csv").open(encoding="utf-8", newline="") as stream:
        meta = sorted((r for r in csv.DictReader(stream) if r["split"] in ("train", "development")),
                      key=lambda r: (r["domain"], int(r["group_index"])))
    groups = {}
    seen_sellers, seen_items = set(), set()
    for split, count in (("train", 60), ("development", 20)):
        rows = [r for r in meta if r["split"] == split]
        if len({r["group_uid"] for r in rows}) != 3 * count or any(
                sum(r["domain"] == d for r in rows) != count for d in "ABC"):
            raise ValueError("Public domain/group partition differs")
        collected: dict = defaultdict(lambda: defaultdict(list))
        with path(f"{split}/items.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if set(r) != {"group_uid", "seller_uid", "item_uid", "title", "description"} or r["item_uid"] in seen_items:
                    raise ValueError("Public fields or item boundary differ")
                seen_items.add(r["item_uid"])
                collected[r["group_uid"]][r["seller_uid"]].append((r["item_uid"], r["title"], r["description"]))
        if set(collected) != {r["group_uid"] for r in rows}:
            raise ValueError("Public group identity differs")
        groups[split] = []
        for row in rows:
            sellers = collected[row["group_uid"]]
            ids = tuple(sorted(sellers))
            if seen_sellers.intersection(ids):
                raise ValueError("Seller reused across groups/splits")
            seen_sellers.update(ids)
            g = data.Group(row["group_uid"], ids, tuple(tuple(sorted(sellers[s])) for s in ids))
            g.validate()
            if int(row["items"]) != sum(map(len, g.items)) or int(row["accounts"]) != 28:
                raise ValueError("Public record count differs")
            groups[split].append(g)
    if [g.uid for g in groups["development"]] != original["evaluation_group_ids"]["development"]:
        raise ValueError("Original valid row alignment differs")
    return groups, meta, checked


def attach_labels(groups: list[data.Group], root: Path, split: str) -> list[data.Group]:
    """One CSV parse per authorized stage; caller writes access receipt first."""
    if split not in ("train", "development"):
        raise ValueError("Unapproved supervision split")
    dm = data.read_json(root / "manifest.json")
    relative = f"{split}/supervision/pairs.csv"
    path = data.verify(root / relative, dm["files"][relative])
    rows = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[row["group_uid"]].append(row)
    if set(rows) != {g.uid for g in groups}:
        raise ValueError("Supervision group identities differ; do not retry")
    result = []
    for g in groups:
        labels = data.align_labels(g, rows[g.uid])
        if sum(labels) != 20:
            raise ValueError("Positive pair count differs")
        result.append(data.Group(g.uid, g.sellers, g.items, labels))
    return result


def reassign(group: data.Group, catalog: dict) -> tuple[data.Group, dict]:
    """Cycle ORIGINAL style tuples, retaining the recipient's labels and content."""
    group.validate()
    parsed = [[probe.decode(t, d, catalog) for _, t, d in rows] for rows in group.items]
    styles = [rows[0][0] for rows in parsed]
    if any(any(s != styles[i] for s, _ in rows) for i, rows in enumerate(parsed)):
        raise ValueError("Account has inconsistent style")
    offset = data.seed_for(20260910, group.uid, "style_cycle") % 27 + 1
    assigned = [styles[(i + offset) % 28] for i in range(28)]
    if Counter(styles) != Counter(assigned):
        raise ValueError("Style multiset changed")
    items = []
    preserved = ("product", "qualifier", "intro", "detail", "packaging", "delivery", "service_position")
    for i, records in enumerate(group.items):
        changed = []
        for (uid, title, desc), (style, fields) in zip(records, parsed[i]):
            if probe.render(style, fields, catalog) != (title, desc):
                raise ValueError("Original roundtrip differs")
            t, d = probe.render(assigned[i], fields, catalog)
            actual_style, actual_fields = probe.decode(t, d, catalog)
            if actual_style != assigned[i] or any(actual_fields[k] != fields[k] for k in preserved):
                raise ValueError("Retained content changed")
            changed.append((uid, t, d))
        items.append(tuple(changed))
    result = data.Group(group.uid, group.sellers, tuple(items), group.labels)
    result.validate()
    return result, {"group_uid": group.uid, "offset": offset}


def public_digest(groups: list[data.Group]) -> str:
    return hashlib.sha256(data.json_bytes([data.Group(g.uid, g.sellers, g.items).payload() for g in groups])).hexdigest()


def manipulation(original: list[data.Group], changed: list[data.Group], catalog: dict) -> dict:
    left, right = np.triu_indices(28, 1)
    tables = {}
    for name, groups in (("original", original), ("reassigned", changed)):
        rows = []
        for g, base in zip(groups, original):
            if g.uid != base.uid or g.labels != base.labels or g.labels is None:
                raise ValueError("Intervention supervision moved")
            styles = np.asarray([probe.decode(items[0][1], items[0][2], catalog)[0] for items in g.items])
            codes = (styles[left] == styles[right]).astype(int) @ np.asarray([4, 2, 1])
            counts = np.zeros((2, 8), dtype=np.int64)
            np.add.at(counts, (np.asarray(g.labels), codes), 1)
            if counts.sum(1).tolist() != [358, 20]:
                raise ValueError("Class totals differ")
            rows.append(counts)
        tables[name] = np.stack(rows)
    if not np.array_equal(tables["original"].sum(1), tables["reassigned"].sum(1)):
        raise ValueError("Full-graph pattern marginal changed")
    return {"group_uids": [g.uid for g in original], "patterns": [f"{i:03b}" for i in range(8)],
            "tables": {k: v.tolist() for k, v in tables.items()},
            "interpretation": "Check actual association weakening; never redraw; residual association is possible."}


def train_stage(model: Any, optimizer: Any, groups: list[data.Group], config: dict,
                order: str, stage: int, branch: str, budget: previous.Budget) -> dict:
    """Same update/schedule as original, plus first-update module checks in-place."""
    import torch
    started = time.monotonic()
    stream = data.seed_for(config["initialization_seed"], order, stage, "current")
    rows = data.schedule(groups, config["epochs_per_stage"], stream)
    losses, norms, module_checks = [], [], {}
    for i, group in enumerate(rows):
        if i == 0:
            before = {k: core.state_digest(m.state_dict()) for k, m in (("encoder", model.encoder), ("head", model.head))}
        log = core.update(model, optimizer, group, None, config,
                          data.seed_for(stream, i, "dropout_current"),
                          data.seed_for(stream, i, "dropout_replay"), budget.check)
        losses.append(log["current_bce"])
        norms.append(log["gradient_norm_before_clip"])
        if i == 0:
            for k, m in (("encoder", model.encoder), ("head", model.head)):
                gradients = [float(p.grad.detach().float().norm()) for p in m.parameters() if p.grad is not None]
                changed = before[k] != core.state_digest(m.state_dict())
                if not gradients or not all(np.isfinite(gradients)) or max(gradients) <= 0 or not changed:
                    raise ValueError(f"Actual first update failed {k} gradient/parameter check")
                module_checks[k] = {"finite_nonzero_gradient": True, "parameters_changed": changed}
            expected_step = (stage - 1) * 180 + 1
            if not optimizer.state or any(float(v["step"]) != expected_step for v in optimizer.state.values()):
                raise ValueError("Adam history did not continue from the correct stage")
        if (i + 1) % 30 == 0:
            print(data.json_bytes({"event": "updates", "order": order, "stage": stage, "branch": branch,
                                  "completed": i + 1, "total": len(rows), **budget.state()}).decode(), flush=True)
    return {"updates": len(rows), "training_seconds": time.monotonic() - started,
            "current_group_ids": [g.uid for g in rows],
            "current_order_sha256": hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest(),
            "current_dropout_stream": stream, "replay_group_ids": [],
            "current_pair_presentations": len(rows) * 378, "replay_pair_presentations": 0,
            "current_bce_mean": float(np.mean(losses)), "replay_bce_mean": None,
            "maximum_gradient_norm_before_clip": max(norms), "first_update_modules": module_checks}


def same_original_point(point: dict, old: dict, out: Path) -> None:
    if (point["model_state_sha256"] != old["model_state_sha256"] or
            point["checkpoint"]["state_sha256"] != old["checkpoint"]["state_sha256"]):
        raise ValueError("Original model/Adam exact replay failed")
    a, b = point["scores"]["development"], old["scores"]["development"]
    if not np.array_equal(np.load(data.verify(out / a["path"], a), allow_pickle=False),
                          np.load(data.verify(probe.RUN / b["path"], b), allow_pickle=False)):
        raise ValueError("Original full scores exact replay failed")


def validate_training(log: dict, meta: list[dict], config: dict, old: dict) -> None:
    order = log["order"]
    for name, stage, branch in [("shared", 1, "shared")] + [
            (f"{b}_stage{s}", s, b) for b in BRANCHES for s in (2, 3)]:
        r = log["training"][name]
        ids = [x["group_uid"] for x in meta if x["split"] == "train" and x["domain"] == order[stage - 1]]
        # schedule only uses uid; labels and text have no role in sampling.
        stream = data.seed_for(config["initialization_seed"], order, stage, "current")
        dummy = [data.Group(uid, (), ()) for uid in ids]
        expected_ids = [g.uid for g in data.schedule(dummy, 3, stream)]
        if (r["updates"] != 180 or r["current_group_ids"] != expected_ids
                or r["current_dropout_stream"] != stream or r["replay_group_ids"] != []
                or r["current_pair_presentations"] != 68040 or r["replay_pair_presentations"] != 0
                or r["current_order_sha256"] != hashlib.sha256(data.json_bytes(expected_ids)).hexdigest()
                or r["first_update_modules"] != {k: {"finite_nonzero_gradient": True, "parameters_changed": True} for k in ("encoder", "head")}):
            raise ValueError("Unpaired/incorrect training schedule or update semantics")
        if branch in ("shared", "original"):
            prior = old["training"]["shared"] if stage == 1 else old["training"]["sequential"][stage - 2]
            for key in ("updates", "current_group_ids", "current_order_sha256", "current_dropout_stream", "current_pair_presentations"):
                if r[key] != prior[key]:
                    raise ValueError("Original training presentations differ")


def run(out: Path) -> dict:
    import torch
    config, original = contract(), probe.original_manifest()
    runtime, base = config["runtime"], original["config"]
    if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires resumed Linux stage and one idle visible GPU")
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < runtime["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient eligible idle GPU; wait")
    available = next(int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    if available < runtime["minimum_free_host_bytes"] or shutil.disk_usage(data.ROOT).free < runtime["maximum_output_bytes"]:
        raise RuntimeError("Insufficient host/disk reserve; wait")
    if out.exists() or not out.is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a new reports directory")
    out.mkdir(parents=True)
    for name in ("work", "scores", "models"):
        (out / name).mkdir()
    budget = previous.Budget(out, config)
    snapshot = sources()
    result = {"status": "RUNNING", "contract": config, "original_manifest_sha256": probe.RUN_SHA,
              "source_files": snapshot, "orders": [],
              "label_parses": {"train": 0, "development": 0, "heldout": 0, "owners": 0},
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(out / "startup.json", result)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        files = core.model_files(base)
        if any(files[k] != base["model"][k] for k in ("file_count", "total_size_bytes", "content_sha256")):
            raise ValueError("Pretrained model bytes differ")
        groups, meta, checked = public_inputs(original)
        root = data.ROOT / base["data_root"]
        data.write_json(out / "access.json", {"train_parse_attempts": 1, "development": 0, "heldout": 0, "owners": 0})
        result["label_parses"]["train"] = 1
        groups["train"] = attach_labels(groups["train"], root, "train")
        catalog = data.read_json(probe.CATALOG)
        pairs = [reassign(g, catalog) for g in groups["train"]]
        reassigned = [g for g, _ in pairs]
        result["inputs"] = {"files": checked, "metadata": meta,
                            "original_train_sha256": public_digest(groups["train"]),
                            "reassigned_train_sha256": public_digest(reassigned),
                            "valid_sha256": public_digest(groups["development"]),
                            "transformations": [r for _, r in pairs]}
        result["manipulation"] = manipulation(groups["train"], reassigned, catalog)
        data.write_json(out / "inputs.json", result["inputs"])
        data.write_json(out / "manipulation.json", result["manipulation"])
        supplies = {"original": {g.uid: g for g in groups["train"]}, "reassigned": {g.uid: g for g in reassigned}}
        eval_groups = {"development": groups["development"]}
        model = core.load_model(base)
        for old in original["orders"]:
            order = old["order"]
            initial_path = out / "work" / f"{order}_initial.pt"
            budget.check(previous.checkpoint_reserve(model, None))
            initial = core.save_state(initial_path, model, None, {"point": order + "_initial", "config": base})
            if initial["state_sha256"] != old["points"]["initial"]["checkpoint"]["state_sha256"]:
                raise ValueError("Original initialization differs")
            log = {"order": order, "points": {}, "training": {}, "models": {}, "branch_start_state_sha256": {}}

            def current(domain: str, branch: str) -> list[data.Group]:
                return [supplies[branch][r["group_uid"]] for r in meta if r["split"] == "train" and r["domain"] == domain]

            optimizer = core.make_optimizer(model, base)
            log["training"]["shared"] = train_stage(model, optimizer, current(order[0], "original"), base, order, 1, "shared", budget)
            shared = previous.point(model, optimizer, order + "_shared", eval_groups, base, out, budget)
            same_original_point(shared, old["points"]["shared"], out)
            log["points"]["shared"] = shared
            shared_path = out / shared["checkpoint"]["path"]
            del optimizer
            for branch in BRANCHES:
                optimizer = core.make_optimizer(model, base)
                core.restore_state(shared_path, model, optimizer, shared["checkpoint"]["state_sha256"])
                log["branch_start_state_sha256"][branch] = shared["checkpoint"]["state_sha256"]
                for stage in (2, 3):
                    key = f"{branch}_stage{stage}"
                    log["training"][key] = train_stage(model, optimizer, current(order[stage - 1], branch), base, order, stage, branch, budget)
                    # Original metadata must also match the old canonical full-state digest.
                    name = f"{order}_{'sequential' if branch == 'original' else branch}_stage{stage}"
                    point = previous.point(model, optimizer, name, eval_groups, base, out, budget)
                    log["points"][key] = point
                    if branch == "original":
                        same_original_point(point, old["points"][f"sequential_stage{stage}"], out)
                    if stage == 3 and branch == "reassigned":
                        log["models"][branch] = previous.retain_inference(model, order + "_reassigned", point, base, out, budget)
                    previous.remove_work_file(out / point["checkpoint"]["path"], out)
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()
            validate_training(log, meta, base, old)
            core.restore_state(initial_path, model, None, initial["state_sha256"])
            previous.remove_work_file(shared_path, out)
            previous.remove_work_file(initial_path, out)
            result["orders"].append(log)
            data.write_json(out / f"order_{order}.json", log)
        result["physical_updates"] = sum(t["updates"] for o in result["orders"] for t in o["training"].values())
        if result["physical_updates"] != 2700 or sources() != snapshot:
            raise ValueError("Update count/source mismatch")
        result["formal_training_seconds"] = sum(t["training_seconds"] for o in result["orders"] for t in o["training"].values())
        result["inference_seconds_including_replay"] = sum(p["timing"]["inference_seconds_including_replay"] for o in result["orders"] for p in o["points"].values())
        result["retained_files_verified"] = [data.record(data.verify(out / o["models"]["reassigned"]["path"],
                                                       o["models"]["reassigned"]), out) for o in result["orders"]]
        result["budget"] = budget.state()
        result["status"] = COMPLETE
        data.write_json(out / "manifest.json", result)
        return result
    except Exception as error:
        data.write_json(out / "failure.json", {"status": "FAILED_DO_NOT_EVALUATE_OR_AUTO_RESTART",
                        "error_type": type(error).__name__, "error": str(error),
                        "elapsed_seconds": time.monotonic() - budget.started, "label_parses": result["label_parses"]})
        raise


def validated_scores(out: Path, original: dict, groups: dict, meta: list[dict], checked: dict) -> tuple[dict, dict]:
    if (out / "failure.json").exists():
        raise ValueError("Failed run cannot be evaluated")
    record, config = data.read_json(out / "manifest.json"), contract()
    if (record["status"] != COMPLETE or record["contract"] != config or record["source_files"] != sources()
            or record["original_manifest_sha256"] != probe.RUN_SHA or record["physical_updates"] != 2700
            or record["label_parses"] != {"train": 1, "development": 0, "heldout": 0, "owners": 0}
            or [r["order"] for r in record["orders"]] != list(probe.ORDERS)
            or record["budget"]["elapsed_seconds"] > config["runtime"]["maximum_gpu_stage_seconds"]
            or record["budget"]["peak_observed_bytes"] > config["runtime"]["maximum_output_bytes"]):
        raise ValueError("Incomplete or mismatched training contract")
    pairs = [reassign(g, data.read_json(probe.CATALOG)) for g in groups["train"]]
    expected_inputs = {"files": checked, "metadata": meta, "original_train_sha256": public_digest(groups["train"]),
                       "reassigned_train_sha256": public_digest([g for g, _ in pairs]),
                       "valid_sha256": public_digest(groups["development"]), "transformations": [r for _, r in pairs]}
    if record["inputs"] != expected_inputs:
        raise ValueError("Training/valid public bytes or transformation differ")
    values = {}
    expected_retained = [{k: o["models"]["reassigned"][k] for k in ("path", "bytes", "sha256")} for o in record["orders"]]
    if record["retained_files_verified"] != expected_retained:
        raise ValueError("Final Linux inference payload verification missing")
    for log, old in zip(record["orders"], original["orders"]):
        if set(log["points"]) != set(POINTS) or set(log["training"]) != set(POINTS) or set(log["models"]) != {"reassigned"}:
            raise ValueError("Incomplete points/models")
        validate_training(log, meta, original["config"], old)
        shared_digest = old["points"]["shared"]["checkpoint"]["state_sha256"]
        if log["branch_start_state_sha256"] != dict.fromkeys(BRANCHES, shared_digest):
            raise ValueError("Branches did not share full original Adam state")
        model = log["models"]["reassigned"]
        if model["path"] != f"models/{log['order']}_reassigned.pt" or not model["actual_loaded_model_equals_replayed_state"]:
            raise ValueError("Retained inference model receipt differs")
        for key, point in log["points"].items():
            if not point["full_model_and_adam_reloaded"] or set(point["scores"]) != {"development"}:
                raise ValueError("Full model/Adam reload or valid-only scope differs")
            if key == "shared" or key.startswith("original"):
                old_key = "shared" if key == "shared" else key.replace("original", "sequential")
                same_original_point(point, old["points"][old_key], out)
            name = key.replace("original", "sequential")
            r = point["scores"]["development"]
            if r["path"] != f"scores/{log['order']}_{name}_development.npy":
                raise ValueError("Score path mapping differs")
            array = np.load(data.verify(out / r["path"], r), allow_pickle=False)
            if array.shape != (60, 378) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Invalid/incomplete score array")
            values[(log["order"], key)] = array
    return values, record


def contrasts(arrays: dict, column: int) -> dict[str, np.ndarray]:
    result = {}
    def rows(key: str) -> np.ndarray:
        return np.stack([arrays[(order, key)][i * 20:(i + 1) * 20, column] for i, order in enumerate(probe.ORDERS)])
    for stage in (2, 3):
        for branch in BRANCHES:
            result[f"{branch}_first_domain_change_stage{stage}"] = rows(f"{branch}_stage{stage}") - rows("shared")
        result[f"original_minus_reassigned_first_domain_change_stage{stage}"] = rows(f"original_stage{stage}") - rows(f"reassigned_stage{stage}")
    return result


def evaluate(out: Path, destination: Path) -> dict:
    if destination.exists() or not destination.is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a new evaluation directory")
    original = probe.original_manifest()
    groups, meta, checked = public_inputs(original)
    scores, run_record = validated_scores(out, original, groups, meta, checked)
    destination.mkdir(parents=True)
    data.write_json(destination / "access.json", {"status": "COMPLETE_GATE_PASSED_VALID_PARSE_STARTING",
                                                  "development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    labelled = attach_labels(groups["development"], data.ROOT / original["config"]["data_root"], "development")
    labels = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    arrays, summaries = {}, {}
    for key, values in scores.items():
        classification = [metrics.classification(y, s) for y, s in zip(labels, values)]
        matrix = np.column_stack((np.asarray([[r[c] for c in metrics.CLASS_KEYS] for r in classification]),
                                  metrics.retrieval(labels, values, 28)))
        arrays[key] = matrix
        path = destination / ("_".join(key) + "_metrics.npy")
        np.save(path, matrix, allow_pickle=False)
        summaries["_".join(key)] = {"file": data.record(path, destination),
            "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[i * 20:(i + 1) * 20].mean(0).tolist())) for i, d in enumerate("ABC")},
            "confusion": [r["confusion"] for r in classification]}
    config = contract()
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, 20, size=(config["bootstrap_draws"], 3, 20))
    comparisons = {}
    for j, c in enumerate(metrics.COLUMNS):
        comparisons[c] = {}
        for name, delta in contrasts(arrays, j).items():
            sampled = np.stack([delta[i][draws[:, i]] for i in range(3)], axis=1).mean((1, 2))
            comparisons[c][name] = {"mean": float(delta.mean()), "by_first_domain": dict(zip("ABC", delta.mean(1).tolist())),
                                    "conditional_95pct_interval": np.quantile(sampled, [.025, .975]).tolist()}
    transitions = {}
    for order in probe.ORDERS:
        for branch in BRANCHES:
            for stage in (2, 3):
                before = "shared" if stage == 2 else f"{branch}_stage2"
                delta = arrays[(order, f"{branch}_stage{stage}")] - arrays[(order, before)]
                transitions[f"{order}_{branch}_stage{stage}"] = {
                    "trained_domain": order[stage - 1], "old_domains": list(order[:stage - 1]),
                    "by_domain_change": {d: dict(zip(metrics.COLUMNS, delta[i * 20:(i + 1) * 20].mean(0).tolist())) for i, d in enumerate("ABC")}}
    result = {"status": "PAIRED_TRAINING_EVALUATED_INTERPRETATION_REQUIRED", "columns": list(metrics.COLUMNS),
              "label_parses": {"development": 1, "train": 0, "heldout": 0, "owners": 0},
              "score_manifest_sha256": data.sha256(out / "manifest.json"), "points": summaries,
              "contrasts": comparisons, "transitions": transitions, "train_manipulation": run_record["manipulation"],
              "bootstrap": {"seed": config["bootstrap_seed"], "draws": config["bootstrap_draws"], "unit": "paired group within first domain"},
              "limit": "Fixed training paths and one intervention; assess manipulation and new learning; not unique cause, pure style, or mediation percentage."}
    data.write_json(destination / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    if args.action == "evaluate" and (platform.system() != "Windows" or args.evaluation is None):
        parser.error("Evaluation requires Windows and a new --evaluation directory")
    result = run(args.out.resolve()) if args.action == "train" else evaluate(args.out.resolve(), args.evaluation.resolve())
    print(data.json_bytes({"status": result["status"], "output": str(args.out)}).decode())


if __name__ == "__main__":
    main()
