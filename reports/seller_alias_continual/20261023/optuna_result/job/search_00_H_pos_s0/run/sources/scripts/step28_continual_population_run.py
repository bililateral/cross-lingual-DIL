"""Linux single-GPU execution of the approved new-population comparison.

Run only after the user resumes this concrete Linux stage. The production path
first checks the changed real LaBSE semantics, then executes the frozen schedule.
It never opens valid/test labels and does not resume a failed run automatically.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import random
import shutil
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_population as core
import step28_continual_population_data as data

ARMS = ("sequential", "er", "cumulative")
COMPLETE = "COMPLETE_BLIND_SCORES_RELOADED_VALID_AND_TEST_LABELS_UNREAD"


def sources() -> list[dict]:
    paths = sorted((data.ROOT / "scripts").glob("step28_continual_population*.py"))
    paths += sorted((data.ROOT / "tests").glob("test_step28_continual_population*.py"))
    paths.append(data.POLICY_PATH)
    return [data.record(path, data.ROOT) for path in paths]


class Budget:
    def __init__(self, root: Path, config: dict):
        self.root, self.config = root, config["runtime"]
        self.started = time.monotonic()
        self.last_disk_check = -float("inf")
        self.peak_bytes = 0

    def check(self, reserve: int = 0) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.config["maximum_gpu_stage_seconds"]:
            raise RuntimeError("Approved GPU-stage time budget reached")
        if reserve or elapsed - self.last_disk_check > 10:
            used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            self.peak_bytes = max(self.peak_bytes, used)
            self.last_disk_check = elapsed
            if used + reserve > self.config["maximum_output_bytes"]:
                raise RuntimeError("Approved output budget would be exceeded")
            if shutil.disk_usage(self.root).free < reserve:
                raise RuntimeError("Insufficient free space; do not affect other users")

    def state(self) -> dict:
        self.check(reserve=1)
        return {"elapsed_seconds": time.monotonic() - self.started, "peak_observed_bytes": self.peak_bytes}


def checkpoint_reserve(model: Any, optimizer: Any | None) -> int:
    # Conservative reserve before creating a file, including future torch metadata.
    parameters = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    return parameters * (3 if optimizer is not None else 1) + 64 * 1024**2


def remove_work_file(path: Path, root: Path) -> None:
    # Only new, verified intermediate states of THIS run. Never a general cleanup.
    resolved, work = path.resolve(), (root / "work").resolve()
    if resolved.parent != work or resolved.suffix != ".pt" or resolved.is_symlink():
        raise ValueError("Not this run's disposable intermediate checkpoint")
    resolved.unlink()


def point(model: Any, optimizer: Any | None, name: str, groups: dict,
          config: dict, out: Path, budget: Budget, memory: data.Memory | None = None) -> dict:
    started = time.monotonic()
    before = {split: core.score(model, rows, config, budget.check) for split, rows in groups.items()}
    inference_seconds = time.monotonic() - started
    checkpoint = out / "work" / f"{name}.pt"
    metadata = {"point": name, "config": config}
    budget.check(checkpoint_reserve(model, optimizer))
    state = core.save_state(checkpoint, model, optimizer, metadata)
    restored = core.restore_state(checkpoint, model, optimizer, state["state_sha256"])
    if restored != metadata:
        raise ValueError("Checkpoint metadata differs")
    result = {"checkpoint": {**state, "path": checkpoint.relative_to(out).as_posix()},
              "model_state_sha256": core.state_digest(model.state_dict()),
              "full_model_and_adam_reloaded": True, "scores": {}}
    for split, rows in groups.items():
        inference_started = time.monotonic()
        after = core.score(model, rows, config, budget.check)
        inference_seconds += time.monotonic() - inference_started
        if not np.array_equal(before[split], after):
            raise ValueError("Complete blind scores differ after actual checkpoint reload")
        path = out / "scores" / f"{name}_{split}.npy"
        np.save(path, before[split], allow_pickle=False)
        result["scores"][split] = data.record(path, out)
    if memory is not None:
        payload = memory.to_bytes()
        path = out / "memory" / f"{name}.json"
        path.write_bytes(payload)
        restored_memory = data.Memory.from_bytes(path.read_bytes())
        if restored_memory.to_bytes() != payload:
            raise ValueError("Actual persisted memory differs")
        result["memory"] = {**data.record(path, out), "seen": memory.seen,
                            "retained_groups": [g.uid for g in memory.groups],
                            "disk_roundtrip_exact": True}
    result["timing"] = {"inference_seconds_including_replay": inference_seconds,
                        "total_point_seconds": time.monotonic() - started}
    print(data.json_bytes({"event": "point_reloaded", "point": name,
                          **result["timing"], **budget.state()}).decode(), flush=True)
    return result


def retain_inference(model: Any, name: str, point_record: dict, config: dict,
                     out: Path, budget: Budget) -> dict:
    target = out / "models" / f"{name}.pt"
    budget.check(checkpoint_reserve(model, None))
    saved = core.save_state(target, model, None, {"point": name, "config": config})
    core.restore_state(target, model, None, saved["state_sha256"])
    if core.state_digest(model.state_dict()) != point_record["model_state_sha256"]:
        raise ValueError("Retained inference model differs from the replayed model")
    return {**saved, "path": target.relative_to(out).as_posix(),
            "actual_loaded_model_equals_replayed_state": True}


def train_stage(model: Any, optimizer: Any, current: list[data.Group], memory: data.Memory | None,
                config: dict, order: str, stage: int, budget: Budget, *, cumulative: bool = False) -> dict:
    started = time.monotonic()
    base_seed = config["initialization_seed"]
    stream = data.seed_for(base_seed, order, stage, "cumulative" if cumulative else "current")
    rows = data.schedule(current, config["epochs_per_stage"], stream)
    replay_rng = random.Random(data.seed_for(base_seed, order, stage, "replay_draw"))
    first_memory = memory.to_bytes() if memory is not None else None
    current_losses, historical_losses, norms, history_ids = [], [], [], []
    for index, group in enumerate(rows):
        replay = replay_rng.choice(memory.groups) if memory is not None and memory.groups else None
        log = core.update(model, optimizer, group, replay, config,
                          data.seed_for(stream, index, "dropout_current"),
                          data.seed_for(stream, index, "dropout_replay"), budget.check)
        current_losses.append(log["current_bce"])
        norms.append(log["gradient_norm_before_clip"])
        if replay is not None:
            historical_losses.append(log["replay_bce"])
            history_ids.append(replay.uid)
        if (index + 1) % 30 == 0:
            print(data.json_bytes({"event": "updates", "order": order, "stage": stage,
                  "arm": "cumulative" if cumulative else "er" if memory is not None else "sequential",
                  "completed": index + 1, "total": len(rows),
                  "stage_training_seconds": time.monotonic() - started, **budget.state()}).decode(), flush=True)
    if memory is not None and memory.to_bytes() != first_memory:
        raise ValueError("Historical memory changed during a stage")
    return {"updates": len(rows), "training_seconds": time.monotonic() - started,
            "current_group_ids": [g.uid for g in rows],
            "current_order_sha256": hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest(),
            "current_dropout_stream": stream, "replay_group_ids": history_ids,
            "current_pair_presentations": sum(len(g.labels) for g in rows),
            "replay_pair_presentations": len(history_ids) * 378,
            "current_bce_mean": float(np.mean(current_losses)),
            "replay_bce_mean": float(np.mean(historical_losses)) if historical_losses else None,
            "maximum_gradient_norm_before_clip": max(norms), "memory_frozen_during_stage": True}


def runtime_check(model: Any, archive: data.Archive, config: dict, out: Path, budget: Budget) -> dict:
    """New actual encoder/head gradients, updates and complete state reload."""
    import torch
    started = time.monotonic()
    sys.path.insert(0, str(data.ROOT / "tests"))
    suite = unittest.defaultTestLoader.loadTestsFromName("test_step28_continual_population_contracts.TorchContracts")
    tests = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.skipped or tests.testsRun != 3:
        raise RuntimeError("Necessary actual PyTorch semantic cases did not all pass")
    current, replay = archive.groups("train", "A")[:2]
    original = out / "work" / "runtime_original.pt"
    budget.check(checkpoint_reserve(model, None))
    original_record = core.save_state(original, model, None, {"role": "precheck_original"})
    optimizer = core.make_optimizer(model, config)
    initial_digest = {name: core.state_digest(module.state_dict())
                      for name, module in (("encoder", model.encoder), ("head", model.head))}
    branch = {}
    for label, group in (("current", current), ("replay", replay)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = core.backward_group(model, group, config, data.seed_for(20260909, "runtime", label), budget.check)
        branch[label] = {"loss": loss}
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            norms = [float(p.grad.detach().float().norm()) for p in module.parameters() if p.grad is not None]
            if not norms or not all(np.isfinite(norms)) or not max(norms) > 0:
                raise ValueError(f"No finite nonzero {label} gradient in {name}")
            branch[label][name + "_gradient_norm"] = float(np.linalg.norm(norms))
    update_log = core.update(model, optimizer, current, replay, config, 701, 702, budget.check)
    for name, module in (("encoder", model.encoder), ("head", model.head)):
        if core.state_digest(module.state_dict()) == initial_digest[name]:
            raise ValueError(f"Runtime update did not change {name}")
    memory = data.Memory(data.seed_for(20260909, "runtime_memory"))
    memory.add_stage(archive.groups("train", "A"))
    # A complete group's 378 scores in both blind splits; full formal points follow.
    groups = {s: archive.groups(s)[:1] for s in ("development", "heldout")}
    result = point(model, optimizer, "runtime", groups, config, out, budget, memory)
    if not optimizer.state or not all(float(v["step"]) == 1 for v in optimizer.state.values()):
        raise ValueError("First actual Adam update/reload differs")
    core.restore_state(original, model, None, original_record["state_sha256"])
    for name, module in (("encoder", model.encoder), ("head", model.head)):
        if core.state_digest(module.state_dict()) != initial_digest[name]:
            raise ValueError("Runtime check altered formal initialization")
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    remove_work_file(out / result["checkpoint"]["path"], out)
    remove_work_file(original, out)
    return {"status": "PASS_REAL_LABSE_NEW_INPUT_HEAD_MEMORY_ADAM_RELOAD",
            "branches": branch, "update": update_log, "point": result,
            "formal_initialization_restored": True, "formal_update_count": 0,
            "torch_contracts": {"passed": tests.testsRun, "skipped": 0, "failed": 0},
            "runtime_check_seconds": time.monotonic() - started}


def run(out: Path, *, check_only: bool = False) -> dict:
    import torch
    config = data.policy()
    if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires the authorized Linux stage and exactly one visible GPU")
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < config["runtime"]["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient eligible idle GPU capacity; wait")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < config["runtime"]["minimum_free_host_bytes"]:
        raise RuntimeError("Insufficient available host RAM; wait")
    if shutil.disk_usage(data.ROOT).free < config["runtime"]["maximum_output_bytes"]:
        raise RuntimeError("Insufficient project disk reserve; wait")
    if out.exists() or not out.resolve().is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a new run directory inside project reports")
    out.mkdir(parents=True)
    for name in ("work", "models", "scores", "memory"):
        (out / name).mkdir()
    budget = Budget(out, config)
    startup_sources = sources()
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest: dict = {"status": "RUNNING", "started_utc": started_utc, "config": config,
                      "source_files": startup_sources, "orders": [],
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(out / "startup.json", manifest)
    try:
        torch.set_num_threads(config["runtime"]["torch_cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model_record = core.model_files(config)
        for key in ("file_count", "total_size_bytes", "content_sha256"):
            if model_record[key] != config["model"][key]:
                raise ValueError("Original pretrained model bytes differ")
        manifest["pretrained_model"] = model_record
        archive = data.Archive(config, train_labels=True)
        manifest["verified_input_files"] = archive.checked
        manifest["label_reads"] = {"new_train_csv_offline_packaging": archive.train_label_parses,
                                   "development": 0, "heldout": 0, "owners": 0, "old_assets": 0}
        model = core.load_model(config)
        manifest["runtime_check"] = runtime_check(model, archive, config, out, budget)
        data.write_json(out / "runtime_check.json", {"source_files": startup_sources, **manifest["runtime_check"]})
        if check_only:
            manifest["status"] = "REAL_CHECK_ONLY_NO_FORMAL_TRAINING"
        else:
            eval_groups = {s: archive.groups(s) for s in ("development", "heldout")}
            manifest["evaluation_group_ids"] = {s: [g.uid for g in rows] for s, rows in eval_groups.items()}
            initial_model_digest = core.state_digest(model.state_dict())
            for order in config["orders"]:
                if core.state_digest(model.state_dict()) != initial_model_digest:
                    raise ValueError("Order did not start from common original state")
                log: dict = {"order": order, "points": {}, "training": {}, "trajectory": {}, "models": {}}
                points = log["points"]
                points["initial"] = point(model, None, order + "_initial", eval_groups, config, out, budget)
                initial_path = out / points["initial"]["checkpoint"]["path"]
                optimizer = core.make_optimizer(model, config)
                first = archive.groups("train", order[0])
                log["training"]["shared"] = train_stage(model, optimizer, first, None, config, order, 1, budget)
                memory = data.Memory(data.seed_for(config["initialization_seed"], order, "memory"))
                memory.add_stage(first)
                memory_bytes = memory.to_bytes()
                points["shared"] = point(model, optimizer, order + "_shared", eval_groups, config, out, budget, memory)
                shared_path = out / points["shared"]["checkpoint"]["path"]
                log["models"]["frozen"] = retain_inference(model, order + "_frozen", points["shared"], config, out, budget)
                log["trajectory"]["frozen"] = ["initial", "shared", "shared", "shared"]
                del optimizer
                for arm in ARMS:
                    log["training"][arm] = []
                    log["trajectory"][arm] = ["initial", "shared"]
                    optimizer = core.make_optimizer(model, config)
                    core.restore_state(shared_path, model, optimizer, points["shared"]["checkpoint"]["state_sha256"])
                    memory = data.Memory.from_bytes(memory_bytes) if arm == "er" else None
                    for stage in (2, 3):
                        if arm == "cumulative":
                            core.restore_state(initial_path, model, None, points["initial"]["checkpoint"]["state_sha256"])
                            del optimizer
                            optimizer = core.make_optimizer(model, config)
                        current = archive.groups("train", order[:stage] if arm == "cumulative" else order[stage - 1])
                        stage_log = train_stage(model, optimizer, current, memory, config, order, stage, budget,
                                                cumulative=arm == "cumulative")
                        log["training"][arm].append(stage_log)
                        if memory is not None:
                            memory.add_stage(current)
                        key = f"{arm}_stage{stage}"
                        points[key] = point(model, optimizer, order + "_" + key, eval_groups, config, out, budget, memory)
                        log["trajectory"][arm].append(key)
                        if stage == 3:
                            log["models"][arm] = retain_inference(model, order + "_" + arm, points[key], config, out, budget)
                        remove_work_file(out / points[key]["checkpoint"]["path"], out)
                    del optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                for index in (0, 1):
                    seq, er = (log["training"][a][index] for a in ("sequential", "er"))
                    if (seq["current_order_sha256"], seq["current_dropout_stream"]) != (er["current_order_sha256"], er["current_dropout_stream"]):
                        raise ValueError("Unpaired current presentations/dropout")
                core.restore_state(initial_path, model, None, points["initial"]["checkpoint"]["state_sha256"])
                remove_work_file(shared_path, out)
                remove_work_file(initial_path, out)
                manifest["orders"].append(log)
                data.write_json(out / f"order_{order}.json", log)
            manifest["physical_updates"] = sum(o["training"]["shared"]["updates"] +
                sum(r["updates"] for arm in ARMS for r in o["training"][arm]) for o in manifest["orders"])
            if manifest["physical_updates"] != config["physical_updates"]:
                raise ValueError("Physical optimizer update count differs")
            manifest["formal_training_seconds"] = sum(o["training"]["shared"]["training_seconds"] +
                sum(r["training_seconds"] for arm in ARMS for r in o["training"][arm]) for o in manifest["orders"])
            manifest["formal_inference_seconds_including_replay"] = sum(
                p["timing"]["inference_seconds_including_replay"] for o in manifest["orders"] for p in o["points"].values())
            manifest["status"] = COMPLETE
        if sources() != startup_sources:
            raise ValueError("Scientific sources changed during execution")
        manifest["budget"] = budget.state()
        data.write_json(out / "manifest.json", manifest)
        return {"status": manifest["status"], "output": str(out), "budget": manifest["budget"]}
    except Exception as error:
        data.write_json(out / "failure.json", {"status": "FAILED_DO_NOT_EVALUATE_OR_AUTO_RESTART",
                        "error_type": type(error).__name__, "error": str(error),
                        "startup_source_files": startup_sources, "failure_source_files": sources(),
                        "elapsed_seconds": time.monotonic() - budget.started,
                        "completed_orders": manifest["orders"]})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New reports directory; never overwrite")
    parser.add_argument("--check-only", action="store_true", help="Run changed real LaBSE checks only")
    args = parser.parse_args()
    print(data.json_bytes(run(args.out.resolve(), check_only=args.check_only)).decode())


if __name__ == "__main__":
    main()
