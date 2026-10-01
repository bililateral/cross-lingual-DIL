"""Linux-only nine-point diagnostic; launch after reported-stage resumption."""
from __future__ import annotations

import argparse
import gc
import platform
import shutil
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import step28_continual_diagnostic as diag
import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics
import step28_continual_population_run as legacy


def module_digests(model) -> dict:
    return {name: core.state_digest(module.state_dict())
            for name, module in (("encoder", model.encoder), ("head", model.head))}


def train_segment(model, optimizer, groups: list[data.Group], config: dict,
                  anchor: str, phase: str, check) -> dict:
    started = time.monotonic()
    seed = diag.segment_seed(config, anchor, phase)
    order = diag.positions(len(groups), config["epochs_per_stage"], seed)
    before = module_digests(model)
    losses, gradient_norms, first_gradients = [], [], {}
    for index, position in enumerate(order):
        log = core.update(model, optimizer, groups[position], None, config,
                          data.seed_for(seed, index, "dropout"), 0, check)
        losses.append(log["current_bce"])
        gradient_norms.append(log["gradient_norm_before_clip"])
        if index == 0:
            for name, module in (("encoder", model.encoder), ("head", model.head)):
                norms = [float(p.grad.detach().float().norm()) for p in module.parameters() if p.grad is not None]
                if not norms or not all(np.isfinite(norms)) or not max(norms) > 0:
                    raise ValueError("Missing finite nonzero encoder/head gradient")
                first_gradients[name] = float(np.linalg.norm(norms))
        if (index + 1) % len(groups) == 0:
            print(data.json_bytes({"event": "epoch", "anchor": anchor, "phase": phase,
                "epoch": (index + 1) // len(groups), "updates": index + 1,
                "online_bce": float(np.mean(losses[-len(groups):]))}).decode(), flush=True)
    after = module_digests(model)
    if any(before[k] == after[k] for k in before):
        raise ValueError("Formal segment did not change both trainable modules")
    expected_step = len(order) * (1 if phase == "shared" else 2)
    steps = {float(value["step"]) for value in optimizer.state.values()}
    if not optimizer.state or steps != {expected_step}:
        raise ValueError("Adam continuation/update accounting differs")
    return {"updates": len(order), "group_ids": [groups[i].uid for i in order],
            "position_order": order, "dropout_stream": seed,
            "online_bce_per_epoch": [float(np.mean(losses[i:i + len(groups)]))
                                     for i in range(0, len(losses), len(groups))],
            "training_seconds": time.monotonic() - started,
            "first_update_clipped_gradient_norms": first_gradients,
            "module_state_before": before, "module_state_after": after,
            "maximum_gradient_norm_before_clip": max(gradient_norms),
            "adam_step": expected_step}


def save_point(model, optimizer, name: str, groups: dict, archive: diag.Archive,
               config: dict, output: Path, budget: legacy.Budget) -> dict:
    # Existing point is generic over supplied splits; no legacy archive or run is called.
    record = legacy.point(model, optimizer, name, groups, config, output, budget)
    labels = np.array([g.labels for g in groups["train"]], dtype=np.uint8)
    scores = np.load(output / record["scores"]["train"]["path"], allow_pickle=False)
    matrix, confusion = metrics.group_metrics(labels, scores)
    path = output / "metrics" / f"{name}_train.npy"
    np.save(path, matrix, allow_pickle=False)
    record["train_metrics"] = {**diag.summarize(matrix, diag.domain_rows(archive, "train")),
        "columns": list(metrics.COLUMNS), "per_group_confusion": confusion,
        "per_group_metrics": data.record(path, output)}
    return record


def run(output: Path) -> dict:
    import torch
    config = diag.policy()
    if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires the resumed Linux stage and one visible GPU")
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < config["runtime"]["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient idle eligible GPU capacity; wait")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < config["runtime"]["minimum_free_host_bytes"] or shutil.disk_usage(data.ROOT).free < config["runtime"]["maximum_output_bytes"]:
        raise RuntimeError("Insufficient shared-server RAM/disk reserve")
    if output.exists() or not output.resolve().is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a fresh diagnostic reports directory")
    output.mkdir(parents=True)
    for name in ("work", "scores", "metrics"):
        (output / name).mkdir()
    budget = legacy.Budget(output, config)
    start_sources = diag.sources()
    manifest = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "source_files": start_sources, "rotations": [],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(output / "startup.json", manifest)
    try:
        torch.set_num_threads(config["runtime"]["torch_cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        sys.path.insert(0, str(data.ROOT / "tests"))
        suite = unittest.defaultTestLoader.loadTestsFromName("test_step28_continual_diagnostic_contracts.TorchContracts")
        tests = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
        if not tests.wasSuccessful() or tests.skipped or tests.testsRun != 2:
            raise RuntimeError("Both actual Torch diagnostic contracts must pass")
        manifest["torch_contracts"] = {"passed": tests.testsRun, "skipped": 0, "failed": 0}
        pretrained = core.model_files(config)
        if any(pretrained[k] != config["model"][k] for k in ("file_count", "total_size_bytes", "content_sha256")):
            raise ValueError("Original pretrained payload differs")
        manifest["pretrained_model"] = pretrained
        archive = diag.Archive(config, train_labels=True)
        groups = {split: archive.groups(split) for split in diag.SPLITS}
        manifest["group_ids"] = {s: [g.uid for g in rows] for s, rows in groups.items()}
        manifest["verified_input_files"] = archive.checked
        manifest["label_reads"] = {"train": archive.train_label_parses, "development": 0,
                                   "heldout": 0, "owners": 0, "old_assets": 0}
        model = core.load_model(config)
        original_path = output / "work/original.pt"
        budget.check(legacy.checkpoint_reserve(model, None))
        original = core.save_state(original_path, model, None, {"role": "diagnostic_original"})
        original_digest = core.state_digest(model.state_dict())
        for anchor, target in config["rotations"]:
            core.restore_state(original_path, model, None, original["state_sha256"])
            if core.state_digest(model.state_dict()) != original_digest:
                raise ValueError("Initial state differs across rotations")
            optimizer = core.make_optimizer(model, config)
            training = {"shared": train_segment(model, optimizer, archive.groups("train", anchor),
                        config, anchor, "shared", budget.check)}
            shared = save_point(model, optimizer, f"{anchor}_shared", groups, archive, config, output, budget)
            points = {"shared": shared}
            branch_starts = {}
            shared_path = output / shared["checkpoint"]["path"]
            del optimizer
            for arm, domain in (("same", anchor), ("cross", target)):
                optimizer = core.make_optimizer(model, config)
                core.restore_state(shared_path, model, optimizer, shared["checkpoint"]["state_sha256"])
                branch_starts[arm] = {"model": core.state_digest(model.state_dict()),
                                      "optimizer": core.state_digest(optimizer.state_dict())}
                training[arm] = train_segment(model, optimizer, archive.groups("train", domain),
                                             config, anchor, "branch", budget.check)
                points[arm] = save_point(model, optimizer, f"{anchor}_{arm}", groups, archive, config, output, budget)
                legacy.remove_work_file(output / points[arm]["checkpoint"]["path"], output)
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()
            if branch_starts["same"] != branch_starts["cross"]:
                raise ValueError("Branches did not start from identical model/Adam")
            legacy.remove_work_file(shared_path, output)
            row = {"anchor": anchor, "target": target, "training": training, "points": points,
                   "initial_model_sha256": original_digest, "branch_starts": branch_starts}
            manifest["rotations"].append(row)
            data.write_json(output / f"rotation_{anchor}.json", row)
        legacy.remove_work_file(original_path, output)
        manifest["physical_updates"] = sum(t["updates"] for r in manifest["rotations"] for t in r["training"].values())
        if manifest["physical_updates"] != config["physical_updates"] or diag.sources() != start_sources:
            raise ValueError("Update count or scientific sources changed")
        manifest["formal_training_seconds"] = sum(t["training_seconds"] for r in manifest["rotations"] for t in r["training"].values())
        manifest["formal_scoring_seconds"] = sum(p["timing"]["inference_seconds_including_replay"]
                                                for r in manifest["rotations"] for p in r["points"].values())
        manifest["formal_point_seconds"] = sum(p["timing"]["total_point_seconds"]
                                               for r in manifest["rotations"] for p in r["points"].values())
        if list((output / "work").iterdir()):
            raise ValueError("Unexpected work-state remainder")
        manifest["intermediate_states_removed_after_reload"] = True
        manifest["budget"] = budget.state()
        manifest["status"] = diag.COMPLETE
        data.write_json(output / "manifest.json", manifest)
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > config["runtime"]["maximum_retained_bytes"]:
            raise RuntimeError("Retained small outputs exceed approval")
        return {"status": diag.COMPLETE, "output": str(output), "training_seconds": manifest["formal_training_seconds"],
                "budget": manifest["budget"]}
    except Exception as error:
        data.write_json(output / "failure.json", {"status": "FAILED_DO_NOT_EVALUATE_OR_AUTO_RESTART",
            "error_type": type(error).__name__, "error": str(error), "source_files": start_sources,
            "completed_rotations": manifest["rotations"], "elapsed_seconds": time.monotonic() - budget.started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(data.json_bytes(run(args.out.resolve())).decode())


if __name__ == "__main__":
    main()
