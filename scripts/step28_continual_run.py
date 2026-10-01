"""Four fixed-feature continual controls; training never opens development truth."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_core as core
import step28_continual_data as data
from step28_continual_evaluate import curve_metrics

ARMS = ("frozen", "sequential", "er", "cumulative")


def same_state(a: Any, b: Any) -> bool:
    torch = core.torch_module()
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def save_point(directory: Path, model, optimizer, prep: core.Preprocessor,
               development: np.ndarray, memory: core.Reservoir | None = None) -> dict[str, Any]:
    """Publish unlabeled artifacts; replay optimizer, preprocessing and scores.

    ER's label-bearing cache is round-tripped in a temporary directory, removed
    immediately, and never put in public reports. Archives are not trainer inputs.
    """
    torch = core.torch_module()
    directory.mkdir()
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "medians": torch.from_numpy(prep.medians.copy()), "means": torch.from_numpy(prep.means.copy()),
               "scales": torch.from_numpy(prep.scales.copy())}
    checkpoint = directory / "checkpoint.pt"
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not same_state(payload, loaded):
        raise ValueError("Checkpoint state differs on disk")
    restored_model = core.make_head()
    restored_model.load_state_dict(loaded["model"])
    restored_optimizer = core.make_optimizer(restored_model)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    if not same_state(optimizer.state_dict(), restored_optimizer.state_dict()):
        raise ValueError("Reloaded optimizer differs")
    restored_prep = core.Preprocessor(*(loaded[k].numpy() for k in ("medians", "means", "scales")))
    if not np.array_equal(prep.transform(development), restored_prep.transform(development)):
        raise ValueError("Reloaded preprocessing differs")
    x = prep.transform(development)
    scores = core.score(model, x)
    if not np.array_equal(scores, core.score(restored_model, x)):
        raise ValueError("Reloaded complete development scores differ")
    np.save(directory / "logits.npy", scores, allow_pickle=False)
    if not np.array_equal(scores, np.load(directory / "logits.npy", allow_pickle=False)):
        raise ValueError("Saved score bytes do not replay")
    memory_record = None
    if memory is not None:
        array_bytes, state_bytes = memory.serialized_parts()
        with tempfile.TemporaryDirectory(prefix="continual_memory_") as scratch:
            ap, sp = Path(scratch) / "memory.npy", Path(scratch) / "memory.json"
            ap.write_bytes(array_bytes)
            sp.write_bytes(state_bytes)
            restored_memory = core.Reservoir.restore(ap.read_bytes(), sp.read_bytes())
            if restored_memory.serialized_parts() != (array_bytes, state_bytes):
                raise ValueError("Saved ER memory differs")
            # Check the subsequent draw without consuming the live sampling RNG.
            first, second = copy.deepcopy(memory).sample(256), restored_memory.sample(256)
            if first is None or second is None or any(not np.array_equal(a, b) for a, b in zip(first, second)):
                raise ValueError("Reloaded ER next draw differs")
        memory_record = {"seen_unique_rows": memory.seen, "records": len(memory.records),
                         "positive_records": int(memory.records["label"].sum()),
                         "array_bytes": len(array_bytes), "state_bytes": len(state_bytes),
                         "total_bytes": len(array_bytes) + len(state_bytes),
                         "array_sha256": hashlib.sha256(array_bytes).hexdigest(),
                         "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
                         "disk_and_next_draw_replayed": True}
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optimizer_bytes = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                          for value in state.values() if torch.is_tensor(value))
    return {"checkpoint": data.record(checkpoint, directory.parent),
            "scores": data.record(directory / "logits.npy", directory.parent),
            "model_tensor_bytes": model_bytes, "optimizer_tensor_bytes": optimizer_bytes,
            "preprocessing_array_bytes": prep.medians.nbytes + prep.means.nbytes + prep.scales.nbytes,
            "memory": memory_record, "disk_model_optimizer_preprocessing_scores_exact": True}


def expected_log(rows: int, epochs: int, batch: int, replay_size: int = 0) -> dict[str, int]:
    updates = epochs * math.ceil(rows / batch)
    return {"updates": updates, "current_presentations": rows * epochs,
            "replay_presentations": updates * min(batch, replay_size)}


def fit(model, optimizer, x: np.ndarray, y: np.ndarray, seed: int, stage: int,
        settings: dict, memory: core.Reservoir | None = None, new_rows: int | None = None) -> dict:
    start = time.perf_counter()
    before = core.score(model, x).astype(np.float64)
    result = core.train_stage(model, optimizer, x, y, order_seed=seed, stage=stage,
                              memory=memory, epochs=settings["epochs"], batch_size=settings["batch_size"])
    expected = expected_log(len(y), settings["epochs"], settings["batch_size"], len(memory.records) if memory else 0)
    if any(result[k] != v for k, v in expected.items()):
        raise ValueError("Actual presentation/update counts differ")
    after = core.score(model, x).astype(np.float64)
    result["training_diagnostic"] = {
        "population": "current rows" if new_rows is None else "arrived prefix",
        "before_ap": curve_metrics(y, before)["average_precision"],
        "after_ap": curve_metrics(y, after)["average_precision"],
        "before_bce": float(np.mean(np.logaddexp(0., before) - y * before)),
        "after_bce": float(np.mean(np.logaddexp(0., after) - y * after)),
        "forward_presentations": 2 * len(y), "selection_or_early_stopping": False}
    result["fit_seconds"] = time.perf_counter() - start
    result["input_array_bytes"] = x.nbytes + y.nbytes
    result["new_arrival_presentations"] = (len(y) if new_rows is None else new_rows) * settings["epochs"]
    result["prior_arrival_presentations"] = (len(y) - (len(y) if new_rows is None else new_rows)) * settings["epochs"]
    return result


def run_order(source: data.StageSource, development: np.ndarray, seed: int,
              output: Path, settings: dict) -> dict[str, Any]:
    output.mkdir()
    raw, y = source.load(1)
    prep = core.Preprocessor.fit(raw)
    x = prep.transform(raw)
    model = core.make_head(settings["initialization_seed"])
    optimizer = core.make_optimizer(model)
    points = {"initial": save_point(output / "initial", model, optimizer, prep, development)}
    initial_state = copy.deepcopy(model.state_dict())
    first_log = fit(model, optimizer, x, y, seed, 1, settings)
    if same_state(initial_state, model.state_dict()):
        raise ValueError("Shared first-stage model did not update")
    points["shared"] = save_point(output / "shared", model, optimizer, prep, development)
    shared_state = copy.deepcopy(model.state_dict())
    sequential, sequential_optimizer = core.clone_training_state(model, optimizer)
    replay, replay_optimizer = core.clone_training_state(model, optimizer)
    for clone, opt in ((sequential, sequential_optimizer), (replay, replay_optimizer)):
        if not same_state(shared_state, clone.state_dict()) or not same_state(optimizer.state_dict(), opt.state_dict()):
            raise ValueError("Branch differs from shared first-stage state")
    memory = core.Reservoir(settings["memory_bytes"], seed)
    memory.update_after_stage(x, y)
    logs = {arm: [{**first_log, "shared_physical_fit": True}] for arm in ARMS}
    trajectory = {arm: ["initial", "shared"] for arm in ARMS}
    first_memory_bytes = sum(map(len, memory.serialized_parts()))
    del raw, x, y
    for stage in range(2, len(source.stages) + 1):
        raw, y = source.load(stage)
        x = prep.transform(raw)
        new_rows = len(y)
        sequential_log = fit(sequential, sequential_optimizer, x, y, seed, stage, settings)
        replay_log = fit(replay, replay_optimizer, x, y, seed, stage, settings, memory)
        if sequential_log["current_order_sha256"] != replay_log["current_order_sha256"]:
            raise ValueError("Incremental arms received different new-sample order")
        # Exactly one insertion opportunity per unique current edge, after all epochs.
        memory.update_after_stage(x, y)
        if memory.seen != sum(map(len, source.stages[:stage])) * source.pairs:
            raise ValueError("Reservoir arrival count differs")
        del raw, x, y
        for arm, head, opt, log, cache in (
                ("sequential", sequential, sequential_optimizer, sequential_log, None),
                ("er", replay, replay_optimizer, replay_log, memory)):
            name = f"{arm}_stage{stage}"
            points[name] = save_point(output / name, head, opt, prep, development, cache)
            trajectory[arm].append(name)
            logs[arm].append(log)
        raw, y = source.load(stage, cumulative=True)
        x = prep.transform(raw)  # Keep the identical FIRST-stage preprocessing.
        cumulative = core.make_head(settings["initialization_seed"])
        cumulative_optimizer = core.make_optimizer(cumulative)
        if not same_state(initial_state, cumulative.state_dict()) or cumulative_optimizer.state:
            raise ValueError("Cumulative reference did not reset to the original initialization")
        cumulative_log = fit(cumulative, cumulative_optimizer, x, y, seed, stage, settings, new_rows=new_rows)
        name = f"cumulative_stage{stage}"
        points[name] = save_point(output / name, cumulative, cumulative_optimizer, prep, development)
        trajectory["cumulative"].append(name)
        logs["cumulative"].append(cumulative_log)
        del raw, x, y, cumulative, cumulative_optimizer
        if not same_state(shared_state, model.state_dict()):
            raise ValueError("Frozen model was changed by another arm")
        trajectory["frozen"].append("shared")
        logs["frozen"].append({"updates": 0, "current_presentations": 0, "replay_presentations": 0,
                                "new_arrival_presentations": 0, "prior_arrival_presentations": 0,
                                "fit_seconds": 0.0, "input_array_bytes": 0})
    return {"seed": seed, "points": points, "trajectory": trajectory, "training": logs,
            "stage_accesses": source.accesses, "first_memory_bytes": first_memory_bytes,
            "preprocessing_fitted_stage": 1, "cumulative_reset_each_stage": True,
            "frozen_state_unchanged": True, "retained_account_coverage": None,
            "account_coverage_note": "ER stores no endpoint identifiers; no inferred coverage is reported."}


def runtime_check(output: Path) -> dict[str, Any]:
    """New orchestration only; reuse the completed core gradient check."""
    torch = core.torch_module()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = {"epochs": 2, "batch_size": 8, "memory_bytes": 2048, "initialization_seed": 20260907}
    with tempfile.TemporaryDirectory(prefix="continual_orchestration_") as scratch:
        root = Path(scratch)
        rng = np.random.default_rng(913)
        matrix = rng.normal(size=(30, 57))
        matrix[:, 0] = np.repeat(np.arange(5), 6)
        labels = (np.arange(30) % 3 == 0).astype(np.uint8)
        np.save(root / "base.npy", matrix[:, :24], allow_pickle=False)
        np.save(root / "identity.npy", matrix[:, 24:], allow_pickle=False)
        np.save(root / "labels.npy", labels, allow_pickle=False)
        stages = [[w] for w in [3, 0, 4, 1, 2]]
        source = data.StageSource(root / "base.npy", root / "identity.npy", root / "labels.npy", stages, 6, 5)
        result = run_order(source, rng.normal(size=(18, 57)), 11, root / "order11", settings)
        if len(result["points"]) != 14 or result["trajectory"]["frozen"] != ["initial"] + ["shared"] * 5:
            raise ValueError("Fixture trajectory/shared-score reuse differs")
        expected_accesses = [(1, "current", [3])]
        for stage in range(2, 6):
            expected_accesses += [(stage, "current", stages[stage - 1]),
                                  (stage, "arrived_prefix", [w for g in stages[:stage] for w in g])]
        actual = [(r["stage"], r["scope"], r["world_ordinals"]) for r in source.accesses]
        if actual != expected_accesses:
            raise ValueError("Fixture current/prefix data boundary differs")
        if result["points"]["er_stage5"]["memory"]["seen_unique_rows"] != 30:
            raise ValueError("Fixture repeated epochs inserted memory more than once")
        for stage in range(1, 6):
            for arm in ("sequential", "er", "cumulative"):
                rows = 6 * stage if arm == "cumulative" else 6
                log = result["training"][arm][stage - 1]
                if log["updates"] != 2 * math.ceil(rows / 8):
                    raise ValueError("Fixture cumulative/current update budget differs")
        verification = {"points_reloaded": len(result["points"]), "stage_accesses": source.accesses,
                        "trajectory": result["trajectory"], "training": result["training"],
                        "final_memory": result["points"]["er_stage5"]["memory"]}
    result = {"status": "PASSED_REAL_CPU_ORCHESTRATION_CHECK", "code": data.code_records(),
              "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "torch_threads": 1,
              "project_label_reads": 0, "verification": verification}
    data.write_json(output, result)
    return result


def run(output: Path, check_result: Path) -> dict[str, Any]:
    policy = data.load_policy()
    checked = data.read_json(check_result)
    if checked["status"] != "PASSED_REAL_CPU_ORCHESTRATION_CHECK" or checked["code"] != data.code_records():
        raise ValueError("Necessary orchestration verification is missing or stale")
    arrival = data.verify_public(policy)
    torch = core.torch_module()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if checked["torch"] != torch.__version__ or checked["numpy"] != np.__version__:
        raise ValueError("Checked numerical environment differs")
    output.mkdir()  # An existing result is never overwritten or resumed silently.
    start = time.perf_counter()
    files = policy["public_files"]
    # Benchmark packaging reads TRAIN labels once, outside every method. Only
    # selected stage rows are exposed thereafter. The temporary tensor is not
    # retained history, is never published, and is removed on every exit path.
    with tempfile.TemporaryDirectory(prefix="continual_training_labels_") as scratch:
        labels = data.aligned_labels(data.verify(policy["train_labels"]),
                                     data.ROOT / files["train_rows"]["path"], "train", 500, 28, 20)
        label_path = Path(scratch) / "labels.npy"
        np.save(label_path, labels, allow_pickle=False)
        del labels
        label_pack = data.record(label_path, label_path.parent)
        development = np.concatenate((np.load(data.ROOT / files["development_base"]["path"], allow_pickle=False),
                                      np.load(data.ROOT / files["development_identity"]["path"], allow_pickle=False)), axis=1)
        if development.shape != (189000, 57) or not np.isfinite(development).all():
            raise ValueError("Fixed development public features differ")
        orders = []
        for order in arrival["orders"]:
            source = data.StageSource(data.ROOT / files["train_base"]["path"],
                                      data.ROOT / files["train_identity"]["path"], label_path,
                                      order["train_world_ordinals_by_stage"], 378, 500)
            orders.append(run_order(source, development, order["seed"], output / f"order{order['seed']}", policy))
        del development
    # Only this final manifest qualifies the full set for development evaluation.
    import resource  # This command is explicitly a Linux CPU execution stage.
    result = {"status": "ALL_FOUR_ARM_SCORES_SAVED_AND_REPLAYED_NO_DEVELOPMENT_LABELS",
              "code": data.code_records(), "policy": policy, "orders": orders,
              "runtime_check": data.record(check_result), "temporary_training_label_pack": label_pack,
              "label_reads": {"train_csv_offline_packaging": 1, "development": 0, "audit_a": 0, "audit_b": 0},
              "elapsed_seconds": time.perf_counter() - start,
              "whole_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "environment": {"torch": torch.__version__, "numpy": np.__version__,
                              "platform": platform.platform(), "device": "cpu", "torch_threads": 1},
              "limits": ["Random same-source arrival, not five distinct domains or a real chronology",
                         "Fixed-budget training calls receive no cumulative-arm historical input",
                         "No development evaluation has run; no forgetting or method advantage claimed"]}
    data.write_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-result", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing check or run")
    if args.command == "run" and args.check_result is None:
        parser.error("run requires --check-result")
    result = runtime_check(args.output) if args.command == "check" else run(args.output, args.check_result)
    print(result["status"])


if __name__ == "__main__":
    main()
