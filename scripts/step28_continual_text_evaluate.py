"""Complete text-run qualification before independent development-label evaluation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import step28_continual_data as shared
import step28_continual_evaluate as metrics
import step28_continual_text_data as data
from step28_continual_text_run import ARMS, expected_counts, trajectory


def validate_memory(value: dict, stage: int, arrived_rows: int, budget: int) -> None:
    if (value["last_stage"] != stage or value["seen_unique_rows"] != arrived_rows
            or value["budget"] != budget or not 1024 < value["total_bytes"] <= budget
            or not 0 <= value["positive_records"] <= value["records"] <= arrived_rows
            or not 2 <= value["retained_accounts"] <= 2 * value["records"]
            or len(value["sha256"]) != 64):
        raise ValueError("Historical byte budget or arrival boundary differs")


def validate_order(order: dict, base: Path, *, stages: int = 5, worlds_per_stage: int = 100,
                   development_worlds: int = 500, memory_budget: int = 524288,
                   arrival_stages: list[list[int]]) -> None:
    expected_trajectory = trajectory(stages)
    required = {name for names in expected_trajectory.values() for name in names}
    retained = {"shared"} | {f"{arm}_stage{stages}" for arm in ARMS[1:]}
    if (set(order["points"]) != required or order["trajectory"] != expected_trajectory
            or order["first_stage_shared_model_and_adam"] is not True
            or order["cumulative_resets_original_and_empty_adam"] is not True
            or order["frozen_artifact_unchanged"] is not True):
        raise ValueError("Incomplete points, sharing, reset or trajectory")
    accesses = [{"arm": "shared", "stage": 1, "scope": "current",
                 "world_ordinals": arrival_stages[0], "rows": worlds_per_stage * 378}]
    for arm in ARMS[1:]:
        for stage in range(2, stages + 1):
            worlds = ([w for group in arrival_stages[:stage] for w in group]
                      if arm == "cumulative" else arrival_stages[stage - 1])
            accesses.append({"arm": arm, "stage": stage, "scope": "arrived_prefix" if arm == "cumulative" else "current",
                             "world_ordinals": worlds, "rows": len(worlds) * 378})
    if order["stage_accesses"] != accesses:
        raise ValueError("Current/prefix access differs from frozen arrival")
    validate_memory(order["first_memory"], 1, worlds_per_stage * 378, memory_budget)
    if order["points"]["shared"]["memory"] != order["first_memory"]:
        raise ValueError("First history snapshot differs")
    for arm in ARMS:
        logs = order["training"][arm]
        if len(logs) != stages or logs[0] != order["training"]["frozen"][0] or logs[0]["shared_physical_fit"] is not True:
            raise ValueError("Shared training log differs")
        for stage, log in enumerate(logs, 1):
            frozen = arm == "frozen" and stage > 1
            count = worlds_per_stage * (stage if arm == "cumulative" else 1)
            old = log.get("history_before_training")
            if arm == "er" and stage > 1:
                previous = order["first_memory"] if stage == 2 else order["points"][f"er_stage{stage - 1}"]["memory"]
                if old != previous:
                    raise ValueError("ER did not start with exactly the previous retained memory/RNG")
            elif old is not None:
                raise ValueError("Unexpected historical method input")
            expected = expected_counts(0 if frozen else count, 2, old["records"] if old is not None else 0)
            if any(log[k] != v for k, v in expected.items()):
                raise ValueError("World/update/presentation budget differs")
            new_rows = 0 if frozen else worlds_per_stage * 2 * 378
            prior_rows = (stage - 1) * worlds_per_stage * 2 * 378 if arm == "cumulative" else 0
            if (log["new_arrival_presentations"], log["prior_arrival_presentations"]) != (new_rows, prior_rows):
                raise ValueError("New/prior arrival cost accounting differs")
            if not frozen and (not np.isfinite(log["mean_update_loss"]) or log["mean_update_loss"] < 0):
                raise ValueError("Invalid training loss record")
            if arm in ("er", "sequential") and log["current_order_sha256"] != order["training"]["sequential"][stage - 1]["current_order_sha256"]:
                raise ValueError("Incremental current ordering differs")
    expected_updates = 2 * worlds_per_stage * (1 + 2 * (stages - 1) + sum(range(2, stages + 1)))
    if order["physical_updates"] != expected_updates:
        raise ValueError("Physical updates count repeated shared fits or omits training")
    with np.load(shared.verify(order["scaler"], base), allow_pickle=False) as prep:
        if set(prep.files) != {"medians", "means", "scales"} or any(
                prep[k].shape != (57,) or not np.isfinite(prep[k]).all() for k in prep.files) or np.any(prep["scales"] <= 0):
            raise ValueError("Invalid first-stage scaler snapshot")
    for name, point in order["points"].items():
        if point != shared.read_json(base / name / "point.json"):
            raise ValueError("Point receipt differs from final manifest")
        stage = 0 if name == "initial" else 1 if name == "shared" else int(name.rsplit("stage", 1)[1])
        arm = 0 if stage < 2 else ARMS.index(name.split("_stage", 1)[0])
        if (point["progress"] != {"order_seed": order["seed"], "stage": stage, "arm": arm}
                or point["full_state_and_all_scores_exact"] is not True
                or point["temporary_state_removed"] is not True
                or point["score_passes"] != 2 or point["scored_world_presentations"] != development_worlds * 2
                or len(point["state_signature"]) != 64
                or len(point["temporary_state"]["sha256"]) != 64 or point["temporary_state"]["size_bytes"] <= 0):
            raise ValueError("Missing exact state/score publication evidence")
        if name.startswith("er_stage"):
            validate_memory(point["memory"], stage, stage * worlds_per_stage * 378, memory_budget)
        elif name != "shared" and point["memory"] is not None:
            raise ValueError("Unexpected cache in a non-ER point")
        model = point["inference_model"]
        if (model is not None) != (name in retained):
            raise ValueError("Inference retention differs")
        if model is not None:
            if model["path"] != f"{name}/model.pt":
                raise ValueError("Model-to-point path mismatch")
            shared.verify(model, base)
        if point["scores"]["path"] != f"{name}/logits.npy":
            raise ValueError("Score-to-point path mismatch")
        scores = np.load(shared.verify(point["scores"], base), allow_pickle=False)
        if scores.shape != (development_worlds * 378,) or scores.dtype != np.dtype("float32") or not np.isfinite(scores).all():
            raise ValueError("Invalid complete score array")


def validate_run(run: Path, settings: dict) -> dict:
    manifest = shared.read_json(run / "manifest.json")
    if (manifest["status"] != "ALL_TEXT_FOUR_ARM_SCORES_SAVED_AND_REPLAYED_NO_DEVELOPMENT_LABELS"
            or manifest["settings"] != settings or manifest["code"] != data.code_records()
            or manifest["label_reads"] != {"train_csv_offline_packaging": 1, "development": 0, "qrels": 0, "audit_a": 0, "audit_b": 0}
            or manifest["temporary_work_removed"] is not True or manifest["physical_updates"] != 13800
            or [o["seed"] for o in manifest["orders"]] != [11, 23, 37]):
        raise ValueError("Full text-run qualification is absent or stale")
    arrival = shared.read_json(data.ROOT / settings["benchmark"]["arrival"]["path"])
    for order, sequence in zip(manifest["orders"], arrival["orders"], strict=True):
        base = run / f"order{order['seed']}"
        if order != shared.read_json(base / "order.json"):
            raise ValueError("Order receipt differs from manifest")
        validate_order(order, base, arrival_stages=sequence["train_world_ordinals_by_stage"])
    return manifest


def summarize(manifest: dict, arrays: dict[str, np.ndarray], columns: list[str],
              *, bootstrap_seed: int, replicates: int) -> tuple[dict, dict, dict]:
    """Same world-paired AP contrasts as the fixed-feature comparison."""
    world_count = next(iter(arrays.values())).shape[0]
    indices = np.random.default_rng(bootstrap_seed).integers(0, world_count, (replicates, world_count), dtype=np.int32)
    def contrast(target: str, after: int, control: str, before: int) -> dict:
        deltas = []
        for order in manifest["orders"]:
            prefix, path = f"order{order['seed']}_", order["trajectory"]
            deltas.append(arrays[prefix + path[target][after]] - arrays[prefix + path[control][before]])
        values = np.stack(deltas)
        return {key: metrics.paired_interval(values[:, :, columns.index(key)], indices)
                for key in ("average_precision", "no_strong_average_precision")}
    comparisons = {target + "_minus_" + control: contrast(target, 5, control, 5)
                   for target, control in (("er", "sequential"), ("sequential", "frozen"),
                                           ("cumulative", "sequential"), ("er", "cumulative"))}
    changes = {arm: {f"stage{after}_minus_stage{before}": contrast(arm, after, arm, before)
                    for before, after in ((0, 1), (1, 5), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5))} for arm in ARMS}
    bootstrap = {"seed": bootstrap_seed, "replicates": replicates,
                 "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
    return comparisons, changes, bootstrap


def evaluate(run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Never overwrite evaluation evidence")
    settings = data.load_settings()
    data.verify_public(settings)
    manifest = validate_run(run, settings)  # All 42 points BEFORE any development truth.
    benchmark = settings["benchmark"]
    y = shared.aligned_labels(shared.verify(benchmark["development_labels"]),
                              data.ROOT / benchmark["public_files"]["development_rows"]["path"],
                              "development", 500, 28, 20).reshape(500, 378)
    identity = np.load(data.ROOT / benchmark["public_files"]["development_identity"]["path"], allow_pickle=False)
    no_strong = ~((identity[:, 0] > 0) | (identity[:, 19] > 0)).reshape(500, 378)
    del identity
    output.mkdir()
    summaries, arrays, curves = {}, {}, {}
    for order in manifest["orders"]:
        for name, point in order["points"].items():
            key = f"order{order['seed']}_{name}"
            path = run / f"order{order['seed']}" / point["scores"]["path"]
            values = np.load(path, allow_pickle=False).reshape(500, 378)
            summary, per_world = metrics.point_metrics(y, values, no_strong, 28)
            dest = output / (key + ".npy")
            np.save(dest, per_world, allow_pickle=False)
            np.testing.assert_array_equal(per_world, np.load(dest, allow_pickle=False))
            summary["world_metrics"] = shared.record(dest, output)
            summaries[key], arrays[key] = summary, per_world
        curves[str(order["seed"])] = {arm: [summaries[f"order{order['seed']}_{name}"]["world_equal"]
                                                 for name in order["trajectory"][arm]] for arm in ARMS}
    columns = next(iter(summaries.values()))["world_metric_columns"]
    contrasts, changes, bootstrap = summarize(manifest, arrays, columns,
            bootstrap_seed=settings["run"]["bootstrap_seed"], replicates=settings["run"]["bootstrap_replicates"])
    if data.code_records() != manifest["code"]:
        raise ValueError("Code or policy changed during development evaluation")
    result = {"status": "EXPLORATORY_TEXT_CONTINUAL_DEVELOPMENT_EVALUATION_COMPLETE",
              "run_manifest": shared.record(run / "manifest.json"), "code": manifest["code"],
              "unique_score_arrays": len(summaries), "points": summaries, "stage_trajectories": curves,
              "final_paired_comparisons": contrasts, "fixed_development_changes": changes, "bootstrap": bootstrap,
              "label_reads": {"development": 1, "train": 0, "qrels": 0, "audit_a": 0, "audit_b": 0},
              "domain_BWT_or_forgetting": None,
              "limits": ["Fixed consumed development population, random same-source arrivals; exploratory only",
                         "Intervals cover overall and rule-excluded AP; other metrics are descriptive",
                         "Within the new text representation only; no sole-unfreezing attribution",
                         "No real-market validity, established forgetting or automatic favorable rerun"]}
    shared.write_json(output / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(args.run, args.output)["status"])


if __name__ == "__main__":
    main()
