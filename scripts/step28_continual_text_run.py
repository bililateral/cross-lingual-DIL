"""One-GPU, four-arm continual text run; no development truth in training."""
from __future__ import annotations

import argparse
import hashlib
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_data as shared
import step28_continual_text as core
import step28_continual_text_data as data
from step28_continual_text_check import digest_tensors

ARMS = ("frozen", "sequential", "er", "cumulative")


def trajectory(stages: int = 5) -> dict[str, list[str]]:
    return {arm: ["initial", "shared"] + (["shared"] * (stages - 1) if arm == "frozen"
                else [f"{arm}_stage{s}" for s in range(2, stages + 1)]) for arm in ARMS}


def expected_counts(worlds: int, epochs: int = 2, history_edges: int = 0) -> dict[str, int]:
    return {"updates": worlds * epochs, "current_presentations": worlds * epochs * 378,
            "replay_presentations": worlds * epochs * min(16, history_edges)}


def memory_record(memory: core.ByteMemory | None) -> dict | None:
    if memory is None:
        return None
    payload = memory.to_bytes()
    batch = core.batch_from_payloads([e[2] for e in memory.entries])
    return {"seen_unique_rows": memory.seen, "last_stage": memory.last_stage,
            "records": len(memory.entries), "positive_records": int(batch.labels.sum()),
            "retained_accounts": len(batch.accounts), "total_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "budget": memory.budget}


class Trainer:
    """A single live encoder/head/Adam instance, reused by explicit state reloads."""

    def __init__(self, encoder: Any, tokenizer: Any, *, bf16: bool = True):
        self.encoder, self.tokenizer = encoder, tokenizer
        self.head = core.make_head().to(next(encoder.parameters()).device)
        self.optimizer = core.make_text_optimizer(encoder, self.head)
        self.scaler: core.Preprocessor | None = None
        self.precision = {"bf16": bf16}

    def signature(self) -> str:
        return digest_tensors([self.encoder.state_dict(), self.head.state_dict(), self.optimizer.state_dict()])

    def save(self, path: Path, progress: dict, memory: core.ByteMemory | None = None) -> None:
        core.save_state(path, self.encoder, self.head, self.optimizer, self.scaler, progress, memory)

    def restore(self, path: Path) -> tuple[dict, core.ByteMemory | None]:
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()  # No second model or old Adam GPU allocation.
        self.scaler, progress, memory = core.restore_state(path, self.encoder, self.head, self.optimizer)
        return progress, memory

    def scores(self, development: data.WorldArchive) -> np.ndarray:
        result = np.concatenate([core.score_batch(self.encoder, self.head, self.tokenizer,
                                  batch, self.scaler, **self.precision) for batch in development.all_worlds()])
        if (result.shape != (development.world_count * development.pairs,)
                or result.dtype != np.dtype("float32") or not np.isfinite(result).all()):
            raise ValueError("Complete development scores differ in shape/dtype")
        return result

    def save_inference(self, path: Path) -> None:
        torch = core.torch_module()
        payload = {"encoder": self.encoder.state_dict(), "head": self.head.state_dict(),
                   "scaler": {k: torch.from_numpy(getattr(self.scaler, k).copy())
                              for k in ("medians", "means", "scales")},
                   "policy_sha256": hashlib.sha256(core.POLICY_PATH.read_bytes()).hexdigest()}
        expected = digest_tensors(payload)
        with path.open("xb") as handle:
            torch.save(payload, handle)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if digest_tensors(loaded) != expected:
            raise ValueError("Published inference weights/scaler differ on disk")
        # Exercise the actual inference reload; leave the already restored Adam intact.
        self.encoder.load_state_dict(loaded["encoder"], strict=True)
        self.head.load_state_dict(loaded["head"], strict=True)
        self.scaler = core.Preprocessor(**{k: v.numpy().copy() for k, v in loaded["scaler"].items()})


def save_point(trainer: Trainer, directory: Path, development: data.WorldArchive,
               scratch: Path, progress: dict, *, retain_model: bool,
               memory: core.ByteMemory | None = None) -> dict:
    """Exact full-score disk replay before deleting a temporary full training state."""
    started = time.perf_counter()
    directory.mkdir()
    print(f"point={directory.name} full_development_replay_started", flush=True)
    signature = trainer.signature()
    scaler = {k: getattr(trainer.scaler, k).copy() for k in ("medians", "means", "scales")}
    memory_bytes = memory.to_bytes() if memory is not None else None
    before = trainer.scores(development)
    with tempfile.TemporaryDirectory(prefix="point_", dir=scratch) as temporary:
        checkpoint = Path(temporary) / "state.pt"
        trainer.save(checkpoint, progress, memory)
        state_receipt = shared.record(checkpoint, checkpoint.parent)
        restored_progress, restored_memory = trainer.restore(checkpoint)
        if (trainer.signature() != signature or restored_progress != progress
                or (restored_memory.to_bytes() if restored_memory is not None else None) != memory_bytes):
            raise ValueError("Full model/Adam/progress/history restore differs")
        for key, expected in scaler.items():
            np.testing.assert_array_equal(getattr(trainer.scaler, key), expected)
        if restored_memory is not None:
            left = core.ByteMemory.from_bytes(memory_bytes).sample(16)
            right = restored_memory.sample(16)
            if [core.edge_payload(left, i) for i in range(len(left.edges))] != [
                    core.edge_payload(right, i) for i in range(len(right.edges))]:
                raise ValueError("Restored next replay draw differs")
        model = None
        if retain_model:
            path = directory / "model.pt"
            trainer.save_inference(path)
            model = shared.record(path, directory.parent)
        after = trainer.scores(development)  # Also after inference reload when retained.
        np.testing.assert_array_equal(before, after)
    score_path = directory / "logits.npy"
    np.save(score_path, before, allow_pickle=False)
    np.testing.assert_array_equal(before, np.load(score_path, allow_pickle=False))
    receipt = {"scores": shared.record(score_path, directory.parent), "inference_model": model,
               "state_signature": signature, "temporary_state": state_receipt,
               "temporary_state_removed": True, "progress": progress,
               "full_state_and_all_scores_exact": True, "memory": memory_record(memory),
               "score_passes": 2, "scored_world_presentations": 2 * development.world_count,
               "publication_seconds": time.perf_counter() - started}
    shared.write_json(directory / "point.json", receipt)
    return receipt


def fit(trainer: Trainer, worlds: list[core.PairBatch], seed: int, stage: int,
        *, memory: core.ByteMemory | None = None, new_worlds: int | None = None) -> dict:
    started = time.perf_counter()
    before = memory_record(memory)
    result = core.train_stage(trainer.encoder, trainer.head, trainer.optimizer, trainer.tokenizer,
                              worlds, trainer.scaler, order_seed=seed, stage=stage,
                              memory=memory, epochs=2, **trainer.precision)
    expected = expected_counts(len(worlds), 2, len(memory.entries) if memory is not None else 0)
    if any(result[k] != v for k, v in expected.items()):
        raise ValueError("Actual world/update/replay budget differs")
    new_worlds = len(worlds) if new_worlds is None else new_worlds
    result.update({"new_arrival_presentations": new_worlds * 2 * 378,
                   "prior_arrival_presentations": (len(worlds) - new_worlds) * 2 * 378,
                   "history_before_training": before, "fit_seconds": time.perf_counter() - started})
    return result


def run_order(trainer: Trainer, source: data.StageSource, development: data.WorldArchive,
              seed: int, output: Path, scratch: Path, *, memory_budget: int = 524288) -> dict:
    output.mkdir()
    first = source.load("shared", 1)
    original_signature = trainer.signature()
    trainer.scaler = core.fit_first_scaler(trainer.encoder, trainer.tokenizer, first, **trainer.precision)
    if trainer.signature() != original_signature or trainer.optimizer.state:
        raise ValueError("Initial scaling trained the model or optimizer was not empty")
    original = scratch / "original.pt"
    common = scratch / "shared.pt"
    trainer.save(original, {"order_seed": seed, "stage": 0})
    prep_path = output / "scaler.npz"
    fixed_scaler = {k: getattr(trainer.scaler, k).copy() for k in ("medians", "means", "scales")}
    np.savez(prep_path, **fixed_scaler)
    with np.load(prep_path, allow_pickle=False) as saved_scaler:
        for key, expected in fixed_scaler.items():
            np.testing.assert_array_equal(saved_scaler[key], expected)
    publish = lambda name, stage, arm, keep, memory=None: save_point(
        trainer, output / name, development, scratch,
        {"order_seed": seed, "stage": stage, "arm": arm}, retain_model=keep, memory=memory)
    points = {"initial": publish("initial", 0, 0, False)}
    print(f"order={seed} arm=shared stage=1 worlds={len(first)} fit_started", flush=True)
    first_log = fit(trainer, first, seed, 1)
    if trainer.signature() == original_signature or not trainer.optimizer.state:
        raise ValueError("Shared first stage did not produce a nonempty updated state")
    trainer.save(common, {"order_seed": seed, "stage": 1})
    cache = core.ByteMemory(memory_budget, seed)
    cache.update_after_stage(first, 1)
    first_memory = memory_record(cache)
    memory_path = scratch / "first_memory.bin"
    memory_path.write_bytes(cache.to_bytes())
    points["shared"] = publish("shared", 1, 0, True, cache)
    shared_signature = trainer.signature()
    del first, cache
    logs = {arm: [{**first_log, "shared_physical_fit": True}] for arm in ARMS}
    stages = len(source.stages)
    for arm in ARMS[1:]:
        cache = None
        if arm != "cumulative":
            progress, unwanted = trainer.restore(common)
            if progress != {"order_seed": seed, "stage": 1} or unwanted is not None or trainer.signature() != shared_signature:
                raise ValueError("Incremental arm did not inherit shared model and Adam")
            for key, expected in fixed_scaler.items():
                np.testing.assert_array_equal(getattr(trainer.scaler, key), expected)
            if arm == "er":
                cache = core.ByteMemory.from_bytes(memory_path.read_bytes())
                memory_path.unlink()  # No separate initial-history backup during ER.
        for stage in range(2, stages + 1):
            if arm == "cumulative":
                progress, unwanted = trainer.restore(original)
                if (progress != {"order_seed": seed, "stage": 0} or unwanted is not None
                        or trainer.signature() != original_signature or trainer.optimizer.state):
                    raise ValueError("Cumulative model/Adam were not reset to original")
                for key, expected in fixed_scaler.items():
                    np.testing.assert_array_equal(getattr(trainer.scaler, key), expected)
            worlds = source.load(arm, stage)
            print(f"order={seed} arm={arm} stage={stage} worlds={len(worlds)} fit_started", flush=True)
            log = fit(trainer, worlds, seed, stage, memory=cache,
                      new_worlds=len(source.stages[stage - 1]))
            if arm == "er":
                if log["current_order_sha256"] != logs["sequential"][stage - 1]["current_order_sha256"]:
                    raise ValueError("Incremental current-world permutations differ")
                cache.update_after_stage(worlds, stage)
                if cache.seen != sum(map(len, source.stages[:stage])) * 378:
                    raise ValueError("Unique arrival/insertion count differs")
            del worlds
            name = f"{arm}_stage{stage}"
            points[name] = publish(name, stage, ARMS.index(arm), stage == stages, cache)
            logs[arm].append(log)
            print(f"order={seed} arm={arm} stage={stage} updates={log['updates']} published", flush=True)
        del cache
    logs["frozen"].extend({"updates": 0, "current_presentations": 0, "replay_presentations": 0,
                            "new_arrival_presentations": 0, "prior_arrival_presentations": 0}
                           for _ in range(2, stages + 1))
    # Frozen is the immutable shared inference artifact, never another mutable GPU object.
    shared.verify(points["shared"]["inference_model"], output)
    physical = first_log["updates"] + sum(log["updates"] for arm in ARMS[1:] for log in logs[arm][1:])
    result = {"seed": seed, "points": points, "trajectory": trajectory(stages), "training": logs,
              "physical_updates": physical, "first_memory": first_memory, "stage_accesses": source.accesses,
              "scaler": shared.record(prep_path, output), "first_stage_shared_model_and_adam": True,
              "cumulative_resets_original_and_empty_adam": True, "frozen_artifact_unchanged": True}
    shared.write_json(output / "order.json", result)
    return result


def runtime_check(output: Path) -> dict:
    """Only new production orchestration/publication on hand-created CPU fixtures."""
    code = data.code_records()
    import sys
    sys.path.insert(0, str(data.ROOT / "tests"))
    import test_step28_continual_text_run_contracts as fixtures
    torch = core.torch_module()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix="text_orchestration_") as temporary:
        root = Path(temporary)
        train, development = fixtures.make_archives(root, training_worlds=5, sellers=28)
        trainer = Trainer(fixtures.toy_encoder(), fixtures.CharacterTokenizer(), bf16=False)
        source = data.StageSource(train, [[3], [0], [4], [1], [2]])
        scratch = root / "scratch"
        scratch.mkdir()
        result = run_order(trainer, source, development, 11, root / "order11", scratch, memory_budget=8192)
        fixtures.assert_order_result(result, root / "order11", stages=5, worlds_per_stage=1,
                                     development_worlds=2, memory_budget=8192)
        verification = {"physical_updates": result["physical_updates"], "points_reloaded": len(result["points"]),
                        "retained_inference_models": sum(p["inference_model"] is not None for p in result["points"].values()),
                        "stage_accesses": source.accesses, "trajectory": result["trajectory"],
                        "last_memory": result["points"]["er_stage5"]["memory"]}
    if data.code_records() != code:
        raise ValueError("Code or policy changed during orchestration verification")
    record = {"status": "PASSED_REAL_CPU_TEXT_ORCHESTRATION_CHECK", "code": code,
              "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": 1,
                              "python": platform.python_version(), "platform": platform.platform()},
              "project_label_reads": 0, "verification": verification}
    shared.write_json(output, record)
    return record


def run(output: Path, check_result: Path) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("Formal text training requires the separately authorized Linux GPU stage")
    settings = data.load_settings()
    code = data.code_records()
    checked = shared.read_json(check_result)
    if checked["status"] != "PASSED_REAL_CPU_TEXT_ORCHESTRATION_CHECK" or checked["code"] != code:
        raise ValueError("New orchestration verification is missing or stale")
    arrival = data.verify_public(settings)
    torch = core.torch_module()
    torch.set_num_threads(1)
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.mem_get_info()[0] < 24 * 1024**3):
        raise RuntimeError("Wait for an available BF16 GPU with at least 24 GiB free")
    if (checked["environment"]["torch"], checked["environment"]["numpy"]) != (torch.__version__, np.__version__):
        raise ValueError("Checked numerical environment differs")
    packages = {name: version(name) for name in ("numpy", "transformers", "sentence-transformers")}
    component_environment = shared.read_json(data.ROOT / settings["run"]["component_runtime"]["path"])["environment"]
    if (component_environment["torch"] != torch.__version__ or component_environment["packages"] != packages
            or component_environment["cuda"] != torch.version.cuda or component_environment["cudnn"] != torch.backends.cudnn.version()):
        raise ValueError("Encoder software differs from the reused real component check")
    output.mkdir()
    start = time.perf_counter()
    benchmark = settings["benchmark"]
    try:
        with tempfile.TemporaryDirectory(prefix="text_work_") as temporary:
            work = Path(temporary)
            labels = shared.aligned_labels(shared.verify(benchmark["train_labels"]),
                        data.ROOT / benchmark["public_files"]["train_rows"]["path"], "train", 500, 28, 20)
            label_path = work / "labels.npy"
            np.save(label_path, labels, allow_pickle=False)
            del labels
            label_pack = shared.record(label_path, work)
            train, development = data.archive(settings, "train", label_path), data.archive(settings, "development")
            data.disjoint_archives(train, development)
            orders = []
            for arrival_order in arrival["orders"]:
                seed = arrival_order["seed"]
                encoder, tokenizer = core.load_labse()
                trainer = Trainer(encoder, tokenizer)
                source = data.StageSource(train, arrival_order["train_world_ordinals_by_stage"])
                with tempfile.TemporaryDirectory(prefix=f"order{seed}_", dir=work) as scratch:
                    orders.append(run_order(trainer, source, development, seed,
                                            output / f"order{seed}", Path(scratch)))
                del trainer, encoder, tokenizer, source
                torch.cuda.empty_cache()
            if sum(order["physical_updates"] for order in orders) != 13800:
                raise ValueError("Formal physical update total differs")
            if data.code_records() != code:
                raise ValueError("Code or policy changed during the formal run")
        import resource
        result = {"status": "ALL_TEXT_FOUR_ARM_SCORES_SAVED_AND_REPLAYED_NO_DEVELOPMENT_LABELS",
                  "settings": settings, "code": code, "orders": orders,
                  "runtime_check": shared.record(check_result), "temporary_training_label_pack": label_pack,
                  "label_reads": {"train_csv_offline_packaging": 1, "development": 0, "qrels": 0, "audit_a": 0, "audit_b": 0},
                  "temporary_work_removed": True, "physical_updates": 13800,
                  "elapsed_seconds": time.perf_counter() - start,
                  "whole_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "environment": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                                  "gpu": torch.cuda.get_device_name(), "platform": platform.platform(),
                                  "python": platform.python_version(), "cudnn": torch.backends.cudnn.version(),
                                  "packages": packages, "torch_threads": 1}}
        shared.write_json(output / "manifest.json", result)
        return result
    except Exception as exc:
        shared.write_json(output / "failure.json", {"status": "INCOMPLETE_NO_DEVELOPMENT_EVALUATION",
                          "exception": f"{type(exc).__name__}: {exc}", "code": code,
                          "observed_code_at_failure": data.code_records(),
                          "temporary_work_removed": True, "automatic_restart": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-result", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite a check or run")
    if args.command == "run" and args.check_result is None:
        parser.error("run requires --check-result")
    result = runtime_check(args.output) if args.command == "check" else run(args.output, args.check_result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
