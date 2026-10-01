"""One necessary py310/GPU check of the NEW text/replay training semantics.

No formal comparison or project label input. The capacity fixture uses three
public train worlds with expressly hand-created numeric values and binary labels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import platform
import sys
import tempfile
import time
import unittest
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_text as core
import step28_labse_finetune_common as common

ROOT = Path(__file__).resolve().parents[1]


def digest_tensors(value: Any) -> str:
    """Hash every model/Adam tensor in bounded CPU chunks, including names/shapes."""
    torch = core.torch_module()
    digest = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            digest.update(core.json_bytes([str(obj.dtype), list(obj.shape)]))
            for chunk in obj.detach().reshape(-1).split(1048576):
                digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                digest.update(core.json_bytes(str(key)))
                visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            digest.update(core.json_bytes(len(obj)))
            for item in obj:
                visit(item)
        else:
            digest.update(core.json_bytes(obj))
    visit(value)
    return digest.hexdigest()


def gradient_norm(module: Any) -> dict[str, float | int]:
    torch = core.torch_module()
    squared, count, absent = 0., 0, 0
    for parameter in module.parameters():
        if parameter.grad is None:
            absent += parameter.numel()
            continue
        for chunk in parameter.grad.detach().reshape(-1).split(1048576):
            if not torch.isfinite(chunk).all():
                raise ValueError("Nonfinite independent branch gradient")
            squared += float(torch.sum(chunk.double().square()))
        count += parameter.numel()
    if not np.isfinite(squared) or squared <= 0 or not count:
        raise ValueError("No nonzero finite DATA gradient in this encoder/head branch")
    return {"l2": float(np.sqrt(squared)), "parameters_with_gradient": count,
            "parameters_without_gradient": absent}


def public_capacity_batches() -> tuple[core.PairBatch, core.ByteMemory, dict]:
    policy = core.load_policy()
    spec = policy["public_texts"]["train"]
    path = ROOT / spec["path"]
    if path.stat().st_size != spec["size_bytes"] or common.sha256_file(path) != spec["sha256"]:
        raise ValueError("Public train text bytes differ")
    groups: dict[str, list[dict]] = {}
    for row in common.iter_jsonl(path):
        uid = row["world_uid"]
        if uid not in groups and len(groups) == 3:
            break
        groups.setdefault(uid, []).append(row)
    if len(groups) != 3 or any(len(rows) != 99 for rows in groups.values()):
        raise ValueError("First three public world layouts differ")
    batches = []
    for index, (uid, rows) in enumerate(groups.items()):
        accounts = common.build_redacted_text_index(rows, expected_worlds=1,
                                                    expected_sellers_per_world=28,
                                                    expected_items_per_world=99)[uid]
        edges = tuple(itertools.combinations(sorted(accounts), 2))
        # These fixtures are NOT the project's numeric features or true labels.
        numeric = np.random.default_rng(81 + index).normal(0, .3, (378, 51))
        labels = (np.arange(378) % 3 == 0).astype(np.uint8)
        batch = core.PairBatch(accounts, edges, numeric, labels)
        batch.validate(training=True, complete_world=True)
        batches.append(batch)
    payloads = []
    for batch, needed in ((batches[1], 14), (batches[2], 2)):
        uids = sorted(batch.accounts)
        for i in range(needed):
            edge = (uids[2 * i], uids[2 * i + 1])
            payloads.append(core.edge_payload(batch, batch.edges.index(edge)))
    memory = core.ByteMemory()
    memory.update_after_stage([core.batch_from_payloads(payloads)], 1)
    return batches[0], memory, {"public_worlds": list(groups), "items": 297,
                               "project_binary_labels_read": 0,
                               "numeric_and_labels": "hand-created verification values only"}


def run_check(output: Path) -> dict:
    torch = core.torch_module()
    if platform.system() != "Linux" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires the existing Linux py310 and available BF16 CUDA GPU")
    free, total = torch.cuda.mem_get_info(0)
    if free < 24 * 1024**3:
        raise RuntimeError("Less than 24 GiB GPU memory free; wait for shared-server availability")
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "tests"))
    import test_step28_continual_text_contracts as fixtures

    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(fixtures)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output / "contracts.log").write_text(log.getvalue(), encoding="utf-8")
    if not result.wasSuccessful() or result.skipped:
        raise ValueError("New focused CPU training contracts failed or skipped")
    print("New CPU gradient/state contracts passed; loading existing LaBSE", flush=True)
    encoder, tokenizer = core.load_labse()
    head = core.make_head().to("cuda:0")
    optimizer = core.make_text_optimizer(encoder, head)
    first, current = fixtures.example_batch(2, width=3), fixtures.example_batch(3, 100, 3)
    initial_encoder = digest_tensors(encoder.state_dict())
    scaler = core.fit_first_scaler(encoder, tokenizer, [first])
    if digest_tensors(encoder.state_dict()) != initial_encoder:
        raise ValueError("Initial eval scaling changed encoder parameters")
    memory = core.ByteMemory()
    memory.update_after_stage([first], 1)
    historical = core.batch_from_payloads([memory.entries[0][2]])
    branches = {}
    encoder.train()
    head.train()
    # The new activation-recomputation setting must preserve the real full gradient.
    encoder[0].auto_model.gradient_checkpointing_disable()
    optimizer.zero_grad(set_to_none=True)
    with core.paired_rng(torch.device("cuda:0"), 31):
        direct_loss = core.batch_loss(encoder, head, tokenizer, current, scaler)
        direct_loss.backward()
    direct_value = float(direct_loss.detach())
    direct_gradient = digest_tensors([p.grad for p in [*encoder.parameters(), *head.parameters()]])
    del direct_loss
    encoder[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    for name, batch, seed in (("current", current, 31), ("history", historical, 73)):
        optimizer.zero_grad(set_to_none=True)
        with core.paired_rng(torch.device("cuda:0"), seed):
            loss = core.batch_loss(encoder, head, tokenizer, batch, scaler)
            loss.backward()
        if name == "current" and (float(loss.detach()) != direct_value or digest_tensors(
                [p.grad for p in [*encoder.parameters(), *head.parameters()]]) != direct_gradient):
            raise ValueError("Activation checkpointing changed paired real loss/full gradients")
        branches[name] = {"bce": float(loss.detach()), "encoder": gradient_norm(encoder),
                          "head": gradient_norm(head)}
        del loss
    optimizer.zero_grad(set_to_none=True)
    initial_head = digest_tensors(head.state_dict())
    core.train_update(encoder, head, optimizer, tokenizer, first, None, scaler,
                      current_seed=11, history_seed=12)
    if digest_tensors(encoder.state_dict()) == initial_encoder or digest_tensors(head.state_dict()) == initial_head:
        raise ValueError("Encoder or head did not actually update")
    if not optimizer.state:
        raise ValueError("Optimizer state is empty after first training update")
    score_before = core.score_batch(encoder, head, tokenizer, current, scaler)
    saved_signature = digest_tensors([encoder.state_dict(), head.state_dict(), optimizer.state_dict()])
    saved_memory = memory.to_bytes()
    saved_scaler = {name: getattr(scaler, name).copy() for name in ("medians", "means", "scales")}
    progress = {"order_seed": 11, "next_stage": 2, "epoch": 0, "step": 0}
    print("Independent real encoder/head gradients passed; checking nonempty state replay", flush=True)
    with tempfile.TemporaryDirectory(prefix="text_state_", dir=output) as temporary:
        path = Path(temporary) / "state.pt"
        core.save_state(path, encoder, head, optimizer, scaler, progress, memory)
        checkpoint_bytes = path.stat().st_size
        old = memory.sample(16)
        core.train_update(encoder, head, optimizer, tokenizer, current, old, scaler,
                          current_seed=21, history_seed=22)
        expected_signature = digest_tensors([encoder.state_dict(), head.state_dict(), optimizer.state_dict()])
        expected_scores = core.score_batch(encoder, head, tokenizer, current, scaler)
        # Reload into the same model to avoid a second large GPU model/Adam copy.
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        scaler, restored_progress, restored_memory = core.restore_state(path, encoder, head, optimizer)
        for name, expected in saved_scaler.items():
            np.testing.assert_array_equal(getattr(scaler, name), expected,
                                          err_msg=f"Restored scaler array differs: {name}")
        if (restored_progress != progress or restored_memory.to_bytes() != saved_memory
                or digest_tensors([encoder.state_dict(), head.state_dict(), optimizer.state_dict()]) != saved_signature):
            raise ValueError("Encoder/head/Adam/progress/memory state replay differs")
        np.testing.assert_array_equal(core.score_batch(encoder, head, tokenizer, current, scaler), score_before)
        replay = restored_memory.sample(16)
        if [core.edge_payload(replay, i) for i in range(len(replay.edges))] != [
                core.edge_payload(old, i) for i in range(len(old.edges))]:
            raise ValueError("Restored replay RNG selected different historical inputs")
        core.train_update(encoder, head, optimizer, tokenizer, current, replay, scaler,
                          current_seed=21, history_seed=22)
        if digest_tensors([encoder.state_dict(), head.state_dict(), optimizer.state_dict()]) != expected_signature:
            raise ValueError("Subsequent real model/Adam update differs after reload")
        np.testing.assert_array_equal(core.score_batch(encoder, head, tokenizer, current, scaler), expected_scores)
    print("Exact state/score/subsequent-update replay passed; checking one real public-world shape", flush=True)
    capacity, retained, provenance = public_capacity_batches()
    history = retained.sample(16)
    if len(history.edges) != 16 or len(history.accounts) != 32:
        raise ValueError("Capacity fixture lacks the maximum 32 distinct historical endpoints")
    chunks, _ = core.prepare_text(capacity, tokenizer)
    old_chunks, _ = core.prepare_text(history, tokenizer)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    report = core.train_update(encoder, head, optimizer, tokenizer, capacity, history, scaler,
                               current_seed=41, history_seed=42)
    torch.cuda.synchronize()
    capacity_seconds = time.perf_counter() - started
    return {"status": "PASSED_NEW_TEXT_COMPONENT_CHECK_NO_FORMAL_EXPERIMENT",
            "contracts_passed": result.testsRun, "contracts_skipped": 0,
            "independent_real_branch_gradients": branches,
            "activation_checkpoint_loss_and_all_gradients_exact": True,
            "encoder_and_head_updated": True, "initial_scaler_encoder_unchanged": True,
            "exact_medians_means_scales_reload": True,
            "exact_nonempty_optimizer_reload": True, "exact_raw_scores_reload": True,
            "exact_subsequent_model_and_adam_update": True,
            "temporary_checkpoint_bytes": checkpoint_bytes, "temporary_checkpoint_removed": True,
            "capacity": {**provenance, "current_edges": 378, "current_accounts": 28,
                         "history_edges": 16, "history_accounts": 32,
                         "current_chunks": len(chunks), "history_chunks": len(old_chunks),
                         "memory_bytes": retained.used_bytes, "update_seconds": capacity_seconds,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
                         "loss_report": report,
                         "scope": "one public current-world shape; not worst-case or formal dataset validation"},
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
                            "initial_free_bytes": free, "total_gpu_bytes": total,
                            "packages": {name: version(name) for name in
                                         ("numpy", "transformers", "sentence-transformers")}}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Actual check requires original Linux py310; --help is available anywhere")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT / "scripts/step28_continual_text.py", core.POLICY_PATH,
             ROOT / "scripts/step28_continual_core.py", ROOT / "scripts/step28_labse_finetune_common.py",
             ROOT / "scripts/step7_authorship_common.py", ROOT / "tests/test_step28_continual_text_contracts.py"]
    records = [{"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
                "sha256": common.sha256_file(path)} for path in files]
    try:
        result = run_check(args.output)
    except Exception as exc:
        (args.output / "runtime.json").write_bytes(core.json_bytes({"status": "FAILED_COMPONENT_CHECK",
            "exception": f"{type(exc).__name__}: {exc}", "files": records,
            "formal_experiment_started": False}) + b"\n")
        raise
    result["files"] = records
    (args.output / "runtime.json").write_bytes(core.json_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
