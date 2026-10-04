"""Linux CPU verification using handmade text only; never formal supervision.

Native checks perform two actual updates per method. The Adam counter is openly
rebased after the first update to isolate stage-two code. Full 288-step continuity
and checkpoint branches are separately exercised with the tiny handmade model.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Callable
import unittest
from unittest import mock

import numpy as np

import step28_bge_continual as method
import step28_bge_continual_run as runner


def native(arm: str, fixtures: object, policy: dict, check: Callable[[], None]) -> dict:
    import torch

    started = time.monotonic()
    c = method.config(policy)
    model = method.base.load_model(c, "split_rank", "cpu")
    optimizer = method.core.make_optimizer(model, c)
    old, current = fixtures.handmade_group("native_A", 1), fixtures.handmade_group("native_B", 2)
    initial = method.core.state_digest(model.state_dict())
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in model.parameters()}:
        raise ValueError("Native optimizer parameter coverage differs")
    warm = method.update(model, optimizer, old, None, None, c, "seq", 1, 1, 81, 82,
                         observe=True, check=check)
    after_warm = method.core.state_digest(model.state_dict())
    reference = method.ranking.score(model, [old], c, check)[0]
    # This is a declared test fixture. No claim that 288 native updates occurred.
    for state in optimizer.state.values():
        state["step"].fill_(288)
    prior_optimizer = method.core.state_digest(optimizer.state_dict())
    component = {}
    if arm == "logit":
        model.train()
        torch.manual_seed(108)
        prediction = method.base.logits(model, old, c, "split_rank", check)
        target = torch.tensor(reference, dtype=prediction.dtype)
        penalty = (prediction - target).square().mean()
        probes = [model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
                  model.head[0].weight]
        gradients = torch.autograd.grad(penalty, probes)
        norms = [float(g.norm()) for g in gradients]
        if target.requires_grad or not all(np.isfinite(v) and v > 0 for v in norms):
            raise ValueError("Historical MSE alone does not reach the real encoder and head")
        component = {"encoder_first_query_weight": norms[0], "head_hidden_weight": norms[1],
                     "reference_is_detached": True, "reference_mode": "origin_model_eval",
                     "student_mode": "train"}
        del prediction, target, penalty, probes, gradients
    probes = {name: parameter for name, parameter in model.named_parameters()
              if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
              or name.startswith("head.")}
    layers = model.encoder[0].auto_model.config.num_hidden_layers
    if sum(".attention.self.query.weight" in name for name in probes) != layers:
        raise ValueError("Native layer probes are incomplete")
    before = {name: method.core.state_digest(parameter) for name, parameter in probes.items()}
    captured = []
    hook = model.head.register_forward_hook(lambda _, __, out: captured.append(out.detach().flatten().numpy().copy()))
    try:
        update = method.update(model, optimizer, current, old if arm != "seq" else None,
                               reference if arm == "logit" else None, c, arm, 2, 1, 107, 108,
                               observe=True, check=check)
    finally:
        hook.remove()
    if len(captured) != (1 if arm == "seq" else 2):
        raise ValueError("Unexpected current/history forwards")
    independent = {"current": fixtures.scalar_losses(captured[0], current.labels)}
    if arm != "seq":
        independent["history"] = fixtures.scalar_losses(captured[1], old.labels)
    for role, losses in independent.items():
        for name, value in losses.items():
            if abs(update[role + "_" + name] - value) > (4e-6 if name == "total" else 2e-6):
                raise ValueError("Native loss differs from scalar reference: " + role + "/" + name)
    expected_mse = float(np.mean((captured[1].astype(np.float64) - reference) ** 2)) if arm == "logit" else 0.
    if abs(update["logit_mse"] - expected_mse) > 1e-6:
        raise ValueError("Native logit MSE differs from scalar reference")
    rows = []
    for name, parameter in probes.items():
        norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.
        changed = before[name] != method.core.state_digest(parameter)
        if not np.isfinite(norm) or norm <= 0 or not changed:
            raise ValueError("Native parameter has no actual update: " + name)
        rows.append({"name": name, "gradient_norm_after_clip": norm, "parameters_changed": changed,
                     "adam_step": float(optimizer.state[parameter]["step"])})
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    if any(".pooler." not in name for name in unused) or method.adam_step(optimizer) != 289:
        raise ValueError("Unexpected unused parameters or Adam counter")
    result = {"method": arm, "native_updates_actually_executed": 2,
              "handmade_counter_fixture": "One actual warm update, then step counters rebased 1 -> 288; native continuity over 288 updates is not claimed",
              "initial_model_state_sha256": initial, "after_warm_model_state_sha256": after_warm,
              "before_stage2_optimizer_sha256": prior_optimizer,
              "after_model_state_sha256": method.core.state_digest(model.state_dict()),
              "after_optimizer_state_sha256": method.core.state_digest(optimizer.state_dict()),
              "warm_update": warm, "stage2_update": update,
              "captured_current_logits": captured[0].tolist(),
              "captured_history_logits": captured[1].tolist() if arm != "seq" else None,
              "origin_eval_reference": reference.tolist(), "independent_loss": independent,
              "independent_logit_mse": expected_mse, "logit_term_gradient": component,
              "parameter_probes": rows, "unused_parameters": unused, "backbone_layers": layers,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "seconds": time.monotonic() - started}
    del model, optimizer, optimized, probes
    gc.collect()
    return result


def run(destination: Path) -> dict:
    import torch

    if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or destination.exists():
        raise ValueError("Use Linux CPU and a new evidence file; CUDA_VISIBLE_DEVICES must be empty")
    policy, source_files = method.contract(), runner.sources()
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)

    def check():
        if time.monotonic() - started >= policy["cpu_verification"]["maximum_seconds"]:
            raise RuntimeError("CPU verification time limit reached")
        if sum(p.stat().st_size for p in destination.parent.rglob("*") if p.is_file()) > policy["cpu_verification"]["maximum_evidence_bytes"]:
            raise RuntimeError("CPU evidence budget exceeded")

    sys.path.insert(0, str(method.data.ROOT / "tests"))
    import test_step28_bge_continual_contracts as fixtures
    try:
        with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal text")), \
             mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal CSV labels")), \
             mock.patch.object(method.data, "Archive", side_effect=AssertionError("No data archive")):
            suite = unittest.defaultTestLoader.loadTestsFromModule(fixtures)
            tested = unittest.TextTestRunner(verbosity=2).run(suite)
            contracts = {"passed": tested.testsRun - len(tested.failures) - len(tested.errors) - len(tested.skipped),
                         "failed": len(tested.failures) + len(tested.errors), "skipped": len(tested.skipped)}
            method.data.write_json(destination.parent / "contracts.json", contracts)
            if not tested.wasSuccessful() or tested.skipped:
                raise RuntimeError("Applicable handmade tests must pass without skips")
            check()
            c = method.config(policy)
            archive = method.core.model_files(method.base.model_config(c, "split_rank"))
            if any(archive[k] != c["models"]["split_rank"][k]
                   for k in ("file_count", "total_size_bytes", "content_sha256")):
                raise ValueError("Native pretrained archive differs")
            reports = {}
            for arm in method.UPDATED:
                print(method.data.json_bytes({"event": "native_start", "method": arm}).decode(), flush=True)
                reports[arm] = native(arm, fixtures, policy, check)
                method.data.write_json(destination.parent / (arm + ".json"), reports[arm])
                print(method.data.json_bytes({"event": "native_end", "method": arm,
                                               "seconds": reports[arm]["seconds"]}).decode(), flush=True)
            for arm in ("er", "logit"):
                for field in ("initial_model_state_sha256", "after_warm_model_state_sha256",
                              "before_stage2_optimizer_sha256", "captured_current_logits", "origin_eval_reference"):
                    if reports[arm][field] != reports["seq"][field]:
                        raise ValueError("Native shared-state/current pairing differs: " + field)
            if reports["er"]["captured_history_logits"] != reports["logit"]["captured_history_logits"]:
                raise ValueError("Native history forwards are not paired")
            if source_files != runner.sources():
                raise ValueError("Sources changed during verification")
            result = {"status": "PASS_BGE_CONTINUAL_HANDMADE_CPU", "source_files": source_files,
                      "contracts": contracts, "native": {arm: method.data.record(destination.parent / (arm + ".json"), destination.parent)
                                                         for arm in method.UPDATED},
                      "native_updates_actually_executed": 6, "formal_inputs": False, "formal_labels": False,
                      "formal_updates": 0, "gpu": False, "retained_native_weights": 0, "archive": archive,
                      "seconds": time.monotonic() - started,
                      "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0))},
                      "limitations": "Tiny full-stage/reload plus native two-update fixtures; no formal data/performance, native full checkpoint, CUDA/BF16 or 288-step native continuity claim"}
            method.data.write_json(destination, result)
            check()
            return result
    except Exception as error:
        method.data.write_json(destination.parent / "failure.json", {
            "status": "CPU_VERIFICATION_FAILED", "error": str(error), "seconds": time.monotonic() - started,
            "formal_inputs": False, "formal_labels": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out.resolve())
    print(method.data.json_bytes({k: result[k] for k in ("status", "native_updates_actually_executed", "seconds")}).decode())
