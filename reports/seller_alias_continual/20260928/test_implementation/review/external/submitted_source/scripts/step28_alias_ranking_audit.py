"""Linux single-CPU verification on handmade inputs; no formal text or labels."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import platform
import resource
import sys
import time
import unittest
from typing import Any, Callable
from unittest import mock

import numpy as np

import step28_alias_ranking as method
import step28_alias_ranking_run as runner


def native(kind: str, group: Any, policy: dict, scalar_reference: Callable,
           check: Callable[[], None]) -> dict:
    import torch

    started = time.monotonic()
    config = method.reference_config(policy, "s0")
    model = (method.base.load_model(config, "split_rank", "cpu") if kind == "original_d"
             else method.load_model(policy, "s0", "cpu"))
    optimizer = method.core.make_optimizer(model, config)
    initial = method.core.state_digest(model.state_dict())
    optimized = [p for g in optimizer.param_groups for p in g["params"]]
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in model.parameters()}:
        raise ValueError("Native optimizer coverage differs")
    before_scores = method.score(model, [group], config, check)
    component_gradients = {}
    if kind == "hard":
        model.train()
        torch.manual_seed(20260927)
        predicted = method.base.logits(model, group, config, "split_rank", check)
        terms = method.objectives(predicted, torch.tensor(group.labels, dtype=torch.float32), .5)
        targets = [model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
                   model.head[0].weight]
        for name in ("bce", "rank", "hard"):
            check()
            gradients = torch.autograd.grad(terms[name], targets, retain_graph=True)
            norms = [float(g.norm()) for g in gradients]
            if not all(np.isfinite(value) and value > 0 for value in norms):
                raise ValueError("Native objective disconnected from encoder or head: " + name)
            component_gradients[name] = {"encoder_first_query_weight": norms[0], "head_hidden_weight": norms[1]}
        del predicted, terms, targets, gradients
    probes = {name: p for name, p in model.named_parameters()
              if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
              or name.startswith("head.")}
    layers = model.encoder[0].auto_model.config.num_hidden_layers
    if sum(".attention.self.query.weight" in name for name in probes) != layers:
        raise ValueError("Native layer probes are incomplete")
    before = {name: method.core.state_digest(parameter) for name, parameter in probes.items()}
    captured = {}

    def capture(module, inputs, output):
        captured["logits"] = output.detach().flatten().numpy().copy()

    check()
    hook = model.head.register_forward_hook(capture)
    try:
        if kind == "original_d":
            update = method.base.update(model, optimizer, group, config, "split_rank", 20260927,
                                        observe=True, check=check)
        else:
            arm = "d" if kind == "wrapped_d" else kind
            update = method.update(model, optimizer, group, config, policy, arm, 1, 20260927,
                                   observe=True, check=check)
    finally:
        hook.remove()
    independent = scalar_reference(captured["logits"], group.labels, .5 if kind == "hard" else 0.)
    del independent["hard_gradient"]
    for name in ("bce", "rank", "total") + (("hard",) if kind == "hard" else ()):
        if abs(update[name] - independent[name]) > (3e-6 if name == "total" else 1e-6):
            raise ValueError("Actual native loss differs from scalar reference: " + name)
    rows = []
    for name, parameter in probes.items():
        norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.
        changed = before[name] != method.core.state_digest(parameter)
        if not np.isfinite(norm) or norm <= 0 or not changed:
            raise ValueError("Native probe has no actual task update: " + name)
        rows.append({"name": name, "gradient_norm_after_clip": norm, "changed": changed,
                     "adam_step": float(optimizer.state[parameter]["step"])})
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    if any(".pooler." not in name for name in unused):
        raise ValueError("Unexpected unused native parameter")
    if not optimizer.state or any(float(state["step"]) != 1 for state in optimizer.state.values()):
        raise ValueError("Native Adam step differs")
    after_scores = method.score(model, [group], config, check)
    if np.array_equal(before_scores, after_scores):
        raise ValueError("Native scores did not change")
    result = {"kind": kind, "initial_model_state_sha256": initial,
              "post_model_state_sha256": method.core.state_digest(model.state_dict()),
              "post_optimizer_state_sha256": method.core.state_digest(optimizer.state_dict()),
              "initial_scores": before_scores.tolist(), "post_scores": after_scores.tolist(),
              "captured_training_logits": captured["logits"].tolist(),
              "native_updates": 1, "update": update, "independent_objectives": independent,
              "loss_component_probe_norms": component_gradients, "parameter_probes": rows,
              "unused_parameter_names": unused, "backbone_layers": layers,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "optimized_parameter_tensors": len(optimized), "adam_parameter_states": len(optimizer.state),
              "actual_optimizer_lrs": [float(g["lr"]) for g in optimizer.param_groups],
              "seconds": time.monotonic() - started}
    del optimizer, model, probes, optimized
    gc.collect()
    return result


def run(destination: Path) -> dict:
    import torch

    if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or destination.exists():
        raise ValueError("Use Linux CPU with CUDA_VISIBLE_DEVICES empty and a new evidence file")
    policy = method.contract()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_files = runner.sources()

    def check():
        if time.monotonic() - started >= policy["cpu_verification"]["maximum_seconds"]:
            raise RuntimeError("CPU verification time limit reached")
        total = sum(p.stat().st_size for p in destination.parent.rglob("*") if p.is_file())
        if total > policy["cpu_verification"]["maximum_evidence_bytes"]:
            raise RuntimeError("CPU evidence limit reached")

    sys.path.insert(0, str(method.data.ROOT / "tests"))
    from test_step28_alias_ranking_contracts import handmade_group, scalar_objectives
    try:
        with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal inputs")), \
             mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
             mock.patch.object(method.data, "Archive", side_effect=AssertionError("No public archive traversal")):
            suite = unittest.defaultTestLoader.loadTestsFromName("test_step28_alias_ranking_contracts")
            tested = unittest.TextTestRunner(verbosity=2).run(suite)
            contracts = {"passed": tested.testsRun - len(tested.failures) - len(tested.errors) - len(tested.skipped),
                         "failed": len(tested.failures) + len(tested.errors), "skipped": len(tested.skipped)}
            method.data.write_json(destination.parent / "contracts.json", contracts)
            if not tested.wasSuccessful() or tested.skipped:
                raise RuntimeError("All applicable contracts must pass without skips before native checks")
            config = method.reference_config(policy, "s0")
            archive = method.core.model_files(method.base.model_config(config, "split_rank"))
            if any(archive[key] != config["models"]["split_rank"][key]
                   for key in ("file_count", "total_size_bytes", "content_sha256")):
                raise ValueError("Actual native pretrained archive differs")
            reports = {}
            for kind in ("original_d", "wrapped_d", "schedule", "hard"):
                print(method.data.json_bytes({"event": "native_start", "kind": kind}).decode(), flush=True)
                reports[kind] = native(kind, handmade_group(), policy, scalar_objectives, check)
                method.data.write_json(destination.parent / (kind + ".json"), reports[kind])
                print(method.data.json_bytes({"event": "native_complete", "kind": kind,
                                               "seconds": reports[kind]["seconds"]}).decode(), flush=True)
            for field in ("initial_model_state_sha256", "post_model_state_sha256", "post_optimizer_state_sha256",
                          "initial_scores", "post_scores", "captured_training_logits", "parameter_count"):
                if reports["original_d"][field] != reports["wrapped_d"][field]:
                    raise ValueError("Actual original D and wrapper differ: " + field)
            for kind in ("schedule", "hard"):
                for field in ("initial_model_state_sha256", "initial_scores", "captured_training_logits", "parameter_count"):
                    if reports[kind][field] != reports["original_d"][field]:
                        raise ValueError("Intervention changed common initial forward or architecture: " + field)
                expected = [method.encoder_lr(policy, kind, 1), .001]
                if reports[kind]["actual_optimizer_lrs"] != expected:
                    raise ValueError("Actual native learning rates differ")
            if runner.sources() != source_files:
                raise ValueError("Sources changed during verification")
            check()
            result = {"status": "PASS_RANKING_NATIVE_AND_CONTRACTS", "contracts": contracts,
                      "policy_sha256": method.POLICY_SHA256, "method_sha256": method.data.sha256(Path(method.__file__)),
                      "source_files": source_files, "archive": archive, "native_handmade_updates": 4,
                      "original_d_wrapper_exact": True, "paired_initial_forward_exact": True,
                      "native_reports": {kind: method.data.record(destination.parent / (kind + ".json"), destination.parent)
                                         for kind in reports},
                      "formal_label_reads": 0, "formal_input_reads": 0, "formal_updates": 0,
                      "gpu_execution": False, "retained_native_model_payloads": 0,
                      "seconds": time.monotonic() - started,
                      "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "cpu_threads": torch.get_num_threads()},
                      "limitations": "Handmade CPU FP32 only; no formal performance or CUDA/BF16 claim. Tiny checkpoint restore and native one-step checks are distinct evidence."}
            method.data.write_json(destination, result)
            check()
            return result
    except Exception as error:
        method.data.write_json(destination.parent / "failure.json", {
            "status": "CPU_VERIFICATION_FAILED", "error": str(error), "seconds": time.monotonic() - started,
            "formal_input_reads": 0, "formal_label_reads": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.out.resolve())
    print(method.data.json_bytes({key: report[key] for key in ("status", "native_handmade_updates", "seconds", "max_rss_kib")}).decode())
