"""Linux CPU handwritten verification; no formal texts, labels or trained states."""
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
from unittest import mock

import numpy as np

import step28_er_weight as method


def native(arm: str, fixtures: object, check: object, p: dict | None = None) -> tuple[dict, dict]:
    import torch

    started = time.monotonic()
    p = method.contract() if p is None else p
    weight = method.all_weights(p)[arm]
    c = method.config(p)
    model = method.base.load_model(c, "split_rank", "cpu")
    optimizer = method.core.make_optimizer(model, c)
    old, current = fixtures.handmade_group("native_old", 1), fixtures.handmade_group("native_current", 2)
    initial = method.core.state_digest(model.state_dict())
    covered = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in covered}) != len(covered) or {id(p) for p in covered} != {id(p) for p in model.parameters()}:
        raise ValueError("Native optimizer coverage differs")
    warm = method.parent.update(model, optimizer, old, None, None, c, "seq", 1, 1, 81, 82,
                                observe=True, check=check)
    after_warm = method.core.state_digest(model.state_dict())
    # An explicit handmade counter fixture, never a claim of 288 native warm updates.
    for state in optimizer.state.values():
        state["step"].fill_(288)
    before_optimizer = method.core.state_digest(optimizer.state_dict())
    probes = {name: parameter for name, parameter in model.named_parameters()
              if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
              or name.startswith("head.")}
    layers = model.encoder[0].auto_model.config.num_hidden_layers
    if sum(".attention.self.query.weight" in name for name in probes) != layers:
        raise ValueError("Native layer probes incomplete")
    before = {name: method.core.state_digest(p) for name, p in probes.items()}
    component_parameters = {
        "encoder_first_query": model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
        "head_hidden": model.head[0].weight}
    reference, mse_gradients = None, {}
    if method.with_logits(p) and arm in p["arms"]:
        reference = method.parent.ranking.score(model, [old], c, check)[0]
        model.train()
        torch.manual_seed(108)
        prediction = method.base.logits(model, old, c, "split_rank", check)
        target = torch.tensor(reference, dtype=prediction.dtype)
        penalty = .5 * (prediction - target).square().mean()
        gradients = torch.autograd.grad(penalty, tuple(component_parameters.values()))
        mse_gradients = {name: gradient.detach().clone() for name, gradient in
                         zip(component_parameters, gradients, strict=True)}
        if target.requires_grad or any(not torch.isfinite(g).all() or float(g.norm()) <= 0
                                       for g in mse_gradients.values()):
            raise ValueError("Detached target MSE must independently reach encoder and head")
        del prediction, target, penalty, gradients
    components = {name: [] for name in component_parameters}
    hooks = [parameter.register_hook(lambda grad, name=name: components[name].append(grad.detach().clone()))
             for name, parameter in component_parameters.items()]
    captured = []
    hooks.append(model.head.register_forward_hook(lambda _, __, output: captured.append(output.detach().flatten().numpy().copy())))
    try:
        update = method.update(model, optimizer, current, old, c, weight, 2, 1, 107, 108,
                               reference=reference, logit_weight=.5 if reference is not None else 0.,
                               observe=True, check=check)
    finally:
        for hook in hooks:
            hook.remove()
    if len(captured) != 2 or any(len(values) != 2 for values in components.values()):
        raise ValueError("Need separate current/history native forwards and gradient contributions")
    independent = {}
    for role, values, group in (("current", captured[0], current), ("history", captured[1], old)):
        independent[role] = fixtures.scalar_losses(values, group.labels)
        for term, value in independent[role].items():
            if abs(update[role + "_" + term] - value) > (4e-6 if term == "total" else 2e-6):
                raise ValueError("Native objective differs from scalar reference")
    expected_total = independent["current"]["total"] + weight * independent["history"]["total"]
    expected_mse = 0.
    if reference is not None:
        expected_mse = float(np.mean((captured[1].astype(np.float64) - reference) ** 2))
        if abs(update["logit_mse"] - expected_mse) > 1e-6:
            raise ValueError("Native MSE differs from independent scalar mean")
        expected_total += .5 * expected_mse
    if abs(update["total"] - expected_total) > 6e-6:
        raise ValueError("Native history weight does not affect the full objective")
    rows = []
    for name, parameter in probes.items():
        norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.
        changed = before[name] != method.core.state_digest(parameter)
        if not np.isfinite(norm) or norm <= 0 or not changed:
            raise ValueError("Native parameter was not updated: " + name)
        rows.append({"name": name, "gradient_norm_after_clip": norm, "parameters_changed": changed})
    for name, contributions in components.items():
        if not all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in contributions):
            raise ValueError("A native loss contribution misses " + name)
    unused = [name for name, p in model.named_parameters() if p.grad is None]
    if any(".pooler." not in name for name in unused) or method.parent.adam_step(optimizer) != 289:
        raise ValueError("Unexpected native optimizer coverage or step")
    result = {"arm": arm, "history_weight": weight, "native_updates_actually_executed": 2,
              "counter_fixture": "One real warm update, then Adam counters 1->288; no native full-stage continuity claim",
              "initial_model_state_sha256": initial, "after_warm_model_state_sha256": after_warm,
              "before_stage2_optimizer_sha256": before_optimizer,
              "after_model_state_sha256": method.core.state_digest(model.state_dict()),
              "after_optimizer_sha256": method.core.state_digest(optimizer.state_dict()),
              "warm_update": warm, "weighted_update": update,
              "captured_current_logits": captured[0].tolist(), "captured_history_logits": captured[1].tolist(),
              "independent_objectives": independent, "independent_total": expected_total,
              "gradient_components": {name: {role: {"norm": float(g.norm()), "sha256": method.core.state_digest(g)}
                                              for role, g in zip(("current", "weighted_history"), values, strict=True)}
                                      for name, values in components.items()},
              "parameter_probes": rows, "unused_parameters": unused, "backbone_layers": layers,
              "seconds": time.monotonic() - started}
    if reference is not None:
        result.update(origin_eval_reference=reference.tolist(), independent_logit_mse=expected_mse,
                      mse_gradient_reference={name: {"norm": float(g.norm()), "sha256": method.core.state_digest(g)}
                                              for name, g in mse_gradients.items()})
        components["mse_reference"] = mse_gradients
    del model, optimizer, covered, probes, component_parameters
    gc.collect()
    return result, components


def run(destination: Path, study: str = "weight") -> dict:
    import torch

    if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or destination.exists():
        raise ValueError("New Linux CPU evidence only; CUDA_VISIBLE_DEVICES must be empty")
    p = method.contract(study)
    source_files = method.sources(p)
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.use_deterministic_algorithms(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def check() -> None:
        if time.monotonic() - started >= p["cpu_verification"]["maximum_seconds"]:
            raise RuntimeError("CPU verification time limit reached")
        if sum(f.stat().st_size for f in destination.parent.rglob("*") if f.is_file()) > p["cpu_verification"]["maximum_evidence_bytes"]:
            raise RuntimeError("CPU evidence budget exceeded")

    sys.path.insert(0, str(method.data.ROOT / "tests"))
    import test_step28_er_weight_contracts as tests
    import test_step28_bge_continual_contracts as fixtures
    try:
        with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal texts")), \
             mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
             mock.patch.object(method.data, "Archive", side_effect=AssertionError("No formal archive")):
            tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
            contracts = {"passed": tested.testsRun - len(tested.failures) - len(tested.errors) - len(tested.skipped),
                         "failed": len(tested.failures) + len(tested.errors), "skipped": len(tested.skipped)}
            method.data.write_json(destination.parent / "contracts.json", contracts)
            if not tested.wasSuccessful() or tested.skipped:
                raise RuntimeError("Affected contracts must pass without skips")
            check()
            c = method.config(p)
            archive = method.core.model_files(method.base.model_config(c, "split_rank"))
            if any(archive[key] != c["models"]["split_rank"][key]
                   for key in ("file_count", "total_size_bytes", "content_sha256")):
                raise ValueError("Pretrained archive differs")
            reports, components = {}, {}
            native_arms = (("quarter", "logit_quarter") if method.with_logits(p) else
                           ("quarter", "tenth") if study == "low" else tuple(p["arms"]))
            for arm in native_arms:
                print(method.data.json_bytes({"event": "native_start", "arm": arm}).decode(), flush=True)
                reports[arm], components[arm] = native(arm, fixtures, check, p)
                method.data.write_json(destination.parent / (arm + ".json"), reports[arm])
                print(method.data.json_bytes({"event": "native_end", "arm": arm, "seconds": reports[arm]["seconds"]}).decode(), flush=True)
            for field in ("initial_model_state_sha256", "after_warm_model_state_sha256",
                          "before_stage2_optimizer_sha256", "captured_current_logits", "captured_history_logits"):
                if reports[native_arms[0]][field] != reports[native_arms[1]][field]:
                    raise ValueError("Native current/history states are not paired: " + field)
            scaling = {}
            ratio = method.all_weights(p)[native_arms[0]] / method.all_weights(p)[native_arms[1]]
            for name in components[native_arms[0]]:
                a_current, a_history = components[native_arms[0]][name]
                b_current, b_history = components[native_arms[1]][name]
                torch.testing.assert_close(a_current, b_current, rtol=0, atol=0)
                if method.with_logits(p):
                    expected = a_history + components["logit_quarter"]["mse_reference"][name]
                    torch.testing.assert_close(b_history, expected, rtol=2e-5, atol=2e-8)
                    scaling[name] = {"current_exactly_equal": True, "compared_arms": list(native_arms),
                                     "history_supervision_weight": .25, "independent_mse_weight": .5,
                                     "history_equals_er_plus_mse_gradient": True,
                                     "maximum_history_difference": float((b_history - expected).abs().max())}
                else:
                    torch.testing.assert_close(a_history, ratio * b_history, rtol=1e-6, atol=1e-8)
                    scaling[name] = {"current_exactly_equal": True,
                                     "compared_arms": list(native_arms), "expected_history_ratio": ratio,
                                     "weighted_history_scaling_matches": True,
                                     "maximum_history_difference": float((a_history - ratio * b_history).abs().max())}
            if method.sources(p) != source_files:
                raise ValueError("Sources changed during verification")
            gradient_evidence = "native_logit_gradient_increment_verified" if method.with_logits(p) else "native_history_gradient_scaling_verified"
            result = {"status": f"PASS_{method.evidence_tag(p)}_HANDMADE_CPU", "source_files": source_files,
                      "contracts": contracts, "native": {arm: method.data.record(destination.parent / (arm + ".json"), destination.parent)
                                                         for arm in p["arms"]},
                      "native_reference": {arm: method.data.record(destination.parent / (arm + ".json"), destination.parent)
                                           for arm in native_arms if arm not in p["arms"]},
                      "native_updates_actually_executed": 4, gradient_evidence: True,
                      "native_component_comparison": scaling, "formal_inputs": False, "formal_labels": False,
                      "formal_updates": 0, "gpu": False, "retained_native_weights": 0, "archive": archive,
                      "seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                                      "cpu_affinity": sorted(os.sched_getaffinity(0))},
                      "limitations": "Handwritten groups and tiny/native CPU checks; no formal training, CUDA/BF16 or native 288-step continuity claim"}
            method.data.write_json(destination, result)
            check()
            return result
    except Exception as error:
        method.data.write_json(destination.parent / "failure.json", {
            "status": "ER_WEIGHT_CPU_FAILED", "error": str(error), "seconds": time.monotonic() - started,
            "formal_inputs": False, "formal_labels": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--study", choices=("weight", "low", "logit"), default="weight")
    args = parser.parse_args()
    result = run(args.out.resolve(), args.study)
    print(method.data.json_bytes({key: result[key] for key in ("status", "native_updates_actually_executed", "seconds")}).decode())
