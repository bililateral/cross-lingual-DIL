"""Linux-only handwritten CPU verification for the risk-replay implementation."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest import mock

import numpy as np

import step28_risk_replay as method


def native(check) -> dict:
    import torch
    import test_step28_risk_replay as tests
    fixtures = tests.fixtures
    c = fixtures.config()
    model = method.base.load_model(c, "split_rank", "cpu")
    optimizer = method.core.make_optimizer(model, c)
    old, new = fixtures.handmade_group("native_old"), fixtures.handmade_group("native_new", 2)
    method.parent.update(model, optimizer, old, None, None, c, "seq", 1, 1, 71, 72, check=check)
    for state in optimizer.state.values():
        state["step"].fill_(288)
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    origin = method.parent.ranking.score(model, [old], c, check)[0]
    reference = method.make_reference(origin, old.labels)
    # Deliberately degraded reference to activate every channel in a handmade
    # gradient probe. This fixture is NOT an actual stage-end scientific target.
    reference = {key: (np.asarray(value) * .5).tolist() for key, value in reference.items()}
    probes = {"encoder_query": model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
              "head_hidden": model.head[0].weight}
    model.train(); torch.manual_seed(82)
    predicted = method.base.logits(model, old, c, "split_rank", check)
    risks, counts = method.query_risks(predicted, torch.tensor(old.labels))
    penalty = method.distribution_penalty(risks, counts, reference).mean()
    gradients = torch.autograd.grad(.5 * penalty, tuple(probes.values()))
    increments = {key: value.detach().clone() for key, value in zip(probes, gradients)}
    if any(not torch.isfinite(value).all() or float(value.norm()) <= 0 for value in increments.values()):
        raise AssertionError("Independent retention gradient misses encoder or head")
    independent = tests.scalar_penalty(predicted.detach().numpy(), old.labels, reference)
    if abs(float(penalty) - independent) > 5e-6:
        raise AssertionError("Native risk objective differs from scalar reference")
    del predicted, risks, counts, penalty, gradients
    captured, rows, digests = {}, {}, {}
    original_clip = torch.nn.utils.clip_grad_norm_
    for name, gamma in (("zero", 0.), ("risk", .5)):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        def capture(parameters, *args, **kwargs):
            captured[name] = {key: parameter.grad.detach().clone() for key, parameter in probes.items()}
            return original_clip(parameters, *args, **kwargs)
        with mock.patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=capture):
            rows[name] = method.update(model, optimizer, new, old, reference, c, 2, 1, 81, 82,
                                       retention_weight=gamma, check=check)
        digests[name] = {key: method.core.state_digest(parameter) for key, parameter in probes.items()}
    decomposition = {}
    for key in probes:
        difference = captured["risk"][key] - captured["zero"][key]
        torch.testing.assert_close(difference, increments[key], rtol=3e-4, atol=2e-6)
        if digests["zero"][key] == digests["risk"][key]:
            raise AssertionError("Retention failed to affect actual native update")
        decomposition[key] = {"increment_norm": float(increments[key].norm()),
                              "max_residual": float((difference - increments[key]).abs().max()),
                              "different_actual_parameter_update": True}
    return {"actual_native_updates": 3, "counter_fixture": "one warm update; step counter 1->288; two paired next-step updates",
            "reference_fixture": "half of actual eval-origin risks; activation probe only, not scientific target",
            "retention_gradient_decomposition": decomposition, "updates": rows,
            "independent_scalar_retention": independent, "parameter_digests": digests}


def main() -> None:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Only a fresh Linux CPU evidence output")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(method.data.ROOT / "tests"))
    import test_step28_risk_replay as tests
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 2700:
            raise RuntimeError("CPU verification deadline exceeded")
    result = {"scope": "handwritten_only", "formal_data_access": False,
              "python": sys.version, "torch": torch.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0))}
    with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal texts")), \
            mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
            mock.patch.object(method.data, "Archive", side_effect=AssertionError("No formal archive")):
        tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        result["tests"] = {"run": tested.testsRun, "failures": len(tested.failures),
                           "errors": len(tested.errors), "skipped": len(tested.skipped)}
        if tested.wasSuccessful() and args.native:
            result["native"] = native(check)
    result["seconds"] = time.monotonic() - started
    result["status"] = "PASS" if tested.wasSuccessful() else "FAIL"
    method.data.write_json(args.output, result)
    if not tested.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
