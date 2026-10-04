"""Linux CPU handwritten verification; no formal data or training entry."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest
from unittest import mock

import torch

import step28_group_meta as method


def native(check) -> dict:
    import test_step28_group_meta as tests
    c, episode, _, _ = tests.setup()
    # Four-text chunks create many retained dense parameter-gradient graphs.
    # Larger text chunks reduce that overhead without dropping accounts/records.
    # This is a handwritten CPU execution setting, not a frozen GPU schedule.
    c["input"]["microbatch"] = 32
    model = method.load_model(c, "cpu")
    transformer = model.encoder[0].auto_model
    transformer.gradient_checkpointing_disable()
    if transformer.config._attn_implementation != "eager":
        raise AssertionError("Native double-backward route requires eager attention")
    optimizer = method.core.make_optimizer(model, c)
    print("native_warm_start", flush=True)
    method.parent.update(model, optimizer, episode.history_support, None, None,
                         c, "seq", 1, 1, 71, 72, check=check)
    for state in optimizer.state.values():
        state["step"].fill_(288)
    model.eval()
    with torch.no_grad():
        reference = method.base.logits(model, episode.history_support, c, "split_rank", check).numpy().copy()
    episode = replace(episode, reference=reference)
    settings = method.Settings(.0001, .01, .3, .7)
    probes = {"encoder_query": transformer.encoder.layer[0].attention.self.query.weight,
              "head_hidden": model.head[0].weight}
    before = {name: p.detach().clone() for name, p in probes.items()}
    optimizer_digest = method.core.state_digest(optimizer.state_dict())
    original_objective = method.objective
    original_clip = torch.nn.utils.clip_grad_norm_
    evidence = {}

    def build(*args, **kwargs):
        print("native_meta_graph_start", flush=True)
        result = original_objective(*args, **kwargs)
        _, terms, fast = result
        if optimizer_digest != method.core.state_digest(optimizer.state_dict()):
            raise AssertionError("Virtual update mutated Adam")
        for name, parameter in probes.items():
            if not torch.equal(parameter, before[name]):
                raise AssertionError("Virtual update mutated original parameter")
        support = torch.autograd.grad(terms["support"], tuple(probes.values()), retain_graph=True)
        names = {id(parameter): "model." + name for name, parameter in model.named_parameters()}
        fast_probes = tuple(fast[names[id(parameter)]] for parameter in probes.values())
        query_fast = torch.autograd.grad(terms["query"], fast_probes, retain_graph=True)
        for i, name in enumerate(probes):
            evidence[name] = {"support": support[i].detach().clone(),
                              "query_fast": query_fast[i].detach().clone()}
        print("native_meta_graph_ready", flush=True)
        return result

    def clip(parameters, *args, **kwargs):
        print("native_outer_backward_complete", flush=True)
        for name, parameter in probes.items():
            record = evidence[name]
            increment = parameter.grad - record.pop("support")
            correction = increment - settings.outer_weight * record.pop("query_fast")
            if (not torch.isfinite(increment).all() or not torch.isfinite(correction).all()
                    or float(increment.norm()) <= 0 or float(correction.norm()) <= 0):
                raise AssertionError("Native outer/Hessian path absent")
            record.update(outer_increment_norm=float(increment.norm()),
                          hessian_correction_norm=float(correction.norm()))
        return original_clip(parameters, *args, **kwargs)

    with mock.patch.object(method, "objective", side_effect=build), \
            mock.patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=clip):
        result = method.update(model, optimizer, episode, c, settings, 2, 1, (81, 82, 83, 84), check)
    for name, parameter in probes.items():
        if torch.equal(parameter, before[name]):
            raise AssertionError("Native actual parameter unchanged")
        evidence[name]["actual_parameter_delta_norm"] = float((parameter - before[name]).norm())
    print("native_meta_update_complete", flush=True)
    return {"actual_updates": 2, "fixture": "one warm update, Adam step rebased 1->288, one actual meta update",
            "settings_fixture_only": settings.__dict__, "attention": "eager", "precision": "float32",
            "checkpointing": False, "text_microbatch": 32, "probes": evidence, "update": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Fresh Linux CPU evidence path required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(method.data.ROOT / "tests"))
    import test_step28_group_meta as tests
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 7200:
            raise TimeoutError("Handwritten budget exhausted")
    result = {"scope": "handwritten_only", "formal_data_access": False,
              "python": sys.version, "torch": torch.__version__,
              "cpu_affinity": sorted(os.sched_getaffinity(0))}
    try:
        with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal texts")), \
                mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
                mock.patch.object(method.data, "Archive", side_effect=AssertionError("No formal archive")):
            if args.native_only:
                result["native"] = native(check)
            else:
                tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
                result["tests"] = {"run": tested.testsRun, "failures": len(tested.failures),
                                   "errors": len(tested.errors), "skipped": len(tested.skipped)}
                if not tested.wasSuccessful():
                    raise AssertionError("Handwritten tests failed")
        result["status"] = "PASS"
    except Exception:
        result.update(status="FAIL", traceback=traceback.format_exc())
    result["seconds"] = time.monotonic() - started
    method.data.write_json(args.output, result)
    if result["status"] != "PASS":
        print(result["traceback"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
