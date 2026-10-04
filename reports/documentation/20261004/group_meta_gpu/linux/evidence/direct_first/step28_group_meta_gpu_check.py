"""Bounded handwritten GPU feasibility; never opens formal data or old weights."""
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import replace
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from unittest import mock

import numpy as np
import torch

import step28_group_meta as method

SETTINGS = method.Settings(1e-5, 1e-3, .1, 1.)


def context(offload: bool):
    return torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else contextlib.nullcontext()


def assert_state_close(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("State keys differ")
        for key in left:
            assert_state_close(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError("State lengths differ")
        for a, b in zip(left, right, strict=True):
            assert_state_close(a, b)
    elif left != right:
        raise AssertionError("State scalar differs")


def tiny(check) -> dict:
    import test_step28_group_meta as fixtures
    c, episode, model, settings = fixtures.setup()
    model = model.to("cuda:0")
    optimizer = method.core.make_optimizer(model, c)
    method.parent.update(model, optimizer, episode.history_support, None, None,
                         c, "seq", 1, 1, 71, 72, check=check)
    for state in optimizer.state.values():
        state["step"].fill_(288)
    original_model, original_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    observations = []
    for offload in (False, True):
        model.load_state_dict(original_model)
        optimizer.load_state_dict(copy.deepcopy(original_adam))
        model.train()
        with context(offload):
            loss, terms, fast = method.objective(model, episode, c, settings, (81, 82, 83, 84), check)
            params = tuple(model.parameters())
            support = torch.autograd.grad(terms["support"], params, retain_graph=True)
            query = torch.autograd.grad(terms["query"], tuple(fast.values()), retain_graph=True)
            full = torch.autograd.grad(loss, params)
        corrections = {}
        for prefix in ("encoder.", "head."):
            indices = [i for i, (name, _) in enumerate(model.named_parameters()) if name.startswith(prefix)]
            corrections[prefix] = sum(float((full[i] - support[i] - settings.outer_weight * query[i]).square().sum())
                                      for i in indices)
            if not np.isfinite(corrections[prefix]) or corrections[prefix] <= 1e-14:
                raise AssertionError("Complete Hessian path differs insufficiently from first-order mutation")
        with context(offload):
            update = method.update(model, optimizer, episode, c, settings, 2, 1, (81, 82, 83, 84), check)
        observations.append({"loss": loss.detach().clone(), "gradient": tuple(g.detach().clone() for g in full),
                             "model": copy.deepcopy(model.state_dict()), "adam": copy.deepcopy(optimizer.state_dict()),
                             "corrections": corrections, "update": update})
    for key in ("loss", "gradient", "model", "adam"):
        assert_state_close(observations[0][key], observations[1][key])
    return {"all_parameter_gradient_and_model_adam_parity": True, "rtol": 1e-5, "atol": 1e-7,
            "hessian_correction_squared_norms": [r["corrections"] for r in observations],
            "settings_fixture_only": settings.__dict__, "actual_updates": 3}


def padded_text(tokenizer, prefix: str, length: int) -> str:
    text = prefix
    for _ in range(length):
        actual = len(tokenizer(text, padding=False, truncation=False)["input_ids"])
        if actual == length:
            return text
        if actual > length:
            break
        text += "文"
    raise AssertionError("Handwritten token shape could not be built without truncation")


def native(offload: bool, check) -> dict:
    import test_step28_group_meta as fixtures
    import test_step28_bge_continual_contracts as groups_fixture
    c, _, _, _ = fixtures.setup()
    c["input"]["microbatch"] = 32
    model = method.load_model(c, "cuda:0")
    groups = []
    for role in range(4):
        group = groups_fixture.handmade_group(f"gpu_meta_{role}", role * 2)
        items = []
        for account in range(28):
            rows = []
            for item in range(3 + int(account < 15)):
                title = padded_text(model.encoder.tokenizer, f"手写款式{role}{account}{item}", 21)
                description = padded_text(model.encoder.tokenizer, f"人工描述{account % 9}内容{role}{item}", 82)
                rows.append((f"{group.uid}_item_{account:02}_{item}", title, description))
            items.append(tuple(rows))
        group = replace(group, items=tuple(items))
        group.validate()
        if sum(map(len, group.items)) != 99:
            raise AssertionError("Incomplete handwritten group")
        groups.append(group)
    optimizer = method.core.make_optimizer(model, c)
    print("native_warm_start", flush=True)
    with context(offload):
        method.parent.update(model, optimizer, groups[1], None, None, c, "seq", 1, 1, 71, 72, check=check)
    for state in optimizer.state.values():
        state["step"].fill_(288)
    reference = method.parent.ranking.score(model, [groups[1]], c, check)[0]
    episode = method.Episode(*groups, reference, groups[1].uid)
    transformer = model.encoder[0].auto_model
    probes = {"encoder_query": transformer.encoder.layer[0].attention.self.query.weight,
              "head_hidden": model.head[0].weight}
    records = []
    for step in (1, 2):
        check()
        before = {name: p.detach().clone() for name, p in probes.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        print(f"native_meta_start step={step} offload={offload}", flush=True)
        with context(offload):
            record = method.update(model, optimizer, episode, c, SETTINGS, 2, step,
                                   tuple(80 + step * 4 + i for i in range(4)), check)
        torch.cuda.synchronize()
        record["seconds"] = time.monotonic() - started
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        record["probe_delta_norms"] = {name: float((p.detach() - before[name]).norm()) for name, p in probes.items()}
        if any(not np.isfinite(v) or v <= 0 for v in record["probe_delta_norms"].values()):
            raise AssertionError("Native encoder/head parameters did not change")
        records.append(record)
        print(method.data.json_bytes(record).decode(), flush=True)
    return {"actual_updates": 3, "fixture": "one real warm update; counter 1->288; two real meta updates",
            "offload_saved_tensors": offload, "pin_memory": False, "settings": SETTINGS.__dict__,
            "precision": "float32", "attention": transformer.config._attn_implementation,
            "checkpointing": False, "text_microbatch": 32,
            "groups": 4, "accounts_per_group": 28, "items_per_group": 99,
            "title_tokens": 21, "description_tokens": 82, "updates": records,
            "scope": "Handwritten shape only; not all formal input shapes or a complete stage"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tiny", "direct", "offload"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or args.output.exists():
        raise ValueError("Fresh isolated Linux GPU0 evidence path required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    sys.path.insert(0, str(method.data.ROOT / "tests"))
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 3600:
            raise TimeoutError("Per-process ceiling; external wrapper enforces remaining aggregate budget")
    result = {"mode": args.mode, "scope": "handwritten_only", "formal_inputs": False, "formal_labels": False,
              "old_experiment_weights": False, "python": sys.version, "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
              "cpu_affinity": sorted(os.sched_getaffinity(0))}
    try:
        with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal texts")), \
                mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
                mock.patch.object(method.data, "Archive", side_effect=AssertionError("No formal archive")):
            result["evidence"] = tiny(check) if args.mode == "tiny" else native(args.mode == "offload", check)
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__,
                      cuda_oom=isinstance(error, torch.cuda.OutOfMemoryError), traceback=traceback.format_exc())
    result["seconds"] = time.monotonic() - started
    result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["last_interval_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["last_interval_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    method.data.write_json(args.output, result)
    if result["status"] != "PASS":
        print(result["traceback"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
