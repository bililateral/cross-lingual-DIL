"""Linux CPU contracts and four native BGE updates on handwritten inputs only."""
from __future__ import annotations

import argparse
import gc
import itertools
import os
from pathlib import Path
import platform
import resource
import sys
import time
import unittest
from typing import Any
from unittest import mock

import numpy as np

import step28_alias_pooling as method
import step28_alias_pooling_run as runner


def scalar_objectives(logits: np.ndarray, labels: tuple[int, ...]) -> dict:
    values = logits.astype(np.float64)
    truth = np.asarray(labels)
    bce = float(np.mean(np.logaddexp(0., values) - truth * values))
    pairs = list(itertools.combinations(range(28), 2))
    terms = []
    for query in range(28):
        indices = [i for i, pair in enumerate(pairs) if query in pair]
        local = values[indices]
        maximum = float(local.max())
        terms.append(maximum + np.log(np.exp(local - maximum).sum()) - local[truth[indices] == 1].mean())
    return {"bce": bce, "rank": float(np.mean(terms)), "total": bce + float(np.mean(terms))}


def native(kind: str, group: Any, policy: dict, check) -> dict:
    import torch

    started = time.monotonic()
    c = method.reference_config(policy, "s0")
    weighted = kind == "weighted"
    model = (method.base.load_model(c, "split_rank", "cpu") if kind == "original_d"
             else method.load_model(policy, "s0", weighted, "cpu"))
    optimizer = (method.core.make_optimizer(model, c) if kind == "original_d"
                 else method.make_optimizer(model, c, policy))
    common = method.common_digest(model)
    if common != policy["historical_d"]["expected_initial_common_state_sha256"]:
        raise ValueError("Native common initial weights differ from archived D")
    optimized = [p for g in optimizer.param_groups for p in g["params"]]
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in model.parameters()}:
        raise ValueError("Native optimizer coverage differs")
    before_scores = method.score(model, [group], c, check)
    objectives = {}
    if weighted:
        model.train()
        torch.manual_seed(20260925)
        predicted = method.logits(model, group, c, check)
        terms = method.base.objectives(predicted, torch.tensor(group.labels, dtype=torch.float32), 1)
        targets = [model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
                   model.head[0].weight, model.aggregation.output.weight]
        for name in ("bce", "rank"):
            gradients = torch.autograd.grad(terms[name], targets, retain_graph=True)
            norms = [float(g.norm()) for g in gradients]
            if not all(np.isfinite(value) and value > 0 for value in norms):
                raise ValueError("Native loss is disconnected from encoder/head/aggregation: " + name)
            objectives[name] = norms
        del terms, predicted, gradients, targets
    probes = {name: p for name, p in model.named_parameters()
              if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
              or name.startswith("head.") or name.startswith("aggregation.")}
    layers = model.encoder[0].auto_model.config.num_hidden_layers
    if sum(".attention.self.query.weight" in name for name in probes) != layers:
        raise ValueError("Native layer probes are incomplete")
    updates = []
    for step in range(2 if weighted else 1):
        check()
        hashes = {name: method.core.state_digest(parameter) for name, parameter in probes.items()}
        captured = {}

        def capture(module, inputs, output):
            captured["logits"] = output.detach().flatten().numpy().copy()

        hook = model.head.register_forward_hook(capture)
        try:
            log = (method.base.update(model, optimizer, group, c, "split_rank", 20260925 + step, observe=True, check=check)
                   if kind == "original_d" else method.update(model, optimizer, group, c, 20260925 + step, observe=True, check=check))
        finally:
            hook.remove()
        independent = scalar_objectives(captured["logits"], group.labels)
        if any(abs(log[key] - independent[key]) > (2e-6 if key == "total" else 1e-6)
               for key in ("bce", "rank", "total")):
            raise ValueError("Native objective differs from independent scalar computation")
        rows = []
        for name, parameter in probes.items():
            norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.
            changed = hashes[name] != method.core.state_digest(parameter)
            initially_zero_hidden = weighted and step == 0 and name.startswith("aggregation.hidden.")
            if initially_zero_hidden:
                if norm != 0:
                    raise ValueError("Zero output weight should block hidden task gradient in step one")
            elif not np.isfinite(norm) or norm <= 0 or not changed:
                raise ValueError("Native probe did not learn from its task gradient: " + name)
            rows.append({"name": name, "gradient_norm_after_clip": norm, "changed": changed,
                         "adam_step": float(optimizer.state[parameter]["step"]),
                         "zero_hidden_task_gradient_expected": initially_zero_hidden})
        unused = [name for name, p in model.named_parameters() if p.grad is None]
        if any(".pooler." not in name for name in unused):
            raise ValueError("Unexpected unused native parameter")
        if any(float(state["step"]) != step + 1 for state in optimizer.state.values()):
            raise ValueError("Native Adam continuity differs")
        updates.append({"step": step + 1, "update": log, "independent_objectives": independent,
                        "probes": rows, "unused_pretrained_pooler_parameters": unused})
        print(method.data.json_bytes({"event": "native_handmade_update", "kind": kind, "step": step + 1,
                                      "seconds": time.monotonic() - started}).decode(), flush=True)
    after_scores = method.score(model, [group], c, check)
    if np.array_equal(before_scores, after_scores):
        raise ValueError("Native scores did not change")
    nonuniform = None
    if weighted:
        # Inspect actual learned weights on a handwritten account, no truth input.
        texts = [row[1] for row in group.items[0]] + [row[2] for row in group.items[0]]
        features = model.encoder.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        model.eval()
        with torch.inference_mode():
            encoded = torch.nn.functional.normalize(model.encoder(features)["sentence_embedding"].float(), dim=1)
            title, description = encoded.chunk(2)
            weights = model.aggregation(title, description)
            nonuniform = float((weights - 1 / len(title)).abs().max())
        if not np.isfinite(nonuniform) or nonuniform <= 0:
            raise ValueError("Learned native item weights remained uniform on the inspected handmade account")
    result = {"kind": kind, "initial_common_state_sha256": common,
              "post_model_state_sha256": method.core.state_digest(model.state_dict()),
              "post_optimizer_state_sha256": method.core.state_digest(optimizer.state_dict()),
              "initial_scores": before_scores.tolist(), "post_scores": after_scores.tolist(),
              "native_updates": len(updates), "updates": updates, "loss_component_probe_norms": objectives,
              "learned_weight_max_distance_from_uniform": nonuniform,
              "backbone_layers": layers, "parameter_count": sum(p.numel() for p in model.parameters()),
              "optimized_parameter_tensors": len(optimized), "adam_parameter_states": len(optimizer.state),
              "seconds": time.monotonic() - started}
    del optimizer, model, probes
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
    from test_step28_alias_pooling_contracts import handmade_group
    with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal inputs")), \
         mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal labels")), \
         mock.patch.object(method.data, "Archive", side_effect=AssertionError("No public archive traversal")):
        suite = unittest.defaultTestLoader.loadTestsFromName("test_step28_alias_pooling_contracts")
        tests = unittest.TextTestRunner(verbosity=2).run(suite)
        if not tests.wasSuccessful() or tests.skipped:
            raise RuntimeError("New computation contracts must all pass without skips")
        contracts = {"passed": tests.testsRun, "failed": len(tests.failures) + len(tests.errors), "skipped": len(tests.skipped)}
        archive = method.core.model_files(method.base.model_config(method.reference_config(policy, "s0"), "split_rank"))
        expected = method.reference_config(policy, "s0")["models"]["split_rank"]
        if any(archive[key] != expected[key] for key in ("file_count", "total_size_bytes", "content_sha256")):
            raise ValueError("Actual native pretrained archive differs")
        group = handmade_group()
        reports = {}
        for kind in ("original_d", "wrapped_d", "weighted"):
            reports[kind] = native(kind, group, policy, check)
            method.data.write_json(destination.parent / (kind + ".json"), reports[kind])
        for field in ("initial_common_state_sha256", "post_model_state_sha256", "post_optimizer_state_sha256",
                      "initial_scores", "post_scores"):
            if reports["original_d"][field] != reports["wrapped_d"][field]:
                raise ValueError("D wrapper differs from actual original native update: " + field)
        if reports["weighted"]["parameter_count"] - reports["original_d"]["parameter_count"] != policy["aggregation"]["additional_parameters"]:
            raise ValueError("Native aggregation parameter count differs from the approved intervention")
        initial_error = float(np.max(np.abs(np.asarray(reports["weighted"]["initial_scores"])
                                             - np.asarray(reports["original_d"]["initial_scores"]))))
        if initial_error > 1e-6:
            raise ValueError("Initial uniform weighted model differs excessively from D")
        if runner.sources() != source_files:
            raise ValueError("Sources changed during CPU verification")
        check()
        result = {"status": "PASS_POOLING_NATIVE_AND_CONTRACTS", "contracts": contracts,
                  "policy_sha256": method.POLICY_SHA256, "method_sha256": method.data.sha256(Path(method.__file__)),
                  "source_files": source_files, "archive": archive,
                  "native_handmade_updates": sum(r["native_updates"] for r in reports.values()),
                  "original_d_wrapper_exact": True, "uniform_initial_logit_max_error": initial_error,
                  "native_reports": {kind: method.data.record(destination.parent / (kind + ".json"), destination.parent) for kind in reports},
                  "formal_label_reads": 0, "formal_input_reads": 0, "formal_updates": 0,
                  "gpu_execution": False, "retained_native_model_payloads": 0,
                  "seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
                  "limitations": "Handwritten CPU FP32 only. No CUDA/BF16/formal-data effect claim; tiny tests and native updates are distinct evidence."}
        method.data.write_json(destination, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.out.resolve())
    print(method.data.json_bytes({key: report[key] for key in ("status", "native_handmade_updates", "seconds", "max_rss_kib")}).decode())
