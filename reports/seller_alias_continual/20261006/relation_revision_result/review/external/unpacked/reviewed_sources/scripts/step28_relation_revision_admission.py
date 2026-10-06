"""Bounded revision admission on handwritten data; one invocation, no retry."""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import unittest
from unittest import mock

STARTED = time.monotonic()

import psutil
import torch

import step28_relation_revision as revision
import step28_relation_memory_verify as previous

old = revision.original
handwritten = previous.handwritten


def native_sources() -> dict:
    paths = {Path(__file__), Path(revision.__file__), Path(previous.__file__), Path(handwritten.__file__)}
    pending = list(paths)
    while pending:
        path = pending.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and name.startswith("step28_"):
                    child = old.data.ROOT / "scripts" / (name + ".py")
                    # CPU-only local import is unrelated to the native computation.
                    if child.name == "step28_relation_revision_run.py":
                        continue
                    if child not in paths:
                        paths.add(child)
                        pending.append(child)
    paths.update(old.data.ROOT / "schema" / name for name in (
        "step28_bge_continual_policy.json", "step28_chinese_base_policy.json", "step28_alias_ranking_policy.json"))
    return {str(p.relative_to(old.data.ROOT)): old.data.sha256(p) for p in sorted(paths)}


def native(check, progress) -> dict:
    progress("preflight", resources=previous.preflight())
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    c = old.config()
    progress("load_model")
    model = old.load_model(c)
    auto = model.encoder[0].auto_model
    if not auto.is_gradient_checkpointing:
        raise ValueError("Native checkpointing missing")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p != 0:
            raise ValueError("Active dropout")
        for key in ("dropout_prob", "attention_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
            value = getattr(module, key, None)
            if isinstance(value, (float, int)) and value != 0:
                raise ValueError("Active attention dropout")
    current = handwritten.handmade_group("native_current", 8, 254)
    history = handwritten.handmade_group("native_history", 8, 254)
    owners = [i for i, n in enumerate([3]*4+[2]*8) for _ in range(n)]
    owners = owners[1:]+owners[:1]
    history = replace(history, labels=tuple(int(owners[i] == owners[j])
        for i, j in itertools.combinations(range(28), 2)))
    assert history.labels != current.labels
    shapes = []
    for group in (current, history):
        texts = old.base.record_texts(group, "separate_moments")
        lengths = [len(v) for v in model.encoder.tokenizer(texts, padding=False, truncation=False)["input_ids"]]
        if len(lengths) != 448 or set(lengths) != {256}:
            raise ValueError("Native maximum input shape differs")
        shapes.append(dict(texts=448, minimum=256, maximum=256))
    progress("reference", input_shapes=shapes)
    x = old.reference(model, history, c, check)
    h, b, constant = old.statistics(torch.from_numpy(x), history.labels)
    memory = old.Memory("ABC")
    memory.h, memory.b, memory.constant, memory.count = h.numpy(), b.numpy(), constant, 1
    y = old.relation_features(model, history, c, check)
    torch.testing.assert_close(y.detach().cpu(), torch.from_numpy(x), atol=1e-5, rtol=1e-4)
    difference = float((y.detach().cpu()-torch.from_numpy(x)).abs().max())
    del y
    name, query = next((n, p) for n, p in model.encoder.named_parameters() if "query.weight" in n)
    probes = (query, model.head[0].weight, model.weight)
    before = [p.detach().clone() for p in probes]
    gradients = {}
    actual_history = revision.history_objective

    def inspect(*args):
        terms = actual_history(*args)
        for key in ("compressed", "rank", "total"):
            gs = torch.autograd.grad(terms[key], probes, retain_graph=True)
            norms = [float(g.float().norm()) for g in gs]
            if not all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in gs):
                raise ValueError("Historical component gradient missing: " + key)
            gradients[key] = norms
            del gs
            check()
        progress("history_gradients", history_only_gradient_norms=gradients)
        return terms

    calls = [0]
    def count(module, args):
        calls[0] += 1
    hook = auto.encoder.layer[0].register_forward_pre_hook(count)
    optimizer = old.make_optimizer(model, c)
    assert not optimizer.state
    tick = time.monotonic()
    try:
        with mock.patch.object(revision, "history_objective", side_effect=inspect), \
                mock.patch.object(torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip, \
                mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            row = revision.optimization_step(model, optimizer, current, history, x, memory, c, 1e-5, check=check)
            assert clip.call_count == step.call_count == 1
        torch.cuda.synchronize()
    finally:
        hook.remove()
    changes = [float((p.detach()-v).abs().max()) for p, v in zip(probes, before)]
    if not all(v > 0 for v in changes) or calls[0] <= 224 or row["adam_step"] != 1:
        raise ValueError("Missing native update or checkpoint recomputation")
    return dict(input_shapes=shapes, optimizer=row, history_only_gradient_norms=gradients,
        encoder_probe=name, probe_max_parameter_changes=changes, first_layer_calls=calls[0],
        dtypes=previous.assert_update_dtypes(model, optimizer), update_seconds=time.monotonic()-tick,
        reference_max_difference=difference, weights_saved=False, formal_data=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("native", "cpu"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    assert len(process.cpu_affinity()) == 1
    assert all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    report = dict(status="RUNNING", mode=args.mode, python=sys.version, affinity=process.cpu_affinity(),
                  started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), scientific_sources=native_sources())
    rss_limit, disk_limit = ((64*2**30, 128*2**20) if args.mode == "native" else (2*2**30, 16*2**20))
    stopped, lock = threading.Event(), threading.RLock()
    peak = 0
    def write():
        with lock:
            report.update(elapsed_seconds=time.monotonic()-STARTED, peak_rss_bytes=peak)
            if torch.cuda.is_initialized():
                report.update(max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved())
            old.data.write_json(args.out/"result.json", report)
    def check():
        nonlocal peak
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try: rss += child.memory_info().rss
            except psutil.NoSuchProcess: pass
        peak = max(peak, rss)
        size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
        if time.monotonic()-STARTED >= 585 or peak > rss_limit or size > disk_limit-65536:
            raise RuntimeError("Admission time/RSS/evidence budget exceeded")
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 28*2**30:
            raise RuntimeError("Native allocator budget exceeded")
    def progress(stage, **fields):
        with lock:
            report.update(stage=stage, **fields)
            write()
    def watch():
        while not stopped.wait(.5):
            try: check()
            except BaseException as exc:
                try:
                    report.update(status="BUDGET_STOP_NO_RETRY", error=str(exc)); write()
                finally: os._exit(2)
    def terminate(signum, frame):
        raise RuntimeError("Admission TERM; no automatic retry")
    signal.signal(signal.SIGTERM, terminate)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        progress("started")
        if args.mode == "native":
            report["native"] = native(check, progress)
        else:
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            import step28_relation_revision_run as run
            sys.path.insert(0, str(old.data.ROOT/"tests"))
            import test_step28_relation_revision_run as tests
            with (args.out/"unittest.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                    unittest.defaultTestLoader.loadTestsFromTestCase(tests.IntegrationTests))
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                          skipped=len(result.skipped), source_files=run.sources())
            if not result.wasSuccessful(): raise RuntimeError("Revision integration verification failed")
        check()
        report["status"] = "PASS_HANDWRITTEN_ONLY"
    except BaseException:
        report.update(status="FAIL_NO_RETRY", traceback=traceback.format_exc())
        raise
    finally:
        stopped.set(); watcher.join(timeout=2); write()


if __name__ == "__main__":
    main()
