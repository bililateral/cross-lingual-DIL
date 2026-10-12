#!/usr/bin/env python3
"""Unexecuted local PyTorch probe supplied for independent primary checking.

NOT RUN in this webpage review: torch is not installed. This file must not be
counted as Linux py310 evidence or as a native/GPU qualification. If the primary
executor chooses to use it, run within the existing handwritten CPU budget.
It uses the actual submitted restore function via AST, and tiny handwritten
CPU tensors only. It imports no project modules and loads no model/data.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import types
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    import torch  # No automatic install or fallback.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    inputs = json.loads(args.inputs.read_text())
    assert hashlib.sha256(args.input_zip.read_bytes()).hexdigest() == inputs["input_zip_sha256"]
    c = inputs["torch_probe_handwritten"]
    with zipfile.ZipFile(args.input_zip) as z:
        submitted = z.read("scripts/step28_crossfit_retention_verify.py").decode()
        numerics = z.read("scripts/step28_record_replay_numerics.py").decode()
    env = {"torch": torch, "copy": copy,
           "diag": types.SimpleNamespace(restore_rng=lambda state: torch.set_rng_state(state))}
    nodes = []
    for text, names in ((numerics, {"cpu_copy"}), (submitted, {"state", "restore"})):
        nodes.extend(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names)
    tree = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, "actual_submitted_restore_ast", "exec"), env)
    cpu_copy, state, restore = (env[n] for n in ("cpu_copy", "state", "restore"))
    def signature(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            return {"shape": list(t.shape), "dtype": str(t.dtype),
                    "bytes": t.reshape(-1).view(torch.uint8).tolist()}
        if isinstance(v, dict):
            return {str(k): signature(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [signature(x) for x in v]
        return v
    def sha(v):
        return hashlib.sha256(json.dumps(signature(v), sort_keys=True).encode()).hexdigest()
    def trial(clone_each_load):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([c["initial_parameters"]]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]),
                                      eps=c["eps"], weight_decay=c["weight_decay"],
                                      foreach=False, capturable=False, fused=False)
        x = torch.tensor([c["linear_loss_coefficients"]])
        def update():
            optimizer.zero_grad(set_to_none=True)
            model(x).sum().backward()
            optimizer.step()
        update()  # One actual update allocates moments, as in native fixture.
        for s in optimizer.state.values():
            s["step"].fill_(c["baseline_step"])
        saved = state(model, optimizer)
        saved_sha = sha(saved["adam"])
        rng = torch.get_rng_state().clone()
        modes = [m.training for m in model.modules()]
        events = []
        for k in range(2):
            incoming = {**saved, "adam": cpu_copy(saved["adam"])} if clone_each_load else saved
            restore(model, optimizer, incoming, rng, modes)
            p = next(model.parameters())
            live_step = optimizer.state[p]["step"]
            saved_step = next(iter(saved["adam"]["state"].values()))["step"]
            step_before = int(live_step.item())
            row = {"attempt": k + 1, "step_before": step_before,
                   "step_aliases_saved": live_step.data_ptr() == saved_step.data_ptr()}
            if step_before != c["baseline_step"]:
                row.update(status="REJECTED_BEFORE_UPDATE", actual_update=False)
                events.append(row)
                break
            update()
            row.update(status="ACTUAL_TINY_CPU_ADAMW_UPDATE", actual_update=True,
                       live_step_after=int(live_step.item()), saved_step_after=int(saved_step.item()),
                       saved_adam_unchanged=sha(saved["adam"]) == saved_sha)
            events.append(row)
        return {"clone_each_load": clone_each_load, "events": events,
                "saved_adam_unchanged": sha(saved["adam"]) == saved_sha}
    original, cloned = trial(False), trial(True)
    if not cloned["saved_adam_unchanged"] or len(cloned["events"]) != 2:
        raise AssertionError("Clone-on-restore failed to preserve handwritten baseline")
    print(json.dumps({"environment": "Actual environment of whoever executes this file",
                      "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                      "elapsed_seconds": time.monotonic() - started, "device": "cpu",
                      "project_modules_imported": False, "formal_inputs": False, "gpu": False,
                      "submitted": original, "clone_on_restore": cloned,
                      "interpretation": "Runtime-specific observation; this is not the seven-update native qualification."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
