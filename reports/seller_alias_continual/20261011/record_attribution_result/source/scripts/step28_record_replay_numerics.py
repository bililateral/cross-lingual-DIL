"""One bounded native C/S update comparison on handwritten inputs; no formal data."""
from __future__ import annotations

import time
STARTED = time.monotonic()

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import traceback

import numpy as np
import psutil
import torch

import step28_record_replay as method
import step28_record_replay_run as run
from step28_record_replay_verify import preflight


def old_numerics() -> dict:
    # Literal settings of the SHA-bound historical LOGIT execute, lines 520--523.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return observed()


def observed() -> dict:
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def equal_state(left, right, counts, path="state"):
    if isinstance(left, torch.Tensor):
        if (left.dtype != right.dtype or left.shape != right.shape or
                not torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8))):
            raise AssertionError("Native tensor differs: " + path)
        counts["tensors"] += 1
        counts["elements"] += left.numel()
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("State keys differ: " + path)
        for key in left:
            equal_state(left[key], right[key], counts, path + "/" + str(key))
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("State sequence differs: " + path)
        for i, (a, b) in enumerate(zip(left, right)):
            equal_state(a, b, counts, path + "/" + str(i))
    elif left != right:
        raise AssertionError("State scalar differs: " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if os.name != "posix" or not out.is_relative_to(method.data.ROOT / "reports") or out.exists():
        raise ValueError("New Linux reports output required; no automatic retry")
    out.mkdir(parents=True)
    report = {"status": "RUNNING", "formal_inputs": False, "formal_labels": False,
              "limits": {"wall_seconds": 300, "gpu_bytes": 28*2**30,
                         "rss_bytes": 64*2**30, "output_bytes": 4*2**30}, "comparisons": {}}
    process, stopped, lock = psutil.Process(), threading.Event(), threading.RLock()
    def write():
        with lock:
            report["elapsed_seconds"] = time.monotonic() - STARTED
            method.data.write_json(out / "result.json", report)
    def check():
        report["peak_rss_bytes"] = max(report.get("peak_rss_bytes", 0), process.memory_info().rss)
        if torch.cuda.is_initialized():
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if (time.monotonic() - STARTED > 285 or report["peak_rss_bytes"] > 64*2**30
                or report.get("peak_reserved_bytes", 0) > 28*2**30
                or sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 4*2**30):
            raise RuntimeError("Approved native maintenance budget exceeded")
    def watchdog():
        while not stopped.wait(.5):
            try:
                check()
            except BaseException as exc:
                report.update(status="BUDGET_STOP_NO_RETRY", error=str(exc))
                write()
                os._exit(2)
    watcher = threading.Thread(target=watchdog, daemon=True)
    write()
    watcher.start()
    try:
        report["resources"] = preflight()
        report["source_files"] = method.data.read_json(args.sources)
        for rec in report["source_files"]:
            method.data.verify(method.data.ROOT / rec["path"], rec)
        old = method.data.ROOT / "reports/seller_alias_continual/20261009/replay_improvement_result/analysis/output/historical/step28_bge_continual_run.py"
        if method.data.sha256(old) != "bc425a88e502f9802aac81a0ac0977368606ed12c4e87e4bb2a6020ce9804db9":
            raise ValueError("Historical setting source changed")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("cuBLAS configuration must precede Python")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(28*2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
        report["environment"] = {"python": sys.version, "torch": torch.__version__,
                                 "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        sys.path.insert(0, str(method.data.ROOT / "tests"))
        from test_step28_record_replay import handmade_group
        current, history = [handmade_group("numerics_" + role, 2) for role in ("current", "history")]
        c = method.config()
        model_record = c["models"]["split_rank"]
        archive = method.core.model_files(method.base.model_config(c, "split_rank"))
        for key in ("file_count", "total_size_bytes", "content_sha256"):
            if archive[key] != model_record[key]:
                raise ValueError("Pinned Chinese model archive differs")
        report["model"] = model_record
        old_numerics()
        model = method.load_model(c)
        if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
            raise ValueError("Native BF16/checkpoint mode unavailable")
        teacher = method.reference(model, history, c, check)
        with torch.no_grad():
            model.head[0].weight.add_(.003)
        initial = cpu_copy(model.state_dict())
        initial_digest = method.core.state_digest(initial)
        rng, py_rng, np_rng = run.previous.rng_state(), random.getstate(), np.random.get_state()
        report["fixture"] = {"records_per_group": 56, "current": current.uid, "history": history.uid,
                             "teacher_values": len(teacher), "teacher_sha256": hashlib.sha256(teacher.tobytes()).hexdigest(),
                             "initial_state_sha256": initial_digest, "stage": 2, "step": 1,
                             "optimizer_initial_state": "fresh AdamW; no full stage continuation claim"}
        for arm in ("C", "S"):
            expected = None
            runs = []
            for label, setup in (("historical_explicit", old_numerics), ("repaired_shared", method.parent.configure_numerics)):
                check()
                report.update(stage="native_update", current_arm=arm, current_profile=label)
                write()
                # Start each profile from wrong flags, so setup cannot pass by inheritance.
                torch.use_deterministic_algorithms(False, warn_only=True)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                numerics = setup()
                if numerics != observed() or numerics != {"deterministic_algorithms": True,
                        "deterministic_warn_only": False, "cuda_matmul_allow_tf32": False,
                        "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8"}:
                    raise AssertionError("Effective numerical settings differ")
                model.load_state_dict(initial, strict=True)
                model.zero_grad(set_to_none=True)
                optimizer = method.core.make_optimizer(model, c)
                run.previous.restore_rng(rng)
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                if method.core.state_digest(model.state_dict()) != initial_digest or optimizer.state:
                    raise AssertionError("Initial parameters or optimizer differ")
                probes = {"encoder_query": model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
                          "head_hidden": model.head[0].weight, "head_output": model.head[2].weight}
                probe_before = {k: p.detach().clone() for k, p in probes.items()}
                previous, increments = {}, {}
                def observe(role, log):
                    increments[role] = {}
                    for name, param in probes.items():
                        delta = param.grad.detach() - previous.get(name, torch.zeros_like(param))
                        norm = float(delta.norm())
                        if not torch.isfinite(delta).all() or norm <= 0:
                            raise AssertionError("Missing native branch gradient: " + name)
                        increments[role][name] = norm
                        previous[name] = param.grad.detach().clone()
                tick = time.monotonic()
                log = method.update(model, optimizer, current, history, teacher, c, arm, "ABC", 2, 1, check, observe)
                torch.cuda.synchronize()
                changed = {k: not torch.equal(p, probe_before[k]) for k, p in probes.items()}
                if not all(changed.values()) or method.parent.adam_step(optimizer) != 1:
                    raise AssertionError("Actual encoder/head optimizer update missing")
                if not (out / "first_update.json").exists():
                    (out / "first_update.json").write_text(json.dumps({"arm": arm, "profile": label, "adam_step": 1}) + "\n")
                state = {"parameters": cpu_copy(model.state_dict()),
                         "gradients": {name: cpu_copy(p.grad) for name, p in model.named_parameters()},
                         "optimizer": cpu_copy(optimizer.state_dict()), "rng": run.previous.rng_state(),
                         "loss_log": log, "branch_gradient_norms": increments}
                row = {"profile": label, "numerics": numerics, "update_seconds": time.monotonic()-tick,
                       "state_sha256": method.core.state_digest(state), "loss_log": log,
                       "branch_gradient_norms": increments, "parameters_changed": changed}
                runs.append(row)
                if expected is None:
                    expected = state
                else:
                    counts = {"tensors": 0, "elements": 0}
                    equal_state(expected, state, counts)
                    report["comparisons"][arm] = {"status": "BITWISE_EQUAL", **counts, "runs": runs}
                    write()
                del state, optimizer, probe_before, previous
            del expected
        if hashlib.sha256(teacher.tobytes()).hexdigest() != report["fixture"]["teacher_sha256"]:
            raise AssertionError("Teacher target mutated")
        check()
        report["status"] = "PASS_NATIVE_NUMERICS_C_S"
    except BaseException:
        report.update(status="FAILED_NO_AUTOMATIC_RETRY", traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
        write()


if __name__ == "__main__":
    main()
