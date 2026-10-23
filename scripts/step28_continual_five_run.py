"""One independent three-stage trajectory; blind endpoints and real restoration."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import psutil

import step28_continual_five as m
from step28_bge_continual_run import save_array

data, core, base, parent, diag = m.data, m.core, m.base, m.parent, m.diag
CONTRACT = "reports/documentation/20261023/continual_implementation"
COMPLETE = "COMPLETE_864_UPDATES_3_ENDPOINTS_VALID_BLIND"


def sources() -> list[dict]:
    paths = {data.ROOT/"scripts"/name for name in (
        "step28_continual_five.py", "step28_continual_five_run.py", "step28_continual_five_control.py",
        "step28_continual_five_verify.py", "step28_continual_five_evaluate.py")}
    paths.add(data.ROOT/"tests/test_step28_continual_five.py")
    todo = list(paths)
    while todo:
        path = todo.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and (name.startswith("step28_") or name.startswith("test_step28_")):
                    child = data.ROOT/("tests" if name.startswith("test_") else "scripts")/(name+".py")
                    if child not in paths:
                        paths.add(child)
                        todo.append(child)
    for name in ("step28_chinese_base", "step28_alias_ranking", "step28_record_replay", "step28_bge_continual", "step28_continual_expression"):
        paths.add(data.ROOT/"schema"/(name+"_policy.json"))
    paths.add(data.ROOT/"schema/step28_continual_expression_catalog.json")
    paths.update(data.ROOT/p for p in (
        CONTRACT+"/authorization.json", CONTRACT+"/execution.zh.txt", CONTRACT+"/gpu_review_waiver.json",
        "scripts/run_step28_continual_five_cpu_linux_20261023.sh",
        "reports/documentation/20261023/continual_baselines/proposal.zh.txt",
        "reports/documentation/20261023/continual_baselines/budget.json",
        "reports/documentation/20261023/continual_baselines/review/primary_disposition.zh.txt"))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def owned_processes(pid: int) -> dict:
    try:
        p = psutil.Process(pid)
        return {q.pid: q for q in [p, *p.children(recursive=True)] if q.is_running()}
    except psutil.NoSuchProcess:
        return {}


def resident_bytes(processes: dict) -> int:
    total = 0
    for p in processes.values():
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass  # Already exited; it owns no live memory.
    return total


class Budget:
    """Child receives the parent's actual absolute outer deadline, including tail."""
    def __init__(self, campaign: Path, deadline: float, tail: float = 60.):
        self.root, self.deadline, self.tail = campaign, deadline, tail
        self.started, self.last_checked = time.monotonic(), -float("inf")
        self.peak_rss = self.peak_disk = self.peak_small = 0

    def check(self, reserve: int = 0) -> None:
        now = time.monotonic()
        if now >= self.deadline-self.tail:
            raise TimeoutError("Allocated task allowance reached; termination tail reserved")
        if reserve or now-self.last_checked >= 5:
            self.last_checked = now
            rss = resident_bytes(owned_processes(os.getpid()))
            used = small = 0
            for p in self.root.rglob("*"):
                if p.is_file() and not p.is_symlink():
                    size = p.stat().st_size
                    used += size
                    if p.suffix != ".pt":  # Actual full/inference checkpoint payloads only.
                        small += size
            self.peak_rss, self.peak_disk = max(self.peak_rss, rss), max(self.peak_disk, used)
            self.peak_small = max(self.peak_small, small)
            if rss > 64*2**30 or used+reserve > 224*2**30:
                raise RuntimeError("Approved worker RSS or campaign storage cap exceeded")
            if small > 2*2**30:
                raise RuntimeError("Approved 2 GiB small-evidence cap exceeded")
            if shutil.disk_usage(self.root).free < reserve+64*2**30:
                raise RuntimeError("System disk safety margin unavailable")

    @contextmanager
    def saving(self, reserve: int):
        with (self.root/"checkpoint.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.check(reserve)
            yield
            fcntl.flock(lock, fcntl.LOCK_UN)

    def snapshot(self) -> dict:
        self.last_checked = -float("inf")
        self.check()
        return dict(elapsed_seconds=time.monotonic()-self.started, deadline_monotonic=self.deadline,
                    stop_before_deadline_seconds=self.tail, peak_worker_rss_bytes=self.peak_rss,
                    peak_campaign_bytes=self.peak_disk, peak_small_evidence_bytes=self.peak_small)


def save(path: Path, model, optimizer, meta: dict, budget: Budget) -> dict:
    with budget.saving(base.persistence.checkpoint_reserve(model, optimizer)):
        return core.save_state(path, model, optimizer, meta)


def checkpoint(root, method, model, optimizer, memory, current, calibration, valid, c, stage, first_map, budget):
    if (len(calibration) != 12 or len(valid) != 60 or any(g.labels is not None for g in valid)
            or parent.adam_step(optimizer) != stage*288):
        raise ValueError("Endpoint/current-cal/blind-dev/Adam scope differs")
    optimizer.zero_grad(set_to_none=True)
    name = f"stage{stage}"
    rng, modes = diag.rng_state(), [v.training for v in model.modules()]
    with diag.neutral(model):
        scores = {"calibration": method.score(model, calibration, c, budget.check),
                  "raw": method.score(model, valid, c, budget.check)}
        fit = parent.calibration.fit(scores["calibration"], np.asarray([g.labels for g in calibration], np.uint8),
                                     role="calibration", check=budget.check)
        if fit["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Calibration fit failed; no retry")
        current_map = dict(zip(("a", "b"), parent.calibration.parameters(fit)))
        first_map = current_map if first_map is None else first_map
        maps = {"first": first_map, "stage": current_map}
        if stage == 1:
            memory = method.new_memory(first_map)
        method.retain(memory, model, current, c, stage, maps, budget.check)
    memory_record = None
    if memory is not None:
        mp = root/"memory"/(name+".bin")
        mp.write_bytes(memory.to_bytes())
        memory = method.memory_from(mp.read_bytes())
        memory_record = data.record(mp, root)
    meta = {**method.metadata(), "stage": stage, "adam_step": stage*288, "rng": rng, "modes": modes,
            "maps": maps, "memory": memory_record}
    full_path, infer_path = root/"work"/(name+".pt"), root/"models"/(name+".pt")
    full = save(full_path, model, optimizer, meta, budget)
    if method.restore(full_path, full, model, optimizer) != meta:
        raise ValueError("Full restored metadata differs")
    with diag.neutral(model):
        for role, groups in (("calibration", calibration), ("raw", valid)):
            values = method.score(model, groups, c, budget.check)
            if not np.array_equal(scores[role], values):
                raise ValueError("Full model/Adam restored prediction differs")
            scores["full-"+role] = values
    inference = save(infer_path, model, None, meta, budget)
    if method.restore(infer_path, inference, model, None) != meta:
        raise ValueError("Inference restored metadata differs")
    with diag.neutral(model):
        for role, groups in (("calibration", calibration), ("raw", valid)):
            values = method.score(model, groups, c, budget.check)
            if not np.array_equal(scores[role], values):
                raise ValueError("Inference restored prediction differs")
            scores["inference-"+role] = values
    for role, mapping in (("stage-cal", current_map), ("first-cal", first_map)):
        scores[role] = parent.calibration.transform(scores["raw"], mapping)
        parent.calibration.preserve_order(scores["raw"], scores[role])
    map_path = root/"maps"/(name+".json")
    data.write_json(map_path, fit)
    result = {**meta, "calibration_ids": [g.uid for g in calibration],
              "development_ids": [g.uid for g in valid], "mapping": data.record(map_path, root),
              "full_restore_verified": True, "inference_restore_verified": True,
              "memory_summary": None if memory is None else memory.summary(),
              "full_state": {**full, "path": full_path.relative_to(root).as_posix()},
              "model": {**inference, "path": infer_path.relative_to(root).as_posix()},
              "scores": {k: save_array(root/"scores"/(name+"_"+k+".npy"), v, root) for k, v in scores.items()},
              "continuity": {"model": core.state_digest(model.state_dict()),
                             "adam": core.state_digest(optimizer.state_dict()), "rng": diag.rng_state()}}
    data.write_json(root/"points"/(name+".json"), result)
    base.persistence.remove_work_file(full_path, root)
    return result, memory, first_map


def train_stage(method, model, optimizer, current, memory, c, stage, root, budget) -> dict:
    """Learner receives only this stage's 48 groups and surviving history."""
    if parent.adam_step(optimizer) != (stage-1)*288:
        raise ValueError("Three-stage Adam continuity failed")
    if memory is not None:
        memory.begin_stage(stage)
    before = None if memory is None else memory.summary()
    sequence, stream = parent.schedule(current, c, method.spec.order, stage)
    rows = []
    for step, group in enumerate(sequence, 1):
        history, target = memory.draw() if memory is not None else (None, None)
        row = method.update(model, optimizer, group, history, target, c, stage, step, budget.check)
        rows.append({**row, "step": step, "current_uid": group.uid,
                     "history_uid": history.uid if history is not None else None})
        if step == 1 or step % 24 == 0:
            print(data.json_bytes(dict(event="updates", run_id=method.spec.run_id, stage=stage,
                                       completed=step, **budget.snapshot())).decode(), flush=True)
    after = None if memory is None else memory.summary()
    if before is not None:
        keys = ("members", "seen", "origins", "tables") if method.spec.is_record else ("members", "seen", "references", "reference_origins")
        if any(before[k] != after[k] for k in keys) or after.get("count", after.get("draw_count")) != 288:
            raise ValueError("Historical membership/targets changed or missing draws")
    path = root/"updates"/f"stage{stage}.json"
    data.write_json(path, dict(stage=stage, updates=288, adam_step=parent.adam_step(optimizer),
        actual_domain=method.spec.order[stage-1], current_stream=stream, rows=rows,
        memory_before=before, memory_after=after))
    return data.record(path, root)


def execute(job: Path, spec: m.Spec, deadline: float, gate_path: Path) -> dict:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gate = data.read_json(gate_path)
    if (gate.get("status") != "APPROVED_IMPLEMENTATION_AND_RESOURCES" or gate["sources"] != sources()
            or gate["runs"] != [s.identity() for s in m.specifications()]):
        raise ValueError("Exact implementation/qualification/45-run gate missing")
    budget = Budget(job.parent, deadline)
    budget.check()
    c, method = m.config(spec), m.Method(spec)
    parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(gate["allocator_bytes"]/torch.cuda.get_device_properties(0).total_memory)
    archive = core.model_files(base.model_config(c, "split_rank"))
    pinned = c["models"]["split_rank"]
    if any(archive[k] != pinned[k] for k in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Pretrained archive differs")
    root = job/"run"
    for sub in ("models", "work", "memory", "points", "maps", "scores", "updates"):
        (root/sub).mkdir(parents=True)
    # Offline dispatch scope may index the full authorized archive. Neither this
    # object nor its future/evicted groups is supplied to train_stage/update.
    public, metadata, checked = base.public.public_inputs(c)
    _, partition = base.partition(public, metadata, c)
    access = dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
    data.write_json(job/"access.json", access)
    public["train"] = base.public.attach_labels(public["train"], c, "train")
    groups, actual_partition = base.partition(public, metadata, c)
    if actual_partition != partition:
        raise ValueError("Labels changed public partition")
    supply = parent.Supply(groups, partition)
    valid = groups["development"]
    manifest = dict(status="RUNNING", spec=spec.identity(), source_files=gate["sources"], public_inputs=checked,
                    partition=partition, pretrained_archive=archive, points={}, training={}, updates=0,
                    gradient_group_presentations=0, access=access)
    data.write_json(root/"startup.json", manifest)
    del public, groups, metadata
    model, optimizer = method.fresh(c)
    manifest["initial"] = dict(model=core.state_digest(model.state_dict()), adam=core.state_digest(optimizer.state_dict()))
    memory, first_map = None, None
    for stage in (1, 2, 3):
        current, calibration = supply.current(spec.run_id, spec.order, stage)
        manifest["training"][str(stage)] = train_stage(method, model, optimizer, current, memory, c, stage, root, budget)
        point, memory, first_map = checkpoint(root, method, model, optimizer, memory, current, calibration,
                                              valid, c, stage, first_map, budget)
        manifest["points"][str(stage)] = point
        manifest["updates"] += 288
        manifest["gradient_group_presentations"] += 288*(1+(stage > 1 and spec.name != "SEQ_ref"))
        data.write_json(root/"progress.json", manifest)
        del current, calibration
    manifest.update(status=COMPLETE, resources=budget.snapshot())
    data.write_json(root/"completion.json", manifest)
    del model, optimizer, memory, supply
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.job, m.Spec.from_dict(data.read_json(args.spec)), args.deadline, args.gate)
    except BaseException as error:
        data.write_json(args.job/"failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
                                                    status="STOP_NO_RETRY", time=time.time()))
        raise


if __name__ == "__main__":
    main()
