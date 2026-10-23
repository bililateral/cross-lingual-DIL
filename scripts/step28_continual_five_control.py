"""One owner of the fixed queue, three children and cumulative reservations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
CONTROLLER_STARTED = time.monotonic()

import psutil

import step28_continual_five as m
import step28_continual_five_run as run

data = m.data
LIMITS = dict(training=1620000., confirmation=129600., cpu=1200., native=5400.,
              generation=600., control=1800., evaluation=3600., analysis=1800.)


class Ledger:
    """Only the controller writes. Active allowances already include elapsed time."""
    def __init__(self, path: Path | None = None, clock=time.monotonic):
        self.path, self.clock = path, clock
        self.used = {k: 0. for k in LIMITS}
        self.active, self.completed = {}, []
        self.failed = False

    def save(self) -> None:
        if self.path:
            temporary = self.path.with_suffix(".tmp")
            data.write_json(temporary, self.snapshot())
            os.replace(temporary, self.path)

    def snapshot(self) -> dict:
        return dict(used=self.used, active=self.active, completed=self.completed, failed=self.failed,
                    cumulative_seconds=sum(self.used.values()), limits=LIMITS)

    def available(self, bucket: str) -> float:
        return LIMITS[bucket]-self.used[bucket]-sum(v["allowance"] for v in self.active.values() if v["bucket"] == bucket)

    def reserve(self, identity: str, bucket: str, allowance: float, *, remaining_projection: float = 0.) -> dict:
        if (self.failed or identity in self.active or any(v["id"] == identity for v in self.completed)
                or allowance <= 0 or allowance > self.available(bucket)
                or remaining_projection+allowance > self.available(bucket)
                or (bucket == "training" and allowance > 86400)):
            raise ValueError("Budget/identity/whole remaining campaign cannot be reserved")
        now = self.clock()
        row = dict(id=identity, bucket=bucket, started=now, allowance=allowance, deadline=now+allowance)
        self.active[identity] = row
        self.save()
        return row

    def settle(self, identity: str, *, success: bool) -> dict:
        row = self.active.pop(identity)
        actual = self.clock()-row["started"]
        self.used[row["bucket"]] += actual
        row = {**row, "actual_seconds": actual, "success": success}
        self.completed.append(row)
        self.failed |= not success or actual > row["allowance"] or self.used[row["bucket"]] > LIMITS[row["bucket"]]
        self.save()
        if self.failed:
            raise RuntimeError("Failed/over-budget task retained; stop all dispatch")
        return row

    def charge(self, bucket: str, actual: float) -> None:
        self.used[bucket] += actual
        over = self.available(bucket) < 0
        self.failed |= over
        self.save()
        if over:
            raise RuntimeError("Cumulative bucket exceeded")


def joint_opening(records: list[dict], expected_sources: list[dict]) -> None:
    expected = {s.run_id: s for s in m.specifications()}
    found = {}
    for row in records:
        spec = m.Spec.from_dict(row["spec"])
        if spec.run_id in found:
            raise ValueError("Duplicate run identity is not a complete set")
        found[spec.run_id] = row
        if (row.get("status") != run.COMPLETE or row.get("source_files") != expected_sources
                or row.get("updates") != 864 or row.get("gradient_group_presentations") != (864 if spec.name == "SEQ_ref" else 1440)
                or row.get("access") != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
                or set(row.get("points", {})) != {"1", "2", "3"}
                or set(row.get("training", {})) != {"1", "2", "3"}):
            raise ValueError("Run completion/source/permission coverage differs")
        for stage, point in row["points"].items():
            if (point.get("continual_spec") != spec.identity() or point.get("adam_step") != 288*int(stage)
                    or not point.get("full_restore_verified") or not point.get("inference_restore_verified")
                    or len(set(point.get("development_ids", []))) != 60
                    or len(set(point.get("calibration_ids", []))) != 12
                    or set(point.get("scores", {})) != {"calibration", "raw", "full-calibration", "full-raw",
                                                        "inference-calibration", "inference-raw", "stage-cal", "first-cal"}):
                raise ValueError("Incomplete blind endpoint/restore evidence")
    if set(found) != set(expected):
        raise ValueError("All 45 exact method/order/seed identities required before opening")
    for seed in m.SEEDS:
        for record_family in (False, True):
            initials = {row["initial"]["model"] for identity, row in found.items()
                        if expected[identity].seed == seed and expected[identity].is_record == record_family}
            if len(initials) != 1:
                raise ValueError("Paired common initialization differs within an architecture/seed")
        for order in m.ORDERS:
            rows = [row for identity, row in found.items() if expected[identity].seed == seed
                    and expected[identity].order == order and expected[identity].name != "SEQ_ref"]
            for stage in ("1", "2", "3"):
                memberships = [row["points"][stage]["memory_summary"]["members"] for row in rows]
                if any(v != memberships[0] or len(v) != 6 for v in memberships):
                    raise ValueError("Paired Algorithm R membership differs across methods")


def validate_blind_files(campaign: Path, records: list[dict]) -> None:
    import numpy as np
    for row in records:
        spec = m.Spec.from_dict(row["spec"])
        root = campaign/spec.run_id/"run"
        for stage, point in row["points"].items():
            for rec in (point["model"], point["mapping"], row["training"][stage]):
                data.verify(root/rec["path"], rec)
            if point["memory"] is not None:
                rec = point["memory"]
                # Hashing the retained payload does not reopen it to a learner.
                data.verify(root/rec["path"], rec)
            values = {}
            for role, rec in point["scores"].items():
                value = np.load(data.verify(root/rec["path"], rec), allow_pickle=False)
                count = 12 if "calibration" in role else 60
                if value.shape != (count, 378) or not np.isfinite(value).all():
                    raise ValueError("Invalid blind-score shape/finite coverage")
                values[role] = value
            for role in ("raw", "calibration"):
                if not all(np.array_equal(values[role], values[prefix+role]) for prefix in ("full-", "inference-")):
                    raise ValueError("Saved actual restored scores differ")


def terminate_owned(process: subprocess.Popen) -> float:
    """Return parent wait time, already covered by the child's outer elapsed."""
    waited = 0.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        waiting = time.monotonic()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            waited += time.monotonic()-waiting
            os.killpg(process.pid, signal.SIGKILL)
            waiting = time.monotonic()
            process.wait(timeout=10)
        waited += time.monotonic()-waiting
    return waited


def dispatch(queue: list[dict], campaign: Path, ledger: Ledger, bucket: str,
             command, *, max_workers: int = 3, tail: float = 60., admit=None,
             poll_seconds: float = 30.) -> list[dict]:
    """Fixed queue, no replacement or retry. Wait time is not controller work."""
    active, completed, next_index = {}, [], 0
    cores = psutil.Process().cpu_affinity()[:max_workers]
    if len(cores) != max_workers:
        raise RuntimeError("Distinct single-core workers unavailable")
    pending_error, tick = None, None
    try:
        while active or next_index < len(queue):
            tick = time.monotonic()
            for identity, (process, log, slot) in list(active.items()):
                if process.poll() is None and time.monotonic() >= ledger.active[identity]["deadline"]-tail:
                    tick += terminate_owned(process)
                    raise TimeoutError("Child stopped within its reserved termination tail")
                if process.poll() is not None:
                    log.close()
                    del active[identity]
                    completed.append(ledger.settle(identity, success=process.returncode == 0))
            owned = run.owned_processes(os.getpid())
            rss = run.resident_bytes(owned)
            if rss > 192*2**30:
                raise RuntimeError("Campaign owned-PID-union RSS exceeded")
            while len(active) < max_workers and next_index < len(queue):
                row = queue[next_index]
                if admit:
                    admit(len(active))
                remaining = sum(q.get("projected_seconds", q["allowance"]) for q in queue[next_index+1:])
                allowance = min(row["allowance"], ledger.available(bucket)-remaining)
                if allowance < row.get("projected_seconds", row["allowance"]):
                    raise RuntimeError("Available allowance cannot cover this run and the pending campaign")
                reservation = ledger.reserve(row["id"], bucket, allowance, remaining_projection=remaining)
                slot = next(v for v in range(max_workers) if v not in [a[2] for a in active.values()])
                job = campaign/row["id"]
                job.mkdir()
                command_line = ["taskset", "-c", str(cores[slot]), *command(row, job, reservation["deadline"])]
                data.write_json(job/"dispatch.json", dict(reservation=reservation, command=command_line,
                                                        source_files=run.sources()))
                log = (job/"console.txt").open("w", encoding="utf-8")
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1"}
                try:
                    process = subprocess.Popen(command_line, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                except BaseException:
                    log.close()
                    ledger.settle(row["id"], success=False)
                    raise
                active[row["id"]] = (process, log, slot)
                data.write_json(job/"process.json", dict(pid=process.pid, owner=psutil.Process(process.pid).username(),
                                                       created=psutil.Process(process.pid).create_time()))
                next_index += 1
            elapsed = time.monotonic()-tick
            tick = None
            ledger.charge("control", elapsed)
            if active:
                next_stop = min(ledger.active[identity]["deadline"]-tail for identity in active)
                time.sleep(max(0., min(poll_seconds, next_stop-time.monotonic())))
    except BaseException as error:
        pending_error = error
    finally:
        if tick is not None:
            try:
                ledger.charge("control", time.monotonic()-tick)
            except RuntimeError as error:
                pending_error = error
        cleanup_started, cleanup_waited = time.monotonic(), 0.
        for identity, (process, log, _) in list(active.items()):
            try:
                cleanup_waited += terminate_owned(process)
            finally:
                log.close()
                try:
                    ledger.settle(identity, success=False)
                except RuntimeError:
                    pass
        # A timeout can remove the child from active only after settling it.
        for identity in list(ledger.active):
            try:
                ledger.settle(identity, success=False)
            except RuntimeError:
                pass
        ledger.charge("control", time.monotonic()-cleanup_started-cleanup_waited)
    if pending_error:
        raise pending_error
    return completed


def resource_admission(gate: dict, active: int, path: Path, *, startup: bool = False) -> dict:
    import shutil
    query = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid,memory.total,memory.free",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(",")
    applications = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_memory",
                                            "--format=csv,noheader,nounits"], text=True).strip()
    own = run.owned_processes(os.getpid())
    foreign = [line for line in applications.splitlines() if int(line.split(",")[0]) not in own]
    free = int(query[2])*2**20
    required = gate["minimum_free_gpu_bytes"] if active == 0 else gate["minimum_free_slot_bytes"]
    if (free < required or foreign or psutil.virtual_memory().available < 64*2**30
            or (startup and (psutil.virtual_memory().available < 208*2**30
                             or shutil.disk_usage(path).free < 288*2**30))):
        raise RuntimeError("Actual resource admission unavailable; no launch")
    return dict(gpu_uuid=query[0].strip(), free_gpu_bytes=free, compute_apps=applications,
                host_available_bytes=psutil.virtual_memory().available, disk_free_bytes=shutil.disk_usage(path).free)


def execute(campaign: Path, gate_path: Path) -> dict:
    gate = data.read_json(gate_path)
    if (gate.get("status") != "APPROVED_IMPLEMENTATION_AND_RESOURCES" or gate["sources"] != run.sources()
            or gate["runs"] != [s.identity() for s in m.specifications()] or campaign.exists()):
        raise ValueError("Fresh approved complete campaign required")
    preflight = resource_admission(gate, 0, campaign.parent, startup=True)
    campaign.mkdir(parents=True)
    data.write_json(campaign/"preflight.json", preflight)
    ledger = Ledger(campaign/"budget.json")
    ledger.used.update(gate["prior_charged_seconds"])
    ledger.charge("control", time.monotonic()-CONTROLLER_STARTED)
    ledger.save()
    queue = [dict(id=s.run_id, spec=s.identity(), allowance=gate["per_run_allowance_seconds"][s.name],
                  projected_seconds=gate["per_run_projected_seconds"][s.name]) for s in m.specifications()]
    def command(row, job, deadline):
        data.write_json(job/"spec.json", row["spec"])
        return [sys.executable, "-B", str(Path(run.__file__)), "--job", str(job), "--spec", str(job/"spec.json"),
                "--deadline", str(deadline), "--gate", str(gate_path)]
    try:
        receipts = dispatch(queue, campaign, ledger, "training", command,
                            admit=lambda active: resource_admission(gate, active, campaign))
        control_tick = time.monotonic()
        records = [data.read_json(campaign/s.run_id/"run/completion.json") for s in m.specifications()]
        joint_opening(records, gate["sources"])
        result = dict(status="ALL_45_BLIND_COMPLETE", run_ids=[s.run_id for s in m.specifications()],
                      receipts=receipts, sources=gate["sources"], budget=ledger.snapshot())
        data.write_json(campaign/"training_complete.json", result)
        ledger.charge("control", time.monotonic()-control_tick)
        # No branch on old-dev effects: the already frozen confirmation follows.
        from step28_continual_five_evaluate import __file__ as evaluator
        for identity, action, bucket in (("old_dev", "old-dev", "evaluation"), ("generation", "generate", "generation"),
                                         ("confirmation", "confirmation", "confirmation"), ("analysis", "analyze", "analysis")):
            row = dict(id=identity, allowance=ledger.available(bucket))
            def post_command(q, job, deadline):
                return [sys.executable, "-B", evaluator, action, "--campaign", str(campaign), "--job", str(job),
                        "--gate", str(gate_path), "--deadline", str(deadline)]
            dispatch([row], campaign, ledger, bucket, post_command, max_workers=1)
        control_tick = time.monotonic()
        result.update(status="ALL_45_AND_FROZEN_CONFIRMATION_ANALYSIS_COMPLETE", budget=ledger.snapshot())
        data.write_json(campaign/"completion.json", result)
        ledger.charge("control", time.monotonic()-control_tick)
        return result
    except BaseException as error:
        data.write_json(campaign/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error),
                                                     traceback=traceback.format_exc(), budget=ledger.snapshot()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    execute(args.campaign, args.gate)
