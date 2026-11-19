"""Use the FIT12 dispatch loop for the Chinese benchmark's one training queue."""
import errno
from pathlib import Path
import signal
import sys
import time

import psutil

from step30_evolution_control import kernel_identity
from step30_evolution_io import Ledger, PYTHON, read, record, verify, write
from step35_market_queue import MarketRunner, OomHandoff, attempt_job, request_recovery


POLICY = "vendorlink_manual_cuda_oom_once"
SHARED_SOURCES = ("step30_evolution_control.py", "step30_evolution_io.py",
                  "step30_evolution_policy.py", "step35_market_queue.py")
# The shared runner monitors GPU jobs in 'gpu' and its own time in 'auxiliary'.
BUCKETS = {"training": "training", "generation": "generation", "confirmation": "gpu",
           "analysis": "analysis", "control": "auxiliary"}


class ChineseRunner(MarketRunner):
    active_poll_seconds = 2.
    worker_threads = 1

    def verify_worker_sources(self, job):
        try:
            for binding in self.source_bindings:
                verify(binding)
        except (OSError, ValueError) as exc:
            write(job["target"]/"shared_failure.json", dict(error=repr(exc)))
            raise

    def command(self, job, slot, deadline):
        self.verify_worker_sources(job)
        cores = psutil.Process().cpu_affinity()
        action = job["action"]
        args = ["--id", job["task_id"] if action == "worker" else action, "--deadline", str(deadline)]
        if action == "worker":
            args += ["--attempt", str(job["attempt"])]
        return ["taskset", "-c", str(cores[slot]), PYTHON, "-B", str(self.worker_entry), action,
                "--gate", str(self.gate_path), *args]


def training_phase(runner, rows, workers, protocol, validate):
    parent = psutil.Process()
    write(runner.root/"controller.json", dict(protocol=protocol, pid=parent.pid, owner=parent.username(),
          cwd=parent.cwd(), command=parent.cmdline(), **kernel_identity(parent.pid)))
    jobs = [attempt_job(dict(task_id=r["id"], task_dir=runner.root/"training"/r["id"], action="worker",
                       seconds=r["seconds"], bucket="training"), 1) for r in rows]
    handoff = OomHandoff(jobs, workers, paired_start=False, cores_per_slot=1, stop_serial_exhausted=False)
    by_id = {r["id"]: r for r in rows}
    def accepted(job):
        try:
            if job["attempt"] == 2:
                verify(job["first_oom_receipt"])
            validate(job["target"], by_id[job["task_id"]])
        except BaseException as exc:
            # Failed source/endpoint validation or parent interruption is shared.
            write(job["target"]/"shared_failure.json", dict(error=repr(exc)))
            raise
    runner.run_many(jobs, workers, accepted, recovery=handoff)
    if set(handoff.winners) != set(by_id):
        raise ValueError("Every training identity needs one successful attempt")
    write(runner.root/"training_complete.json", dict(status="ALL_TRAINING_COMPLETE_EVALUATION_CLOSED",
          sources=runner.allowed["sources"], winners=handoff.winners))


def winner_directory(campaign, task_id, winner):
    number = winner["attempt"]
    if type(number) is not int or number not in (1, 2):
        raise ValueError("Invalid winning attempt")
    root = campaign/"training"/task_id
    job = root/f"attempt-{number}"
    def receipt(binding, path):
        if Path(binding["path"]) != path:
            raise ValueError("Winner refers to a different training attempt")
        return read(verify(binding))
    success = receipt(winner["receipt"], job/"dispatch_receipt.json")
    if (success["task_id"], success["attempt"], success["success"], success["released"]) != (task_id, number, True, True):
        raise ValueError("Successful released receipt required")
    if number == 2:
        prior = receipt(winner["first_oom_receipt"], root/"attempt-1"/"dispatch_receipt.json")
        reopened = read(root/"attempt-1"/"slot_reopened.json")
        if (reopened["previous"]["receipt"] != winner["first_oom_receipt"] or
                (reopened["next_task"], reopened["next_attempt"]) != (task_id, 2)):
            raise ValueError("Winner differs from the first failure accepted for recovery")
        if (prior["task_id"], prior["attempt"], prior["success"], prior["released"],
            prior["failure_kind"], prior["slot"]) != (task_id, 1, False, True, "CUDA_OOM", success["slot"]):
            raise ValueError("Second attempt requires the original released CUDA OOM in the same slot")
    if Path(winner["completion"]["path"]) != job/"run"/"completion.json":
        raise ValueError("Winning completion path differs")
    verify(winner["completion"])
    return job


def record_worker_failure(job, action, error, traceback_text):
    import torch
    message = str(error).lower()
    oom = action == "worker" and (
        (isinstance(error, torch.cuda.OutOfMemoryError) and "cpu" not in message)
        or "cuda out of memory" in message or "cuda error: out of memory" in message)
    evidence = dict(kind="CUDA_OOM" if oom else "OTHER", error=repr(error), traceback=traceback_text)
    write(job/"worker_failure.json", evidence)
    write(job/"failure.json", evidence)
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        write(job/"shared_failure.json", dict(error=repr(error)))


def campaign(allowed, gate, rows, entry, validate, post_status):
    started = time.monotonic()
    root = Path(allowed["campaign"])
    caps = {BUCKETS[k]: v for k, v in allowed["budget_seconds"].items()}
    prior = {BUCKETS[k]: v for k, v in allowed["prior_used_seconds"].items()}
    # Reserve all first attempts before claiming the new directory. Retries reserve
    # their full method allowance against remaining and in-flight time at dispatch.
    if (sum(r["seconds"] for r in rows)+prior["training"] > caps["training"] or
            any(prior[BUCKETS[k]]+v > caps[BUCKETS[k]] for k,v in allowed["post_seconds"].items()) or
            prior["auxiliary"]+90 >= caps["auxiliary"]):
        raise ValueError("Fixed first attempts and serial work exceed their approved budgets")
    root.mkdir()
    ledger = Ledger(root, prior, allowed["cpu_test_seconds"], caps=caps, cpu_test_cap=caps["auxiliary"])
    limits = allowed["limits"]
    runtime_limits = dict(limits, workers=allowed["workers"], rss_bytes=limits["worker_rss_bytes"],
                          owned_rss_bytes=limits["rss_bytes"], free_disk_reserve=limits["free_reserve_bytes"])
    runner = ChineseRunner(dict(allowed, experiment_root=str(root)), gate, root, ledger, worker_entry=entry,
                           caps=caps, limits=runtime_limits, halt_dispatch_on_failure=True, continue_after_oom=True,
                           source_bindings=[record(gate), allowed["authorization"], allowed["analysis_plan"], *allowed["sources"]])
    runner.handoff_since = started
    # The loop checks this request between dispatch/settlement critical sections.
    previous_handler = signal.signal(signal.SIGTERM, lambda sig, frame: setattr(runner, "stop_signal", sig))
    try:
        training_phase(runner, rows, allowed["workers"], allowed["experiment"], validate)
        for action, status in post_status.items():
            job = dict(action=action, target=root/action, bucket=BUCKETS[action], seconds=allowed["post_seconds"][action])
            def post_complete(item):
                if read(item["target"]/"result.json")["status"] != status:
                    raise ValueError("Post-training step incomplete")
            runner.run_many([job], 1, post_complete)
        runner.settle_handoff()
        runner.check_controller_stop()
        write(root/"completion.json", dict(status="ALL_FORMAL_ACTIONS_COMPLETE", sources=allowed["sources"]))
    except BaseException as error:
        cleanup_errors = []
        try:
            runner.settle_handoff(False)
        except BaseException as secondary:
            cleanup_errors.append(repr(secondary))
        try:
            write(root/"failure.json", dict(error=repr(error), cleanup_errors=cleanup_errors, automatic_restart=False))
        except BaseException as secondary:
            print("Cannot persist campaign failure:", repr(error), cleanup_errors, repr(secondary), file=sys.stderr)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
