"""Finite Optuna decisions and disjoint execution accounting; no launch on import."""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

from step28_optuna import ARMS, TrialSpec

S0 = {"A": .49413171588434307, "B": .5058704368257183}
SHARED = {"control": 900., "evaluation": 3600., "analysis": 600.}


def objective(values: dict, reference: dict) -> float:
    if set(values) != {"A", "B"} or set(reference) != {"A", "B"}:
        raise ValueError("Both domains required")
    if any(not math.isfinite(float(x)) for x in (*values.values(), *reference.values())):
        raise ValueError("Nonfinite objective stops all dispatch")
    return min(values[d]-reference[d] for d in "AB")


def select_trial(rows: list[dict], reference: dict) -> dict:
    if len(rows) != 12 or sorted(r["number"] for r in rows) != list(range(12)):
        raise ValueError("Twelve actual slots required")
    keys = []
    for row in rows:
        if row["state"] != "COMPLETE":
            raise ValueError("Incomplete/failed study cannot be selected")
        u = objective(row["domains"], reference)
        if row["value"] != u:
            raise ValueError("Stored objective differs")
        keys.append((u, sum(row["domains"][d]-reference[d] for d in "AB")/2, -row["number"]))
    return rows[max(range(12), key=keys.__getitem__)]


def projected_gate(projected: dict) -> None:
    if set(projected["arms"]) != set(ARMS):
        raise ValueError("Each arm needs its own measured projection")
    search = replication = 0.
    for p in projected["arms"].values():
        for key in ("search", "replication"):
            if not math.isfinite(p[key]) or not 0 < p[key] <= 21600:
                raise ValueError("Old PASS cannot satisfy the new single-run budget")
        search += 12*p["search"]
        replication += 3*p["replication"]
    shared = projected["shared"]
    if set(shared) != set(SHARED) or any(not math.isfinite(shared[k]) or not 0 < shared[k] <= SHARED[k] for k in SHARED):
        raise ValueError("Shared evaluation/analysis reserve not demonstrated")
    if search > 777600 or replication > 194400 or 300+1800+search+replication+sum(shared.values()) > 979200:
        raise ValueError("Stage/total projection exceeds proposal")


class Ledger:
    """One live monotonic interval; waiting outside intervals is calendar time."""

    def __init__(self, clock=time.monotonic, prior_cpu=0., prior_native=0.):
        if not 0 <= prior_cpu <= 300 or not 0 <= prior_native <= 1800:
            raise ValueError("Qualification time outside allotted budgets")
        self.clock = clock
        self.qualification = prior_cpu+prior_native
        self.used = {**{k: 0. for k in SHARED}, "search": 0., "replication": 0.}
        self.arm_used = {p: {a: 0. for a in ARMS} for p in ("search", "replication")}
        self.intervals = []
        self.active = None
        self.failed = False

    def remaining(self, kind: str, arm: str | None = None) -> float:
        total = 979200-self.qualification-sum(self.used.values())
        shared = 5100-sum(self.used[k] for k in SHARED)
        if kind in SHARED:
            return min(SHARED[kind]-self.used[kind], shared, total)
        per_arm, stage = (259200., 777600.) if kind == "search" else (64800., 194400.)
        return min(21600., per_arm-self.arm_used[kind][arm], stage-self.used[kind], total)

    @contextmanager
    def interval(self, kind: str, *, arm=None, run_id=None):
        if self.failed or self.active is not None or kind not in self.used:
            raise ValueError("Failed or overlapping accounting interval")
        if kind in ("search", "replication") and (arm not in ARMS or not run_id):
            raise ValueError("Own run identity required")
        allowed = self.remaining(kind, arm)
        if allowed <= 0:
            self.failed = True
            raise TimeoutError("No allotted budget remains")
        started = self.clock()
        self.active = (kind, started, allowed)
        error = None
        try:
            yield allowed
            self.check()
        except BaseException as exc:
            error = type(exc).__name__
            self.failed = True
            raise
        finally:
            elapsed = self.clock()-started
            if elapsed < 0:
                raise ValueError("Clock must be monotonic")
            self.used[kind] += elapsed
            if arm is not None:
                self.arm_used[kind][arm] += elapsed
            self.intervals.append(dict(kind=kind, run_id=run_id, arm=arm, seconds=elapsed, error=error))
            self.active = None

    def check(self) -> None:
        if self.active is None:
            raise ValueError("No executing interval")
        _, started, allowed = self.active
        if self.clock()-started >= allowed:
            raise TimeoutError("Allotted continuous interval exhausted")

    def snapshot(self) -> dict:
        return dict(qualification_seconds=self.qualification, used=self.used.copy(),
                    by_arm={k: v.copy() for k, v in self.arm_used.items()},
                    intervals=list(self.intervals), failed=self.failed)


class Search:
    """Actual independent samplers; caller supplies only completed production results."""

    def __init__(self, storage: str | None, reference: dict):
        import optuna
        if optuna.__version__ != "4.6.0":
            raise ValueError("Pinned Optuna 4.6.0 required")
        objective(reference, reference)
        self.reference = dict(reference)
        self.studies, self.rows = {}, {a: [] for a in ARMS}
        self.pending = None
        self.failed = False
        for arm in ARMS:
            sampler = optuna.samplers.TPESampler(seed=20261018, n_startup_trials=4,
                n_ei_candidates=24, multivariate=False, group=False, constant_liar=False)
            study = optuna.create_study(study_name=arm, storage=storage, direction="maximize",
                sampler=sampler, pruner=optuna.pruners.NopPruner(), load_if_exists=False)
            study.enqueue_trial(dict(lambda_rank=1., lr_multiplier=1.), skip_if_exists=False)
            self.studies[arm] = study

    @staticmethod
    def order(number: int) -> tuple:
        shift = number % 3
        return ARMS[shift:]+ARMS[:shift]

    def ask(self, arm: str, number: int):
        if self.failed or self.pending is not None or arm not in ARMS or len(self.rows[arm]) != number or number not in range(12):
            raise ValueError("Invalid slot or stopped campaign")
        if number and any(len(self.rows[a]) < number for a in ARMS):
            raise ValueError("All preceding arm slots, including anchors, must finish")
        trial = self.studies[arm].ask()
        if trial.number != number:
            raise ValueError("Study history/slot mismatch")
        weight = trial.suggest_float("lambda_rank", .25, 2., log=True)
        rate = trial.suggest_float("lr_multiplier", .5, 2., log=True)
        spec = TrialSpec(arm, "search", number, weight, rate)
        self.pending = (spec, trial)
        return spec

    def fail(self, reason: str) -> None:
        import optuna
        self.failed = True
        if self.pending is not None:
            spec, trial = self.pending
            trial.set_user_attr("failure_reason", reason)
            self.studies[spec.arm].tell(trial, state=optuna.trial.TrialState.FAIL)
            self.rows[spec.arm].append(dict(number=spec.number, state="FAIL", spec=spec.identity(), reason=reason))
            self.pending = None

    def tell(self, result: dict) -> None:
        if self.failed or self.pending is None:
            raise ValueError("No running trial")
        spec, trial = self.pending
        try:
            if (result.get("status") != "SEARCH_COMPLETE" or result.get("trial") != spec.identity()
                    or result.get("child_exit_code") != 0 or result.get("resources_released") is not True
                    or result.get("access") != dict(train=1, valid=1, heldout=0, owners=0)
                    or result.get("anchor_compatible") is not True and spec.number == 0):
                raise ValueError("Completion/exit/anchor evidence incomplete")
            value = objective(result["domains"], self.reference)
            for key in ("domains", "trial", "completion_identity"):
                trial.set_user_attr(key, result[key])
            self.studies[spec.arm].tell(trial, value)
            self.rows[spec.arm].append(dict(number=spec.number, state="COMPLETE", value=value,
                                           domains=result["domains"], spec=spec.identity()))
            self.pending = None
        except BaseException as exc:
            self.fail(str(exc))
            raise

    def selected(self) -> dict:
        if self.failed or self.pending is not None:
            raise ValueError("Campaign failed or pending")
        return {arm: select_trial(self.rows[arm], self.reference) for arm in ARMS}


def replication_specs(selected: dict) -> list[TrialSpec]:
    if set(selected) != set(ARMS):
        raise ValueError("Freeze all three configurations first")
    return [TrialSpec(arm, "replication", seed-1, selected[arm]["spec"]["lambda_rank"],
                      selected[arm]["spec"]["lr_multiplier"], seed)
            for seed in range(1, 4) for arm in Search.order(seed-1)]


def joint_opening(records: list[dict], specs: list[TrialSpec]) -> None:
    if len(records) != 9 or len(specs) != 9 or {(s.arm, s.seed) for s in specs} != {(a, s) for a in ARMS for s in range(1, 4)}:
        raise ValueError("All nine prescribed trajectories required")
    for result, spec in zip(records, specs, strict=True):
        if (result.get("status") != "REPLICATION_BLIND_COMPLETE" or result.get("trial") != spec.identity()
                or result.get("access") != dict(train=1, valid=0, heldout=0, owners=0)
                or result.get("child_exit_code") != 0 or result.get("resources_released") is not True):
            raise ValueError("Joint valid opening requires every own blind completion and exit")
