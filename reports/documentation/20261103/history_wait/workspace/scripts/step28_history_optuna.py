"""Two history coefficients on the unchanged H_pos_tuned three-stage learner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import step28_continual_five as five

data, core, base, parent, diag = five.data, five.core, five.base, five.parent, five.diag
NAME = "H_pos_history"
ORDERS, REPLICATION_SEEDS = five.ORDERS, five.SEEDS
SEARCH_SEED = 7
TRIALS = 12
SPACE = {"history_weight": (.03, .3), "teacher_weight": (.1, 1.)}
ANCHOR = {"history_weight": .1, "teacher_weight": .5}
SAMPLER_SEED = 2026110203


@dataclass(frozen=True)
class Spec:
    phase: str
    number: int
    order: str
    seed: int
    history_weight: float
    teacher_weight: float

    def __post_init__(self):
        if (self.phase not in ("search", "replication") or type(self.number) is not int
                or self.number not in range(TRIALS) or self.order not in ORDERS
                or type(self.seed) is not int
                or self.seed not in ((SEARCH_SEED,) if self.phase == "search" else REPLICATION_SEEDS)):
            raise ValueError("Outside the fixed history search/replication identities")
        for key, (low, high) in SPACE.items():
            value = getattr(self, key)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("History coefficient outside the fixed space")
        if self.number == 0 and self.parameters != ANCHOR:
            raise ValueError("Trial zero is the original H anchor")

    @property
    def parameters(self):
        return {key: getattr(self, key) for key in SPACE}

    @property
    def name(self):
        return NAME

    @property
    def arm(self):
        return "H_pos"

    @property
    def is_record(self):
        return True

    @property
    def lambda_rank(self):
        return five.PARAMETERS["H_pos_tuned"][0]

    @property
    def lr_multiplier(self):
        return five.PARAMETERS["H_pos_tuned"][1]

    @property
    def roots(self):
        # Replications match the retained H trajectories, including memory draws.
        return dict(initialization=2026101900+self.seed, schedule=2026102900+self.seed,
                    memory=2026103900+self.seed, partition=20260918, diagnostic=20261008)

    @property
    def run_id(self):
        return f"{self.phase}_{self.number:02}_{self.order}_s{self.seed}_{NAME}"

    def identity(self):
        return {**asdict(self), "name": NAME, "roots": self.roots,
                "lambda_rank": self.lambda_rank, "lr_multiplier": self.lr_multiplier}

    @classmethod
    def from_dict(cls, value):
        result = cls(**{key: value[key] for key in cls.__dataclass_fields__})
        if value != result.identity():
            raise ValueError("Foreign coefficients, frozen H settings or random roots")
        return result

    def rates(self, step):
        return self.lr_multiplier*parent.stage_lr(step), self.lr_multiplier*.001


def specifications(phase, number, parameters):
    seeds = (SEARCH_SEED,) if phase == "search" else REPLICATION_SEEDS
    return [Spec(phase, number, order, seed, **parameters) for order in ORDERS for seed in seeds]


config = five.config


class Objective(five.tuned.Method):
    def metadata(self, arm):
        value = super().metadata(arm)
        value["training"].update(history_supervision=self.spec.history_weight,
                                 teacher_A0=self.spec.teacher_weight)
        return value

    def objective(self, table, group, seed, arm, reference=None):
        import torch
        self.check_arm(arm)
        if group.labels is None:
            raise ValueError("Authorized training labels required")
        record = five.record
        slots = (record.assignment(group), record.assignment(group, seed))
        predictions = [self.account_logits(table, assignment, arm) for assignment in slots]
        truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
        pieces = []
        for predicted in predictions:
            original = five.tuned.positive.objectives(predicted, truth)
            pieces.append(dict(bce=original["bce"], rank_used=original["rank_pos"], hard=original["hard"],
                total=original["bce"]+self.spec.lambda_rank*original["rank_pos"]+.5*original["hard"]))
        supervised = (pieces[0]["total"]+pieces[1]["total"])/2
        mse = table.new_zeros(())
        if reference is not None:
            target = torch.as_tensor(reference, dtype=table.dtype, device=table.device)
            mse = (predictions[0]-self.account_logits(target, slots[0], arm)).square().mean()
        weight = 1. if reference is None else self.spec.history_weight
        result = weight*supervised+self.spec.teacher_weight*mse
        if not torch.isfinite(result):
            raise ValueError("Nonfinite history-weight objective")
        return result, dict(supervised=float(supervised.detach()), mse0=float(mse.detach()), mse1=None,
            distillation=float(mse.detach()), supervised_contribution=float((weight*supervised).detach()),
            teacher_contribution=float((self.spec.teacher_weight*mse).detach()), weighted=float(result.detach()),
            views=[{key: float(value.detach()) for key, value in row.items()} for row in pieces])


class Method(five.Method):
    def __init__(self, spec):
        self.spec, self.tuned = spec, Objective(spec)


def select_trial(rows):
    if (len(rows) != TRIALS or [r["number"] for r in rows] != list(range(TRIALS))
            or any(r["state"] != "COMPLETE" or not math.isfinite(r["value"]) for r in rows)):
        raise ValueError("All twelve finite complete trials are required")
    # Prefer the actual original configuration on an exact tie; never round U.
    return max(rows, key=lambda r: (r["value"], r["parameters"] == ANCHOR, -r["number"]))


class Search:
    """One actual Optuna study; three orders complete before every tell."""
    def __init__(self, storage=None):
        import optuna
        if optuna.__version__ != "4.6.0":
            raise ValueError("Use the existing pinned Optuna 4.6.0")
        self.study = optuna.create_study(study_name="H_pos_history", storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED, n_startup_trials=4, n_ei_candidates=24),
            pruner=optuna.pruners.NopPruner(), load_if_exists=False)
        self.study.enqueue_trial(ANCHOR)
        self.rows, self.pending, self.failed = [], None, False

    def ask(self):
        if self.failed or self.pending is not None or len(self.rows) >= TRIALS:
            raise ValueError("No further candidate is available")
        trial = self.study.ask()
        if trial.number != len(self.rows):
            raise ValueError("Fresh sequential trial numbering required")
        parameters = {key: trial.suggest_float(key, *bounds, log=True) for key, bounds in SPACE.items()}
        specs = specifications("search", trial.number, parameters)
        self.pending = (trial, specs)
        return specs

    def tell(self, feedback):
        if self.pending is None or self.failed:
            raise ValueError("No complete pending candidate")
        trial, specs = self.pending
        values = feedback.get("final_map_by_order", {})
        if (feedback.get("status") != "THREE_ORDERS_VERIFIED_DEV_COMPLETE"
                or feedback.get("specs") != [s.identity() for s in specs] or set(values) != set(ORDERS)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values())
                or not feedback.get("result_identity") or feedback.get("all_workers_released") is not True):
            raise ValueError("Optuna cannot consume partial or unbound feedback")
        value = sum(values[order] for order in ORDERS)/3
        trial.set_user_attr("feedback", feedback)
        self.study.tell(trial, value)
        self.rows.append(dict(number=trial.number, state="COMPLETE", value=value,
                              parameters=specs[0].parameters, feedback=feedback))
        self.pending = None

    def fail(self, reason):
        import optuna
        self.failed = True
        if self.pending is not None:
            trial, _ = self.pending
            trial.set_user_attr("failure", reason)
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            self.pending = None

    def selected(self):
        if self.failed or self.pending is not None:
            raise ValueError("Failed/pending study cannot select")
        return select_trial(self.rows)
