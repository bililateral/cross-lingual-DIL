"""Original S and historical H; reuse their actual objectives and five-run engine."""
from __future__ import annotations

import numpy as np

import step28_continual_five as five
import step28_g_half as historical

data, core, base, parent, diag = five.data, five.core, five.base, five.parent, five.diag
METHODS = ("S_default", "H_default")


class Spec(five.Spec):
    def __post_init__(self) -> None:
        if self.name not in METHODS or self.order not in five.ORDERS or self.seed not in five.SEEDS:
            raise ValueError("Outside the two original methods, three orders and s4-s6")

    @property
    def arm(self) -> str:
        return "S" if self.name == "S_default" else "G_half"

    @property
    def is_record(self) -> bool:
        return True

    @property
    def lambda_rank(self) -> float:
        return 1.

    @property
    def lr_multiplier(self) -> float:
        return 1.


def specifications() -> list[Spec]:
    return [Spec(name, order, seed) for order in five.ORDERS for seed in five.SEEDS for name in METHODS]


config = five.config


class OriginalObjective:
    """Small adapter: the original loss/scoring implementations are not rewritten."""
    def __init__(self, spec: Spec):
        self.spec = spec

    def metadata(self, arm: str) -> dict:
        if arm != self.spec.arm:
            raise ValueError("Foreign original method")
        pooling = (five.tuned.positive.identity("S_pos") if arm == "S" else historical.identity())
        return dict(pooling=pooling, original_default=self.spec.identity(), training=dict(
            arm=arm, rank="original_rank", lambda_rank=1., history_supervision=.1,
            teacher_A0=.5, hard_weight=.5))

    def bind(self, model, metadata: dict) -> None:
        expected = self.metadata(self.spec.arm)
        if metadata != expected:
            raise ValueError("Restored original objective differs")
        model.original_default_identity = expected

    def objective(self, table, group, seed, arm, reference=None):
        self.metadata(arm)
        function = five.record.objective if arm == "S" else historical.objective
        return function(table, group, seed, arm, reference)

    def score(self, model, groups, c, arm, check=lambda: None):
        self.metadata(arm)
        self.bind(model, getattr(model, "original_default_identity", {}))
        if not groups:
            return np.empty((0, 378), np.float32)
        return (five.record.score(model, groups, c, check) if arm == "S"
                else historical.score(model, groups, c, arm, check))


class Method(five.Method):
    def __init__(self, spec: Spec):
        self.spec = spec
        # Inherited update, score, Memory and restore dispatch through this adapter.
        # The historical field name does not run an Optuna study.
        self.tuned = OriginalObjective(spec)
