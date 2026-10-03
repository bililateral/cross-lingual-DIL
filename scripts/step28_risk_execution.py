"""Small risk-specific adapters for the existing paired continuation runner."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

import step28_risk_replay as risk


class Memory(risk.RiskMemory):
    """Expose the existing runner's metadata interface, preserving charged risks."""

    def __getattr__(self, name):
        return getattr(self.cache, name)

    @property
    def auxiliary(self):
        return self.cache.auxiliary

    @auxiliary.setter
    def auxiliary(self, value):
        self.cache.auxiliary = {**value, "risk_replay": self.cache.auxiliary["risk_replay"]}

    def summary(self):
        self.to_bytes()
        result = self.cache.summary()
        refs = self.cache.auxiliary["risk_replay"]["groups"]
        result["risk_references"] = {uid: hashlib.sha256(risk.data.json_bytes(v)).hexdigest()
                                     for uid, v in refs.items()}
        result["risk_origins"] = {uid: v["stage"] for uid, v in refs.items()}
        return result

    @classmethod
    def from_er(cls, payload, model, optimizer, c, check):
        cache = risk.parent.Memory.from_bytes(payload)
        if (cache.with_logits or cache.reservoir.seen != 48 or cache.draw_stage != 0
                or risk.parent.adam_step(optimizer) != 288):
            raise ValueError("Risk first memory requires the shared stage-one ER cache")
        result = cls(cache.order, cache.seed, cache.first_map)
        result.cache = cache
        rows = risk.parent.ranking.score(model, cache.reservoir.groups, c, check)
        result.cache.auxiliary["risk_replay"] = {"schema": 1, "groups": {
            g.uid: {"stage": 1, "values": risk.make_reference(row, g.labels)}
            for g, row in zip(cache.reservoir.groups, rows, strict=True)}}
        result.to_bytes()
        return result


def mode_probe(model, memory, c, check):
    """One live cached group, four fixed train draws, no update or new labels."""
    import torch
    from step28_bge_continual_run import rng_state, restore_rng

    state, mode = rng_state(), model.training
    group = memory.cache.reservoir.groups[0]
    origin = memory.cache.auxiliary["risk_replay"]["groups"][group.uid]["values"]
    try:
        row = risk.parent.ranking.score(model, [group], c, check)[0]
        reference = risk.make_reference(row, group.labels)
        scores = torch.tensor(row)
        risks, counts = risk.query_risks(scores, torch.tensor(group.labels))
        zero = risk.distribution_penalty(risks, counts, reference).tolist()
        if any(zero):
            raise ValueError("Unchanged eval model differs from its own reference")
        measurements = []
        with torch.no_grad():
            model.train()
            for seed in (610301, 610302, 610303, 610304):
                torch.manual_seed(seed)
                logits = risk.base.logits(model, group, c, "split_rank", check)
                live, degrees = risk.query_risks(logits, torch.tensor(group.labels, device=logits.device))
                measurements.append(risk.distribution_penalty(live, degrees, reference).cpu().tolist())
        return {"group_uid": group.uid, "eval_self": zero, "train_against_same_model_eval": measurements,
                "eval_against_origin": risk.distribution_penalty(risks, counts, origin).tolist(),
                "origin_channel_std_by_stratum": {k: np.asarray(v).std(0).tolist() for k, v in origin.items()},
                "scope": "One retained group, fixed four seeds; noise diagnostic, no adaptive tuning"}
    finally:
        restore_rng(state)
        model.train(mode)


def verify_references(root, retained, start, log, stage):
    """Check saved full risk memory, origin ages and unchanged surviving targets."""
    memory = Memory.from_bytes((root / retained["file"]["path"]).read_bytes())
    actual = memory.summary()
    if any(actual[k] != retained[k] for k in actual):
        raise ValueError("Risk memory record differs from serialized memory")
    first = start["memory_summary"]
    for uid, digest in first["risk_references"].items():
        if uid in actual["risk_references"] and actual["risk_references"][uid] != digest:
            raise ValueError("Old risk reference refreshed")
    expected = first if stage == 2 else retained
    if any(log["memory_after_training"][key] != expected[key]
           for key in ("risk_references", "risk_origins")):
        raise ValueError("Training consumed different risk references")
