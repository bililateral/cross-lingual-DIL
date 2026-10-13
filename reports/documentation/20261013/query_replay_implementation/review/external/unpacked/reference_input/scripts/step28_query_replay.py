"""Natural-query replay and exact finite-population response retention."""
from __future__ import annotations

import hashlib
import random
from typing import Any, Callable

import numpy as np

import step28_record_replay as record

data = record.data
ARMS = ("S", "Q", "C_q", "C_b")
RNG_ORDER = "ABC"


def structure(group: data.Group) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    group.validate()
    if group.labels is None:
        raise ValueError("Authorized training relations required")
    owner = record.assignment(group)
    truth = np.eye(28, dtype=bool)
    i, j = np.triu_indices(28, 1)
    truth[i, j] = group.labels
    truth[j, i] = group.labels
    components, remaining = [], set(range(28))
    while remaining:
        members = np.flatnonzero(truth[min(remaining)]).tolist()
        if not set(members) <= remaining or any(not np.array_equal(truth[a], truth[members[0]]) for a in members):
            raise ValueError("Training relations are not disjoint positive cliques")
        remaining.difference_update(members)
        components.append(members)
    if sorted(map(len, components)) != [2]*8+[3]*4:
        raise ValueError("Expected eight K2 and four K3")
    np.fill_diagonal(truth, False)
    return owner, components, truth


def candidates(group: data.Group, stage: int, step: int, rule: str) -> list[list[list[int]]]:
    if rule not in ("query_replay", "within_account_bootstrap") or stage != 2 or not 1 <= step <= 288:
        raise ValueError("Fixed B historical augmentation scope required")
    owner, components, _ = structure(group)
    natural = [np.flatnonzero(owner == a).tolist() for a in range(28)]
    result = []
    for q in range(28):
        row = [[] for _ in range(28)]
        if rule == "query_replay":
            for index, component in enumerate(components):
                accounts = [a for a in component if a != q]
                pool = [r for a in accounts for r in natural[a]]
                rng = random.Random(data.seed_for(20260918, rule, RNG_ORDER, stage, step, group.uid, q, index))
                if len(accounts) > 1:
                    rng.shuffle(pool)
                start = 0
                for j in accounts:
                    row[j] = pool[start:start+len(natural[j])]
                    start += len(natural[j])
            if sorted(r for slot in row for r in slot) != [r for r in range(len(owner)) if owner[r] != q]:
                raise ValueError("Query exclusion or complete-record conservation failed")
        else:
            for j in range(28):
                if j != q:
                    rng = random.Random(data.seed_for(20260918, rule, RNG_ORDER, stage, step, group.uid, q, j))
                    row[j] = [rng.choice(natural[j]) for _ in natural[j]]
        if row[q] or any(len(row[j]) != len(natural[j]) or any(owner[r] == q for r in row[j]) for j in range(28) if j != q):
            raise ValueError("Wrong row size or query-origin leakage")
        result.append(row)
    return result


def marginal(table: Any, group: data.Group) -> tuple[Any, Any, Any]:
    """Return natural-query means, account-centered means and exact b weights."""
    import torch
    owner, components, _ = structure(group)
    n = len(owner)
    if table.shape != (n*(n-1)//2,) or not torch.isfinite(table).all():
        raise ValueError("Complete finite symmetric record table required")
    matrix = table.new_zeros((n, n))
    i, j = torch.triu_indices(n, n, 1, device=table.device)
    matrix[i, j], matrix[j, i] = table, table
    owned = torch.as_tensor(owner, device=table.device)
    pooling = torch.nn.functional.one_hot(owned, 28).T.to(table.dtype)
    pooling = pooling / pooling.sum(1, keepdim=True)
    u = pooling @ matrix
    centered = u - (u @ pooling.T)[:, owned]
    coefficients = np.zeros((28, n), dtype=np.float64)
    counts = np.bincount(owner, minlength=28)
    for q in range(28):
        for component in components:
            accounts = [a for a in component if a != q]
            total = int(sum(counts[a] for a in accounts))
            b = sum((total-int(counts[j]))/(int(counts[j])*total*(total-1)) for j in accounts)
            coefficients[q, np.isin(owner, accounts)] = b
    return u, centered, torch.as_tensor(coefficients, dtype=table.dtype, device=table.device)


def response_loss(table: Any, group: data.Group, reference: Any) -> Any:
    import torch
    target = torch.as_tensor(reference, dtype=table.dtype, device=table.device).detach()
    _, live, weights = marginal(table, group)
    _, old, _ = marginal(target, group)
    return (weights*(live-old).square()).sum()/756


def directed_scores(table: Any, group: data.Group, rows: list[list[list[int]]]) -> Any:
    import torch
    u, _, _ = marginal(table, group)
    n = u.shape[1]
    weights = np.zeros((28, 28, n), dtype=np.float64)
    for q, row in enumerate(rows):
        for j, indices in enumerate(row):
            if j != q:
                weights[q, j] = np.bincount(indices, minlength=n)/len(indices)
    return (torch.as_tensor(weights, dtype=table.dtype, device=table.device)*u[:, None, :]).sum(2)


def row_objective(scores: Any, group: data.Group) -> Any:
    import torch
    _, _, labels = structure(group)
    if scores.shape != (28, 28) or not torch.isfinite(scores).all():
        raise ValueError("Complete finite directed rows required")
    positive = torch.as_tensor(labels, device=scores.device)
    diagonal = torch.eye(28, dtype=torch.bool, device=scores.device)
    negative = ~positive & ~diagonal
    counts = positive.sum(1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(scores[~diagonal], positive[~diagonal].to(scores.dtype))
    rank = (torch.logsumexp(scores.masked_fill(diagonal, -torch.inf), 1)-(scores*positive).sum(1)/counts).mean()
    chosen = torch.argsort(scores.detach().masked_fill(~negative, -torch.inf), dim=1, descending=True, stable=True)[:, :5]
    hard = torch.nn.functional.softplus(scores.gather(1, chosen)[:, None, :]-scores[:, :, None])
    hard = ((hard*positive[:, :, None]).sum((1, 2))/(counts*5)).mean()
    return bce+rank+.5*hard


def objective(table: Any, group: data.Group, regroup_seed: int, arm: str,
              reference: Any = None, *, stage: int = 2, step: int = 1) -> tuple[Any, dict]:
    import torch
    if arm not in ARMS:
        raise ValueError("Unknown query-replay arm")
    if arm == "S" or reference is None:
        return record.objective(table, group, regroup_seed, "S", reference)
    natural = record.account_logits(table, record.assignment(group))
    truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
    a0 = record.parent.ranking.objectives(natural, truth, .5)["total"]
    rule = "within_account_bootstrap" if arm == "C_b" else "query_replay"
    rows = candidates(group, stage, step, rule)
    augmented = row_objective(directed_scores(table, group, rows), group)
    target = torch.as_tensor(reference, dtype=table.dtype, device=table.device).detach()
    d0 = (natural-record.account_logits(target, record.assignment(group))).square().mean()
    response = response_loss(table, group, target)
    supervised = (a0+augmented)/2
    weight = 0. if arm == "Q" else .25
    loss = .1*supervised+.5*d0+weight*response
    return loss, {"supervised": float(supervised.detach()), "natural_supervised": float(a0.detach()),
        "augmented_supervised": float(augmented.detach()), "mse0": float(d0.detach()), "mse1": None,
        "distillation": float(d0.detach()), "response": float(response.detach()), "response_weight": weight,
        "weighted_response": float((weight*response).detach()), "weighted": float(loss.detach()),
        "augmentation_rule": rule, "assignment_sha256": hashlib.sha256(data.json_bytes(rows)).hexdigest()}


def update(model: Any, optimizer: Any, current: data.Group, history: data.Group | None,
           reference: Any, c: dict, arm: str, stage: int, step: int, check: Callable = lambda: None) -> dict:
    if arm not in ARMS or stage not in (1, 2) or (stage == 1 and (arm != "S" or history is not None)):
        raise ValueError("Shared A and fixed B branches required")
    def loss(table, group, regroup_seed, name, target):
        return objective(table, group, regroup_seed, name, target, stage=stage, step=step)
    return record.update(model, optimizer, current, history, reference, c, arm, RNG_ORDER, stage, step, check,
                         objective_fn=None if arm == "S" else loss)


def diagnostic(table: np.ndarray, group: data.Group, reference: np.ndarray) -> dict:
    import torch
    with torch.no_grad():
        value = float(response_loss(torch.from_numpy(table), group, reference))
    return {"R_q": value, "weighted_R_q": .25*value}
