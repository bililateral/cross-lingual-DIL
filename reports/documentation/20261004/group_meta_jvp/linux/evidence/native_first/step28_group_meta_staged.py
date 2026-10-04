"""Exact whole-group staged gradient; no offload, approximation or data loader."""
from __future__ import annotations

import math
from typing import Any, Callable

import torch

import step28_group_meta as direct


def staged_gradient(parameters: dict, support: tuple, query: tuple, rates: dict,
                    beta: float, progress: Callable = lambda phase: None,
                    hessian_product: Callable | None = None) -> tuple:
    """Return gs + beta*(v - Hs.T@(A*v)), with one group's graph at a time.

    Each closure evaluates one complete group, is free of mutable forward state,
    and replays its own RNG. A is fixed diagonal. Returned diagnostics have no
    graph; the caller assigns gradients and performs exactly one outer update.
    """
    names, theta = tuple(parameters), tuple(parameters.values())
    if (not theta or set(names) != set(rates) or len(support) != 2 or len(query) != 2
            or any(not p.requires_grad for p in theta)
            or not math.isfinite(beta) or beta < 0
            or any(not math.isfinite(v) or v < 0 for v in rates.values())):
        raise ValueError("Invalid fixed-rate complete-group staged objective")
    result = [torch.zeros_like(p) for p in theta]
    used = [False] * len(theta)
    support_value = query_value = 0.

    def accumulate(destination: list, gradients: tuple, scale: float = 1.,
                   track_used: bool = True) -> None:
        with torch.no_grad():
            for i, gradient in enumerate(gradients):
                if gradient is not None:
                    destination[i].add_(gradient, alpha=scale)
                    if track_used:
                        used[i] = True

    for i, term in enumerate(support):
        progress(f"support_{i}_start")
        loss = term(parameters)
        support_value += float(loss.detach())
        gradients = torch.autograd.grad(loss, theta, allow_unused=True)
        accumulate(result, gradients)
        del loss, gradients
        progress(f"support_{i}_done")
    correction_norms = {"encoder": 0., "head": 0.}
    if beta:
        with torch.no_grad():
            fast = {name: (p - rates[name] * g).requires_grad_(True)
                    for name, p, g in zip(names, theta, result, strict=True)}
        vector = [torch.zeros_like(p) for p in theta]
        for i, term in enumerate(query):
            progress(f"query_{i}_start")
            loss = term(fast)
            query_value += float(loss.detach())
            gradients = torch.autograd.grad(loss, tuple(fast.values()), allow_unused=True)
            accumulate(vector, gradients)
            del loss, gradients
            progress(f"query_{i}_done")
        del fast
        with torch.no_grad():
            for name, total, v in zip(names, result, vector, strict=True):
                total.add_(v, alpha=beta)
                v.mul_(rates[name])  # Apply A BEFORE Hs; cross-block Hessian retained.

        def one_hvp(term: Callable) -> tuple:
            loss = term(parameters)
            gradients = torch.autograd.grad(loss, theta, create_graph=True, allow_unused=True)
            active = [(g, v) for g, v in zip(gradients, vector, strict=True)
                      if g is not None and g.requires_grad]
            if not active:
                return (None,) * len(theta)
            return torch.autograd.grad(tuple(g for g, _ in active), theta,
                                       grad_outputs=tuple(v for _, v in active), allow_unused=True)

        for i, term in enumerate(support):
            progress(f"hvp_{i}_start")
            correction = (one_hvp(term) if hessian_product is None
                          else hessian_product(i, parameters, tuple(vector)))
            # Diagnostics are sums of per-support squared norms, not the norm of
            # the summed correction. No extra full-model vector is retained.
            for name, g in zip(names, correction, strict=True):
                if g is not None:
                    block = "encoder" if name.startswith("model.encoder.") else "head"
                    correction_norms[block] += float(g.detach().square().sum()) * beta ** 2
            # torch.func materializes unused derivatives as zero. Only the
            # ordinary support/query gradients establish participation here.
            accumulate(result, correction, -beta, track_used=hessian_product is None)
            del correction
            progress(f"hvp_{i}_done")
        del vector
    if any(not torch.isfinite(g).all() for g in result):
        raise ValueError("Nonfinite staged gradient")
    gradients = {name: g if present else None
                 for name, g, present in zip(names, result, used, strict=True)}
    return gradients, {"support": support_value, "query": query_value,
                       "total": support_value + beta * query_value,
                       "physical_group_forwards": 6 if beta else 2,
                       "hvp_per_support_squared_norm_sum": correction_norms}


def gradient(model: Any, episode: Any, config: dict, settings: Any, seeds: tuple,
             check: Callable = lambda: None, progress: Callable = lambda phase: None,
             hessian_product: Callable | None = None) -> tuple:
    episode.validate()
    settings.validate()
    if (len(seeds) != 4 or any(type(s) is not int for s in seeds)
            or config["input"]["encoder_bf16"]
            or any(getattr(m, "gradient_checkpointing", False) for m in model.modules())):
        raise ValueError("Same explicit four seeds, FP32 and non-checkpointed path required")
    scorer = direct.GroupScorer(model, config, check if hessian_product is None else lambda: None)
    parameters = dict(scorer.named_parameters())
    original_buffers = dict(scorer.named_buffers())
    buffers = {n: b.detach().clone() for n, b in original_buffers.items()}
    rates = {n: settings.inner_encoder_lr if n.startswith("model.encoder.")
             else settings.inner_head_lr for n in parameters}
    devices = sorted({p.device.index for p in parameters.values() if p.is_cuda})
    terms = {}
    groups = episode.groups()

    def scalar_loss(params: dict, role: int) -> tuple:
        # No diagnostic logging/extraction or RNG reset in the transformed loss.
        # Existing forward/tokenization and complete loss validation are reused.
        values = torch.func.functional_call(scorer, (params, buffers), (groups[role],), strict=True)
        truth = torch.tensor(groups[role].labels, dtype=values.dtype, device=values.device)
        base = direct.parent.ranking.objectives(values, truth, .5)["total"]
        if role == 1:
            target = torch.tensor(episode.reference, dtype=values.dtype, device=values.device)
            mse = (values - target).square().mean()
            return .1 * base + .5 * mse, base, mse
        return (settings.query_history_weight * base if role == 3 else base), base, None

    def check_buffers() -> None:
        if any(not torch.equal(buffers[n], b) for n, b in original_buffers.items()):
            raise ValueError("Mutable forward buffers are outside the verified staged path")

    def role_loss(params: dict, role: int) -> torch.Tensor:
        check()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seeds[role])
            total, base, mse = scalar_loss(params, role)
        check_buffers()
        key = ("current_support", "history_support", "current_query", "history_query")[role]
        terms[key] = float(base.detach())
        if mse is not None:
            terms["logit_mse"] = float(mse.detach())
        return total

    def transformed_hvp(role: int, params: dict, vector: tuple) -> tuple:
        check()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seeds[role])
            correction = hessian_product(lambda p: scalar_loss(p, role)[0], params, vector)
        check_buffers()
        check()
        return correction

    support = tuple(lambda params, role=i: role_loss(params, role) for i in (0, 1))
    query = tuple(lambda params, role=i: role_loss(params, role) for i in (2, 3))
    gradients, diagnostics = staged_gradient(parameters, support, query, rates,
                                             settings.outer_weight, progress,
                                             transformed_hvp if hessian_product is not None else None)
    terms.update(diagnostics)
    return gradients, terms


def update(model: Any, optimizer: Any, episode: Any, config: dict, settings: Any,
           stage: int, step: int, seeds: tuple, check: Callable = lambda: None,
           progress: Callable = lambda phase: None,
           hessian_product: Callable | None = None) -> dict:
    if stage not in (2, 3) or direct.parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1:
        raise ValueError("Wrong continuous Adam history")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    gradients, terms = gradient(model, episode, config, settings, seeds, check, progress, hessian_product)
    for name, p in model.named_parameters():
        p.grad = gradients["model." + name]
    del gradients
    progress("outer_update_start")
    result = direct.apply_gradient(model, optimizer, terms, stage, step, check)
    progress("outer_update_done")
    return result
