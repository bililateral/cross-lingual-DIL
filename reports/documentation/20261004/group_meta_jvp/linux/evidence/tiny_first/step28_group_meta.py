"""Whole-group look-ahead replay; handwritten implementation, no formal loader.

The inner step is differentiable SGD, not an Adam simulation. The outer step is
one ordinary AdamW update. This is a mechanism candidate, not a novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch

import step28_bge_continual as parent

base, data, core = parent.base, parent.data, parent.core


def load_model(config: dict, device: str = "cpu") -> Any:
    """The same registered BGE/head, with eager attention selected at load time."""
    from sentence_transformers import SentenceTransformer
    if config["input"]["encoder_bf16"] or config["input"]["activation_checkpointing"]:
        raise ValueError("Explicit float32, non-checkpointed meta configuration required")
    spec = config["models"]["split_rank"]
    torch.manual_seed(config["initialization_seed"])
    encoder = SentenceTransformer(str(data.ROOT / spec["path"]), device=device,
                                  local_files_only=True, trust_remote_code=False,
                                  model_kwargs={"attn_implementation": "eager"})
    encoder.default_prompt_name = None
    encoder.max_seq_length = config["input"]["token_budget"]
    pools = [m for m in encoder if type(m).__name__ == "Pooling"]
    if (len(pools) != 1 or base.pooling_modes(pools[0]) != [spec["pooling"]]
            or any(type(m).__name__ == "Dense" for m in encoder) != spec["dense_module"]
            or encoder.get_sentence_embedding_dimension() != spec["embedding_dim"]):
        raise ValueError("Registered native BGE representation differs")
    encoder[0].auto_model.gradient_checkpointing_disable()
    torch.manual_seed(config["initialization_seed"])
    return core.build_model(encoder, config["interventions"]["split_rank"]["account_dim"],
                            config["input"]["head_hidden"]).to(device)


@dataclass(frozen=True)
class Settings:
    inner_encoder_lr: float
    inner_head_lr: float
    query_history_weight: float
    outer_weight: float

    def validate(self) -> None:
        values = (self.inner_encoder_lr, self.inner_head_lr,
                  self.query_history_weight, self.outer_weight)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Finite nonnegative explicit meta settings required")
        if self.inner_head_lr == 0 or self.query_history_weight == 0:
            raise ValueError("Head look-ahead and historical query must be active")


@dataclass(frozen=True)
class Episode:
    current_support: Any
    history_support: Any
    current_query: Any
    history_query: Any
    reference: np.ndarray
    # Explicit binding prevents accepting another historical group's target.
    reference_uid: str

    def validate(self) -> None:
        groups = self.groups()
        seen_groups, seen_sellers, seen_items = set(), set(), set()
        for group in groups:
            group.validate()
            sellers = set(group.sellers)
            items = {row[0] for rows in group.items for row in rows}
            if (group.labels is None or group.uid in seen_groups
                    or sellers & seen_sellers or items & seen_items):
                raise ValueError("Four complete, supervised, identity-disjoint groups required")
            seen_groups.add(group.uid)
            seen_sellers.update(sellers)
            seen_items.update(items)
        if (self.reference_uid != self.history_support.uid
                or not isinstance(self.reference, np.ndarray)
                or self.reference.shape != (378,) or self.reference.dtype != np.float32
                or not np.isfinite(self.reference).all()):
            raise ValueError("Historical support requires its own fixed float32 logits")

    def groups(self) -> tuple:
        return (self.current_support, self.history_support,
                self.current_query, self.history_query)


def lookahead(parameters: dict, support: Callable, query: Callable,
              rates: dict, beta: float) -> tuple:
    """J = Ls(theta) + beta Lq(theta - A grad Ls), retaining the Hessian path.

    Closures must evaluate the supplied parameter dictionary. Neither a copied
    module nor an optimizer step is a differentiable replacement for this path.
    """
    if (set(parameters) != set(rates) or not math.isfinite(beta) or beta < 0
            or any(not math.isfinite(v) or v < 0 for v in rates.values())):
        raise ValueError("Invalid look-ahead coefficients")
    loss_s = support(parameters)
    if beta == 0:
        return loss_s, loss_s, loss_s.new_zeros(()), parameters
    grads = torch.autograd.grad(loss_s, tuple(parameters.values()),
                                create_graph=True, allow_unused=True)
    fast = {name: value if grad is None else value - rates[name] * grad
            for (name, value), grad in zip(parameters.items(), grads, strict=True)}
    loss_q = query(fast)
    return loss_s + beta * loss_q, loss_s, loss_q, fast


class GroupScorer(torch.nn.Module):
    def __init__(self, model: Any, config: dict, check: Callable) -> None:
        super().__init__()
        self.model, self.config, self.check = model, config, check

    def forward(self, group: Any) -> Any:
        return base.logits(self.model, group, self.config, "split_rank", self.check)


def objective(model: Any, episode: Episode, config: dict, settings: Settings,
              seeds: tuple[int, int, int, int], check: Callable = lambda: None) -> tuple:
    """Build the actual four-group objective, without changing model/Adam state.

    Checkpoint recomputation outside functional_call may restore original rather
    than fast weights. This implementation rejects it instead of silently using
    the wrong weights. Full-float eager attention is the verification route.
    """
    episode.validate()
    settings.validate()
    if len(seeds) != 4 or any(type(seed) is not int for seed in seeds):
        raise ValueError("Four explicit dropout streams required")
    if (config["input"]["encoder_bf16"]
            or any(getattr(m, "gradient_checkpointing", False) for m in model.modules())):
        raise ValueError("Meta route requires float32 and checkpointing disabled")
    scorer = GroupScorer(model, config, check)
    parameters = dict(scorer.named_parameters())
    if any(not p.requires_grad for p in parameters.values()):
        raise ValueError("Head-only/frozen-encoder meta training is outside this implementation")
    buffers = {name: value.detach().clone() for name, value in scorer.named_buffers()}
    rates = {name: settings.inner_encoder_lr if name.startswith("model.encoder.")
             else settings.inner_head_lr for name in parameters}
    groups = episode.groups()
    devices = sorted({p.device.index for p in parameters.values() if p.is_cuda})
    terms = {}

    def forward(params: dict, role: int) -> tuple:
        check()
        # Stateless per-role streams also make recomputation/finite differences
        # reproducible without perturbing the caller's global RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seeds[role])
            values = torch.func.functional_call(scorer, (params, buffers), (groups[role],), strict=True)
        truth = torch.tensor(groups[role].labels, dtype=values.dtype, device=values.device)
        loss = parent.ranking.objectives(values, truth, .5)["total"]
        terms[("current_support", "history_support", "current_query", "history_query")[role]] = loss
        return values, loss

    def support(params: dict) -> Any:
        _, current = forward(params, 0)
        values, history = forward(params, 1)
        target = torch.tensor(episode.reference, dtype=values.dtype, device=values.device)
        mse = (values - target).square().mean()
        terms["logit_mse"] = mse
        return current + .1 * history + .5 * mse

    def query(params: dict) -> Any:
        _, current = forward(params, 2)
        _, history = forward(params, 3)
        return current + settings.query_history_weight * history

    total, support_loss, query_loss, fast = lookahead(
        parameters, support, query, rates, settings.outer_weight)
    terms.update(total=total, support=support_loss, query=query_loss)
    return total, terms, fast


def update(model: Any, optimizer: Any, episode: Episode, config: dict,
           settings: Settings, stage: int, step: int, seeds: tuple,
           check: Callable = lambda: None) -> dict:
    if stage not in (2, 3) or parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1:
        raise ValueError("Meta update requires the declared continuous Adam history")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, terms, _ = objective(model, episode, config, settings, seeds, check)
    loss.backward()
    return apply_gradient(model, optimizer, terms, stage, step, check)


def apply_gradient(model: Any, optimizer: Any, terms: dict, stage: int, step: int,
                   check: Callable = lambda: None) -> dict:
    """Shared single outer update, after either exact gradient construction."""
    if stage not in (2, 3) or parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1:
        raise ValueError("Wrong outer Adam history")
    encoder_lr = parent.stage_lr(step)
    evidence = {}
    for name, module in (("encoder", model.encoder), ("head", model.head)):
        squared = sum(float(p.grad.detach().float().square().sum())
                      for p in module.parameters() if p.grad is not None)
        if not math.isfinite(squared) or squared <= 0:
            raise ValueError("Missing finite encoder/head gradient")
        evidence[name] = math.sqrt(squared)
    if len(optimizer.param_groups) != 2:
        raise ValueError("Expected encoder and head optimizer groups")
    for group, rate in zip(optimizer.param_groups, (encoder_lr, .001), strict=True):
        group["lr"] = rate
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    if parent.adam_step(optimizer) != (stage - 1) * 288 + step:
        raise ValueError("Expected exactly one actual AdamW step")
    return {**{name: float(value.detach()) if isinstance(value, torch.Tensor) else value
               for name, value in terms.items()},
            "stage": stage, "step": step, "adam_step": parent.adam_step(optimizer),
            "gradient_norm": float(norm), "module_gradient_norms": evidence,
            "encoder_lr": encoder_lr, "head_lr": .001}


def train_stage(model: Any, optimizer: Any, current: list, memory: Any,
                config: dict, order: str, stage: int, settings: Settings,
                check: Callable = lambda: None) -> dict:
    """Complete stage API; four group presentations per update, no data loader.

    Current queries are a cyclic shift of the same epoch permutation (each group
    has six support and six query appearances). Historical query draws use an
    independent stateless stream and exclude the historical support group.
    """
    import random
    policy = parent.contract()
    settings.validate()
    if (not memory.with_logits or memory.order != order
            or memory.seed != policy["memory_seed"]):
        raise ValueError("Expected paired LOGIT memory with stage-end references")
    sequence, stream = parent.schedule(current, policy, order, stage)
    if {g.uid for g in current} & {g.uid for g in memory.reservoir.groups}:
        raise ValueError("Current and historical fitting groups overlap")
    if parent.adam_step(optimizer) != (stage - 1) * 288:
        raise ValueError("Wrong stage start")
    memory.begin_stage(stage)
    records, episodes = [], []
    for index, current_support in enumerate(sequence):
        history_support, reference = memory.draw()
        choices = sorted((g for g in memory.reservoir.groups if g.uid != history_support.uid),
                         key=lambda g: g.uid)
        rng = random.Random(data.seed_for(policy["memory_seed"], order, stage, index, "meta_query"))
        history_query = choices[rng.randrange(len(choices))]
        current_query = sequence[(index // 48) * 48 + (index % 48 + 1) % 48]
        episode = Episode(current_support, history_support, current_query, history_query,
                          reference, history_support.uid)
        seeds = (data.seed_for(stream, index, "dropout"),
                 data.seed_for(policy["memory_seed"], order, stage, index, "history_dropout"),
                 data.seed_for(stream, index, "meta_current_dropout"),
                 data.seed_for(stream, index, "meta_history_dropout"))
        records.append(update(model, optimizer, episode, config, settings,
                              stage, index + 1, seeds, check))
        episodes.append([g.uid for g in episode.groups()])
    memory.auxiliary["group_meta"] = {"settings": settings.__dict__, "completed_stage": stage,
                                      "query_schedule": "cyclic_current_stateless_history"}
    memory.to_bytes()  # All persistent method-specific state counts toward 1 MiB.
    return {"updates": records, "episodes": episodes, "physical_updates": len(records),
            "gradient_group_presentations": (2 if settings.outer_weight == 0 else 4) * len(records)}
