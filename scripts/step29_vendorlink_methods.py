"""Four actual methods for the approved observed-account proxy implementation.

VendorLink's missing token reduction is an explicitly approved reproduction
assumption: masked token mean of the sum of its last four layers. Not exact
reproduction of an undocumented author operation. No model/data IO on import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from step29_vendorlink_data import Group, assignment, seed_for

METHODS = ("VendorLink_SEQ", "H_SEQ", "H_ER", "H_full")
UPSTREAM = "9d0ba5dba665f0e057426cb3b88fae402658d361"


@dataclass(frozen=True)
class Settings:
    method: str
    lambda_rank: float = 1.3051794547562026
    lr_multiplier: float = 1.2832655111666686
    encoder_lr: float = 1e-5
    head_lr: float = .001
    vendor_lr: float = 4e-5
    weight_decay: float = .01
    vendor_weight_decay: float = 0.
    microbatch: int = 8
    vendor_batch: int = 32
    channel_tokens: int = 256
    vendor_tokens: int = 512
    memory_groups: int = 6
    memory_bytes: int = 1048576

    def __post_init__(self):
        if self.method not in METHODS or min(self.microbatch, self.vendor_batch, self.memory_groups) < 1:
            raise ValueError("Invalid method or batch/memory capacity")
        if not 2 <= self.channel_tokens <= 256 or not 4 <= self.vendor_tokens <= 512:
            raise ValueError("Explicit BERT input bounds required")


def vendor_embeddings(hidden_states, mask):
    if len(hidden_states) < 4 or mask.ndim != 2 or (mask.sum(1) == 0).any():
        raise ValueError("Four layers and nonempty masks required")
    values = torch.stack(hidden_states[-4:]).sum(0).float()
    return (values*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)


def vendor_scores(embeddings, slots):
    """Exact mean-cosine normalization; self terms INCLUDE advertisement diagonal."""
    z = F.normalize(embeddings.float(), dim=1)
    if (embeddings.norm(dim=1) == 0).any():
        raise ValueError("Zero advertisement representation")
    ids = torch.as_tensor(slots, device=z.device)
    n = int(ids.max())+1
    means = torch.stack([z[ids == i].mean(0) for i in range(n)])
    similarity = means @ means.T  # Equals all-pair cosine mean, including self diagonal.
    i, j = torch.triu_indices(n, n, 1, device=z.device)
    denominator = similarity.diag()[i]+similarity.diag()[j]
    if (denominator <= 0).any():
        raise ValueError("Undefined normalization; do not silently add an epsilon")
    return 2*similarity[i, j]/denominator


def h_pool(table, slots):
    nrecords, naccounts = len(slots), max(slots)+1
    if table.shape != (nrecords*(nrecords-1)//2,):
        raise ValueError("Full off-diagonal record table required")
    i, j = torch.triu_indices(nrecords, nrecords, 1, device=table.device)
    matrix = table.new_zeros((nrecords, nrecords))
    matrix[i, j], matrix[j, i] = table, table
    assigned = torch.as_tensor(slots, device=table.device)
    members = [torch.where(assigned == k)[0] for k in range(naccounts)]
    values = []
    for a in range(naccounts):
        for b in range(a+1, naccounts):
            block = matrix[members[a][:, None], members[b][None, :]].flatten().double()
            values.append((.5*block.mean()+.5*(torch.logsumexp(block, 0)-math.log(len(block)))).to(table.dtype))
    return torch.stack(values)


def h_loss(scores, labels, n, rank_weight):
    if scores.shape != (n*(n-1)//2,) or labels.shape != scores.shape:
        raise ValueError("Unordered account-pair scores/labels mismatch")
    if not torch.isin(labels, labels.new_tensor([0, 1])).all():
        raise ValueError("Binary observed-account supervision required")
    i, j = torch.triu_indices(n, n, 1, device=scores.device)
    matrix = scores.new_zeros((n, n))
    matrix[i, j], matrix[j, i] = scores, scores
    positive = torch.zeros((n, n), dtype=torch.bool, device=scores.device)
    positive[i, j], positive[j, i] = labels.bool(), labels.bool()
    negative = ~positive & ~torch.eye(n, dtype=torch.bool, device=scores.device)
    ranks, hard = [], []
    for q in range(n):
        pos, neg = matrix[q, positive[q]], matrix[q, negative[q]]
        if len(pos) == 0 or len(neg) < 5:
            raise ValueError("The unchanged hard-top-5 objective requires five negatives")
        differences = neg[None, :]-pos[:, None]
        extended = torch.cat((differences.new_zeros((len(pos), 1)), differences), 1)
        peak = extended.detach().amax(1, keepdim=True)
        ranks.append((torch.logsumexp(extended-peak, 1)+peak[:, 0]).mean())
        selected = torch.argsort(neg.detach(), descending=True, stable=True)[:5]
        hard.append(F.softplus(neg[selected][None, :]-pos[:, None]).mean())
    # Preserve the existing unweighted mean over unordered pairs.
    bce = F.binary_cross_entropy_with_logits(scores, labels)
    rank, hard_loss = torch.stack(ranks).mean(), torch.stack(hard).mean()
    return bce+rank_weight*rank+.5*hard_loss


def h_objective(table, group, regroup_seed, settings, teacher=None, history=False):
    group.validate(training=True)
    natural, regrouped = assignment(group), assignment(group, regroup_seed)
    predictions = [h_pool(table, slots) for slots in (natural, regrouped)]
    labels = table.new_tensor(group.labels())
    supervised = sum(h_loss(p, labels, len(group.views), settings.lambda_rank) for p in predictions)/2
    distillation = table.new_zeros(())
    if history and settings.method == "H_full":
        if teacher is None:
            raise ValueError("Full H requires its retained birth teacher")
        target = h_pool(teacher.to(table), natural).detach()
        distillation = (predictions[0]-target).square().mean()
    if history and settings.method == "H_SEQ":
        raise ValueError("H_SEQ cannot consume a history sample")
    total = (.1 if history else 1.)*supervised+.5*distillation
    return total, {"supervised": float(supervised.detach()), "distillation": float(distillation.detach()),
                   "weighted": float(total.detach())}


class Model(nn.Module):
    def __init__(self, bert, tokenizer, settings: Settings):
        super().__init__()
        self.encoder, self.tokenizer, self.settings = bert, tokenizer, settings
        self.head = (nn.Identity() if settings.method == "VendorLink_SEQ" else
                     nn.Sequential(nn.Linear(4*bert.config.hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)))
        self.classifier = None
        self.classes = ()
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)

    def start_vendor_stage(self, classes, initialization_seed):
        if self.settings.method != "VendorLink_SEQ" or len(set(classes)) < 2 or len(classes) != len(set(classes)):
            raise ValueError("VendorLink requires current-stage training classes")
        self.classes = tuple(sorted(classes))
        torch.manual_seed(initialization_seed)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(self.classes)).to(next(self.encoder.parameters()).device)
        nn.init.normal_(self.classifier.weight, std=self.encoder.config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, texts, limit, hidden=False):
        device = next(self.parameters()).device
        features = self.tokenizer(texts, padding=True, truncation=True, max_length=limit, return_tensors="pt")
        features = {k: v.to(device) for k, v in features.items()}
        output = self.encoder(**features, output_hidden_states=hidden, return_dict=True)
        return output, features["attention_mask"]

    def records(self, group, check=lambda: None):
        records = [r for v in group.views for r in v.records]
        texts = [r.title for r in records]+[r.description for r in records]
        parts = []
        for start in range(0, len(texts), self.settings.microbatch):
            check()
            output, _ = self.encode(texts[start:start+self.settings.microbatch], self.settings.channel_tokens)
            parts.append(output.last_hidden_state[:, 0].float())
        title, description = F.normalize(torch.cat(parts), dim=1).chunk(2)
        return torch.cat((title, description), dim=1)/math.sqrt(2)

    def table(self, z):
        i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
        return torch.cat([self.head(torch.cat(((z[i[k:k+512]]-z[j[k:k+512]]).abs(),
            z[i[k:k+512]]*z[j[k:k+512]]), 1)).flatten() for k in range(0, len(i), 512)])

    def backward_h(self, group, step_seed, teacher=None, history=False, check=lambda: None):
        torch.manual_seed(step_seed)
        z = self.records(group, check)
        with torch.no_grad():
            values = self.table(z.detach())
        proxy = values.detach().requires_grad_(True)
        loss, logs = h_objective(proxy, group, seed_for(step_seed, "regroup"), self.settings, teacher, history)
        gradient, = torch.autograd.grad(loss, proxy)
        leaf = z.detach().requires_grad_(True)
        i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
        for k in range(0, len(i), 512):
            check()
            u, v = leaf[i[k:k+512]], leaf[j[k:k+512]]
            self.head(torch.cat(((u-v).abs(), u*v), 1)).flatten().backward(gradient[k:k+512])
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise ValueError("Missing/nonfinite encoder-chain gradient")
        z.backward(leaf.grad)
        return logs

    @torch.no_grad()
    def teacher(self, group, check=lambda: None):
        self.eval()
        return self.table(self.records(group, check)).float().cpu().clone()

    @torch.no_grad()
    def score(self, group, check=lambda: None):
        self.eval()
        slots = assignment(group)
        if self.settings.method != "VendorLink_SEQ":
            return h_pool(self.table(self.records(group, check)), slots).cpu()
        texts = [r.title+" [SEP] "+r.description for v in group.views for r in v.records]
        parts = []
        for k in range(0, len(texts), self.settings.microbatch):
            check()
            output, mask = self.encode(texts[k:k+self.settings.microbatch], self.settings.vendor_tokens, True)
            parts.append(vendor_embeddings(output.hidden_states, mask))
        return vendor_scores(torch.cat(parts), slots).cpu()


def make_optimizer(model):
    s = model.settings
    if s.method == "VendorLink_SEQ":
        if model.classifier is None:
            raise ValueError("Current-stage class head must be initialized first")
        return torch.optim.AdamW(model.parameters(), lr=s.vendor_lr, weight_decay=s.vendor_weight_decay, eps=1e-8)
    return torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": s.encoder_lr*s.lr_multiplier},
                             {"params": model.head.parameters(), "lr": s.head_lr*s.lr_multiplier}],
                            weight_decay=s.weight_decay, eps=1e-8)


def update_h(model, optimizer, current, history, teacher, step_seed, check=lambda: None):
    if model.settings.method not in METHODS[1:]:
        raise ValueError("H entry called with another method")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logs = {"current": model.backward_h(current, seed_for(step_seed, "current"), check=check)}
    if history is not None:
        logs["history"] = model.backward_h(history, seed_for(step_seed, "history"), teacher, True, check)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    return {**logs, "gradient_norm": float(norm), "optimizer_updates": 1}


def update_vendor(model, optimizer, records, identities, step_seed, check=lambda: None):
    model.train()
    torch.manual_seed(step_seed)
    optimizer.zero_grad(set_to_none=True)
    lookup = {identity: k for k, identity in enumerate(model.classes)}
    labels = torch.tensor([lookup[k] for k in identities], device=next(model.parameters()).device)
    output, _ = model.encode([r.title+" [SEP] "+r.description for r in records], model.settings.vendor_tokens)
    loss = F.cross_entropy(model.classifier(model.dropout(output.pooler_output)), labels)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    return {"cross_entropy": float(loss.detach()), "gradient_norm": float(norm), "optimizer_updates": 1}


class Memory:
    def __init__(self, seed, settings):
        self.rng = random.Random(seed)
        self.settings = settings
        self.seen, self.entries = 0, []

    def retain(self, groups, stage, model, check=lambda: None):
        if stage not in (1, 2):
            raise ValueError("Only completed nonfinal stages produce history")
        # Select first, then encode new survivors only; existing teachers never refresh.
        selected = list(self.entries)
        for group in sorted(groups, key=lambda g: g.uid):
            self.seen += 1
            slot = len(selected) if len(selected) < self.settings.memory_groups else self.rng.randrange(self.seen)
            if slot < self.settings.memory_groups:
                row = {"group": asdict(group), "origin": stage, "teacher": None}
                if slot == len(selected):
                    selected.append(row)
                else:
                    selected[slot] = row
        for row in selected:
            if row["teacher"] is None:
                row["teacher"] = model.teacher(Group.from_dict(row["group"]), check)
        self.entries = selected
        if len(self.bytes()) > self.settings.memory_bytes:
            raise ValueError("Retained text, teachers and sampler state exceed the memory budget")

    def state(self):
        return {"seen": self.seen, "entries": self.entries, "rng": self.rng.getstate()}

    def restore(self, state):
        self.seen, self.entries = state["seen"], state["entries"]
        self.rng.setstate(state["rng"])

    def bytes(self):
        buffer = io.BytesIO()
        torch.save(self.state(), buffer)
        return buffer.getvalue()

    def draw(self, draw_seed):
        row = random.Random(draw_seed).choice(self.entries)
        return Group.from_dict(row["group"]), row["teacher"]
