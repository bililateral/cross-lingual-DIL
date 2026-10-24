"""L2R text adaptation: paid query packages, real selection and encoder CE."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import contextlib
import json
import math
import random

import numpy as np

import step28_record_replay as record

data = record.data


@dataclass(frozen=True)
class Account:
    uid: str
    domain: str
    group: str
    component: int
    items: tuple

    @classmethod
    def restore(cls, row: dict) -> Account:
        return cls(row['uid'], row['domain'], row['group'], row['component'],
                   tuple(tuple(item) for item in row['items']))


def accounts(group, domain: str, *, supervised: bool = True) -> list[Account]:
    group.validate()
    components = np.arange(28)
    if supervised:
        # Reuses the clique-shape validation, without using its randomized view.
        record.assignment(group, 0)
        adjacency = np.eye(28, dtype=bool)
        i, j = np.triu_indices(28, 1)
        adjacency[i, j] = adjacency[j, i] = group.labels
        components = np.array([np.flatnonzero(row)[0] for row in adjacency])
    return [Account(uid, domain, group.uid, int(component), items)
            for uid, component, items in zip(group.sellers, components, group.items, strict=True)]


def same_identity(a: Account, b: Account) -> bool:
    return (a.domain, a.group, a.component) == (b.domain, b.group, b.component)


def unique(rows: list[Account]) -> list[Account]:
    by_id = {}
    for row in rows:
        if row.uid in by_id and row != by_id[row.uid]:
            raise ValueError('Conflicting surviving account relation/text')
        by_id[row.uid] = row
    return [by_id[key] for key in sorted(by_id)]


def encode(model, rows: list[Account], c: dict, check=lambda: None):
    """Raw transformer CLS, bypassing SentenceTransformer Normalize and the head."""
    import torch
    if not rows:
        raise ValueError('Empty encoding block')
    records = [item for row in rows for item in row.items]
    texts = [r[1] for r in records] + [r[2] for r in records]
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(texts), c['input']['microbatch']):
        check()
        f = model.encoder.tokenizer(texts[start:start+c['input']['microbatch']],
                                   padding=True, truncation=False, return_tensors='pt')
        if f['input_ids'].shape[1] > c['input']['token_budget']:
            raise ValueError('Token limit exceeded; no silent truncation')
        f = {key: value.to(device) for key, value in f.items()}
        cast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
        with cast:
            outputs.append(model.encoder[0](f)['token_embeddings'][:, 0].float())
    title, description = torch.cat(outputs).chunk(2)
    counts = [len(a.items) for a in rows]
    result = torch.stack([torch.cat((t.mean(0), d.mean(0)))/math.sqrt(2)
                          for t, d in zip(title.split(counts), description.split(counts), strict=True)])
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite account representation')
    return result


class BM25:
    """Only current-fit accounts; no archive callback, persistent vectors or labels."""
    def __init__(self, rows: list[Account], tokenizer):
        self.rows = unique(rows)
        self.by_id = {a.uid: a for a in self.rows}
        self.tokenizer = tokenizer
        self.counts = [Counter(self.tokens(a)) for a in self.rows]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=np.float64)
        self.avgdl = float(self.lengths.mean())
        if not self.avgdl > 0:
            raise ValueError('Entire current BM25 corpus has no tokens')
        self.df = Counter(t for count in self.counts for t in count)

    def tokens(self, account: Account) -> list[int]:
        return [token for _, title, description in account.items for text in (title, description)
                for token in self.tokenizer.encode(text, add_special_tokens=False, truncation=False)]

    def candidates(self, query: Account) -> list[Account]:
        terms, n = set(self.tokens(query)), len(self.rows)
        scores = np.zeros(n)
        for t in sorted(terms):
            tf = np.array([count[t] for count in self.counts], dtype=np.float64)
            idf = math.log(1+(n-self.df[t]+.5)/(self.df[t]+.5))
            scores += idf*tf*1.2/(tf+.2*(1-.72+.72*self.lengths/self.avgdl))
        legal = [i for i, a in enumerate(self.rows) if a.uid != query.uid and not same_identity(a, query)]
        return [self.rows[i] for i in sorted(legal, key=lambda i: (-scores[i], self.rows[i].uid))[:500]]


def perpendicular(q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    denom = float(q @ q)
    if denom <= 0 or not np.isfinite(denom) or not np.isfinite(rows).all():
        raise ValueError('Invalid projection denominator or vector')
    return rows - (rows @ q)[:, None]*q[None, :]/denom


def isd(q: np.ndarray, rows: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not len(reference):
        raise ValueError('ISD reference must be nonempty')
    left, right = perpendicular(q, rows), perpendicular(q, reference)
    return np.linalg.norm(left[:, None, :]-right[None, :, :], axis=2).mean(1)


def priority(q: np.ndarray, p: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    perpendicular(q, candidates)  # Validate the projection, including q.
    pp = float(p @ q)*q/float(q @ q)
    dp = (candidates @ q)[:, None]*q/float(q @ q)
    difference = pp-dp
    pss = np.where(difference @ pp > 0, 1., -1.)*np.linalg.norm(difference, axis=1)
    values = .8*pss+.2*isd(q, candidates, candidates)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite PSS/ISD')
    return values


def choose(rows: list[Account], values: np.ndarray, count: int, *, largest: bool = True) -> list[Account]:
    if len(rows) != len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid selection scores')
    return [rows[i] for i in sorted(range(len(rows)), key=lambda i: (
        -float(values[i]) if largest else float(values[i]), rows[i].uid))[:count]]


class Memory:
    def __init__(self, spec: dict):
        self.spec, self.seen, self.stage = spec, 0, 0
        self.packages: dict[str, dict] = {}
        self.rng = random.Random(data.seed_for(spec['roots']['memory'], spec['order'], 'l2r_admission'))

    def to_bytes(self) -> bytes:
        rows = {uid: {'query': asdict(p['query']), 'positive': asdict(p['positive']),
                      'negatives': [asdict(a) for a in p['negatives']]}
                for uid, p in sorted(self.packages.items())}
        slots = sum(2+len(p['negatives']) for p in self.packages.values())
        payload = data.json_bytes(dict(spec=self.spec, seen=self.seen, stage=self.stage,
                                      packages=rows, rng=self.rng.getstate()))
        if len(rows) > 14 or slots > 168 or len(payload) > 1048576:
            raise ValueError('L2R paid historical capacity exceeded')
        for uid, p in self.packages.items():
            if (uid != p['query'].uid or not same_identity(p['query'], p['positive'])
                    or uid == p['positive'].uid or len(p['negatives']) > 10
                    or len({a.uid for a in p['negatives']}) != len(p['negatives'])
                    or any(same_identity(a, p['query']) for a in p['negatives'])):
                raise ValueError('Invalid L2R query package')
        unique([a for p in self.packages.values() for a in (p['query'], p['positive'], *p['negatives'])])
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes, spec: dict) -> Memory:
        row = json.loads(payload)
        if row['spec'] != spec:
            raise ValueError('Foreign L2R history')
        result = cls(spec)
        result.seen, result.stage = row['seen'], row['stage']
        result.rng.setstate(data.tuple_state(row['rng']))
        result.packages = {uid: {'query': Account.restore(p['query']),
                                'positive': Account.restore(p['positive']),
                                'negatives': [Account.restore(a) for a in p['negatives']]}
                           for uid, p in row['packages'].items()}
        if result.to_bytes() != payload:
            raise ValueError('L2R history roundtrip differs')
        return result

    def summary(self) -> dict:
        rows = [a for p in self.packages.values() for a in (p['query'], p['positive'], *p['negatives'])]
        return dict(bytes=len(self.to_bytes()), seen=self.seen, stage=self.stage,
                    slots=len(rows), unique_accounts=len(unique(rows)), queries=sorted(self.packages),
                    negative_counts={uid: len(p['negatives']) for uid, p in self.packages.items()})


class Stage:
    def __init__(self, spec: dict, stage: int, current: list, memory: Memory, model):
        if stage != memory.stage+1 or len(current) != 48:
            raise ValueError('L2R current-stage scope differs')
        self.spec, self.stage, self.memory = spec, stage, memory
        self.current = {g.uid: accounts(g, spec['order'][stage-1]) for g in current}
        self.index = BM25([a for rows in self.current.values() for a in rows], model.encoder.tokenizer)
        self.mc: dict[str, set[str]] = {}

    def rng(self, *parts) -> random.Random:
        return random.Random(data.seed_for(self.spec['roots']['memory'], self.spec['name'],
                                          self.spec['order'], self.stage, *parts))

    def select(self, model, c: dict, query: Account, positive: Account, step: int, check) -> tuple:
        import torch
        old = self.memory.packages.get(query.uid, {}).get('negatives', [])
        new_count = 5-min(2, len(old))
        available = self.index.candidates(query)
        if len(available) < new_count:
            raise ValueError('Current fixed corpus lacks expected legal negatives')
        rng = self.rng(step, query.uid, positive.uid, 'current_pool')
        pool = rng.sample(sorted(available, key=lambda a: a.uid), min(2*new_count, len(available)))
        old_rng = self.rng(step, query.uid, positive.uid, 'history_pool')
        old_pool = old_rng.sample(sorted(old, key=lambda a: a.uid), min(4, len(old)))
        rows = unique([query, positive, *pool, *old_pool])
        model.eval()
        with torch.no_grad():
            vectors = encode(model, rows, c, check).cpu().numpy()
        lookup = dict(zip((a.uid for a in rows), vectors, strict=True))
        q, p = lookup[query.uid], lookup[positive.uid]
        new = choose(pool, priority(q, p, np.stack([lookup[a.uid] for a in pool])), new_count)
        previous = (choose(old_pool, isd(q, np.stack([lookup[a.uid] for a in old_pool]),
                                        np.stack([lookup[a.uid] for a in new])), 2) if old_pool else [])
        self.mc.setdefault(query.uid, set()).update(a.uid for a in new)
        return new, previous

    def block(self, model, c: dict, queries: list[tuple[Account, list[Account]]], step: int,
              role: str, check) -> tuple:
        import torch
        selected, items = [], []
        for q, positives in queries:
            for p in sorted(positives, key=lambda a: a.uid):
                new, old = self.select(model, c, q, p, step, check)
                selected.extend((p, *new, *old))
                items.append((q, p, [a.uid for a in new], [a.uid for a in old]))
        documents = unique(selected)
        rows = unique(documents+[q for q, _ in queries])
        by_id = {a.uid: i for i, a in enumerate(rows)}
        model.train()
        torch.manual_seed(data.seed_for(self.spec['roots']['schedule'], self.spec['order'],
                                       self.stage, step, role, 'l2r_dropout'))
        z = encode(model, rows, c, check)
        losses: dict[str, list] = {}
        evidence = []
        for q, p, new, old in items:
            legal = [a for a in documents if a.uid == p.uid or
                     (a.uid != q.uid and not same_identity(q, a))]
            ids = [a.uid for a in legal]
            if p.uid not in ids or len(ids) < 2:
                raise ValueError('Missing positive or legal in-batch denominator')
            scores = z[by_id[q.uid]] @ z[[by_id[a.uid] for a in legal]].T
            loss = torch.logsumexp(scores, 0)-scores[ids.index(p.uid)]
            losses.setdefault(q.uid, []).append(loss)
            evidence.append(dict(query=q.uid, positive=p.uid, new=new, old=old, denominator=ids))
        objective = torch.stack([torch.stack(losses[q.uid]).mean() for q, _ in queries]).mean()
        return objective, dict(queries=len(queries), positive_items=len(items), accounts=len(rows), selections=evidence)

    def backward(self, model, group, c: dict, step: int, check=lambda: None) -> dict:
        current = self.current[group.uid]
        queries = [(q, [p for p in current if p.uid != q.uid and same_identity(q, p)]) for q in current]
        loss, log = self.block(model, c, queries, step, 'current', check)
        log['weighted'] = float(loss.detach())
        loss.backward()
        result = {'current': log}
        if self.memory.packages:
            pairs = [(p['query'], [p['positive']]) for _, p in sorted(self.memory.packages.items())]
            loss, log = self.block(model, c, pairs, step, 'history', check)
            log['weighted'] = float(.1*loss.detach())
            (.1*loss).backward()
            result['history'] = log
        return result

    def retain(self, model, c: dict, check=lambda: None) -> dict:
        import torch
        if self.stage == 3:
            return {'skipped_final_future_maintenance': True}
        memory = self.memory
        selected = list(sorted(memory.packages))
        current = self.index.by_id
        if not set(current) <= set(self.mc):
            raise ValueError('Not every current query has entered the training selections')
        for uid in sorted(current):
            memory.seen += 1
            if len(selected) < 14:
                selected.append(uid)
            else:
                index = memory.rng.randrange(memory.seen)
                if index < 14:
                    selected[index] = uid
        kept, evidence = {}, {}
        for uid in sorted(selected):
            check()
            rng = self.rng(uid, 'retain')
            candidates = [current[k] for k in sorted(self.mc.get(uid, set()))]
            if uid not in memory.packages:
                q = current[uid]
                positives = [a for a in self.current[q.group] if a.uid != uid and same_identity(a, q)]
                # Streaming reservoir on stable candidate IDs, not free archive fill.
                negative = []
                for count, a in enumerate(candidates, 1):
                    if len(negative) < 10:
                        negative.append(a)
                    else:
                        j = rng.randrange(count)
                        if j < 10:
                            negative[j] = a
                kept[uid] = dict(query=q, positive=rng.choice(positives), negatives=sorted(negative, key=lambda a: a.uid))
                evidence[uid] = dict(created=True, replaced=0)
                continue
            p = memory.packages[uid]
            old = p['negatives']
            k = max(1, len(old)//3) if old else 0
            fresh = [a for a in candidates if a.uid not in {a.uid for a in old} and not same_identity(a, p['query'])]
            r = min(3, len(old)-k, len(fresh)) if k else 0
            negative = old.copy()
            if r:
                anchors = rng.sample(old, k)
                remaining = [a for a in old if a.uid not in {a.uid for a in anchors}]
                old_pool = rng.sample(remaining, min(2*r, len(remaining)))
                new_pool = rng.sample(fresh, min(2*r, len(fresh)))
                rows = unique([p['query'], *anchors, *old_pool, *new_pool])
                model.eval()
                with torch.no_grad():
                    vectors = encode(model, rows, c, check).cpu().numpy()
                z = dict(zip((a.uid for a in rows), vectors, strict=True))
                q, reference = z[uid], np.stack([z[a.uid] for a in anchors])
                removed = choose(old_pool, isd(q, np.stack([z[a.uid] for a in old_pool]), reference), r, largest=False)
                added = choose(new_pool, isd(q, np.stack([z[a.uid] for a in new_pool]), reference), r)
                negative = [a for a in old if a.uid not in {a.uid for a in removed}]+added
            kept[uid] = {**p, 'negatives': sorted(negative, key=lambda a: a.uid)}
            evidence[uid] = dict(created=False, replaced=r, count=len(negative))
        memory.packages, memory.stage = kept, self.stage
        memory.to_bytes()
        return evidence

    def transient(self) -> dict:
        return dict(stage=self.stage, current_ids=sorted(self.current),
                    mc={q: sorted(ids) for q, ids in sorted(self.mc.items())})

    def restore_transient(self, row: dict) -> None:
        if row['stage'] != self.stage or row['current_ids'] != sorted(self.current):
            raise ValueError('L2R transient current source differs')
        if any(uid not in self.index.by_id for values in row['mc'].values() for uid in values):
            raise ValueError('L2R transient references noncurrent text')
        self.mc = {q: set(ids) for q, ids in row['mc'].items()}
