"""Trainable-text continual components; no project data loader or formal runner.

Only a supplied current batch and self-contained retained edges enter this module.
The existing fixed-feature experiment and closed transfer runners are unchanged.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from step28_continual_core import Preprocessor, current_permutation, make_head, torch_module
import step28_labse_finetune_common as text_common

FIELDS = ("title", "description")
NUMERIC_COUNT = 51
TOKEN_BUDGET = 256
HEADER_BYTES = 1024
ENTRY_HEADER = struct.Struct("<QQI")
NUMERIC_BYTES = NUMERIC_COUNT * 8
POLICY_PATH = Path(__file__).resolve().parents[1] / "schema/step28_continual_text_policy.json"


def load_policy() -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    opt = policy["optimizer"]
    if (policy["input_order"] != ["legacy18", "dynamic_redacted_labse6", "identity33"]
            or policy["head"] != [57, 32, 1]
            or policy["order_seeds"] != [11, 23, 37]
            or (policy["epochs_per_stage"], policy["chunk_batch_size"], policy["token_budget"],
                policy["replay_maximum_edges"], policy["memory_bytes"], policy["threshold_logit"])
            != (2, 4, 256, 16, 524288, 0)
            or (opt["name"], opt["encoder_lr"], opt["encoder_weight_decay"], opt["head_lr"],
                opt["head_weight_decay"], opt["betas"], opt["epsilon"], opt["foreach"], opt["combined_clip_norm"])
            != ("AdamW", 2e-5, .01, .001, 0., [.9, .999], 1e-8, False, 1.)):
        raise ValueError("Component numerical policy differs from implementation")
    return policy


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def text_order(value: str) -> tuple[bytes, bytes]:
    raw = value.encode("utf-8")
    return hashlib.sha256(raw).digest(), raw


def canonical_fields(fields: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if set(fields) != set(FIELDS):
        raise ValueError("Only redacted title and description are model text")
    result = {}
    for field in FIELDS:
        values = fields[field]
        if isinstance(values, str) or not values or any(
                not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError("Every field needs nonempty unique text")
        result[field] = tuple(sorted(set(values), key=text_order))
    return result


@dataclass
class PairBatch:
    accounts: dict[str, dict[str, tuple[str, ...]]]
    edges: tuple[tuple[str, str], ...]
    numeric: np.ndarray  # raw legacy18 + identity33; NaN allowed before imputation
    labels: np.ndarray | None = None

    def validate(self, *, training: bool = False, complete_world: bool = False) -> None:
        if not self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("Empty or duplicate edges")
        endpoints = set()
        for left, right in self.edges:
            if not isinstance(left, str) or not isinstance(right, str) or not left or not left < right:
                raise ValueError("Edges must use distinct canonical public endpoints")
            endpoints.update((left, right))
        if endpoints != set(self.accounts):
            raise ValueError("Batch must contain exactly the requested endpoints")
        for fields in self.accounts.values():
            if fields != canonical_fields(fields):
                raise ValueError("Account fields are not canonical unique text")
        if (self.numeric.shape != (len(self.edges), NUMERIC_COUNT)
                or self.numeric.dtype.kind != "f" or np.isinf(self.numeric).any()):
            raise ValueError("Expected raw 51-column public numeric inputs")
        if training and self.labels is None:
            raise ValueError("Training requires supplied binary labels")
        if self.labels is not None and (self.labels.shape != (len(self.edges),)
                                       or not np.isin(self.labels, [0, 1]).all()):
            raise ValueError("Invalid supplied labels")
        if complete_world and (len(endpoints) != 28 or len(self.edges) != 378
                               or set(self.edges) != set(itertools.combinations(sorted(endpoints), 2))):
            raise ValueError("Current-world input is not a complete K28 graph")


def edge_payload(batch: PairBatch, index: int) -> bytes:
    """One edge owns all its inputs; no hidden cross-edge dictionary or old vectors."""
    left, right = batch.edges[index]
    if batch.labels is None:
        raise ValueError("Memory insertion requires an arrived training label")
    numeric = np.asarray(batch.numeric[index], dtype="<f8")
    if numeric.shape != (NUMERIC_COUNT,) or np.isinf(numeric).any() or batch.labels[index] not in (0, 1):
        raise ValueError("Invalid retained numeric input or label")
    fields = [canonical_fields(batch.accounts[uid]) for uid in (left, right)]
    return numeric.tobytes() + bytes([int(batch.labels[index])]) + json_bytes([left, right, *fields])


def batch_from_payloads(payloads: Sequence[bytes]) -> PairBatch:
    accounts, edges, numeric, labels = {}, [], [], []
    for payload in payloads:
        if len(payload) <= NUMERIC_BYTES + 1:
            raise ValueError("Truncated retained edge")
        values = np.frombuffer(payload[:NUMERIC_BYTES], dtype="<f8").copy()
        label = payload[NUMERIC_BYTES]
        left, right, left_fields, right_fields = json.loads(payload[NUMERIC_BYTES + 1:])
        for uid, fields in ((left, left_fields), (right, right_fields)):
            fields = canonical_fields(fields)
            if uid in accounts and accounts[uid] != fields:
                raise ValueError("Retained endpoint has conflicting text")
            accounts[uid] = fields
        edges.append((left, right))
        numeric.append(values)
        labels.append(label)
    batch = PairBatch(accounts, tuple(edges), np.asarray(numeric, dtype="<f8"),
                      np.asarray(labels, dtype=np.uint8))
    batch.validate(training=True)
    return batch


class ByteMemory:
    """Random-priority PREFIX under an additive byte budget, not uniform reservoir.

    Give each unique arriving edge an independent random priority. Keep the
    longest priority-sorted prefix that fits; never skip the first nonfitting
    edge to pack cheaper later edges. The cutoff only decreases as arrivals are
    added, so discarded payloads are never needed again. Inclusion probabilities
    depend on costs. This is a simple label-blind reference, not a new selector.
    """

    def __init__(self, budget: int = 524288, seed: int = 11):
        if type(budget) is not int or budget <= HEADER_BYTES:
            raise ValueError("Memory budget too small")
        self.budget, self.seen, self.last_stage = budget, 0, 0
        self.entries: list[tuple[int, int, bytes]] = []
        self.cutoff: tuple[int, int] | None = None
        self.selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))

    @property
    def used_bytes(self) -> int:
        return HEADER_BYTES + sum(ENTRY_HEADER.size + len(x[2]) for x in self.entries)

    def _insert(self, payload: bytes) -> None:
        if HEADER_BYTES + ENTRY_HEADER.size + len(payload) > self.budget:
            raise ValueError("A full edge exceeds the budget; do not truncate or silently exclude it")
        self.seen += 1
        key = (int(self.selection_rng.bit_generator.random_raw()), self.seen)
        if self.cutoff is not None and key >= self.cutoff:
            return
        self.entries.append((*key, payload))
        self.entries.sort(key=lambda x: x[:2])
        while self.used_bytes > self.budget:
            removed = self.entries.pop()
            self.cutoff = removed[:2]  # smaller than any previously discarded key

    def update_after_stage(self, batches: Iterable[PairBatch], stage: int) -> None:
        if stage != self.last_stage + 1:
            raise ValueError("Insert each arrival stage exactly once, after training")
        seen_before = self.seen
        for batch in batches:
            batch.validate(training=True)
            for index in range(len(batch.edges)):
                self._insert(edge_payload(batch, index))
        if self.seen == seen_before:
            raise ValueError("Cannot advance memory past an empty arrival")
        self.last_stage = stage
        self.to_bytes()

    def sample(self, maximum: int = 16) -> PairBatch | None:
        if maximum < 1:
            raise ValueError("Replay size must be positive")
        if not self.entries:
            return None
        indices = self.replay_rng.choice(len(self.entries), min(maximum, len(self.entries)), replace=False)
        return batch_from_payloads([self.entries[int(i)][2] for i in indices])

    def to_bytes(self) -> bytes:
        state = json_bytes({"budget": self.budget, "seen": self.seen,
                            "last_stage": self.last_stage, "cutoff": self.cutoff,
                            "selection_rng": self.selection_rng.bit_generator.state,
                            "replay_rng": self.replay_rng.bit_generator.state})
        if len(state) + 4 > HEADER_BYTES or self.used_bytes > self.budget:
            raise ValueError("Retained state exceeds its declared bytes")
        header = struct.pack("<I", len(state)) + state + bytes(HEADER_BYTES - 4 - len(state))
        return header + b"".join(ENTRY_HEADER.pack(key, arrival, len(payload)) + payload
                                 for key, arrival, payload in self.entries)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ByteMemory":
        if len(payload) < HEADER_BYTES:
            raise ValueError("Truncated memory header")
        size, = struct.unpack_from("<I", payload)
        if size > HEADER_BYTES - 4:
            raise ValueError("Invalid memory header length")
        state = json.loads(payload[4:4 + size])
        obj = cls(state["budget"])
        obj.seen, obj.last_stage = state["seen"], state["last_stage"]
        obj.cutoff = tuple(state["cutoff"]) if state["cutoff"] is not None else None
        obj.selection_rng.bit_generator.state = state["selection_rng"]
        obj.replay_rng.bit_generator.state = state["replay_rng"]
        offset = HEADER_BYTES
        while offset < len(payload):
            if offset + ENTRY_HEADER.size > len(payload):
                raise ValueError("Truncated memory entry header")
            key, arrival, count = ENTRY_HEADER.unpack_from(payload, offset)
            offset += ENTRY_HEADER.size
            if offset + count > len(payload) or not 1 <= arrival <= obj.seen:
                raise ValueError("Invalid memory entry size/arrival")
            edge = payload[offset:offset + count]
            batch_from_payloads([edge]).validate(training=True)
            obj.entries.append((key, arrival, edge))
            offset += count
        keys = [entry[:2] for entry in obj.entries]
        if (keys != sorted(set(keys)) or obj.last_stage < 0
                or (obj.cutoff is not None and any(key >= obj.cutoff for key in keys))
                or obj.to_bytes() != payload):
            raise ValueError("Saved memory state/ordering differs")
        if obj.entries:
            batch_from_payloads([entry[2] for entry in obj.entries])
        return obj


def prepare_text(batch: PairBatch, tokenizer: Any) -> tuple[list[str], dict[str, slice]]:
    batch.validate()
    texts = sorted({text for fields in batch.accounts.values()
                    for field in FIELDS for text in fields[field]}, key=text_order)
    chunks, spans = [], {}
    for text in texts:
        start = len(chunks)
        chunks.extend(text_common.chunk_text_exact(tokenizer, text, TOKEN_BUDGET))
        spans[text] = slice(start, len(chunks))
    return chunks, spans


def aggregate_chunks(batch: PairBatch, chunks: Any, spans: Mapping[str, slice]) -> Any:
    """chunks -> original unique text -> account field -> six pair features."""
    torch = torch_module()
    vectors = {text: text_common.torch_unit_mean(chunks[span]) for text, span in spans.items()}
    accounts = {uid: {field: torch.stack([vectors[t] for t in fields[field]])
                      for field in FIELDS} for uid, fields in batch.accounts.items()}
    return torch.stack([text_common.torch_six_pair_aggregates(accounts[left], accounts[right])
                        for left, right in batch.edges])


def dynamic_features(encoder: Any, tokenizer: Any, batch: PairBatch, *,
                     chunk_batch_size: int = 4, bf16: bool = True) -> Any:
    torch = torch_module()
    if chunk_batch_size < 1:
        raise ValueError("Invalid chunk microbatch size")
    chunks, spans = prepare_text(batch, tokenizer)
    device = next(encoder.parameters()).device
    outputs = []
    for start in range(0, len(chunks), chunk_batch_size):
        features = tokenizer(chunks[start:start + chunk_batch_size], padding=True, truncation=False,
                             add_special_tokens=True, return_tensors="pt")
        if int(features["attention_mask"].sum(dim=1).max()) > TOKEN_BUDGET:
            raise ValueError("Chunk exceeds the tokenizer budget")
        features = {name: tensor.to(device) for name, tensor in features.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            encoded = encoder(features)["sentence_embedding"]
        outputs.append(torch.nn.functional.normalize(encoded.float(), dim=1, eps=1e-12))
    values = aggregate_chunks(batch, torch.cat(outputs), spans)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite dynamic features")
    return values


def combine_numpy(numeric: np.ndarray, dynamic: np.ndarray) -> np.ndarray:
    if numeric.shape != (len(dynamic), 51) or dynamic.shape != (len(numeric), 6):
        raise ValueError("Numeric/dynamic columns differ")
    return np.column_stack((numeric[:, :18], dynamic, numeric[:, 18:]))


def scaled_features(numeric: np.ndarray, dynamic: Any, scaler: Preprocessor) -> Any:
    torch = torch_module()
    # Match the old preprocessor's float64 affine calculation then float32 head.
    fixed = torch.as_tensor(numeric, dtype=torch.float64, device=dynamic.device)
    raw = torch.cat((fixed[:, :18], dynamic.double(), fixed[:, 18:]), dim=1)
    medians, means, scales = [torch.as_tensor(value, dtype=torch.float64, device=raw.device)
                              for value in (scaler.medians, scaler.means, scaler.scales)]
    result = ((torch.where(torch.isnan(raw), medians, raw) - means) / scales).float()
    if result.shape != (len(numeric), 57) or not torch.isfinite(result).all():
        raise ValueError("Invalid scaled features")
    return result


def fit_first_scaler(encoder: Any, tokenizer: Any, first_batches: Iterable[PairBatch], **kwargs) -> Preprocessor:
    """Caller supplies only first-arrival worlds and the untouched original encoder."""
    torch = torch_module()
    previous_mode = encoder.training
    encoder.eval()
    values = []
    try:
        with torch.no_grad():
            for batch in first_batches:
                dynamic = dynamic_features(encoder, tokenizer, batch, **kwargs).cpu().numpy()
                values.append(combine_numpy(batch.numeric, dynamic))
    finally:
        encoder.train(previous_mode)
    if not values:
        raise ValueError("Missing first-arrival inputs")
    return Preprocessor.fit(np.concatenate(values))


def make_text_optimizer(encoder: Any, head: Any):
    torch = torch_module()
    return torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": 2e-5, "weight_decay": .01},
        {"params": head.parameters(), "lr": .001, "weight_decay": 0.},
    ], betas=(.9, .999), eps=1e-8, foreach=False)


def load_labse(device: str = "cuda:0") -> tuple[Any, Any]:
    """Existing local payload only; activation recomputation preserves dropout RNG."""
    torch = torch_module()
    from sentence_transformers import SentenceTransformer

    policy = load_policy()
    text_common.verify_labse_payload(policy)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = SentenceTransformer(str(POLICY_PATH.parents[1] / policy["labse_model"]["path"]),
                                  device=device, local_files_only=True)
    if encoder.max_seq_length != TOKEN_BUDGET or encoder.get_sentence_embedding_dimension() != 768:
        raise ValueError("Existing LaBSE structure differs")
    encoder.default_prompt_name = None
    encoder[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    encoder.eval()
    return encoder, encoder.tokenizer


def branch_seed(order_seed: int, stage: int, epoch: int, step: int, branch: str) -> int:
    if branch not in ("current", "history"):
        raise ValueError("Unknown stochastic branch")
    return int.from_bytes(hashlib.sha256(json_bytes([order_seed, stage, epoch, step, branch])).digest()[:8],
                          "little") % (2**63)


@contextmanager
def paired_rng(device: Any, seed: int):
    torch = torch_module()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def batch_logits(encoder: Any, head: Any, tokenizer: Any, batch: PairBatch,
                 scaler: Preprocessor, **kwargs) -> Any:
    values = dynamic_features(encoder, tokenizer, batch, **kwargs)
    return head(scaled_features(batch.numeric, values, scaler)).squeeze(1)


def batch_loss(encoder: Any, head: Any, tokenizer: Any, batch: PairBatch,
               scaler: Preprocessor, **kwargs) -> Any:
    torch = torch_module()
    batch.validate(training=True)
    logits = batch_logits(encoder, head, tokenizer, batch, scaler, **kwargs)
    labels = torch.as_tensor(batch.labels, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def train_update(encoder: Any, head: Any, optimizer: Any, tokenizer: Any,
                 current: PairBatch, history: PairBatch | None, scaler: Preprocessor, *,
                 current_seed: int, history_seed: int, **kwargs) -> dict[str, float]:
    """mean current BCE + mean historical BCE, then ONE clip and AdamW update.

    Sequential backward frees the current graph before history encoding. It is
    the same summed objective; no optimizer update or gradient reset in between.
    Each branch uses its own deterministic RNG context through checkpoint backward.
    This function neither reads old files nor inserts current examples in memory.
    """
    torch = torch_module()
    encoder.train()
    head.train()
    device = next(encoder.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    report = {}
    for name, batch, seed in (("current", current, current_seed), ("history", history, history_seed)):
        if batch is None:
            continue
        with paired_rng(device, seed):
            loss = batch_loss(encoder, head, tokenizer, batch, scaler, **kwargs)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite branch loss")
            loss.backward()
            report[name + "_bce"] = float(loss.detach())
        del loss
    parameters = list(encoder.parameters()) + list(head.parameters())
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    report["pre_clip_gradient_norm"] = float(norm)
    return report


def train_stage(encoder: Any, head: Any, optimizer: Any, tokenizer: Any,
                current_worlds: Sequence[PairBatch], scaler: Preprocessor, *,
                order_seed: int, stage: int, memory: ByteMemory | None = None,
                epochs: int = 2, **kwargs) -> dict[str, Any]:
    """Current complete worlds only; caller inserts each stage once AFTER this call."""
    if not current_worlds or stage < 1 or epochs < 1 or (stage == 1 and memory is not None):
        raise ValueError("Invalid stage or first-stage replay")
    if memory is not None and memory.last_stage != stage - 1:
        raise ValueError("Replay must contain only preceding arrivals")
    endpoints = set()
    for batch in current_worlds:
        batch.validate(training=True, complete_world=True)
        if endpoints.intersection(batch.accounts):
            raise ValueError("Current worlds overlap")
        endpoints.update(batch.accounts)
    digest = hashlib.sha256()
    updates, current_rows, replay_rows, loss_sum = 0, 0, 0, 0.
    seen_before = memory.seen if memory is not None else 0
    for epoch in range(epochs):
        order = current_permutation(len(current_worlds), order_seed, stage, epoch)
        digest.update(np.asarray(order, dtype="<i8").tobytes())
        for step, index in enumerate(order):
            history = memory.sample(16) if memory is not None else None
            report = train_update(encoder, head, optimizer, tokenizer, current_worlds[int(index)],
                                  history, scaler,
                                  current_seed=branch_seed(order_seed, stage, epoch, step, "current"),
                                  history_seed=branch_seed(order_seed, stage, epoch, step, "history"),
                                  **kwargs)
            updates += 1
            current_rows += len(current_worlds[int(index)].edges)
            replay_rows += len(history.edges) if history is not None else 0
            loss_sum += report["current_bce"] + report.get("history_bce", 0.)
    if memory is not None and memory.seen != seen_before:
        raise ValueError("Repeated optimization inserted history")
    return {"updates": updates, "current_presentations": current_rows,
            "replay_presentations": replay_rows, "mean_update_loss": loss_sum / updates,
            "current_order_sha256": digest.hexdigest()}


def score_batch(encoder: Any, head: Any, tokenizer: Any, batch: PairBatch,
                scaler: Preprocessor, **kwargs) -> np.ndarray:
    torch = torch_module()
    encoder.eval()
    head.eval()
    with torch.no_grad():
        result = batch_logits(encoder, head, tokenizer, batch, scaler, **kwargs).cpu().numpy().copy()
    if result.shape != (len(batch.edges),) or not np.isfinite(result).all():
        raise ValueError("Invalid raw-logit scores")
    return result


def save_state(path: Path, encoder: Any, head: Any, optimizer: Any, scaler: Preprocessor,
               progress: dict[str, int], memory: ByteMemory | None) -> None:
    torch = torch_module()
    state = {"encoder": encoder.state_dict(), "head": head.state_dict(),
             "optimizer": optimizer.state_dict(), "progress": progress,
             "scaler": {name: torch.from_numpy(getattr(scaler, name).copy())
                        for name in ("medians", "means", "scales")},
             "memory": memory.to_bytes() if memory is not None else None,
             "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()}
    with path.open("xb") as handle:
        torch.save(state, handle)


def restore_state(path: Path, encoder: Any, head: Any, optimizer: Any) -> tuple[Preprocessor, dict, ByteMemory | None]:
    torch = torch_module()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["policy_sha256"] != hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest():
        raise ValueError("Checkpoint uses another numerical contract")
    encoder.load_state_dict(state["encoder"], strict=True)
    head.load_state_dict(state["head"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler = Preprocessor(**{name: value.numpy().copy() for name, value in state["scaler"].items()})
    memory = ByteMemory.from_bytes(state["memory"]) if state["memory"] is not None else None
    return scaler, state["progress"], memory
