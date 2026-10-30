"""Trainable item/account pooling and symmetric pure-text pair classification."""
from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_continual_population_data as data


def build_model(encoder: Any, dimension: int, hidden: int = 128) -> Any:
    import torch

    class PairModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Sequential(torch.nn.Linear(dimension * 2, hidden),
                                            torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

        def pair_logits(self, accounts: Any) -> Any:
            left, right = torch.triu_indices(len(accounts), len(accounts), 1, device=accounts.device)
            u, v = accounts[left], accounts[right]
            return self.head(torch.cat(((u - v).abs(), u * v), dim=1)).flatten()

    return PairModel()


def make_optimizer(model: Any, config: dict) -> Any:
    import torch
    p = config["optimizer"]
    return torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": p["encoder_lr"],
          "weight_decay": p["encoder_weight_decay"]},
         {"params": model.head.parameters(), "lr": p["head_lr"],
          "weight_decay": p["head_weight_decay"]}],
        betas=tuple(p["betas"]), eps=p["eps"], foreach=p["foreach"])


def model_files(config: dict) -> dict:
    base = data.ROOT / config["model"]["path"]
    rows = [{"path": path.relative_to(base).as_posix(), "size_bytes": path.stat().st_size,
             "sha256": data.sha256(path)}
            for path in sorted(base.rglob("*"), key=lambda p: p.relative_to(base).as_posix())
            if path.is_file() and ".cache" not in path.parts and "__pycache__" not in path.parts
            and path.suffix != ".pyc"]
    # Preserve step7's canonical_hash: size_bytes keys and no trailing newline.
    content = hashlib.sha256(data.json_bytes(rows).rstrip(b"\n")).hexdigest()
    return {"files": rows, "file_count": len(rows),
            "total_size_bytes": sum(row["size_bytes"] for row in rows), "content_sha256": content}


def load_model(config: dict, device: str = "cuda:0") -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(config["initialization_seed"])
    encoder = SentenceTransformer(str(data.ROOT / config["model"]["path"]),
                                  device=device, local_files_only=True)
    encoder.default_prompt_name = None
    encoder.max_seq_length = config["model"]["token_budget"]
    if encoder.get_sentence_embedding_dimension() != config["model"]["embedding_dim"]:
        raise ValueError("LaBSE embedding dimension differs")
    if config["model"]["activation_checkpointing"]:
        encoder[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    # Head initialization must not depend on encoder loader's incidental RNG use.
    torch.manual_seed(config["initialization_seed"])
    return build_model(encoder, config["model"]["embedding_dim"], config["model"]["pair_head"][1]).to(device)


def pool_records(vectors: Any, counts: list[int]) -> Any:
    import torch
    if sum(counts) != len(vectors) or any(n <= 0 for n in counts):
        raise ValueError("Record/account alignment differs")
    normalized = torch.nn.functional.normalize(vectors.float(), dim=1)
    means = torch.stack([chunk.mean(dim=0) for chunk in normalized.split(counts)])
    return torch.nn.functional.normalize(means, dim=1)


def account_vectors(model: Any, group: data.Group, config: dict,
                    check_budget: Callable[[], None] = lambda: None) -> Any:
    import torch
    device = next(model.parameters()).device
    texts = [title + "\n" + description for rows in group.items for _, title, description in rows]
    counts = list(map(len, group.items))
    size = config["model"]["microbatch"]
    blocks = []
    for start in range(0, len(texts), size):
        check_budget()
        batch = texts[start:start + size]
        # Do not use SentenceTransformer.tokenize: it silently truncates long input.
        features = model.encoder.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        if features["input_ids"].shape[1] > config["model"]["token_budget"]:
            raise ValueError("An item exceeds the frozen token budget")
        features = {key: value.to(device) for key, value in features.items()}
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if device.type == "cuda" and config["model"]["encoder_bf16"] else contextlib.nullcontext())
        with context:
            vectors = model.encoder(features)["sentence_embedding"]
        blocks.append(vectors.float())
    return pool_records(torch.cat(blocks), counts)


def logits(model: Any, group: data.Group, config: dict,
           check_budget: Callable[[], None] = lambda: None) -> Any:
    return model.pair_logits(account_vectors(model, group, config, check_budget))


def backward_group(model: Any, group: data.Group, config: dict, seed: int,
                   check_budget: Callable[[], None] = lambda: None) -> float:
    import torch
    if group.labels is None:
        raise ValueError("Training requires authorized current/replay supervision")
    torch.manual_seed(seed)
    predicted = logits(model, group, config, check_budget)
    truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
    if predicted.shape != truth.shape or not torch.isfinite(predicted).all():
        raise ValueError("Nonfinite or misaligned train logits")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, truth, reduction="mean")
    loss.backward()
    return float(loss.detach())


def update(model: Any, optimizer: Any, current: data.Group, replay: data.Group | None,
           config: dict, current_seed: int, replay_seed: int,
           check_budget: Callable[[], None] = lambda: None) -> dict:
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = backward_group(model, current, config, current_seed, check_budget)
    historical = backward_group(model, replay, config, replay_seed, check_budget) if replay else None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"],
                                         error_if_nonfinite=True)
    check_budget()
    optimizer.step()
    return {"current_bce": loss, "replay_bce": historical, "gradient_norm_before_clip": float(norm)}


def score(model: Any, groups: list[data.Group], config: dict,
          check_budget: Callable[[], None] = lambda: None) -> np.ndarray:
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for group in groups:
            result.append(logits(model, group, config, check_budget).float().cpu().numpy())
    values = np.stack(result).astype(np.float32, copy=False)
    if values.shape != (len(groups), 378) or not np.isfinite(values).all():
        raise ValueError("Incomplete/nonfinite blind scores")
    return values


def state_digest(value: Any) -> str:
    """All tensors, scalar state and optimizer parameter groups, without aliases."""
    import torch
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(data.json_bytes([str(tensor.dtype), list(tensor.shape)]))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(data.json_bytes([str(type(key)), str(key)]))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(data.json_bytes(["sequence", len(item)]))
            for child in item:
                visit(child)
        else:
            digest.update(data.json_bytes(item))

    visit(value)
    return digest.hexdigest()


def save_state(path: Path, model: Any, optimizer: Any | None, metadata: dict) -> dict:
    import torch
    if path.exists():
        raise FileExistsError(path)
    payload = {"model": model.state_dict(), "metadata": metadata,
               "optimizer": optimizer.state_dict() if optimizer is not None else None}
    digest = state_digest(payload)
    torch.save(payload, path)
    return {"state_sha256": digest, **data.record(path, path.parent)}


def restore_state(path: Path, model: Any, optimizer: Any | None, expected_digest: str) -> dict:
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if state_digest(saved) != expected_digest:
        raise ValueError("Saved model/optimizer state differs")
    model.load_state_dict(saved["model"], strict=True)
    if optimizer is not None:
        if saved["optimizer"] is None:
            raise ValueError("Missing required Adam state")
        optimizer.load_state_dict(saved["optimizer"])
    loaded = {"model": model.state_dict(), "metadata": saved["metadata"],
              "optimizer": optimizer.state_dict() if optimizer is not None else None}
    # Inference loads may deliberately omit saved Adam, but model bytes must match.
    if state_digest(loaded["model"]) != state_digest(saved["model"]):
        raise ValueError("Actual loaded model differs")
    if optimizer is not None and state_digest(loaded) != expected_digest:
        raise ValueError("Actual loaded Adam state differs")
    return saved["metadata"]
