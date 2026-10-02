    1  """Trainable item/account pooling and symmetric pure-text pair classification."""
    2  from __future__ import annotations
    3  
    4  import contextlib
    5  import hashlib
    6  from pathlib import Path
    7  from typing import Any, Callable
    8  
    9  import numpy as np
   10  
   11  import step28_continual_population_data as data
   12  
   13  
   14  def build_model(encoder: Any, dimension: int, hidden: int = 128) -> Any:
   15      import torch
   16  
   17      class PairModel(torch.nn.Module):
   18          def __init__(self) -> None:
   19              super().__init__()
   20              self.encoder = encoder
   21              self.head = torch.nn.Sequential(torch.nn.Linear(dimension * 2, hidden),
   22                                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
   23  
   24          def pair_logits(self, accounts: Any) -> Any:
   25              left, right = torch.triu_indices(len(accounts), len(accounts), 1, device=accounts.device)
   26              u, v = accounts[left], accounts[right]
   27              return self.head(torch.cat(((u - v).abs(), u * v), dim=1)).flatten()
   28  
   29      return PairModel()
   30  
   31  
   32  def make_optimizer(model: Any, config: dict) -> Any:
   33      import torch
   34      p = config["optimizer"]
   35      return torch.optim.AdamW(
   36          [{"params": model.encoder.parameters(), "lr": p["encoder_lr"],
   37            "weight_decay": p["encoder_weight_decay"]},
   38           {"params": model.head.parameters(), "lr": p["head_lr"],
   39            "weight_decay": p["head_weight_decay"]}],
   40          betas=tuple(p["betas"]), eps=p["eps"], foreach=p["foreach"])
   41  
   42  
   43  def model_files(config: dict) -> dict:
   44      base = data.ROOT / config["model"]["path"]
   45      rows = [{"path": path.relative_to(base).as_posix(), "size_bytes": path.stat().st_size,
   46               "sha256": data.sha256(path)}
   47              for path in sorted(base.rglob("*"), key=lambda p: p.relative_to(base).as_posix())
   48              if path.is_file() and ".cache" not in path.parts and "__pycache__" not in path.parts
   49              and path.suffix != ".pyc"]
   50      # Preserve step7's canonical_hash: size_bytes keys and no trailing newline.
   51      content = hashlib.sha256(data.json_bytes(rows).rstrip(b"\n")).hexdigest()
   52      return {"files": rows, "file_count": len(rows),
   53              "total_size_bytes": sum(row["size_bytes"] for row in rows), "content_sha256": content}
   54  
   55  
   56  def load_model(config: dict, device: str = "cuda:0") -> Any:
   57      import torch
   58      from sentence_transformers import SentenceTransformer
   59  
   60      torch.manual_seed(config["initialization_seed"])
   61      encoder = SentenceTransformer(str(data.ROOT / config["model"]["path"]),
   62                                    device=device, local_files_only=True)
   63      encoder.default_prompt_name = None
   64      encoder.max_seq_length = config["model"]["token_budget"]
   65      if encoder.get_sentence_embedding_dimension() != config["model"]["embedding_dim"]:
   66          raise ValueError("LaBSE embedding dimension differs")
   67      if config["model"]["activation_checkpointing"]:
   68          encoder[0].auto_model.gradient_checkpointing_enable(
   69              gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
   70      # Head initialization must not depend on encoder loader's incidental RNG use.
   71      torch.manual_seed(config["initialization_seed"])
   72      return build_model(encoder, config["model"]["embedding_dim"], config["model"]["pair_head"][1]).to(device)
   73  
   74  
   75  def pool_records(vectors: Any, counts: list[int]) -> Any:
   76      import torch
   77      if sum(counts) != len(vectors) or any(n <= 0 for n in counts):
   78          raise ValueError("Record/account alignment differs")
   79      normalized = torch.nn.functional.normalize(vectors.float(), dim=1)
   80      means = torch.stack([chunk.mean(dim=0) for chunk in normalized.split(counts)])
   81      return torch.nn.functional.normalize(means, dim=1)
   82  
   83  
   84  def account_vectors(model: Any, group: data.Group, config: dict,
   85                      check_budget: Callable[[], None] = lambda: None) -> Any:
   86      import torch
   87      device = next(model.parameters()).device
   88      texts = [title + "\n" + description for rows in group.items for _, title, description in rows]
   89      counts = list(map(len, group.items))
   90      size = config["model"]["microbatch"]
   91      blocks = []
   92      for start in range(0, len(texts), size):
   93          check_budget()
   94          batch = texts[start:start + size]
   95          # Do not use SentenceTransformer.tokenize: it silently truncates long input.
   96          features = model.encoder.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
   97          if features["input_ids"].shape[1] > config["model"]["token_budget"]:
   98              raise ValueError("An item exceeds the frozen token budget")
   99          features = {key: value.to(device) for key, value in features.items()}
  100          context = (torch.autocast("cuda", dtype=torch.bfloat16)
  101                     if device.type == "cuda" and config["model"]["encoder_bf16"] else contextlib.nullcontext())
  102          with context:
  103              vectors = model.encoder(features)["sentence_embedding"]
  104          blocks.append(vectors.float())
  105      return pool_records(torch.cat(blocks), counts)
  106  
  107  
  108  def logits(model: Any, group: data.Group, config: dict,
  109             check_budget: Callable[[], None] = lambda: None) -> Any:
  110      return model.pair_logits(account_vectors(model, group, config, check_budget))
  111  
  112  
  113  def backward_group(model: Any, group: data.Group, config: dict, seed: int,
  114                     check_budget: Callable[[], None] = lambda: None) -> float:
  115      import torch
  116      if group.labels is None:
  117          raise ValueError("Training requires authorized current/replay supervision")
  118      torch.manual_seed(seed)
  119      predicted = logits(model, group, config, check_budget)
  120      truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
  121      if predicted.shape != truth.shape or not torch.isfinite(predicted).all():
  122          raise ValueError("Nonfinite or misaligned train logits")
  123      loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, truth, reduction="mean")
  124      loss.backward()
  125      return float(loss.detach())
  126  
  127  
  128  def update(model: Any, optimizer: Any, current: data.Group, replay: data.Group | None,
  129             config: dict, current_seed: int, replay_seed: int,
  130             check_budget: Callable[[], None] = lambda: None) -> dict:
  131      import torch
  132      model.train()
  133      optimizer.zero_grad(set_to_none=True)
  134      loss = backward_group(model, current, config, current_seed, check_budget)
  135      historical = backward_group(model, replay, config, replay_seed, check_budget) if replay else None
  136      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"],
  137                                           error_if_nonfinite=True)
  138      check_budget()
  139      optimizer.step()
  140      return {"current_bce": loss, "replay_bce": historical, "gradient_norm_before_clip": float(norm)}
  141  
  142  
  143  def score(model: Any, groups: list[data.Group], config: dict,
  144            check_budget: Callable[[], None] = lambda: None) -> np.ndarray:
  145      import torch
  146      model.eval()
  147      result = []
  148      with torch.inference_mode():
  149          for group in groups:
  150              result.append(logits(model, group, config, check_budget).float().cpu().numpy())
  151      values = np.stack(result).astype(np.float32, copy=False)
  152      if values.shape != (len(groups), 378) or not np.isfinite(values).all():
  153          raise ValueError("Incomplete/nonfinite blind scores")
  154      return values
  155  
  156  
  157  def state_digest(value: Any) -> str:
  158      """All tensors, scalar state and optimizer parameter groups, without aliases."""
  159      import torch
  160      digest = hashlib.sha256()
  161  
  162      def visit(item: Any) -> None:
  163          if isinstance(item, torch.Tensor):
  164              tensor = item.detach().contiguous().cpu()
  165              digest.update(data.json_bytes([str(tensor.dtype), list(tensor.shape)]))
  166              digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
  167          elif isinstance(item, dict):
  168              for key in sorted(item, key=str):
  169                  digest.update(data.json_bytes([str(type(key)), str(key)]))
  170                  visit(item[key])
  171          elif isinstance(item, (tuple, list)):
  172              digest.update(data.json_bytes(["sequence", len(item)]))
  173              for child in item:
  174                  visit(child)
  175          else:
  176              digest.update(data.json_bytes(item))
  177  
  178      visit(value)
  179      return digest.hexdigest()
  180  
  181  
  182  def save_state(path: Path, model: Any, optimizer: Any | None, metadata: dict) -> dict:
  183      import torch
  184      if path.exists():
  185          raise FileExistsError(path)
  186      payload = {"model": model.state_dict(), "metadata": metadata,
  187                 "optimizer": optimizer.state_dict() if optimizer is not None else None}
  188      digest = state_digest(payload)
  189      torch.save(payload, path)
  190      return {"state_sha256": digest, **data.record(path, path.parent)}
  191  
  192  
  193  def restore_state(path: Path, model: Any, optimizer: Any | None, expected_digest: str) -> dict:
  194      import torch
  195      saved = torch.load(path, map_location="cpu", weights_only=True)
  196      if state_digest(saved) != expected_digest:
  197          raise ValueError("Saved model/optimizer state differs")
  198      model.load_state_dict(saved["model"], strict=True)
  199      if optimizer is not None:
  200          if saved["optimizer"] is None:
  201              raise ValueError("Missing required Adam state")
  202          optimizer.load_state_dict(saved["optimizer"])
  203      loaded = {"model": model.state_dict(), "metadata": saved["metadata"],
  204                "optimizer": optimizer.state_dict() if optimizer is not None else None}
  205      # Inference loads may deliberately omit saved Adam, but model bytes must match.
  206      if state_digest(loaded["model"]) != state_digest(saved["model"]):
  207          raise ValueError("Actual loaded model differs")
  208      if optimizer is not None and state_digest(loaded) != expected_digest:
  209          raise ValueError("Actual loaded Adam state differs")
  210      return saved["metadata"]
