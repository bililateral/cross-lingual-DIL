"""Chinese alias identification: account representation by ranking-objective factorial.

Preparation is authorized; a new Linux stage must be reported and resumed.
No owners/test access, automatic retry, validation threshold tuning, or winner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import shutil
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics
import step28_continual_population_run as persistence
import step28_continual_expression_run as public

POLICY = data.ROOT / "schema/step28_chinese_base_policy.json"
POLICY_SHA256 = "add77dd222c8308eb013bd8226b1f6f51dc3a2403a533010322d27848c597f2a"
ARMS = ("mean_bce", "mean_rank", "split_bce", "split_rank")
ROLES = ("fit", "calibration", "development")
ROLE_SIZES = {"fit": 144, "calibration": 36, "development": 60}
COMPLETE = "COMPLETE_CHINESE_BASE_3456_CALIBRATED_VALID_BLIND"
CPU_CONTRACT_COUNT = 12


def contract() -> dict:
    if data.sha256(POLICY) != POLICY_SHA256:
        raise ValueError("Frozen Chinese-model policy differs")
    return data.read_json(POLICY)


def sources() -> list[dict]:
    paths = [Path(__file__), POLICY, Path(core.__file__), Path(data.__file__),
             Path(metrics.__file__), Path(persistence.__file__), Path(public.__file__),
             data.ROOT / "tests/test_step28_chinese_base_contracts.py",
             data.ROOT / "scripts/step28_chinese_base_audit.py",
             data.ROOT / "scripts/run_step28_chinese_base_linux_20260924.sh"]
    return [data.record(p, data.ROOT) for p in paths]


def partition(groups: dict, metadata: list[dict], c: dict) -> tuple[dict, dict]:
    """Only public group IDs/domain membership determine fit/calibration roles."""
    by_id = {r["group_uid"]: r for r in metadata}
    train = groups["train"]
    if len(train) != 180 or len({g.uid for g in train}) != 180:
        raise ValueError("Expected the original 180 complete training groups")
    fit_ids: set[str] = set()
    for domain in "ABC":
        ids = [g.uid for g in train if by_id[g.uid]["domain"] == domain]
        if len(ids) != 60 or any(by_id[uid]["split"] != "train" for uid in ids):
            raise ValueError("Training domain membership differs")
        ids.sort(key=lambda uid: hashlib.sha256(data.json_bytes(
            [c["partition"]["seed"], domain, uid])).hexdigest())
        fit_ids.update(ids[:48])
    result = {"fit": [g for g in train if g.uid in fit_ids],
              "calibration": [g for g in train if g.uid not in fit_ids],
              "development": groups["development"]}
    if any(len(result[r]) != ROLE_SIZES[r] for r in ROLES):
        raise ValueError("Fit/calibration/valid counts differ")
    all_ids = [g.uid for r in ROLES for g in result[r]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Groups cross fit/calibration/valid roles")
    rows = {r: [{"group_uid": g.uid, "domain": by_id[g.uid]["domain"]}
                for g in result[r]] for r in ROLES}
    for role, count in (("fit", 48), ("calibration", 12), ("development", 20)):
        if any(sum(x["domain"] == d for x in rows[role]) != count for d in "ABC"):
            raise ValueError("Per-domain role counts differ")
    return result, rows


def schedule(groups: list[data.Group], c: dict) -> tuple[list, int]:
    if len(groups) != 144 or len({g.uid for g in groups}) != 144:
        raise ValueError("Only 144 fitting groups can enter the optimizer")
    seed = data.seed_for(c["schedule_seed"], "BASE_JOINT", "current")
    return data.schedule(groups, 6, seed), seed


def model_config(c: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown model")
    return {"model": {**c["models"][arm], **c["input"]},
            "optimizer": c["optimizer"], "initialization_seed": c["initialization_seed"]}


def pooling_modes(pool: Any) -> list[str]:
    """Read actual native pooling across the installed and legacy module APIs."""
    mode = getattr(pool, "pooling_mode", None)
    if isinstance(mode, str):
        return [mode]
    if isinstance(mode, (tuple, list)) and all(isinstance(value, str) for value in mode):
        return list(mode)
    if mode is not None:
        raise ValueError("Unrecognized native pooling representation")
    names = {"pooling_mode_cls_token": "cls", "pooling_mode_mean_tokens": "mean"}
    return [names.get(key, key) for key, value in vars(pool).items()
            if key.startswith("pooling_mode_") and value is True]


def load_model(c: dict, arm: str, device: str = "cuda:0") -> Any:
    import torch
    from sentence_transformers import SentenceTransformer
    m = c["models"][arm]
    torch.manual_seed(c["initialization_seed"])
    encoder = SentenceTransformer(str(data.ROOT / m["path"]), device=device,
                                  local_files_only=True, trust_remote_code=False)
    encoder.default_prompt_name = None
    encoder.max_seq_length = c["input"]["token_budget"]
    pools = [v for v in encoder if type(v).__name__ == "Pooling"]
    has_dense = any(type(v).__name__ == "Dense" for v in encoder)
    if len(pools) != 1 or has_dense != m["dense_module"]:
        raise ValueError("Native encoder modules differ from the registered model")
    pool = pools[0]
    if pooling_modes(pool) != [m["pooling"]] or encoder.get_sentence_embedding_dimension() != m["embedding_dim"]:
        raise ValueError("Native pooling or output dimension differs")
    if c["input"]["activation_checkpointing"]:
        encoder[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    torch.manual_seed(c["initialization_seed"])
    return core.build_model(encoder, c["interventions"][arm]["account_dim"],
                            c["input"]["head_hidden"]).to(device)


def record_texts(group: data.Group, representation: str) -> list[str]:
    records = [row for items in group.items for row in items]
    if representation == "combined_mean":
        return [title + "\n" + description for _, title, description in records]
    if representation == "separate_moments":
        return [title for _, title, _ in records] + [description for _, _, description in records]
    raise ValueError("Unknown account representation")


def pool_channels(vectors: Any, counts: list[int], representation: str,
                  epsilon: float = 1e-8) -> Any:
    """Population moments, all items, with no ID/count/domain feature."""
    import torch
    if representation == "combined_mean":
        return core.pool_records(vectors, counts)
    if (representation != "separate_moments" or not counts or any(n <= 0 for n in counts)
            or len(vectors) != 2 * sum(counts) or epsilon != 1e-8):
        raise ValueError("Channel/account alignment or moment definition differs")
    normalized = torch.nn.functional.normalize(vectors.float(), dim=1)
    title, description = normalized.split(sum(counts))
    accounts = []
    for left, right in zip(title.split(counts), description.split(counts), strict=True):
        parts = []
        for channel in (left, right):
            mean = channel.mean(0)
            std = ((channel - mean).square().mean(0) + epsilon).sqrt()
            parts.extend((mean, std))
        accounts.append(torch.cat(parts))
    return torch.nn.functional.normalize(torch.stack(accounts), dim=1)


def objectives(predicted: Any, truth: Any, rank_weight: float) -> dict:
    """Each query averages its positive logits, never positive or negative pairs twice in BCE."""
    import torch
    if (predicted.shape != (378,) or truth.shape != (378,)
            or not torch.isfinite(predicted).all() or not torch.isin(truth, truth.new_tensor([0, 1])).all()
            or int(truth.sum()) != 20 or rank_weight not in (0, 1)):
        raise ValueError("Expected complete finite 28-account supervision and fixed loss weight")
    left, right = torch.triu_indices(28, 28, 1, device=predicted.device)
    positive = torch.zeros((28, 28), dtype=torch.bool, device=predicted.device)
    positive[left, right] = truth.bool()
    positive[right, left] = truth.bool()
    degrees = positive.sum(1)
    if (int((degrees == 1).sum()) != 16 or int((degrees == 2).sum()) != 12):
        raise ValueError("Expected eight two-alias and four three-alias positive components")
    # Same-controller labels must be transitive; a degree-two cycle is not a controller.
    paths = positive.float() @ positive.float()
    diagonal = torch.eye(28, dtype=torch.bool, device=predicted.device)
    if ((paths > 0) & ~positive & ~diagonal).any():
        raise ValueError("Positive relations are not disjoint controller cliques")
    scores = predicted.new_zeros((28, 28))
    scores[left, right] = predicted
    scores[right, left] = predicted
    log_denominator = torch.logsumexp(scores.masked_fill(diagonal, -torch.inf), dim=1)
    positive_mean = scores.masked_fill(~positive, 0).sum(1) / degrees
    rank = (log_denominator - positive_mean).mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(predicted, truth, reduction="mean")
    return {"bce": bce, "rank": rank, "total": bce + rank_weight * rank}


def account_vectors(model: Any, group: data.Group, c: dict, arm: str,
                    check: Any = lambda: None) -> Any:
    import contextlib
    import torch
    device = next(model.parameters()).device
    representation = c["interventions"][arm]["representation"]
    texts = record_texts(group, representation)
    batch_size = c["input"]["microbatch"]
    vectors = []
    for i in range(0, len(texts), batch_size):
        check()
        batch = model.encoder.tokenizer(texts[i:i + batch_size], padding=True,
                                        truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[1] > c["input"]["token_budget"]:
            raise ValueError("Input exceeds token limit; no truncation is allowed")
        batch = {key: value.to(device) for key, value in batch.items()}
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if device.type == "cuda" and c["input"]["encoder_bf16"]
                   else contextlib.nullcontext())
        with context:
            vectors.append(model.encoder(batch)["sentence_embedding"].float())
    return pool_channels(torch.cat(vectors), [len(rows) for rows in group.items],
                         representation, c["representation"]["variance_epsilon"])


def logits(model: Any, group: data.Group, c: dict, arm: str, check: Any = lambda: None) -> Any:
    return model.pair_logits(account_vectors(model, group, c, arm, check))


def score(model: Any, groups: list, c: dict, arm: str, check: Any = lambda: None) -> np.ndarray:
    import torch
    model.eval()
    with torch.inference_mode():
        values = np.stack([logits(model, g, c, arm, check).float().cpu().numpy() for g in groups])
    if values.shape != (len(groups), 378) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Incomplete/nonfinite blind scores")
    return values


def update(model: Any, optimizer: Any, group: data.Group, c: dict, arm: str,
           seed: int, *, observe: bool = False, check: Any = lambda: None) -> dict:
    import torch
    if group.labels is None:
        raise ValueError("A fitting group must carry authorized supervision")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {name: core.state_digest(module.state_dict())
              for name, module in (("encoder", model.encoder), ("head", model.head))} if observe else {}
    torch.manual_seed(seed)
    predicted = logits(model, group, c, arm, check)
    truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
    if predicted.shape != truth.shape or not torch.isfinite(predicted).all():
        raise ValueError("Nonfinite or misaligned logits")
    terms = objectives(predicted, truth, c["interventions"][arm]["rank_weight"])
    loss = terms["total"]
    loss.backward()
    evidence = {}
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            norms = [float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None]
            if not norms or not np.isfinite(norms).all() or max(norms) <= 0:
                raise ValueError("Encoder/head has no finite nonzero gradient")
            evidence[name] = {"finite_nonzero_gradient": True}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["optimizer"]["clip_norm"], error_if_nonfinite=True)
    check()
    optimizer.step()
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            if core.state_digest(module.state_dict()) == before[name]:
                raise ValueError("Encoder/head parameters did not change")
            evidence[name]["parameters_changed"] = True
    return {**{name: float(value.detach()) for name, value in terms.items()},
            "gradient_norm_before_clip": float(norm), "modules": evidence}


def binary_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (y.ndim != 2 or y.shape != s.shape or not np.isin(y, (0, 1)).all()
            or not np.isfinite(s).all() or not np.isfinite(threshold)):
        raise ValueError("Aligned binary labels and finite scores/threshold required")
    positive, predicted = y == 1, s >= float(threshold)
    return np.stack([(positive & predicted).sum(1), (~positive & predicted).sum(1),
                     (positive & ~predicted).sum(1), (~positive & ~predicted).sum(1)], axis=1)


def calibrate(labels: np.ndarray, scores: np.ndarray, domains: list[str], maximum_fpr: float = .001) -> dict:
    """One global threshold; calibrated solely against each train-calibration domain."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (maximum_fpr != .001 or y.shape != s.shape or y.ndim != 2 or len(domains) != len(y)
            or set(domains) != set("ABC") or not np.isin(y, (0, 1)).all() or not np.isfinite(s).all()):
        raise ValueError("Invalid calibration inputs or unapproved FPR target")
    bounds = {}
    for d in "ABC":
        rows = np.asarray(domains) == d
        negative = np.sort(s[rows][y[rows] == 0])[::-1]
        if not len(negative) or not np.any(y[rows] == 1):
            raise ValueError("Calibration domain needs both classes")
        allowed = len(negative) // 1000
        boundary = float(negative[allowed])
        threshold = float(np.nextafter(np.float64(boundary), np.float64(np.inf)))
        if not np.isfinite(threshold):
            raise ValueError("Calibration threshold overflow")
        bounds[d] = {"negative_pairs": len(negative), "allowed_false_positives": allowed,
                     "next_negative_logit": boundary, "threshold": threshold}
    threshold = max(row["threshold"] for row in bounds.values())
    counts = binary_counts(y, s, threshold)
    totals = {d: counts[np.asarray(domains) == d].sum(0).tolist() for d in "ABC"}
    if any(totals[d][1] > bounds[d]["allowed_false_positives"] for d in "ABC"):
        raise ValueError("Calibration did not satisfy its false-positive cap")
    return {"threshold": threshold, "prediction": "float64(logit)>=float64(threshold)",
            "source_role": "train_calibration_only", "bounds": bounds,
            "counts_by_domain": totals, "counts_by_group": counts.tolist()}


def checkpoint_point(model: Any, optimizer: Any, c: dict, arm: str, epoch: int,
                     groups: dict, out: Path, budget: persistence.Budget) -> dict:
    started = time.monotonic()
    name = f"epoch{epoch}"
    before = {role: score(model, groups[role], c, arm, budget.check) for role in ROLES}
    checkpoint = out / "work" / f"{name}.pt"
    metadata = {"arm": arm, "epoch": epoch, "config": c}
    budget.check(persistence.checkpoint_reserve(model, optimizer))
    saved = core.save_state(checkpoint, model, optimizer, metadata)
    if core.restore_state(checkpoint, model, optimizer, saved["state_sha256"]) != metadata:
        raise ValueError("Restored model/optimizer metadata differs")
    point = {"epoch": epoch, "arm": arm, "full_model_and_adam_reloaded": True,
             "checkpoint": saved, "model_state_sha256": core.state_digest(model.state_dict()), "scores": {}}
    for role in ROLES:
        after = score(model, groups[role], c, arm, budget.check)
        if not np.array_equal(before[role], after):
            raise ValueError("Complete scores differ after actual model/Adam restore")
        path = out / "scores" / f"{name}_{role}.npy"
        np.save(path, before[role], allow_pickle=False)
        point["scores"][role] = data.record(path, out)
    target = out / "models" / f"{name}.pt"
    budget.check(persistence.checkpoint_reserve(model, None))
    retained = core.save_state(target, model, None, metadata)
    core.restore_state(target, model, None, retained["state_sha256"])
    if core.state_digest(model.state_dict()) != point["model_state_sha256"]:
        raise ValueError("Retained inference model differs")
    point["model"] = {**retained, "path": target.relative_to(out).as_posix(), "actual_reload_verified": True}
    # Force a disk observation while inference and full optimizer files coexist.
    budget.check(1)
    persistence.remove_work_file(checkpoint, out)
    point["point_seconds"] = time.monotonic() - started
    return point


def train_arm(out: Path, arm: str, c: dict, groups: dict, partition_rows: dict,
              budget: persistence.Budget, preflight: dict) -> dict:
    import torch
    out.mkdir()
    for directory in ("work", "models", "scores"):
        (out / directory).mkdir()
    model = load_model(c, arm)
    if core.state_digest(model.state_dict()) != preflight["initial_model_state_sha256"]:
        raise ValueError("Preflight and formal initialization differ")
    optimizer = core.make_optimizer(model, c)
    rows, seed = schedule(groups["fit"], c)
    r = {"arm": arm, "points": {}, "training": [], "label_parses": 0,
         "preflight": preflight, "fit_group_ids": [g.uid for g in groups["fit"]],
         "calibration_group_ids": [g.uid for g in groups["calibration"]]}
    total_training = 0.
    for start, stop, epoch in ((0, 432, 3), (432, 864, 6)):
        started = time.monotonic()
        losses, evidence = [], {}
        for i in range(start, stop):
            log = update(model, optimizer, rows[i], c, arm, data.seed_for(seed, i, "dropout"),
                         observe=(i == start), check=budget.check)
            losses.append([log[name] for name in ("bce", "rank", "total")])
            if i == start:
                evidence = log["modules"]
            if i in (start, stop - 1) and (not optimizer.state or any(
                    float(v["step"]) != i + 1 for v in optimizer.state.values())):
                raise ValueError("Adam continuity or update count differs")
            if (i + 1) % 24 == 0:
                print(data.json_bytes({"event": "updates", "arm": arm, "completed": i + 1,
                                       **budget.state()}).decode(), flush=True)
        elapsed = time.monotonic() - started
        total_training += elapsed
        r["training"].append({"start": start, "stop": stop, "updates": stop - start,
                              "first_update_modules": evidence, "seconds": elapsed,
                              "losses_by_epoch": dict(zip(("bce", "rank", "total"),
                                  np.asarray(losses).reshape(3, 144, 3).mean(1).T.tolist()))})
        point = checkpoint_point(model, optimizer, c, arm, epoch, groups, out, budget)
        r["points"][str(epoch)] = point
        point["train_metrics"] = {}
        for role in ("fit", "calibration"):
            scores = np.load(out / point["scores"][role]["path"], allow_pickle=False)
            truth = np.asarray([g.labels for g in groups[role]], dtype=np.uint8)
            matrix, counts = metrics.group_metrics(truth, scores)
            path = out / "scores" / f"epoch{epoch}_{role}_metrics.npy"
            np.save(path, matrix, allow_pickle=False)
            point["train_metrics"][role] = {"file": data.record(path, out), "counts": counts,
                "stable_bce_by_group": np.mean(np.logaddexp(0., scores.astype(np.float64)) - truth * scores, axis=1).tolist()}
        if epoch == 6:
            truth = np.asarray([g.labels for g in groups["calibration"]], dtype=np.uint8)
            scores = np.load(out / point["scores"]["calibration"]["path"], allow_pickle=False)
            calibrated = calibrate(truth, scores, [x["domain"] for x in partition_rows["calibration"]])
            calibrated.update(model_state_sha256=point["model_state_sha256"], epoch=6,
                              score_sha256=point["scores"]["calibration"]["sha256"])
            data.write_json(out / "calibration.json", calibrated)
            r["calibration"] = data.record(out / "calibration.json", out)
        data.write_json(out / "progress.json", r)
    r.update(updates=864, formal_training_seconds=total_training,
             group_schedule_sha256=hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest(),
             dropout_stream=seed)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    data.write_json(out / "manifest.json", r)
    return r


def preflight_model(c: dict, arm: str, groups: dict, budget: persistence.Budget) -> dict:
    """Public text/tokenization and actual forward only; zero supervised updates."""
    import torch
    cfg = model_config(c, arm)
    files = core.model_files(cfg)
    if any(files[key] != c["models"][arm][key]
           for key in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Archived pretrained model bytes differ")
    model = load_model(c, arm)
    maximum, records = 0, 0
    for split in ("train", "development"):
        for group in groups[split]:
            budget.check()
            texts = record_texts(group, c["interventions"][arm]["representation"])
            batch = model.encoder.tokenizer(texts, padding=False, truncation=False)
            lengths = [len(row) for row in batch["input_ids"]]
            maximum, records = max(maximum, max(lengths)), records + len(lengths)
            if maximum > c["input"]["token_budget"]:
                raise ValueError("A complete public record exceeds token budget")
    blind = score(model, groups["train"][:1], c, arm, budget.check)
    result = {"pretrained": files, "max_tokens": maximum, "records_checked": records,
              "initial_model_state_sha256": core.state_digest(model.state_dict()),
              "actual_forward_shape": list(blind.shape), "supervision_reads": 0, "updates": 0}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def require_cpu_contracts() -> dict:
    sys.path.insert(0, str(data.ROOT / "tests"))
    suite = unittest.defaultTestLoader.loadTestsFromName("test_step28_chinese_base_contracts")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != CPU_CONTRACT_COUNT:
        raise RuntimeError("All Chinese CPU contracts must pass without skips")
    return {"passed": result.testsRun, "skipped": 0}


def run(out: Path) -> dict:
    import torch
    c = contract()
    runtime = c["runtime"]
    if (platform.system() != "Linux" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Requires separately resumed Linux stage and one eligible visible GPU")
    available = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
                     if row.startswith("MemAvailable:"))
    if (torch.cuda.mem_get_info()[0] < runtime["minimum_free_gpu_bytes"]
            or available < runtime["minimum_free_host_bytes"]
            or shutil.disk_usage(data.ROOT).free < runtime["maximum_output_bytes"]):
        raise RuntimeError("Insufficient shared resources; wait")
    if out.exists() or not out.is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a new project reports run directory")
    out.mkdir(parents=True)
    budget = persistence.Budget(out.parent, c)
    snapshot = sources()
    r = {"status": "RUNNING", "config": c, "source_files": snapshot, "arms": {},
         "label_parses": {"train": 0, "development": 0, "heldout": 0, "owners": 0},
         "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(out / "startup.json", r)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        r["torch_contracts"] = require_cpu_contracts()
        groups, metadata, checked = public.public_inputs(c)
        _, partition_rows = partition(groups, metadata, c)
        data.write_json(out / "partition.json", partition_rows)
        r["partition"] = data.record(out / "partition.json", out)
        r["inputs"] = checked
        preflights = {arm: preflight_model(c, arm, groups, budget) for arm in ARMS}
        for left, right in (("mean_bce", "mean_rank"), ("split_bce", "split_rank")):
            if preflights[left]["initial_model_state_sha256"] != preflights[right]["initial_model_state_sha256"]:
                raise ValueError("Objective comparison must share the exact initial model state")
        r["preflights"] = preflights
        data.write_json(out / "preflight.json", r)
        data.write_json(out / "access.json", {"train_parse_attempts": 1, "development": 0, "heldout": 0, "owners": 0})
        r["label_parses"]["train"] = 1
        groups["train"] = public.attach_labels(groups["train"], c, "train")
        selected, aligned = partition(groups, metadata, c)
        if aligned != partition_rows:
            raise ValueError("Supervision changed the public partition")
        for arm in ARMS:
            result = train_arm(out / arm, arm, c, selected, partition_rows, budget, preflights[arm])
            r["arms"][arm] = {"manifest": data.record(out / arm / "manifest.json", out),
                               "updates": result["updates"], "formal_training_seconds": result["formal_training_seconds"]}
            data.write_json(out / "progress.json", r)
        if sources() != snapshot:
            raise ValueError("Source files changed during execution")
        r.update(status=COMPLETE, physical_updates=sum(x["updates"] for x in r["arms"].values()),
                 formal_training_seconds=sum(x["formal_training_seconds"] for x in r["arms"].values()),
                 budget=budget.state())
        if r["physical_updates"] != 3456:
            raise ValueError("Unexpected formal update count")
        data.write_json(out / "manifest.json", r)
        verify_models(out)
        data.write_json(out / "completion.json", {"status": COMPLETE, "budget": budget.state(),
                        "manifest_sha256": data.sha256(out / "manifest.json")})
        return r
    except Exception as error:
        data.write_json(out / "failure.json", {"status": "FAILED_NO_RETRY_OR_PARTIAL_EVALUATION",
            "error_type": type(error).__name__, "error": str(error), "label_parses": r["label_parses"]})
        raise


def verify_models(out: Path) -> dict:
    r = data.read_json(out / "manifest.json")
    path = out / "model_verification.json"
    if r["status"] != COMPLETE or path.exists() or (out / "failure.json").exists():
        raise ValueError("Only a complete run and new model receipt are permitted")
    records = []
    for arm in ARMS:
        a = data.read_json(data.verify(out / r["arms"][arm]["manifest"]["path"], r["arms"][arm]["manifest"]))
        for epoch in ("3", "6"):
            model = a["points"][epoch]["model"]
            actual = data.verify(out / arm / model["path"], model)
            records.append(data.record(actual, out))
    result = {"status": "EIGHT_MODELS_FRESH_SIZE_SHA_VERIFIED", "files": records,
              "manifest_sha256": data.sha256(out / "manifest.json")}
    data.write_json(path, result)
    return result


def load_array(path: Path, record: dict, shape: tuple, dtype: Any) -> np.ndarray:
    a = np.load(data.verify(path, record), allow_pickle=False)
    if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
        raise ValueError("Saved score/metric shape or dtype differs")
    return a


def validated_run(out: Path, c: dict, groups: dict, metadata: list, checked: dict) -> tuple[dict, dict, dict]:
    if (out / "failure.json").exists() or (out.parent / "exit_status.txt").read_text().strip() != "0":
        raise ValueError("Failed or incomplete outer job")
    for name in ("started.txt", "finished.txt", "resource_usage.log"):
        if not (out.parent / name).is_file():
            raise ValueError("Missing outer execution evidence")
    started = datetime.fromisoformat((out.parent / "started.txt").read_text().strip())
    finished = datetime.fromisoformat((out.parent / "finished.txt").read_text().strip())
    if (started.utcoffset() is None or finished.utcoffset() is None
            or not 0 <= (finished - started).total_seconds() <= c["runtime"]["maximum_gpu_stage_seconds"] + 60):
        raise ValueError("Invalid outer job timing")
    r = data.read_json(out / "manifest.json")
    if (r["status"] != COMPLETE or r["config"] != c or r["source_files"] != sources()
            or set(r["arms"]) != set(ARMS) or r["physical_updates"] != 3456
            or r["inputs"] != checked or r["torch_contracts"] != {"passed": CPU_CONTRACT_COUNT, "skipped": 0}
            or r["label_parses"] != {"train": 1, "development": 0, "heldout": 0, "owners": 0}):
        raise ValueError("Incomplete/stale base-model stage")
    completion = data.read_json(out / "completion.json")
    if completion["status"] != COMPLETE or completion["manifest_sha256"] != data.sha256(out / "manifest.json"):
        raise ValueError("Completion is not bound to this manifest")
    for budget in (r["budget"], completion["budget"]):
        if (not 0 <= budget["elapsed_seconds"] < c["runtime"]["maximum_gpu_stage_seconds"]
                or not 0 <= budget["peak_observed_bytes"] <= c["runtime"]["maximum_output_bytes"]):
            raise ValueError("Budget exceeded")
    selected, part = partition(groups, metadata, c)
    saved_part = data.read_json(data.verify(out / r["partition"]["path"], r["partition"]))
    if saved_part != part:
        raise ValueError("Saved fitting/calibration group mapping differs")
    rows, seed = schedule(selected["fit"], c)
    expected_schedule = hashlib.sha256(data.json_bytes([g.uid for g in rows])).hexdigest()
    result, expected_models = {}, []
    for arm in ARMS:
        ar = r["arms"][arm]
        if ar["manifest"]["path"] != f"{arm}/manifest.json" or ar["updates"] != 864:
            raise ValueError("Arm identity or update count differs")
        a = data.read_json(data.verify(out / ar["manifest"]["path"], ar["manifest"]))
        pre = a["preflight"]
        if (a["arm"] != arm or a["label_parses"] != 0 or a["updates"] != 864
                or a["group_schedule_sha256"] != expected_schedule or a["dropout_stream"] != seed
                or a["fit_group_ids"] != [g.uid for g in selected["fit"]]
                or a["calibration_group_ids"] != [g.uid for g in selected["calibration"]]
                or set(a["points"]) != {"3", "6"} or len(a["training"]) != 2
                or pre != r["preflights"][arm] or pre["updates"] != 0 or pre["supervision_reads"] != 0
                or pre["actual_forward_shape"] != [1, 378] or not 0 < pre["max_tokens"] <= 256
                or pre["records_checked"] != 24103 * (2 if c["interventions"][arm]["representation"] == "separate_moments" else 1)):
            raise ValueError("Training supply, prefix preflight, or scope differs")
        if any(pre["pretrained"][key] != c["models"][arm][key]
               for key in ("file_count", "total_size_bytes", "content_sha256")):
            raise ValueError("Pretrained model source differs")
        for log, start, stop in zip(a["training"], (0, 432), (432, 864), strict=True):
            if (log["start"] != start or log["stop"] != stop or log["updates"] != 432
                    or set(log["first_update_modules"]) != {"encoder", "head"}
                    or any(v != {"finite_nonzero_gradient": True, "parameters_changed": True}
                           for v in log["first_update_modules"].values())):
                raise ValueError("Actual optimizer/encoder/head evidence is incomplete")
            losses = log["losses_by_epoch"]
            if (set(losses) != {"bce", "rank", "total"}
                    or any(np.asarray(v).shape != (3,) or not np.isfinite(v).all() for v in losses.values())
                    or not np.allclose(losses["total"], np.asarray(losses["bce"]) +
                                       c["interventions"][arm]["rank_weight"] * np.asarray(losses["rank"]),
                                       rtol=1e-6, atol=1e-7)):
                raise ValueError("Saved objective components differ from the frozen intervention")
        points = {}
        for epoch in ("3", "6"):
            p = a["points"][epoch]
            if (p["arm"] != arm or p["epoch"] != int(epoch)
                    or not p["full_model_and_adam_reloaded"] or set(p["scores"]) != set(ROLES)
                    or not p["model"]["actual_reload_verified"]
                    or p["model"]["path"] != f"models/epoch{epoch}.pt"):
                raise ValueError("Incomplete or mismatched model/score checkpoint")
            arrays = {}
            for role in ROLES:
                rec = p["scores"][role]
                if rec["path"] != f"scores/epoch{epoch}_{role}.npy":
                    raise ValueError("Score role or point mismatch")
                arrays[role] = load_array(out / arm / rec["path"], rec, (ROLE_SIZES[role], 378), np.float32)
            for role in ("fit", "calibration"):
                rec = p["train_metrics"][role]["file"]
                if rec["path"] != f"scores/epoch{epoch}_{role}_metrics.npy":
                    raise ValueError("Metric role or point mismatch")
                load_array(out / arm / rec["path"], rec, (ROLE_SIZES[role], 22), np.float64)
            points[epoch] = arrays
            expected_models.append({k: p["model"][k] for k in ("bytes", "sha256")}
                                   | {"path": f"{arm}/models/epoch{epoch}.pt"})
        if a["calibration"]["path"] != "calibration.json":
            raise ValueError("Calibration file role differs")
        cal = data.read_json(data.verify(out / arm / a["calibration"]["path"], a["calibration"]))
        point = a["points"]["6"]
        if (cal["epoch"] != 6 or cal["source_role"] != "train_calibration_only"
                or cal["prediction"] != "float64(logit)>=float64(threshold)"
                or cal["model_state_sha256"] != point["model_state_sha256"]
                or cal["score_sha256"] != point["scores"]["calibration"]["sha256"]
                or set(cal["bounds"]) != set("ABC")):
            raise ValueError("Threshold is not tied to this model's calibration scores")
        counts = np.asarray(cal["counts_by_group"])
        if (counts.shape != (36, 4) or counts.dtype.kind not in "iu" or (counts < 0).any()
                or not np.all(counts[:, 0] + counts[:, 2] == 20)
                or not np.all(counts[:, 1] + counts[:, 3] == 358)):
            raise ValueError("Calibration group counts differ")
        for d in "ABC":
            mask = np.asarray([row["domain"] for row in part["calibration"]]) == d
            bound = cal["bounds"][d]
            if (bound["negative_pairs"] != 4296 or bound["allowed_false_positives"] != 4
                    or counts[mask].sum(0).tolist() != cal["counts_by_domain"][d]
                    or bound["threshold"] != float(np.nextafter(np.float64(bound["next_negative_logit"]), np.inf))
                    or cal["counts_by_domain"][d][1] > 4):
                raise ValueError("Calibration boundary/count metadata differs")
        if not np.isfinite(cal["threshold"]) or cal["threshold"] != max(x["threshold"] for x in cal["bounds"].values()):
            raise ValueError("Not the single globally fixed calibration threshold")
        result[arm] = {"points": points, "threshold": cal["threshold"]}
    receipt = data.read_json(out / "model_verification.json")
    if (receipt["status"] != "EIGHT_MODELS_FRESH_SIZE_SHA_VERIFIED"
            or receipt["manifest_sha256"] != data.sha256(out / "manifest.json")
            or receipt["files"] != expected_models):
        raise ValueError("Missing or mismatched actual Linux model hash receipt")
    # Evaluation now runs on Linux alongside retained weights. A historical
    # receipt cannot establish that the actual model files still match today.
    for model in expected_models:
        data.verify(out / model["path"], model)
    return result, part, r


def rates(counts: np.ndarray) -> dict:
    tp, fp, fn, tn = np.asarray(counts, dtype=np.float64).sum(axis=0)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "fpr": float(fp / (fp + tn)), "recall": float(tp / (tp + fn)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.}


def automatic_report(counts: np.ndarray, domains: list[str], draws: np.ndarray) -> dict:
    out = {}
    domain_boot_counts = []
    for index, domain in enumerate("ABC"):
        rows = np.flatnonzero(np.asarray(domains) == domain)
        values = counts[rows]
        point = rates(values)
        boot = values[draws[:, index]].sum(axis=1).astype(np.float64)
        domain_boot_counts.append(boot)
        fpr = boot[:, 1] / (boot[:, 1] + boot[:, 3])
        recall = boot[:, 0] / (boot[:, 0] + boot[:, 2])
        precision = np.divide(boot[:, 0], boot[:, 0] + boot[:, 1],
                              out=np.zeros(len(boot)), where=boot[:, 0] + boot[:, 1] > 0)
        point["conditional_95pct_intervals"] = {k: np.quantile(v, [.025, .975]).tolist()
                                                 for k, v in (("fpr", fpr), ("recall", recall), ("precision", precision))}
        point["passes_point_gates"] = point["fp"] * 1000 <= point["fp"] + point["tn"] and point["tp"] * 2 >= point["tp"] + point["fn"]
        out[domain] = point
    pooled = rates(counts)
    boot = np.stack(domain_boot_counts).sum(axis=0)
    fpr = boot[:, 1] / (boot[:, 1] + boot[:, 3])
    recall = boot[:, 0] / (boot[:, 0] + boot[:, 2])
    precision = np.divide(boot[:, 0], boot[:, 0] + boot[:, 1],
                          out=np.zeros(len(boot)), where=boot[:, 0] + boot[:, 1] > 0)
    pooled["conditional_95pct_intervals"] = {k: np.quantile(v, [.025, .975]).tolist()
                                             for k, v in (("fpr", fpr), ("recall", recall), ("precision", precision))}
    return {"by_domain": out, "pooled": pooled,
            "all_domains_pass": all(v["passes_point_gates"] for v in out.values()),
            "interval_scope": "Group bootstrap conditional on fixed models, calibration groups and thresholds; zero observed FP may produce [0,0], not a population upper bound or deployment guarantee."}


def automatic_comparison(candidate: np.ndarray, reference: np.ndarray,
                         domains: list[str], draws: np.ndarray) -> dict:
    """Paired group resampling at each model's already fixed threshold."""
    rows = [np.flatnonzero(np.asarray(domains) == d) for d in "ABC"]
    bootstrap = [np.stack([counts[index][draws[:, j]].sum(1)
                           for j, index in enumerate(rows)], axis=1)
                 for counts in (candidate, reference)]
    result = {}
    for name, numerator, other in (("fpr", 1, 3), ("recall", 0, 2), ("precision", 0, 1)):
        scopes = {}
        for domain_index, scope in enumerate((*"ABC", "pooled")):
            values, points = [], []
            for counts, sampled in zip((candidate, reference), bootstrap, strict=True):
                totals = sampled[:, domain_index] if scope != "pooled" else sampled.sum(1)
                denom = totals[:, numerator] + totals[:, other]
                values.append(np.divide(totals[:, numerator], denom,
                                        out=np.zeros(len(totals)), where=denom > 0))
                points.append(rates(counts[rows[domain_index]] if scope != "pooled" else counts)[name])
            scopes[scope] = {"difference": points[0] - points[1],
                             "conditional_95pct_interval": np.quantile(values[0] - values[1], [.025, .975]).tolist()}
        result[name] = scopes
    return result


def historical_reference(c: dict, partition_rows: dict) -> dict:
    """Reuse frozen predictions, not archived weights or a new historical label parse."""
    records = c["historical_reference"]["files"]
    paths = {name: data.verify(data.ROOT / row["path"], row) for name, row in records.items()}
    parent = data.read_json(paths["manifest.json"])
    completion = data.read_json(paths["completion.json"])
    arm = data.read_json(paths["labse/manifest.json"])
    calibration = data.read_json(paths["labse/calibration.json"])
    score_record = records["labse/scores/epoch6_development.npy"]
    if (parent["status"] != "COMPLETE_BASE_MODELS_2592_CALIBRATED_VALID_BLIND"
            or completion["manifest_sha256"] != records["manifest.json"]["sha256"]
            or data.read_json(paths["partition.json"]) != partition_rows
            or parent["arms"]["labse"]["manifest"]["sha256"] != records["labse/manifest.json"]["sha256"]
            or arm["points"]["6"]["scores"]["development"]["sha256"] != score_record["sha256"]
            or arm["calibration"]["sha256"] != records["labse/calibration.json"]["sha256"]
            or calibration["model_state_sha256"] != arm["points"]["6"]["model_state_sha256"]):
        raise ValueError("Historical LaBSE prediction/calibration/partition binding differs")
    return {"scores": load_array(paths["labse/scores/epoch6_development.npy"], score_record,
                                  (60, 378), np.float32),
            "threshold": calibration["threshold"], "files": records}


def metric_comparison(candidate: np.ndarray, reference: np.ndarray,
                      domains: list[str], draws: np.ndarray) -> dict:
    domain_rows = [np.flatnonzero(np.asarray(domains) == d) for d in "ABC"]
    result = {}
    for i, column in enumerate(metrics.COLUMNS):
        delta = candidate[:, i] - reference[:, i]
        boot = np.stack([delta[rows][draws[:, j]].mean(1)
                         for j, rows in enumerate(domain_rows)]).mean(0)
        result[column] = {"mean": float(delta.mean()),
                          "conditional_95pct_interval": np.quantile(boot, [.025, .975]).tolist()}
    return result


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != "Linux" or destination.exists():
        raise ValueError("Requires resumed Linux evaluation and a new directory")
    c = contract()
    groups, metadata, checked = public.public_inputs(c)
    arrays, part, r = validated_run(out, c, groups, metadata, checked)
    reference = historical_reference(c, part)
    destination.mkdir(parents=True)
    data.write_json(destination / "access.json", {"status": "ALL_FOUR_MODELS_COMPLETE_BEFORE_VALID",
                    "development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    try:
        labelled = public.attach_labels(groups["development"], c, "development")
        truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
        e = c["evaluation"]
        domains = [row["domain"] for row in part["development"]]
        result = {"status": "CHINESE_BASE_METRICS_COLLECTED_BEFORE_UNCERTAINTY", "config": c,
                  "manifest_sha256": data.sha256(out / "manifest.json"), "columns": list(metrics.COLUMNS),
                  "label_parses": {"train": 0, "development": 1, "heldout": 0, "owners": 0},
                  "arms": {}, "comparisons": {}, "primary_comparison": c["primary_comparison"],
                  "historical_reference": reference["files"],
                  "formal_training_seconds": r["formal_training_seconds"]}
        matrices, auto_counts = {}, {}
        for arm in ARMS:
            (destination / arm).mkdir()
            points = {}
            for epoch in ("3", "6"):
                scores = arrays[arm]["points"][epoch]["development"]
                matrix, counts = metrics.group_metrics(truth, scores)
                path = destination / arm / f"epoch{epoch}_metrics.npy"
                np.save(path, matrix, allow_pickle=False)
                points[epoch] = {"file": data.record(path, destination), "counts_at_logit_zero": counts,
                                 "mean": dict(zip(metrics.COLUMNS, matrix.mean(0).tolist())),
                                 "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(domains) == d].mean(0).tolist())) for d in "ABC"}}
                if epoch == "6":
                    matrices[arm] = matrix
                    count = binary_counts(truth, scores, arrays[arm]["threshold"])
                    auto_counts[arm] = count
                    automatic = {"threshold": arrays[arm]["threshold"], "counts_by_group": count.tolist()}
            result["arms"][arm] = {"points": points, "automatic_classification": automatic}
        matrices["historical_labse"], historical_counts = metrics.group_metrics(truth, reference["scores"])
        auto_counts["historical_labse"] = binary_counts(truth, reference["scores"], reference["threshold"])
        path = destination / "historical_labse_metrics.npy"
        np.save(path, matrices["historical_labse"], allow_pickle=False)
        result["historical_labse"] = {"file": data.record(path, destination),
            "counts_at_logit_zero": historical_counts,
            "threshold": reference["threshold"], "counts_by_group": auto_counts["historical_labse"].tolist()}
        # Persist all nine matrices and all counts before ANY uncertainty calculation.
        data.write_json(destination / "collected.json", result)
        draws = np.random.default_rng(e["bootstrap_seed"]).integers(
            0, 20, size=(e["bootstrap_replicates"], 3, 20))
        for arm in ARMS:
            raw = result["arms"][arm]["automatic_classification"]
            result["arms"][arm]["automatic_classification"] = {
                **automatic_report(auto_counts[arm], domains, draws), **raw}
        result["historical_labse"]["automatic_classification"] = automatic_report(
            auto_counts["historical_labse"], domains, draws)
        for candidate, baseline in c["comparisons"]:
            result["comparisons"][f"{candidate}_minus_{baseline}"] = {
                "metrics": metric_comparison(matrices[candidate], matrices[baseline], domains, draws),
                "automatic": automatic_comparison(auto_counts[candidate], auto_counts[baseline], domains, draws)}
        result["interpretation"] = "Each domain's fixed-threshold FPR<=0.001 AND recall>=0.5 is required. Point-value development qualification, not independent test, training-seed robustness, calibrated deployment guarantee, or automatic model selection. AP comparisons are secondary."
        result["status"] = "CHINESE_BASE_DEVELOPMENT_EVALUATED_NO_AUTOMATIC_WINNER"
        data.write_json(destination / "evaluation.json", result)
        return result
    except Exception as error:
        data.write_json(destination / "failure.json", {"status": "EVALUATION_FAILED_NO_RETRY",
                        "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "verify-models"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    if args.action == "evaluate" and args.evaluation is None:
        parser.error("evaluate requires --evaluation")
    if args.action == "train":
        result = run(args.out.resolve())
    elif args.action == "evaluate":
        result = evaluate(args.out.resolve(), args.evaluation.resolve())
    else:
        result = verify_models(args.out.resolve())
    print(data.json_bytes({"status": result["status"], "out": str(args.out)}).decode())


if __name__ == "__main__":
    main()
