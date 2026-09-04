"""Run the pre-frozen equal-budget Chinese fine-tuning comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_v6_style_transfer_common_v2 as common
import step28_v13_v1_13_v9_4_1_v6_style_transfer_source_linux_v2 as source


ROOT = common.REPO_ROOT


class TargetTransferRuntimeError(RuntimeError):
    pass


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_source_stage_contract(policy: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / policy["output_roots"]["source_and_zero_shot"]
    summary_path = root / "run_summary.json"
    selection_path = root / "source_epoch_selection.json"
    if not summary_path.is_file() or not selection_path.is_file():
        raise TargetTransferRuntimeError(
            "Run and return the frozen source/zero-shot stage first"
        )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise TargetTransferRuntimeError("Source result manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]:
        raise TargetTransferRuntimeError("Source manifest policy hash drift")
    records = {record["path"]: record for record in manifest["files"]}
    expected_model_ids = tuple(
        f"{arm}_seed_{seed}"
        for arm in policy["arms"]
        if arm != "generic"
        for seed in policy["source_optimization"]["confirmation_seeds"]
    )
    required_paths = {
        "run_summary.json": summary_path,
        "source_epoch_selection.json": selection_path,
        "training_audits.json": root / "training_audits.json",
        **{
            f"scores_{model_id}.npz": root / f"scores_{model_id}.npz"
            for model_id in expected_model_ids
        },
    }
    for name, path in required_paths.items():
        if name not in records:
            raise TargetTransferRuntimeError(f"Source manifest lacks {name}")
        record = records[name]
        if path.stat().st_size != int(record["size_bytes"]):
            raise TargetTransferRuntimeError(f"Source {name} size drift")
        if common.sha256_file(path) != record["sha256"]:
            raise TargetTransferRuntimeError(f"Source {name} hash drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if summary.get("policy_canonical_self_hash") != policy["canonical_self_hash"]:
        raise TargetTransferRuntimeError("Source result policy hash drift")
    if summary.get("status") != (
        "COMPLETED_V6_STYLE_SOURCE_AND_CHINESE_DEVELOPMENT_ZERO_SHOT"
    ):
        raise TargetTransferRuntimeError("Source result status drift")
    epochs = int(selection["selected_epochs"])
    if int(summary.get("selected_source_epochs", -1)) != epochs:
        raise TargetTransferRuntimeError("Source selected epoch records disagree")
    if epochs < 0 or epochs > int(
        policy["source_optimization"]["extended_max_epochs"]
    ):
        raise TargetTransferRuntimeError("Selected source epoch is invalid")
    training_audits = json.loads(
        required_paths["training_audits.json"].read_text(encoding="utf-8")
    )
    if set(training_audits) != set(expected_model_ids):
        raise TargetTransferRuntimeError("Source training audit model registry drift")
    return {
        "root": root,
        "selected_epochs": epochs,
        "manifest_sha256": common.sha256_file(manifest_path),
        "records": records,
        "model_ids": expected_model_ids,
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_source_reference_scores(
    contract: Mapping[str, Any], model_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_id not in contract["model_ids"]:
        raise TargetTransferRuntimeError(f"Unknown source replay model: {model_id}")
    name = f"scores_{model_id}.npz"
    record = contract["records"][name]
    path = contract["root"] / name
    with np.load(path, allow_pickle=False) as payload:
        if "v6_synthetic_audit" not in payload.files:
            raise TargetTransferRuntimeError(
                f"Source replay reference lacks synthetic-audit scores: {model_id}"
            )
        values = np.ascontiguousarray(
            payload["v6_synthetic_audit"], dtype="<f8"
        )
    if values.ndim != 1 or not np.isfinite(values).all():
        raise TargetTransferRuntimeError(
            f"Invalid source replay reference scores: {model_id}"
        )
    return values, {
        "path": name,
        "size_bytes": int(record["size_bytes"]),
        "sha256": str(record["sha256"]),
        "array_sha256": _array_sha256(values),
    }


def verify_source_replay_scores(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    prepared: Mapping[str, Sequence[Sequence[str]]],
    v6: Mapping[str, Any],
    reference: np.ndarray,
) -> dict[str, Any]:
    observed, _metrics, embeddings, _accounts = source._evaluate_v6_split(
        torch, encoder, tokenizer, prepared, v6, "synthetic_audit"
    )
    del embeddings
    if observed.shape != reference.shape:
        raise TargetTransferRuntimeError("Source replay score shape drift")
    maximum_error = float(np.max(np.abs(observed - reference)))
    if not np.array_equal(observed, reference):
        raise TargetTransferRuntimeError(
            "Source replay differs from the evaluated source-stage encoder: "
            f"max_abs_error={maximum_error}"
        )
    return {
        "score_count": len(observed),
        "exact_array_match": True,
        "max_abs_error": maximum_error,
        "array_sha256": _array_sha256(observed),
    }


def _target_head(torch: Any, policy: Mapping[str, Any]) -> Any:
    config = policy["target_optimization"]

    class PositiveCosineHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_scale = torch.nn.Parameter(
                torch.tensor(float(config["log_scale_initial"]), dtype=torch.float32)
            )
            self.bias = torch.nn.Parameter(
                torch.tensor(float(config["bias_initial"]), dtype=torch.float32)
            )

        def forward(self, cosines: Any) -> Any:
            lower, upper = config["log_scale_bounds"]
            scale = torch.exp(torch.clamp(self.log_scale, float(lower), float(upper)))
            return scale * cosines + self.bias

    return PositiveCosineHead().to("cuda:0")


def load_target_encoder(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    initialization: Path,
    seed: int,
) -> tuple[Any, Any]:
    source.set_determinism(torch, seed)
    encoder = SentenceTransformer(
        str(initialization), device="cuda:0", local_files_only=True
    )
    if int(encoder.max_seq_length) != int(
        policy["labse_model"]["native_max_sequence_length"]
    ):
        raise TargetTransferRuntimeError("Target LaBSE sequence length drift")
    if getattr(encoder, "default_prompt_name", None) is not None:
        encoder.default_prompt_name = None
    source.set_determinism(torch, seed)
    return encoder, encoder.tokenizer


def _selected_worlds(
    all_worlds: Sequence[str], world_count: int, selection_seed: int
) -> tuple[str, ...]:
    domain = f"step28-v6-style-target-v2:{selection_seed}:".encode("ascii")
    ordered = sorted(
        all_worlds,
        key=lambda value: (
            hashlib.sha256(domain + value.encode("utf-8")).digest(),
            value.encode("utf-8"),
        ),
    )
    if world_count <= 0 or world_count > len(ordered):
        raise TargetTransferRuntimeError("Invalid target world budget")
    return tuple(ordered[:world_count])


def train_target(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    initialization: Path,
    arm: str,
    source_seed: int,
    target_seed: int,
    budget_name: str,
    train_worlds: Mapping[str, Mapping[str, str]],
    train_prepared: Mapping[str, Sequence[Sequence[str]]],
    train_pairs: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    config = policy["target_optimization"]
    budget = config["budgets"][budget_name]
    encoder, tokenizer = load_target_encoder(
        torch, SentenceTransformer, policy, initialization, target_seed
    )
    head = _target_head(torch, policy)
    if config["dropout_enabled"]:
        encoder.train()
    else:
        encoder.eval()
    head.train()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(config["encoder_learning_rate"]),
                "weight_decay": float(config["encoder_weight_decay"]),
            },
            {
                "params": head.parameters(),
                "lr": float(config["head_learning_rate"]),
                "weight_decay": float(config["head_weight_decay"]),
            },
        ],
        betas=tuple(float(value) for value in config["adamw_betas"]),
        eps=float(config["adamw_eps"]),
    )
    pairs_by_world: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_pairs:
        pairs_by_world[str(row["world_uid"])].append(row)
    selected = _selected_worlds(
        tuple(train_worlds), int(budget["world_count"]), target_seed
    )
    update_count = 0
    first_gradient_norm = None
    trace = []
    width = int(budget["worlds_per_gradient_step"])
    for epoch in range(int(budget["epochs"])):
        generator = np.random.Generator(np.random.PCG64(target_seed + epoch))
        order = list(selected)
        generator.shuffle(order)
        epoch_losses = []
        for batch_number, start in enumerate(range(0, len(order), width), start=1):
            group = order[start : start + width]
            if len(group) != width:
                raise TargetTransferRuntimeError(
                    "Target world budget is not divisible by batch width"
                )
            optimizer.zero_grad(set_to_none=True)
            for world_uid in group:
                account_uids = tuple(
                    sorted(
                        train_worlds[world_uid],
                        key=lambda value: value.encode("utf-8"),
                    )
                )
                world_pairs = pairs_by_world[world_uid]
                if len(account_uids) != 28 or len(world_pairs) != 378:
                    raise TargetTransferRuntimeError(
                        f"Target world layout drift: {world_uid}"
                    )
                embeddings = source.encode_accounts(
                    torch,
                    encoder,
                    tokenizer,
                    train_prepared,
                    account_uids,
                )
                account_index = {
                    uid: index for index, uid in enumerate(account_uids)
                }
                left = [
                    account_index[str(row["seller_uid_left"])]
                    for row in world_pairs
                ]
                right = [
                    account_index[str(row["seller_uid_right"])]
                    for row in world_pairs
                ]
                cosines = (
                    embeddings[left].float() * embeddings[right].float()
                ).sum(dim=1)
                labels = torch.tensor(
                    [float(row["label"]) for row in world_pairs],
                    device="cuda:0",
                    dtype=torch.float32,
                )
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    head(cosines), labels
                )
                (loss / width).backward()
                epoch_losses.append(float(loss.detach().cpu()))
                del embeddings, cosines, labels, loss
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    float(config["gradient_clip_norm"]),
                )
                .detach()
                .cpu()
            )
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise TargetTransferRuntimeError("Invalid target parameter gradient")
            if first_gradient_norm is None:
                first_gradient_norm = gradient_norm
            optimizer.step()
            update_count += 1
            if batch_number % 10 == 0 or start + width >= len(order):
                print(
                    f"{arm} source={source_seed} target={target_seed} "
                    f"{budget_name} 第{epoch + 1}轮："
                    f"{batch_number}/{math.ceil(len(order) / width)}批",
                    flush=True,
                )
        trace.append(
            {
                "epoch": epoch + 1,
                "mean_world_loss": float(np.mean(epoch_losses)),
                "optimizer_updates": update_count,
            }
        )
    if update_count != int(budget["optimizer_updates"]):
        raise TargetTransferRuntimeError("Target optimizer update count drift")
    return (
        encoder,
        tokenizer,
        head,
        {
            "arm": arm,
            "source_seed": source_seed,
            "target_seed": target_seed,
            "budget": budget_name,
            "world_count": len(selected),
            "selected_world_uids_sha256": hashlib.sha256(
                json.dumps(
                    selected,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "epochs": int(budget["epochs"]),
            "optimizer_updates": update_count,
            "first_pre_clip_gradient_norm": first_gradient_norm,
            "trace": trace,
            "audit_a_truth_reads": 0,
            "audit_b_truth_reads": 0,
        },
    )


def target_metric_views(
    labels: Sequence[int],
    ranking_scores: Sequence[float],
    probabilities: Sequence[float],
    threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not (len(labels) == len(ranking_scores) == len(probabilities)):
        raise TargetTransferRuntimeError("Target metric view length mismatch")
    return (
        source.ranking_metrics(labels, ranking_scores),
        source.probability_metrics(probabilities, labels, None, threshold),
    )


def evaluate_target(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    head: Any,
    policy: Mapping[str, Any],
    development_worlds: Mapping[str, Mapping[str, str]],
    development_prepared: Mapping[str, Sequence[Sequence[str]]],
    development_pairs: Sequence[Mapping[str, Any]],
    qrel_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    encoder.eval()
    head.eval()
    pairs_by_world: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(development_pairs):
        pairs_by_world[str(row["world_uid"])].append((index, row))
    qrels = {
        str(row["query_seller_uid"]): set(row["relevant_seller_uids"])
        for row in qrel_rows
    }
    ranking_scores = np.empty(len(development_pairs), dtype="<f8")
    probabilities = np.empty(len(development_pairs), dtype="<f8")
    per_world_ranking = []
    per_world_probability = []
    per_world_retrieval = []
    threshold = float(policy["target_optimization"]["classification_threshold"])
    with torch.no_grad():
        for number, (world_uid, sellers) in enumerate(
            development_worlds.items(), start=1
        ):
            account_uids = tuple(
                sorted(sellers, key=lambda value: value.encode("utf-8"))
            )
            indexed_pairs = pairs_by_world[world_uid]
            world_pairs = [row for _index, row in indexed_pairs]
            embeddings = source.encode_accounts(
                torch,
                encoder,
                tokenizer,
                development_prepared,
                account_uids,
            )
            account_index = {
                uid: index for index, uid in enumerate(account_uids)
            }
            left = [
                account_index[str(row["seller_uid_left"])] for row in world_pairs
            ]
            right = [
                account_index[str(row["seller_uid_right"])] for row in world_pairs
            ]
            cosines = (
                embeddings[left].float() * embeddings[right].float()
            ).sum(dim=1)
            world_ranking_scores = np.ascontiguousarray(
                cosines.double().cpu().numpy(), dtype="<f8"
            )
            world_probabilities = (
                torch.sigmoid(head(cosines)).double().cpu().numpy()
            )
            for (index, _row), score, probability in zip(
                indexed_pairs, world_ranking_scores, world_probabilities
            ):
                ranking_scores[index] = score
                probabilities[index] = probability
            labels = [int(row["label"]) for row in world_pairs]
            ranking_record, probability_record = target_metric_views(
                labels, world_ranking_scores, world_probabilities, threshold
            )
            per_world_ranking.append(ranking_record)
            per_world_probability.append(probability_record)
            candidates: dict[str, dict[str, float]] = defaultdict(dict)
            for row, score in zip(world_pairs, world_ranking_scores):
                left_uid = str(row["seller_uid_left"])
                right_uid = str(row["seller_uid_right"])
                candidates[left_uid][right_uid] = float(score)
                candidates[right_uid][left_uid] = float(score)
            per_world_retrieval.append(
                source.aggregate_retrieval(
                    candidates, {uid: qrels[uid] for uid in account_uids}
                )
            )
            del embeddings, cosines
            if number % 50 == 0:
                print(
                    f"中文微调开发推理：{number}/{len(development_worlds)}世界",
                    flush=True,
                )
    labels = [int(row["label"]) for row in development_pairs]

    def mean_records(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
        return {
            name: float(np.mean([row[name] for row in records]))
            for name in records[0]
        }

    pooled_ranking, pooled_probability = target_metric_views(
        labels, ranking_scores, probabilities, threshold
    )
    return ranking_scores, probabilities, {
        "ranking_score": "raw_account_cosine",
        "probability_score": "sigmoid_positive_cosine_head",
        "pooled_ranking": pooled_ranking,
        "pooled_probability_and_threshold": pooled_probability,
        "world_equal_ranking": mean_records(per_world_ranking),
        "world_equal_probability_and_threshold": mean_records(
            per_world_probability
        ),
        "retrieval_world_equal": mean_records(per_world_retrieval),
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def paired_world_intervals(
    policy: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    generic_scores: Sequence[np.ndarray],
    correct_scores: Sequence[np.ndarray],
    permuted_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not (
        len(generic_scores) == len(correct_scores) == len(permuted_scores) == 3
    ):
        raise TargetTransferRuntimeError("Target paired seed count drift")
    generic_world = [
        source._per_world_ap(pairs, values)[1] for values in generic_scores
    ]
    correct_world = [
        source._per_world_ap(pairs, values)[1] for values in correct_scores
    ]
    permuted_world = [
        source._per_world_ap(pairs, values)[1] for values in permuted_scores
    ]
    correct_minus_generic = np.mean(
        np.vstack(
            [
                correct_world[index] - generic_world[index]
                for index in range(3)
            ]
        ),
        axis=0,
    )
    correct_minus_permuted = np.mean(
        np.vstack(
            [
                correct_world[index] - permuted_world[index]
                for index in range(3)
            ]
        ),
        axis=0,
    )
    world_count = len(correct_minus_generic)
    repetitions = int(policy["evaluation"]["bootstrap_repetitions"])
    rng = np.random.Generator(
        np.random.PCG64(int(policy["evaluation"]["bootstrap_seed"]))
    )
    draws = rng.integers(
        0, world_count, size=(repetitions, world_count), dtype=np.int32
    )

    def interval(values: np.ndarray) -> dict[str, float]:
        series = values[draws].mean(axis=1)
        return {
            "q025": float(np.quantile(series, 0.025, method="linear")),
            "q975": float(np.quantile(series, 0.975, method="linear")),
        }

    return {
        "unit": "world",
        "world_count": world_count,
        "repetitions": repetitions,
        "estimand": "mean_per_world_average_precision_delta",
        "correct_minus_generic": interval(correct_minus_generic),
        "correct_minus_permuted": interval(correct_minus_permuted),
    }


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
        )
    return records


def _safe_remove_temporary_checkpoint(path: Path, building: Path) -> None:
    resolved = path.resolve()
    checkpoint_root = (building / "_source_checkpoints").resolve()
    if resolved.parent != checkpoint_root:
        raise TargetTransferRuntimeError("Unsafe temporary checkpoint path")
    if resolved.exists():
        shutil.rmtree(resolved)


def run() -> dict[str, Any]:
    policy = common.load_policy()
    static = common.validate_static_inputs()
    source_contract = load_source_stage_contract(policy)
    selected_epochs = int(source_contract["selected_epochs"])
    torch, SentenceTransformer = source.require_gpu_runtime(policy)
    implementation_records = source.implementation_file_records(
        (
            "scripts/step28_v13_v1_13_v9_4_1_v6_style_transfer_common_v2.py",
            "scripts/step28_v13_v1_13_v9_4_1_v6_style_transfer_source_linux_v2.py",
            "scripts/step28_v13_v1_13_v9_4_1_v6_style_transfer_target_linux_v2.py",
            "scripts/run_step28_v13_v1_13_v9_4_1_v6_style_transfer_target_v2_linux_20260904.sh",
        )
    )
    runtime_versions = source.runtime_package_versions()
    output_root = ROOT / policy["output_roots"]["same_budget_target"]
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or building.exists():
        raise TargetTransferRuntimeError("Target output path already exists")
    building.mkdir(parents=True)
    checkpoint_root = building / "_source_checkpoints"
    checkpoint_root.mkdir()
    started = time.time()

    v6 = common.load_v6(policy)
    train_worlds = common.load_chinese_style_streams(policy, "train")
    development_worlds = common.load_chinese_style_streams(policy, "development")
    train_pairs = common.load_chinese_pairs(policy, "train", include_labels=True)
    development_pairs = common.load_chinese_pairs(
        policy, "development", include_labels=True
    )
    development_qrels = common.load_chinese_development_qrels(policy)
    train_streams = {
        seller_uid: stream
        for sellers in train_worlds.values()
        for seller_uid, stream in sellers.items()
    }
    development_streams = {
        seller_uid: stream
        for sellers in development_worlds.values()
        for seller_uid, stream in sellers.items()
    }
    v6_streams = {
        uid: row["style_stream"] for uid, row in v6["accounts"].items()
    }
    temporary_encoder, tokenizer = source.load_encoder(
        policy,
        SentenceTransformer,
        torch,
        int(policy["source_optimization"]["selection_seed"]),
    )
    v6_prepared = source.prepare_stream_chunks(
        tokenizer, v6_streams, policy
    )
    train_prepared = source.prepare_stream_chunks(
        tokenizer, train_streams, policy
    )
    development_prepared = source.prepare_stream_chunks(
        tokenizer, development_streams, policy
    )
    del temporary_encoder
    gc.collect()
    torch.cuda.empty_cache()

    results: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}
    source_training: dict[str, Any] = {}
    model_ids: dict[str, dict[str, list[str]]] = {
        budget: defaultdict(list)
        for budget in policy["target_optimization"]["budgets"]
    }
    seed_pairs = [
        (int(source_seed), int(target_seed))
        for source_seed, target_seed in policy["target_optimization"][
            "source_target_seed_pairs"
        ]
    ]
    for arm in policy["arms"]:
        for source_seed, target_seed in seed_pairs:
            if arm == "generic":
                initialization = ROOT / policy["labse_model"]["path"]
            else:
                checkpoint = checkpoint_root / f"{arm}_seed_{source_seed}"
                print(
                    f"重放来源初始化：{arm} seed={source_seed} "
                    f"epochs={selected_epochs}",
                    flush=True,
                )
                encoder, _tokenizer, trace, audit = source.train_source_encoder(
                    torch,
                    SentenceTransformer,
                    policy,
                    v6,
                    arm,
                    source_seed,
                    selected_epochs,
                    development_prepared=v6_prepared,
                )
                model_id = f"{arm}_seed_{source_seed}"
                reference_scores, reference_record = load_source_reference_scores(
                    source_contract, model_id
                )
                pre_save_replay = verify_source_replay_scores(
                    torch,
                    encoder,
                    _tokenizer,
                    v6_prepared,
                    v6,
                    reference_scores,
                )
                encoder.save(str(checkpoint))
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
                reloaded_encoder, reloaded_tokenizer = load_target_encoder(
                    torch,
                    SentenceTransformer,
                    policy,
                    checkpoint,
                    target_seed,
                )
                reloaded_encoder.eval()
                post_load_replay = verify_source_replay_scores(
                    torch,
                    reloaded_encoder,
                    reloaded_tokenizer,
                    v6_prepared,
                    v6,
                    reference_scores,
                )
                del reloaded_encoder
                gc.collect()
                torch.cuda.empty_cache()
                source_training[model_id] = {
                    **audit,
                    "trace": trace,
                    "source_result_manifest_sha256": source_contract[
                        "manifest_sha256"
                    ],
                    "reference_scores": reference_record,
                    "pre_save_replay": pre_save_replay,
                    "post_load_replay": post_load_replay,
                }
                initialization = checkpoint
            for budget_name in policy["target_optimization"]["budgets"]:
                model_id = (
                    f"{arm}_source_{source_seed}_target_{target_seed}_{budget_name}"
                )
                print(f"开始中文同预算模型：{model_id}", flush=True)
                encoder, tokenizer, head, audit = train_target(
                    torch,
                    SentenceTransformer,
                    policy,
                    initialization,
                    arm,
                    source_seed,
                    target_seed,
                    budget_name,
                    train_worlds,
                    train_prepared,
                    train_pairs,
                )
                ranking_score, probability, metrics = evaluate_target(
                    torch,
                    encoder,
                    tokenizer,
                    head,
                    policy,
                    development_worlds,
                    development_prepared,
                    development_pairs,
                    development_qrels,
                )
                results[model_id] = {"training": audit, "development": metrics}
                scores[model_id] = ranking_score
                model_ids[budget_name][arm].append(model_id)
                np.savez_compressed(
                    building / f"development_scores_{model_id}.npz",
                    ranking_scores=ranking_score,
                    probabilities=probability,
                )
                del encoder, head
                gc.collect()
                torch.cuda.empty_cache()
            if arm != "generic":
                _safe_remove_temporary_checkpoint(initialization, building)
    if any(checkpoint_root.iterdir()):
        raise TargetTransferRuntimeError("Temporary source checkpoints remain")
    checkpoint_root.rmdir()

    comparisons = {}
    for budget_name, arm_registry in model_ids.items():
        generic_ids = arm_registry["generic"]
        correct_ids = arm_registry["v6_correct"]
        permuted_ids = arm_registry["v6_permuted"]
        intervals = paired_world_intervals(
            policy,
            development_pairs,
            [scores[value] for value in generic_ids],
            [scores[value] for value in correct_ids],
            [scores[value] for value in permuted_ids],
        )

        def mean_world_equal(ids: Sequence[str]) -> float:
            return float(
                np.mean(
                    [
                        results[value]["development"]["world_equal_ranking"][
                            "average_precision"
                        ]
                        for value in ids
                    ]
                )
            )

        generic_ap = mean_world_equal(generic_ids)
        correct_ap = mean_world_equal(correct_ids)
        permuted_ap = mean_world_equal(permuted_ids)
        comparisons[budget_name] = {
            "primary_aggregation": "world_equal",
            "generic_mean_world_equal_average_precision": generic_ap,
            "correct_mean_world_equal_average_precision": correct_ap,
            "permuted_mean_world_equal_average_precision": permuted_ap,
            "correct_minus_generic_mean_world_equal_ap": correct_ap - generic_ap,
            "correct_minus_permuted_mean_world_equal_ap": correct_ap - permuted_ap,
            "pooled_average_precision_descriptive": {
                arm_name: float(
                    np.mean(
                        [
                            results[value]["development"]["pooled_ranking"]
                            ["average_precision"]
                            for value in arm_registry[arm_name]
                        ]
                    )
                )
                for arm_name in policy["arms"]
            },
            "bootstrap": intervals,
            "development_candidate": bool(
                correct_ap > generic_ap
                and correct_ap > permuted_ap
                and intervals["correct_minus_generic"]["q025"] > 0.0
                and intervals["correct_minus_permuted"]["q025"] > 0.0
            ),
        }
    overall = {
        "primary_budget": policy["evaluation"]["primary_budget"],
        "primary_aggregation": "world_equal",
        "low_resource_development_candidate": comparisons[
            policy["evaluation"]["primary_budget"]
        ][
            "development_candidate"
        ],
        "saturation_budget": policy["evaluation"]["saturation_budget"],
        "learning_curve_budgets": policy["evaluation"][
            "learning_curve_budgets"
        ],
        "claim_boundary": (
            "CHINESE_DEVELOPMENT_CANDIDATE_ONLY_NOT_BLIND_TRANSFER_CONFIRMATION"
        ),
    }
    _write_json(building / "model_results.json", results)
    _write_json(building / "source_replay_audits.json", source_training)
    _write_json(building / "comparisons.json", comparisons)
    summary = {
        "status": "COMPLETED_EQUAL_BUDGET_CHINESE_DEVELOPMENT_COMPARISON",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "selected_source_epochs": selected_epochs,
        "model_count": len(results),
        "comparisons": overall,
        "elapsed_seconds": time.time() - started,
        "temporary_source_checkpoints_retained": 0,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
        "static_input_validation": static,
    }
    _write_json(building / "run_summary.json", summary)
    manifest = {
        "version": policy["version"],
        "status": summary["status"],
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "files": _file_manifest(building),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "packages": runtime_versions,
        },
        "implementation_files": implementation_records,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }
    _write_json(building / "manifest.json", manifest)
    os.replace(building, output_root)
    return summary


def validate() -> dict[str, Any]:
    policy = common.load_policy()
    static = common.validate_static_inputs()
    return {
        "status": "PASSED_V6_STYLE_TRANSFER_TARGET_CONTRACT",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "budgets": policy["target_optimization"]["budgets"],
        "source_target_seed_pairs": policy["target_optimization"][
            "source_target_seed_pairs"
        ],
        "static": static,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def smoke() -> dict[str, Any]:
    policy = common.load_policy()
    torch, SentenceTransformer = source.require_gpu_runtime(policy)
    target_seed = int(
        policy["target_optimization"]["source_target_seed_pairs"][0][1]
    )
    encoder, tokenizer = load_target_encoder(
        torch,
        SentenceTransformer,
        policy,
        ROOT / policy["labse_model"]["path"],
        target_seed,
    )
    encoder.train()
    head = _target_head(torch, policy)
    head.train()
    prepared = source.prepare_stream_chunks(
        tokenizer,
        {
            "a": " ".join(["W5, N2!"] * 32),
            "b": " ".join(["W4. N3?"] * 31),
            "c": " ".join(["W7; N1."] * 30),
            "d": " ".join(["W3: N4!"] * 29),
        },
        policy,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(policy["target_optimization"]["encoder_learning_rate"]),
                "weight_decay": float(
                    policy["target_optimization"]["encoder_weight_decay"]
                ),
            },
            {
                "params": head.parameters(),
                "lr": float(policy["target_optimization"]["head_learning_rate"]),
                "weight_decay": float(
                    policy["target_optimization"]["head_weight_decay"]
                ),
            },
        ],
        betas=tuple(
            float(value)
            for value in policy["target_optimization"]["adamw_betas"]
        ),
        eps=float(policy["target_optimization"]["adamw_eps"]),
    )
    optimizer.zero_grad(set_to_none=True)
    embeddings = source.encode_accounts(
        torch, encoder, tokenizer, prepared, ("a", "b", "c", "d"), batch_size=4
    )
    cosines = torch.stack(
        ((embeddings[0] * embeddings[1]).sum(), (embeddings[2] * embeddings[3]).sum())
    )
    labels = torch.tensor([1.0, 0.0], device="cuda:0", dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(cosines), labels
    )
    loss.backward()
    encoder_gradient = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )
    head_gradient = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in head.parameters()
        if parameter.grad is not None
    )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (encoder_gradient, head_gradient)
    ):
        raise TargetTransferRuntimeError("Target GPU smoke produced invalid gradients")
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(head.parameters()),
        float(policy["target_optimization"]["gradient_clip_norm"]),
    )
    optimizer.step()
    return {
        "status": "PASSED_V6_STYLE_TRANSFER_TARGET_GPU_SMOKE",
        "loss": float(loss.detach().cpu()),
        "encoder_gradient_l1": encoder_gradient,
        "head_gradient_l1": head_gradient,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "smoke", "run"))
    arguments = parser.parse_args()
    if arguments.command == "validate":
        result = validate()
    elif arguments.command == "smoke":
        result = smoke()
    else:
        result = run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
