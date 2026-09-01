#!/usr/bin/env python3
"""Directly fine-tune LaBSE on complete redacted Chinese seller text.

The formal path is Linux/GPU-only.  Audit-A/B truth is never addressable from
this module.  Development model selection is fixed to the last epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_labse_finetune_common_v1 as common
import step28_v13_v1_13_v9_4_1_transfer_claim_controls_v4 as controls


AUTHORIZATION_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_labse_direct_finetune_v1_execution.json"
)
MODEL_IDS = ("ft_base", "ft_joint")


class DirectFinetuneError(ValueError):
    """Raised when direct fine-tuning violates its scientific contract."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(value, dtype="<f8"), allow_pickle=False)


def load_execution_authorization(policy: Mapping[str, Any]) -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise DirectFinetuneError(
            "Direct fine-tuning is implementation-only; freeze the minimal execution "
            "authorization before formal GPU training"
        )
    value = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    if (
        value.get("version")
        != "step28-v13-v1.13-v9.4.1-labse-direct-finetune-v1-execution"
        or value.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or value.get("scientific_scope")
        != "train_and_development_direct_finetune_comparison_only"
        or value.get("formal_gpu_training_authorized") is not True
        or value.get("audit_a_prediction_authorized") is not False
        or value.get("audit_a_truth_authorized") is not False
        or value.get("audit_b_prediction_authorized") is not False
        or value.get("audit_b_truth_authorized") is not False
    ):
        raise DirectFinetuneError("Direct fine-tune execution authorization drift")
    return value


def require_gpu_runtime(policy: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    if platform.system() != "Linux":
        raise DirectFinetuneError("Formal direct fine-tuning is Linux-only")
    try:
        import sentence_transformers
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise DirectFinetuneError("PyTorch/Transformers stack is incomplete") from exc
    expected = policy["runtime"]
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_runtime": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }
    for key in (
        "python",
        "numpy",
        "torch",
        "torch_cuda_runtime",
        "cudnn_runtime",
        "transformers",
        "sentence_transformers",
    ):
        if str(observed[key]) != str(expected[key]):
            raise DirectFinetuneError(
                f"Direct fine-tune runtime drift: {key} expected={expected[key]} "
                f"observed={observed[key]}"
            )
    if not torch.cuda.is_available():
        raise DirectFinetuneError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if (
        properties.name != expected["gpu_name"]
        or [properties.major, properties.minor] != expected["compute_capability"]
    ):
        raise DirectFinetuneError("Direct fine-tune GPU identity drift")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise DirectFinetuneError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    if os.environ.get("TOKENIZERS_PARALLELISM", "").lower() != "false":
        raise DirectFinetuneError("TOKENIZERS_PARALLELISM must equal false")
    return torch, SentenceTransformer, transformers


def set_determinism(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _world_row_indices(rows: Mapping[str, Any]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, world_uid in enumerate(rows["world_uids"]):
        grouped.setdefault(str(world_uid), []).append(index)
    result = {
        world_uid: np.asarray(indices, dtype=np.int64)
        for world_uid, indices in grouped.items()
    }
    if any(len(indices) != 378 for indices in result.values()):
        raise DirectFinetuneError("Fine-tune pair rows are not complete K28 worlds")
    return result


def _build_text_indices(
    policy: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, dict[str, tuple[str, ...]]]]]:
    layout = policy["formal_layout"]
    output = {}
    for split in policy["allowed_splits"]:
        path = common.verified_redacted_items_path(policy, split)
        output[split] = common.build_redacted_text_index(
            common.iter_jsonl(path),
            expected_worlds=int(layout["worlds_per_split"]),
            expected_sellers_per_world=int(layout["sellers_per_world"]),
            expected_items_per_world=int(layout["items_per_world"]),
        )
    return output


def _chunk_cache_for_world(
    world: Mapping[str, Mapping[str, Sequence[str]]],
    tokenizer: Any,
    budget: int,
    cache: dict[str, tuple[str, ...]],
) -> None:
    for seller in world.values():
        for field in common.FIELDS:
            for text in seller[field]:
                if text not in cache:
                    cache[text] = common.chunk_text_exact(tokenizer, text, budget)


def _batch_sentence_embeddings(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    chunks: Sequence[str],
    *,
    batch_size: int,
    device: Any,
    use_autocast: bool,
) -> Any:
    outputs = []
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        features = tokenizer(
            batch,
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if int(features["attention_mask"].sum(dim=1).max()) > 256:
            raise DirectFinetuneError("A fine-tune chunk exceeds 256 tokens")
        features = {key: value.to(device) for key, value in features.items()}
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            encoded = encoder(features)["sentence_embedding"]
        outputs.append(
            torch.nn.functional.normalize(encoded.float(), p=2, dim=1, eps=1e-12)
        )
    return torch.cat(outputs, dim=0)


def differentiable_world_semantics(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    world_texts: Mapping[str, Mapping[str, Sequence[str]]],
    seller_uid_left: Sequence[str],
    seller_uid_right: Sequence[str],
    chunk_cache: dict[str, tuple[str, ...]],
    *,
    token_budget: int,
    chunk_batch_size: int,
    device: Any,
    use_autocast: bool,
) -> Any:
    _chunk_cache_for_world(world_texts, tokenizer, token_budget, chunk_cache)
    unique_texts = sorted(
        {
            text
            for seller in world_texts.values()
            for field in common.FIELDS
            for text in seller[field]
        },
        key=lambda value: (
            hashlib.sha256(value.encode("utf-8")).digest(),
            value.encode("utf-8"),
        ),
    )
    chunks: list[str] = []
    text_slices: dict[str, slice] = {}
    for text in unique_texts:
        start = len(chunks)
        chunks.extend(chunk_cache[text])
        text_slices[text] = slice(start, len(chunks))
    chunk_vectors = _batch_sentence_embeddings(
        torch,
        encoder,
        tokenizer,
        chunks,
        batch_size=chunk_batch_size,
        device=device,
        use_autocast=use_autocast,
    )
    text_vectors = {
        text: common.torch_unit_mean(chunk_vectors[text_slices[text]])
        for text in unique_texts
    }
    seller_vectors = {
        seller_uid: {
            field: torch.stack([text_vectors[text] for text in fields[field]])
            for field in common.FIELDS
        }
        for seller_uid, fields in world_texts.items()
    }
    if len(seller_uid_left) != 378 or len(seller_uid_right) != 378:
        raise DirectFinetuneError("Fine-tune world lacks 378 pair endpoints")
    return torch.stack(
        [
            common.torch_six_pair_aggregates(
                seller_vectors[str(left)], seller_vectors[str(right)], top_k=3
            )
            for left, right in zip(seller_uid_left, seller_uid_right)
        ]
    )


def _standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype="<f8")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise DirectFinetuneError("Numeric fine-tune input is non-finite")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale <= 1e-12] = 1.0
    return np.ascontiguousarray(mean, dtype="<f8"), np.ascontiguousarray(
        scale, dtype="<f8"
    )


def _numeric_views(
    rows: Mapping[str, Any],
    identity_phi: np.ndarray,
    legacy_mean: np.ndarray,
    legacy_scale: np.ndarray,
    include_identity: bool,
) -> np.ndarray:
    legacy = (np.asarray(rows["base24"][:, :18], dtype="<f8") - legacy_mean) / legacy_scale
    pieces = [legacy]
    if include_identity:
        pieces.append(np.asarray(identity_phi, dtype="<f8"))
    result = np.ascontiguousarray(np.column_stack(pieces), dtype="<f8")
    if not np.isfinite(result).all():
        raise DirectFinetuneError("Fine-tune numeric view is non-finite")
    return result


def _load_encoder(
    policy: Mapping[str, Any], SentenceTransformer: Any, device: str
) -> tuple[Any, Any]:
    model_path = ROOT / str(policy["labse_model"]["path"])
    encoder = SentenceTransformer(
        str(model_path), device=device, local_files_only=True
    )
    if int(encoder.max_seq_length) != 256:
        raise DirectFinetuneError("Loaded LaBSE native sequence length drift")
    if getattr(encoder, "default_prompt_name", None) is not None:
        encoder.default_prompt_name = None
    tokenizer = encoder.tokenizer
    return encoder, tokenizer


def _train_one(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    model_id: str,
    selected_worlds: tuple[str, ...],
    text_index: Mapping[str, Any],
    train: Mapping[str, Any],
    development: Mapping[str, Any],
    train_labels: np.ndarray,
    train_rows_by_world: Mapping[str, np.ndarray],
    development_rows_by_world: Mapping[str, np.ndarray],
    *,
    smoke: bool,
) -> tuple[np.ndarray, dict[str, Any], Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_id not in MODEL_IDS:
        raise DirectFinetuneError("Unknown direct fine-tune model")
    optimization = policy["optimization"]
    seed = int(optimization["development_screen_seed"])
    set_determinism(torch, seed)
    device = torch.device("cuda:0")
    encoder, tokenizer = _load_encoder(policy, SentenceTransformer, "cuda:0")
    include_identity = model_id == "ft_joint"
    selected_world_set = set(selected_worlds)
    selected_mask = np.fromiter(
        (world_uid in selected_world_set for world_uid in train["world_uids"]),
        dtype=bool,
        count=len(train["world_uids"]),
    )
    selected_row_worlds = [
        world_uid
        for world_uid, keep in zip(train["world_uids"], selected_mask)
        if keep
    ]
    identity_scale, identity_mu = controls.core.common_v1.fit_identity_transform(
        train["identity33"][selected_mask], selected_row_worlds
    )
    train_identity_phi, _ = controls.core.common_v1.apply_identity_transform(
        train["identity33"], identity_scale, identity_mu
    )
    development_identity_phi, _ = controls.core.common_v1.apply_identity_transform(
        development["identity33"], identity_scale, identity_mu
    )
    legacy_mean, legacy_scale = _standardizer(
        train["base24"][selected_mask, :18]
    )
    train_numeric = _numeric_views(
        train, train_identity_phi, legacy_mean, legacy_scale, include_identity
    )
    development_numeric = _numeric_views(
        development,
        development_identity_phi,
        legacy_mean,
        legacy_scale,
        include_identity,
    )
    numeric_width = 18 + (33 if include_identity else 0)
    head = torch.nn.Linear(numeric_width + 6, 1, bias=True).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.constant_(head.bias, math.log(20.0 / 358.0))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(optimization["encoder_learning_rate"]),
                "weight_decay": float(optimization["weight_decay"]),
            },
            {
                "params": head.parameters(),
                "lr": float(optimization["head_learning_rate"]),
                "weight_decay": 0.0,
            },
        ]
    )
    epochs = 1 if smoke else int(optimization["epochs"])
    worlds_per_step = int(optimization["worlds_per_gradient_step"])
    chunk_cache: dict[str, tuple[str, ...]] = {}
    training_log = []
    encoder.train()
    head.train()
    for epoch in range(epochs):
        generator = np.random.Generator(np.random.PCG64(seed + epoch))
        order = list(selected_worlds)
        generator.shuffle(order)
        epoch_losses = []
        for group_start in range(0, len(order), worlds_per_step):
            group = order[group_start : group_start + worlds_per_step]
            optimizer.zero_grad(set_to_none=True)
            for world_uid in group:
                indices = train_rows_by_world[world_uid]
                semantics = differentiable_world_semantics(
                    torch,
                    encoder,
                    tokenizer,
                    text_index["train"][world_uid],
                    [train["seller_uid_left"][index] for index in indices],
                    [train["seller_uid_right"][index] for index in indices],
                    chunk_cache,
                    token_budget=256,
                    chunk_batch_size=int(optimization["chunk_batch_size"]),
                    device=device,
                    use_autocast=True,
                )
                numeric = torch.from_numpy(train_numeric[indices]).to(
                    device=device, dtype=torch.float32
                )
                labels = torch.from_numpy(train_labels[indices].astype(np.float32)).to(
                    device
                )
                logits = head(torch.cat((numeric, semantics.float()), dim=1)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="mean"
                )
                (loss / len(group)).backward()
                epoch_losses.append(float(loss.detach().cpu()))
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                float(optimization["gradient_clip_norm"]),
            )
            optimizer.step()
        training_log.append(
            {
                "epoch": epoch + 1,
                "mean_world_loss": float(np.mean(epoch_losses)),
                "minimum_world_loss": float(np.min(epoch_losses)),
                "maximum_world_loss": float(np.max(epoch_losses)),
            }
        )
    predictions = np.empty(len(development["world_uids"]), dtype="<f8")
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for world_uid in sorted(
            development_rows_by_world, key=lambda value: value.encode("utf-8")
        ):
            indices = development_rows_by_world[world_uid]
            semantics = differentiable_world_semantics(
                torch,
                encoder,
                tokenizer,
                text_index["development"][world_uid],
                [development["seller_uid_left"][index] for index in indices],
                [development["seller_uid_right"][index] for index in indices],
                chunk_cache,
                token_budget=256,
                chunk_batch_size=int(optimization["chunk_batch_size"]),
                device=device,
                use_autocast=True,
            )
            numeric = torch.from_numpy(development_numeric[indices]).to(
                device=device, dtype=torch.float32
            )
            logits = head(torch.cat((numeric, semantics.float()), dim=1)).squeeze(1)
            predictions[indices] = torch.sigmoid(logits).double().cpu().numpy()
    if not np.isfinite(predictions).all():
        raise DirectFinetuneError("Direct fine-tune development prediction is non-finite")
    audit = {
        "model_id": model_id,
        "world_count": len(selected_worlds),
        "pair_count": len(selected_worlds) * 378,
        "epochs": epochs,
        "training_log": training_log,
        "chunk_cache_text_count": len(chunk_cache),
        "whole_document_truncation_count": 0,
    }
    return (
        predictions,
        audit,
        encoder,
        head,
        legacy_mean,
        legacy_scale,
        identity_scale,
        identity_mu,
    )


def validate_contract() -> dict[str, Any]:
    policy = common.load_policy()
    separately_authorized = False
    if AUTHORIZATION_PATH.is_file():
        load_execution_authorization(policy)
        separately_authorized = True
    return {
        "status": "PASSED_LABSE_DIRECT_FINETUNE_IMPLEMENTATION_CONTRACT_NO_TRUTH_READ",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "implementation_policy_embedded_authorization": policy[
            "formal_gpu_training_authorized"
        ],
        "separate_execution_authorization_present": separately_authorized,
        "formal_gpu_training_authorized": separately_authorized,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def smoke_runtime() -> dict[str, Any]:
    """Exercise one real LaBSE backward step without reading formal supervision."""

    policy = common.load_policy()
    torch, SentenceTransformer, _transformers = require_gpu_runtime(policy)
    common.verify_labse_payload(policy)
    set_determinism(torch, int(policy["optimization"]["development_screen_seed"]))
    encoder, tokenizer = _load_encoder(policy, SentenceTransformer, "cuda:0")
    device = torch.device("cuda:0")
    texts = (
        "原装配件，工作日发货。支持批量订购。",
        "商品说明：包装完整；数量请提前确认。",
        "现货样品，付款后依订单顺序处理。",
        "规格和交付时间以页面说明为准。",
    )
    chunks: list[str] = []
    text_slices: list[slice] = []
    for text in texts:
        parts = common.chunk_text_exact(tokenizer, text, 256)
        if "".join(parts) != text:
            raise DirectFinetuneError("Runtime smoke chunk reconstruction failed")
        start = len(chunks)
        chunks.extend(parts)
        text_slices.append(slice(start, len(chunks)))
    encoder.train()
    vectors = _batch_sentence_embeddings(
        torch,
        encoder,
        tokenizer,
        chunks,
        batch_size=4,
        device=device,
        use_autocast=True,
    )
    text_vectors = [common.torch_unit_mean(vectors[index]) for index in text_slices]
    seller_left = {
        "title": torch.stack((text_vectors[0],)),
        "description": torch.stack((text_vectors[1],)),
    }
    seller_right = {
        "title": torch.stack((text_vectors[2],)),
        "description": torch.stack((text_vectors[3],)),
    }
    pair_features = torch.stack(
        (
            common.torch_six_pair_aggregates(seller_left, seller_left, top_k=3),
            common.torch_six_pair_aggregates(seller_left, seller_right, top_k=3),
        )
    )
    head = torch.nn.Linear(6, 1, bias=True).to(device)
    torch.nn.init.constant_(head.weight, 0.01)
    torch.nn.init.zeros_(head.bias)
    logits = head(pair_features.float()).squeeze(1)
    labels = torch.tensor((1.0, 0.0), dtype=torch.float32, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    encoder_gradient = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )
    if not math.isfinite(float(loss.detach().cpu())) or not math.isfinite(
        encoder_gradient
    ) or encoder_gradient <= 0.0:
        raise DirectFinetuneError("Runtime smoke did not backpropagate into LaBSE")
    del encoder, head
    torch.cuda.empty_cache()
    return {
        "status": "PASSED_LABSE_DIRECT_FINETUNE_RUNTIME_SMOKE_NO_FORMAL_TRUTH_READ",
        "loss": float(loss.detach().cpu()),
        "encoder_gradient_l1": encoder_gradient,
        "formal_train_labels_read": 0,
        "formal_development_labels_read": 0,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def run() -> dict[str, Any]:
    policy = common.load_policy()
    authorization = load_execution_authorization(policy)
    torch, SentenceTransformer, _transformers = require_gpu_runtime(policy)
    common.verify_labse_payload(policy)
    (
        _execution,
        _v3_policy,
        train,
        development,
        train_labels,
        development_labels,
        relevance,
    ) = controls._load_inputs(policy)
    text_index = _build_text_indices(policy)
    train_rows = _world_row_indices(train)
    development_rows = _world_row_indices(development)
    subsets = common.nested_world_subsets(
        tuple(train_rows), policy["training_budgets"]["world_counts"]
    )
    if set(text_index["train"]) != set(train_rows) or set(
        text_index["development"]
    ) != set(development_rows):
        raise DirectFinetuneError("Text and pair world universes do not match")
    output = ROOT / str(policy["output_root"])
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise DirectFinetuneError("Direct fine-tune output already exists")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        budget_counts = list(policy["training_budgets"]["world_counts"])
        model_ids = list(MODEL_IDS)
        reports = {}
        for world_count in budget_counts:
            selected = subsets[int(world_count)]
            for model_id in model_ids:
                (
                    predictions,
                    training_audit,
                    encoder,
                    head,
                    legacy_mean,
                    legacy_scale,
                    identity_scale,
                    identity_mu,
                ) = _train_one(
                    torch,
                    SentenceTransformer,
                    policy,
                    model_id,
                    selected,
                    text_index,
                    train,
                    development,
                    train_labels,
                    train_rows,
                    development_rows,
                    smoke=False,
                )
                threshold, evaluation = controls._evaluate_model(
                    predictions,
                    development_labels,
                    development,
                    relevance,
                )
                key = f"worlds_{int(world_count):03d}/{model_id}"
                root = building / key
                save_array(root / "development_probabilities.npy", predictions)
                save_array(root / "legacy_mean.npy", legacy_mean)
                save_array(root / "legacy_scale.npy", legacy_scale)
                save_array(root / "identity_scale.npy", identity_scale)
                save_array(root / "identity_mu.npy", identity_mu)
                write_json(
                    root / "training_and_evaluation.json",
                    {
                        "training": training_audit,
                        "development_threshold": float(threshold),
                        "development": evaluation,
                    },
                )
                if int(world_count) == 500:
                    encoder.save_pretrained(str(root / "encoder"))
                    torch.save(head.state_dict(), root / "head_state.pt")
                reports[key] = {
                    "training": training_audit,
                    "development_threshold": float(threshold),
                    "development": evaluation,
                }
                del encoder, head
                torch.cuda.empty_cache()
        write_json(
            building / "direct_finetune_evaluation.json",
            {
                "status": "LABSE_DIRECT_FINETUNE_DEVELOPMENT_COMPLETE_AUDIT_TRUTH_SEALED",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "execution_authorization": authorization,
                "models": reports,
                "truth_read_counts": {
                    "train_labels": 1,
                    "development_labels": 1,
                    "development_qrels": 1,
                    "audit_a_labels_or_qrels": 0,
                    "audit_b_labels_or_qrels": 0,
                },
            },
        )
        files = [
            controls.file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        write_json(
            building / "manifest.json",
            {
                "status": "LABSE_DIRECT_FINETUNE_OUTPUT_AUDIT_TRUTH_SEALED",
                "producer_sha256": controls.sha256_file(Path(__file__)),
                "files": files,
                "audit_a_truth_reads": 0,
                "audit_b_truth_reads": 0,
            },
        )
        building.replace(output)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    return {
        "status": "LABSE_DIRECT_FINETUNE_COMPLETE_AUDIT_TRUTH_SEALED",
        "output_root": output.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-contract", "smoke", "run"))
    args = parser.parse_args()
    if args.command == "validate-contract":
        result = validate_contract()
    elif args.command == "smoke":
        result = smoke_runtime()
    else:
        result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
