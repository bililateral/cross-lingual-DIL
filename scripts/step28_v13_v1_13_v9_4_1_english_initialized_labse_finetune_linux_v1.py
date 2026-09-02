#!/usr/bin/env python3
"""Train an English-supervised LaBSE encoder, then fine-tune it on Chinese.

Only English train labels and Chinese train/development supervision are used.
Audit-A/B truth is deliberately not addressable from this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_labse_direct_finetune_linux_v1 as direct
import step28_v13_v1_13_v9_4_1_transfer_claim_controls_v4 as controls


POLICY_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_english_initialized_labse_finetune_v1_policy.json"
)
EXPECTED_VERSION = "step28-v13-v1.13-v9.4.1-english-initialized-labse-finetune-v1"
MODEL_IDS = ("generic_init_base", "english_init_base")
SOURCE_FEATURE_COUNT = 6
TARGET_FEATURE_COUNT = 24
PAIR_TOP_K = 3


class EnglishInitializedFinetuneError(ValueError):
    """Raised when this development experiment violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(value, dtype="<f8"), allow_pickle=False)


def _verified_file(spec: Mapping[str, Any], label: str) -> Path:
    path = ROOT / str(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["size_bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise EnglishInitializedFinetuneError(f"Frozen input drift: {label}")
    return path


def load_policy() -> dict[str, Any]:
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if value.get("version") != EXPECTED_VERSION:
        raise EnglishInitializedFinetuneError("Policy identity drift")
    if value.get("canonical_self_hash") != canonical_self_hash(value):
        raise EnglishInitializedFinetuneError("Policy canonical self-hash drift")
    if (
        value.get("development_training_authorized") is not True
        or value.get("audit_a_prediction_authorized") is not False
        or value.get("audit_a_truth_authorized") is not False
        or value.get("audit_b_prediction_authorized") is not False
        or value.get("audit_b_truth_authorized") is not False
        or value["chinese_target"]["forbidden_splits"] != ["audit_a", "audit_b"]
    ):
        raise EnglishInitializedFinetuneError("Training or audit boundary drift")
    source = value["english_source"]
    text = value["text_input"]
    models = value["models"]
    decision = value["development_decision"]
    if (
        source["allowed_split"] != "train"
        or source["validation_or_test_labels_allowed"] is not False
        or source["source_trainable_features"]
        != "six_differentiable_labse_aggregates_only"
        or text["fields"] != ["title", "description"]
        or int(text["token_budget_including_special_tokens"]) != 256
        or text["whole_document_truncation_allowed"] is not False
        or text["all_seller_text_mappings_used_in_each_english_epoch"] is not True
        or text["exact_character_reconstruction_required"] is not True
        or int(text["top_k"]) != PAIR_TOP_K
        or set(models) != set(MODEL_IDS)
        or any(model["identity33"] is not False for model in models.values())
        or any(
            model["target_features"]
            != "legacy18_plus_six_differentiable_labse_aggregates"
            for model in models.values()
        )
        or decision["primary_metric"] != "pooled_average_precision"
        or decision["comparison"]
        != "english_init_base_minus_generic_init_base"
        or decision["continue_to_confirmation_only_if_strictly_positive"]
        is not True
        or decision["single_seed_result_is_confirmatory"] is not False
    ):
        raise EnglishInitializedFinetuneError("Scientific comparison contract drift")
    for label, spec in value["english_source"].items():
        if isinstance(spec, dict) and "path" in spec:
            _verified_file(spec, f"english_source.{label}")
    for label, spec in value["chinese_target"].items():
        if isinstance(spec, dict) and "path" in spec:
            _verified_file(spec, f"chinese_target.{label}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EnglishInitializedFinetuneError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise EnglishInitializedFinetuneError("JSONL row is not an object")
            yield value


def _component_class_weights(
    labels: np.ndarray, components: Sequence[str]
) -> np.ndarray:
    group_counts = Counter(zip(labels.tolist(), components))
    class_components = {
        label: len({component for observed, component in group_counts if observed == label})
        for label in (0, 1)
    }
    if any(class_components[label] <= 0 for label in (0, 1)):
        raise EnglishInitializedFinetuneError("English source lacks one label class")
    weights = np.asarray(
        [
            0.5
            / class_components[int(label)]
            / group_counts[(int(label), str(component))]
            for label, component in zip(labels, components)
        ],
        dtype="<f8",
    )
    if not np.isclose(weights[labels == 0].sum(), 0.5) or not np.isclose(
        weights[labels == 1].sum(), 0.5
    ):
        raise EnglishInitializedFinetuneError("English source weights do not balance")
    return weights


def load_english_source(policy: Mapping[str, Any]) -> dict[str, Any]:
    source = policy["english_source"]
    pairs_all = _read_csv(_verified_file(source["pair_manifest"], "pair_manifest"))
    pairs = [row for row in pairs_all if row["split_name"] == source["allowed_split"]]
    if len(pairs) != int(source["expected_pairs"]):
        raise EnglishInitializedFinetuneError("English train pair count drift")
    pair_uids = [row["pair_uid"] for row in pairs]
    if len(pair_uids) != len(set(pair_uids)):
        raise EnglishInitializedFinetuneError("Duplicate English train pair")

    label_rows = _read_csv(_verified_file(source["train_labels"], "train_labels"))
    label_index = {row["pair_uid"]: row for row in label_rows}
    if len(label_index) != len(label_rows):
        raise EnglishInitializedFinetuneError("Duplicate English train label row")
    if set(label_index) != set(pair_uids):
        raise EnglishInitializedFinetuneError("English labels do not match train pairs")
    label_values = {row["review_label"] for row in label_rows}
    if not label_values <= {"positive", "negative"} or label_values != {
        "positive",
        "negative",
    }:
        raise EnglishInitializedFinetuneError("Unexpected English label value")
    labels = np.asarray(
        [1 if label_index[uid]["review_label"] == "positive" else 0 for uid in pair_uids],
        dtype=np.int64,
    )
    if (
        int(labels.sum()) != int(source["expected_positive"])
        or int((labels == 0).sum()) != int(source["expected_negative"])
    ):
        raise EnglishInitializedFinetuneError("English label counts drift")
    components = tuple(row["component_id"] for row in pairs)
    if any(label_index[uid]["component_id"] != component for uid, component in zip(pair_uids, components)):
        raise EnglishInitializedFinetuneError("English component assignment drift")

    sellers = {
        seller
        for row in pairs
        for seller in (row["seller_uid_left"], row["seller_uid_right"])
    }
    if len(sellers) != int(source["expected_sellers"]):
        raise EnglishInitializedFinetuneError("English train seller count drift")
    seller_components: dict[str, set[str]] = defaultdict(set)
    component_pair_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(pairs):
        component = str(row["component_id"])
        component_pair_indices[component].append(index)
        seller_components[row["seller_uid_left"]].add(component)
        seller_components[row["seller_uid_right"]].add(component)
    if any(len(values) != 1 for values in seller_components.values()):
        raise EnglishInitializedFinetuneError("English seller crosses components")
    if len(component_pair_indices) != int(source["expected_components"]):
        raise EnglishInitializedFinetuneError("English component count drift")
    mappings = []
    required_text_uids = set()
    for row in _iter_jsonl(_verified_file(source["seller_text_index"], "seller_text_index")):
        if row["seller_uid"] in sellers:
            if row["split_name"] != source["allowed_split"]:
                raise EnglishInitializedFinetuneError("English seller text split drift")
            mappings.append(row)
            required_text_uids.add(str(row["text_uid"]))
    if len(mappings) != int(source["expected_text_mappings"]):
        raise EnglishInitializedFinetuneError("English seller-text mapping count drift")
    if len(required_text_uids) != int(source["expected_global_unique_texts"]):
        raise EnglishInitializedFinetuneError("English global unique text count drift")

    text_lookup = {}
    for row in _iter_jsonl(_verified_file(source["unique_texts"], "unique_texts")):
        uid = str(row["text_uid"])
        if uid in required_text_uids:
            if uid in text_lookup:
                raise EnglishInitializedFinetuneError("Duplicate English text payload")
            text = str(row["text"])
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != uid:
                raise EnglishInitializedFinetuneError("English text hash drift")
            text_lookup[uid] = text
    if set(text_lookup) != required_text_uids:
        raise EnglishInitializedFinetuneError("English text payload is incomplete")
    text_uids = tuple(sorted(required_text_uids, key=lambda value: value.encode("utf-8")))
    text_index = {uid: index for index, uid in enumerate(text_uids)}
    seller_fields: dict[str, dict[str, list[int]]] = {
        seller: {"title": [], "description": []} for seller in sellers
    }
    seen_mapping = set()
    for row in mappings:
        key = (row["seller_uid"], row["field_name"], row["text_uid"])
        if key in seen_mapping or row["field_name"] not in ("title", "description"):
            raise EnglishInitializedFinetuneError("English seller-text mapping drift")
        seen_mapping.add(key)
        seller_fields[row["seller_uid"]][row["field_name"]].append(
            text_index[row["text_uid"]]
        )
    frozen_fields = {
        seller: {
            field: tuple(sorted(indices))
            for field, indices in fields.items()
        }
        for seller, fields in seller_fields.items()
    }
    if any(not values[field] for values in frozen_fields.values() for field in ("title", "description")):
        raise EnglishInitializedFinetuneError("English seller lacks one text field")

    return {
        "pairs": tuple(pairs),
        "labels": labels,
        "components": components,
        "weights": _component_class_weights(labels, components),
        "text_uids": text_uids,
        "texts": tuple(text_lookup[uid] for uid in text_uids),
        "seller_fields": frozen_fields,
        "component_pair_indices": {
            component: tuple(indices)
            for component, indices in component_pair_indices.items()
        },
    }


def _chunk_all_texts(
    tokenizer: Any,
    texts: Sequence[str],
    budget: int,
    *,
    report_progress: bool = False,
) -> tuple[tuple[str, ...], ...]:
    result = []
    for index, text in enumerate(texts, start=1):
        chunks = direct.common.chunk_text_exact(tokenizer, text, budget)
        if "".join(chunks) != text:
            raise EnglishInitializedFinetuneError("English chunk reconstruction failed")
        result.append(tuple(chunks))
        if report_progress and (index % 2500 == 0 or index == len(texts)):
            print(f"英文文本分块：{index}/{len(texts)}", flush=True)
    return tuple(result)


def _encode_text_groups(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    chunks_by_text: Sequence[Sequence[str]],
    indices: Sequence[int],
    *,
    chunk_batch_size: int,
    device: Any,
    use_autocast: bool,
) -> Any:
    chunks = []
    slices = []
    for index in indices:
        start = len(chunks)
        chunks.extend(chunks_by_text[index])
        slices.append(slice(start, len(chunks)))
    vectors = direct._batch_sentence_embeddings(
        torch,
        encoder,
        tokenizer,
        chunks,
        batch_size=chunk_batch_size,
        device=device,
        use_autocast=use_autocast,
    )
    return torch.stack([direct.common.torch_unit_mean(vectors[item]) for item in slices])


def _all_text_embeddings(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    chunks_by_text: Sequence[Sequence[str]],
    *,
    chunk_batch_size: int,
    device: Any,
) -> Any:
    outputs = []
    group = []
    group_chunks = 0
    # Keep this grouping identical to _vjp_all_texts.  The English source
    # stage recomputes the same deterministic forward pass during the VJP;
    # changing group boundaries can change padding shapes and defeat that
    # equality even though each individual text is unchanged.
    limit = chunk_batch_size * 2
    for index, chunks in enumerate(chunks_by_text):
        if group and group_chunks + len(chunks) > limit:
            outputs.append(
                _encode_text_groups(
                    torch,
                    encoder,
                    tokenizer,
                    chunks_by_text,
                    group,
                    chunk_batch_size=chunk_batch_size,
                    device=device,
                    use_autocast=True,
                )
            )
            group = []
            group_chunks = 0
        group.append(index)
        group_chunks += len(chunks)
    if group:
        outputs.append(
            _encode_text_groups(
                torch,
                encoder,
                tokenizer,
                chunks_by_text,
                group,
                chunk_batch_size=chunk_batch_size,
                device=device,
                use_autocast=True,
            )
        )
    return torch.cat(outputs, dim=0)


def _pair_semantics(
    torch: Any,
    text_vectors: Any,
    pair: Mapping[str, str],
    seller_fields: Mapping[str, Mapping[str, Sequence[int]]],
) -> Any:
    left = {
        field: text_vectors[list(seller_fields[pair["seller_uid_left"]][field])]
        for field in ("title", "description")
    }
    right = {
        field: text_vectors[list(seller_fields[pair["seller_uid_right"]][field])]
        for field in ("title", "description")
    }
    return direct.common.torch_six_pair_aggregates(left, right, top_k=PAIR_TOP_K)


def _source_matrix(
    torch: Any,
    text_vectors: Any,
    source: Mapping[str, Any],
) -> Any:
    return torch.stack(
        [
            _pair_semantics(torch, text_vectors, pair, source["seller_fields"])
            for pair in source["pairs"]
        ]
    )


def _weighted_bce(torch: Any, logits: Any, labels: Any, weights: Any) -> Any:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return torch.sum(losses * weights)


def _full_encoder_contract(encoder: Any, label: str) -> dict[str, Any]:
    named_parameters = tuple(encoder.named_parameters())
    if not named_parameters or any(
        not parameter.requires_grad for _name, parameter in named_parameters
    ):
        raise EnglishInitializedFinetuneError(
            f"{label} does not expose every LaBSE parameter for fine-tuning"
        )
    count = sum(int(parameter.numel()) for _name, parameter in named_parameters)
    if count <= 0:
        raise EnglishInitializedFinetuneError(f"{label} parameter count is invalid")
    architecture = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in named_parameters
    ]
    return {
        "parameter_tensor_count": len(named_parameters),
        "parameter_count": count,
        "architecture_sha256": hashlib.sha256(
            canonical_json_bytes(architecture)
        ).hexdigest(),
        "all_parameters_trainable": True,
    }


def _vjp_all_texts(
    torch: Any,
    encoder: Any,
    tokenizer: Any,
    chunks_by_text: Sequence[Sequence[str]],
    text_gradient: Any,
    *,
    chunk_batch_size: int,
    device: Any,
) -> None:
    group = []
    group_chunks = 0
    limit = chunk_batch_size * 2

    def backward_group(current: Sequence[int]) -> None:
        vectors = _encode_text_groups(
            torch,
            encoder,
            tokenizer,
            chunks_by_text,
            current,
            chunk_batch_size=chunk_batch_size,
            device=device,
            use_autocast=True,
        )
        gradient = text_gradient[list(current)].to(device=device, dtype=torch.float32)
        torch.sum(vectors.float() * gradient).backward()

    for index, chunks in enumerate(chunks_by_text):
        if group and group_chunks + len(chunks) > limit:
            backward_group(group)
            group = []
            group_chunks = 0
        group.append(index)
        group_chunks += len(chunks)
    if group:
        backward_group(group)


def train_english_source(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    output: Path,
) -> tuple[Path, dict[str, Any]]:
    config = policy["source_optimization"]
    seed = int(config["seed"])
    direct.set_determinism(torch, seed)
    encoder, tokenizer = direct._load_encoder(policy, SentenceTransformer, "cuda:0")
    encoder_contract = _full_encoder_contract(
        encoder, "English source encoder"
    )
    encoder.eval()  # deterministic exact two-pass gradient; gradients remain enabled
    device = torch.device("cuda:0")
    chunks_by_text = _chunk_all_texts(
        tokenizer,
        source["texts"],
        int(policy["text_input"]["token_budget_including_special_tokens"]),
        report_progress=True,
    )
    chunk_count = sum(len(chunks) for chunks in chunks_by_text)
    labels = torch.from_numpy(source["labels"].astype(np.float32)).to(device)
    weights = torch.from_numpy(source["weights"].astype(np.float32)).to(device)
    # Only text-derived LaBSE features enter the English objective.  Since
    # only the encoder is transferred, legacy18 would give the source head a
    # dominant bypass around the representation that this experiment tests.
    head = torch.nn.Linear(SOURCE_FEATURE_COUNT, 1, bias=True).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)

    print(
        f"英文来源：401 对，{len(source['texts'])} 条全局唯一文本，{chunk_count} 个分块",
        flush=True,
    )
    with torch.no_grad():
        frozen_texts = _all_text_embeddings(
            torch,
            encoder,
            tokenizer,
            chunks_by_text,
            chunk_batch_size=int(config["chunk_batch_size"]),
            device=device,
        )
        frozen_matrix = _source_matrix(torch, frozen_texts, source).detach()
        warmup_initial_loss = float(
            _weighted_bce(
                torch, head(frozen_matrix).squeeze(1), labels, weights
            ).cpu()
        )
    warmup = torch.optim.AdamW(
        head.parameters(), lr=float(config["head_learning_rate"]), weight_decay=0.0
    )
    for _step in range(int(config["head_warmup_steps"])):
        warmup.zero_grad(set_to_none=True)
        loss = _weighted_bce(
            torch, head(frozen_matrix).squeeze(1), labels, weights
        )
        loss.backward()
        warmup.step()
    with torch.no_grad():
        warmup_final_loss = float(
            _weighted_bce(
                torch, head(frozen_matrix).squeeze(1), labels, weights
            ).cpu()
        )
        warmup_semantic_head_l2 = float(torch.linalg.vector_norm(head.weight[0]).cpu())
    del frozen_texts, frozen_matrix, warmup
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(config["encoder_learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            },
            {
                "params": head.parameters(),
                "lr": float(config["head_learning_rate"]),
                "weight_decay": 0.0,
            },
        ]
    )
    training_log = []
    optimizer_update_count = 0
    encoder_anchor_name = None
    encoder_anchor_before = None
    encoder_anchor_first_gradient_l2 = None
    component_ids = tuple(
        sorted(source["component_pair_indices"], key=lambda value: value.encode("utf-8"))
    )
    components_per_step = int(config["components_per_gradient_step"])
    steps_per_epoch = math.ceil(len(component_ids) / components_per_step)
    for epoch in range(int(config["epochs"])):
        generator = np.random.Generator(np.random.PCG64(seed + epoch))
        order = list(component_ids)
        generator.shuffle(order)
        epoch_losses = []
        epoch_gradient_norms = []
        visited_pairs = set()
        visited_sellers = set()
        for batch_number, start in enumerate(
            range(0, len(order), components_per_step), start=1
        ):
            component_batch = order[start : start + components_per_step]
            pair_indices = sorted(
                index
                for component in component_batch
                for index in source["component_pair_indices"][component]
            )
            batch_pairs = tuple(source["pairs"][index] for index in pair_indices)
            batch_sellers = {
                seller
                for pair in batch_pairs
                for seller in (pair["seller_uid_left"], pair["seller_uid_right"])
            }
            global_text_indices = sorted(
                {
                    index
                    for seller in batch_sellers
                    for field in ("title", "description")
                    for index in source["seller_fields"][seller][field]
                }
            )
            local_index = {
                global_index: local
                for local, global_index in enumerate(global_text_indices)
            }
            local_source = {
                "pairs": batch_pairs,
                "seller_fields": {
                    seller: {
                        field: tuple(
                            local_index[index]
                            for index in source["seller_fields"][seller][field]
                        )
                        for field in ("title", "description")
                    }
                    for seller in batch_sellers
                },
            }
            local_chunks = tuple(chunks_by_text[index] for index in global_text_indices)
            batch_labels = labels[pair_indices]
            # A constant scale preserves the global component/class-balanced
            # objective across the deterministic partition into optimizer steps.
            batch_weights = weights[pair_indices] * steps_per_epoch
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                current = _all_text_embeddings(
                    torch,
                    encoder,
                    tokenizer,
                    local_chunks,
                    chunk_batch_size=int(config["chunk_batch_size"]),
                    device=device,
                )
            leaf = current.detach().requires_grad_(True)
            matrix = _source_matrix(torch, leaf, local_source)
            loss = _weighted_bce(
                torch, head(matrix).squeeze(1), batch_labels, batch_weights
            )
            loss.backward()
            if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                raise EnglishInitializedFinetuneError("English text gradient is invalid")
            text_gradient = leaf.grad.detach().clone()
            loss_value = float(loss.detach().cpu())
            del matrix, leaf, current
            _vjp_all_texts(
                torch,
                encoder,
                tokenizer,
                local_chunks,
                text_gradient,
                chunk_batch_size=int(config["chunk_batch_size"]),
                device=device,
            )
            if encoder_anchor_name is None:
                for name, parameter in reversed(tuple(encoder.named_parameters())):
                    if parameter.grad is None:
                        continue
                    candidate_norm = float(
                        torch.linalg.vector_norm(parameter.grad.detach().float()).cpu()
                    )
                    if math.isfinite(candidate_norm) and candidate_norm > 0.0:
                        encoder_anchor_name = name
                        encoder_anchor_before = parameter.detach().float().cpu().clone()
                        encoder_anchor_first_gradient_l2 = candidate_norm
                        break
                if encoder_anchor_name is None:
                    raise EnglishInitializedFinetuneError(
                        "English loss produced no nonzero LaBSE parameter gradient"
                    )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    float(config["gradient_clip_norm"]),
                ).detach().cpu()
            )
            if not math.isfinite(gradient_norm):
                raise EnglishInitializedFinetuneError("English gradient norm is invalid")
            optimizer.step()
            optimizer_update_count += 1
            epoch_losses.append(loss_value)
            epoch_gradient_norms.append(gradient_norm)
            visited_pairs.update(pair_indices)
            visited_sellers.update(batch_sellers)
            del text_gradient
            torch.cuda.empty_cache()
            if batch_number % 5 == 0 or batch_number == steps_per_epoch:
                print(
                    f"英文微调第 {epoch + 1} 轮：批次 "
                    f"{batch_number}/{steps_per_epoch}",
                    flush=True,
                )
        if visited_pairs != set(range(len(source["pairs"]))) or visited_sellers != set(
            source["seller_fields"]
        ):
            raise EnglishInitializedFinetuneError("English epoch did not cover all data")
        epoch_report = {
            "epoch": epoch + 1,
            "optimizer_updates_after_epoch": optimizer_update_count,
            "mean_component_batch_loss": float(np.mean(epoch_losses)),
            "minimum_component_batch_loss": float(np.min(epoch_losses)),
            "maximum_component_batch_loss": float(np.max(epoch_losses)),
            "mean_pre_clip_gradient_norm": float(np.mean(epoch_gradient_norms)),
            "pair_coverage": len(visited_pairs),
            "seller_coverage": len(visited_sellers),
            "seller_text_mapping_coverage": sum(
                len(indices)
                for seller in visited_sellers
                for indices in source["seller_fields"][seller].values()
            ),
        }
        training_log.append(epoch_report)
        print(
            f"英文微调 {epoch + 1}/{config['epochs']}："
            f"loss={epoch_report['mean_component_batch_loss']:.6f}，"
            f"累计更新={optimizer_update_count}",
            flush=True,
        )

    if encoder_anchor_name is None or encoder_anchor_before is None:
        raise EnglishInitializedFinetuneError("English encoder anchor was not established")
    encoder_anchor_after = dict(encoder.named_parameters())[encoder_anchor_name]
    encoder_anchor_delta_l2 = float(
        torch.linalg.vector_norm(
            encoder_anchor_after.detach().float().cpu() - encoder_anchor_before
        )
    )
    if not math.isfinite(encoder_anchor_delta_l2) or encoder_anchor_delta_l2 <= 0.0:
        raise EnglishInitializedFinetuneError("English training did not update LaBSE parameters")

    encoder.eval()
    head.eval()
    with torch.no_grad():
        final_texts = _all_text_embeddings(
            torch,
            encoder,
            tokenizer,
            chunks_by_text,
            chunk_batch_size=int(config["chunk_batch_size"]),
            device=device,
        )
        final_matrix = _source_matrix(torch, final_texts, source)
        final_probabilities = (
            torch.sigmoid(head(final_matrix).squeeze(1)).double().cpu().numpy()
        )
    source_training_fit = {
        "scope": "english_training_fit_diagnostic_only_not_source_validation",
        "unweighted": controls.core.score_curve_metrics(
            source["labels"], final_probabilities
        )
        | controls.core.probabilistic_metrics(source["labels"], final_probabilities),
        "component_and_class_balanced": controls.core.score_curve_metrics(
            source["labels"], final_probabilities, source["weights"]
        )
        | controls.core.probabilistic_metrics(
            source["labels"], final_probabilities, source["weights"]
        ),
    }
    del final_texts, final_matrix, final_probabilities
    torch.cuda.empty_cache()

    source_root = output / "english_source"
    encoder_path = source_root / "encoder"
    encoder.save_pretrained(str(encoder_path))
    torch.save(head.state_dict(), source_root / "head_state.pt")
    import step7_v4_common as step7_common

    trained_encoder_payload = step7_common.model_content_fingerprint(encoder_path)
    if trained_encoder_payload["content_sha256"] == policy["labse_model"]["content_sha256"]:
        raise EnglishInitializedFinetuneError("English training did not change LaBSE bytes")
    audit = {
        "pair_count": len(source["pairs"]),
        "positive_count": int(source["labels"].sum()),
        "negative_count": int((source["labels"] == 0).sum()),
        "seller_count": len(source["seller_fields"]),
        "seller_text_mapping_count": sum(
            len(indices)
            for fields in source["seller_fields"].values()
            for indices in fields.values()
        ),
        "global_unique_text_count": len(source["texts"]),
        "chunk_count": chunk_count,
        "component_count": len(component_ids),
        "encoder_contract": encoder_contract,
        "epochs": int(config["epochs"]),
        "optimizer_update_count": optimizer_update_count,
        "all_seller_text_mappings_used_per_epoch": True,
        "whole_document_truncation_count": 0,
        "validation_or_test_label_reads": 0,
        "head_warmup": {
            "steps": int(config["head_warmup_steps"]),
            "initial_balanced_loss": warmup_initial_loss,
            "final_balanced_loss": warmup_final_loss,
            "semantic6_weight_l2": warmup_semantic_head_l2,
        },
        "trained_encoder_payload": trained_encoder_payload,
        "encoder_update_proof": {
            "anchor_parameter": encoder_anchor_name,
            "first_gradient_l2": encoder_anchor_first_gradient_l2,
            "final_parameter_delta_l2": encoder_anchor_delta_l2,
        },
        "training_fit_diagnostic": source_training_fit,
        "training_log": training_log,
    }
    write_json(source_root / "training.json", audit)
    del encoder, head
    torch.cuda.empty_cache()
    return encoder_path, audit


def _load_target_encoder(
    policy: Mapping[str, Any], SentenceTransformer: Any, initialization: str
) -> tuple[Any, Any]:
    encoder = SentenceTransformer(initialization, device="cuda:0", local_files_only=True)
    if int(encoder.max_seq_length) != int(policy["labse_model"]["native_max_sequence_length"]):
        raise EnglishInitializedFinetuneError("Target encoder sequence length drift")
    if getattr(encoder, "default_prompt_name", None) is not None:
        encoder.default_prompt_name = None
    return encoder, encoder.tokenizer


def train_target_model(
    torch: Any,
    SentenceTransformer: Any,
    policy: Mapping[str, Any],
    model_id: str,
    initialization: str,
    text_index: Mapping[str, Any],
    train: Mapping[str, Any],
    development: Mapping[str, Any],
    train_labels: np.ndarray,
    train_rows: Mapping[str, np.ndarray],
    development_rows: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], Any, Any, np.ndarray, np.ndarray]:
    config = policy["target_optimization"]
    seed = int(config["seed"])
    direct.set_determinism(torch, seed)
    encoder, tokenizer = _load_target_encoder(policy, SentenceTransformer, initialization)
    encoder_contract = _full_encoder_contract(
        encoder, f"{model_id} target encoder"
    )
    # Loading two byte-different checkpoints may consume a different number
    # of random values.  Reset after loading so target dropout and every later
    # stochastic operation are matched; encoder weights remain the sole
    # intended difference between the two target arms.
    direct.set_determinism(torch, seed)
    device = torch.device("cuda:0")
    selected_worlds = tuple(
        sorted(
            train_rows,
            key=lambda value: (
                hashlib.sha256(value.encode("utf-8")).digest(),
                value.encode("utf-8"),
            ),
        )
    )
    if len(selected_worlds) != int(config["world_count"]):
        raise EnglishInitializedFinetuneError("Chinese target world count drift")
    legacy_mean, legacy_scale = direct._standardizer(train["base24"][:, :18])
    train_numeric = np.ascontiguousarray(
        (train["base24"][:, :18] - legacy_mean) / legacy_scale, dtype="<f8"
    )
    development_numeric = np.ascontiguousarray(
        (development["base24"][:, :18] - legacy_mean) / legacy_scale, dtype="<f8"
    )
    head = torch.nn.Linear(TARGET_FEATURE_COUNT, 1, bias=True).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.constant_(head.bias, math.log(20.0 / 358.0))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(config["encoder_learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            },
            {
                "params": head.parameters(),
                "lr": float(config["head_learning_rate"]),
                "weight_decay": 0.0,
            },
        ]
    )
    chunk_cache: dict[str, tuple[str, ...]] = {}
    training_log = []
    optimizer_update_count = 0
    encoder_anchor_name = None
    encoder_anchor_before = None
    encoder_anchor_first_gradient_l2 = None
    encoder.train()
    head.train()
    for epoch in range(int(config["epochs"])):
        generator = np.random.Generator(np.random.PCG64(seed + epoch))
        order = list(selected_worlds)
        generator.shuffle(order)
        epoch_losses = []
        width = int(config["worlds_per_gradient_step"])
        steps_per_epoch = math.ceil(len(order) / width)
        for batch_number, start in enumerate(range(0, len(order), width), start=1):
            group = order[start : start + width]
            optimizer.zero_grad(set_to_none=True)
            for world_uid in group:
                indices = train_rows[world_uid]
                semantics = direct.differentiable_world_semantics(
                    torch,
                    encoder,
                    tokenizer,
                    text_index["train"][world_uid],
                    [train["seller_uid_left"][index] for index in indices],
                    [train["seller_uid_right"][index] for index in indices],
                    chunk_cache,
                    token_budget=int(
                        policy["text_input"]["token_budget_including_special_tokens"]
                    ),
                    chunk_batch_size=int(config["chunk_batch_size"]),
                    device=device,
                    use_autocast=True,
                )
                numeric = torch.from_numpy(train_numeric[indices]).to(
                    device=device, dtype=torch.float32
                )
                labels = torch.from_numpy(train_labels[indices].astype(np.float32)).to(device)
                logits = head(torch.cat((numeric, semantics.float()), dim=1)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                (loss / len(group)).backward()
                epoch_losses.append(float(loss.detach().cpu()))
            if encoder_anchor_name is None:
                for name, parameter in reversed(tuple(encoder.named_parameters())):
                    if parameter.grad is None:
                        continue
                    candidate_norm = float(
                        torch.linalg.vector_norm(parameter.grad.detach().float()).cpu()
                    )
                    if math.isfinite(candidate_norm) and candidate_norm > 0.0:
                        encoder_anchor_name = name
                        encoder_anchor_before = parameter.detach().float().cpu().clone()
                        encoder_anchor_first_gradient_l2 = candidate_norm
                        break
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    float(config["gradient_clip_norm"]),
                ).detach().cpu()
            )
            if not math.isfinite(gradient_norm):
                raise EnglishInitializedFinetuneError(
                    f"{model_id} Chinese gradient norm is invalid"
                )
            optimizer.step()
            optimizer_update_count += 1
            if batch_number % 5 == 0 or batch_number == steps_per_epoch:
                print(
                    f"{model_id} 第 {epoch + 1} 轮：批次 "
                    f"{batch_number}/{steps_per_epoch}",
                    flush=True,
                )
        mean_loss = float(np.mean(epoch_losses))
        training_log.append({"epoch": epoch + 1, "mean_world_loss": mean_loss})
        print(
            f"{model_id} 中文微调 {epoch + 1}/{config['epochs']}：loss={mean_loss:.6f}",
            flush=True,
        )

    if encoder_anchor_name is None or encoder_anchor_before is None:
        raise EnglishInitializedFinetuneError(
            f"{model_id} Chinese loss produced no nonzero LaBSE parameter gradient"
        )
    encoder_anchor_after = dict(encoder.named_parameters())[encoder_anchor_name]
    encoder_anchor_delta_l2 = float(
        torch.linalg.vector_norm(
            encoder_anchor_after.detach().float().cpu() - encoder_anchor_before
        )
    )
    if not math.isfinite(encoder_anchor_delta_l2) or encoder_anchor_delta_l2 <= 0.0:
        raise EnglishInitializedFinetuneError(
            f"{model_id} Chinese training did not update LaBSE parameters"
        )

    predictions = np.empty(len(development["world_uids"]), dtype="<f8")
    encoder.eval()
    head.eval()
    with torch.no_grad():
        development_order = sorted(
            development_rows, key=lambda value: value.encode("utf-8")
        )
        for world_number, world_uid in enumerate(development_order, start=1):
            indices = development_rows[world_uid]
            semantics = direct.differentiable_world_semantics(
                torch,
                encoder,
                tokenizer,
                text_index["development"][world_uid],
                [development["seller_uid_left"][index] for index in indices],
                [development["seller_uid_right"][index] for index in indices],
                chunk_cache,
                token_budget=int(
                    policy["text_input"]["token_budget_including_special_tokens"]
                ),
                chunk_batch_size=int(config["chunk_batch_size"]),
                device=device,
                use_autocast=True,
            )
            numeric = torch.from_numpy(development_numeric[indices]).to(
                device=device, dtype=torch.float32
            )
            logits = head(torch.cat((numeric, semantics.float()), dim=1)).squeeze(1)
            predictions[indices] = torch.sigmoid(logits).double().cpu().numpy()
            if world_number % 100 == 0 or world_number == len(development_order):
                print(
                    f"{model_id} 开发集推理：{world_number}/{len(development_order)} 世界",
                    flush=True,
                )
    if not np.isfinite(predictions).all():
        raise EnglishInitializedFinetuneError("Chinese development predictions are invalid")
    return (
        predictions,
        {
            "model_id": model_id,
            "world_count": len(selected_worlds),
            "encoder_contract": encoder_contract,
            "epochs": int(config["epochs"]),
            "optimizer_updates": optimizer_update_count,
            "encoder_update_proof": {
                "anchor_parameter": encoder_anchor_name,
                "first_gradient_l2": encoder_anchor_first_gradient_l2,
                "final_parameter_delta_l2": encoder_anchor_delta_l2,
            },
            "training_log": training_log,
            "whole_document_truncation_count": 0,
        },
        encoder,
        head,
        legacy_mean,
        legacy_scale,
    )


def development_comparison(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    generic = reports["generic_init_base"]["development"]
    english = reports["english_init_base"]["development"]

    def deltas(section: str) -> dict[str, float]:
        return {
            key: float(english[section][key] - generic[section][key])
            for key in english[section]
            if isinstance(english[section][key], (int, float))
            and isinstance(generic[section].get(key), (int, float))
        }

    threshold_sections = ("raw_rows", "world_equal_confusion")
    primary_delta = float(
        english["pooled"]["average_precision"]
        - generic["pooled"]["average_precision"]
    )
    return {
        "comparison": "english_init_base_minus_generic_init_base",
        "primary_metric": "pooled_average_precision",
        "primary_delta": primary_delta,
        "continue_to_confirmation": primary_delta > 0.0,
        "single_seed_result_is_confirmatory": False,
        "pooled_deltas": deltas("pooled"),
        "world_equal_sensitivity_deltas": deltas("world_equal_sensitivity"),
        "threshold_metric_deltas": {
            section: {
                key: float(
                    english["threshold"][section][key]
                    - generic["threshold"][section][key]
                )
                for key in english["threshold"][section]
            }
            for section in threshold_sections
        },
        "retrieval_deltas": deltas("retrieval"),
        "lower_is_better_metrics": ["brier", "log_loss"],
    }


def validate_contract() -> dict[str, Any]:
    policy = load_policy()
    source = load_english_source(policy)
    return {
        "status": "PASSED_ENGLISH_INITIALIZED_FINETUNE_CONTRACT_NO_AUDIT_TRUTH",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "english_train_pairs": len(source["pairs"]),
        "english_positive": int(source["labels"].sum()),
        "english_negative": int((source["labels"] == 0).sum()),
        "english_sellers": len(source["seller_fields"]),
        "english_components": len(source["component_pair_indices"]),
        "english_global_unique_texts": len(source["texts"]),
        "english_validation_or_test_label_reads": 0,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def smoke_runtime() -> dict[str, Any]:
    """Exercise the two-pass text-gradient path without formal supervision."""

    policy = load_policy()
    torch, SentenceTransformer, _transformers = direct.require_gpu_runtime(policy)
    direct.common.verify_labse_payload(policy)
    direct.set_determinism(torch, int(policy["source_optimization"]["seed"]))
    encoder, tokenizer = direct._load_encoder(policy, SentenceTransformer, "cuda:0")
    encoder.eval()
    device = torch.device("cuda:0")
    texts = (
        " ".join(["Fast dispatch with plain packaging and clear order notes."] * 180),
        "Product details are listed carefully. Contact is not included.",
        "Orders are prepared daily with simple packaging.",
        "Please read the item description before placing an order.",
    )
    chunks_by_text = _chunk_all_texts(
        tokenizer,
        texts,
        int(policy["text_input"]["token_budget_including_special_tokens"]),
    )
    if max(len(chunks) for chunks in chunks_by_text) <= 1:
        raise EnglishInitializedFinetuneError(
            "Smoke did not exercise a multi-chunk English text"
        )
    with torch.no_grad():
        current = _all_text_embeddings(
            torch,
            encoder,
            tokenizer,
            chunks_by_text,
            chunk_batch_size=4,
            device=device,
        )
    leaf = current.detach().requires_grad_(True)
    seller_fields = {
        "left": {"title": (0,), "description": (1,)},
        "right": {"title": (2,), "description": (3,)},
    }
    pair = {"seller_uid_left": "left", "seller_uid_right": "right"}
    semantics = _pair_semantics(torch, leaf, pair, seller_fields)
    head = torch.nn.Linear(SOURCE_FEATURE_COUNT, 1, bias=True).to(device)
    torch.nn.init.constant_(head.weight, 0.01)
    torch.nn.init.zeros_(head.bias)
    features = semantics.float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(features).squeeze(), torch.tensor(1.0, device=device)
    )
    loss.backward()
    if leaf.grad is None or float(leaf.grad.abs().sum().cpu()) <= 0.0:
        raise EnglishInitializedFinetuneError("Smoke produced no text gradient")
    encoder.zero_grad(set_to_none=True)
    _vjp_all_texts(
        torch,
        encoder,
        tokenizer,
        chunks_by_text,
        leaf.grad.detach(),
        chunk_batch_size=4,
        device=device,
    )
    encoder_gradient = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )
    if not math.isfinite(encoder_gradient) or encoder_gradient <= 0.0:
        raise EnglishInitializedFinetuneError("Smoke did not update LaBSE gradient")
    anchor_name = None
    two_pass_gradient = None
    for name, parameter in reversed(tuple(encoder.named_parameters())):
        if parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0.0:
            anchor_name = name
            two_pass_gradient = parameter.grad.detach().float().cpu().clone()
            break
    if anchor_name is None or two_pass_gradient is None:
        raise EnglishInitializedFinetuneError("Smoke has no two-pass gradient anchor")

    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    direct_vectors = _all_text_embeddings(
        torch,
        encoder,
        tokenizer,
        chunks_by_text,
        chunk_batch_size=4,
        device=device,
    )
    direct_semantics = _pair_semantics(
        torch, direct_vectors, pair, seller_fields
    ).float()
    direct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(direct_semantics).squeeze(), torch.tensor(1.0, device=device)
    )
    direct_loss.backward()
    direct_gradient = dict(encoder.named_parameters())[anchor_name].grad
    if direct_gradient is None:
        raise EnglishInitializedFinetuneError("Direct smoke lost its gradient anchor")
    direct_gradient = direct_gradient.detach().float().cpu()
    maximum_gradient_difference = float(
        torch.max(torch.abs(two_pass_gradient - direct_gradient))
    )
    relative_gradient_l2_difference = float(
        torch.linalg.vector_norm(two_pass_gradient - direct_gradient)
        / torch.clamp(torch.linalg.vector_norm(direct_gradient), min=1e-12)
    )
    if not torch.allclose(
        two_pass_gradient, direct_gradient, rtol=1e-4, atol=1e-7
    ):
        raise EnglishInitializedFinetuneError(
            "Two-pass English gradient does not match direct autograd"
        )
    return {
        "status": "PASSED_TWO_PASS_LABSE_GRADIENT_SMOKE_NO_FORMAL_TRUTH",
        "loss": float(loss.detach().cpu()),
        "encoder_gradient_l1": encoder_gradient,
        "gradient_equivalence": {
            "anchor_parameter": anchor_name,
            "maximum_absolute_difference": maximum_gradient_difference,
            "relative_l2_difference": relative_gradient_l2_difference,
        },
        "english_train_label_reads": 0,
        "chinese_train_or_development_truth_reads": 0,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def run() -> dict[str, Any]:
    policy = load_policy()
    torch, SentenceTransformer, _transformers = direct.require_gpu_runtime(policy)
    direct.common.verify_labse_payload(policy)
    output = ROOT / str(policy["output_root"])
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise EnglishInitializedFinetuneError("Output already exists")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        print("读取并核对英文 401 对训练边界", flush=True)
        source = load_english_source(policy)
        english_encoder, source_audit = train_english_source(
            torch, SentenceTransformer, policy, source, building
        )
        (
            _execution,
            _v3_policy,
            train,
            development,
            train_labels,
            development_labels,
            relevance,
        ) = controls._load_inputs(policy)
        direct_policy = direct.common.load_policy()
        print("构建中文训练／开发完整文本索引", flush=True)
        text_index = direct._build_text_indices(direct_policy)
        train_rows = direct._world_row_indices(train)
        development_rows = direct._world_row_indices(development)
        initializations = {
            "generic_init_base": str(ROOT / str(policy["labse_model"]["path"])),
            "english_init_base": str(english_encoder),
        }
        reports = {}
        for model_id in MODEL_IDS:
            predictions, audit, encoder, head, legacy_mean, legacy_scale = train_target_model(
                torch,
                SentenceTransformer,
                policy,
                model_id,
                initializations[model_id],
                text_index,
                train,
                development,
                train_labels,
                train_rows,
                development_rows,
            )
            threshold, evaluation = controls._evaluate_model(
                predictions, development_labels, development, relevance
            )
            root = building / "chinese_target" / model_id
            save_array(root / "development_probabilities.npy", predictions)
            save_array(root / "legacy_mean.npy", legacy_mean)
            save_array(root / "legacy_scale.npy", legacy_scale)
            encoder.save_pretrained(str(root / "encoder"))
            torch.save(head.state_dict(), root / "head_state.pt")
            report = {
                "training": audit,
                "development_threshold": float(threshold),
                "development": evaluation,
            }
            write_json(root / "training_and_evaluation.json", report)
            reports[model_id] = report
            del encoder, head
            torch.cuda.empty_cache()
        generic_training = reports["generic_init_base"]["training"]
        english_training = reports["english_init_base"]["training"]
        if (
            generic_training["encoder_contract"]
            != english_training["encoder_contract"]
            or source_audit["encoder_contract"]
            != english_training["encoder_contract"]
            or generic_training["world_count"] != english_training["world_count"]
            or generic_training["epochs"] != english_training["epochs"]
            or generic_training["optimizer_updates"]
            != english_training["optimizer_updates"]
        ):
            raise EnglishInitializedFinetuneError(
                "Chinese target arms are not capacity/schedule matched"
            )
        summary = {
            "status": "ENGLISH_INITIALIZED_LABSE_FINETUNE_DEVELOPMENT_COMPLETE_AUDIT_TRUTH_SEALED",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "english_source": source_audit,
            "models": reports,
            "development_primary_comparison": development_comparison(reports),
            "truth_read_counts": {
                "english_train_labels": 1,
                "english_validation_or_test_labels": 0,
                "chinese_train_labels": 1,
                "chinese_development_labels": 1,
                "chinese_development_qrels": 1,
                "audit_a_labels_or_qrels": 0,
                "audit_b_labels_or_qrels": 0,
            },
        }
        write_json(building / "evaluation.json", summary)
        files = [
            controls.file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        write_json(
            building / "manifest.json",
            {
                "status": "ENGLISH_INITIALIZED_LABSE_FINETUNE_OUTPUT_AUDIT_TRUTH_SEALED",
                "producer_sha256": sha256_file(Path(__file__)),
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
        "status": "ENGLISH_INITIALIZED_LABSE_FINETUNE_COMPLETE_AUDIT_TRUTH_SEALED",
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
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
