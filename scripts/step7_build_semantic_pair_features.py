from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json"
POLICY_PATH = ROOT / "schema" / "step7_semantic_model_policy.json"
PROFILE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step3_seller_profiles.en_content_train_pool.jsonl",
    "zh_target_strict": ROOT / "reports" / "step3_seller_profiles.zh_target_strict.jsonl",
    "zh_target_aux": ROOT / "reports" / "step3_seller_profiles.zh_target_aux.jsonl",
}
PREVIEW_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step7_pair_feature_preview.en_content_train_pool.csv",
    "zh_target_strict": ROOT / "reports" / "step7_pair_feature_preview.zh_target_strict.csv",
    "zh_target_aux": ROOT / "reports" / "step7_pair_feature_preview.zh_target_aux.csv",
}
SENTENCE_SPLIT_RE = re.compile(r"(?:[。！？!?；;]+|\n+)")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def require_torch_and_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime
        raise SystemExit(
            "torch and transformers are required for Step 7 semantic feature extraction."
        ) from exc
    return torch, AutoTokenizer, AutoModel, AutoModelForSequenceClassification


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def index_rows_by_pair_uid(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        pair_uid = row.get("pair_uid")
        if pair_uid:
            indexed[pair_uid] = row
    return indexed


def iter_auto_map_refs(auto_map: dict | None):
    if not isinstance(auto_map, dict):
        return
    for value in auto_map.values():
        if isinstance(value, str):
            yield value
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield item


def module_path_from_auto_map_ref(ref: str) -> Path:
    local_ref = ref.split("--", 1)[1] if "--" in ref else ref
    module_name = local_ref.rsplit(".", 1)[0]
    return Path(*module_name.split(".")).with_suffix(".py")


def create_init_files_for_python_modules(model_dir: Path) -> None:
    py_files = [path for path in model_dir.rglob("*.py") if path.name != "__init__.py"]
    for py_path in py_files:
        parent = py_path.parent
        while parent != model_dir.parent and parent.is_relative_to(model_dir):
            init_path = parent / "__init__.py"
            if not init_path.exists():
                init_path.write_text("", encoding="utf-8")
            if parent == model_dir:
                break
            parent = parent.parent


def localize_remote_code_config(model_dir: Path, model_key: str) -> None:
    config_path = model_dir / "config.json"
    config = load_json(config_path)
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return

    remote_refs = [ref for ref in iter_auto_map_refs(auto_map) if "--" in ref]
    if not remote_refs:
        return

    missing_paths = []
    for ref in remote_refs:
        module_path = model_dir / module_path_from_auto_map_ref(ref)
        if not module_path.exists():
            missing_paths.append(str(module_path.relative_to(model_dir)))

    if missing_paths:
        missing_text = ", ".join(sorted(set(missing_paths)))
        raise SystemExit(
            f"Local Step 7 model directory for {model_key} is incomplete. "
            f"Missing trust_remote_code files: {missing_text}. "
            "Repair the local model first with:\n"
            "python scripts/step7_download_models.py "
            f"--embedding-model {model_key} --force\n"
            "If this is a reranker, replace --embedding-model with --reranker-model."
        )

    normalized = {}
    changed = False
    for key, value in auto_map.items():
        if isinstance(value, str):
            new_value = value.split("--", 1)[1] if "--" in value else value
            normalized[key] = new_value
            changed = changed or new_value != value
            continue
        if isinstance(value, (list, tuple)):
            new_values = []
            for item in value:
                if isinstance(item, str) and "--" in item:
                    new_values.append(item.split("--", 1)[1])
                    changed = True
                else:
                    new_values.append(item)
            normalized[key] = new_values
            continue
        normalized[key] = value

    if changed:
        config["auto_map"] = normalized
        write_json(config_path, config)
    create_init_files_for_python_modules(model_dir)


def choose_device(torch_module, preferred: list[str], explicit: str | None):
    candidates = [explicit] if explicit else preferred
    for name in candidates:
        if name == "cuda" and torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if name == "cpu":
            return torch_module.device("cpu")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def batch_iterable(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def mean_pool(last_hidden_state, attention_mask, torch_module):
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * expanded_mask).sum(dim=1)
    denom = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def build_profile_index(rows: list[dict], text_field: str) -> dict[str, str]:
    index = {}
    for row in rows:
        text = str(row.get(text_field, "") or "").strip()
        if text:
            index[row["seller_uid"]] = text
    return index


def normalize_text_segment(value: str) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    if CJK_RE.search(text):
        return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def split_profile_text_segments(text: str) -> list[str]:
    segments = []
    for segment in SENTENCE_SPLIT_RE.split(str(text or "")):
        cleaned = segment.strip()
        if cleaned:
            segments.append(cleaned)
    return segments


def build_profile_texts(rows: list[dict], text_field: str, cfg: dict | None) -> tuple[dict[str, str], dict]:
    base_texts = build_profile_index(rows, text_field)
    diagnostics = {
        "enabled": False,
        "max_sentence_seller_share": None,
        "min_normalized_char_length": None,
        "min_kept_segments": None,
        "seller_count": len(base_texts),
        "profile_changed_count": 0,
        "dropped_segment_count": 0,
        "top_dropped_segments": [],
    }
    if not cfg or not bool(cfg.get("enabled", False)):
        return base_texts, diagnostics

    max_share = float(cfg.get("max_sentence_seller_share", 0.01))
    min_normalized_char_length = int(cfg.get("min_normalized_char_length", 8))
    min_kept_segments = int(cfg.get("min_kept_segments", 1))
    diagnostics.update(
        {
            "enabled": True,
            "max_sentence_seller_share": round(max_share, 6),
            "min_normalized_char_length": min_normalized_char_length,
            "min_kept_segments": min_kept_segments,
        }
    )

    seller_segment_sets: dict[str, set[str]] = {}
    segment_surface_map: dict[str, str] = {}
    df_counter: Counter[str] = Counter()
    for seller_uid, text in base_texts.items():
        norms = set()
        for segment in split_profile_text_segments(text):
            norm = normalize_text_segment(segment)
            if len(norm) < min_normalized_char_length:
                continue
            norms.add(norm)
            segment_surface_map.setdefault(norm, segment)
        seller_segment_sets[seller_uid] = norms
        df_counter.update(norms)

    seller_count = max(len(base_texts), 1)
    filtered_texts: dict[str, str] = {}
    dropped_counter: Counter[str] = Counter()
    profile_changed_count = 0
    dropped_segment_count = 0
    for seller_uid, text in base_texts.items():
        kept_segments = []
        dropped_any = False
        for segment in split_profile_text_segments(text):
            norm = normalize_text_segment(segment)
            if len(norm) < min_normalized_char_length:
                kept_segments.append(segment)
                continue
            seller_share = df_counter.get(norm, 0) / seller_count
            if seller_share > max_share:
                dropped_counter[norm] += 1
                dropped_segment_count += 1
                dropped_any = True
                continue
            kept_segments.append(segment)
        if dropped_any and len(kept_segments) >= min_kept_segments:
            filtered_texts[seller_uid] = "\n".join(kept_segments)
            profile_changed_count += 1
        else:
            filtered_texts[seller_uid] = text

    diagnostics["profile_changed_count"] = int(profile_changed_count)
    diagnostics["dropped_segment_count"] = int(dropped_segment_count)
    diagnostics["top_dropped_segments"] = [
        {
            "segment_preview": segment_surface_map.get(norm, norm)[:180],
            "seller_df": int(df_counter.get(norm, 0)),
            "seller_share": round(df_counter.get(norm, 0) / seller_count, 6),
        }
        for norm, _count in dropped_counter.most_common(10)
    ]
    return filtered_texts, diagnostics


def pool_pair_uids(rows: list[dict]) -> tuple[list[str], list[str]]:
    sellers = []
    pair_uids = []
    seen = set()
    for row in rows:
        pair_uids.append(row["pair_uid"])
        for key in ("seller_uid_left", "seller_uid_right"):
            seller_uid = row[key]
            if seller_uid not in seen:
                seen.add(seller_uid)
                sellers.append(seller_uid)
    return sellers, pair_uids


def embedding_cache_paths(policy: dict, pool: str, model_key: str) -> tuple[Path, Path]:
    npy_path = ROOT / policy["output_files"]["embedding_cache_template"].format(pool=pool, model_key=model_key)
    manifest_path = ROOT / policy["output_files"]["embedding_manifest_template"].format(pool=pool, model_key=model_key)
    return npy_path, manifest_path


def resolve_local_model_dir(model_key: str, model_cfg: dict) -> Path:
    model_dir = ROOT / model_cfg["local_path"]
    if not model_dir.is_dir() or not (model_dir / "config.json").exists():
        raise SystemExit(
            f"Local Step 7 model directory is missing for {model_key}: {model_cfg['local_path']}. "
            "Run `python scripts/step7_download_models.py` before semantic feature extraction."
        )
    localize_remote_code_config(model_dir, model_key)
    return model_dir


def load_cached_embeddings(npy_path: Path, manifest_path: Path, seller_uids: list[str]) -> np.ndarray | None:
    if not npy_path.exists() or not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    if manifest.get("seller_uids") != seller_uids:
        return None
    return np.load(npy_path)


def save_cached_embeddings(
    npy_path: Path,
    manifest_path: Path,
    seller_uids: list[str],
    model_key: str,
    model_cfg: dict,
    embeddings: np.ndarray,
) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings)
    write_json(
        manifest_path,
        {
            "model_key": model_key,
            "model_repo_id": model_cfg["repo_id"],
            "model_local_path": model_cfg["local_path"],
            "feature_name": model_cfg["feature_name"],
            "seller_uids": seller_uids,
            "shape": list(embeddings.shape),
        },
    )


def encode_texts(model_key: str, model_cfg: dict, texts: list[str], device, torch_module, tokenizer_cls, model_cls) -> np.ndarray:
    model_dir = resolve_local_model_dir(model_key, model_cfg)
    tokenizer = tokenizer_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = model_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.to(device)
    model.eval()

    all_embeddings = []
    prefix = model_cfg.get("text_prefix", "")
    with torch_module.no_grad():
        for batch_texts in batch_iterable(texts, int(model_cfg["batch_size"])):
            prefixed = [prefix + text for text in batch_texts]
            encoded = tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=int(model_cfg["max_length"]),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = mean_pool(hidden, encoded["attention_mask"], torch_module)
            if model_cfg.get("normalize_embeddings", True):
                pooled = torch_module.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.detach().cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    del model
    if str(device) == "cuda":
        torch_module.cuda.empty_cache()
    return embeddings


def encode_or_load_embeddings(
    pool: str,
    seller_uids: list[str],
    profile_texts: dict[str, str],
    model_key: str,
    model_cfg: dict,
    policy: dict,
    device,
    torch_module,
    tokenizer_cls,
    model_cls,
    force_recompute: bool,
) -> np.ndarray:
    npy_path, manifest_path = embedding_cache_paths(policy, pool, model_key)
    if not force_recompute:
        cached = load_cached_embeddings(npy_path, manifest_path, seller_uids)
        if cached is not None:
            return cached
    texts = [profile_texts[seller_uid] for seller_uid in seller_uids]
    embeddings = encode_texts(model_key, model_cfg, texts, device, torch_module, tokenizer_cls, model_cls)
    save_cached_embeddings(npy_path, manifest_path, seller_uids, model_key, model_cfg, embeddings)
    return embeddings


def cosine_scores(rows: list[dict], seller_uids: list[str], embeddings: np.ndarray) -> list[float]:
    index = {seller_uid: idx for idx, seller_uid in enumerate(seller_uids)}
    scores = []
    for row in rows:
        left_vec = embeddings[index[row["seller_uid_left"]]]
        right_vec = embeddings[index[row["seller_uid_right"]]]
        scores.append(round(float(np.dot(left_vec, right_vec)), 6))
    return scores


def reranker_scores(rows: list[dict], profile_texts: dict[str, str], model_key: str, model_cfg: dict, device, torch_module, tokenizer_cls, reranker_cls) -> list[float]:
    model_dir = resolve_local_model_dir(model_key, model_cfg)
    tokenizer = tokenizer_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = reranker_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.to(device)
    model.eval()

    scores: list[float] = []
    with torch_module.no_grad():
        for batch_rows in batch_iterable(rows, int(model_cfg["batch_size"])):
            left_texts = [profile_texts[row["seller_uid_left"]] for row in batch_rows]
            right_texts = [profile_texts[row["seller_uid_right"]] for row in batch_rows]
            encoded = tokenizer(
                left_texts,
                right_texts,
                padding=True,
                truncation=True,
                max_length=int(model_cfg["max_length"]),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_scores = torch_module.sigmoid(logits.view(-1))
            else:
                batch_scores = torch_module.softmax(logits, dim=-1)[:, -1]
            scores.extend(round(float(item), 6) for item in batch_scores.detach().cpu().tolist())

    del model
    if str(device) == "cuda":
        torch_module.cuda.empty_cache()
    return scores


def finalize_rows(
    preview_rows: list[dict],
    schema: dict,
    feature_columns: dict[str, list[float]],
    existing_rows_by_pair_uid: dict[str, dict] | None = None,
) -> list[dict]:
    final_rows = []
    all_fields = schema["semantic_enriched_output_fields"]
    semantic_fields = set(schema["feature_views"]["future_multilingual_semantics"])
    for idx, preview in enumerate(preview_rows):
        existing = existing_rows_by_pair_uid.get(preview["pair_uid"], {}) if existing_rows_by_pair_uid else {}
        row = {}
        for field in all_fields:
            if field in semantic_fields:
                # Preserve previously built semantic columns when the current run only rebuilds
                # a subset of backbones/rerankers.
                row[field] = existing.get(field, "")
            else:
                row[field] = preview.get(field, existing.get(field, ""))
        for field, values in feature_columns.items():
            row[field] = values[idx]
        final_rows.append(row)
    return final_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step-7 semantic pair features with backbone and reranker scores.")
    parser.add_argument(
        "--pool",
        action="append",
        dest="pools",
        choices=sorted(PREVIEW_PATHS.keys()),
        help="Pool to process. Repeat to select multiple pools. Defaults to all pools.",
    )
    parser.add_argument(
        "--embedding-model",
        action="append",
        dest="embedding_models",
        help="Embedding model key from step7_semantic_model_policy.json. Repeat to select multiple models.",
    )
    parser.add_argument(
        "--reranker-model",
        action="append",
        dest="reranker_models",
        help="Reranker model key from step7_semantic_model_policy.json. Repeat to select multiple models.",
    )
    parser.add_argument("--device", default=None, help="Optional explicit device, e.g. cuda or cpu.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cached seller embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_module, tokenizer_cls, model_cls, reranker_cls = require_torch_and_transformers()
    schema = load_json(SCHEMA_PATH)
    policy = load_json(POLICY_PATH)
    pools = args.pools or list(PREVIEW_PATHS.keys())
    embedding_keys = args.embedding_models or list(policy["embedding_models"].keys())
    reranker_keys = args.reranker_models or list(policy["reranker_models"].keys())

    unknown_embedding = sorted(set(embedding_keys) - set(policy["embedding_models"].keys()))
    unknown_reranker = sorted(set(reranker_keys) - set(policy["reranker_models"].keys()))
    if unknown_embedding or unknown_reranker:
        raise SystemExit(
            f"Unknown semantic model keys: embedding={unknown_embedding}, reranker={unknown_reranker}"
        )

    device = choose_device(torch_module, policy["device_preference"], args.device)
    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "device": str(device),
        "selected_pools": pools,
        "selected_embedding_models": embedding_keys,
        "selected_reranker_models": reranker_keys,
        "selected_embedding_local_paths": {
            key: policy["embedding_models"][key]["local_path"] for key in embedding_keys
        },
        "selected_reranker_local_paths": {
            key: policy["reranker_models"][key]["local_path"] for key in reranker_keys
        },
        "pool_summaries": {},
    }

    for pool in pools:
        preview_rows = load_csv(PREVIEW_PATHS[pool])
        seller_uids, _ = pool_pair_uids(preview_rows)
        profile_rows = load_jsonl(PROFILE_PATHS[pool])
        profile_texts, deboilerplate_diagnostics = build_profile_texts(
            profile_rows,
            policy["profile_text_field"],
            policy.get("deboilerplate_profile_text"),
        )
        missing_profiles = [seller_uid for seller_uid in seller_uids if seller_uid not in profile_texts]
        if missing_profiles:
            raise ValueError(f"{pool} is missing {len(missing_profiles)} seller texts for semantic scoring")

        feature_columns: dict[str, list[float]] = {}
        embedding_cache_files = []
        for model_key in embedding_keys:
            model_cfg = policy["embedding_models"][model_key]
            embeddings = encode_or_load_embeddings(
                pool,
                seller_uids,
                profile_texts,
                model_key,
                model_cfg,
                policy,
                device,
                torch_module,
                tokenizer_cls,
                model_cls,
                args.force_recompute,
            )
            feature_name = model_cfg["feature_name"]
            feature_columns[feature_name] = cosine_scores(preview_rows, seller_uids, embeddings)
            npy_path, manifest_path = embedding_cache_paths(policy, pool, model_key)
            embedding_cache_files.append(str(npy_path.relative_to(ROOT)))
            embedding_cache_files.append(str(manifest_path.relative_to(ROOT)))

        for model_key in reranker_keys:
            model_cfg = policy["reranker_models"][model_key]
            feature_columns[model_cfg["feature_name"]] = reranker_scores(
                preview_rows,
                profile_texts,
                model_key,
                model_cfg,
                device,
                torch_module,
                tokenizer_cls,
                reranker_cls,
            )

        output_path = ROOT / policy["output_files"]["pair_feature_tables"][pool]
        existing_rows_by_pair_uid = {}
        if output_path.exists():
            existing_rows_by_pair_uid = index_rows_by_pair_uid(load_csv(output_path))
        final_rows = finalize_rows(preview_rows, schema, feature_columns, existing_rows_by_pair_uid)
        write_csv(output_path, final_rows, schema["semantic_enriched_output_fields"])

        preserved_semantic_columns = []
        if existing_rows_by_pair_uid:
            for field in schema["feature_views"]["future_multilingual_semantics"]:
                if field in feature_columns:
                    continue
                if any(existing_rows_by_pair_uid.get(row["pair_uid"], {}).get(field, "") not in ("", None) for row in preview_rows):
                    preserved_semantic_columns.append(field)
        summary["pool_summaries"][pool] = {
            "row_count": len(final_rows),
            "unique_seller_count": len(seller_uids),
            "embedding_cache_files": embedding_cache_files,
            "output_file": str(output_path.relative_to(ROOT)),
            "semantic_merge_mode": "preserve_existing_semantic_columns",
            "semantic_feature_columns": sorted(feature_columns.keys()),
            "preserved_semantic_feature_columns": preserved_semantic_columns,
            "deboilerplate_profile_text": deboilerplate_diagnostics,
        }

    write_json(ROOT / policy["output_files"]["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
