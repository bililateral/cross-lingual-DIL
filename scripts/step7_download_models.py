from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step7_semantic_model_policy.json"
DOWNLOAD_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.model",
    "added_tokens.json",
    "model.safetensors",
    "pytorch_model.bin",
    "*.py",
]
REMOTE_CODE_ALLOW_PATTERNS = [
    "*.py",
]
REMOTE_CODE_MANIFEST_NAME = ".step7_remote_code_manifest.json"


def require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime
        raise SystemExit(
            "huggingface_hub is required for Step 7 model download. Install it, then rerun this script."
        ) from exc
    return snapshot_download


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all Step 7 backbone and reranker models into local relative paths."
    )
    parser.add_argument(
        "--embedding-model",
        action="append",
        dest="embedding_models",
        help="Embedding model key from step7_semantic_model_policy.json. Repeat to limit downloads.",
    )
    parser.add_argument(
        "--reranker-model",
        action="append",
        dest="reranker_models",
        help="Reranker model key from step7_semantic_model_policy.json. Repeat to limit downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download model files even if the local directory already exists.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN if set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel download workers passed to snapshot_download. Use 1 for unstable networks.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="How many times to retry a failed model download before marking it as failed.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=8.0,
        help="Seconds to wait between model-level retries.",
    )
    return parser.parse_args()


def validate_selection(selected: list[str], allowed: set[str], model_type: str) -> list[str]:
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise SystemExit(f"Unknown {model_type} model keys: {unknown}")
    return selected


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


def external_remote_code_dependencies(config: dict) -> dict[str, list[str]]:
    dependencies: dict[str, set[str]] = {}
    for ref in iter_auto_map_refs(config.get("auto_map")):
        if "--" not in ref:
            continue
        repo_id, _ = ref.split("--", 1)
        dependencies.setdefault(repo_id, set()).add(str(module_path_from_auto_map_ref(ref)))
    return {repo_id: sorted(paths) for repo_id, paths in dependencies.items()}


def normalize_auto_map(config: dict) -> bool:
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return False
    changed = False
    normalized = {}
    for key, value in auto_map.items():
        if isinstance(value, str):
            new_value = value.split("--", 1)[1] if "--" in value else value
            changed = changed or new_value != value
            normalized[key] = new_value
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
    return changed


def remote_code_manifest_path(model_dir: Path) -> Path:
    return model_dir / REMOTE_CODE_MANIFEST_NAME


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


def remote_code_complete(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False
    config = load_json(config_path)
    for ref in iter_auto_map_refs(config.get("auto_map")):
        if "--" in ref:
            return False
        module_path = model_dir / module_path_from_auto_map_ref(ref)
        if not module_path.exists():
            return False
    return True


def is_download_complete(model_dir: Path) -> bool:
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        return False
    weight_patterns = ("model.safetensors", "pytorch_model.bin")
    for pattern in weight_patterns:
        if any(model_dir.rglob(pattern)):
            return remote_code_complete(model_dir)
    return False


def ensure_remote_code_localized(
    snapshot_download,
    model_dir: Path,
    token: str | None,
    force: bool,
    max_workers: int,
) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {"dependency_repos": [], "localized_auto_map": False}

    config = load_json(config_path)
    dependencies = external_remote_code_dependencies(config)
    manifest_path = remote_code_manifest_path(model_dir)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        for row in manifest.get("dependency_repos", []):
            repo_id = row.get("repo_id")
            module_paths = row.get("module_paths", [])
            if repo_id:
                dependencies.setdefault(repo_id, [])
                dependencies[repo_id] = sorted(set(dependencies[repo_id]) | set(module_paths))

    downloaded_dependency_repos = []
    for repo_id in sorted(dependencies):
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            token=token,
            force_download=force,
            max_workers=max_workers,
            allow_patterns=REMOTE_CODE_ALLOW_PATTERNS,
        )
        downloaded_dependency_repos.append(repo_id)

    localized_auto_map = normalize_auto_map(config)
    if localized_auto_map:
        write_json(config_path, config)

    if downloaded_dependency_repos or localized_auto_map:
        write_json(
            manifest_path,
            {
                "dependency_repos": [
                    {
                        "repo_id": repo_id,
                        "module_paths": dependencies[repo_id],
                    }
                    for repo_id in sorted(dependencies)
                ],
                "localized_auto_map": localized_auto_map or remote_code_complete(model_dir),
            },
        )
    if downloaded_dependency_repos:
        create_init_files_for_python_modules(model_dir)

    return {
        "dependency_repos": downloaded_dependency_repos,
        "localized_auto_map": localized_auto_map,
    }


def download_one_model(
    snapshot_download,
    model_type: str,
    model_key: str,
    model_cfg: dict,
    token: str | None,
    force: bool,
    max_workers: int,
    retries: int,
    retry_wait_seconds: float,
) -> dict:
    local_dir = ROOT / model_cfg["local_path"]
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    status = "reused_existing"
    if force or not is_download_complete(local_dir):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                snapshot_download(
                    repo_id=model_cfg["repo_id"],
                    local_dir=str(local_dir),
                    token=token,
                    force_download=force,
                    max_workers=max_workers,
                    allow_patterns=DOWNLOAD_ALLOW_PATTERNS,
                )
                remote_code_info = ensure_remote_code_localized(
                    snapshot_download,
                    local_dir,
                    token,
                    force,
                    max_workers,
                )
                status = "downloaded"
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - depends on runtime/network
                last_error = exc
                if attempt < retries:
                    time.sleep(retry_wait_seconds)
        if last_error is not None:
            return {
                "model_type": model_type,
                "model_key": model_key,
                "repo_id": model_cfg["repo_id"],
                "local_path": model_cfg["local_path"],
                "status": "failed",
                "error_type": type(last_error).__name__,
                "error_message": str(last_error),
            }
    else:
        remote_code_info = ensure_remote_code_localized(
            snapshot_download,
            local_dir,
            token,
            force,
            max_workers,
        )
    record = {
        "model_type": model_type,
        "model_key": model_key,
        "repo_id": model_cfg["repo_id"],
        "local_path": model_cfg["local_path"],
        "status": status,
    }
    dependency_repos = remote_code_info.get("dependency_repos", [])
    if dependency_repos:
        record["remote_code_dependency_repos"] = dependency_repos
    if remote_code_info.get("localized_auto_map"):
        record["localized_auto_map"] = True
    if not is_download_complete(local_dir):
        record["status"] = "incomplete"
    return record


def main() -> None:
    args = parse_args()
    snapshot_download = require_huggingface_hub()
    policy = load_json(POLICY_PATH)

    embedding_keys = args.embedding_models or list(policy["embedding_models"].keys())
    reranker_keys = args.reranker_models or list(policy["reranker_models"].keys())
    embedding_keys = validate_selection(embedding_keys, set(policy["embedding_models"].keys()), "embedding")
    reranker_keys = validate_selection(reranker_keys, set(policy["reranker_models"].keys()), "reranker")

    downloaded_models = []
    for model_key in embedding_keys:
        downloaded_models.append(
            download_one_model(
                snapshot_download,
                "embedding",
                model_key,
                policy["embedding_models"][model_key],
                args.token,
                args.force,
                args.max_workers,
                args.retries,
                args.retry_wait_seconds,
            )
        )
    for model_key in reranker_keys:
        downloaded_models.append(
            download_one_model(
                snapshot_download,
                "reranker",
                model_key,
                policy["reranker_models"][model_key],
                args.token,
                args.force,
                args.max_workers,
                args.retries,
                args.retry_wait_seconds,
            )
        )

    failed_models = [row["model_key"] for row in downloaded_models if row["status"] in {"failed", "incomplete"}]
    summary = {
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "model_root": policy["model_root"],
        "selected_embedding_models": embedding_keys,
        "selected_reranker_models": reranker_keys,
        "max_workers": args.max_workers,
        "retries": args.retries,
        "downloaded_models": downloaded_models,
        "failed_models": failed_models,
        "allow_patterns": DOWNLOAD_ALLOW_PATTERNS,
    }
    write_json(ROOT / policy["output_files"]["download_summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed_models:
        raise SystemExit(f"Step 7 model download incomplete. Failed model keys: {failed_models}")


if __name__ == "__main__":
    main()
