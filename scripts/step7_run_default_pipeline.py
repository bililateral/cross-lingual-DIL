from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
STEP7_POLICY_PATH = ROOT / "schema" / "step7_semantic_model_policy.json"
TRAINING_POLICY_PATH = ROOT / "schema" / "step7_training_policy.json"
DEFAULT_POOLS = ["en_content_train_pool", "zh_target_strict"]
DEFAULT_EMBEDDING_MODEL = "gte_multilingual_base"
DEFAULT_RERANKER_MODEL = "gte_multilingual_reranker_base"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the default Step 7 pipeline: preview -> semantic features -> LightGBM training."
    )
    parser.add_argument(
        "--pool",
        action="append",
        dest="pools",
        choices=DEFAULT_POOLS,
        help="Pool to process. Repeat to select multiple pools. Defaults to en_content_train_pool + zh_target_strict.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model key from step7_semantic_model_policy.json.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Reranker model key from step7_semantic_model_policy.json.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip Stage A preview rebuild.",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip Stage B semantic feature extraction.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Stage C LightGBM training.",
    )
    parser.add_argument(
        "--force-recompute-semantic",
        action="store_true",
        help="Pass --force-recompute to semantic feature extraction.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit device for semantic feature extraction, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def require_dependency(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(
            f"Missing Python dependency: {module_name}. Install it in the active environment, then rerun Step 7."
        )


def verify_runtime_dependencies(skip_semantic: bool, skip_train: bool) -> None:
    if not skip_semantic:
        require_dependency("torch")
        require_dependency("transformers")
    if not skip_train:
        require_dependency("lightgbm")


def verify_model_exists(policy: dict, embedding_model: str, reranker_model: str) -> None:
    unknown_embedding = embedding_model not in policy["embedding_models"]
    unknown_reranker = reranker_model not in policy["reranker_models"]
    if unknown_embedding or unknown_reranker:
        raise SystemExit(
            f"Unknown model key(s): embedding={embedding_model}, reranker={reranker_model}"
        )

    required = {
        embedding_model: policy["embedding_models"][embedding_model]["local_path"],
        reranker_model: policy["reranker_models"][reranker_model]["local_path"],
    }
    missing = []
    for model_key, relative_path in required.items():
        model_dir = ROOT / relative_path
        if not model_dir.is_dir() or not (model_dir / "config.json").exists():
            missing.append((model_key, relative_path))
    if missing:
        lines = [f"{model_key}: {relative_path}" for model_key, relative_path in missing]
        raise SystemExit(
            "Step 7 local models are missing. Download them first with:\n"
            "python scripts/step7_download_models.py "
            f"--embedding-model {embedding_model} --reranker-model {reranker_model}\n"
            "Missing:\n" + "\n".join(lines)
        )


def verify_required_inputs(skip_preview: bool, skip_semantic: bool, skip_train: bool, pools: list[str]) -> None:
    required = []
    preview_by_pool = {
        "en_content_train_pool": ROOT / "reports" / "step7_pair_feature_preview.en_content_train_pool.csv",
        "zh_target_strict": ROOT / "reports" / "step7_pair_feature_preview.zh_target_strict.csv",
    }
    if skip_preview and not skip_semantic:
        for pool in pools:
            required.append(preview_by_pool[pool])
    if not skip_train:
        required.extend(
            [
                ROOT / "reports" / "step5_en_frozen_silver_labels.csv",
                ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing Step 7 prerequisite files:\n" + "\n".join(str(path) for path in missing))


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def semantic_outputs_exist(pools: list[str]) -> bool:
    paths = {
        "en_content_train_pool": REPORTS_DIR / "step7_pair_features.en_content_train_pool.csv",
        "zh_target_strict": REPORTS_DIR / "step7_pair_features.zh_target_strict.csv",
    }
    return all(paths[pool].exists() for pool in pools)


def main() -> None:
    args = parse_args()
    pools = args.pools or DEFAULT_POOLS
    semantic_policy = load_json(STEP7_POLICY_PATH)
    load_json(TRAINING_POLICY_PATH)

    verify_runtime_dependencies(args.skip_semantic, args.skip_train)
    if not args.skip_semantic:
        verify_model_exists(semantic_policy, args.embedding_model, args.reranker_model)
    verify_required_inputs(args.skip_preview, args.skip_semantic, args.skip_train, pools)

    python_exe = sys.executable
    if not args.skip_preview:
        run_command([python_exe, "scripts/step7_build_pair_feature_preview.py"])

    if not args.skip_semantic:
        semantic_cmd = [python_exe, "scripts/step7_build_semantic_pair_features.py"]
        for pool in pools:
            semantic_cmd.extend(["--pool", pool])
        semantic_cmd.extend(["--embedding-model", args.embedding_model])
        semantic_cmd.extend(["--reranker-model", args.reranker_model])
        if args.force_recompute_semantic:
            semantic_cmd.append("--force-recompute")
        if args.device:
            semantic_cmd.extend(["--device", args.device])
        run_command(semantic_cmd)

    if not args.skip_train:
        if not semantic_outputs_exist(pools):
            raise SystemExit(
                "Semantic-enriched pair tables are missing for the selected pools. "
                "Run Stage B before training."
            )
        run_command([python_exe, "scripts/step7_train_baseline_models.py"])


if __name__ == "__main__":
    main()
