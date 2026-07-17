#!/usr/bin/env python3
"""Download the two pinned Step24 models on Windows and record provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step24_common as common


def write_provenance(model_path: Path, repo_id: str, revision: str) -> None:
    payload = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": revision,
        "download_contract": "huggingface_hub_snapshot_download_pinned_revision",
    }
    path = model_path / "step24_model_provenance.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to replace different Step24 model provenance: {path}")
    path.write_text(rendered, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    common.validate_policy(policy)
    planned = {
        key: {
            "repo_id": cfg["repo_id"],
            "revision": cfg["revision"],
            "local_path": cfg["local_path"],
        }
        for key, cfg in policy["frozen_style_encoders"].items()
    }
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "planned_models": planned}, indent=2))
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub in the Windows environment before downloading Step24 models"
        ) from exc

    api = HfApi()
    completed = {}
    for encoder_key, cfg in policy["frozen_style_encoders"].items():
        model_info = api.model_info(cfg["repo_id"], revision=cfg["revision"])
        if model_info.sha != cfg["revision"]:
            raise ValueError(
                f"Hugging Face resolved an unexpected revision for {cfg['repo_id']}: "
                f"{model_info.sha}"
            )
        model_path = common.resolve(cfg["local_path"])
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=cfg["repo_id"],
            revision=cfg["revision"],
            local_dir=str(model_path),
        )
        for required_name in ("config.json", "modules.json"):
            if not (model_path / required_name).is_file():
                raise FileNotFoundError(
                    f"Downloaded Step24 model is incomplete: {model_path / required_name}"
                )
        write_provenance(model_path, cfg["repo_id"], cfg["revision"])
        common.validate_model_provenance(model_path, cfg)
        completed[encoder_key] = {
            **planned[encoder_key],
            "directory_fingerprint": common.directory_fingerprint(model_path),
        }
    print(json.dumps({"status": "pass", "models": completed}, indent=2))


if __name__ == "__main__":
    main()
