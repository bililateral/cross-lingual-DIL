#!/usr/bin/env python3
"""Download the two pinned Step24 models on Windows and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import httpx

import step24_common as common


def write_provenance(model_path: Path, repo_id: str, revision: str) -> None:
    payload = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": revision,
        "download_contract": "huggingface_model_info_plus_streaming_get_pinned_revision",
    }
    path = model_path / "step24_model_provenance.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to replace different Step24 model provenance: {path}")
    path.write_text(rendered, encoding="utf-8", newline="\n")


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lfs_sha256(sibling: object) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None) if lfs is not None else None


def verify_snapshot_file(path: Path, sibling: object) -> bool:
    expected_size = int(getattr(sibling, "size"))
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    expected_lfs_sha256 = lfs_sha256(sibling)
    if expected_lfs_sha256:
        return common.sha256_file(path) == expected_lfs_sha256
    expected_blob_sha1 = str(getattr(sibling, "blob_id", "") or "")
    return not expected_blob_sha1 or git_blob_sha1(path) == expected_blob_sha1


def stream_snapshot_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    sibling: object,
    model_path: Path,
) -> None:
    relative_name = str(getattr(sibling, "rfilename"))
    destination = model_path / Path(relative_name)
    if verify_snapshot_file(destination, sibling):
        print(f"Verified existing file: {repo_id}/{relative_name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".step24.part")
    expected_size = int(getattr(sibling, "size"))
    encoded_name = quote(relative_name.replace("\\", "/"), safe="/")
    url = f"{endpoint.rstrip('/')}/{repo_id}/resolve/{revision}/{encoded_name}?download=true"
    for attempt in range(1, 6):
        existing_size = partial.stat().st_size if partial.is_file() else 0
        if existing_size > expected_size:
            partial.unlink()
            existing_size = 0
        headers = {"User-Agent": "cross-lingual-step24-pinned-downloader/1.0"}
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
        try:
            timeout = httpx.Timeout(connect=60.0, read=300.0, write=300.0, pool=60.0)
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    append = existing_size > 0 and response.status_code == 206
                    mode = "ab" if append else "wb"
                    if existing_size and not append:
                        existing_size = 0
                    print(
                        f"Downloading {repo_id}/{relative_name} "
                        f"from byte {existing_size} ({expected_size} bytes)",
                        flush=True,
                    )
                    with partial.open(mode) as handle:
                        for block in response.iter_bytes(chunk_size=8 * 1024 * 1024):
                            handle.write(block)
            if partial.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch: observed={partial.stat().st_size} expected={expected_size}"
                )
            if lfs_sha256(sibling):
                if common.sha256_file(partial) != lfs_sha256(sibling):
                    raise IOError("LFS SHA-256 mismatch")
            else:
                expected_blob_sha1 = str(getattr(sibling, "blob_id", "") or "")
                if expected_blob_sha1 and git_blob_sha1(partial) != expected_blob_sha1:
                    raise IOError("Git blob SHA-1 mismatch")
            os.replace(partial, destination)
            return
        except Exception as exc:
            if attempt == 5:
                raise RuntimeError(
                    f"Failed to download pinned snapshot file after five attempts: "
                    f"{repo_id}/{relative_name}"
                ) from exc
            wait_seconds = attempt * 5
            print(
                f"Retry {attempt}/5 for {repo_id}/{relative_name} after {exc}; "
                f"waiting {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)


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

    # The Windows research host reaches the Hub through a mirror/proxy. Serial
    # metadata requests avoid intermittent HEAD failures from concurrent range
    # requests without changing the pinned snapshot or its final fingerprint.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub in the Windows environment before downloading Step24 models"
        ) from exc

    api = HfApi()
    completed = {}
    for encoder_key, cfg in policy["frozen_style_encoders"].items():
        model_info = api.model_info(
            cfg["repo_id"], revision=cfg["revision"], files_metadata=True
        )
        if model_info.sha != cfg["revision"]:
            raise ValueError(
                f"Hugging Face resolved an unexpected revision for {cfg['repo_id']}: "
                f"{model_info.sha}"
            )
        model_path = common.resolve(cfg["local_path"])
        model_path.mkdir(parents=True, exist_ok=True)
        endpoint = str(getattr(api, "endpoint", "") or os.environ.get("HF_ENDPOINT", ""))
        if not endpoint:
            raise ValueError("Hugging Face endpoint is empty")
        for sibling in model_info.siblings:
            stream_snapshot_file(
                endpoint,
                cfg["repo_id"],
                cfg["revision"],
                sibling,
                model_path,
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
