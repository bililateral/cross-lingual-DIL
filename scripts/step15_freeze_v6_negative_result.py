#!/usr/bin/env python3
"""Materialize an immutable hash manifest for the frozen Step15-v6 negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v6_negative_freeze.json"
DEFAULT_OUTPUT = ROOT / "reports" / "step15_v6" / "manifests" / "step15_v6_negative_freeze_manifest.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    output_path = resolve(args.output)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "frozen_strict_negative_result":
        raise ValueError("The v6 freeze policy is not marked frozen_strict_negative_result")
    if policy.get("promotion_eligible") is not False:
        raise ValueError("The v6 negative freeze must record promotion_eligible=false")

    records = []
    for value in policy.get("frozen_outputs", []):
        path = resolve(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen v6 output: {path}")
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    step12_path = resolve(policy["frozen_outputs"][1])
    step12 = json.loads(step12_path.read_text(encoding="utf-8"))
    if (step12.get("promotion") or {}).get("eligible") is not False:
        raise ValueError("Frozen Step12-v6 result no longer reproduces promotion.eligible=false")
    if (step12.get("selection") or {}).get("final_selected") != policy.get("selected_model"):
        raise ValueError("Frozen v6 selected model disagrees with the Step12-v6 result")

    manifest = {
        "step": "step15_freeze_v6_negative_result",
        "version": policy["version"],
        "status": policy["status"],
        "selected_model": policy["selected_model"],
        "promotion_eligible": False,
        "evaluation_role": policy["evaluation_role"],
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": sha256(policy_path),
        "producer": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "records": records,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    if output_path.exists():
        observed = json.loads(output_path.read_text(encoding="utf-8"))
        if observed != manifest:
            raise FileExistsError(
                f"Refusing to replace a different v6 negative freeze manifest: {output_path}"
            )
        print(json.dumps({"status": "verified", "manifest": str(output_path.relative_to(ROOT))}, indent=2))
        return
    if args.verify_existing:
        raise FileNotFoundError(f"V6 negative freeze manifest does not exist: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)
    print(json.dumps({"status": "created", "manifest": str(output_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
