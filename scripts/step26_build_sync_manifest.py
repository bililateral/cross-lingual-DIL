#!/usr/bin/env python3
"""Build an explicit input/output manifest for the Step26 Linux run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step26_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def record(path: Path, role: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Step26 manifest input/output is missing: {path}")
    return {
        "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256(path),
    }


def main() -> None:
    args = parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    output_root = common.resolve(policy["outputs_root"])
    required_code = [
        policy_path,
        common.resolve(policy["frozen_sources"]["step24_policy"]),
        common.ROOT / "scripts" / "step26_common.py",
        common.ROOT / "scripts" / "step26_build_frozen_style_cache.py",
        common.ROOT / "scripts" / "step26_evaluate_frozen_authorship_bridge.py",
        Path(__file__).resolve(),
        common.ROOT / "scripts" / "run_step26_frozen_authorship_bridge_linux_20260718.sh",
        common.ROOT / "tests" / "test_step26_frozen_authorship_bridge_contracts.py",
    ]
    required_inputs = [
        common.resolve(value) for value in policy["frozen_sources"].values()
    ]
    data_cfg = policy["evaluation_data"]
    required_inputs.extend(
        common.resolve(data_cfg[key])
        for key in (
            "frozen_labels",
            "evidence_labels",
            "seller_profiles",
            "item_identity_signals",
            "identifier_redacted_e5_metadata",
            "identifier_redacted_e5_matrix",
        )
    )
    for split_cfg in data_cfg["split_allowlists"].values():
        required_inputs.extend(
            common.resolve(split_cfg[key])
            for key in ("v8_b0_predictions", "v8_clean_predictions", "v8_contextual_predictions")
        )
    for encoder_key in policy["frozen_models"]["encoder_keys"]:
        model_path = common.resolve(step24_policy["frozen_style_encoders"][encoder_key]["local_path"])
        required_inputs.append(model_path / "step24_model_provenance.json")
    required_code = list(dict.fromkeys(required_code))
    required_inputs = list(dict.fromkeys(required_inputs))
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "required_code_files": len(required_code),
                    "required_input_files": len(required_inputs),
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    outputs = []
    for key, relative in policy["outputs"].items():
        if key == "sync_manifest":
            continue
        outputs.append(output_root / relative)
    for encoder_key in policy["frozen_models"]["encoder_keys"]:
        stem = output_root / "embeddings" / f"{encoder_key}.evaluation"
        outputs.extend((Path(f"{stem}.npy"), Path(f"{stem}.json")))
    manifest = {
        "step": "step26_sync_manifest",
        "version": policy["version"],
        "status": "complete",
        "required_code": [record(path, "code_or_policy") for path in required_code],
        "required_inputs": [record(path, "frozen_input") for path in required_inputs],
        "generated_outputs": [record(path, "generated_output") for path in outputs],
        "publication_claim_allowed": False,
        "policy_sha256": common.sha256(policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    manifest["manifest_payload_sha256"] = common.canonical_hash(manifest)
    output_path = output_root / policy["outputs"]["sync_manifest"]
    common.write_json_immutable(output_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "generated_output_count": len(outputs),
                "manifest": str(output_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
