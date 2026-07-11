#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "step15_v5r_output_contract_validation.json"
SEEDS = (20260320, 20260321, 20260322)
EXPERIMENTS = {
    "step15_v5r_identity_only_curriculum_public_noise_weighted_strong_weighted_mixup": False,
    "step15_v5r_identity_only_curriculum_domain_balanced_public_noise_weighted_strong_weighted_mixup": True,
}
PHASE = "phase4_add_positive_pair_mixup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step15 v5r artifact and mixup-manifest contracts.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_run(experiment: str, seed: int, domain_balanced: bool) -> dict[str, Any]:
    token = f"{experiment}_{PHASE}_seed_{seed}"
    artifact_path = ROOT / "reports" / f"{token}_artifact.json"
    manifest_path = ROOT / "reports" / f"{token}_positive_mixup_manifest.csv"
    require(artifact_path.is_file(), f"Missing Step15 v5r artifact: {artifact_path}")
    require(manifest_path.is_file(), f"Missing Step15 v5r mixup manifest: {manifest_path}")

    artifact = load_json(artifact_path)
    diagnostics = artifact.get("training_diagnostics") or {}
    mixup = diagnostics.get("positive_pair_mixup") or {}
    synthetic_count = int(mixup.get("synthetic_row_count", 0) or 0)
    require(synthetic_count > 0, f"{token}: mixup generated no synthetic rows")
    require(mixup.get("scope") == "same_domain_same_evidence_type", f"{token}: incorrect mixup scope")
    require(int(mixup.get("cross_domain_parent_count", -1)) == 0, f"{token}: cross-domain parents detected")
    require(
        int(mixup.get("cross_evidence_type_parent_count", -1)) == 0,
        f"{token}: cross-evidence-type parents detected",
    )
    require(float(mixup.get("synthetic_weight_min", 0.0)) >= 0.55 - 1e-9, f"{token}: synthetic weight below 0.55")
    require(float(mixup.get("synthetic_weight_max", 2.0)) <= 1.0 + 1e-9, f"{token}: synthetic weight above 1.0")
    require(mixup.get("synthetic_weight_mode") == "minimum_parent_weight", f"{token}: wrong weight mode")

    manifest_rows = load_csv(manifest_path)
    require(len(manifest_rows) == synthetic_count, f"{token}: manifest/synthetic count mismatch")
    for row_number, row in enumerate(manifest_rows, start=2):
        require(
            row["mixup_parent_left_pool"] == row["mixup_parent_right_pool"] == row["step15_pool"],
            f"{token}:{row_number}: manifest contains cross-domain parents",
        )
        require(
            row["mixup_parent_left_evidence_type"] == row["mixup_parent_right_evidence_type"],
            f"{token}:{row_number}: manifest contains cross-evidence parents",
        )
        expected_weight = min(
            float(row["mixup_parent_left_training_sample_weight"]),
            float(row["mixup_parent_right_training_sample_weight"]),
        )
        require(
            abs(float(row["training_sample_weight"]) - expected_weight) <= 1e-9,
            f"{token}:{row_number}: synthetic weight does not equal minimum parent weight",
        )

    domain = diagnostics.get("effective_domain_balance") or {}
    if domain_balanced:
        require(domain.get("enabled") is True, f"{token}: effective domain balance is not enabled")
        require(
            domain.get("method") == "post_quality_effective_weight_mass",
            f"{token}: wrong effective domain-balance method",
        )
        mass_after = domain.get("mass_after") or {}
        require(set(mass_after) == {"en_content_train_pool", "zh_target_strict"}, f"{token}: wrong balanced domains")
        require(
            abs(float(mass_after["en_content_train_pool"]) - float(mass_after["zh_target_strict"])) <= 1e-5,
            f"{token}: effective domain masses are not equal",
        )
    else:
        require(domain.get("enabled") is False, f"{token}: non-domain experiment unexpectedly balances domains")

    return {
        "experiment": experiment,
        "seed": seed,
        "synthetic_row_count": synthetic_count,
        "eligible_positive_source_count": int(mixup.get("eligible_positive_source_count", 0) or 0),
        "cross_domain_parent_count": 0,
        "cross_evidence_type_parent_count": 0,
        "synthetic_weight_min": float(mixup["synthetic_weight_min"]),
        "synthetic_weight_mean": float(mixup["synthetic_weight_mean"]),
        "synthetic_weight_max": float(mixup["synthetic_weight_max"]),
        "domain_balance_method": domain.get("method"),
        "domain_mass_after": domain.get("mass_after"),
        "artifact": str(artifact_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
    }


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    runs = [
        validate_run(experiment, seed, domain_balanced)
        for experiment, domain_balanced in EXPERIMENTS.items()
        for seed in SEEDS
    ]
    payload = {
        "step": "step15_validate_v5r_outputs",
        "status": "pass",
        "validated_run_count": len(runs),
        "hard_rules": {
            "same_domain_parents_only": True,
            "same_evidence_type_parents_only": True,
            "minimum_parent_weight_inherited": True,
            "effective_domain_mass_balanced_over_real_domains_only": True,
        },
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "pass", "output": str(output_path.relative_to(ROOT)), "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
