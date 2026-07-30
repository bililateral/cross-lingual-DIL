#!/usr/bin/env python3
"""Build the pre-key fixed-sample sensitivity artifact for Step28-v13.

This is deliberately not a confirmatory power simulation.  Before the formal
split keys exist, there is no defensible empirical distribution for the
world-level AP differences, five-placebo correlation, retrieval scores or
hard-negative errors.  The artifact therefore fixes the maximum feasible
500-world design and reports transparent normal-approximation sensitivity
over a registered paired-world standard-deviation grid.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import step28_v13_common as common


VERSION = (
    "2026-07-30-step28-v13-training-ready-"
    "fixed-sample-sensitivity-v1"
)
STATUS = (
    "PASS_FIXED_SAMPLE_ESTIMATION_DESIGN_"
    "NOT_CONFIRMATORY_POWER_CERTIFIED"
)
BASE_POLICY = (
    common.ROOT
    / "schema"
    / "step28_v13_synthetic_chinese_dataset_policy.json"
)
EXPECTED_BASE_POLICY_SHA256 = (
    "ce18015199c864df0f76a240df782c331020e5e76d483c5440cea6a673c74729"
)
WORLD_COUNT = 500
PAIRED_WORLD_SD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
FAMILYWISE_ALPHA = 0.05
AP_COMPARISON_COUNT = 7
TARGET_POINT_POWER = 0.80


def _producer_path() -> Path:
    return (
        common.ROOT
        / "scripts"
        / "step28_v13_build_fixed_sample_sensitivity.py"
    )


def _minimum_detectable_difference(
    *,
    paired_world_standard_deviation: float,
    margin: float,
    world_count: int,
    familywise_alpha: float,
    comparison_count: int,
    target_power: float,
) -> float:
    if (
        paired_world_standard_deviation <= 0.0
        or margin < 0.0
        or world_count < 2
        or not 0.0 < familywise_alpha < 1.0
        or comparison_count < 1
        or not 0.0 < target_power < 1.0
    ):
        raise common.ContractError(
            "Invalid fixed-sample sensitivity parameter"
        )
    normal = NormalDist()
    critical = normal.inv_cdf(
        1.0 - familywise_alpha / comparison_count
    )
    power_quantile = normal.inv_cdf(target_power)
    return margin + (
        (critical + power_quantile)
        * paired_world_standard_deviation
        / math.sqrt(world_count)
    )


def build_artifact() -> dict[str, Any]:
    if (
        not BASE_POLICY.is_file()
        or common.sha256_file(BASE_POLICY)
        != EXPECTED_BASE_POLICY_SHA256
    ):
        raise common.ContractError(
            "Fixed-sample sensitivity base-policy pin drift"
        )
    normal = NormalDist()
    critical = normal.inv_cdf(
        1.0 - FAMILYWISE_ALPHA / AP_COMPARISON_COUNT
    )
    power_quantile = normal.inv_cdf(TARGET_POINT_POWER)
    comparisons = {
        "audit_a_ap": {
            "success_margin": 0.03,
            "design_alternative": 0.06,
        },
        "audit_b_ap": {
            "success_margin": 0.015,
            "design_alternative": 0.04,
        },
    }
    sensitivity = {}
    for name, specification in comparisons.items():
        margin = float(specification["success_margin"])
        alternative = float(specification["design_alternative"])
        rows = []
        for standard_deviation in PAIRED_WORLD_SD_GRID:
            detectable = _minimum_detectable_difference(
                paired_world_standard_deviation=standard_deviation,
                margin=margin,
                world_count=WORLD_COUNT,
                familywise_alpha=FAMILYWISE_ALPHA,
                comparison_count=AP_COMPARISON_COUNT,
                target_power=TARGET_POINT_POWER,
            )
            rows.append(
                {
                    "paired_world_standard_deviation": (
                        standard_deviation
                    ),
                    "minimum_detectable_true_difference": detectable,
                    "registered_design_alternative_exceeds_mde": (
                        alternative >= detectable
                    ),
                }
            )
        maximum_sd = (
            (alternative - margin)
            * math.sqrt(WORLD_COUNT)
            / (critical + power_quantile)
        )
        sensitivity[name] = {
            **specification,
            "maximum_paired_world_standard_deviation_at_which_"
            "the_registered_alternative_reaches_nominal_point_power": (
                maximum_sd
            ),
            "grid": rows,
        }
    equivalence_half_width = 0.01
    equivalence_confidence = 0.90
    equivalence_z = normal.inv_cdf(
        0.5 + equivalence_confidence / 2.0
    )
    artifact: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "producer": {
            "path": (
                "scripts/"
                "step28_v13_build_fixed_sample_sensitivity.py"
            ),
            "sha256": common.sha256_file(_producer_path()),
        },
        "base_policy": {
            "path": (
                "schema/"
                "step28_v13_synthetic_chinese_dataset_policy.json"
            ),
            "sha256": EXPECTED_BASE_POLICY_SHA256,
        },
        "formal_private_structure_key_created_or_read": False,
        "formal_world_materialized": False,
        "design_type": (
            "fixed_maximal_resource_estimation_with_sensitivity_"
            "not_power_selected"
        ),
        "world_counts": {
            "train": WORLD_COUNT,
            "development": WORLD_COUNT,
            "audit_a": WORLD_COUNT,
            "audit_b": WORLD_COUNT,
        },
        "independent_analysis_unit": "world",
        "independent_worlds_per_split": WORLD_COUNT,
        "parent_confirmatory_power_contract": {
            "status_for_this_training_ready_child": (
                "EXPLICITLY_SUPERSEDED_BEFORE_ANY_PRIVATE_KEY"
            ),
            "reason": (
                "No empirical pre-key distribution exists for paired "
                "world AP differences, five-M1 dependence, retrieval "
                "scores or hard-negative errors. Supplying favourable "
                "values would fabricate power."
            ),
            "old_5000_replicate_monte_carlo_claimed": False,
            "old_grid_selected_world_count_claimed": False,
        },
        "claim_boundary": {
            "confirmatory_power_certified": False,
            "nominal_point_power_is_only_a_sensitivity_calculation": True,
            "binary_success_or_failure_from_power_is_forbidden": True,
            "post_release_sample_size_change_is_forbidden": True,
            "primary_reporting": (
                "effect estimates and preregistered paired world-cluster "
                "bootstrap intervals"
            ),
        },
        "normal_approximation_sensitivity": {
            "purpose": (
                "show how detectable AP differences depend on the unknown "
                "paired-world standard deviation; not a substitute for the "
                "future exact AP/bootstrap analysis"
            ),
            "familywise_alpha": FAMILYWISE_ALPHA,
            "comparison_count": AP_COMPARISON_COUNT,
            "multiplicity_method": "Bonferroni",
            "per_comparison_one_sided_alpha": (
                FAMILYWISE_ALPHA / AP_COMPARISON_COUNT
            ),
            "critical_normal_quantile": critical,
            "target_point_power": TARGET_POINT_POWER,
            "power_normal_quantile": power_quantile,
            "paired_world_standard_deviation_grid": list(
                PAIRED_WORLD_SD_GRID
            ),
            "formula": (
                "MDE = margin + "
                "(z_(1-alpha/7)+z_power)*paired_world_sd/sqrt(W)"
            ),
            "comparisons": sensitivity,
        },
        "m1_equivalence_sensitivity": {
            "equivalence_half_width": equivalence_half_width,
            "confidence_level": equivalence_confidence,
            "maximum_paired_world_standard_deviation_for_nominal_"
            "interval_half_width": (
                equivalence_half_width
                * math.sqrt(WORLD_COUNT)
                / equivalence_z
            ),
            "formula": (
                "maximum_sd = equivalence_half_width*sqrt(W)/z_.95"
            ),
            "confirmatory_tost_power_certified": False,
        },
        "unknown_until_formal_evaluation": [
            "paired-world AP-difference standard deviations",
            "five-M1 cross-seed correlation",
            "world intraclass dependence of row scores",
            "retrieval score and relevance dependence",
            "hard-negative error dependence",
        ],
    }
    artifact["canonical_self_hash"] = common.canonical_sha256(artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    output = parse_args().output.resolve()
    try:
        output.relative_to(common.ROOT.resolve())
    except ValueError as exc:
        raise common.ContractError(
            "Fixed-sample sensitivity output must stay in repository"
        ) from exc
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite fixed-sample artifact: {output}"
        )
    common.write_json(output, build_artifact())
    print(f"Wrote fixed-sample sensitivity artifact: {output}")


if __name__ == "__main__":
    main()
