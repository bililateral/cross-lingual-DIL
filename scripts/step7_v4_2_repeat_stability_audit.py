#!/usr/bin/env python3
"""Repeat Step7-v4.1 with frozen new component-fold seeds and no valid labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import lightgbm
import numpy as np
import scipy
import sklearn

import step7_v4_1_select_style_free_m0 as parent


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
POLICY_PATH = ROOT / "schema" / "step7_v4_2_repeat_stability_policy.json"
EXPECTED_VERSION = "2026-07-27-step7-v4.2-new-seed-stability-v1"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_record(record: dict, role: str) -> Path:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"Step7-v4.2 malformed file record: {role}")
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v4.2 missing {role}: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v4.2 byte-size drift: {role}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v4.2 SHA-256 drift: {role}")
    return path


def load_policy(*, require_frozen: bool = True) -> tuple[dict, dict]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("version") != EXPECTED_VERSION:
        raise ValueError("Step7-v4.2 policy version drift")
    if set(policy["frozen_parent"]) != {
        "policy",
        "selector",
        "formal_summary",
        "formal_train_oof",
        "formal_no_clone_oof",
    }:
        raise ValueError("Step7-v4.2 frozen-parent universe drift")
    for role, record in policy["frozen_parent"].items():
        verify_file_record(record, role)
    implementation = policy["implementation"]
    if set(implementation) != {"path", "sha256"}:
        raise ValueError("Step7-v4.2 implementation record drift")
    if resolve(implementation["path"]).resolve() != SCRIPT_PATH.resolve():
        raise ValueError("Step7-v4.2 implementation path drift")
    expected_script_hash = implementation["sha256"]
    if require_frozen or expected_script_hash != "TO_BE_FROZEN_AFTER_IMPLEMENTATION":
        if sha256_file(SCRIPT_PATH) != expected_script_hash:
            raise ValueError("Step7-v4.2 implementation SHA-256 drift")

    design = policy["repeat_design"]
    seeds = [int(value) for value in design["outer_seeds"]]
    if (
        seeds != [
            2026072701,
            2026072702,
            2026072703,
            2026072704,
            2026072705,
        ]
        or len(set(seeds)) != 5
        or int(design["bootstrap_seed"]) != 2026072717
        or int(design["outer_fold_count"]) != 5
        or int(design["inner_fold_count"]) != 4
        or design["repeat_complete_no_exact_clone_stress_test"] is not True
    ):
        raise ValueError("Step7-v4.2 repeat design drift")
    expected_outputs = {
        "root",
        "summary",
        "train_oof",
        "no_clone_oof",
    }
    if set(policy["outputs"]) != expected_outputs:
        raise ValueError("Step7-v4.2 output universe drift")
    root = policy["outputs"]["root"].rstrip("/")
    if root != "reports/step7_v4_2_repeat_stability/v1_20260727":
        raise ValueError("Step7-v4.2 output root drift")
    for role, value in policy["outputs"].items():
        if role != "root" and not value.startswith(root + "/"):
            raise ValueError(f"Step7-v4.2 output escapes root: {role}")
    if policy["operational_m0_primary"] != "lightgbm__legacy18_labse":
        raise ValueError("Step7-v4.2 operational M0 drift")
    boundary = policy["claim_boundary"]
    if (
        boundary["new_real_english_data"] is not False
        or boundary["old_valid_label_values_may_be_read"] is not False
        or boundary["historical_test_label_values_may_be_read"] is not False
        or boundary["candidate_or_grid_changes_allowed"] is not False
    ):
        raise ValueError("Step7-v4.2 claim boundary drift")

    base_policy = parent.load_policy()
    parent_record = policy["frozen_parent"]["policy"]
    if sha256_file(parent.POLICY_PATH) != parent_record["sha256"]:
        raise ValueError("Step7-v4.2 loaded parent policy drift")
    base_seeds = set(int(value) for value in base_policy["training"]["outer_seeds"])
    if base_seeds & set(seeds):
        raise ValueError("Step7-v4.2 repeat seed overlaps Step7-v4.1")
    return policy, base_policy


def runtime_policy(policy: dict, base_policy: dict) -> dict:
    output = copy.deepcopy(base_policy)
    output["training"]["outer_seeds"] = list(policy["repeat_design"]["outer_seeds"])
    output["evaluation"]["bootstrap"]["seed"] = int(
        policy["repeat_design"]["bootstrap_seed"]
    )
    return output


def strict_retrieval_metrics(rows: list[dict], scores: np.ndarray) -> dict:
    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.shape != (len(rows),) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Step7-v4.2 retrieval score shape/value drift")
    queries: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for row, score in zip(rows, probabilities, strict=True):
        label = int(row["review_label"] == "positive")
        left = row["seller_uid_left"]
        right = row["seller_uid_right"]
        queries[left].append((right, label, float(score)))
        queries[right].append((left, label, float(score)))

    ks = (1, 3, 5, 10)
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    hits_at_1: list[float] = []
    precisions = {k: [] for k in ks}
    recalls = {k: [] for k in ks}
    excluded_no_positive = 0
    excluded_no_negative = 0
    for _query, candidates in sorted(queries.items()):
        positive_count = sum(label == 1 for _uid, label, _score in candidates)
        negative_count = sum(label == 0 for _uid, label, _score in candidates)
        if positive_count == 0:
            excluded_no_positive += 1
            continue
        if negative_count == 0:
            excluded_no_negative += 1
            continue
        ranked = sorted(candidates, key=lambda item: (-item[2], item[0]))
        positive_ranks = [
            rank
            for rank, (_uid, label, _score) in enumerate(ranked, start=1)
            if label == 1
        ]
        reciprocal_ranks.append(1.0 / positive_ranks[0])
        average_precisions.append(
            float(
                np.mean(
                    [
                        positive_index / rank
                        for positive_index, rank in enumerate(
                            positive_ranks, start=1
                        )
                    ]
                )
            )
        )
        hits_at_1.append(float(positive_ranks[0] == 1))
        for k in ks:
            cutoff = min(k, len(ranked))
            found = sum(item[1] == 1 for item in ranked[:cutoff])
            precisions[k].append(found / cutoff)
            recalls[k].append(found / positive_count)
    if not reciprocal_ranks:
        return {
            "status": "not_estimable_no_query_with_both_label_classes",
            "eligible_query_count": 0,
            "excluded_no_positive_count": excluded_no_positive,
            "excluded_no_negative_count": excluded_no_negative,
        }
    return {
        "status": "diagnostic_incomplete_labelled_candidate_graph",
        "eligible_query_count": len(reciprocal_ranks),
        "excluded_no_positive_count": excluded_no_positive,
        "excluded_no_negative_count": excluded_no_negative,
        "map": float(np.mean(average_precisions)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "hits_at_1": float(np.mean(hits_at_1)),
        **{
            f"precision_at_{k}": float(np.mean(precisions[k]))
            for k in ks
        },
        **{f"recall_at_{k}": float(np.mean(recalls[k])) for k in ks},
    }


def data_counts(rows: list[dict]) -> dict:
    return {
        "pair_count": len(rows),
        "positive_count": sum(
            row["review_label"] == "positive" for row in rows
        ),
        "negative_count": sum(
            row["review_label"] == "negative" for row in rows
        ),
        "component_count": len({row["component_id"] for row in rows}),
    }


def assert_expected_counts(policy: dict, train_rows: list[dict], no_clone_rows: list[dict]) -> None:
    expected = policy["expected_data"]
    train = data_counts(train_rows)
    no_clone = data_counts(no_clone_rows)
    expected_train = {
        "pair_count": int(expected["train_pair_count"]),
        "positive_count": int(expected["train_positive_count"]),
        "negative_count": int(expected["train_negative_count"]),
        "component_count": int(expected["train_component_count"]),
    }
    if train != expected_train:
        raise ValueError(f"Step7-v4.2 train boundary drift: {train}")
    if (
        no_clone["pair_count"] != int(expected["no_clone_pair_count"])
        or no_clone["positive_count"] != int(expected["no_clone_positive_count"])
        or no_clone["negative_count"] != int(expected["no_clone_negative_count"])
    ):
        raise ValueError(f"Step7-v4.2 no-clone boundary drift: {no_clone}")


def compact_results(
    rows: list[dict],
    ranking: list[str],
    results: dict[str, dict],
) -> dict:
    output = {}
    for candidate_id in ranking:
        result = parent.compact_candidate_result(results[candidate_id])
        result["retrieval_metrics_extended"] = strict_retrieval_metrics(
            rows,
            results[candidate_id]["mean_repeated_nested_oof_scores"],
        )
        output[candidate_id] = result
    return output


def run_repeat(policy: dict, base_policy: dict) -> dict:
    run_policy = runtime_policy(policy, base_policy)
    (
        parent_policy,
        preparation_manifest,
        preparation_bundle,
        _fixed_features,
        _seller_records,
        _seller_markets,
        data_audit,
    ) = parent.load_style_free_parent_data(base_policy)
    factory = data_audit.pop("factory")
    pair_rows = preparation_bundle["pair_rows"]
    train_rows = parent.parent_selector.load_label_split(
        parent_policy, pair_rows, "train"
    )
    overlap_by_pair = parent.parent_common.exact_overlap_audit_by_pair(
        pair_rows, preparation_bundle["seller_text_rows"]
    )
    no_clone_rows = [
        row
        for row in train_rows
        if not overlap_by_pair[row["pair_uid"]][
            "any_exact_clean_text_overlap"
        ]
    ]
    assert_expected_counts(policy, train_rows, no_clone_rows)
    preflight = {
        "train": parent.preflight_nested_support(
            run_policy, train_rows, role="repeat_train"
        ),
        "no_exact_clone": parent.preflight_nested_support(
            run_policy, no_clone_rows, role="repeat_no_exact_clone"
        ),
    }
    print("[Step7-v4.2] preflight passed; starting new-seed main repeat", flush=True)
    main_results, main_ranking, main_audit = parent.run_nested_selection(
        run_policy,
        parent_policy,
        factory,
        train_rows,
        progress_label="v4.2-main",
    )
    print("[Step7-v4.2] starting new-seed no-clone repeat", flush=True)
    no_clone_results, no_clone_ranking, no_clone_audit = (
        parent.run_nested_selection(
            run_policy,
            parent_policy,
            factory,
            no_clone_rows,
            progress_label="v4.2-no-clone",
        )
    )
    decision = parent.assess_selection(
        run_policy,
        train_rows,
        main_results,
        main_ranking,
        main_audit,
        no_clone_rows,
        no_clone_results,
        no_clone_ranking,
        no_clone_audit,
    )

    train_oof_path = resolve(policy["outputs"]["train_oof"])
    no_clone_oof_path = resolve(policy["outputs"]["no_clone_oof"])
    parent.parent_common.write_csv_immutable(
        train_oof_path,
        parent.train_prediction_rows(
            run_policy, train_rows, main_ranking, main_results
        ),
    )
    parent.parent_common.write_csv_immutable(
        no_clone_oof_path,
        parent.train_prediction_rows(
            run_policy,
            no_clone_rows,
            no_clone_ranking,
            no_clone_results,
        ),
    )

    # The old summary is opened only after the repeat scores and ranking exist.
    old_summary_path = verify_file_record(
        policy["frozen_parent"]["formal_summary"], "formal_summary"
    )
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    primary = policy["operational_m0_primary"]
    comparison = {
        "operational_m0_primary": primary,
        "v4_1_main_winner": old_summary["train_only_candidate_ranking"][0],
        "v4_2_main_winner": main_ranking[0],
        "v4_2_no_clone_winner": no_clone_ranking[0],
        "primary_remains_main_aggregate_winner": main_ranking[0] == primary,
        "primary_main_rank": main_ranking.index(primary) + 1,
        "primary_no_clone_rank": no_clone_ranking.index(primary) + 1,
        "v4_1_and_v4_2_full_ranking_identical": (
            old_summary["train_only_candidate_ranking"] == main_ranking
        ),
        "v4_1_and_v4_2_no_clone_ranking_identical": (
            old_summary["no_exact_clone_candidate_ranking"]
            == no_clone_ranking
        ),
        "new_seed_main_winners": [
            row["seed_winner"] for row in main_audit["outer_seed_audit"]
        ],
        "new_seed_no_clone_winners": [
            row["seed_winner"]
            for row in no_clone_audit["outer_seed_audit"]
        ],
        "same_data_repeat_is_independent_confirmation": False,
    }
    summary = {
        "step": "step7_v4_2_repeat_stability_audit",
        "version": policy["version"],
        "objective": policy["objective"],
        "claim_boundary": policy["claim_boundary"],
        "repeat_design": policy["repeat_design"],
        "preflight": preflight,
        "data": {
            "train": data_counts(train_rows),
            "no_exact_clone": data_counts(no_clone_rows),
            "parent_preparation_manifest_content_sha256": (
                preparation_manifest["manifest_content_sha256"]
            ),
        },
        "comparison": comparison,
        "selection_decision": decision,
        "train_candidate_ranking": main_ranking,
        "train_nested_audit": main_audit,
        "train_candidate_results": compact_results(
            train_rows, main_ranking, main_results
        ),
        "no_exact_clone_candidate_ranking": no_clone_ranking,
        "no_exact_clone_nested_audit": no_clone_audit,
        "no_exact_clone_candidate_results": compact_results(
            no_clone_rows, no_clone_ranking, no_clone_results
        ),
        "old_valid_label_values_read": False,
        "historical_test_label_values_read": False,
        "frozen_parent": policy["frozen_parent"],
        "outputs": {
            "train_oof": parent.parent_common.file_record(train_oof_path),
            "no_clone_oof": parent.parent_common.file_record(
                no_clone_oof_path
            ),
        },
        "execution_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "lightgbm_version": lightgbm.__version__,
            "platform": platform.platform(),
            "gpu_used": False,
        },
        "policy_sha256": sha256_file(POLICY_PATH),
        "producer_sha256": sha256_file(SCRIPT_PATH),
    }
    summary = parent.json_ready(summary)
    summary["summary_content_sha256"] = parent.canonical_hash(summary)
    summary_path = resolve(policy["outputs"]["summary"])
    parent.parent_common.write_json_immutable(summary_path, summary)
    return summary


def validate_config_only(policy: dict, base_policy: dict) -> dict:
    return {
        "status": "pass",
        "version": policy["version"],
        "operational_m0_primary": policy["operational_m0_primary"],
        "new_outer_seeds": policy["repeat_design"]["outer_seeds"],
        "base_outer_seeds": base_policy["training"]["outer_seeds"],
        "candidate_count": len(
            parent.candidate_specs(
                base_policy,
                parent.parent_common.load_json(
                    parent.resolve(base_policy["parent_contract"]["policy_path"])
                ),
            )
        ),
        "old_valid_label_values_read": False,
        "historical_test_label_values_read": False,
        "numerical_execution_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy, base_policy = load_policy()
    if args.validate_config_only:
        print(
            json.dumps(
                validate_config_only(policy, base_policy),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = run_repeat(policy, base_policy)
    print(
        json.dumps(
            {
                "status": "completed",
                **summary["comparison"],
                "summary": policy["outputs"]["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
