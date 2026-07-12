#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/yongpeng/cross-lingual}"
cd "$ROOT"

if [[ "${SKIP_PRE_STEP9:-0}" != "1" ]]; then
echo "[1/10] Verify the synchronized Step16G boundary and refresh Step5 metadata"
python3 scripts/step5_refresh_frozen_summary.py \
  --boundary-id step16g_hard_negative_imbalance_20260710 \
  --reason "Step16G train-only hard-negative expansion; valid/test unchanged"

python3 - <<'PY'
import csv
from collections import Counter

def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

labels = read("reports/step5_zh_target_strict_frozen_silver_labels.csv")
features = {row["pair_uid"] for row in read("reports/step7_pair_features.zh_target_strict.csv")}
counts = Counter(
    (row.get("split_name"), row.get("review_label"))
    for row in labels
    if row.get("review_label") in {"positive", "negative"}
)
expected = {
    ("train", "positive"): 229,
    ("train", "negative"): 344,
    ("valid", "positive"): 30,
    ("valid", "negative"): 90,
    ("test", "positive"): 50,
    ("test", "negative"): 150,
}
if any(counts[key] != value for key, value in expected.items()):
    raise SystemExit(f"Unexpected Step16G split counts: {dict(counts)}")
step16g = [row for row in labels if row.get("reviewer_id") == "step16g_hard_negative_imbalance_20260710"]
if len(step16g) != 115 or any(row["pair_uid"] not in features for row in step16g):
    raise SystemExit("Step16G rows are missing or lack Step7 feature coverage")
print("Step16G boundary verified:", dict(counts))
PY

echo "[2/10] Rerun Step7 source-domain baselines"
# The 115 new rows already have Step7 pair features, so semantic embeddings do not need recomputation.
python3 scripts/step7_train_baseline_models.py

echo "[3/10] Rerun Step9 calibration controls"
python3 scripts/step9_run_calibration_adaptation.py \
  --experiment core_calibrated_default \
  --experiment core_calibrated_bge_m3 \
  --experiment identifier_augmented_calibrated_default
else
  echo "[1-3/10] SKIP_PRE_STEP9=1: keeping completed Step5/Step7/calibration outputs"
fi

echo "[4/10] Rerun the complete Step9 support-ratio grid"
python3 scripts/step9_run_few_shot_adaptation.py \
  --ratio 0.1 \
  --ratio 0.2 \
  --ratio 0.5 \
  --ratio 1.0 \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322

echo "[5/10] Rebuild Step15 evidence types and rerun frozen v5"
python3 scripts/step15_build_evidence_type_labels.py \
  --policy schema/step15_evidence_type_policy.json \
  --pool en_content_train_pool \
  --pool zh_target_strict

python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json \
  --experiment step15_v5_identity_only_curriculum_public_noise_weighted_strong \
  --experiment step15_v5_identity_only_curriculum_domain_balanced_public_noise_weighted_strong \
  --phase phase0_identity_warm_start \
  --phase phase1_add_semantic_topic_negative \
  --phase phase2_add_template_clone_negative \
  --phase phase3_add_contact_url_noise \
  --phase phase4_add_positive_pair_mixup \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322

python3 scripts/step15_slice_level_audit.py \
  --policy schema/step15_evidence_type_policy.json

echo "[6/10] Run Step12 grouped-bootstrap robustness and same-ratio mixup ablation"
python3 scripts/step12_statistical_robustness_audit.py \
  --output-json reports/step12_v5_statistical_robustness_zh_test_step16g_imbalance_20260710.json \
  --output-metrics reports/step12_v5_statistical_robustness_model_metrics_step16g_imbalance_20260710.csv \
  --output-comparisons reports/step12_v5_statistical_robustness_paired_comparisons_step16g_imbalance_20260710.csv

echo "[7/10] Rerun the explicit Step11 publication candidates"
python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family step15 \
  --step15-experiment step15_v5_identity_only_curriculum_domain_balanced_public_noise_weighted_strong \
  --step15-phase phase4_add_positive_pair_mixup \
  --step15-seed 20260320 \
  --step15-seed 20260321 \
  --step15-seed 20260322

python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family step15 \
  --step15-experiment step15_v5_identity_only_curriculum_public_noise_weighted_strong \
  --step15-phase phase4_add_positive_pair_mixup \
  --step15-seed 20260320 \
  --step15-seed 20260321 \
  --step15-seed 20260322

for seed in 20260320 20260321 20260322; do
  python3 scripts/step11_cluster_chinese_graph.py \
    --policy schema/step11_clustering_policy.json \
    --scorer-family step9 \
    --step9-experiment core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup \
    --step9-ratio 1.0 \
    --step9-seed "$seed"
done

python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family step7 \
  --step7-experiment core_zero_shot_bge_m3

echo "[8/10] Generate the explicit six-summary Step11 cluster audit"
python3 scripts/step11_cluster_level_audit.py \
  --summary reports/step11_step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean_clustering_summary.json \
  --summary reports/step11_step15_v5_public_noise_weighted_strong_phase4_seed_mean_clustering_summary.json \
  --summary reports/step11_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260320_clustering_summary.json \
  --summary reports/step11_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260321_clustering_summary.json \
  --summary reports/step11_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260322_clustering_summary.json \
  --summary reports/step11_core_zero_shot_bge_m3_clustering_summary.json \
  --output-csv reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.csv \
  --output-summary reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.json

echo "[9/10] Regenerate Step13 against the explicit Step11 audit"
python3 scripts/step13_concept_drift_audit.py \
  --step11-audit reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.json \
  --output-json reports/step13_concept_drift_audit.step16g_imbalance_validation_20260710.json \
  --output-csv reports/step13_concept_drift_audit.step16g_imbalance_validation_20260710.csv \
  --output-md docs/STEP13_CONCEPT_DRIFT_AUDIT_STEP16G_IMBALANCE_VALIDATION_20260710.md

echo "[10/10] Verify mixup execution and publication-audit isolation"
python3 - <<'PY'
import csv
import json
from pathlib import Path

seeds = (20260320, 20260321, 20260322)
for seed in seeds:
    artifact_path = Path(
        "reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_"
        f"ratio_100pct_seed_{seed}_artifact.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    mixup = artifact["positive_pair_mixup"]
    if mixup.get("synthetic_row_count") != 115:
        raise SystemExit(f"{artifact_path} has unexpected mixup diagnostics: {mixup}")
    if mixup.get("eligible_positive_source_count") != 72:
        raise SystemExit(f"{artifact_path} has unexpected eligible source count: {mixup}")

    def scores(path):
        with open(path, encoding="utf-8-sig", newline="") as handle:
            return {row["pair_uid"]: float(row["prob_positive"]) for row in csv.DictReader(handle)}

    mixed = scores(
        "reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_"
        f"ratio_100pct_seed_{seed}_predictions.zh_test.csv"
    )
    plain = scores(
        "reports/step9_core_few_shot_multilingual_e5_large_lr_l2_"
        f"ratio_100pct_seed_{seed}_predictions.zh_test.csv"
    )
    if mixed.keys() != plain.keys() or max(abs(mixed[key] - plain[key]) for key in mixed) <= 1e-12:
        raise SystemExit(f"Mixup and non-mixup predictions are not a valid distinct paired control for seed {seed}")

step12 = json.loads(
    Path("reports/step12_v5_statistical_robustness_zh_test_step16g_imbalance_20260710.json").read_text(
        encoding="utf-8"
    )
)
roles = {row["comparison_role"] for row in step12["paired_comparisons"]}
if "step9_mixup100_vs_non_mixup100_same_ratio" not in roles:
    raise SystemExit("Step12 is missing the same-ratio mixup comparison")

audit = json.loads(
    Path("reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.json").read_text(
        encoding="utf-8"
    )
)
if audit.get("summary_selection_mode") != "explicit" or audit.get("input_summary_count") != 6:
    raise SystemExit("Step11 cluster audit did not use the explicit six-summary allow-list")

step13 = json.loads(
    Path("reports/step13_concept_drift_audit.step16g_imbalance_validation_20260710.json").read_text(
        encoding="utf-8"
    )
)
expected_audit = "reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.json"
if not any(str(path).replace("\\", "/") == expected_audit for path in step13.get("inputs", {})):
    raise SystemExit("Step13 does not record the explicit Step16G Step11 audit input")

print("Step16G rerun verification passed for all three mixup seeds, Step12, Step11, and Step13.")
PY
