# Step 12: Statistical Robustness

Status: current-boundary fixed `zh_test` robustness audit updated on `2026-06-02`; this page records the grouped bootstrap and paired-comparison result for the `2026-04-23` active Step 5 boundary plus Step 15 v2 evidence-type curriculum scorers.

## Role

Step 12 turns the current Step 7 / Step 9 / Step 15 point estimates into uncertainty-aware reporting evidence.

It does not:

- reopen Step 5
- mix `zh_train`, `zh_valid`, and `zh_test`
- relabel uncertain rows
- replace the fixed benchmark with random K-fold CV

It does:

- keep the fixed `zh_target_strict` test split
- resample by Step 5 `split_component_id`
- report 95% grouped bootstrap confidence intervals
- run paired bootstrap comparisons against raw semantic, Step 7 fusion, Step 9 adaptation, and Step 15 v2 curriculum baselines

## Inputs

- policy: `schema/step12_statistical_robustness_policy.json`
- runner: `scripts/step12_statistical_robustness_audit.py`
- frozen labels: `reports/step5_zh_target_strict_frozen_silver_labels.csv`
- pair features: `reports/step7_pair_features.zh_target_strict.csv`
- Step 7 prediction CSVs for current clean / identifier controls
- Step 9 prediction CSVs for current E5 LR/L2, BGE residual, LaBSE LR/L2, and identifier operational controls
- Step 15 v2 prediction CSVs for identity-only curriculum, domain-balanced, target/source-only, mixup-scope, multitask ablation, and identifier operational controls

## Fixed Test Container

The audit uses only current fixed `zh_test` supervision:

- rows: `106`
- positives: `21`
- negatives: `85`
- bootstrap groups: `39`
- largest group size: `14`

The bootstrap unit is `split_component_id`, not individual edge rows. This keeps pair dependencies from making the confidence intervals look narrower than they are.

## Outputs

- summary: `reports/step12_v2_statistical_robustness_zh_test_20260602.json`
- model metrics: `reports/step12_v2_statistical_robustness_model_metrics_20260602.csv`
- paired comparisons: `reports/step12_v2_statistical_robustness_paired_comparisons_20260602.csv`

## Current Metric Reading

Key ROC-AUC readings:

- `step15_v2_domain_balanced_phase4_seed_mean`: observed `0.901401`, CI `[0.797735, 0.968098]`
- `step15_v2_identity_from_scratch_phase4_seed_mean`: observed `0.889636`, CI `[0.775385, 0.963048]`
- `step15_v2_zh_positive_mixup_phase4_seed_mean`: observed `0.892997`, CI `[0.784912, 0.965366]`
- `step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean`: observed `0.842017`, CI `[0.720812, 0.928571]`
- `raw_e5_cosine`: observed `0.806723`, CI `[0.638524, 0.916667]`
- `raw_labse_cosine`: observed `0.806162`, CI `[0.708636, 0.906667]`
- `raw_bge_m3_cosine`: observed `0.783754`, CI `[0.624193, 0.936095]`

Key average-precision readings:

- `step15_v2_identifier_operational_phase4_seed_mean`: observed `0.745849`, CI `[0.479067, 0.906404]`
- `step15_v2_domain_balanced_phase4_seed_mean`: observed `0.714371`, CI `[0.390322, 0.905488]`
- `step15_v2_identity_from_scratch_phase4_seed_mean`: observed `0.699299`, CI `[0.421039, 0.884818]`
- `step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean`: observed `0.588995`, CI `[0.275049, 0.836304]`
- `raw_e5_cosine`: observed `0.520573`, CI `[0.220730, 0.745153]`
- `raw_labse_cosine`: observed `0.518581`, CI `[0.296728, 0.711879]`

The identifier operational control has the strongest AP point estimate, but it uses direct identifier features and is not the clean transfer-safe scientific mainline.

## Paired Comparisons

Primary clean Step 15 v2 seed-mean versus raw E5:

- ROC-AUC diff: `+0.082913`, 95% CI `[-0.076627, 0.250000]`, bootstrap sign p `0.269200`
- AP diff: `+0.178725`, 95% CI `[-0.176207, 0.466066]`, bootstrap sign p `0.268800`

Primary clean Step 15 v2 versus Step 9 mixup 100pct:

- ROC-AUC diff: `+0.047619`, 95% CI `[-0.037055, 0.125878]`, bootstrap sign p `0.232800`
- AP diff: `+0.110304`, 95% CI `[-0.086075, 0.273455]`, bootstrap sign p `0.252000`

Other important comparisons:

- Step 15 v2 domain-balanced vs from-scratch: ROC-AUC diff `+0.011765`, CI `[-0.033585, 0.053476]`; AP diff `+0.015072`, CI `[-0.108300, 0.074526]`
- Step 15 v2 identifier operational vs clean primary: ROC-AUC diff `-0.006723`, CI `[-0.063915, 0.060250]`; AP diff `+0.046550`, CI `[-0.083281, 0.221741]`
- Step 9 mixup 100pct vs Step 7 default fusion: ROC-AUC diff `+0.253782`, CI `[0.018175, 0.410672]`

## Interpretation

The current Step 12 v2 result supports a cautious reading:

- Step 15 v2 provides the strongest current clean point estimates, especially the domain-balanced control.
- The grouped bootstrap intervals for Step 15 v2 primary versus raw E5 and versus Step 9 mixup 100pct still cross zero, so the project should not claim statistically robust superiority on the current fixed `zh_test`.
- Domain-balanced is a useful clean control, but its improvement over from-scratch is also not statistically separable under paired grouped bootstrap.
- The correct paper wording is therefore: evidence-type incremental hard-negative learning gives a strong and coherent point-estimate improvement, but publication claims must remain uncertainty-bounded unless more positive test evidence or external validation reduces the grouped-bootstrap uncertainty.

## Run Command

```bash
python scripts/step12_statistical_robustness_audit.py \
  --labels reports/step5_zh_target_strict_frozen_silver_labels.csv \
  --features reports/step7_pair_features.zh_target_strict.csv \
  --resamples 5000 \
  --seed 20260513 \
  --output-json reports/step12_v2_statistical_robustness_zh_test_20260602.json \
  --output-metrics reports/step12_v2_statistical_robustness_model_metrics_20260602.csv \
  --output-comparisons reports/step12_v2_statistical_robustness_paired_comparisons_20260602.csv
```
