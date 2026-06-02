# Step 12: Statistical Robustness

Status: current-boundary fixed `zh_test` robustness audit completed locally on `2026-05-13`; this page records the grouped bootstrap and paired-comparison result for the `2026-04-23` active Step 5 boundary.

## Role

Step 12 turns the current Step 7 / Step 9 point estimates into uncertainty-aware reporting evidence.

It does not:

- reopen Step 5
- mix `zh_train`, `zh_valid`, and `zh_test`
- relabel uncertain rows
- replace the fixed benchmark with random K-fold CV

It does:

- keep the fixed `zh_target_strict` test split
- resample by Step 5 `split_component_id`
- report 95% grouped bootstrap confidence intervals
- run paired bootstrap comparisons against raw semantic and Step 7 fusion baselines

## Inputs

- policy: `schema/step12_statistical_robustness_policy.json`
- runner: `scripts/step12_statistical_robustness_audit.py`
- frozen labels: `reports/step5_zh_target_strict_frozen_silver_labels.csv`
- pair features: `reports/step7_pair_features.zh_target_strict.csv`
- Step 7 prediction CSVs for current clean / identifier controls
- Step 9 prediction CSVs for current E5 LR/L2, BGE residual, LaBSE LR/L2, and identifier operational controls

## Fixed Test Container

The audit uses only current fixed `zh_test` supervision:

- rows: `106`
- positives: `21`
- negatives: `85`
- bootstrap groups: `39`
- largest group size: `14`

The bootstrap unit is `split_component_id`, not individual edge rows. This keeps pair dependencies from making the confidence intervals look narrower than they are.

## Outputs

- summary: `reports/step12_statistical_robustness_zh_test_20260513.json`
- model metrics: `reports/step12_statistical_robustness_model_metrics_20260513.csv`
- paired comparisons: `reports/step12_statistical_robustness_paired_comparisons_20260513.csv`

## Current Metric Reading

Key ROC-AUC readings:

- `step9_e5_lr_l2_50pct_seed_mean`: observed `0.819048`, 95% grouped CI `[0.701728, 0.916886]`
- `step9_bge_m3_residual_lr_100pct_seed_mean`: observed `0.817367`, CI `[0.726018, 0.914507]`
- `raw_e5_cosine`: observed `0.806723`, CI `[0.638524, 0.916667]`
- `raw_labse_cosine`: observed `0.806162`, CI `[0.708636, 0.906667]`
- `raw_bge_m3_cosine`: observed `0.783754`, CI `[0.624193, 0.936095]`
- `step7_core_zero_shot_bge_m3`: observed `0.601681`, CI `[0.420804, 0.819306]`
- `step7_core_zero_shot_default`: observed `0.588235`, CI `[0.410808, 0.819153]`

Key average-precision readings:

- `step9_e5_lr_l2_50pct_seed_mean`: observed `0.540494`, CI `[0.301265, 0.763798]`
- `raw_e5_cosine`: observed `0.520573`, CI `[0.220730, 0.745153]`
- `raw_labse_cosine`: observed `0.518581`, CI `[0.296728, 0.711879]`
- `step9_identifier_augmented_lr_l2_100pct_seed_mean`: observed `0.647686`, CI `[0.347406, 0.866396]`

Identifier-augmented AP remains the strongest operational AP point estimate, but it is not a clean transfer-safe scientific line.

## Paired Comparisons

Primary clean E5 LR/L2 seed-mean versus raw E5:

- ROC-AUC diff: `+0.012325`, 95% CI `[-0.108240, 0.147650]`, bootstrap sign p `0.736800`
- AP diff: `+0.019920`, 95% CI `[-0.251152, 0.326280]`, bootstrap sign p `0.805600`

Primary clean E5 LR/L2 seed-mean versus Step 7 BGE fusion:

- ROC-AUC diff: `+0.217367`, 95% CI `[-0.005893, 0.382821]`, bootstrap sign p `0.053200`
- AP diff: `+0.091733`, 95% CI `[-0.094190, 0.391933]`, bootstrap sign p `0.168400`

Primary clean E5 LR/L2 seed-mean versus Step 7 default fusion:

- ROC-AUC diff: `+0.230812`, 95% CI `[-0.011315, 0.400449]`, bootstrap sign p `0.060000`
- AP diff: `+0.091946`, 95% CI `[-0.093644, 0.406710]`, bootstrap sign p `0.163600`

Other clean controls:

- BGE residual vs raw BGE: ROC-AUC diff `+0.033613`, CI `[-0.140493, 0.176280]`
- LaBSE LR/L2 vs raw LaBSE: ROC-AUC diff `-0.006723`, CI `[-0.092913, 0.073583]`
- identifier operational vs raw E5: AP diff `+0.127113`, CI `[-0.278566, 0.523619]`

## Interpretation

The current Step 12 result supports a cautious reading:

- Step 9 E5 LR/L2 is a useful clean graph-triage scorer and has a better point estimate than raw E5.
- The grouped bootstrap interval for E5 LR/L2 versus raw E5 crosses zero, so the project should not claim a statistically robust improvement over raw E5 on the current fixed `zh_test`.
- E5 LR/L2 strongly improves over the collapsed Step 7 fusion point estimate, but grouped CIs still barely cross zero because the fixed test split has only `21` positives across dependent seller components.
- The correct paper wording is therefore: few-shot LR/L2 repairs the collapsed fusion baseline and provides a useful graph-triage surface, while the strongest clean claim against raw semantic baselines remains modest and statistically uncertain under grouped bootstrap.

## Run Command

```bash
python scripts/step12_statistical_robustness_audit.py --resamples 5000
```
