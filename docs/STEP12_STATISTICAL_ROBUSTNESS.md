# Step 12: Statistical Robustness

Status: current fixed `zh_test` grouped-bootstrap audit updated on `2026-07-04` after adding extended ranking metrics to Step 7, Step 9, Step 12, and Step 15.

## Role

Step 12 turns the fixed-test point estimates from Step 7, Step 9, and Step 15 into uncertainty-aware reporting evidence.

It does not:

- reopen Step 5 labels
- mix `zh_train`, `zh_valid`, and `zh_test`
- relabel uncertain rows
- tune thresholds on `zh_test`
- replace the fixed benchmark with random K-fold cross validation

It does:

- keep the fixed `zh_target_strict` test split
- resample by Step 5 `split_component_id`
- report 95% grouped bootstrap confidence intervals
- run paired bootstrap comparisons against raw semantic, Step 7 fusion, Step 9 target-domain adaptation, and Step 15 v5 curriculum scorers

## Metric Scope

The current Step 12 output reports:

- `roc_auc`
- `average_precision`
- `pr_auc`
- `map`
- `mrr`

Current definitions:

- `average_precision` is the binary pair-ranking average precision over the fixed `zh_test` rows.
- `pr_auc` is an explicit alias of `average_precision` under this binary setup.
- `map` is the same global pair-ranking value as `average_precision`; there is no seller-group query partition in the current Step 12 audit.
- `mrr` is the reciprocal rank of the first positive pair in the global ranked list.

Because `pr_auc` and `map` are aliases of `average_precision` in the current global pair-ranking audit, they should not be interpreted as independent evidence. `mrr` is useful as a supplementary ranking diagnostic, but it is weakly discriminative here because multiple strong models rank at least one positive pair first.

## Inputs

- runner: `scripts/step12_statistical_robustness_audit.py`
- frozen labels: `reports/step5_zh_target_strict_frozen_silver_labels.csv`
- pair features: `reports/step7_pair_features.zh_target_strict.csv`
- Step 7 prediction CSVs for current clean and identifier controls
- Step 9 prediction CSVs for E5 LR/L2, E5 positive-pair mixup, BGE residual/LR controls, LaBSE controls, and identifier operational controls
- Step 15 v5 prediction CSVs for public-noise weighted and domain-balanced public-noise weighted scorers

## Fixed Test Container

The audit uses only the fixed `zh_test` supervision:

- rows: `106`
- positives: `21`
- negatives: `85`
- bootstrap groups: `39`

The bootstrap unit is `split_component_id`, not individual edge rows. This keeps pair dependencies from making the confidence intervals look narrower than they are.

## Outputs

- summary: `reports/step12_v5_statistical_robustness_zh_test_20260603.json`
- model metrics: `reports/step12_v5_statistical_robustness_model_metrics_20260603.csv`
- paired comparisons: `reports/step12_v5_statistical_robustness_paired_comparisons_20260603.csv`

The filenames retain the original v5 date token because this is the same frozen Step 15 v5 evaluation boundary. The `2026-07-04` rerun extends the metric columns without changing the underlying train/test split.

## Current Metric Reading

Key model readings:

| Model | ROC-AUC | AP / PR-AUC / MAP | MRR |
| --- | ---: | ---: | ---: |
| `raw_e5_cosine` | `0.806723` | `0.520573` | `0.500000` |
| `step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean` | `0.842017` | `0.588995` | `1.000000` |
| `step15_v5_public_noise_weighted_strong_phase4_seed_mean` | `0.904202` | `0.701809` | `1.000000` |
| `step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean` | `0.913725` | `0.738951` | `1.000000` |

Grouped-bootstrap intervals for the current clean main scorer:

| Metric | Observed | Bootstrap Mean | 95% CI |
| --- | ---: | ---: | ---: |
| ROC-AUC | `0.913725` | `0.913844` | `[0.837521, 0.968700]` |
| AP / PR-AUC / MAP | `0.738951` | `0.733387` | `[0.480227, 0.894451]` |
| MRR | `1.000000` | `0.998106` | `[1.000000, 1.000000]` |

## Paired Comparisons

Step 15 v5 domain-balanced versus Step 9 E5 mixup 100pct:

- ROC-AUC diff: `+0.071709`, 95% CI `[0.006751, 0.152423]`, supports positive difference: `yes`
- AP / PR-AUC / MAP diff: `+0.149956`, 95% CI `[-0.032434, 0.304387]`, supports positive difference: `no`
- MRR diff: `0.000000`, 95% CI `[0.000000, 0.000000]`, supports positive difference: `no`

Step 15 v5 domain-balanced versus raw E5:

- ROC-AUC diff: `+0.107003`, 95% CI `[-0.019282, 0.269503]`, supports positive difference: `no`
- AP / PR-AUC / MAP diff: `+0.218378`, 95% CI `[-0.105231, 0.485687]`, supports positive difference: `no`
- MRR diff: `+0.500000`, 95% CI `[0.000000, 0.800000]`, supports positive difference: `no`

## Interpretation

The current Step 12 v5 result supports a careful claim:

- Step 15 v5 domain-balanced is the strongest current clean fixed-test point-estimate scorer.
- Its ROC-AUC improvement over Step 9 E5 positive-pair mixup 100pct is supported by grouped paired bootstrap.
- Its AP / PR-AUC / MAP improvement over Step 9 mixup 100pct is positive but uncertainty-bounded because the confidence interval crosses zero.
- Its point estimates are stronger than raw E5, but the paired grouped-bootstrap intervals versus raw E5 still cross zero.
- MRR should be treated as a supplemental sanity check, not a primary claim, because the global first-positive rank saturates for several models.

The correct paper wording remains uncertainty-bounded: Step 15 v5 improves target-domain hard-negative robustness and gives the strongest current fixed-test point estimates, but the current `zh_test` positive count is still too small to claim broad statistically robust superiority over every raw semantic baseline.

## Run Command

```bash
python3 scripts/step12_statistical_robustness_audit.py \
  --labels reports/step5_zh_target_strict_frozen_silver_labels.csv \
  --features reports/step7_pair_features.zh_target_strict.csv \
  --resamples 5000 \
  --seed 20260513 \
  --output-json reports/step12_v5_statistical_robustness_zh_test_20260603.json \
  --output-metrics reports/step12_v5_statistical_robustness_model_metrics_20260603.csv \
  --output-comparisons reports/step12_v5_statistical_robustness_paired_comparisons_20260603.csv
```
