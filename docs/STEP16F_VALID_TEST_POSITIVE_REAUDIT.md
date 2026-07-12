# Step16F Valid/Test Positive Reaudit

Date: 2026-07-09

## Scope

This audit rechecks only the current `zh_target_strict` validation and test positive rows after the Step16C/E refreeze.
It does not modify Step5 labels. Its purpose is to stratify positive labels by evidence strength for paper reporting.

## Summary

- Audited positive rows: `80`
- Split counts: `{"valid": 30, "test": 50}`
- Paper bucket counts: `{"direct_or_component_primary": 22, "soft_primary_or_slice": 14, "secondary_or_sensitivity_only": 44}`
- Rows needing manual recheck before a strongest paper claim: `50`

## Evidence-Tier Counts

| Value | Count |
|---|---:|
| `weak_component_or_semantic_positive` | 31 |
| `gold_direct_seller_contact` | 19 |
| `strong_soft_structural_clone` | 11 |
| `weak_soft_positive_needs_reaudit` | 6 |
| `soft_product_data_clone_not_direct_identity` | 4 |
| `component_or_contact_supported_soft_positive` | 3 |
| `moderate_soft_structural_positive` | 3 |
| `gold_direct_seller_contact_weaker_type` | 2 |
| `gold_component_anchor` | 1 |

## Evidence-Tier Counts by Split

| Split | `component_or_contact_supported_soft_positive` | `gold_component_anchor` | `gold_direct_seller_contact` | `gold_direct_seller_contact_weaker_type` | `moderate_soft_structural_positive` | `soft_product_data_clone_not_direct_identity` | `strong_soft_structural_clone` | `weak_component_or_semantic_positive` | `weak_soft_positive_needs_reaudit` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test` | 3 | 1 | 16 | 1 | 1 | 4 | 3 | 19 | 2 |
| `valid` | 0 | 0 | 3 | 1 | 2 | 0 | 8 | 12 | 4 |

## Recommended-Use Counts

| Value | Count |
|---|---:|
| `sensitivity_only_or_reaudit` | 31 |
| `primary_gold_benchmark` | 19 |
| `primary_gold_benchmark_soft_slice` | 11 |
| `sensitivity_only_or_exclude_from_primary` | 6 |
| `secondary_or_sensitivity_only` | 4 |
| `primary_gold_benchmark_with_soft_slice` | 3 |
| `secondary_or_slice_reported_gold` | 3 |
| `primary_gold_benchmark_with_contact_slice` | 2 |
| `primary_gold_benchmark_component_slice` | 1 |

## Recommended-Use Counts by Split

| Split | `primary_gold_benchmark` | `primary_gold_benchmark_component_slice` | `primary_gold_benchmark_soft_slice` | `primary_gold_benchmark_with_contact_slice` | `primary_gold_benchmark_with_soft_slice` | `secondary_or_sensitivity_only` | `secondary_or_slice_reported_gold` | `sensitivity_only_or_exclude_from_primary` | `sensitivity_only_or_reaudit` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test` | 16 | 1 | 3 | 1 | 3 | 4 | 1 | 2 | 19 |
| `valid` | 3 | 0 | 8 | 1 | 0 | 0 | 2 | 4 | 12 |

## Risk-Flag Counts

| Value | Count |
|---|---:|
| `weak_soft_positive` | 37 |
| `contact_context_also_mentions_data_product` | 14 |
| `not_direct_identity` | 14 |
| `product_data_email_not_seller_identity` | 4 |
| `direct_contact_not_in_pair_feature` | 3 |

## Interpretation

The current validation/test positives are usable for continued experiments, but they should not be reported as one undifferentiated gold class.
The paper should report at least these positive slices:

1. `gold_direct_seller_contact`
2. `gold_component_anchor`
3. `strong_soft_structural_clone`
4. softer or risk-flagged positives used only in secondary/sensitivity analysis

Rows flagged as `product_data_email_not_seller_identity` should not be described as direct identity-anchor positives. They can remain as clone/soft positives only if the cloned listing evidence is accepted by the annotation protocol.

The strictest primary-positive subset is `direct_or_component_primary`.
The broader internal benchmark can additionally include `soft_primary_or_slice`, but this must be stated explicitly because these rows are not direct seller-identity anchors.
Rows in `secondary_or_sensitivity_only` should be used for sensitivity analysis or manual follow-up, not for the strongest paper claim.

## Outputs

- CSV: `reports/step16f_valid_test_positive_reaudit.csv`
- JSON summary: `reports/step16f_valid_test_positive_reaudit_summary.json`
