# Step 13 Concept Drift Audit

Generated at: `2026-06-06`

## Scope

This is a read-only audit. It joins Step 5 frozen supervision rows, Step 7 pair features, existing Step 7/9 predictions, and the current manifest-only Step 11 audit. It does not train a model and does not write labels back to Step 5.

## Dataset

| domain | joined_supervision | positive | negative | train | valid | test |
| --- | --- | --- | --- | --- | --- | --- |
| EN | 734 | 209 | 525 | {'positive': 116, 'negative': 285} | {'negative': 110, 'positive': 42} | {'positive': 51, 'negative': 130} |
| ZH | 522 | 96 | 426 | {'positive': 61, 'negative': 274} | {'positive': 14, 'negative': 67} | {'positive': 21, 'negative': 85} |


## Key Findings

- Frozen supervision is still small and imbalanced: EN 734 rows (209 positive / 525 negative); ZH 522 rows (96 positive / 426 negative).
- The largest EN->ZH marginal feature shifts are: digit_ratio_mean_raw_gap_abs (SMD=0.854419), repeated_description_share_percentile_gap_abs (SMD=-0.852279), repeated_title_share_percentile_gap_abs (SMD=-0.849901), repeated_title_share_raw_gap_abs (SMD=-0.676433), punct_ratio_mean_raw_gap_abs (SMD=0.673458).
- High-semantic negatives are not uniformly inflated in ZH under the EN-negative q90 E5 threshold; the ZH rate is similar: EN 53/525 (0.100952); ZH 35/426 (0.08216).
- Feature-specific high-negative rate deltas (ZH minus EN) are mixed, not a single semantic-collapse pattern: bge_m3: -0.061046; gte_multilingual_base: -0.009403; labse: 0.014071; multilingual_e5_large: -0.018792.
- Raw E5 remains stronger than Step 7 E5 fusion on fixed ZH test: raw AUC 0.806723 / AP 0.520573 vs fusion AUC 0.55014 / AP 0.384493.
- The current E5 LR/L2 few-shot seed-mean is only a small global ranking improvement over raw E5: AUC delta 0.012325 and AP delta 0.019921.
- The E5 LR/L2 positive-pair mixup 100pct seed-mean is the strongest current Step 9 minority-regularization baseline, with AUC delta 0.035294 and AP delta 0.068422 versus raw E5; Step 12 paired bootstrap decides whether this can be treated as a robust improvement.
- Step 15 v5 domain-balanced public-noise-weighted curriculum has the strongest current fixed-test point estimate, with AUC delta 0.107002 and AP delta 0.218378 versus raw E5; Step 12 v5 paired bootstrap supports its ROC-AUC improvement over Step 9 mixup100 but not yet over raw E5.
- Step 7 fusion diagnostics still show collapse/early-stop risk for 4/5 tracked experiments; this supports a source-domain shortcut/transfer drift diagnosis rather than a simple hyperparameter issue.
- Current manifest-only Step 11 cluster audit is dominated by non-controller evidence types: 68 template/topic clusters vs 0 anchored same-controller cores.

## Largest EN to ZH Feature Shifts

| feature_group | feature | mean_en | mean_zh | smd_zh_minus_en | ks_statistic |
| --- | --- | --- | --- | --- | --- |
| style_gap | digit_ratio_mean_raw_gap_abs | 0.024915 | 0.112342 | 0.854419 | 0.458131 |
| style_gap | repeated_description_share_percentile_gap_abs | 0.116047 | 0.02145 | -0.852279 | 0.385689 |
| style_gap | repeated_title_share_percentile_gap_abs | 0.095917 | 0.004797 | -0.849901 | 0.463983 |
| style_gap | repeated_title_share_raw_gap_abs | 0.111012 | 0.009721 | -0.676433 | 0.390841 |
| style_gap | punct_ratio_mean_raw_gap_abs | 0.031131 | 0.073857 | 0.673458 | 0.398619 |
| style_gap | digit_ratio_mean_percentile_gap_abs | 0.191704 | 0.352371 | 0.658126 | 0.306101 |
| style_gap | title_length_median_percentile_gap_abs | 0.177962 | 0.317737 | 0.641652 | 0.28839 |
| style_gap | punct_ratio_mean_percentile_gap_abs | 0.203969 | 0.357554 | 0.625685 | 0.253965 |
| semantic | embedding_cosine_bge_m3 | 0.888488 | 0.856805 | -0.603442 | 0.292054 |
| style_gap | max_category_share_percentile_gap_abs | 0.180058 | 0.356819 | 0.528869 | 0.361552 |
| semantic | embedding_cosine_paraphrase_multilingual_mpnet | 0.809767 | 0.739974 | -0.523472 | 0.269191 |
| identifier | has_shared_contact_exact | 0.230245 | 0.055556 | -0.515466 | 0.17469 |


## Label-Conditional Drift

Top positive-label shifts:

| feature_group | feature | mean_en | mean_zh | smd_zh_minus_en | ks_statistic |
| --- | --- | --- | --- | --- | --- |
| style_gap | repeated_description_share_percentile_gap_abs | 0.167058 | 0.017777 | -1.203035 | 0.63746 |
| style_gap | repeated_title_share_percentile_gap_abs | 0.167971 | 0.005791 | -1.171099 | 0.716856 |
| style_gap | repeated_title_share_raw_gap_abs | 0.221822 | 0.020162 | -1.054163 | 0.702203 |
| identifier | has_shared_contact_exact | 0.708134 | 0.270833 | -0.972785 | 0.437301 |
| structural | shared_category_count_capped | 1.631579 | 0.635417 | -0.758214 | 0.373754 |
| style_gap | digit_ratio_mean_raw_gap_abs | 0.020653 | 0.044992 | 0.710782 | 0.363786 |
| structural | candidate_rule_count_raw | 2.760766 | 2.020833 | -0.708148 | 0.296202 |
| style_gap | uppercase_ratio_mean_raw_gap_abs | 0.078778 | 0.029739 | -0.684912 | 0.367225 |


Top negative-label shifts:

| feature_group | feature | mean_en | mean_zh | smd_zh_minus_en | ks_statistic |
| --- | --- | --- | --- | --- | --- |
| style_gap | digit_ratio_mean_raw_gap_abs | 0.026612 | 0.127519 | 0.920908 | 0.499812 |
| style_gap | digit_ratio_mean_percentile_gap_abs | 0.204536 | 0.393906 | 0.753491 | 0.347015 |
| style_gap | repeated_title_share_percentile_gap_abs | 0.067233 | 0.004573 | -0.749396 | 0.367311 |
| style_gap | repeated_description_share_percentile_gap_abs | 0.095739 | 0.022277 | -0.722819 | 0.314286 |
| structural | sparse_lexical_similarity_raw | 0.474022 | 0.326928 | -0.722791 | 0.452636 |
| semantic | embedding_cosine_bge_m3 | 0.882815 | 0.847012 | -0.715842 | 0.332622 |
| style_gap | punct_ratio_mean_raw_gap_abs | 0.035711 | 0.082191 | 0.690734 | 0.429631 |
| style_gap | title_length_median_percentile_gap_abs | 0.186602 | 0.341743 | 0.67948 | 0.319812 |


## High-Semantic Negative Ratio

Thresholds are defined as the English negative q90 for each semantic feature. This asks whether target-domain negatives enter a source-domain high-similarity region more often.

| domain | feature | threshold | negative_n | high_semantic_negative_n | high_semantic_negative_rate | high_semantic_no_identifier_negative_n | high_semantic_template_no_identifier_negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | embedding_cosine_multilingual_e5_large | 0.952618 | 525 | 53 | 0.100952 | 53 | 0 |
| zh | embedding_cosine_multilingual_e5_large | 0.952618 | 426 | 35 | 0.08216 | 35 | 4 |
| en | embedding_cosine_bge_m3 | 0.939324 | 525 | 53 | 0.100952 | 53 | 1 |
| zh | embedding_cosine_bge_m3 | 0.939324 | 426 | 17 | 0.039906 | 17 | 3 |
| en | embedding_cosine_labse | 0.949055 | 525 | 53 | 0.100952 | 53 | 2 |
| zh | embedding_cosine_labse | 0.949055 | 426 | 49 | 0.115023 | 49 | 10 |
| en | embedding_cosine_gte_multilingual_base | 0.989466 | 525 | 53 | 0.100952 | 53 | 1 |
| zh | embedding_cosine_gte_multilingual_base | 0.989466 | 426 | 39 | 0.091549 | 39 | 11 |


## ZH Test Slice Performance

Slices with fewer than five positives or five negatives are marked unstable in the CSV/JSON and should be treated as diagnostics, not conclusions.

| slice_name | model_id | n | n_positive | n_negative | roc_auc | average_precision | delta_auc_vs_raw_e5 | delta_ap_vs_raw_e5 | unstable_slice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_zh_test | raw_e5_cosine | 106 | 21 | 85 | 0.806723 | 0.520573 |  |  | False |
| all_zh_test | step7_core_zero_shot_multilingual_e5_large | 106 | 21 | 85 | 0.55014 | 0.384493 | -0.256583 | -0.13608 | False |
| all_zh_test | step9_e5_lr_l2_50pct_seed_mean | 106 | 21 | 85 | 0.819048 | 0.540494 | 0.012325 | 0.019921 | False |
| all_zh_test | step9_identifier_augmented_lr_l2_100pct_seed_mean | 106 | 21 | 85 | 0.783754 | 0.647686 | -0.022969 | 0.127113 | False |
| all_zh_test | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 106 | 21 | 85 | 0.826891 | 0.549271 | 0.020168 | 0.028698 | False |
| all_zh_test | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 106 | 21 | 85 | 0.842017 | 0.588995 | 0.035294 | 0.068422 | False |
| all_zh_test | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 106 | 21 | 85 | 0.904202 | 0.701809 | 0.097479 | 0.181236 | False |
| all_zh_test | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 106 | 21 | 85 | 0.913725 | 0.738951 | 0.107002 | 0.218378 | False |
| identifier_present | raw_e5_cosine | 5 | 5 | 0 |  | 1.0 |  |  | True |
| identifier_present | step7_core_zero_shot_multilingual_e5_large | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_e5_lr_l2_50pct_seed_mean | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_identifier_augmented_lr_l2_100pct_seed_mean | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 5 | 5 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_absent | raw_e5_cosine | 101 | 16 | 85 | 0.809559 | 0.5029 |  |  | False |
| identifier_absent | step7_core_zero_shot_multilingual_e5_large | 101 | 16 | 85 | 0.53125 | 0.345732 | -0.278309 | -0.157168 | False |
| identifier_absent | step9_e5_lr_l2_50pct_seed_mean | 101 | 16 | 85 | 0.819118 | 0.511861 | 0.009559 | 0.008961 | False |
| identifier_absent | step9_identifier_augmented_lr_l2_100pct_seed_mean | 101 | 16 | 85 | 0.716176 | 0.459684 | -0.093383 | -0.043216 | False |
| identifier_absent | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 101 | 16 | 85 | 0.836765 | 0.533889 | 0.027206 | 0.030989 | False |
| identifier_absent | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 101 | 16 | 85 | 0.838971 | 0.549525 | 0.029412 | 0.046625 | False |
| identifier_absent | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 101 | 16 | 85 | 0.884559 | 0.564994 | 0.075 | 0.062094 | False |
| identifier_absent | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 101 | 16 | 85 | 0.893382 | 0.605713 | 0.083823 | 0.102813 | False |
| high_e5_semantic_no_identifier | raw_e5_cosine | 7 | 4 | 3 | 0.416667 | 0.566667 |  |  | True |
| high_e5_semantic_no_identifier | step7_core_zero_shot_multilingual_e5_large | 7 | 4 | 3 | 0.625 | 0.617857 | 0.208333 | 0.05119 | True |
| high_e5_semantic_no_identifier | step9_e5_lr_l2_50pct_seed_mean | 7 | 4 | 3 | 0.416667 | 0.667857 | 0.0 | 0.10119 | True |
| high_e5_semantic_no_identifier | step9_identifier_augmented_lr_l2_100pct_seed_mean | 7 | 4 | 3 | 0.25 | 0.617857 | -0.166667 | 0.05119 | True |
| high_e5_semantic_no_identifier | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 7 | 4 | 3 | 0.416667 | 0.667857 | 0.0 | 0.10119 | True |
| high_e5_semantic_no_identifier | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 7 | 4 | 3 | 0.333333 | 0.642857 | -0.083334 | 0.07619 | True |
| high_e5_semantic_no_identifier | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 7 | 4 | 3 | 1.0 | 1.0 | 0.583333 | 0.433333 | True |
| high_e5_semantic_no_identifier | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 7 | 4 | 3 | 1.0 | 1.0 | 0.583333 | 0.433333 | True |


## Step 7 Fusion Diagnostics

| experiment_name | best_iteration | collapse_guard_triggered | collapse_guard_reasons | unique_valid_probabilities | zh_test_auc | zh_test_ap | top_feature_importance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| core_zero_shot_default | 1 | True | best_iteration_below_minimum | 6 | 0.588235 | 0.448547 | shared_title_count_capped\|max_category_share_percentile_gap_abs\|reranker_score_gte_multilingual_reranker_base\|repeated_description_share_percentile_gap_abs\|same_market_raw_bool |
| core_zero_shot_bge_m3 | 1 | True | best_iteration_below_minimum | 6 | 0.601681 | 0.448761 | shared_title_count_capped\|max_category_share_percentile_gap_abs\|embedding_cosine_bge_m3\|profile_category_jaccard\|reranker_score_bge_reranker_v2_m3 |
| core_zero_shot_multilingual_e5_large | 1 | True | best_iteration_below_minimum\|insufficient_unique_valid_probabilities | 5 | 0.55014 | 0.384493 | shared_description_count_capped\|max_category_share_percentile_gap_abs\|profile_category_jaccard\|embedding_cosine_multilingual_e5_large\|boilerplate_ratio_gap_abs |
| core_zero_shot_default_no_structural | 54 | False |  | 116 | 0.623529 | 0.287652 | reranker_score_gte_multilingual_reranker_base\|embedding_cosine_gte_multilingual_base\|repeated_title_share_percentile_gap_abs\|max_category_share_percentile_gap_abs\|price_median_percentile_gap_abs |
| identifier_augmented_default | 1 | True | best_iteration_below_minimum | 6 | 0.606443 | 0.418989 | shared_description_count_capped\|has_shared_contact_exact\|repeated_title_share_percentile_gap_abs\|item_count_percentile_gap_abs\|max_category_share_percentile_gap_abs |


## Step 11 Evidence Context

| row_type | decision | count | current_summary_count | summary_selection_mode | unique_cluster_set_count |
| --- | --- | --- | --- | --- | --- |
| step11_cluster_audit_decision | same_controller_high_confidence | 0 |  | explicit | 79 |
| step11_cluster_audit_decision | same_controller_core_with_possible_expansion | 0 |  | explicit | 79 |
| step11_cluster_audit_decision | partial_anchor | 5 |  | explicit | 79 |
| step11_cluster_audit_decision | template_clone_not_controller | 33 |  | explicit | 79 |
| step11_cluster_audit_decision | semantic_topic_not_controller | 35 |  | explicit | 79 |
| step11_cluster_audit_decision | uncertain | 6 |  | explicit | 79 |


## Interpretation

The audit supports a concept-drift framing: source-domain fusion features are not simply weak; they encode source-domain shortcuts that do not transfer cleanly to Chinese target-domain pairs. Raw semantic ranking remains useful, but high-semantic target negatives and template-dense no-identifier slices explain why graph-level identity claims need reliability filtering and direct-anchor audit.

Current few-shot gains should be reported as slice-dependent diagnostics unless Step 12 bootstrap comparisons and future Step 11 reliability-filter reruns show stable improvements.
