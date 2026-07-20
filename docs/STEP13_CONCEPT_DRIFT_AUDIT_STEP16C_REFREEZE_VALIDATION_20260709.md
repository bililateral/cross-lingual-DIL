# Step 13 Concept Drift Audit

Generated at: `2026-07-09`

## Scope

This is a read-only audit. It joins Step 5 frozen supervision rows, Step 7 pair features, existing Step 7/9 predictions, and the current manifest-only Step 11 audit. It does not train a model and does not write labels back to Step 5.

## Dataset

| domain | joined_supervision | positive | negative | train | valid | test |
| --- | --- | --- | --- | --- | --- | --- |
| EN | 734 | 209 | 525 | {'positive': 116, 'negative': 285} | {'negative': 110, 'positive': 42} | {'positive': 51, 'negative': 130} |
| ZH | 778 | 309 | 469 | {'positive': 229, 'negative': 229} | {'positive': 30, 'negative': 90} | {'positive': 50, 'negative': 150} |


## Key Findings

- Frozen supervision is still small and imbalanced: EN 734 rows (209 positive / 525 negative); ZH 778 rows (309 positive / 469 negative).
- The largest EN->ZH marginal feature shifts are: repeated_description_share_percentile_gap_abs (SMD=-0.868537), repeated_title_share_percentile_gap_abs (SMD=-0.840919), digit_ratio_mean_raw_gap_abs (SMD=0.706297), repeated_title_share_raw_gap_abs (SMD=-0.664669), punct_ratio_mean_raw_gap_abs (SMD=0.542888).
- High-semantic negatives are not uniformly inflated in ZH under the EN-negative q90 E5 threshold; the ZH rate is lower: EN 53/525 (0.100952); ZH 35/469 (0.074627).
- Feature-specific high-negative rate deltas (ZH minus EN) are mixed, not a single semantic-collapse pattern: bge_m3: -0.064705; gte_multilingual_base: -0.005003; labse: 0.003526; multilingual_e5_large: -0.026325.
- Raw E5 remains stronger than Step 7 E5 fusion on fixed ZH test: raw AUC 0.748 / AP 0.542839 vs fusion AUC 0.623733 / AP 0.519158.
- The current E5 LR/L2 few-shot seed-mean is only a small global ranking improvement over raw E5: AUC delta -0.0168 and AP delta 0.014789.
- The E5 LR/L2 positive-pair mixup 100pct seed-mean is the strongest current Step 9 minority-regularization baseline, with AUC delta 0.014267 and AP delta 0.013356 versus raw E5; Step 12 paired bootstrap decides whether this can be treated as a robust improvement.
- Step 15 v5 domain-balanced public-noise-weighted curriculum has the strongest current fixed-test point estimate, with AUC delta 0.134 and AP delta 0.162544 versus raw E5; Step 12 v5 paired bootstrap supports its ROC-AUC improvement over Step 9 mixup100 but not yet over raw E5.
- Step 7 fusion diagnostics still show collapse/early-stop risk for 4/5 tracked experiments; this supports a source-domain shortcut/transfer drift diagnosis rather than a simple hyperparameter issue.
- Current manifest-only Step 11 cluster audit is dominated by non-controller evidence types: 143 template/topic clusters vs 2 anchored same-controller cores.

## Largest EN to ZH Feature Shifts

| feature_group | feature | mean_en | mean_zh | smd_zh_minus_en | ks_statistic |
| --- | --- | --- | --- | --- | --- |
| style_gap | repeated_description_share_percentile_gap_abs | 0.116047 | 0.020053 | -0.868537 | 0.397274 |
| style_gap | repeated_title_share_percentile_gap_abs | 0.095917 | 0.005753 | -0.840919 | 0.467765 |
| style_gap | digit_ratio_mean_raw_gap_abs | 0.024915 | 0.089592 | 0.706297 | 0.369346 |
| style_gap | repeated_title_share_raw_gap_abs | 0.111012 | 0.010923 | -0.664669 | 0.379027 |
| style_gap | punct_ratio_mean_raw_gap_abs | 0.031131 | 0.064201 | 0.542888 | 0.320346 |
| style_gap | uppercase_ratio_mean_raw_gap_abs | 0.09254 | 0.042443 | -0.52619 | 0.33143 |
| structural | shared_category_count_capped | 1.141689 | 0.616967 | -0.485347 | 0.18585 |
| style_gap | title_length_median_percentile_gap_abs | 0.177962 | 0.276589 | 0.464979 | 0.205344 |
| style_gap | digit_ratio_mean_percentile_gap_abs | 0.191704 | 0.298377 | 0.457073 | 0.216933 |
| style_gap | description_length_median_raw_gap_abs | 390.101499 | 98.3991 | -0.435721 | 0.292366 |
| semantic | embedding_cosine_bge_m3 | 0.888488 | 0.866252 | -0.42744 | 0.231096 |
| style_gap | max_category_share_percentile_gap_abs | 0.180058 | 0.312943 | 0.409902 | 0.307124 |


## Label-Conditional Drift

Top positive-label shifts:

| feature_group | feature | mean_en | mean_zh | smd_zh_minus_en | ks_statistic |
| --- | --- | --- | --- | --- | --- |
| style_gap | repeated_description_share_percentile_gap_abs | 0.167058 | 0.016016 | -1.219666 | 0.680138 |
| style_gap | repeated_title_share_percentile_gap_abs | 0.167971 | 0.006145 | -1.168574 | 0.717564 |
| style_gap | repeated_title_share_raw_gap_abs | 0.221822 | 0.014613 | -1.097107 | 0.700686 |
| identifier | has_shared_contact_exact | 0.708134 | 0.265372 | -0.988034 | 0.442762 |
| style_gap | uppercase_ratio_mean_raw_gap_abs | 0.078778 | 0.018669 | -0.888525 | 0.450767 |
| style_gap | repeated_description_share_raw_gap_abs | 0.222512 | 0.052749 | -0.860852 | 0.706895 |
| semantic | reranker_score_bge_reranker_v2_m3 | 0.395031 | 0.197622 | -0.833678 | 0.376179 |
| identifier | shared_contact_count_capped | 0.899522 | 0.333333 | -0.732844 | 0.442762 |


Top negative-label shifts:

| feature_group | feature | mean_en | mean_zh | smd_zh_minus_en | ks_statistic |
| --- | --- | --- | --- | --- | --- |
| style_gap | digit_ratio_mean_raw_gap_abs | 0.026612 | 0.118936 | 0.864562 | 0.453049 |
| structural | sparse_lexical_similarity_raw | 0.474022 | 0.3087 | -0.80583 | 0.482416 |
| style_gap | repeated_title_share_percentile_gap_abs | 0.067233 | 0.005496 | -0.738235 | 0.368387 |
| semantic | embedding_cosine_bge_m3 | 0.882815 | 0.846883 | -0.730657 | 0.351016 |
| style_gap | repeated_description_share_percentile_gap_abs | 0.095739 | 0.022713 | -0.719983 | 0.314286 |
| style_gap | digit_ratio_mean_percentile_gap_abs | 0.204536 | 0.377783 | 0.69407 | 0.324662 |
| structural | structural_support_score_raw | 0.621423 | 0.476736 | -0.679918 | 0.29032 |
| style_gap | max_category_share_percentile_gap_abs | 0.167181 | 0.382531 | 0.638252 | 0.387946 |


## High-Semantic Negative Ratio

Thresholds are defined as the English negative q90 for each semantic feature. This asks whether target-domain negatives enter a source-domain high-similarity region more often.

| domain | feature | threshold | negative_n | high_semantic_negative_n | high_semantic_negative_rate | high_semantic_no_identifier_negative_n | high_semantic_template_no_identifier_negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | embedding_cosine_multilingual_e5_large | 0.952618 | 525 | 53 | 0.100952 | 53 | 0 |
| zh | embedding_cosine_multilingual_e5_large | 0.952618 | 469 | 35 | 0.074627 | 35 | 4 |
| en | embedding_cosine_bge_m3 | 0.939324 | 525 | 53 | 0.100952 | 53 | 1 |
| zh | embedding_cosine_bge_m3 | 0.939324 | 469 | 17 | 0.036247 | 17 | 3 |
| en | embedding_cosine_labse | 0.949055 | 525 | 53 | 0.100952 | 53 | 2 |
| zh | embedding_cosine_labse | 0.949055 | 469 | 49 | 0.104478 | 49 | 10 |
| en | embedding_cosine_gte_multilingual_base | 0.989466 | 525 | 53 | 0.100952 | 53 | 1 |
| zh | embedding_cosine_gte_multilingual_base | 0.989466 | 469 | 45 | 0.095949 | 45 | 12 |


## ZH Test Slice Performance

Slices with fewer than five positives or five negatives are marked unstable in the CSV/JSON and should be treated as diagnostics, not conclusions.

| slice_name | model_id | n | n_positive | n_negative | roc_auc | average_precision | delta_auc_vs_raw_e5 | delta_ap_vs_raw_e5 | unstable_slice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_zh_test | raw_e5_cosine | 200 | 50 | 150 | 0.748 | 0.542839 |  |  | False |
| all_zh_test | step7_core_zero_shot_multilingual_e5_large | 200 | 50 | 150 | 0.623733 | 0.519158 | -0.124267 | -0.023681 | False |
| all_zh_test | step9_e5_lr_l2_50pct_seed_mean | 200 | 50 | 150 | 0.7312 | 0.557628 | -0.0168 | 0.014789 | False |
| all_zh_test | step9_identifier_augmented_lr_l2_100pct_seed_mean | 200 | 50 | 150 | 0.869467 | 0.789611 | 0.121467 | 0.246772 | False |
| all_zh_test | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 200 | 50 | 150 | 0.7312 | 0.557628 | -0.0168 | 0.014789 | False |
| all_zh_test | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 200 | 50 | 150 | 0.762267 | 0.556195 | 0.014267 | 0.013356 | False |
| all_zh_test | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 200 | 50 | 150 | 0.862267 | 0.733382 | 0.114267 | 0.190543 | False |
| all_zh_test | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 200 | 50 | 150 | 0.882 | 0.705383 | 0.134 | 0.162544 | False |
| identifier_present | raw_e5_cosine | 21 | 21 | 0 |  | 1.0 |  |  | True |
| identifier_present | step7_core_zero_shot_multilingual_e5_large | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_e5_lr_l2_50pct_seed_mean | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_identifier_augmented_lr_l2_100pct_seed_mean | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_present | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 21 | 21 | 0 |  | 1.0 |  | 0.0 | True |
| identifier_absent | raw_e5_cosine | 179 | 29 | 150 | 0.810805 | 0.514589 |  |  | False |
| identifier_absent | step7_core_zero_shot_multilingual_e5_large | 179 | 29 | 150 | 0.529655 | 0.320314 | -0.28115 | -0.194275 | False |
| identifier_absent | step9_e5_lr_l2_50pct_seed_mean | 179 | 29 | 150 | 0.772874 | 0.49896 | -0.037931 | -0.015629 | False |
| identifier_absent | step9_identifier_augmented_lr_l2_100pct_seed_mean | 179 | 29 | 150 | 0.775862 | 0.473377 | -0.034943 | -0.041212 | False |
| identifier_absent | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 179 | 29 | 150 | 0.772874 | 0.49896 | -0.037931 | -0.015629 | False |
| identifier_absent | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 179 | 29 | 150 | 0.766207 | 0.444381 | -0.044598 | -0.070208 | False |
| identifier_absent | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 179 | 29 | 150 | 0.816092 | 0.526227 | 0.005287 | 0.011638 | False |
| identifier_absent | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 179 | 29 | 150 | 0.845057 | 0.515803 | 0.034252 | 0.001214 | False |
| high_e5_semantic_no_identifier | raw_e5_cosine | 15 | 7 | 8 | 0.607143 | 0.711844 |  |  | False |
| high_e5_semantic_no_identifier | step7_core_zero_shot_multilingual_e5_large | 15 | 7 | 8 | 0.598214 | 0.569741 | -0.008929 | -0.142103 | False |
| high_e5_semantic_no_identifier | step9_e5_lr_l2_50pct_seed_mean | 15 | 7 | 8 | 0.732143 | 0.79325 | 0.125 | 0.081406 | False |
| high_e5_semantic_no_identifier | step9_identifier_augmented_lr_l2_100pct_seed_mean | 15 | 7 | 8 | 0.678571 | 0.785714 | 0.071428 | 0.07387 | False |
| high_e5_semantic_no_identifier | step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean | 15 | 7 | 8 | 0.732143 | 0.79325 | 0.125 | 0.081406 | False |
| high_e5_semantic_no_identifier | step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean | 15 | 7 | 8 | 0.642857 | 0.714744 | 0.035714 | 0.0029 | False |
| high_e5_semantic_no_identifier | step15_v5_public_noise_weighted_strong_phase4_seed_mean | 15 | 7 | 8 | 0.75 | 0.799743 | 0.142857 | 0.087899 | False |
| high_e5_semantic_no_identifier | step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean | 15 | 7 | 8 | 0.589286 | 0.665079 | -0.017857 | -0.046765 | False |


## Step 7 Fusion Diagnostics

| experiment_name | best_iteration | collapse_guard_triggered | collapse_guard_reasons | unique_valid_probabilities | zh_test_auc | zh_test_ap | top_feature_importance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| core_zero_shot_default | 1 | True | best_iteration_below_minimum | 6 | 0.604 | 0.490156 | shared_title_count_capped\|max_category_share_percentile_gap_abs\|reranker_score_gte_multilingual_reranker_base\|repeated_description_share_percentile_gap_abs\|same_market_raw_bool |
| core_zero_shot_bge_m3 | 1 | True | best_iteration_below_minimum | 6 | 0.6042 | 0.494441 | shared_title_count_capped\|max_category_share_percentile_gap_abs\|embedding_cosine_bge_m3\|profile_category_jaccard\|reranker_score_bge_reranker_v2_m3 |
| core_zero_shot_multilingual_e5_large | 1 | True | best_iteration_below_minimum\|insufficient_unique_valid_probabilities | 5 | 0.623733 | 0.519158 | shared_description_count_capped\|max_category_share_percentile_gap_abs\|profile_category_jaccard\|embedding_cosine_multilingual_e5_large\|boilerplate_ratio_gap_abs |
| core_zero_shot_default_no_structural | 54 | False |  | 116 | 0.7688 | 0.514122 | reranker_score_gte_multilingual_reranker_base\|embedding_cosine_gte_multilingual_base\|repeated_title_share_percentile_gap_abs\|max_category_share_percentile_gap_abs\|price_median_percentile_gap_abs |
| identifier_augmented_default | 1 | True | best_iteration_below_minimum | 6 | 0.6912 | 0.599287 | shared_description_count_capped\|has_shared_contact_exact\|repeated_title_share_percentile_gap_abs\|item_count_percentile_gap_abs\|max_category_share_percentile_gap_abs |


## Step 11 Evidence Context

| row_type | decision | count | current_summary_count | summary_selection_mode | unique_cluster_set_count |
| --- | --- | --- | --- | --- | --- |
| step11_cluster_audit_decision | same_controller_high_confidence | 0 |  | explicit | 161 |
| step11_cluster_audit_decision | same_controller_core_with_possible_expansion | 2 |  | explicit | 161 |
| step11_cluster_audit_decision | partial_anchor | 7 |  | explicit | 161 |
| step11_cluster_audit_decision | template_clone_not_controller | 76 |  | explicit | 161 |
| step11_cluster_audit_decision | semantic_topic_not_controller | 67 |  | explicit | 161 |
| step11_cluster_audit_decision | uncertain | 9 |  | explicit | 161 |


## Interpretation

The audit supports a concept-drift framing: source-domain fusion features are not simply weak; they encode source-domain shortcuts that do not transfer cleanly to Chinese target-domain pairs. Raw semantic ranking remains useful, but high-semantic target negatives and template-dense no-identifier slices explain why graph-level identity claims need reliability filtering and direct-anchor audit.

Current few-shot gains should be reported as slice-dependent diagnostics unless Step 12 bootstrap comparisons and future Step 11 reliability-filter reruns show stable improvements.
