# Step 7: Source-Domain Seller-Pair Verification

Step 7 trains the source-domain seller-pair verifier and exposes it to `zh_target_strict` as the first zero-shot transfer check.

Run Step 7 on the Linux training/runtime server. Use the Windows sync workspace only for audit and synchronization.

## Role

Step 7 learns a pairwise verification function:

- input: two seller profiles
- output: whether the pair should be judged `positive` or `negative`

What is transferred later is this relation function, not direct English-to-Chinese seller alignment.

## Inputs

Step 7 uses:

- Step 3 seller profiles
- Step 5 frozen pair supervision
- Step 7 pair-feature tables

Pool boundaries remain:

- source training pool: `en_content_train_pool`
- strict target pool: `zh_target_strict`
- auxiliary target pool: `zh_target_aux`

## Transfer-Safe Features

The core feature view still consists of:

- relation-level structural overlap
- market-relative style and scale gaps
- multilingual embedding cosine
- multilingual reranker score

The active pair tables also include boilerplate-aware controls:

- `shared_title_idf_sum`
- `shared_description_idf_sum`
- `shared_title_idf_mean`
- `shared_description_idf_mean`
- `boilerplate_ratio_max`
- `boilerplate_ratio_gap_abs`
- `shared_boilerplate_count`
- `shared_low_df_sentence_count`
- `shared_rare_ngram_count`

## Active Experiment Views

Current synchronized Step 7 experiments:

- `core_zero_shot_default`
- `core_zero_shot_bge_m3`
- `core_zero_shot_bge_m3_embedding_only`
- `identifier_augmented_default`
- `en_only_ablation_default`
- `core_zero_shot_multilingual_e5_large`
- `core_zero_shot_multilingual_e5_large_plus_gte_reranker`
- `core_zero_shot_labse`
- `core_zero_shot_labse_plus_gte_reranker`
- `core_zero_shot_paraphrase_multilingual_mpnet`
- `core_zero_shot_paraphrase_multilingual_mpnet_plus_gte_reranker`
- `core_zero_shot_default_no_reranker`
- `core_zero_shot_default_reranker_only`
- `core_zero_shot_default_no_semantics`
- `core_zero_shot_default_no_style_gap`
- `core_zero_shot_default_no_structural`
- `core_zero_shot_default_raw_style_gap_control`

## Training Stability Status

The earlier small-validation failure has been repaired, but the one-tree LightGBM fusion collapse is not fully repaired.

The active training path uses:

- label-stratified English validation split repair
- enlarged English valid/test containers from the `2026-04-23` English valid/test top-up
- small-validation guard diagnostics for the repaired English `valid` split
- post-train iteration scanning
- collapse diagnostics in the synchronized summary

The `small_validation_guard` is now off for all `17` selected experiments. However, `collapse_guard` is still triggered for `10 / 17` experiments because several LightGBM fusion models select shallow solutions. Current one-iteration examples include `core_zero_shot_default`, `core_zero_shot_bge_m3`, `core_zero_shot_bge_m3_embedding_only`, `core_zero_shot_multilingual_e5_large`, `core_zero_shot_labse`, `core_zero_shot_default_no_reranker`, `core_zero_shot_default_reranker_only`, and `identifier_augmented_default`.

This is a scientific result, not a synchronization defect: the enlarged English source split fixed the minimum validation-size explanation risk, but the current transfer-safe feature fusion still struggles to learn a robust source-domain tree ensemble that transfers to the Chinese strict target.

## Current Reading

The readings below are current for the `2026-04-23` English valid/test top-up Step 5 refreeze and were rechecked locally on `2026-05-13`.

Current synchronized inputs:

- semantic summary: `reports/step7_semantic_feature_summary.json`
- training summary: `reports/step7_training_summary.json`
- pair-feature rows: `en_content_train_pool = 6683`, `zh_target_strict = 3857`, `zh_target_aux = 580`
- current test containers: English `181` rows (`51 positive / 130 negative`), Chinese strict `106` rows (`21 positive / 85 negative`)
- complete semantic columns: five embeddings plus two rerankers

Current Chinese strict zero-shot test results:

- `core_zero_shot_default`: `balanced_accuracy = 0.562465`, `roc_auc = 0.588235`, `average_precision = 0.448547`, `best_iteration = 1`
- `core_zero_shot_bge_m3`: `balanced_accuracy = 0.562465`, `roc_auc = 0.601681`, `average_precision = 0.448761`, `best_iteration = 1`
- `core_zero_shot_bge_m3_embedding_only`: `balanced_accuracy = 0.562465`, `roc_auc = 0.601681`, `average_precision = 0.448761`, `best_iteration = 1`
- `core_zero_shot_multilingual_e5_large`: `balanced_accuracy = 0.538655`, `roc_auc = 0.550140`, `average_precision = 0.384493`, `best_iteration = 1`
- `core_zero_shot_multilingual_e5_large_plus_gte_reranker`: `balanced_accuracy = 0.578711`, `roc_auc = 0.549580`, `average_precision = 0.384533`, `best_iteration = 65`
- `core_zero_shot_labse`: `balanced_accuracy = 0.538655`, `roc_auc = 0.551541`, `average_precision = 0.383870`, `best_iteration = 1`
- `core_zero_shot_labse_plus_gte_reranker`: `balanced_accuracy = 0.527731`, `roc_auc = 0.584874`, `average_precision = 0.387568`, `best_iteration = 7`
- `core_zero_shot_paraphrase_multilingual_mpnet`: `balanced_accuracy = 0.596359`, `roc_auc = 0.557983`, `average_precision = 0.387651`, `best_iteration = 69`
- `core_zero_shot_paraphrase_multilingual_mpnet_plus_gte_reranker`: `balanced_accuracy = 0.524650`, `roc_auc = 0.604482`, `average_precision = 0.366364`, `best_iteration = 47`
- `core_zero_shot_default_no_semantics`: `balanced_accuracy = 0.527731`, `roc_auc = 0.535574`, `average_precision = 0.365896`, `best_iteration = 11`
- `core_zero_shot_default_no_style_gap`: `balanced_accuracy = 0.526331`, `roc_auc = 0.517927`, `average_precision = 0.375092`, `best_iteration = 2`
- `core_zero_shot_default_no_structural`: `balanced_accuracy = 0.572269`, `roc_auc = 0.623529`, `average_precision = 0.287652`, `best_iteration = 54`
- `core_zero_shot_default_raw_style_gap_control`: `balanced_accuracy = 0.513165`, `roc_auc = 0.503641`, `average_precision = 0.325214`, `best_iteration = 57`
- `identifier_augmented_default`: `balanced_accuracy = 0.619888`, `roc_auc = 0.606443`, `average_precision = 0.418989`, `best_iteration = 1`

## Interpretation

The current Step 7 reading is:

- the technical sync issues are fixed: semantic features are complete, E5/LaBSE/Paraphrase embedding-only views are valid, and the English validation container is no longer below the small-validation threshold
- the LightGBM fusion collapse remains a current modeling limitation for many source-domain fusion models
- clean zero-shot transfer is weaker than earlier small-test readings
- `core_zero_shot_bge_m3` is no longer the strongest current clean zero-shot baseline
- `core_zero_shot_default_no_structural` is the best clean Step 7 ROC-AUC ablation, while `core_zero_shot_default_raw_style_gap_control` remains a non-mainline diagnostic control
- the identifier-augmented line remains an operational control, not the clean scientific claim
- raw semantic rankings remain stronger reporting baselines than the current Step 7 LightGBM fusion: raw E5 ROC-AUC `0.806723`, raw LaBSE `0.806162`, raw BGE-M3 `0.783754`

## Relation To Later Steps

- Step 8 uses the Step 7 baselines for strict zero-shot transfer evaluation
- Step 9 few-shot adapts these baselines with limited Chinese supervision
- Step 9 calibration keeps these baselines frozen and calibrates score space
- Step 11 projects selected Step 7 / Step 9 scorers into the Chinese candidate graph
