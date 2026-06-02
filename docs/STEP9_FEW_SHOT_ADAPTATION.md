# Step 9: Chinese Few-Shot Adaptation

Status: current-boundary rerun synchronized after the `2026-04-23` English valid/test top-up; locally rechecked on `2026-05-13`; run any new Step 9 training on the Linux training/runtime server only

This page covers the Step 9 few-shot branch. The old mixed-source LightGBM runs are now legacy controls; the new candidate branch uses frozen Step 7 scoring plus smooth logistic/residual adaptation.

## Role

Step 9 asks whether a small amount of reviewed Chinese supervision improves the synchronized Step 7 pair scorer on `zh_target_strict`.

It does not:

- reopen Step 5 automatically
- redefine the benchmark
- change the Step 7 pair-feature schema

## Active Data Boundary

The active Chinese strict supervision container after the `2026-04-23` English valid/test top-up refreeze is:

- `train = 335`
- `valid = 81`
- `test = 106`

Current Chinese strict split label counts:

- `train = 61 positive / 274 negative`
- `valid = 14 positive / 67 negative`
- `test = 21 positive / 85 negative`

2026-04-21 boundary-expansion note:

- the earlier `38`-row Chinese strict test split was too small for a confirmation-level few-shot improvement claim
- Step 5 boundary expansion has now been reviewed, applied, and refrozen
- the Codex review added `261` hard negatives and `259` uncertain rows, but no new positives under the conservative identity-closure rubric
- the first follow-up positive-anchor pass added `13` positive and `3` uncertain review labels; `5` of those positives are closure-derived audit-only
- the direct-identity v2 pass added `104` uncertain labels and `1` additional closure-derived audit-only positive; it added no primary positive supervision because the mined URL/email evidence was product/victim-data content, not seller identity evidence
- Chinese strict supervision remains `522` rows with `96` primary positive supervision rows
- Step 4 now contains `64` net-new Chinese candidate pairs across the positive-anchor passes, and the refreshed Step 7 pair-feature table has `3857` `zh_target_strict` rows
- current positive targets remain unmet: `zh_train positive = 61` versus target `100+`; `zh_valid positive = 14` versus target `25+`
- Step 7 and Step 9 have now both been rerun on Linux against this boundary and synchronized back
- few-shot promotion remains ROC-AUC/AP first; balanced accuracy is still a threshold-conditioned diagnostic
- the current L2/residual few-shot improvement is real against collapsed Step 7 LightGBM fusion baselines, but still modest against raw semantic E5 ranking; direct identity positives remain scarce

After the `2026-04-23` English valid/test top-up, the synchronized English source supervision container is:

- `train = 401`
- `valid = 152`
- `test = 181`

This replaces the earlier tiny English container (`85 / 19 / 19`) and the intermediate `251 / 75 / 114`, `280 / 77 / 119` boundaries.

## Training Recipe

The active few-shot runner keeps:

- the referenced Step 7 experiment fixed
- English source `train` rows only for backends that explicitly allow source training
- sampled Chinese few-shot `train` rows
- fixed Chinese `valid`
- fixed Chinese `test`

Current important repairs:

- the low-ratio guard is active
- `10% / 20%` are no longer shallow-tree artifacts
- the runner now also asserts that sampled `zh_train` sellers never overlap the fixed `zh_valid` / `zh_test` sellers
- the new hard-boundary sampler builds a curated support set from frozen Step 7 scores: high-score negatives, low-score positives, and typical anchors
- the new residual branch freezes `core_zero_shot_bge_m3` and learns only a small L2-regularized correction layer
- the new direct LR branch uses the same NumPy L2 logistic solver without adding a sklearn dependency
- the E5 positive-pair mixup branch is a training-only data augmentation control: it interpolates Chinese `zh_train` positive pair feature vectors, never writes synthetic rows into Step 5 labels, and never touches `zh_valid` or `zh_test`
- `reports/step9_few_shot_summary.json` is now merge-safe across repeated commands: runs from the same Step 5/7 input fingerprint are merged, while a changed input fingerprint causes the old summary to be backed up before replacement
- the `2026-04-22` policy no longer centers Step 9 only on BGE-M3; it adds default/GTE, no-structural clean ablation, BGE embedding-only, GTE embedding-only/reranker-only, E5/LaBSE/Paraphrase embedding-only and `+gte_reranker` controls, plus an identifier-augmented operational LR/L2 control

## Synchronized Experiments

Current synchronized few-shot experiments:

- `core_few_shot_default_residual_lr`
- `core_few_shot_default_lr_l2`
- `core_few_shot_bge_m3_residual_lr`
- `core_few_shot_bge_m3_no_semantics_residual_lr`
- `core_few_shot_bge_m3_lr_l2`
- `core_few_shot_bge_m3_no_semantics_lr_l2`
- `core_few_shot_bge_m3_embedding_only_lr_l2`
- `core_few_shot_default_no_structural_residual_lr`
- `core_few_shot_default_no_structural_lr_l2`
- `core_few_shot_default_no_reranker_lr_l2`
- `core_few_shot_default_reranker_only_lr_l2`
- `core_few_shot_multilingual_e5_large_lr_l2`
- `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup`
- `core_few_shot_multilingual_e5_large_plus_gte_reranker_lr_l2`
- `core_few_shot_labse_lr_l2`
- `core_few_shot_labse_plus_gte_reranker_lr_l2`
- `core_few_shot_paraphrase_multilingual_mpnet_lr_l2`
- `core_few_shot_paraphrase_multilingual_mpnet_plus_gte_reranker_lr_l2`
- `identifier_augmented_few_shot_default_lr_l2`

Each selected experiment has `12` synchronized runs: ratios `0.1 / 0.2 / 0.5 / 1.0` crossed with seeds `20260320 / 20260321 / 20260322`.

## Positive-Pair Mixup Design

The mixup experiment is:

- `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup`

It is a data augmentation strategy, but only in feature space. It does not rewrite item text, generate new sellers, create new review labels, or expand the frozen Step 5 benchmark. It takes two real Chinese training positive pair feature vectors and interpolates them:

```text
x_new = (1 - lambda) * x_positive_i + lambda * x_positive_j
y_new = positive
```

Here `x_positive_i` and `x_positive_j` are Step 7 pair-level feature vectors from sampled `zh_train`. The feature vector includes E5 semantic cosine plus transfer-safe structural, overlap, and style-gap features. The synthetic row is therefore a new training point in the seller-pair representation space, not a new real-world seller pair.

The purpose is to regularize the minority positive region. Chinese strict supervision is imbalanced:

- `zh_train = 61 positive / 274 negative`
- `zh_valid = 14 positive / 67 negative`
- `zh_test = 21 positive / 85 negative`

Without this regularization, LR/L2 has very few target-domain positive examples to define the positive boundary, while many high-semantic negative examples come from template reuse and same-topic sellers. Mixup gives the positive class a smoother local geometry without relaxing the conservative positive-label standard.

The current branch is source-retained adaptation, not target-only training. For the `100pct` run the actual training matrix is:

```text
English source train: 401 rows
Chinese zh_train: 335 rows = 61 positive / 274 negative
Synthetic train-only positives: 122 rows
Final training matrix: 858 rows = 299 positive / 559 negative
```

Safety constraints:

- only sampled `zh_train` positives are eligible
- require `usable_for_core_transfer = 1`
- require `core_transfer_eligible = 1`
- exclude `positive_component_closure_audit`
- exclude `audit_only`, `audit_only_soft_alias`, and `uncertain_holdout`
- never write synthetic rows to Step 5 frozen labels
- never use `zh_valid` or `zh_test`
- mark the generated artifact as `synthetic_train_only`

Current effect:

- `50pct` mixup reaches ROC-AUC `0.816246` to `0.831373`, AP `0.530994` to `0.558135`
- `100pct` mixup reaches ROC-AUC `0.839216` to `0.844818`, AP `0.579418` to `0.593742`

Interpretation: mixup is the strongest clean Step 9 point estimate so far, especially on AP, but Step 12 grouped bootstrap still does not support a statistically robust claim that it beats raw E5 semantic ranking.

## Remote Linux Commands

Use these only when deliberately regenerating the current boundary. The current synchronized artifacts already exist.

```bash
python3 scripts/step7_build_pair_feature_preview.py

python3 scripts/step7_build_semantic_pair_features.py \
  --pool en_content_train_pool \
  --pool zh_target_strict \
  --pool zh_target_aux \
  --embedding-model gte_multilingual_base \
  --embedding-model bge_m3 \
  --embedding-model multilingual_e5_large \
  --embedding-model labse \
  --embedding-model paraphrase_multilingual_mpnet_base_v2 \
  --reranker-model gte_multilingual_reranker_base \
  --reranker-model bge_reranker_v2_m3

python3 scripts/step7_train_baseline_models.py

python3 scripts/step9_run_few_shot_adaptation.py \
  --ratio 0.1 \
  --ratio 0.2 \
  --ratio 0.5 \
  --ratio 1.0 \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322

python3 scripts/step9_run_calibration_adaptation.py
```

## Current Reading

Current clean Step 7 context:

- the Step 7 LightGBM fusion baselines remain weak on Chinese strict transfer: `core_zero_shot_bge_m3` ROC-AUC `0.601681`, AP `0.448761`; `core_zero_shot_default` ROC-AUC `0.588235`, AP `0.448547`
- raw semantic baselines remain strong and must be reported separately: raw E5 ROC-AUC `0.806723`, raw LaBSE `0.806162`, raw BGE-M3 `0.783754`

Current clean promoted few-shot runs:

- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct / seed 20260320`: ROC-AUC `0.819048`, AP `0.540482`, balanced accuracy `0.589356`, threshold `0.586845`
- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct / seed 20260321`: ROC-AUC `0.824650`, AP `0.541473`, balanced accuracy `0.583473`, threshold `0.566787`
- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct / seed 20260322`: ROC-AUC `0.811765`, AP `0.534180`, balanced accuracy `0.589356`, threshold `0.584435`

Current clean controls:

- `core_few_shot_bge_m3_residual_lr / 100pct`: ROC-AUC `0.817367`, AP `0.515857`; seed thresholds `0.735461 / 0.759046 / 0.735461`
- `core_few_shot_labse_lr_l2 / 100pct`: ROC-AUC `0.799440`, AP `0.531286`, balanced accuracy `0.607843`
- `core_few_shot_bge_m3_lr_l2 / 100pct`: ROC-AUC `0.780952`, AP `0.534253`, balanced accuracy `0.637255`

Operational identifier controls:

- `identifier_augmented_few_shot_default_lr_l2 / 100pct`: ROC-AUC `0.783754`, AP `0.647686`, balanced accuracy `0.720448`, threshold `0.463940`

## Interpretation

The active-boundary few-shot reading is now:

- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct` is the current clean graph-triage few-shot line
- it repairs the collapsed Step 7 LightGBM fusion baseline, but only modestly exceeds the raw E5 semantic ranking baseline
- legacy mixed-source LightGBM few-shot remains a control because it is more threshold-sensitive
- identifier-augmented few-shot is operationally strong but belongs outside the clean transfer-safe few-shot claim

Important caution:

- the clean E5 LR/L2 AUC/AP lift is not enough by itself for a strong significance claim against raw E5
- the Step 11 graph is a candidate-cluster audit surface, not proof-level same-controller evidence
- current Step 11 policy excludes the legacy mixed-LightGBM few-shot experiments from dynamic Step 9 candidate selection

## Relation To Step 11

The current active Step 11 discovery family is:

- `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct` with graph threshold override `0.56`

Current clean controls are:

- `core_few_shot_bge_m3_residual_lr_ratio_100pct` with graph threshold override `0.735`
- `core_few_shot_labse_lr_l2_ratio_100pct` with graph threshold override `0.47`

The current conservative zero-shot anchor is:

- `core_zero_shot_bge_m3`, with no graph override; it resolves to pairwise selected threshold `0.483444`

The active downstream comparison is:

- clean E5 LR/L2 few-shot discovery graph
- BGE residual and LaBSE LR/L2 clean control graphs
- zero-shot BGE anchor/control graph
- identifier-augmented operational control graphs

Current Step 11 outputs are manifest-bound:

- manifest: `reports/step11_current_manifest_20260424.json`
- retained current summaries: `13`
- current cluster-level audit: `reports/step11_cluster_level_audit.current_20260424.csv/json`
