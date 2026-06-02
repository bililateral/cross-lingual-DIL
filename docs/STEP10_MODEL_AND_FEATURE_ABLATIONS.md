# Step 10: Model Robustness and Feature Ablations

Status: archived robustness report assembled from synchronized outputs dated `2026-03-26`

Current mainline note:

- this page preserves the Step 10 robustness snapshot built on the earlier `10`-pair Chinese strict test container
- it predates the repaired GTE semantic-feature-table rerun
- it is not the active `2026-04-10` mainline progress boundary
- the refreshed Step 7 / Step 9 record on the active `43`-row Chinese strict test container now lives in `docs/PROJECT_PROGRESS.md`

## Role

Step 10 does not redefine the benchmark and does not replace the fixed Step 8 / Step 9 protocol.

Its job is to answer:

- which multilingual backbone views are genuinely useful under the fixed Step 7 transfer-safe feature protocol
- whether the then-active mainline result depended on identifier augmentation or English-only auxiliary features
- which feature groups still need ablation before the project moves on to clustering and final reporting

## Fixed Protocol

The following boundaries remain fixed for the current Step 10 subset:

- source supervision pool: `en_content_train_pool`
- strict target evaluation pool: `zh_target_strict`
- English threshold selection split: `valid`
- Chinese zero-shot reporting split: `test`
- no Chinese test retuning
- no Step 9 result rewriting

This means the current Step 10 results are robustness comparisons under the same split and threshold discipline as Step 7 / Step 8, not new benchmark definitions.

## Inputs

The finalized Step 10 record is assembled from three frozen snapshots:

- `reports/step10_backbone_snapshot_20260326`
- `reports/step10_feature_ablation_snapshot_20260326`
- `reports/step10_raw_vs_relative_snapshot_20260326`

The top-level `reports/step7_training_summary.json` is now the latest targeted rerun summary for the raw-vs-relative control and should not be treated as the complete Step 10 archive by itself.

## Completed Comparison Set

The archived Step 10 subset contains:

- backbone comparisons:
  - `core_zero_shot_default`
  - `core_zero_shot_bge_m3`
  - `core_zero_shot_multilingual_e5_large`
  - `core_zero_shot_labse`
  - `core_zero_shot_paraphrase_multilingual_mpnet`
- control views:
  - `identifier_augmented_default`
  - `en_only_ablation_default`

## Archived Snapshot Results

| Experiment | EN test balanced_accuracy | EN test roc_auc | ZH zero-shot balanced_accuracy | ZH zero-shot roc_auc | ZH zero-shot average_precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| `core_zero_shot_default` | `0.777778` | `0.925926` | `0.785714` | `0.809524` | `0.722222` |
| `core_zero_shot_bge_m3` | `0.722222` | `0.740741` | `0.833333` | `1.000000` | `1.000000` |
| `core_zero_shot_multilingual_e5_large` | `0.666667` | `0.759259` | `0.619048` | `0.642857` | `0.411111` |
| `core_zero_shot_labse` | `0.666667` | `0.648148` | `0.619048` | `0.785714` | `0.666667` |
| `core_zero_shot_paraphrase_multilingual_mpnet` | `0.777778` | `0.851852` | `0.619048` | `0.714286` | `0.633333` |
| `identifier_augmented_default` | `0.666667` | `0.907407` | `0.714286` | `0.785714` | `0.698413` |
| `en_only_ablation_default` | `0.666667` | `0.851852` | `N/A` | `N/A` | `N/A` |

## Interpretation

The archived backbone / control-view subset supports the following claims.

### 1. Historical note: `core_zero_shot_default` was the archived baseline in this snapshot

- it is still the best source-domain balance point
- within this archived ten-pair Chinese strict zero-shot test container, it preserves full recall
- it remains the correct anchor for Step 8 and Step 9 reporting

### 2. Historical note: `core_zero_shot_bge_m3` was the strongest robustness comparison in this snapshot

- within this archived snapshot, it removes the Chinese `semantic_only` false-positive cluster
- within this archived snapshot, it achieves perfect target-side ranking on the Chinese test container
- but it loses one Chinese positive at the fixed English-valid threshold
- it also degrades English source-domain performance

Therefore this archived snapshot should not be used by itself to decide the current default-vs-BGE ranking after the repaired GTE rerun.

### 3. Other backbone swaps are not competitive in the current protocol

- `multilingual_e5_large`, `LaBSE`, and `paraphrase-multilingual-mpnet-base-v2` do not outperform the default mainline
- they should remain secondary backbone controls only

### 4. Identifier augmentation still does not provide a stronger zero-shot result

- the explicit identifier features remain zero-gain in the fitted `identifier_augmented_default` ensemble
- the augmented view therefore does not strengthen the current zero-shot claim

### 5. The archived result set is still small-sample

- the Chinese strict zero-shot test split still contains only `10` reviewed pairs
- these Step 10 comparisons should be described as robustness evidence rather than final confirmation-level proof

## Encoded Feature-Ablation Results

| Experiment | EN test balanced_accuracy | EN test roc_auc | ZH zero-shot balanced_accuracy | ZH zero-shot roc_auc | ZH zero-shot average_precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| `core_zero_shot_default_no_reranker` | `0.388889` | `0.518519` | `0.714286` | `0.714286` | `0.609524` |
| `core_zero_shot_default_reranker_only` | `0.722222` | `0.870370` | `0.619048` | `0.785714` | `0.666667` |
| `core_zero_shot_default_no_semantics` | `0.500000` | `0.740741` | `0.833333` | `1.000000` | `1.000000` |
| `core_zero_shot_default_no_style_gap` | `0.777778` | `0.870370` | `0.571429` | `0.714286` | `0.777778` |
| `core_zero_shot_default_no_structural` | `0.777778` | `0.925926` | `0.619048` | `0.738095` | `0.642857` |
| `core_zero_shot_default_raw_style_gap_control` | `0.666667` | `0.814815` | `0.714286` | `0.666667` | `0.609524` |

These ablations support the following interpretations:

- reranker removal causes the largest source-domain collapse, which confirms that reranker signal is a major contributor to the current mainline model
- style-gap removal hurts Chinese zero-shot performance the most among the currently encoded ablations, which supports the current market-relative style-gap design
- structural-feature removal hurts Chinese zero-shot performance while leaving English test metrics almost unchanged, which suggests structural overlap contributes more target-side than source-side value in the current setup
- replacing market-relative style-gap features with raw absolute gaps also degrades both English and Chinese evaluation, which supports keeping the relative normalization as the mainline design
- the strong Chinese result under `core_zero_shot_default_no_semantics` should not be over-interpreted:
  - this archived Chinese test container still has only `10` reviewed pairs
  - the same experiment degrades strongly on English source-domain evaluation
  - this result is best treated as a small-sample ranking / threshold artifact rather than evidence that semantics are unnecessary

## Raw-vs-Relative Control

The final Step 10 control run uses:

- `core_zero_shot_default`
  - transfer-safe market-relative style-gap features
- `core_zero_shot_default_raw_style_gap_control`
  - the same structural features and default semantic scores
  - raw absolute style/scale gaps in place of market-relative gaps
  - `zero_shot_safe = false`, reported only as a control

Observed result:

- the raw-gap control underperforms the relative baseline on both English and Chinese evaluation
- English test:
  - balanced_accuracy drops from `0.777778` to `0.666667`
  - roc_auc drops from `0.925926` to `0.814815`
- Chinese strict zero-shot test:
  - balanced_accuracy drops from `0.785714` to `0.714286`
  - roc_auc drops from `0.809524` to `0.666667`
  - average_precision drops from `0.722222` to `0.609524`

Interpretation:

- the market-relative normalization is not just a policy preference; it also outperforms the raw absolute-gap control under the fixed Step 7 protocol
- this strengthens the existing Step 10 conclusion that the current transfer-safe style-gap design should remain in the mainline baseline

## Current Mainline Note

Step 10 is complete as an archived robustness snapshot set.

The current `2026-04-10` mainline action is no longer an automatic move into Step 11 based on this archived page alone.

1. keep the synchronized Step 10 result snapshots fixed as archived robustness controls:
   - `reports/step10_backbone_snapshot_20260326`
   - `reports/step10_feature_ablation_snapshot_20260326`
   - `reports/step10_raw_vs_relative_snapshot_20260326`
2. keep the refreshed repaired Step 7 / Step 9 records fixed as the active mainline baseline and adaptation record
3. use this Step 10 page only as a historical robustness reference
4. decide Step 11 scorer comparison from the repaired `2026-04-10` summaries, not from this archived snapshot alone
