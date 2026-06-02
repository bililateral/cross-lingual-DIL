# Step 11: Chinese Graph Clustering

Status: current-boundary Step 11 rerun synchronized and manifest-cleaned on `2026-04-24`; locally rechecked on `2026-05-13` with exactly `13` current summaries still present; authoritative project-level status is in `docs/PROJECT_PROGRESS.md`; run new Step 11 scoring on the Linux runtime server only

## Role

Step 11 projects a selected pair scorer into the Chinese candidate graph.

It does not:

- retrain Step 7
- reopen Step 5
- modify Step 10 snapshots

It does:

- reuse a current synchronized Step 7 / Step 9 / Step 9 calibration scorer
- score all `zh_target_strict` candidate pairs
- threshold the pair graph
- prune unsupported edges before connected-components extraction
- emit first-pass suspicious seller clusters

## Freshness Rules

Step 11 now enforces two hard constraints:

1. it may only resolve scorers from the current main synchronized summaries:
   - `reports/step7_training_summary.json`
   - `reports/step9_few_shot_summary.json`
   - `reports/step9_calibration_summary.json`
2. dynamic candidate selection is rebuilt fresh at runtime from those summaries

Archive snapshots and stale `step10_*` summary paths are no longer valid Step 11 scorer sources.

As of the `2026-04-16` audit patch, Step 11 also enforces two additional runtime rules:

1. Step 9 few-shot family-best selection is no longer allowed to overfit the downstream graph to a single perfect small-test seed; ratio-level stability is preferred before seed-level peak metrics.
2. Sensitivity views are no longer assumed to be universally `0.8 / 0.9`; if those absolute thresholds exceed the selected scorer's observed score ceiling, Step 11 backfills high-quantile thresholds from the current score distribution instead of emitting guaranteed-empty views.

As of the `2026-04-20` corrected-freeze patch:

1. `core_few_shot_bge_m3` `10pct` was initially treated as the primary discovery family for the next audit pass.
2. `core_calibrated_bge_m3` is a calibration sensitivity/control scorer only; it has a Step 11 graph threshold override of `0.4` and must not be used as a Step 5 queue source.
3. Step 11 summaries now emit `graph_threshold_diagnostics` and `acceptance_checks_failed` so empty graph thresholds are visible.
4. Step 5 v3 queue building refuses to consume an empty Step 11 source graph or a graph whose primary threshold exceeds its score ceiling.

As of the `2026-04-21` methodological re-audit patch:

1. `core_zero_shot_bge_m3` was promoted as the temporary discovery scorer for the then-current corrected-freeze boundary; this has been superseded by the `2026-04-22` active-boundary LR/L2 few-shot rerun.
2. `core_zero_shot_bge_m3` used a Step 11 graph threshold override of `0.56` in that historical boundary; this is superseded. The current boundary has no BGE graph override and resolves to pairwise selected threshold `0.483444`.
3. Legacy `core_few_shot_bge_m3` `10pct` is demoted to adaptation sensitivity/stress-test evidence because its earlier balanced-accuracy gain was threshold-driven while ROC-AUC degraded relative to the then-current zero-shot BGE line.
4. Step 11 now supports new Step 9 `residual_logistic` and `logistic_regression_l2` scorer artifacts in addition to legacy LightGBM model files.
5. The zero-shot BGE Linux Step 11 rerun has been synchronized; current `core_zero_shot_bge_m3` outputs use primary graph threshold `0.483444`.
6. Step 11 dynamic Step 9 candidate selection is backend-aware: legacy LightGBM candidates still use tree-iteration guards, while residual/logistic candidates use logistic solver semantics.

As of the previous `2026-04-22` Step 11 rerun before the later Step 5 label-stratified split repair:

1. The clean scientific discovery family is `core_few_shot_bge_m3_lr_l2_ratio_10pct`.
2. The Step 11 policy default scorer family is `auto`, with Step 9 priority before Step 7 and calibration.
3. The three active LR/L2 seeds use a graph threshold override of `0.2`.
4. `core_zero_shot_bge_m3` was the conservative zero-shot anchor/control at graph threshold `0.56` in that previous boundary.
5. `identifier_augmented_few_shot_default` remains an operational/direct-identifier control, not the clean transfer-safe mainline.
6. The previous-boundary Step 11 target set had six synchronized summaries, but stale root Step 11 outputs were removed during the `2026-04-22` reports cleanup. Any new current-boundary audit must use a refreshed manifest generated after rerun.
7. Step 11 filters the six known `core_transfer_eligible != 1` rows and scores `3851` eligible Chinese candidate pairs.

As of the current `2026-04-24` manifest cleanup:

1. Current Step 11 output selection is manifest-bound, not glob-bound.
2. Authoritative manifest: `reports/step11_current_manifest_20260424.json`.
3. Current retained Step 11 summaries: `13`.
4. Current retained Step 11 files referenced by the manifest: `66`.
5. Stale/unreferenced `reports/step11_*` files deleted: `200`.
6. Current clean discovery family: `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct` at graph threshold `0.56`.
7. Current clean controls: `core_few_shot_bge_m3_residual_lr_ratio_100pct` at `0.735` and `core_few_shot_labse_lr_l2_ratio_100pct` at `0.47`.
8. Current zero-shot BGE anchor/control: `core_zero_shot_bge_m3` at pairwise selected threshold `0.483444`.
9. Current operational identifier control: `identifier_augmented_few_shot_default_lr_l2_ratio_100pct` at graph threshold `0.46`.
10. Current cluster-level audit artifacts:
   - `reports/step11_cluster_level_audit.current_20260424.csv`
   - `reports/step11_cluster_level_audit.current_20260424.json`
11. The old `current_20260422` cluster audit files and previous-boundary Step 11 outputs were removed from `reports/`; historical readings below are text records only.
12. `2026-05-13` recheck: root `reports/` still contains exactly `13` current `step11_*_clustering_summary.json` files; current audit summary still records `summary_selection_mode = explicit`, `input_summary_count = 13`, `primary_cluster_count_total = 447`, and `unique_cluster_set_count = 140`.

## Graph Hardening Rules

The active clean policy now applies:

- no `direct_keep` escape hatch
- reciprocal top-`5`
- shared-neighbor minimum `1`
- iterative shared-neighbor pruning
- triangle participation disabled by default

This means the graph primary view is:

- threshold pass
- then graph support pruning
- then connected components

## Historical v2 Comparison Trio

The `2026-04-16` v2 dynamic family-best comparison set was:

- zero-shot baseline:
  - `core_zero_shot_bge_m3`
- few-shot mainline:
  - `core_few_shot_bge_m3 / 10pct / seed 20260320`
- calibration control:
  - `core_calibrated_default`

These are historical v2 outputs. They are no longer the active mainline after the corrected calibrated-default Step 5 v3 cleanup and refreshed freeze.

They were also the exact Step 11 artifacts frozen inside the historical snapshot:

- `reports/step5_v2_milestone_snapshot_20260416`

That snapshot directory was removed during the `2026-04-22` reports cleanup after it stopped being an active runtime dependency.

## Fixed Inputs

Step 11 consumes:

- `reports/step7_pair_features.zh_target_strict.csv`
- `reports/step3_seller_profiles.zh_target_strict.jsonl`
- `reports/step7_training_summary.json`
- `reports/step9_few_shot_summary.json`
- `reports/step9_calibration_summary.json`
- `schema/step11_clustering_policy.json`

## Current-Boundary Outputs

The current manifest-retained summary set is:

- `reports/step11_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_20260320_clustering_summary.json`
- `reports/step11_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_20260321_clustering_summary.json`
- `reports/step11_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_20260322_clustering_summary.json`
- `reports/step11_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_20260320_clustering_summary.json`
- `reports/step11_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_20260321_clustering_summary.json`
- `reports/step11_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_20260322_clustering_summary.json`
- `reports/step11_core_few_shot_labse_lr_l2_ratio_100pct_seed_20260320_clustering_summary.json`
- `reports/step11_core_few_shot_labse_lr_l2_ratio_100pct_seed_20260321_clustering_summary.json`
- `reports/step11_core_few_shot_labse_lr_l2_ratio_100pct_seed_20260322_clustering_summary.json`
- `reports/step11_core_zero_shot_bge_m3_clustering_summary.json`
- `reports/step11_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_20260320_clustering_summary.json`
- `reports/step11_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_20260321_clustering_summary.json`
- `reports/step11_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_20260322_clustering_summary.json`

Current audit boundary:

- do not glob all Step 11 summaries in `reports/`
- read `reports/step11_current_manifest_20260424.json`
- pass each retained summary explicitly with repeated `--summary`; the audit runner now rejects no-summary glob mode
- use only each retained summary's `output_paths`
- current audit input summaries: `13`
- current audit primary-view cluster rows: `447`
- deduplicated exact seller-set clusters: `140`
- decisions:
  - `same_controller_high_confidence`: `7`
  - `same_controller_core_with_possible_expansion`: `1`
  - `partial_anchor`: `6`
  - `template_clone_not_controller`: `66`
  - `semantic_topic_not_controller`: `60`
  - `uncertain`: `0`

## Previous-Boundary Outputs

The previous-boundary target summaries were:

- `reports/step11_core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_20260320_clustering_summary.json`
- `reports/step11_core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_20260321_clustering_summary.json`
- `reports/step11_core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_20260322_clustering_summary.json`
- `reports/step11_core_zero_shot_bge_m3_clustering_summary.json`
- `reports/step11_identifier_augmented_few_shot_default_ratio_50pct_seed_20260321_clustering_summary.json`
- `reports/step11_identifier_augmented_few_shot_default_ratio_100pct_seed_20260322_clustering_summary.json`

These root files were deleted during reports cleanup. They remain documented here as previous-boundary readings only.

Primary graph readings:

- `core_few_shot_bge_m3_lr_l2 / 10pct / seed 20260320`: threshold `0.2`, threshold-pass edges `1222`, retained edges `348`, clusters `67`, largest cluster `14`, retained sellers `273`
- `core_few_shot_bge_m3_lr_l2 / 10pct / seed 20260321`: threshold `0.2`, threshold-pass edges `1260`, retained edges `358`, clusters `69`, largest cluster `9`, retained sellers `280`
- `core_few_shot_bge_m3_lr_l2 / 10pct / seed 20260322`: threshold `0.2`, threshold-pass edges `1210`, retained edges `350`, clusters `66`, largest cluster `14`, retained sellers `272`
- `core_zero_shot_bge_m3`: threshold `0.56`, threshold-pass edges `603`, retained edges `183`, clusters `39`, largest cluster `7`, retained sellers `145`
- `identifier_augmented_few_shot_default / 50pct / seed 20260321`: threshold `0.283981`, threshold-pass edges `1125`, retained edges `311`, clusters `51`, largest cluster `13`, retained sellers `226`
- `identifier_augmented_few_shot_default / 100pct / seed 20260322`: threshold `0.484093`, threshold-pass edges `735`, retained edges `206`, clusters `34`, largest cluster `12`, retained sellers `149`

LR/L2 seed stability:

- primary threshold-pass edge Jaccard across LR/L2 seeds: `0.9088` to `0.9534`
- primary cluster-member Jaccard across LR/L2 seeds: `0.8840` to `0.9534`

Previous-boundary audit boundary:

- do not glob all Step 11 summaries in `reports/` because stale summaries from earlier policies/universes remain present
- previous-boundary audit used only the six active summaries above via repeated `--summary` inputs
- same-controller claims still require direct identifier/contact cores; low-threshold LR/L2 components remain discovery surfaces unless the audited row is identifier/contact anchored

Previous-boundary cluster-level audit:

- runner: `scripts/step11_cluster_level_audit.py`
- audit CSV: removed from `reports/`; historical path was `reports/step11_cluster_level_audit.current_20260422.csv`
- audit summary: removed from `reports/`; historical path was `reports/step11_cluster_level_audit.current_20260422.json`
- summary selection mode: `explicit`
- input summaries: `6`
- primary-view cluster rows: `326`
- deduplicated exact seller-set clusters: `160`
- decisions:
  - `same_controller_high_confidence`: `5`
  - `same_controller_core_with_possible_expansion`: `4`
  - `partial_anchor`: `8`
  - `template_clone_not_controller`: `81`
  - `semantic_topic_not_controller`: `62`
  - `uncertain`: `0`

LR/L2 audit reading:

- LR/L2 contributes `83` deduplicated seller-set clusters across the three active seeds.
- `54` LR/L2 seller-set clusters are stable across all three seeds.
- The mechanical identifier-count audit originally surfaced high/medium candidates, but a strict direct-identity recheck downgraded all whole-cluster claims.
- Strict direct recheck artifacts:
  - `reports/step11_cluster_manual_review.strict_direct_all.current_20260422.csv`
  - `reports/step11_cluster_manual_review_edges.strict_direct_all.current_20260422.csv`
  - `reports/step11_cluster_manual_review.strict_direct_all.current_20260422.json`
- The strict recheck reviewed all `160` deduplicated clusters and `947` retained edges.
- It found `4` unique proof-level direct Telegram pairs, appearing in `8` cluster rows, but no whole cluster that can be claimed as one same-controller ring.
- The previous `121394 || 435064 || 95895` LR/L2 candidate is downgraded because its three retained edges were already reviewed as `uncertain` external-URL/product-context evidence.
- This supports LR/L2 as a clean discovery/ranking family, but graph-derived identity claims must remain pair-level until direct-contact cores are manually expanded.

## Historical Dynamic Outputs

### Zero-shot baseline

- file:
  - `reports/step11_core_zero_shot_bge_m3_clustering_summary.json`
- graph threshold:
  - `0.686852`
- threshold-pass edges:
  - `240`
- retained edges:
  - `36`
- clusters:
  - `12`
- largest cluster:
  - `3`
- `tree_cluster_share = 0.0`
- `leaf_member_share_in_clusters = 0.0`

Interpretation:

- this is the clean zero-shot graph baseline
- it is very conservative
- it is useful as a high-precision control, not as the main discovery graph

### Few-shot mainline

- file:
  - `reports/step11_core_few_shot_bge_m3_ratio_10pct_seed_20260320_clustering_summary.json`
- graph threshold:
  - `0.457259`
- threshold-pass edges:
  - `1998`
- retained edges:
  - `418`
- clusters:
  - `85`
- largest cluster:
  - `14`
- `tree_cluster_share = 0.0`
- `leaf_member_share_in_clusters = 0.0`
- `removed_by_shared_neighbor = 1365`
- `removed_by_reciprocal_top_k = 215`

Interpretation:

- this is a historical few-shot graph, not the current discovery mainline
- pruning is active and topology is healthy, but the scorer is now treated as a threshold-driven adaptation control
- dense semantic or template-copy cliques still require human review

### Calibration control

- file:
  - `reports/step11_core_calibrated_default_clustering_summary.json`
- graph threshold:
  - `0.800000`
- threshold-pass edges:
  - `464`
- retained edges:
  - `119`
- clusters:
  - `28`
- largest cluster:
  - `10`
- `tree_cluster_share = 0.0`
- `leaf_member_share_in_clusters = 0.0`

Interpretation:

- this is the strongest clean calibrated control graph
- it is much richer than zero-shot baseline
- it is still materially smaller than the few-shot BGE mainline

## Current Fine-Grained Review Of The Few-Shot BGE Mainline

Retained graph facts:

- retained edges: `418`
- connected components: `85`
- retained sellers: `337`
- global bridge-edge count: `0`

This means the current few-shot BGE mainline no longer suffers from the old bridge-driven topology failure.

### Top cluster 1

- `14` nodes
- `22` retained edges
- density `0.241758`
- all retained edges are `text_clone_primary`

Interpretation:

- this is a dense four-piece banking-card template-copy clique
- it is topologically valid
- it is not, by itself, proof that all `14` accounts share one controller

### Top cluster 2

- `10` nodes
- `15` retained edges
- density `0.333333`
- all retained edges are `text_clone_primary`

Interpretation:

- this is also a template-copy clique
- it is semantically mixed and should be treated as a market-template clique rather than a confirmed same-controller ring

### Top cluster 3

- `9` nodes
- `13` retained edges
- density `0.361111`
- retained edge mix:
  - `7 text_clone_primary`
  - `5 semantic_structural`
  - `1 identifier_plus_text`

Most important anchor:

- `/shop/444654 <-> /shop/459141`
  - `identifier_plus_text`
  - shared Telegram `fz12120`

Interpretation:

- the cluster contains a real anchor pair
- but the whole cluster still mixes anchor edges with template-copy and semantic expansion edges
- it should not be treated as uniformly true

### Top clusters 4 and 5

- cluster 4:
  - pure `semantic_structural`
  - same-market `网贷数据` buyer clique
- cluster 5:
  - mixed `semantic_structural + semantic_only`
  - gift-card / payment-instrument topic clique

Interpretation:

- these are no longer topology bugs
- they are semantic/topic cliques and need business interpretation

## Historical Downstream Use

The historical few-shot BGE Step 11 mainline fed the earlier Step 5 v3 targeted rereview. The current corrected-freeze rerun should be audited from the latest Step 11 summaries listed in `docs/PROJECT_PROGRESS.md`.

Operational rule:

- do **not** treat the top cliques as direct final claims
- do **not** reopen the active freeze inline from Step 11
- use these cliques only to populate Step 5 v3 review / rereview queues
- cut a new freeze only after the v3 human review finishes

## Outputs

Each run emits:

- one scored-pair CSV
- one clustering summary JSON
- one cluster CSV at graph primary threshold
- additional sensitivity cluster CSVs resolved from the selected scorer's actual score distribution

Audit note:

- the current runner emits explicit graph-threshold diagnostics
- compressed calibration scorers must be treated through explicit graph overrides or sensitivity/control labels, not as automatic Step 5 queue sources

## Linux Commands

Current clean mainline:

```bash
python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family auto
```

Explicit current target rerun set:

```bash
for seed in 20260320 20260321 20260322; do
  python3 scripts/step11_cluster_chinese_graph.py \
    --policy schema/step11_clustering_policy.json \
    --scorer-family step9 \
    --step9-experiment core_few_shot_bge_m3_lr_l2 \
    --step9-ratio 0.1 \
    --step9-seed "$seed"
done

python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family step7 \
  --step7-experiment core_zero_shot_bge_m3

python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family step9 \
  --step9-experiment identifier_augmented_few_shot_default \
  --step9-ratio 0.5 \
  --step9-seed 20260321

python3 scripts/step11_cluster_chinese_graph.py \
  --policy schema/step11_clustering_policy.json \
  --scorer-family step9 \
  --step9-experiment identifier_augmented_few_shot_default \
  --step9-ratio 1.0 \
  --step9-seed 20260322
```

Calibration controls remain available, but current reporting should not use calibration as the discovery mainline.

## Reporting Boundary

Current Step 11 supports these claims:

- freshness is fixed
- archive fallback is blocked
- graph pruning now removes unsupported bridge trees
- the clean LR/L2 few-shot line is the current scientific discovery graph family
- the zero-shot BGE line is the current conservative anchor/control graph
- backend-aware Step 9 candidate filtering prevents residual/logistic scorer artifacts from being rejected by LightGBM-only iteration guards
- LR/L2 graph surfaces are stable across the three `10pct` seeds

Current Step 11 does not support these claims:

- that dense template-copy cliques are automatically true same-controller clusters
- that semantic/topic cliques without anchors should be treated as proof-level identity clusters
- that the archived default/GTE Step 11 outputs remain active mainline evidence
- that all Step 11 summaries in `reports/` are current active-boundary summaries
- that LR/L2 low-threshold graph expansions are proof-level same-controller clusters at whole-component level; only audited direct identifier/contact cores can support same-controller claims
