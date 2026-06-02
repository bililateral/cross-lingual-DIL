# Project Progress

Updated: 2026-06-02

## Current Stage

`2026-06-02` evidence-type incremental hard-negative method branch initialized:

- Active branch: `method/evidence-type-incremental-hard-negative`.
- Added the runnable Step 15 design for evidence-type incremental hard-negative learning:
  - `schema/step15_evidence_type_policy.json`
  - `scripts/step15_build_evidence_type_labels.py`
  - `scripts/step15_train_incremental_hard_negative.py`
  - `docs/STEP15_EVIDENCE_TYPE_INCREMENTAL_HARD_NEGATIVE.md`
- Step 15 keeps Step 5 frozen labels unchanged and adds auxiliary `evidence_type` labels only for training diagnostics.
- The final task remains binary `same_controller` vs `different_controller`; evidence types are an auxiliary objective, not a replacement label space.
- The first Step 15 model family is intentionally lightweight: a NumPy MLP with an identity head and an evidence-type head. It does not download or train large language models.
- Clean Step 15 experiments exclude direct identifier features; `step15_e5_multitask_identifier_operational` is explicitly marked as an operational control.
- Local Step 15 smoke/full-run validation completed on Windows: auxiliary labels were generated for both pools, and the first-pass runner completed `45` runs across `3` experiments, `5` curriculum phases, and `3` seeds.
- The local point-estimate run suggests the clean Step 15 phase-4 seed mean is stronger than raw E5 and the previous Step 9 E5 mixup point estimate, but this is not yet a publication claim because the official `5000`-resample Step 12 grouped bootstrap has not been rerun.
- `scripts/step12_statistical_robustness_audit.py` and `schema/step12_statistical_robustness_policy.json` now include Step 15 prediction specs for robustness testing only; Step 11 must not consume Step 15 until Step 12 justifies it.
- Required next action: reproduce Step 15 on Linux, rerun the official fixed-test Step 12 grouped bootstrap, and only then decide whether any Step 15 scorer should enter Step 11. Step 11 must not consume Step 15 from local point estimates alone.

`2026-06-01` discontinued parameter-efficient adaptation branch removed from the active project tree:

- Removed the branch's active policy, training/scoring/evaluation scripts, experiment plans, generated reports, model artifacts, and bytecode caches.
- The active project state now returns to the established Step 5 / Step 7 / Step 9 / Step 11 / Step 12 / Step 13 pipeline.
- Current method baseline remains: raw E5, Step 9 E5 positive-pair mixup controls, relation-reliability filtering, grouped bootstrap robustness audit, and manifest-bound cluster audit.
- No active training command or policy should reference the removed branch. Future concept-drift experiments should be added as a new, separately named method branch after design review.

`2026-05-17` follow-up rerun completed for the RABot-inspired method branch:

- Step 12 policy and runner now include `step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260320/20260321/20260322` and `step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean`.
- Step 12 grouped bootstrap was rerun on the fixed `zh_test = 106` benchmark. Mixup 100pct has the strongest current point estimate: ROC-AUC `0.842017`, AP `0.588995`.
- Paired bootstrap does not support a robust claim that mixup 100pct beats raw E5: ROC-AUC diff `+0.035294`, CI `[-0.086161, 0.158168]`; AP diff `+0.068422`, CI `[-0.207105, 0.331671]`.
- Mixup 100pct does robustly beat collapsed Step 7 default fusion on ROC-AUC: diff `+0.253782`, CI `[0.018175, 0.410672]`; AP still crosses zero.
- Step 11 was explicitly rerun for `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup / 100pct` across seeds `20260320/20260321/20260322`, not only through auto selection.
- Step 11 current manifest was regenerated as `reports/step11_current_manifest_20260517.json`: `19` current summaries, `75` referenced CSVs, `0` unreferenced Step 11 CSVs.
- Step 11 current manifest was re-audited on `2026-05-18` with a stricter proof-edge rule. Identifier-like Step 11 features are no longer sufficient for a same-controller claim; a retained edge must join to Step 5 frozen `positive` / `usable_for_core_transfer = 1` evidence with seller-facing direct contact or PGP support.
- The strict cluster-level audit remains manifest-only and explicit-summary-bound: `reports/step11_cluster_level_audit.current_20260517.json/csv` now records `125` unique cluster sets, `0` high-confidence same-controller full clusters, `0` same-controller cores with expansion, `6` partial anchors, `8` uncertain identifier-like clusters, and `111` template/topic non-controller clusters.
- Step 13 was regenerated on `2026-05-18` and now reads the strict `20260517` Step 11 audit plus both mixup 50pct and 100pct prediction ensembles.

Current scientific interpretation: positive-pair mixup is a useful training-only minority regularization control and improves point estimates, especially at 100pct support, but the fixed-test grouped bootstrap still does not justify a statistically robust claim over raw E5. The strict cluster review further shows that Step 11 should be treated as candidate triage rather than discovery proof: current retained clusters do not support a full same-controller cluster claim without pair-level seller-facing identity evidence.

Local full-project audit on `2026-05-13`: no newer Step 3 / Step 4 / Step 5 / Step 7 / Step 9 / Step 11 artifacts were found after the `2026-04-24` synchronized boundary. The current scientific boundary therefore remains the `2026-04-23` English valid/test top-up refreeze propagated through Linux Step 7, Step 9, and Step 11. The audit confirms that the current project record should continue to treat Step 11 as manifest-bound and evidence-audited, not as a loose `reports/` glob.

Method-branch update on `2026-05-14`, inspired by RABot-style minority augmentation and spurious-edge filtering, was propagated through the `2026-05-17` reruns:

- Step 9 policy and runner now support `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup`, a training-only minority regularization control.
- The mixup branch creates `synthetic_train_only` positive feature rows only from sampled `zh_train` positives with `usable_for_core_transfer = 1` and `core_transfer_eligible = 1`.
- The mixup branch excludes `positive_component_closure_audit`, `audit_only`, `audit_only_soft_alias`, and `uncertain_holdout`; it never writes synthetic rows into Step 5 frozen labels and never touches `zh_valid` or `zh_test`.
- Step 9 summary merging has been hardened so policy-context changes preserve existing experiments and replace only the selected rerun runs, while still backing up the old summary.
- Step 11 policy and runner now support `relation_reliability_score` and an enabled `relation_reliability_filter` before reciprocal-top-k/shared-neighbor graph pruning.
- The relation reliability filter is rule/validation driven, not reinforcement learning. It gives positive support to direct seller-facing contact/PGP, rare clone/rare-ngram support, structural support, and style consistency, and penalizes boilerplate/template and semantic-topic-only edges.
- Step 12 robustness audit now includes `positive_pair_mixup_50pct` and `positive_pair_mixup_100pct` model specs and paired comparisons against raw E5, non-mixup E5 LR/L2, and Step 7 fusion controls.
- These changes add experimental controls and updated audited outputs, but they still do not justify a blanket claim that clean few-shot statistically robustly beats raw E5 on the fixed Chinese test split.

The active supervision boundary is now the `2026-04-23` English valid/test top-up refreeze. It builds on the corrected calibrated-default Step 5 v3 cleanup, the zero-shot BGE Chinese boundary-expansion queue, conservative Chinese positive-anchor passes, English source-domain expansion, English source-domain top-up, English item-level direct-identifier expansion, the validation-split repair, and the `2026-04-23` English direct-identifier plus hard-boundary top-up.

The `2026-04-23` Step 5 boundary has now been propagated through Linux Step 7, Step 9, and Step 11 and synchronized back. Step 11 policy now uses the current clean E5/LabSE/BGE residual scorer set, and the conservative `core_zero_shot_bge_m3` control falls back to its pairwise selected threshold instead of the removed `0.56` override.

Current active Step 5 frozen supervision after the `2026-04-23` English valid/test top-up:

- English `en_content_train_pool`: `1321` reviewed, `734` supervision rows
- English split counts: `train = 401`, `valid = 152`, `test = 181`
- English split labels:
  - `train = 116 positive / 285 negative`
  - `valid = 42 positive / 110 negative`
  - `test = 51 positive / 130 negative`
- English primary positive supervision rows: `209`
- English non-identifier positive share: `0.291866`; this is slightly below the per-English `0.3` target, but the global acceptance check still passes because Chinese non-identifier positives keep the global share at `0.429508`
- Chinese `zh_target_strict` was not expanded in this round and remains `1016` reviewed, `522` supervision rows
- Chinese strict split counts remain `train = 335`, `valid = 81`, `test = 106`
- seller overlap across English supervision splits: `0`
- normalized-alias overlap across English supervision splits: `0`
- Step 5 acceptance checks: pass; coverage warnings/errors: `0 / 0`

Current synchronized Step 7 summary after the `2026-04-23` English valid/test top-up refreeze:

- summary: `reports/step7_training_summary.json`
- semantic summary: `reports/step7_semantic_feature_summary.json`
- selected experiments: `17`
- pair-feature rows: `en_content_train_pool = 6683`, `zh_target_strict = 3857`, `zh_target_aux = 580`
- semantic columns are complete for all five embeddings and both rerankers: `gte_multilingual_base`, `bge_m3`, `multilingual_e5_large`, `labse`, `paraphrase_multilingual_mpnet_base_v2`, `gte_multilingual_reranker_base`, `bge_reranker_v2_m3`
- all selected experiments now have current-boundary prediction files with `152` English validation rows, `181` English test rows, and `106` Chinese strict test rows for zero-shot-safe experiments
- `small_validation_guard.triggered = false` for all 17 experiments; the previous small-validation explanation risk is resolved
- `collapse_guard.triggered = true` for 10 of 17 experiments because several LightGBM fusion models still select very shallow iterations, including `best_iteration = 1` for default, BGE, E5, LaBSE, BGE embedding-only, default no-reranker, default reranker-only, and identifier controls
- E5/LaBSE/Paraphrase experiments now have valid embedding-only views plus separate `+gte_reranker` controls; they are no longer silently GTE-reranker-only runs
- `schema/step7_training_policy.json` now sets all 17 experiments as `default_experiments`, so a plain `python3 scripts/step7_train_baseline_models.py` will not silently rerun only three default models

Current Step 7 Chinese strict zero-shot readings:

- clean default: `core_zero_shot_default` ROC-AUC `0.588235`, AP `0.448547`, balanced accuracy `0.562465`, best iteration `1`
- clean BGE: `core_zero_shot_bge_m3` ROC-AUC `0.601681`, AP `0.448761`, balanced accuracy `0.562465`, best iteration `1`
- best clean ranking ablation: `core_zero_shot_default_no_structural` ROC-AUC `0.623529`, AP `0.287652`, balanced accuracy `0.572269`, best iteration `54`
- strongest non-BGE semantic sensitivity: `core_zero_shot_paraphrase_multilingual_mpnet_plus_gte_reranker` ROC-AUC `0.604482`, AP `0.366364`, balanced accuracy `0.524650`, best iteration `47`
- operational identifier control: `identifier_augmented_default` ROC-AUC `0.606443`, AP `0.418989`, balanced accuracy `0.619888`, best iteration `1`
- non-mainline diagnostic control: `core_zero_shot_default_raw_style_gap_control` ROC-AUC `0.503641`, AP `0.325214`, balanced accuracy `0.513165`, best iteration `57`

Interpretation: the technical Step 7 sync defects are repaired, but the scientific reading remains cautious. BGE-M3 is no longer a uniquely dominant clean zero-shot reference; `core_zero_shot_default_no_structural` has the best clean Step 7 ROC-AUC, while BGE/default have higher AP but collapse to one-tree solutions. Raw semantic features are strong on the Chinese strict test set and must remain reporting baselines: raw E5 ROC-AUC `0.806723`, raw LaBSE `0.806162`, raw BGE-M3 `0.783754`.

Current synchronized Step 9 few-shot summary after the `2026-04-23` Step 7 rerun:

- summary: `reports/step9_few_shot_summary.json`
- selected experiments: `18`
- each experiment has `12` runs: ratios `0.1 / 0.2 / 0.5 / 1.0` across seeds `20260320 / 20260321 / 20260322`
- fixed Chinese evaluation containers remain unchanged: `zh_valid = 81`, `zh_test = 106`
- best clean current candidate: `core_few_shot_multilingual_e5_large_lr_l2 / 50pct`
  - seed `20260320`: ROC-AUC `0.819048`, AP `0.540482`, balanced accuracy `0.589356`
  - seed `20260321`: ROC-AUC `0.824650`, AP `0.541473`, balanced accuracy `0.583473`
  - seed `20260322`: ROC-AUC `0.811765`, AP `0.534180`, balanced accuracy `0.589356`
- residual clean candidate: `core_few_shot_bge_m3_residual_lr / 100pct`, ROC-AUC `0.817367`, AP `0.515857` across all three seeds
- semantic control: `core_few_shot_labse_lr_l2 / 100pct`, ROC-AUC `0.799440`, AP `0.531286` across all three seeds
- operational identifier control: `identifier_augmented_few_shot_default_lr_l2 / 100pct`, ROC-AUC `0.783754`, AP `0.647686`, balanced accuracy `0.720448`
- interpretation: Step 9 LR/residual scorers repair the collapsed Step 7 LightGBM fusion baseline, but the top clean E5 few-shot scorer only modestly exceeds raw E5 semantic ranking. Current Step 11 is therefore a candidate-cluster triage step, not proof that every cluster is same-controller.

Current Step 9 calibration status:

- calibration summaries are synchronized, and Platt scaling now converges numerically
- fixed `0.5` calibration thresholds still predict zero positives on `zh_test`, so calibration remains a diagnostic/control branch and must not be used as the Step 11 discovery mainline

The Step 11 policy for the current boundary now promotes the clean E5/LabSE/BGE residual Step 9 scorer set for graph triage:

- default scorer family: `auto`
- default Step 9 experiment: `core_few_shot_multilingual_e5_large_lr_l2`
- default Step 9 ratio: `0.5`
- default Step 9 seed: `20260321`
- dynamic family priority: `step9`, then `step7`; calibration is disabled for current auto-mainline selection
- clean current Step 11 candidate families:
  - `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct`
  - `core_few_shot_bge_m3_residual_lr_ratio_100pct`
  - `core_few_shot_labse_lr_l2_ratio_100pct`
- conservative anchor/control: `core_zero_shot_bge_m3`
- operational identifier control: `identifier_augmented_few_shot_default_lr_l2_ratio_100pct`
- graph threshold overrides:
  - all three `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct` seeds: `0.56`
  - all three `core_few_shot_bge_m3_residual_lr_ratio_100pct` seeds: `0.735`
  - all three `core_few_shot_labse_lr_l2_ratio_100pct` seeds: `0.47`
  - all three `identifier_augmented_few_shot_default_lr_l2_ratio_100pct` seeds: `0.46`
- `core_zero_shot_bge_m3` has no graph override and resolves to its pairwise selected threshold `0.483444`

Step 11 has been regenerated from the current Step 9 artifacts and synchronized back. Its purpose is to turn edge-level scorers into candidate seller clusters for audit. It must not be interpreted as ground truth; same-controller claims require independent direct-identifier/contact or similarly strong seller-facing evidence.

Current Step 11 manifest and cleanup status after the `2026-04-24` hygiene pass:

- manifest: `reports/step11_current_manifest_20260424.json`
- current retained Step 11 summary files: `13`
- current retained Step 11 files referenced by the manifest: `66`
- stale/unreferenced `reports/step11_*` files deleted: `200`
- remaining `reports/step11_*_clustering_summary.json` files: exactly the `13` current summaries
- remaining cluster-level audit files: `reports/step11_cluster_level_audit.current_20260424.csv` and `reports/step11_cluster_level_audit.current_20260424.json`
- all current manifest keep files exist; no delete candidates remain on disk
- `2026-05-13` audit recheck: root `reports/` still contains exactly `13` Step 11 clustering summaries, and the active audit summary still records `summary_selection_mode = explicit`, `input_summary_count = 13`, `primary_cluster_count_total = 447`, and `unique_cluster_set_count = 140`

Step 11 dynamic Step 9 candidate selection is backend-aware: legacy LightGBM runs still require tree-iteration guards, while `residual_logistic` and `logistic_regression_l2` runs are filtered using logistic solver semantics. Step 11 can score Step 9 residual/logistic scorer artifacts as well as legacy LightGBM model files.

Reports cleanup on `2026-04-22`:

- cleanup manifests:
  - `reports/reports_cleanup_delete_manifest_20260422.csv`
  - `reports/reports_cleanup_keep_manifest_20260422.csv`
  - `reports/reports_cleanup_removed_manifest_20260422.csv`
  - `reports/reports_cleanup_removed_summary_20260422.json`
- removed: `1816` files and `8` snapshot/archive directories
- space released: about `611 MB`
- removed categories: stale Step 11 outputs, Step 9 prediction/sample CSVs, Step 7 embedding caches and preview tables, `.bak/.codexbak` backups, Step 10/Step 11/Step 5 snapshot/archive directories
- retained categories: current Step 5 frozen files and provenance summaries, current Step 7 pair features/models/summaries/predictions, current Step 9 summaries/artifact JSON/model files, and cleanup manifests
- superseded by the `2026-04-24` Step 11 manifest cleanup: current root `reports/step11_*` outputs now contain only the current manifest-retained Step 11 files, plus the current explicit cluster-level audit

To test whether target-domain supervision can improve few-shot adaptation under a larger and cleaner Chinese boundary, a new Step 5 boundary-expansion queue was added and consumed. It selected pending Chinese rows from the current zero-shot BGE Step 11 score surface, not random easy negatives:

- policy: `schema/step5_boundary_expansion_policy.json`
- builder: `scripts/step5_build_boundary_expansion_queue.py`
- apply reviewed labels: `scripts/step5_apply_boundary_expansion_reviews.py`
- target queue: `reports/step5_zh_target_strict_boundary_expansion_queue.zero_shot_bge_20260421.csv`
- Codex review summary: `reports/step5_boundary_expansion_codex_review_summary.zero_shot_bge_20260421.json`
- apply summary: `reports/step5_boundary_expansion_apply_summary.zero_shot_bge_20260421.json`
- reviewed size: `520` rows, split across high-semantic/high-structure positive probes and high-semantic/style-divergent hard-negative probes
- applied labels: `261 negative / 259 uncertain / 0 positive`

The later positive-anchor passes deliberately targeted high-confidence positive evidence instead of more hard negatives:

- policy: `schema/step5_positive_anchor_expansion_policy.json`
- builder: `scripts/step5_build_positive_anchor_expansion_queue.py`
- Codex review: `scripts/step5_codex_review_positive_anchor_expansion.py`
- apply reviewed labels: `scripts/step5_apply_positive_anchor_reviews.py`
- target queue: `reports/step5_zh_target_strict_positive_anchor_expansion_queue.20260421.csv`
- queue summary: `reports/step5_positive_anchor_expansion_queue_summary.20260421.json`
- Codex review summary: `reports/step5_positive_anchor_expansion_codex_review_summary.20260421.json`
- apply summary: `reports/step5_positive_anchor_expansion_apply_summary.20260421.json`
- first pass selected rows: `16`
- first pass applied labels: `13 positive / 3 uncertain`
- first pass net-new Step 4 candidates appended: `10`
- direct-identity v2 pass selected rows: `105`
- direct-identity v2 applied labels: `104 uncertain / 1 positive`
- direct-identity v2 net-new Step 4 candidates appended: `54`
- closure-derived positives are retained as `positive_component_closure_audit` and excluded from supervision/evaluation counts
- interpretation: direct positive anchors are genuinely scarce. The second pass found mostly product/victim-data URLs rather than seller identity anchors, so it did not add primary positive supervision.

The English source-domain expansion pass was added to make the project a defensible large-source/small-target transfer setup rather than a tiny-English/tiny-Chinese comparison:

- policy: `schema/step5_en_source_expansion_policy.json`
- builder: `scripts/step5_build_en_source_expansion_queue.py`
- Codex review: `scripts/step5_codex_review_en_source_expansion.py`
- apply reviewed labels: `scripts/step5_apply_en_source_expansion_reviews.py`
- target queue: `reports/step5_en_source_expansion_queue.20260421.csv`
- queue summary: `reports/step5_en_source_expansion_queue_summary.20260421.json`
- Codex review summary: `reports/step5_en_source_expansion_codex_review_summary.20260421.json`
- apply summary: `reports/step5_en_source_expansion_apply_summary.20260421.json`
- selected rows: `544`
- applied labels: `61 positive / 198 negative / 285 uncertain`
- interpretation: positive labels were kept conservative; product/victim/sample credentials and generic text clones were not promoted. The added negatives are hard-boundary/template controls, not random easy negatives.

Because the first English expansion only barely cleared the requested floor, a smaller source-domain top-up pass was added:

- policy: `schema/step5_en_source_expansion_topup_policy.json`
- target queue: `reports/step5_en_source_expansion_topup_queue.20260421.csv`
- queue summary: `reports/step5_en_source_expansion_topup_queue_summary.20260421.json`
- Codex review summary: `reports/step5_en_source_expansion_topup_codex_review_summary.20260421.json`
- apply summary: `reports/step5_en_source_expansion_topup_apply_summary.20260421.json`
- selected rows: `220`
- applied labels: `2 positive / 56 negative / 162 uncertain`
- interpretation: remaining direct English positive anchors are mostly exhausted under the conservative rubric; further blind expansion would mainly add uncertain rows or hard negatives.

The `2026-04-22` English item-level identity expansion then mined the new Step 3 item-level direct identifiers rather than broad pending queues:

- policy: `schema/step5_en_item_identity_expansion_policy.json`
- generalized builder: `scripts/step5_build_item_identity_expansion_queue.py`
- Codex review: `scripts/step5_codex_review_item_identity_expansion.py`
- apply reviewed labels: `scripts/step5_apply_item_identity_expansion_reviews.py`
- target queue: `reports/step5_en_item_identity_expansion_queue.20260422.csv`
- queue summary: `reports/step5_en_item_identity_expansion_queue_summary.20260422.json`
- Codex review summary: `reports/step5_en_item_identity_expansion_codex_review_summary.20260422.json`
- apply summary: `reports/step5_en_item_identity_expansion_apply_summary.20260422.json`
- selected rows after quality gates: `36`
- applied labels: `35 positive / 1 negative`
- Step 4 candidates appended: `22`
- existing Step 4 candidates updated with item-level identity evidence: `14`
- quality gates: generic parser tokens such as `joinchat`, `messenger`, `before`, `download`, `application`, and soft same-alias continuity rows were excluded before review; the lone negative is a YouTube video ID misparsed as a WeChat-like token
- interpretation: this pass adds genuine source-domain direct-identifier positives without padding random negatives, strengthening the large-source/small-target transfer setup.

The `2026-04-23` English valid/test top-up then raised the source-domain validation split above the Step 7 small-validation guard threshold:

- direct-identifier policy: `schema/step5_en_item_identity_expansion_valid_test_topup_policy.json`
- direct-identifier queue: `reports/step5_en_item_identity_expansion_valid_test_topup_queue.20260423.csv`
- direct-identifier review summary: `reports/step5_en_item_identity_expansion_valid_test_topup_codex_review_summary.20260423.json`
- direct-identifier apply summary: `reports/step5_en_item_identity_expansion_valid_test_topup_apply_summary.20260423.json`
- direct-identifier selected rows: `46`
- direct-identifier applied labels: `30 positive / 16 negative`
- Step 4 candidates appended by direct-identifier top-up: `38`
- source top-up policy: `schema/step5_en_source_expansion_valid_test_topup_policy.json`
- source top-up queue: `reports/step5_en_source_expansion_valid_test_topup_queue.20260423.csv`
- source top-up review summary: `reports/step5_en_source_expansion_valid_test_topup_codex_review_summary.20260423.json`
- source top-up apply summary: `reports/step5_en_source_expansion_valid_test_topup_apply_summary.20260423.json`
- source top-up selected rows: `330`
- source top-up applied labels: `212 negative / 118 uncertain / 0 positive`
- source top-up bucket counts: `10` remaining identifier-primary rows, `100` text-clone probes, `220` hard-negative template probes
- English freeze ratio changed to `55/20/25` for `train/valid/test`
- final English supervision after refreeze: `734` rows with `train = 401`, `valid = 152`, `test = 181`
- interpretation: this round accomplishes the source-domain stability goal without expanding Chinese blindly. Remaining English direct positives are finite; the broad Step 4 tail now mainly contributes hard negatives and uncertain text/template rows.

Local threshold-surface preview from the existing zero-shot BGE scored pairs:

- threshold `0.56`: threshold-pass edges `1764`, post-filter edges `421`, clusters `82`, largest cluster `15`
- threshold `0.53`: threshold-pass edges `1940`, post-filter edges `449`, clusters `87`, largest cluster `15`
- old threshold `0.686852`: threshold-pass edges `240`, post-filter edges `36`, clusters `12`, largest cluster `3`

## Active Step 5 Boundary

Invalidated calibrated-BGE Step 5 v3 cleanup:

- previous cleanup policy: `schema/step5_v3_targeted_cleanup_policy.calibrated_bge_20260420.json` was removed from the active schema tree on `2026-06-01`
- status: deprecated and must not be applied
- archived invalid outputs: `reports/deprecated_step5_v3_calibrated_bge_20260420/`
- invalidation reason: the archived V3 queue summary recorded stale `1663 -> 389` graph filtering while the current `core_calibrated_bge_m3` Step 11 graph is `273 -> 45` at threshold `0.5`
- rollback backup of the invalid active queue state: `reports/step5_zh_target_strict_balanced_review_queue.codexbak.rollback_phantom_calibrated_bge.20260420-201249.csv`

Frozen supervision after the `2026-04-23` English valid/test top-up refreeze:

- English `en_content_train_pool`: `1321` reviewed, `734` supervision rows
- English split counts: `train = 401`, `valid = 152`, `test = 181`
- English split labels:
  - `train = 116 positive / 285 negative`
  - `valid = 42 positive / 110 negative`
  - `test = 51 positive / 130 negative`
- English validation label-stratum coverage now includes positive `identifier_plus_text = 11`, positive `identifier_primary = 19`, positive `text_clone_primary = 12`, negative `identifier_primary = 2`, negative `semantic_only = 33`, and negative `semantic_structural = 75`
- English primary positive supervision rows: `209`
- English non-identifier positive share: `0.291866`
- Chinese `zh_target_strict`: `1016` reviewed, `522` supervision rows
- Chinese strict split counts: `train = 335`, `valid = 81`, `test = 106`
- Chinese strict split labels:
  - `train = 61 positive / 274 negative`
  - `valid = 14 positive / 67 negative`
  - `test = 21 positive / 85 negative`
- split seller overlaps remain zero
- split normalized-alias overlaps remain zero
- coverage requirements pass

Boundary-expansion status:

- the new expansion policy selected `520` pending rows
- the conservative review did not add positive labels because the high-semantic/high-structure probes lacked direct identity closure strong enough for positive supervision
- a follow-up positive-anchor pass added `13` defensible positives and `3` uncertain labels
- the direct-identity v2 follow-up selected `105` additional candidates but added no primary supervision positives: `104` were product/victim-data URL or email ambiguities and `1` was closure-derived audit-only positive
- the active queue has been updated and Step 5 has been refrozen again
- positive split targets remain unmet: `zh_train positive = 61` versus target `100+`; `zh_valid positive = 14` versus target `25+`
- Step 7 has now been rerun and synchronized on this boundary; Step 9 and Step 11 still need refreshed current-boundary reruns before any updated few-shot or graph claim is made

## Step 5 v3 Corrected Queue Rebuild

The corrected V3 queue was built from the then-current calibrated-default Step 11 primary graph before the cleanup was applied:

- policy: `schema/step5_v3_targeted_review_policy.json`
- builder: `scripts/step5_build_targeted_review_queue_v3.py`
- Step 11 summary: `reports/step11_core_calibrated_default_clustering_summary.json`
- scored pairs: `reports/step11_core_calibrated_default_zh_target_strict_scored_pairs.csv`
- cluster CSV: `reports/step11_core_calibrated_default_zh_target_strict_clusters.threshold_0800000.csv`
- review queue: `reports/step5_zh_target_strict_targeted_review_queue.step11_calibrated_default_v3.csv`
- rereview queue: `reports/step5_zh_target_strict_targeted_rereview_queue.step11_calibrated_default_v3.csv`
- summary: `reports/step5_v3_targeted_review_queue_summary.calibrated_default.json`

Current rebuilt V3 queue diagnostics:

- source scorer: `core_calibrated_default`
- graph threshold: `0.8`
- pair-score distribution: min `0.009255`, p95 `0.848353`, max `0.919232`
- threshold-pass edges: `305`
- post-filter edges: `65`
- clusters: `16`
- selected V3 edges: `65`
- net-new review rows: `0`
- rereview rows: `65`
- missing retained pairs: `0`
- consistency checks: all true

After the corrected cleanup and refreshed freeze, a new Step 11 rerun no longer reproduces this calibrated-default queue surface: `core_calibrated_default` now has score max `0.620037`, while the policy graph override remains `0.8`, so the current calibrated-default primary graph has `0` threshold-pass edges. This is not a phantom metric; the scored-pair CSV count and summary threshold view both report `0`.

Corrected calibrated-default cleanup and freeze:

- cleanup policy: `schema/step5_v3_targeted_cleanup_policy.calibrated_default_20260420.json`
- cleanup summary: `reports/step5_v3_targeted_cleanup_summary.calibrated_default_20260420.json`
- reviewer id: `codex_step5_v3_calibrated_default_rereview_20260420`
- changed rows: `65`
- reviewed labels applied: `2 positive / 35 negative / 28 uncertain`
- targeted rereview queue status: `65 / 65` reviewed
- refreshed freeze summary: `reports/step5_frozen_silver_summary.json`

The V3 builder now supports `selection_mode = all_retained_cluster_edges`; it reconstructs retained edges with the Step 11 graph filter and derives selected edges from the current cluster CSV instead of hard-coded cluster ranks or stale explicit pairs.

Downstream alignment note:

- the earlier Step 7 / Step 9 / Step 11 outputs from the invalid calibrated-BGE cleanup boundary remain diagnostic only
- the synchronized Step 7 / Step 9 / Step 11 outputs from the corrected calibrated-default freeze are historical pre-expansion diagnostics and were superseded by the later `2026-04-21` rerun
- the synchronized Step 7 / Step 9 / Step 11 outputs from the `2026-04-21` English top-up boundary are now previous-boundary diagnostics because the `2026-04-22` English item-level identity expansion changed Step 5 supervision
- future Step 5 queue building must not reuse the old `305 -> 65` calibrated-default graph surface after the later boundary-expansion refreeze

## Step 7 Rerun

Step 7 has been rerun on Linux and synchronized for the active `2026-04-23` English valid/test top-up Step 5 boundary:

- semantic summary: `reports/step7_semantic_feature_summary.json`
- training summary: `reports/step7_training_summary.json`
- selected embedding models: `gte_multilingual_base`, `bge_m3`, `multilingual_e5_large`, `labse`, `paraphrase_multilingual_mpnet_base_v2`
- selected rerankers: `gte_multilingual_reranker_base`, `bge_reranker_v2_m3`
- pair-feature rows:
  - `en_content_train_pool = 6683`
  - `zh_target_strict = 3857`
  - `zh_target_aux = 580`
- current Step 7 test containers:
  - English valid: `152` rows (`42 positive / 110 negative`)
  - English test: `181` rows (`51 positive / 130 negative`)
  - Chinese strict test: `106` rows (`21 positive / 85 negative`)

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

Current Chinese strict zero-shot test readings:

- `core_zero_shot_default`: threshold `0.480114`, balanced accuracy `0.562465`, ROC-AUC `0.588235`, AP `0.448547`, best iteration `1`
- `core_zero_shot_bge_m3`: threshold `0.483444`, balanced accuracy `0.562465`, ROC-AUC `0.601681`, AP `0.448761`, best iteration `1`
- `core_zero_shot_bge_m3_embedding_only`: balanced accuracy `0.562465`, ROC-AUC `0.601681`, AP `0.448761`, best iteration `1`
- `core_zero_shot_default_no_style_gap`: balanced accuracy `0.526331`, ROC-AUC `0.517927`, AP `0.375092`, best iteration `2`
- `core_zero_shot_default_no_structural`: balanced accuracy `0.572269`, ROC-AUC `0.623529`, AP `0.287652`, best iteration `54`
- `core_zero_shot_default_raw_style_gap_control`: balanced accuracy `0.513165`, ROC-AUC `0.503641`, AP `0.325214`, best iteration `57`
- `identifier_augmented_default`: threshold `0.503060`, balanced accuracy `0.619888`, ROC-AUC `0.606443`, AP `0.418989`, best iteration `1`

Interpretation:

- the earlier Step 7 sync defects are fixed: semantic columns are present and E5/LaBSE/Paraphrase embedding-only experiments use their own embeddings
- `small_validation_guard` is now off, but 10 of 17 LightGBM experiments trigger `collapse_guard`; several source-domain fusion models select `best_iteration = 1`
- clean zero-shot transfer remains weak on the larger repaired Chinese test split; BGE-M3 is no longer the strongest clean baseline
- `core_zero_shot_default_no_structural` is the best clean ranking ablation by ROC-AUC, while `core_zero_shot_default_raw_style_gap_control` is a non-mainline diagnostic/control and must not be promoted as the clean protocol
- `identifier_augmented_default` remains the operational/direct-identifier control; it has the strongest AP among current Step 7 outputs, but it is not a clean transfer-safe mainline
- Step 9 has now rerun from these refreshed Step 7 artifacts; Step 11 remains the next current-boundary rerun

## Step 9 Few-Shot Rerun

Last synchronized few-shot summary: `reports/step9_few_shot_summary.json`

Status after the active `2026-04-23` English valid/test top-up boundary: current-boundary Step 9 few-shot has been rerun on Linux and synchronized.

Selected grid:

- default experiments: `18`
- backend scope: residual logistic and L2 logistic smooth-fusion candidates; legacy mixed-source LightGBM remains a control family, not the clean mainline
- ratios: `0.1`, `0.2`, `0.5`, `1.0`
- seeds: `20260320`, `20260321`, `20260322`
- fixed Chinese validation rows: `81`
- fixed Chinese test rows: `106`
- all seller-overlap checks are zero

Current clean promoted few-shot line:

- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct / seed 20260320`: ROC-AUC `0.819048`, AP `0.540482`, balanced accuracy `0.589356`, threshold `0.586845`
- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct / seed 20260321`: ROC-AUC `0.824650`, AP `0.541473`, balanced accuracy `0.583473`, threshold `0.566787`
- `core_few_shot_multilingual_e5_large_lr_l2 / 50pct / seed 20260322`: ROC-AUC `0.811765`, AP `0.534180`, balanced accuracy `0.589356`, threshold `0.584435`
- aggregate reading: `50pct` mean ROC-AUC `0.818488`, AP `0.538712`, balanced accuracy `0.587395`

Current clean controls:

- `core_few_shot_bge_m3_residual_lr / 100pct`: ROC-AUC `0.817367`, AP `0.515857`
- `core_few_shot_labse_lr_l2 / 100pct`: ROC-AUC `0.799440`, AP `0.531286`

Operational/direct-identifier controls:

- `identifier_augmented_few_shot_default_lr_l2 / 100pct`: ROC-AUC `0.783754`, AP `0.647686`, balanced accuracy `0.720448`, threshold `0.463940`

Interpretation:

- Step 9 LR/residual few-shot repairs the Step 7 LightGBM fusion collapse and strongly beats the collapsed Step 7 model outputs.
- The top clean few-shot scorer only modestly exceeds the raw E5 semantic baseline, so claims must compare against raw E5/LaBSE/BGE metrics and should not overstate statistical significance.
- Identifier-augmented few-shot is useful operationally but should be reported separately from the clean transfer-safe few-shot result.
- The positive result still needs Step 11 cluster-level audit before any graph-derived same-controller claim is made.

## Step 9 Calibration Rerun

Last synchronized calibration summary: `reports/step9_calibration_summary.json`

Current Chinese strict calibration readings at fixed `0.5` threshold:

- `core_calibrated_default`: balanced accuracy `0.500000`, ROC-AUC `0.588235`, AP `0.448547`, predicts `0` positives on `zh_test`
- `core_calibrated_bge_m3`: balanced accuracy `0.500000`, ROC-AUC `0.601681`, AP `0.448761`, predicts `0` positives on `zh_test`
- `identifier_augmented_calibrated_default`: balanced accuracy `0.500000`, ROC-AUC `0.606443`, AP `0.418989`, predicts `0` positives on `zh_test`

Interpretation:

- Calibration converges numerically, but it calibrates collapsed Step 7 scores into a narrow low-probability range and is not a discovery mainline.
- `step9_calibration` is disabled for current Step 11 auto-mainline selection.
- Policy patch: `schema/step11_clustering_policy.json` no longer assigns `core_zero_shot_bge_m3` a graph threshold override; the conservative anchor/control falls back to the Step 7 pairwise selected threshold `0.483444`.
- Script patch: `scripts/step9_run_calibration_adaptation.py` now supports per-experiment calibration threshold policy through `calibration.experiment_threshold_overrides` or experiment-level threshold keys.

## Step 11 Full Rerun Audit

Targeted Step 11 rerun for the previous `2026-04-21` active boundary was synchronized and audited. It must be regenerated after the `2026-04-22` English item-level identity refreeze.

Previous-boundary target summary coverage:

- expected previous-boundary target summaries: `6`
- actual previous-boundary target summaries: `6`
- missing target summaries: `0`
- checked output paths referenced by those summaries: all present
- Step 11 acceptance checks failed: `0 / 6`
- pair-feature input: `reports/step7_pair_features.zh_target_strict.csv`
- pair-feature rows: `3857`
- eligible rows scored by Step 11: `3851`
- non-core-transfer rows skipped by Step 11: `6`

The six previous-boundary Step 11 summaries were:

- `core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_20260320`
- `core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_20260321`
- `core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_20260322`
- `core_zero_shot_bge_m3`
- `identifier_augmented_few_shot_default_ratio_50pct_seed_20260321`
- `identifier_augmented_few_shot_default_ratio_100pct_seed_20260322`

Important sync hygiene note:

- The stale root Step 11 outputs and unreferenced result CSVs were deleted in the `2026-04-22` reports cleanup.
- Historical comparison now relies on the text record and cleanup manifests unless those old Step 11 outputs are restored from external backup.
- After the next Step 11 rerun, create a new current-summary manifest and do not glob loose files.

Previous-boundary primary graph readings:

- `core_few_shot_bge_m3_lr_l2 / 10pct / seed 20260320`: graph threshold `0.2`, threshold-pass edges `1222`, post-filter edges `348`, clusters `67`, largest cluster `14`, retained sellers `273`
- `core_few_shot_bge_m3_lr_l2 / 10pct / seed 20260321`: graph threshold `0.2`, threshold-pass edges `1260`, post-filter edges `358`, clusters `69`, largest cluster `9`, retained sellers `280`
- `core_few_shot_bge_m3_lr_l2 / 10pct / seed 20260322`: graph threshold `0.2`, threshold-pass edges `1210`, post-filter edges `350`, clusters `66`, largest cluster `14`, retained sellers `272`
- `core_zero_shot_bge_m3`: graph threshold `0.56`, threshold-pass edges `603`, post-filter edges `183`, clusters `39`, largest cluster `7`, retained sellers `145`
- `identifier_augmented_few_shot_default / 50pct / seed 20260321`: graph threshold `0.283981`, threshold-pass edges `1125`, post-filter edges `311`, clusters `51`, largest cluster `13`, retained sellers `226`
- `identifier_augmented_few_shot_default / 100pct / seed 20260322`: graph threshold `0.484093`, threshold-pass edges `735`, post-filter edges `206`, clusters `34`, largest cluster `12`, retained sellers `149`

Seed-stability checks for the clean LR/L2 graph:

- primary threshold-pass edge Jaccard across LR/L2 seeds: `0.9088` to `0.9534`
- primary cluster-member Jaccard across LR/L2 seeds: `0.8840` to `0.9534`
- interpretation: the LR/L2 discovery graph is stable across seeds and is not a single-seed artifact.

High-threshold sensitivity:

- LR/L2 at `0.8` and `0.9` has threshold-pass edges but no post-filter clusters across all three seeds.
- zero-shot BGE remains more conservative at high thresholds: threshold `0.8` yields `9` clusters / `31` sellers; threshold `0.9` yields `5` clusters / `15` sellers.

Interpretation:

- the clean scientific Step 11 discovery family is now `core_few_shot_bge_m3_lr_l2_ratio_10pct`
- zero-shot BGE remains the conservative precision anchor/control
- identifier-augmented few-shot remains an operational/direct-identifier control and must not be merged with the clean cross-lingual few-shot claim
- the LR/L2 graph is broader than zero-shot and stable across seeds, but its `0.2` threshold surface must be cluster-audited before same-controller claims are made
- Script patch: `scripts/step11_cluster_chinese_graph.py` now writes `graph_threshold_diagnostics` and `acceptance_checks_failed`, including whether the graph threshold exceeds the scorer score ceiling or yields no candidate/post-filter edges.
- Step 11 now filters non-core-transfer rows instead of crashing on the six known ineligible rows; future summaries should record this skip count explicitly for audit clarity.

## Step 11 Cluster-Level Audit

Current-boundary cluster-level audit was rerun by Codex on `2026-04-24` against only the thirteen manifest-retained Step 11 summaries. The audit runner now requires repeated `--summary` arguments, so audits cannot fall back to `reports/` globbing.

Audit artifacts:

- audit CSV: `reports/step11_cluster_level_audit.current_20260424.csv`
- audit summary: `reports/step11_cluster_level_audit.current_20260424.json`
- runner: `scripts/step11_cluster_level_audit.py`

Audit method:

- inputs: exactly the thirteen current Step 11 summaries from `reports/step11_current_manifest_20260424.json`, passed explicitly with `--summary`
- summary selection mode: `explicit`
- only primary graph cluster files referenced by each summary's `output_paths` section were read
- primary-view cluster rows across summaries: `447`
- deduplicated exact seller-set clusters: `140`
- retained edges were reconstructed with the Step 11 graph filters, rather than inferred from loose CSV filenames
- same-controller claims are allowed only for direct identifier/contact-anchored cores
- template, description-clone, or semantic/topic cliques are not accepted as whole-cluster same-controller evidence

Audit decisions:

- `same_controller_high_confidence`: `7`
- `same_controller_core_with_possible_expansion`: `1`
- `partial_anchor`: `6`
- `template_clone_not_controller`: `66`
- `semantic_topic_not_controller`: `60`
- `uncertain`: `0`

Previous-boundary strict direct-identity recheck:

- follow-up strict review artifacts were removed from `reports/` during cleanup; historical paths were:
  - `reports/step11_cluster_manual_review.strict_direct_all.current_20260422.csv`
  - `reports/step11_cluster_manual_review_edges.strict_direct_all.current_20260422.csv`
  - `reports/step11_cluster_manual_review.strict_direct_all.current_20260422.json`
- scope: all `160` deduplicated previous-boundary audit clusters at their best primary-graph appearance
- retained edges checked: `947`
- proof-level identity requirement: a Step 5 frozen `positive` edge with `usable_for_core_transfer = 1` and seller-facing direct contact/PGP/wallet evidence
- external URLs, product/victim-data emails, pure semantic positives, and template positives are not counted as proof-level identity cores
- strict direct decisions:
  - `proof_direct_contact_pair_only`: `8`
  - `supporting_positive_but_no_direct_identity_core`: `22`
  - `no_proof_identity_core_after_recheck`: `95`
  - `no_direct_identity_evidence_in_best_cluster`: `35`
- unique proof-level direct Telegram pairs:
  - `/shop/410559 -- /shop/413373` via `telegram:hjm910414`
  - `/shop/444654 -- /shop/459141` via `telegram:fz12120`
  - `/shop/449035 -- /shop/461222` via `telegram:brofish8`
  - `/shop/452097 -- /shop/452596` via `telegram:lucas9999999`
- no current active Step 11 cluster can be claimed as a whole same-controller ring under the strict direct-identity rule.
- the previously suspected `121394 || 435064 || 95895` LR/L2 cluster is downgraded: all three retained edges were already reviewed as `uncertain`, with `external_url` evidence compatible with product/data content rather than seller-operated identity infrastructure.
- the `/shop/3501 || /shop/9484 || 564091` Taiwan-data email cluster is also support/triage only under this strict rule because the shared emails appear inside product/data sample text, not as seller-facing contact.

Interpretation:

- the audit confirms the LR/L2 few-shot graph is useful as a broader discovery surface, but current proof-level identity evidence is only pair-level, not whole-cluster-level
- LR/L2 still supports the ranking/discovery claim, but it does not currently support a new confirmed multi-seller same-controller ring beyond direct Telegram pairs
- the dominant mass of LR/L2 output remains template/topic expansion; these rows should feed review triage, not direct paper claims
- identifier-augmented outputs recover larger contact-heavy components, but strict review shows many are external-URL/product-context clusters and must remain operational controls

## Step 5 Paper-Targeted Expansion Check

Codex implemented a non-destructive paper-targeted Step 5 expansion check on `2026-04-22`:

- policy: `schema/step5_paper_targeted_expansion_policy.json`
- builder: `scripts/step5_build_paper_targeted_expansion_queue.py`
- conservative reviewer: `scripts/step5_codex_review_paper_targeted_expansion.py`
- queue: `reports/step5_zh_target_strict_paper_targeted_expansion_queue.20260422.csv`
- queue summary: `reports/step5_paper_targeted_expansion_queue_summary.20260422.json`
- review summary: `reports/step5_paper_targeted_expansion_codex_review_summary.20260422.json`

The check used exactly the active six Step 11 summaries plus `reports/step11_cluster_manual_review_edges.strict_direct_all.current_20260422.csv`. Step 11 was used only as a candidate miner, not as ground truth.

Selection result:

- selected rows: `20`
- robust LR/L2 high-score unreviewed rows: `8`
- identifier-control high-score unreviewed rows: `8`
- strict direct-proof anchor neighbor rows: `4`
- unreviewed non-URL shared direct-contact pairs available in current Step 4/5: `0`

Conservative review result:

- `20` rows reviewed
- `20 uncertain`
- `0 positive`
- `0 negative`
- no row should be applied to Step 5 supervision at this point

Interpretation:

- the useful part of the paper-targeted expansion proposal is now implemented and reproducible
- the result is negative but important: current raw profiles do not expose additional unreviewed seller-facing shared direct identifiers
- high model scores without direct identity anchors remain review/triage evidence only
- the current evidence still supports LR/L2 as the clean few-shot discovery/ranking mainline, but it does not expand proof-level same-controller claims beyond the four direct Telegram pairs

## Step 3 Item-Level Identity Extraction Check

Codex then moved the positive-anchor search upstream into Step 3 item-level extraction on `2026-04-22`:

- Step 3 parser: `scripts/step3_build_seller_profiles.py`
- Step 3 schema update: `schema/step3_seller_profile_schema.json`
- new item signal outputs:
  - `reports/step3_item_identity_signals.en_content_train_pool.csv`
  - `reports/step3_item_identity_signals.zh_target_strict.csv`
  - `reports/step3_item_identity_signals.zh_target_aux.csv`
- Step 5 item-identity policy: `schema/step5_item_identity_expansion_policy.json`
- Step 5 item-identity builder: `scripts/step5_build_item_identity_expansion_queue.py`
- item-identity queue: `reports/step5_zh_target_strict_item_identity_expansion_queue.20260422.csv`
- queue summary: `reports/step5_item_identity_expansion_queue_summary.20260422.json`
- empty-review summary: `reports/step5_item_identity_expansion_codex_review_summary.20260422.json`

Step 3 rerun result after the 2026-04-23 high-precision Chinese contact patch:

- seller and item count acceptance checks still pass against Step 2
- total item-level identity signals across eligible buckets: `298,775`
- Chinese strict item-level identity signals: `4,430`
- Chinese strict direct-identity-eligible signals: `1,890`
- Chinese strict sellers with any identity signal: `1,259`
- Chinese strict sellers with direct-eligible identity signal: `662`
- Chinese strict shared seller-facing direct token groups: `44`
- new Step 5 review candidates after excluding frozen/reviewed pairs: `0`
- skipped shared-token pairs: `50 frozen_pair`

Interpretation:

- the parser now preserves item-level context for Telegram/TG/纸飞机 compact forms, Wechat/VX/WX/V: compact forms, QQ/企鹅 compact numeric forms, Jabber/XMPP, phone, wallet, PGP, Bat/蝙蝠 numeric forms, email, and support-only URL evidence
- no new Step 5 labels were applied, because all shared seller-facing direct-token pairs found in current Chinese strict raw item text were already frozen/reviewed
- no Step 5 freeze or downstream Step 7/9/11 rerun is triggered by this check
- this strengthens the evidence-scarcity diagnosis: current raw profiles/items have likely exhausted direct positive anchors under the conservative seller-facing standard

## Current Claims

The active evidence currently supports:

- active Step 5 has been corrected from the invalid calibrated-BGE cleanup, expanded with a conservative boundary review plus a positive-anchor pass, and refrozen
- English source-domain supervision has been further expanded and topped up to the active `2026-04-23` boundary: `734` supervision rows, with `209` primary positive supervision rows and no seller/alias split leakage
- Step 5 freeze was repaired on `2026-04-22` after the English validation split was found to be shortcut-leaked by `review_label x review_stratum`; after the later `2026-04-23` English valid/test top-up, active English splits are now `train 401 = 116 positive / 285 negative`, `valid 152 = 42 positive / 110 negative`, and `test 181 = 51 positive / 130 negative`
- the repaired English validation split now includes identifier positives, text-clone positives, semantic negatives, and hard identifier negatives; the old all-identifier-positive/all-semantic-structural-negative validation boundary is invalidated
- active Chinese strict splits after the same component-safe refreeze are now `train 335 = 61 positive / 274 negative`, `valid 81 = 14 positive / 67 negative`, and `test 106 = 21 positive / 85 negative`, with seller/alias split overlap still `0`
- corrected calibrated-default Step 5 v3 rereview queue was rebuilt from current Step 11 files, fully adjudicated, applied, and frozen
- Step 7 has been rerun on Linux and synchronized for the active `2026-04-23` English valid/test top-up boundary
- Step 7 now has complete semantic features for all selected embeddings/rerankers and valid embedding-only E5/LaBSE/Paraphrase controls; several LightGBM fusion models still collapse to shallow one-tree solutions, which is part of the current scientific diagnosis
- current Step 7 clean zero-shot transfer is technically valid but scientifically weak: `core_zero_shot_default_no_structural` is the best clean Step 7 ROC-AUC ablation at `0.623529`, while `core_zero_shot_bge_m3` is `0.601681`
- Step 9 few-shot has been rerun and synchronized on the active boundary; the current clean promoted candidate is `core_few_shot_multilingual_e5_large_lr_l2 / 50pct`
- current Step 9 calibration converges numerically but fixed `0.5` thresholds predict no positives, so calibration is a diagnostic/control branch and not a discovery mainline
- Step 11 policy has been updated and rerun for current E5/BGE residual/LaBSE candidates, conservative BGE, and identifier operational controls; current summaries and referenced outputs are synchronized
- strict direct-identity recheck has reviewed all `160` current audit clusters and found only four unique proof-level direct Telegram pairs; no whole cluster is currently safe to claim as a same-controller ring
- paper-targeted Step 5 expansion from current Step 11 produced a reviewed 20-row candidate queue, but no new high-confidence positive or negative labels
- item-level Step 3 identity extraction now includes the 2026-04-23 high-precision Chinese contact patch and still produced no new unreviewed Chinese strict direct-identity pairs after frozen/reviewed exclusion
- `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct` is the current clean Step 11 discovery candidate family; `core_few_shot_bge_m3_residual_lr_ratio_100pct` and `core_few_shot_labse_lr_l2_ratio_100pct` are current clean controls
- `core_zero_shot_bge_m3` remains the conservative zero-shot anchor/control and now falls back to its pairwise selected graph threshold `0.483444`
- Step 11 dynamic Step 9 candidate filtering is backend-aware, so residual/logistic scorer artifacts are not rejected by LightGBM-only iteration guards
- Step 12 fixed-test robustness audit now exists: `scripts/step12_statistical_robustness_audit.py` with policy `schema/step12_statistical_robustness_policy.json`; outputs are `reports/step12_statistical_robustness_zh_test_20260513.json`, `reports/step12_statistical_robustness_model_metrics_20260513.csv`, and `reports/step12_statistical_robustness_paired_comparisons_20260513.csv`
- Step 12 keeps fixed `zh_test = 106` rows (`21` positive / `85` negative) and uses grouped bootstrap over `39` Step 5 split components; it does not mix `zh_train`, `zh_valid`, and `zh_test`
- RABot-inspired method branches are now encoded for the next rerun: Step 9 positive-pair mixup as training-only minority regularization, and Step 11 relation reliability filtering as a deterministic spurious-edge control
- targeted Step 5 boundary-expansion and positive-anchor routes now exist for testing whether a larger Chinese support/test boundary helps few-shot adaptation
- identifier-augmented few-shot is a strong operational/direct-identifier control, but it is not the clean transfer-safe few-shot result
- clean calibration is not a usable discovery mainline on the current freeze

The active evidence does not yet support:

- making final same-controller claims from current-boundary Step 11 graph outputs before the cluster-level audit is complete
- claiming BGE-M3 is the strongest current clean zero-shot baseline; the latest Step 7 rerun does not support that claim
- using the Step 7 run with `best_iteration = 1` and validation AUC/AP/BAcc `1.0` as a valid benchmark, because it was selected against the shortcut-leaked English validation split
- treating every loose `reports/step11_*clusters*.csv` as current without checking whether it is referenced by a current summary
- globbing all `reports/step11_*_clustering_summary.json` for current audit; use only the explicit current-summary manifest and each summary's `output_paths`
- treating the archived calibrated-BGE V3 queue or cleanup as active
- treating Step 7 / Step 9 / Step 11 metrics from the invalidated cleanup boundary as the current active benchmark after the calibrated-default freeze
- treating the old `core_calibrated_default` `305 -> 65` queue source as reproducible after the corrected cleanup freeze
- using `core_calibrated_bge_m3` as a Step 5 queue source; it is a sensitivity/control scorer only
- selecting the widest Step 11 graph solely by cluster count
- treating any low-threshold or model-expanded Step 11 graph as proof-level same-controller evidence at whole-component level
- treating `external_url` or product/sample email overlap as seller identity proof without seller-facing context
- treating `121394 || 435064 || 95895` as a confirmed LR/L2 same-controller core
- treating template-clone or semantic-only Step 11 clusters as final same-controller rings
- merging identifier-augmented control results into the clean cross-lingual few-shot claim
- claiming the current clean E5 LR/L2 few-shot line statistically robustly beats raw E5 semantic ranking on the fixed `zh_test`; Step 12 finds E5 LR/L2 seed-mean vs raw E5 ROC-AUC diff `+0.012325` with grouped 95% CI `[-0.108240, 0.147650]`, and AP diff `+0.019920` with CI `[-0.251152, 0.326280]`

## Recommended Next Actions

1. Use `reports/step12_statistical_robustness_zh_test_20260513.json/csv` as the current statistical evidence for Step 7/9 pairwise claims; report few-shot gains as modest and uncertainty-bounded against raw semantic baselines.
2. Use `reports/step11_current_manifest_20260424.json` as the authoritative current Step 11 allow-list; do not glob `reports/step11_*` files.
3. Use `reports/step11_cluster_level_audit.current_20260424.csv/json` as the current cluster-level audit; it was built from summary `output_paths` only and classifies cluster evidence by direct identifier/contact, partial anchor, template clone, semantic topic, or uncertain.
4. Use the 2026-04-23 Step 3 high-precision parser outputs as audit evidence, but do not refreeze Step 5 unless a future review queue contains new reviewed labels.
5. Do not apply the 2026-04-22 paper-targeted Chinese queue to Step 5 supervision; it found only uncertain rows.
6. Sync the 2026-05-14 Step 9/Step 11 code and policy updates to Linux, rerun the new `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` branch, then rerun Step 11 so `relation_reliability_filter` is reflected in scored-pair and cluster summaries.
7. If publication requires stronger Chinese identity-proof claims, the next evidence source must be new raw/OCR/source fields or external corroborating evidence; the current Chinese item-level text extraction has exhausted unreviewed direct-token pairs.
