# Step 15 Evidence-Type Incremental Hard-Negative Experiment

Updated: 2026-06-02

Branch: `method/step15-v2-curriculum-audit`

## Purpose

Step 15 implements the first runnable version of the evidence-type incremental hard-negative plan.

The goal is not to replace the project task with ordinary multiclass classification. The final task remains:

```text
same_controller vs different_controller
```

The new idea is to keep the binary identity head, but add conservative evidence/noise-type supervision during training. This targets the current Chinese-domain failure mode:

```text
semantic/topic/template similarity is often high even when identity reliability is low.
```

The experiment asks whether teaching the model these hard-negative regimes can reduce template/topic false positives while preserving or improving fixed `zh_test` ranking.

## Non-Negotiable Controls

Step 15 does not modify Step 5 frozen labels.

Step 15 does not mix `zh_train`, `zh_valid`, and `zh_test`.

Step 15 does not train the binary identity head on `uncertain` rows.

Step 15 does not use Step 11 cluster decisions as same-controller ground truth.

Step 15 does not write synthetic rows into Step 5, `zh_valid`, or `zh_test`.

Clean scientific experiments are kept separate from identifier-augmented operational controls.

## Files

Policy:

```text
schema/step15_evidence_type_policy.json
```

Auxiliary label builder:

```text
scripts/step15_build_evidence_type_labels.py
```

Curriculum training runner:

```text
scripts/step15_train_incremental_hard_negative.py
```

Planned outputs:

```text
reports/step15_evidence_type_label_summary.json
reports/step15_evidence_type_labels.en_content_train_pool.csv
reports/step15_evidence_type_labels.zh_target_strict.csv
reports/step15_incremental_hard_negative_summary.json
reports/step15_<experiment>_<phase>_seed_<seed>_predictions.zh_valid.csv
reports/step15_<experiment>_<phase>_seed_<seed>_predictions.zh_test.csv
reports/step15_<experiment>_<phase>_seed_<seed>_artifact.json
```

## Auxiliary Label Construction

The auxiliary label builder joins:

```text
reports/step5_en_frozen_silver_labels.csv
reports/step5_zh_target_strict_frozen_silver_labels.csv
reports/step7_pair_features.en_content_train_pool.csv
reports/step7_pair_features.zh_target_strict.csv
```

The join key is:

```text
pair_uid
```

The Step 7 feature CSV contains stale or blank review columns in some rows, so the builder always overlays labels, splits, review notes, and supervision flags from the Step 5 frozen files. This is important: Step 5 remains the source of truth for labels.

The builder emits two label layers.

Identity label:

```text
positive  -> same_controller
negative  -> different_controller
uncertain -> uncertain
```

Evidence type:

```text
same_controller_direct_identifier
same_controller_component_anchor
same_controller_style_structural_soft
template_clone_not_controller
semantic_topic_not_controller
public_contact_or_url_noise
ordinary_negative
uncertain_insufficient_evidence
```

The evidence-type rules are conservative and rule-based. They use only frozen labels, review strata, review notes, candidate rule hits, and Step 7 transfer-safe features. They do not create new positives.

## Curriculum Phases

Phase 0:

```text
same_controller_direct_identifier
ordinary_negative
```

Purpose: learn the cleanest identity boundary from direct positive anchors and ordinary negatives.

Phase 1:

```text
add semantic_topic_not_controller
```

Purpose: teach the model that same topic or same product does not imply same controller.

Phase 2:

```text
add template_clone_not_controller
```

Purpose: reduce false positives from copied product text, reusable templates, and boilerplate.

Phase 3:

```text
add public_contact_or_url_noise
```

Purpose: prevent public or non-seller-specific contact-like overlap from acting as identity proof.

Phase 4:

```text
add synthetic_train_only positive pair mixup
```

Purpose: regularize the minority positive boundary without changing Step 5 labels.

## Model

The first implementation deliberately avoids large models.

Backend:

```text
numpy_mlp_multitask
```

Architecture:

```text
standardized pair features
-> small tanh hidden layer
-> identity head: P(same_controller)
-> evidence head: P(evidence_type)
```

Loss:

```text
L_total = L_identity + lambda_evidence * L_evidence + L2
```

The identity head uses only `positive` and `negative` rows that are eligible for core transfer.

The evidence head uses confident evidence-type rows only.

`uncertain` rows are excluded from identity training.

Synthetic mixup rows, when enabled in Phase 4, are positive identity rows only and have masked evidence-type loss.

## Experiments

Clean scientific candidate:

```text
step15_e5_multitask_clean_curriculum
```

This uses semantic, structural, style, lexical, and template-proxy features. It excludes direct identifier features.

Identity-only control:

```text
step15_e5_identity_only_clean_curriculum
```

This uses the same clean feature set but sets `lambda_evidence = 0.0`. It tests whether the auxiliary evidence head matters.

Operational identifier control:

```text
step15_e5_multitask_identifier_operational
```

This includes direct identifier features and must not be mixed into the clean scientific claim.

## Evaluation

Primary evaluation remains the fixed Chinese test split:

```text
zh_target_strict test = 106 rows
positive = 21
negative = 85
```

Primary metrics:

```text
ROC-AUC
Average Precision
balanced accuracy
precision
recall
F1
```

Current reference baselines:

```text
raw E5:
  ROC-AUC = 0.806723
  AP      = 0.520573

Step9 E5 LR/L2 positive-pair mixup 100pct:
  ROC-AUC = 0.842017
  AP      = 0.588995
```

Step15 should not be claimed as successful from point estimates alone. After pair-level training, it must be added to Step 12 grouped bootstrap and then, only if justified, to Step 11 graph-level audit.

The Step 12 runner is prepared to read the main Step 15 candidates:

```text
step15_clean_multitask_phase2_seed_mean
step15_clean_multitask_phase3_seed_mean
step15_clean_multitask_phase4_seed_mean
step15_identity_only_phase4_seed_mean
step15_identifier_operational_phase4_seed_mean
```

## V2 Revision

The 2026-06-02 v2 revision keeps all first-pass Step 15 artifacts intact and writes new experiments under a separate `step15_v2_*` namespace. This prevents old and new experiment dimensions from overwriting each other.

New v2 summary output:

```text
reports/step15_v2_incremental_hard_negative_summary.json
```

New v2 Step 12 outputs:

```text
reports/step12_v2_statistical_robustness_zh_test_20260602.json
reports/step12_v2_statistical_robustness_model_metrics_20260602.csv
reports/step12_v2_statistical_robustness_paired_comparisons_20260602.csv
```

New v2 slice-level audit outputs:

```text
reports/step15_v2_slice_level_audit.json
reports/step15_v2_slice_level_audit.csv
```

The legacy first-pass Step 15 experiments remain available only as controls:

```text
step15_e5_multitask_clean_curriculum
step15_e5_identity_only_clean_curriculum
step15_e5_multitask_identifier_operational
```

They are no longer the default experiments in the v2 policy.
The Step 15 runner refuses these legacy names by default because their artifact and prediction templates resolve to the original first-pass paths. If a legacy control must be intentionally regenerated, the command must include `--allow-legacy-output-overwrite`; otherwise v2 work must use the `step15_v2_*` names.

## V2 Experiment Set

The v2 primary clean candidate is identity-only rather than multitask:

```text
step15_v2_identity_only_curriculum_from_scratch
```

Reason: the first-pass results did not support a claim that the auxiliary evidence-type head improved the binary identity ranking. The identity-only phase-4 seed mean was slightly stronger than the multitask phase-4 seed mean. Therefore v2 treats the evidence head as an ablation, not the main claim.

V2 controls:

```text
step15_v2_identity_only_curriculum_warm_start
step15_v2_identity_only_curriculum_domain_balanced
step15_v2_identity_only_curriculum_target_only
step15_v2_identity_only_curriculum_source_only
step15_v2_identity_only_curriculum_zh_positive_mixup
step15_v2_identity_only_curriculum_same_evidence_mixup
step15_v2_multitask_curriculum_ablation
step15_v2_identifier_operational
```

### Warm-Start Curriculum

The first implementation retrained each phase from scratch. That design can show staged hard-negative ablation, but it is not a strict incremental curriculum.

The v2 warm-start control changes this:

```text
phase0 -> phase1 -> phase2 -> phase3 -> phase4
```

For the same experiment and seed, each phase initializes from the previous phase's best parameters.

To keep parameter scales meaningful, the warm-start run uses one shared standardizer fitted on the final phase's training rows. The artifact records:

```text
initialization: random / warm_start
standardizer_source: warm_start_final_phase_train
training_mode: warm_start_curriculum
```

This tests whether gradual exposure to semantic-topic, template-clone, and contact/URL-noise negatives is better than independently retraining each phase.

### Domain-Balanced Control

The source English pool is larger than the target Chinese pool. V2 adds a domain-balanced identity loss control:

```text
step15_v2_identity_only_curriculum_domain_balanced
```

This keeps class-balanced positive/negative weighting, but also weights source and target domains so the larger English side cannot dominate the identity loss purely by row count.

Source/target controls are also added:

```text
step15_v2_identity_only_curriculum_source_only
step15_v2_identity_only_curriculum_target_only
```

These controls answer whether the observed ranking gain comes from source-domain strength, target-domain adaptation, or their combination.

### Mixup Scope Controls

The first implementation mixed all positive training rows. V2 separates mixup scope:

```text
all_positive
target_train_only
same_evidence_type_only
```

The two new v2 controls are:

```text
step15_v2_identity_only_curriculum_zh_positive_mixup
step15_v2_identity_only_curriculum_same_evidence_mixup
```

`target_train_only` tests whether synthetic minority regularization should be restricted to Chinese positives. `same_evidence_type_only` prevents direct-identifier positives and style/structural soft positives from being mixed into unrealistic synthetic rows.

Synthetic rows remain train-only. They are not written into Step 5, `zh_valid`, or `zh_test`, and their evidence-type loss is masked.

## V2 Slice-Level Audit

Overall ROC-AUC/AP is not enough for Step 15. The method is designed to reduce target-domain hard-negative concept drift, so v2 adds:

```text
scripts/step15_slice_level_audit.py
```

The audit reads fixed `zh_test` only and reports metrics by:

```text
evidence_type
review_stratum
hard_negative_any
negative_template_or_topic
positive_direct_or_component_anchor
positive_style_structural_soft
identifier_present
identifier_absent
```

The key scientific question is not only whether Step 15 improves global AP. It is whether it reduces scores for:

```text
template_clone_not_controller
semantic_topic_not_controller
public_contact_or_url_noise
```

while preserving recall for:

```text
same_controller_direct_identifier
same_controller_style_structural_soft
```

Only if the slice audit shows the expected hard-negative behavior should Step 15 be considered for exploratory Step 11 scoring.

This integration is for robustness testing only. It does not authorize Step 11 consumption until the fixed-test bootstrap comparisons justify it.

## Commands

Build auxiliary labels:

```bash
python3 scripts/step15_build_evidence_type_labels.py \
  --policy schema/step15_evidence_type_policy.json
```

Run the full first-pass Step15 experiment set:

```bash
python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json
```

Run only the v2 clean primary candidate:

```bash
python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json \
  --experiment step15_v2_identity_only_curriculum_from_scratch
```

Run only the final mixup phase for all seeds:

```bash
python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json \
  --experiment step15_v2_identity_only_curriculum_zh_positive_mixup \
  --phase phase4_add_positive_pair_mixup
```

Run the official Step 12 grouped bootstrap after Step 15 predictions exist:

```bash
python3 scripts/step12_statistical_robustness_audit.py
```

## Interpretation Rules

If Step15 improves AP but Step11 still produces mostly template/topic clusters, it is not a success.

If Step15 improves hard-negative slice behavior but global AP changes only modestly, it is still useful evidence that structured hard-negative training addresses concept drift.

If Step15 fails to improve over raw E5 and Step9 mixup, the result remains scientifically meaningful: it strengthens the conclusion that the current bottleneck is direct target-domain positive evidence scarcity, not simply model design.
