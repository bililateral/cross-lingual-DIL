# Step 16B Silver Positive Training Expansion

Date: 2026-07-06

## Purpose

The project has repeatedly hit the same bottleneck: the strict Chinese target-domain benchmark has too few positive seller-pair labels. The earlier strict policy required seller-facing identity anchors before a pair could be treated as a high-confidence positive. That policy protects benchmark validity, but it also leaves too few Chinese positives for stable target-domain adaptation.

Step 16B changes the data strategy for training only:

- keep the existing gold `zh_valid` and `zh_test` benchmark splits unchanged;
- expand only `zh_train` with weakly supervised `silver_train_only` positive pairs;
- assign lower training weights to silver rows so they do not dominate gold labels;
- record every selected silver pair in an audit CSV.

This is not a claim that all silver rows are gold same-controller truth. It is a controlled way to increase positive training support when strict proof-level Chinese positives are scarce.

## Inputs

- `reports/step5_zh_target_strict_frozen_silver_labels.csv`
- `reports/step4_zh_target_strict_silver_candidate_pairs.csv`
- `reports/step7_pair_features.zh_target_strict.csv`
- `schema/step16b_silver_positive_expansion_policy.json`

## Selection Logic

Step 16B selects candidates from the existing Chinese pair universe. It does not invent sellers, items, contacts, or labels outside the project data.

Hard exclusions:

- reviewed negative rows are never converted to positive;
- existing `valid` / `test` rows are not modified;
- any candidate whose seller appears in current `zh_valid` or `zh_test` supervision is excluded;
- every selected pair must already have a Step 7 pair-feature row.

Weak positive rules:

- `shared_contact_weak`: shared contact plus at least weak structural, lexical, or text-overlap support;
- `template_structural_weak`: shared title and description overlap with structural and lexical support;
- `clone_overlap_weak`: stronger title/description clone overlap with structural support;
- `high_similarity_weak`: high lexical similarity plus structural support;
- `rank_structural_weak`: high Step 4 candidate rank plus structural and lexical support.

Component closure:

After direct weak positives are selected, Step 16B builds seller components from existing train positives plus selected silver positives. If an unlabelled pair inside the same component already has a pair-feature row, it can be added as `silver_component_closure`.

## Training Weights

The expanded label file contains `training_sample_weight`:

- existing gold rows: `1.0`
- `silver_direct_or_contact`: `0.55`
- `silver_template_structural`: `0.40`
- `silver_component_closure`: `0.25`

Step 7, Step 9, and Step 15 were updated to multiply their class-balanced weights by this field. This keeps silver positives useful while reducing the risk that weak labels overwhelm stricter labels.

## Applied Counts

Before Step 16B:

- Chinese train positives: `61`
- Chinese valid positives: `14`
- Chinese test positives: `21`

After Step 16B:

- Chinese train positives: `231`
- Chinese valid positives: `14`
- Chinese test positives: `21`
- Added silver train-only positives: `170`

Silver composition:

- `silver_template_structural`: `85`
- `silver_direct_or_contact`: `56`
- `silver_component_closure`: `29`

Safety checks:

- duplicate `pair_uid`: `0`
- missing Step 7 pair features: `0`
- train-valid/test seller overlap: `0`
- existing negatives converted to positives: `0`

## Outputs

- `reports/step16b_silver_positive_candidate_pairs.csv`
- `reports/step16b_silver_positive_training_pairs.csv`
- `reports/step16b_silver_positive_expansion_summary.json`
- updated `reports/step5_zh_target_strict_frozen_silver_labels.csv`
- updated `reports/step5_frozen_silver_summary.json`
- updated `reports/step15_evidence_type_labels.zh_target_strict.csv`
- updated `reports/step15_evidence_type_label_summary.json`

## Interpretation

Step 16B is designed to solve training support scarcity, not to create a stronger gold benchmark. In paper writing, the correct description is:

> We use weakly supervised silver same-controller candidates only for target-domain training support. The fixed validation and test splits remain gold-only and unchanged.

The key ablation should compare:

- gold-only training;
- gold plus Step16B silver-train positives;
- Step16B plus Step15 evidence-type curriculum;
- Step16B plus Step9 positive-pair mixup.

If Step16B improves training but degrades final `zh_test`, the weak labels are too noisy and should be downweighted further or limited to `silver_direct_or_contact`.
