# Evidence-Type Incremental Hard-Negative Learning Plan

Updated: 2026-06-02

Branch: `method/evidence-type-incremental-hard-negative`

## 1. Research Motivation

The current active project has already established a strict cross-lingual seller-pair verification pipeline:

- Step 5 freezes binary supervision labels: `positive`, `negative`, `uncertain`.
- Step 7 builds transfer-safe pair features and raw semantic controls.
- Step 9 performs target-domain adaptation and positive-pair mixup.
- Step 11 turns pair scores into candidate graph clusters for audit.
- Step 12 performs grouped bootstrap robustness checks on the fixed Chinese test split.
- Step 13 audits English-to-Chinese concept drift.

The strongest current point estimate is:

```text
Step9 E5 LR/L2 positive-pair mixup 100pct:
ROC-AUC = 0.842017
AP      = 0.588995
```

However, Step 12 grouped bootstrap does not yet support a statistically robust claim over raw E5:

```text
raw E5:
ROC-AUC = 0.806723
AP      = 0.520573

mixup 100pct vs raw E5:
ROC-AUC diff = +0.035294, CI = [-0.086161, 0.158168]
AP diff      = +0.068422, CI = [-0.207105, 0.331671]
```

Step 11 cluster audit also shows that most retained clusters are not high-confidence same-controller clusters:

```text
same_controller_high_confidence              = 0
same_controller_core_with_possible_expansion = 0
partial_anchor                               = 6
template_clone_not_controller                = 59
semantic_topic_not_controller                = 52
uncertain                                    = 8
```

This means the current bottleneck is not simply model capacity. The main problem is that Chinese target-domain negative examples are heterogeneous and hard:

- Some are ordinary unrelated sellers.
- Some are high-semantic template clones.
- Some are same-topic or same-product sellers.
- Some share public or non-seller-specific identifiers.
- Some are uncertain because evidence is insufficient.

The binary label `negative` hides this internal structure. A model trained only on `positive` vs `negative` can learn that a pair is negative, but it does not learn why it is negative. This is especially harmful in Chinese target-domain transfer, where the dominant error source is not random noise, but high-similarity non-controller pairs.

This branch proposes an evidence-type incremental hard-negative learning method to address that issue.

## 2. What This Is Not

This is not standard class-incremental learning.

Standard class-incremental learning usually means:

```text
Stage 1: learn classes A and B
Stage 2: a new class C arrives
Stage 3: new classes D and E arrive
Goal: learn new classes without forgetting old classes
```

Our final task does not introduce new final prediction classes. The final target remains:

```text
same_controller vs different_controller
```

Therefore this branch should not be described as classic class-incremental learning. The correct framing is:

```text
evidence-type incremental learning
hard-negative incremental learning
curriculum learning over target-domain negative evidence types
```

The goal is not to replace binary seller-pair verification. The goal is to make the binary scorer more robust by teaching it the internal structure of difficult negative cases.

## 3. Core Scientific Hypothesis

The current binary supervision setup collapses several evidence regimes into one `negative` class. This causes the model to under-learn target-domain hard-negative mechanisms such as template reuse and semantic-topic similarity.

Hypothesis:

```text
If the model is incrementally exposed to evidence-specific negative subtypes
and trained with an auxiliary evidence-type objective,
then it will reduce template/topic false positives while preserving or improving
same-controller ranking on the fixed Chinese test set.
```

The expected improvement should appear in:

- fixed `zh_test` AP;
- hard-negative slice AP;
- reduced false positives in template/topic slices;
- improved Step 11 cluster audit distribution;
- more stable Step 12 grouped bootstrap comparisons.

## 4. Label Design

### 4.1 Keep Step 5 Labels Frozen

This method must not rewrite Step 5 ground truth.

The frozen Step 5 label vocabulary remains:

```text
positive
negative
uncertain
```

These labels define the final identity supervision and evaluation. The proposed method adds an auxiliary training-only label, not a replacement label.

### 4.2 Two-Layer Label Structure

The method uses two label layers.

First layer: identity label.

```text
identity_label:
  same_controller
  different_controller
  uncertain
```

Mapping from Step 5:

```text
positive  -> same_controller
negative  -> different_controller
uncertain -> uncertain
```

Second layer: evidence/noise type.

```text
evidence_type:
  same_controller_direct_identifier
  same_controller_component_anchor
  same_controller_style_structural_soft
  template_clone_not_controller
  semantic_topic_not_controller
  public_contact_or_url_noise
  ordinary_negative
  uncertain_insufficient_evidence
```

The final deployed score remains:

```text
P(same_controller)
```

The auxiliary objective predicts:

```text
P(evidence_type)
```

This is a multi-task setup:

```text
main task:      binary identity verification
auxiliary task: evidence/noise-type classification
```

### 4.3 Why Not Flat Multiclass

A flat multiclass design would use classes like:

```text
same_controller_direct_identifier
template_clone_not_controller
semantic_topic_not_controller
ordinary_negative
...
```

That is less suitable because the final research task is still same-controller verification. A flat multiclass model can become unstable when small evidence subtypes have few samples, and it obscures the primary question:

```text
Is this pair same-controller?
```

The multi-task design keeps the identity decision as the main output and uses evidence-type labels to structure training.

## 5. Evidence-Type Definitions

The evidence-type labels must be conservative. They should be derived only from existing frozen labels, Step 7 features, Step 11 audit decisions, and explicit rule outputs. They must not introduce new positive ground truth.

### 5.1 same_controller_direct_identifier

Definition:

```text
Step 5 label is positive
and pair has seller-facing direct identity support.
```

Allowed evidence:

- shared PGP fingerprint;
- shared Telegram/Jabber/QQ/WeChat/phone handle;
- shared seller-facing wallet address;
- direct contact reuse that is not parser noise, public URL, product/victim data, or marketplace boilerplate.

This is the strongest positive subtype.

### 5.2 same_controller_component_anchor

Definition:

```text
Step 5 label is positive
and pair belongs to a seller component supported by direct anchor evidence,
but the exact pair may be component-derived rather than direct-contact-derived.
```

Use with caution:

- allowed only for training diagnostics;
- should not be used to create new test positives;
- must not include closure-derived audit-only positives excluded from core transfer.

### 5.3 same_controller_style_structural_soft

Definition:

```text
Step 5 label is positive
but direct identifier evidence is absent;
the positive label is supported by strong style/structure/content continuity.
```

This subtype is weaker than direct identifier positives. It should be separated so the model can learn that not all positives have the same evidence profile.

### 5.4 template_clone_not_controller

Definition:

```text
Step 5 label is negative
or Step 11 audit decision is template_clone_not_controller,
and the pair has high textual/template overlap without seller-specific identity evidence.
```

Signals:

- high shared title/description/template overlap;
- high boilerplate ratio;
- rare-ngram or low-df sentence overlap that appears to be reusable market text;
- no shared seller-facing contact;
- distinct seller contacts if available.

This is the central hard-negative subtype for the Chinese target domain.

### 5.5 semantic_topic_not_controller

Definition:

```text
Step 5 label is negative
or Step 11 audit decision is semantic_topic_not_controller,
and the pair is similar because of topic/product/category rather than identity.
```

Signals:

- high raw semantic embedding similarity;
- same product category or market topic;
- low direct structural/identifier support;
- no seller-facing direct anchor.

This subtype separates "same topic" from "same controller".

### 5.6 public_contact_or_url_noise

Definition:

```text
Pair appears to share contact-like or URL-like evidence,
but the evidence is public, non-seller-specific, parser noise, or product/victim data.
```

Examples:

- public website URL;
- marketplace support URL;
- product demo URL;
- leaked/victim email;
- contact string extracted from product content rather than seller identity;
- high-frequency contact reused by many sellers.

This subtype is needed because identifier features can become dangerous if all contact-like overlap is treated as identity evidence.

### 5.7 ordinary_negative

Definition:

```text
Step 5 label is negative
and there is no strong semantic/topic/template/contact ambiguity.
```

This is the easy negative class.

It should be used in early training phases but should not dominate the final hard-negative curriculum.

### 5.8 uncertain_insufficient_evidence

Definition:

```text
Step 5 label is uncertain
or Step 11 audit decision is uncertain.
```

Uncertain rows must not be treated as positive or negative identity supervision. They can be used only for:

- auxiliary uncertainty diagnostics;
- calibration review queues;
- active learning candidate selection.

They should not directly train the binary identity head unless a separate semi-supervised method is explicitly designed.

## 6. Data Sources for Auxiliary Labels

Auxiliary `evidence_type` labels can be derived from:

```text
reports/step5_zh_target_strict_frozen_silver_labels.csv
reports/step5_en_frozen_silver_labels.csv
reports/step7_pair_features.zh_target_strict.csv
reports/step7_pair_features.en_content_train_pool.csv
reports/step11_cluster_level_audit.current_20260517.json/csv
reports/step11_current_manifest_20260517.json
```

Allowed feature families:

- raw semantic cosine features;
- structural overlap features;
- style-gap features;
- identifier features;
- template proxy features;
- relation reliability features if produced by Step 11.

Forbidden:

- writing auxiliary labels into Step 5 frozen files;
- using Step 11 cluster labels as ground truth same-controller labels;
- using `uncertain` as positive or negative identity labels;
- putting synthetic or auxiliary labels into `zh_valid` or `zh_test`;
- using future test information to define train labels.

## 7. Proposed Training Design

### 7.1 Model Family

Do not start with a large model.

The first implementation should use small, auditable models:

```text
Option A: Logistic Regression / Linear SVM style residual scorer
Option B: shallow MLP with two heads
Option C: gradient-boosted model only as a diagnostic, not mainline
```

Recommended first model:

```text
shared linear/MLP trunk
identity head: binary same_controller score
evidence head: evidence_type distribution
```

Input features:

```text
raw E5
raw LaBSE
raw BGE-M3
style-gap features
structural overlap features
identifier features
template proxy features
relation reliability features
```

The primary output should be a residual score:

```text
final_score = raw_e5 + correction(features)
```

or a calibrated identity probability:

```text
P(same_controller | features)
```

The raw E5 baseline must remain visible in the output so the method can be compared against the current strongest unsupervised semantic scorer.

### 7.2 Loss Function

The training loss should combine:

```text
L_total = L_identity + lambda_evidence * L_evidence + lambda_reg * L_regularization
```

Identity loss:

- binary cross entropy;
- class-balanced or sample-weighted only if carefully audited;
- no use of uncertain rows as binary labels.

Evidence loss:

- multiclass cross entropy over `evidence_type`;
- only for rows with confident evidence-type labels;
- missing/uncertain evidence-type labels should be masked out.

Regularization:

- L2 regularization for linear/logistic models;
- dropout only for shallow MLP;
- early stopping on fixed `zh_valid`.

### 7.3 Incremental Curriculum

The method should be trained/evaluated in phases.

Each phase adds a harder evidence type while preserving prior categories.

Phase 0: identity warm start.

```text
Train on:
  same_controller_direct_identifier
  ordinary_negative

Purpose:
  learn the cleanest binary identity boundary.
```

Phase 1: add semantic-topic negatives.

```text
Add:
  semantic_topic_not_controller

Purpose:
  teach the model that same topic / same product does not imply same controller.
```

Phase 2: add template-clone negatives.

```text
Add:
  template_clone_not_controller

Purpose:
  reduce false positives caused by copied product descriptions, reused templates, and boilerplate.
```

Phase 3: add contact/URL noise.

```text
Add:
  public_contact_or_url_noise

Purpose:
  prevent identifier features from treating public or non-seller-specific contacts as same-controller proof.
```

Phase 4: add positive-pair mixup.

```text
Add:
  synthetic_train_only positive pair mixup

Purpose:
  regularize the minority positive boundary without changing Step 5 labels.
```

Phase 5: final binary scorer.

```text
Train/evaluate final identity scorer using the best curriculum configuration.
Report:
  P(same_controller)
  evidence_type diagnostics
```

## 8. Evaluation Protocol

Evaluation must use the current fixed Chinese test boundary:

```text
zh_target_strict test = 106 rows
positive = 21
negative = 85
```

No train/valid/test mixing is allowed.

Primary pair-level metrics:

```text
ROC-AUC
Average Precision
balanced accuracy
precision
recall
F1
```

Robustness metrics:

```text
Step12 grouped bootstrap CI
paired comparison against raw E5
paired comparison against Step9 E5 mixup 100pct
```

Slice metrics:

```text
identifier_present
identifier_absent
high_e5_semantic_no_identifier
template_clone_not_controller
semantic_topic_not_controller
public_contact_or_url_noise
```

Graph-level metrics:

```text
Step11 cluster audit decision distribution
template_clone_not_controller cluster count
semantic_topic_not_controller cluster count
partial_anchor count
same_controller_high_confidence count
```

The method is not successful if it only improves global AUC while Step11 remains dominated by template/topic false positives.

## 9. Baselines

The new branch must compare against the current baselines:

Baseline A: raw E5.

```text
ROC-AUC = 0.806723
AP      = 0.520573
```

Baseline B: Step9 E5 LR/L2 50pct.

```text
ROC-AUC ~= 0.819048
AP      ~= 0.540494
```

Baseline C: Step9 E5 LR/L2 positive-pair mixup 100pct.

```text
ROC-AUC = 0.842017
AP      = 0.588995
```

Operational control:

```text
identifier_augmented_few_shot_default_lr_l2 / 100pct:
ROC-AUC = 0.783754
AP      = 0.647686
```

The identifier control has high AP but is not a clean scientific baseline because it uses direct identifier features. It should remain an operational control.

## 10. Success Criteria

The method should be considered promising only if it satisfies most of the following:

```text
1. AP exceeds 0.588995 on fixed zh_test.
2. ROC-AUC is competitive with or above 0.842017.
3. Step12 grouped bootstrap is more stable than current mixup comparisons.
4. Template/topic false positives decrease in slice analysis.
5. Step11 audit shows fewer template_clone_not_controller and semantic_topic_not_controller clusters.
6. partial_anchor or anchor-supported candidates increase without using identifier leakage in the clean scorer.
7. The model does not rely on direct identifier controls unless explicitly labeled as operational.
```

Minimum acceptable result:

```text
The method improves hard-negative slice behavior even if global AUC/AP gains are small.
```

Failure condition:

```text
Global AP/AUC improves slightly, but Step11 audit remains dominated by template/topic non-controller clusters.
```

## 11. Leakage Controls

The following controls are mandatory:

- keep Step 5 frozen labels unchanged;
- keep fixed `zh_valid` and `zh_test` unchanged;
- generate auxiliary labels only for training and diagnostics;
- do not use Step11 cluster membership as same-controller ground truth;
- do not allow synthetic rows into validation or test;
- component/group-aware bootstrap must remain the robustness audit;
- report clean scorer separately from identifier-augmented operational controls.

## 12. Implementation Plan

This branch should start with planning only. Implementation should be staged.

Stage A: auxiliary-label audit builder.

Potential files:

```text
schema/step15_evidence_type_policy.json
scripts/step15_build_evidence_type_labels.py
reports/step15_evidence_type_label_summary.json
reports/step15_evidence_type_labels.zh_target_strict.csv
reports/step15_evidence_type_labels.en_content_train_pool.csv
```

Stage B: curriculum training runner.

Potential files:

```text
scripts/step15_train_incremental_hard_negative.py
reports/step15_incremental_hard_negative_summary.json
reports/step15_predictions.<experiment>.zh_test.csv
```

Stage C: robustness and graph audit integration.

Potential changes:

```text
schema/step12_statistical_robustness_policy.json
scripts/step12_statistical_robustness_audit.py
schema/step11_clustering_policy.json
scripts/step11_cluster_chinese_graph.py
```

Only after pair-level and Step12 results justify it should Step11 consume a new scorer.

## 13. Expected Scientific Contribution

The expected contribution is not "class-incremental learning solves class imbalance."

The correct contribution is:

```text
Evidence-type incremental hard-negative learning reduces target-domain
hard-negative concept drift by separating semantic/topic/template similarity
from identity reliability in cross-lingual seller-pair verification.
```

This directly addresses the current project bottleneck:

```text
semantic similarity != identity reliability
```

If successful, the method can support a stronger claim than ordinary few-shot adaptation:

```text
Target-domain adaptation becomes more reliable when the model is trained not only
on positive scarcity, but also on structured negative evidence types.
```

If unsuccessful, the negative result is still useful:

```text
Even evidence-type hard-negative modeling cannot overcome the current scarcity
of direct target-domain same-controller evidence.
```

That would further support the evidence-scarcity conclusion already suggested by Step 11 and Step 13.

