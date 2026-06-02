# Step 4: Silver-Label Candidate Construction

Step 4 does not create final labels. It builds review-ready seller pairs that are plausible sockpuppet candidates and preserves explicit evidence for later manual adjudication.

## Why This Step Exists

The downstream supervision pipeline requires candidate seller pairs before any human review can happen:

- English silver-label candidates for supervised training
- Chinese silver-label candidates for transfer evaluation and adaptation
- deterministic pre-review ranking and explicit evidence that can be handed off to Step 5

Without a controlled candidate-construction step, later labels are either too noisy or impossible to audit.

## Inputs

- `reports/step3_seller_profiles.en_content_train_pool.jsonl`
- `reports/step3_seller_profiles.zh_target_strict.jsonl`
- `reports/step3_seller_profiles.zh_target_aux.jsonl`
- `reports/step2_aux_pgp_evidence_manifest.csv`

## Candidate Pools

Step 4 mines candidates separately for:

- `en_content_train_pool`
- `zh_target_strict`
- `zh_target_aux`

`zh_target_aux` remains separate on purpose because `products_data.csv` still lacks a strict market provenance field.

## Evidence Rules

### High-precision rules

- `shared_contact_exact`
- `shared_description_clone`
- `shared_title_clone`
- `shared_pgp_fingerprint_via_aux_alias`

These rules are designed to be explicit and auditable. Each retained pair stores the actual shared values.

### Recall-oriented rule

- `profile_lexical_neighbor`

This rule uses deterministic sparse retrieval over seller-profile text components. It is a retrieval rule only, not a final labeling rule.

- English uses word-level tokens and bigrams
- Chinese uses character 2-3 grams plus Latin alphanumeric tokens
- signature titles and signature description segments from Step 3 are included when available

Important boundary:

- raw `lexical_similarity` is pool-specific retrieval evidence
- it must not be treated as a language-invariant calibrated score for Step 8 zero-shot transfer
- if a final pairwise classifier uses sparse lexical signals later, they must be standardized within language/pool or replaced by multilingual embedding similarities

### Support rule

- `structural_support`

This rule is not used alone to auto-create labels. It acts as supporting evidence using:

- category overlap
- item-count ratio
- approximate price-median ratio
- style-stat similarity

## Important Filtering

To reduce false positives, Step 4 applies additional controls:

- noisy contact tokens are filtered with a stoplist and per-type validation
- very common duplicate clusters are ignored
- short or generic titles and descriptions are ignored as clone evidence
- Chinese retrieval does not rely on whitespace tokenization
- same-alias pairs are retained but tagged as `same_alias_identity_continuity`, not mixed with the primary sockpuppet review scope

## Outputs

- pool-specific candidate-pair tables in CSV
- a summary JSON with rule counts, scope counts, and acceptance checks

Operational note:

- Step 4 can export raw review queues as intermediate artifacts
- once Step 5 balanced review queues are built, those raw queues are superseded and are not the active review entrypoint

## Acceptance Criteria

1. Every candidate pair retains explicit evidence codes
2. Any exported raw review-queue row includes review fields with default pending state
3. No candidate pair is auto-labeled as final positive or final negative
4. `zh_target_aux` remains separate from `zh_target_strict`
