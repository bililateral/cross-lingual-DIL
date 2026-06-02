# Step 5: Stratified Review and Silver-Label Freeze

Status: active boundary = 2026-04-23 English valid/test top-up refreeze; the v2 milestone snapshot is historical, and its report folder was removed during the 2026-04-22 reports cleanup

## 2026-04-22 / 2026-04-23 Item-Level Identity Extraction Check

Codex moved the next positive-anchor search upstream into Step 3 item-level parsing:

- Step 3 parser: `scripts/step3_build_seller_profiles.py`
- Step 3 schema update: `schema/step3_seller_profile_schema.json`
- item-level identity signal files:
  - `reports/step3_item_identity_signals.en_content_train_pool.csv`
  - `reports/step3_item_identity_signals.zh_target_strict.csv`
  - `reports/step3_item_identity_signals.zh_target_aux.csv`
- Step 5 item-identity policy: `schema/step5_item_identity_expansion_policy.json`
- Step 5 item-identity builder: `scripts/step5_build_item_identity_expansion_queue.py`
- queue: `reports/step5_zh_target_strict_item_identity_expansion_queue.20260422.csv`
- queue summary: `reports/step5_item_identity_expansion_queue_summary.20260422.json`
- review summary: `reports/step5_item_identity_expansion_codex_review_summary.20260422.json`

Step 3 rerun acceptance checks still pass:

- seller counts match Step 2 for all eligible buckets
- item counts match Step 2 for all eligible buckets
- all profile text remains non-empty

Chinese strict item-level extraction result after the 2026-04-23 high-precision Chinese contact patch:

- item-level identity signals: `4,430`
- direct-identity-eligible signals after seller-facing / product-risk gates: `1,890`
- eligible token groups: `1,175`
- shared seller-facing direct token groups: `44`
- candidate pairs surviving frozen/reviewed exclusion: `0`
- skipped shared-token pairs: `50 frozen_pair`

Interpretation:

- the parser upgrade is implemented and reproducible, including Telegram/TG/纸飞机 compact forms, Wechat/VX/WX/V: compact forms, QQ/企鹅 compact numeric forms, Jabber/XMPP, phone, wallet, PGP, Bat/蝙蝠 numeric forms, email, and support-only external URL extraction with context flags
- the current raw item text does not expose new unreviewed seller-facing shared direct identifiers in `zh_target_strict`
- every shared direct-token seller pair discovered by the item-level parser is already represented in the frozen/reviewed Step 5 boundary
- therefore no Step 5 labels were added, no Step 5 freeze was rerun, and no Step 7/9/11 rerun is warranted from this check alone
- the bottleneck is now raw evidence availability, not downstream model triage; further positive expansion requires new raw/OCR/source fields or manual external evidence, not repeated Step 11 graph sampling

## 2026-04-22 Paper-Targeted Expansion Check

Codex added a non-destructive paper-targeted expansion route after the strict Step 11 direct-identity audit:

- policy: `schema/step5_paper_targeted_expansion_policy.json`
- builder: `scripts/step5_build_paper_targeted_expansion_queue.py`
- conservative reviewer: `scripts/step5_codex_review_paper_targeted_expansion.py`
- queue: `reports/step5_zh_target_strict_paper_targeted_expansion_queue.20260422.csv`
- queue summary: `reports/step5_paper_targeted_expansion_queue_summary.20260422.json`
- review summary: `reports/step5_paper_targeted_expansion_codex_review_summary.20260422.json`

This queue used the then-current Step 11 outputs only as a candidate miner. It read exactly the six previous-boundary Step 11 summaries and the strict direct review output; it did not glob stale `reports/` files and it did not treat clusters as ground truth.

Result:

- selected rows: `20`
- `robust_lr_l2_high_score_unreviewed`: `8`
- `identifier_control_high_score_unreviewed`: `8`
- `direct_proof_anchor_neighbor_unreviewed`: `4`
- unreviewed non-URL shared direct-contact pairs: `0`
- conservative Codex labels: `20 uncertain`, `0 positive`, `0 negative`

Interpretation:

- the useful part of the paper-targeted expansion proposal is now implemented: Step 11 can feed a small, reproducible review queue around stable LR/L2 highs, identifier-control highs, and the four strict Telegram proof pairs
- the strict review found no new defensible positive labels because every selected row lacked an independent seller-facing shared identifier
- these rows should not be applied to Step 5 supervision; they document direct-identity evidence scarcity and can be revisited only with additional raw item-level evidence
- this reinforces the current publication discipline: LR/L2 supports a ranking/discovery improvement claim, while proof-level same-controller claims remain limited to direct-contact pairs unless new anchors are found

## 2026-04-20 Step 5 v3 Source Correction

The first Step 5 v3 targeted review queue was built from an earlier stale Step 11 graph:

- `reports/step11_core_few_shot_bge_m3_ratio_10pct_seed_20260320_clustering_summary.json`
- graph threshold `0.457259`

That source is now treated as invalid for future queue construction because it came from a small-sample BGE few-shot run with a suspicious perfect test score and an overly broad threshold-pass graph.

The calibrated-BGE correction was invalidated on 2026-04-20. Its queue summary recorded stale graph diagnostics (`1663 -> 389`) that do not match the current Step 11 `core_calibrated_bge_m3` graph (`273 -> 45` at threshold `0.5`). Those artifacts have been archived under:

- `reports/deprecated_step5_v3_calibrated_bge_20260420/`

The current corrected Step 5 v3 queue source is:

- `reports/step11_core_calibrated_default_clustering_summary.json`
- `reports/step11_core_calibrated_default_zh_target_strict_scored_pairs.csv`
- `reports/step11_core_calibrated_default_zh_target_strict_clusters.threshold_0800000.csv`

The active corrected policy and outputs are:

- `schema/step5_v3_targeted_review_policy.json`
- `reports/step5_zh_target_strict_targeted_review_queue.step11_calibrated_default_v3.csv`
- `reports/step5_zh_target_strict_targeted_rereview_queue.step11_calibrated_default_v3.csv`
- `reports/step5_v3_targeted_review_queue_summary.calibrated_default.json`

The corrected queue has:

- `0` net-new review rows
- `65` rereview rows
- `0` missing retained pairs
- source scorer `core_calibrated_default`
- graph threshold `0.8`
- threshold-pass edges `305`
- post-filter edges `65`
- consistency checks all true

The active freeze has now consumed the corrected calibrated-default queue. The 65-row rereview queue was fully adjudicated, applied through `schema/step5_v3_targeted_cleanup_policy.calibrated_default_20260420.json`, and frozen into `reports/step5_frozen_silver_summary.json`.

Step 5 is where candidate pairs become usable supervision. The main risk at this stage is not random noise. The main risk is supervision collapse: if reviewers mostly confirm identifier-heavy pairs, the later classifier learns a shortcut and stops looking at content.

## Why This Step Exists

Step 4 produced review-ready candidates, but not balanced supervision. Step 5 must therefore do two things at once:

- adjudicate candidate pairs
- preserve evidence diversity for later supervised training

## Review Strata

Every candidate pair is assigned one review stratum:

- `identifier_plus_text`
- `text_clone_primary`
- `semantic_structural`
- `identifier_primary`
- `semantic_only`
- `same_alias_continuity`

These strata are not cosmetic. They are later used to audit label balance and shortcut risk.

## Hard Rules

- `same_alias_continuity` stays outside the primary sockpuppet supervision pool
- review should not proceed only by global rank score
- reviewed positives must not be dominated by identifier-led strata when non-identifier positives are available
- each frozen silver row must retain `review_stratum`, `review_status`, `review_label`, `reviewer_id`, and `review_notes`

## Training Link

Step 5 is already coupled to Step 7:

- the core transfer model should preferentially learn from non-identifier-supported positives
- identifier-supported positives remain valuable, but must later be masked or downweighted in training

## Outputs

- balanced review queues by pool
- review-stratum summary JSON
- frozen silver labels after human adjudication

## Freeze Rules

Once reviewers fill `review_label`, Step 5 must freeze labels under explicit controls:

- only rows with final `review_label in {positive, negative, uncertain}` may enter frozen outputs
- `same_alias_continuity` may be frozen for audit, but not for primary supervision
- if raw seller aliases collapse to the same normalized alias after stripping market trust-score suffixes such as `(100%)`, they must also be treated as audit-only continuity rows
- reviewed `positive` and `negative` rows must be split by seller-connected components, not by raw pair rows, to avoid seller leakage across train/valid/test
- component assignment should minimize the global train/valid/test deviation from target ratios, not greedily optimize one split in isolation
- every frozen row must retain:
  - `review_stratum`
  - `review_status`
  - `review_label`
  - `reviewer_id`
  - `review_notes`

## Frozen Artifacts

The freeze stage is executed by:

- `schema/step5_freeze_policy.json`
- `scripts/step5_freeze_silver_labels.py`

It produces:

- `reports/step5_en_frozen_silver_labels.csv`
- `reports/step5_zh_target_strict_frozen_silver_labels.csv`
- `reports/step5_zh_target_aux_frozen_silver_labels.csv`
- `reports/step5_frozen_silver_summary.json`

## 2026-04-21 English Source-Domain Expansion

The English source pool was expanded after the Chinese positive-anchor pass because the previous source-domain supervision was too small for a convincing large-source/small-target transfer setup:

- previous English supervision: `123` rows
- previous English split counts: `train = 85`, `valid = 19`, `test = 19`
- new English supervision: `382` rows
- new English split counts: `train = 221`, `valid = 63`, `test = 98`

Artifacts:

- policy: `schema/step5_en_source_expansion_policy.json`
- builder: `scripts/step5_build_en_source_expansion_queue.py`
- conservative review: `scripts/step5_codex_review_en_source_expansion.py`
- apply reviewed labels: `scripts/step5_apply_en_source_expansion_reviews.py`
- queue: `reports/step5_en_source_expansion_queue.20260421.csv`
- queue summary: `reports/step5_en_source_expansion_queue_summary.20260421.json`
- review summary: `reports/step5_en_source_expansion_codex_review_summary.20260421.json`
- apply summary: `reports/step5_en_source_expansion_apply_summary.20260421.json`

The expansion selected `544` pending English rows:

- `54` seller-facing identifier-plus-text candidates
- `90` seller-facing identifier-primary candidates
- `180` strong text-clone positive probes
- `220` hard-negative template probes

Codex review applied:

- `61 positive`
- `198 negative`
- `285 uncertain`

Important review discipline:

- product/victim/sample credentials were not treated as seller identity anchors
- strong text clones without identity closure were usually marked `uncertain`
- alias-related hard-negative candidates were marked `uncertain`
- negatives were hard-boundary/template controls, not random easy negatives

The English freeze policy ratio is now `60/15/25` for `train/valid/test`, which is necessary for an English test set near `100` rows. The refreeze passes seller-overlap and coverage checks.

## 2026-04-21 English Source-Domain Top-Up

The first English expansion reached the requested floor but remained close to the lower bound. A conservative top-up pass was added to test whether more high-quality English supervision was still available without padding random easy negatives:

- policy: `schema/step5_en_source_expansion_topup_policy.json`
- queue: `reports/step5_en_source_expansion_topup_queue.20260421.csv`
- queue summary: `reports/step5_en_source_expansion_topup_queue_summary.20260421.json`
- review summary: `reports/step5_en_source_expansion_topup_codex_review_summary.20260421.json`
- apply summary: `reports/step5_en_source_expansion_topup_apply_summary.20260421.json`

The top-up selected `220` rows:

- `120` seller-facing identifier-primary candidates
- `40` strong text-clone positive probes
- `60` hard-negative template probes

Codex review applied:

- `2 positive`
- `56 negative`
- `162 uncertain`

Final English freeze after top-up:

- `909` reviewed rows
- `440` supervision rows
- `train = 251`, `valid = 75`, `test = 114`
- `train = 87 positive / 164 negative`
- `valid = 21 positive / 54 negative`
- `test = 36 positive / 78 negative`

Interpretation: the English source set is now adequate for the next Linux Step 7 rerun. The top-up also shows that remaining direct positive anchors are mostly depleted under the conservative evidence rubric; additional expansion should move upstream to better seller-facing contact extraction rather than continuing to label broad pending queues.

The final refreeze also includes a normalized-alias split guard: reviewed supervision rows that share the same normalized seller alias are kept inside one split even when their direct pair evidence is uncertain. This removed the `bestgroup` soft train/test overlap while preserving the same supervision row count.

## 2026-04-22 English Item-Level Identity Expansion

Codex used the Step 3 item-level identity signal extraction to add a final conservative English source-domain positive pass:

- policy: `schema/step5_en_item_identity_expansion_policy.json`
- builder: `scripts/step5_build_item_identity_expansion_queue.py`
- conservative review: `scripts/step5_codex_review_item_identity_expansion.py`
- apply reviewed labels: `scripts/step5_apply_item_identity_expansion_reviews.py`
- queue: `reports/step5_en_item_identity_expansion_queue.20260422.csv`
- queue summary: `reports/step5_en_item_identity_expansion_queue_summary.20260422.json`
- review summary: `reports/step5_en_item_identity_expansion_codex_review_summary.20260422.json`
- apply summary: `reports/step5_en_item_identity_expansion_apply_summary.20260422.json`

Queue construction result:

- input English item-level identity signals: `294,261`
- direct-identity-eligible signals after seller-facing/product-risk gates: `183,215`
- eligible token groups: `1,221`
- shared token groups passing frequency bounds: `128`
- selected rows after excluding generic parser tokens, soft same-alias continuity, stale reviewed/frozen pairs, and per-token clique inflation: `36`
- selected anchor types: `2 PGP public key`, `2 crypto wallet`, `3 Telegram`, `21 Wickr`, `2 phone`, `5 email`, `1 WeChat-like parser hit`

Conservative review:

- `35 positive`
- `1 negative`
- the negative is a public YouTube video ID that the parser captured as a WeChat-like token
- no random easy negatives were added

Application/refreeze:

- Step 4 candidates appended: `22`
- existing Step 4 candidates updated with item-level direct-identifier evidence: `14`
- Step 5 active English rows appended: `22`
- existing Step 5 active English rows updated: `14`
- final English reviewed rows: `945`
- final English supervision rows: `476`
- final English split counts before validation repair: `train = 273`, `valid = 80`, `test = 123`
- final English split labels:
  - `train = 108 positive / 165 negative`
  - `valid = 27 positive / 53 negative`
  - `test = 44 positive / 79 negative`
- primary English positive supervision rows: `179`
- seller overlap across train/valid/test: `0`
- normalized-alias overlap across train/valid/test: `0`
- non-identifier positive share: `0.340782`, still above the `0.3` policy floor

Interpretation: the English side became a stronger source domain for the large-source/small-target transfer narrative. This item-level identity refreeze was later followed by the label-stratified validation repair below; use the repaired split counts as the active boundary.

## Step 5 v2 Targeted Review Discipline

After the final Step 11 hardening pass, the project is no longer blocked by tree-like bridge explosions. The remaining uncertainty sits inside a small number of dense few-shot cliques, especially:

- the 14-node four-piece template-copy clique
- the 8-node mixed-evidence clique that contains one strong identifier anchor pair plus several semantic expansion edges

That means any Step 5 reopening must now follow a stricter rule:

- do **not** edit the active `reports/step5_zh_target_strict_balanced_review_queue.csv` directly
- do **not** treat the current `140 / 41 / 38` Chinese strict split as untouched once the targeted review starts feeding new labels back into Step 5
- build a separate targeted review queue first, finish the human review there, and only then decide whether to cut a new Step 5 freeze and a new final holdout

The current targeted queue artifact is:

- `schema/step5_targeted_review_policy.json`
- `scripts/step5_build_targeted_review_queue.py`
- `reports/step5_zh_target_strict_targeted_review_queue.step11_v1.csv`
- `reports/step5_zh_target_strict_targeted_rereview_queue.step11_v1.csv`
- `reports/step5_targeted_review_queue_summary.json`

Before any Step 5 v3 work started, the project froze the full v2 milestone here:

- `schema/step5_v2_milestone_snapshot_policy.json`
- `scripts/step5_snapshot_milestone.py`
- `reports/step5_v2_milestone_snapshot_20260416`
- `reports/step5_v2_milestone_snapshot_summary.json`

This queue is intentionally narrow. It currently contains:

- representative cross-market template-clone edges sampled from the 14-node clique
- retained uncertain expansion edges from the 8-node mixed-evidence clique

If those edges are already present inside the active Step 5 boundary, the builder does **not** duplicate them as net-new rows. Instead, it writes them to the targeted **re-review** queue so the reviewer can audit them without silently rewriting the current freeze first.

It intentionally does **not** queue:

- the strong positive anchor `/shop/444654 <-> /shop/459141`
- `/shop/454420 <-> /shop/462498`, because that audited false-positive candidate is not retained in the current final Step 11 primary graph and therefore should stay on a secondary watchlist

Manual-review entrypoint:

- `scripts/step5_manual_review_cli.py`

Example usage:

```bash
python3 scripts/step5_manual_review_cli.py --summary
python3 scripts/step5_manual_review_cli.py --pool zh_target_strict --reviewer-id manual_zhang_v1
python3 scripts/step5_manual_review_cli.py --pool en_content_train_pool --reviewer-id manual_zhang_v1 --start-rank 200 --priority high

For the Step 5 v2 targeted queue, use the same CLI with `--queue-path` instead of `--pool`:

```bash
python3 scripts/step5_build_targeted_review_queue.py
python3 scripts/step5_manual_review_cli.py --summary --queue-path reports/step5_zh_target_strict_targeted_review_queue.step11_v1.csv
python3 scripts/step5_manual_review_cli.py --queue-path reports/step5_zh_target_strict_targeted_review_queue.step11_v1.csv --queue-label step5_v2_targeted_zh --reviewer-id manual_zhang_v2
python3 scripts/step5_manual_review_cli.py --summary --queue-path reports/step5_zh_target_strict_targeted_rereview_queue.step11_v1.csv
python3 scripts/step5_manual_review_cli.py --queue-path reports/step5_zh_target_strict_targeted_rereview_queue.step11_v1.csv --queue-label step5_v2_rereview_zh --reviewer-id manual_zhang_v2
```
```

Interactive commands:

- `p`: mark `positive`
- `n`: mark `negative`
- `u`: mark `uncertain`
- `s`: skip current row
- `c`: clear current review fields back to pending
- `m`: print full evidence block
- `q`: quit after saving current progress

Operational note:

- `zh_target_strict` uses the same fixed `train/valid/test` split container as English, but Step 8 zero-shot may only read its `test` split; `train` and `valid` stay reserved for Step 9 adaptation.

## Step 5 v3 Targeted Rereview, Cleanup, and Refreshed Freeze

2026-04-20 correction: the earlier BGE and calibrated-BGE Step 5 v3 cleanup artifacts in this section are historical only. The calibrated-BGE corrected cleanup was rolled back because its queue summary was stale relative to the current Step 11 graph. Use the calibrated-default artifacts listed above for the active corrected V3 queue.

Step 5 v3 is intentionally **not** a broad sample expansion. It is a targeted rereview pass built from a current Step 11 graph and must be regenerated whenever the Step 11 source graph changes.

The active v3 queue artifacts are:

- `schema/step5_v3_targeted_review_policy.json`
- `scripts/step5_build_targeted_review_queue_v3.py`
- `reports/step5_zh_target_strict_targeted_review_queue.step11_calibrated_default_v3.csv`
- `reports/step5_zh_target_strict_targeted_rereview_queue.step11_calibrated_default_v3.csv`
- `reports/step5_v3_targeted_review_queue_summary.calibrated_default.json`

The invalidated calibrated-BGE cleanup artifacts are archived, not active:

- `reports/deprecated_step5_v3_calibrated_bge_20260420/`
- `schema/step5_v3_targeted_cleanup_policy.calibrated_bge_20260420.json`

Current v3 queue state:

- net-new review rows: `0`
- rereview rows: `65`
- requested but not retained in the current primary graph: `0`
- cleanup-and-freeze status: complete
- cleanup policy: `schema/step5_v3_targeted_cleanup_policy.calibrated_default_20260420.json`
- cleanup summary: `reports/step5_v3_targeted_cleanup_summary.calibrated_default_20260420.json`
- applied labels: `2 positive / 35 negative / 28 uncertain`
- targeted rereview queue status: `65 / 65` reviewed

Current stricter v3 review rubric:

- mark `negative` only when all hold:
  - no shared contact / PGP
  - public-template clone or obvious topic-resale overlap
  - no seller-specific anchor
  - business logic favors independent sellers reusing the same copy
- mark `uncertain` when:
  - overlap is strong, but still insufficient to prove different controllers
- mark `positive` only for:
  - true anchor edges
  - or seller-specific evidence as strong as a shared-contact anchor

Direct anchor edges retained by the calibrated-default v3 cleanup:

- `/shop/449035 <-> /shop/461222`
- `/shop/452097 <-> /shop/452596`

Current operational note:

- the calibrated-default V3 rereview queue has been consumed into the active Step 5 cleanup and refreshed freeze
- this corrected freeze was later superseded by the 2026-04-21 Chinese boundary-expansion / positive-anchor refreeze, the English source-domain expansion top-up refreeze, and finally the 2026-04-22 English item-level identity expansion refreeze
- rerun Step 7 / Step 9 / Step 11 on Linux against the latest 2026-04-22 English item-level identity expansion refreeze, not the intermediate V3-only, Chinese-only, or English top-up freezes

## Historical v2 Review Round

In the current synchronized queue snapshot, the `zh_target_strict` `semantic_only` review queue is:

- total rows: `765`
- reviewed rows: `165`
- reviewed labels: `34 positive / 74 negative / 57 uncertain`
- pending rows: `600`
- pending `medium`: `428`, starting at `balanced_review_rank = 307`
- pending `low`: `172`, starting at `balanced_review_rank = 1792`

In the same synchronized snapshot, the `zh_target_strict` `semantic_structural` review queue is:

- total rows: `2290`
- reviewed rows: `80`
- reviewed labels: `21 positive / 49 negative / 10 uncertain`
- pending rows: `2210`
- pending `medium`: `838`, starting at `balanced_review_rank = 150`
- pending `low`: `1372`, starting at `balanced_review_rank = 2421`

The intermediate frozen Step 5 boundary after calibrated-default V3 cleanup crossed the project-side rerun gate for Step 7 / Step 9 before the later boundary expansion:

- `en_content_train_pool` reviewed rows: `145`
- `en_content_train_pool` supervision rows: `123`
- English source supervision split counts: `train = 85`, `valid = 19`, `test = 19`
- English source non-identifier positive share: `0.580247`
- `zh_target_strict` reviewed rows: `375`
- `zh_target_strict` supervision rows: `253`
- supervision split counts: `train = 169`, `valid = 46`, `test = 38`
- `semantic_only` positive supervision split counts: `train = 24`, `valid = 3`, `test = 5`
- `semantic_structural` positive supervision split counts: `train = 14`, `valid = 1`, `test = 6`
- `reports/step5_frozen_silver_summary.json` now reports:
  - `acceptance_checks.coverage_requirements_pass = true`
  - `acceptance_checks.non_identifier_positive_share_pass = true`
  - `zh_target_strict non_identifier_positive_share = 0.761364`

This corrected Step 5 boundary is materially larger than the previous `95 / 27 / 26` Chinese strict container. It has been adjudicated, frozen, propagated through Step 7 / Step 9 / Step 11 on Linux, and synchronized back.

That earlier English-source refresh was superseded by the 2026-04-21 English source-domain expansion described above:

- `en_content_train_pool` supervision is now `382`
- English source `train` is now `221`
- English source `test` is now `98`
- the active Step 7 source pool is no longer the earlier `51 / 12 / 11` or `85 / 19 / 19` micro-container that triggered the strongest source-capacity concerns

Historical note:

- the recommendations below describe the pre-v3 decision point
- the project moved past that point once, but the calibrated-BGE V3 cleanup was later invalidated and rolled back
- they are kept here only as audit history for the earlier v2 expansion round

## Step 5 Boundary Expansion for Few-Shot Retest

2026-04-21 update: the earlier `38`-row Chinese strict test split was too small for a confirmation-level few-shot claim. The project now has a boundary-expansion queue that targets hard, informative Chinese rows instead of random easy negatives.

Active boundary-expansion artifacts:

- policy: `schema/step5_boundary_expansion_policy.json`
- builder: `scripts/step5_build_boundary_expansion_queue.py`
- apply reviewed labels: `scripts/step5_apply_boundary_expansion_reviews.py`
- Codex review: `scripts/step5_codex_review_boundary_expansion.py`
- review queue: `reports/step5_zh_target_strict_boundary_expansion_queue.zero_shot_bge_20260421.csv`
- summary: `reports/step5_boundary_expansion_queue_summary.zero_shot_bge_20260421.json`
- Codex review summary: `reports/step5_boundary_expansion_codex_review_summary.zero_shot_bge_20260421.json`
- apply summary: `reports/step5_boundary_expansion_apply_summary.zero_shot_bge_20260421.json`

Selection logic:

- `positive_probe_high_semantic_high_structure`: high BGE-M3 similarity, high structural support, zero-shot BGE graph score above the `0.56` recall threshold, with graph-retained rows preferred
- `negative_probe_high_semantic_style_divergence`: high BGE-M3 similarity, strong profile/style divergence, no direct contact/PGP anchor, and zero-shot BGE score near the decision boundary

Completed review and freeze result:

- selected rows: `520`
- applied labels: `261 negative / 259 uncertain / 0 positive`
- intermediate Chinese strict reviewed rows: `895`
- intermediate Chinese strict supervision rows: `514`
- intermediate Chinese strict split counts: `train = 286`, `valid = 78`, `test = 150`
- intermediate split labels:
  - `train = 57 positive / 229 negative`
  - `valid = 14 positive / 64 negative`
  - `test = 17 positive / 133 negative`
- seller overlap across train/valid/test: `0`
- coverage requirements: pass

Interpretation:

- the high-semantic/high-structure positive-probe bucket did not contain defensible new positive supervision under the current conservative rubric
- this is negative evidence for the claim that simply expanding target-domain boundary rows will create high-quality positives
- the boundary is nevertheless now much larger for testing whether few-shot can exploit target-domain hard negatives and a larger holdout

Hard discipline:

- the queue contains pending rows only
- no selected row was promoted to positive without direct identity closure
- the active frozen benchmark was updated only after the reviewed queue was explicitly applied back to `reports/step5_zh_target_strict_balanced_review_queue.csv`
- after this refreeze, rerun Step 7 / Step 9 / Step 11 on Linux before treating downstream metrics as current

Rebuild commands:

```bash
python3 scripts/step5_build_boundary_expansion_queue.py
python3 scripts/step5_codex_review_boundary_expansion.py
python3 scripts/step5_apply_boundary_expansion_reviews.py --require-complete
python3 scripts/step5_freeze_silver_labels.py
```

The expansion goal is not to assume few-shot improvement. The goal is to create a defensible target-domain support/test boundary large enough to test that claim with ROC-AUC/AP first, then balanced accuracy as a threshold diagnostic.

## Step 5 Positive-Anchor Expansion

2026-04-21 update: after the hard-negative boundary expansion, the remaining bottleneck was not negative supervision but high-confidence Chinese positives. Positive-anchor expansion now mines only seller-specific identity evidence and refuses to pad the queue with generic semantic neighbors:

- supplemental low-frequency seller-facing Telegram/QQ handles extracted from profile text
- extended direct-identity mining over email, Telegram, Wickr, WeChat, Jabber, QQ, phone, crypto-wallet, and external URL evidence
- existing pending direct contact/PGP rows
- contradiction-free reviewed positive-component transitive closure

Artifacts:

- policy: `schema/step5_positive_anchor_expansion_policy.json`
- builder: `scripts/step5_build_positive_anchor_expansion_queue.py`
- Codex review: `scripts/step5_codex_review_positive_anchor_expansion.py`
- apply reviewed labels: `scripts/step5_apply_positive_anchor_reviews.py`
- review queue: `reports/step5_zh_target_strict_positive_anchor_expansion_queue.20260421.csv`
- queue summary: `reports/step5_positive_anchor_expansion_queue_summary.20260421.json`
- Codex review summary: `reports/step5_positive_anchor_expansion_codex_review_summary.20260421.json`
- apply summary: `reports/step5_positive_anchor_expansion_apply_summary.20260421.json`

Completed review and freeze result:

- first pass selected rows: `16`
- first pass reviewed labels: `13 positive / 3 uncertain`
- first pass net-new Step 4 candidate pairs appended: `10`
- direct-identity v2 selected rows: `105`
- direct-identity v2 reviewed labels: `104 uncertain / 1 positive`
- direct-identity v2 net-new Step 4 candidate pairs appended: `54`
- active Chinese strict reviewed rows: `1016`
- active Chinese strict supervision rows: `522`
- active Chinese strict label counts: `102 positive / 426 negative / 488 uncertain`
- active Chinese strict primary positive supervision rows: `96`
- Chinese strict split counts before the later label-stratified repair: `train = 290`, `valid = 80`, `test = 152`
- split labels:
  - `train = 61 positive / 229 negative`
  - `valid = 16 positive / 64 negative`
  - `test = 19 positive / 133 negative`
- seller overlap across train/valid/test: `0`
- coverage requirements: pass

Interpretation:

- the first pass improved the positive pool from `88` to `101`, but positive anchors remain scarce
- `6` current positives are closure-derived audit edges and are excluded from primary supervision/evaluation counts
- the direct-identity v2 pass did not add primary positive supervision; its URL/email candidates were overwhelmingly product-data or victim-data references, not seller identity anchors
- current positive split targets are not met: `zh_train positive = 61` versus target `100+`; `zh_valid positive = 16` versus target `25+`
- the Chinese test positive count is now `19`, still below the desired `50+` confirmation-level target
- because Step 4 gained net-new pairs, the Linux runtime must rerun Step 7 preview and semantic feature extraction before Step 7 training
- the correct conclusion is now evidence scarcity: with current raw profiles, target-domain direct positive identity anchors are too sparse to justify claiming that few-shot should necessarily improve AUC

Rebuild commands:

```bash
python3 scripts/step5_build_positive_anchor_expansion_queue.py
python3 scripts/step5_codex_review_positive_anchor_expansion.py
python3 scripts/step5_apply_positive_anchor_reviews.py --require-complete
python3 scripts/step5_freeze_silver_labels.py
```

Recommended Linux review entrypoint for the current round:

```bash
python3 scripts/step5_manual_review_cli.py --summary
python3 scripts/step5_manual_review_cli.py \
  --pool zh_target_strict \
  --reviewer-id <your_reviewer_id> \
  --stratum semantic_only \
  --priority medium \
  --start-rank 307 \
  --limit 40
```

Recommended follow-up batch after the current `semantic_only` pass:

```bash
python3 scripts/step5_manual_review_cli.py \
  --pool zh_target_strict \
  --reviewer-id <your_reviewer_id> \
  --stratum semantic_structural \
  --priority medium \
  --limit 40
```

Operational guidance:

- exhaust pending `medium` `semantic_only` rows before dropping into the `low` tail
- then review `semantic_structural` `medium` rows to add harder positives and negatives around the current decision boundary
- freeze after each batch of roughly `20-40` newly reviewed target rows
- do not rerun Step 7 / Step 9 on a newer boundary until the freeze summary shows that the new non-identifier positives actually survive component-based split assignment

Current review-round goal:

- keep adding high-confidence Chinese strict supervision only where the evidence is seller specific rather than generic resale overlap
- treat `semantic_only` and `semantic_structural` differently:
  - for `semantic_only`, only keep expanding with non-template, seller-specific pairs; do not force generic `私拍 / 闲人勿扰` style candidates into the positive pool just to pad counts
  - for `semantic_structural`, keep widening the hard-case evidence base so Step 9 and calibration are not judged on a nearly empty target-domain slice
- current active split coverage targets are no longer just `semantic_only`; they are also:
  - `semantic_structural valid > 2`
  - `semantic_structural test > 1`

Practical adjudication rubric for `semantic_only`:

- mark `positive` only when the pair shows a seller-specific portfolio or long-form copy pattern that is too distinctive to explain as a generic market template alone
- current positive references:
  - rank `238`: `market_item.xlsx|中文暗网交易市场|seller_raw:26383||market_item.xlsx|茶马古道|seller_raw:/shop/404544`
  - rank `520`: `market_item.xlsx|中文暗网交易市场|seller_raw:443620||market_item.xlsx|茶马古道|seller_raw:/shop/824`
- both positives are cross-market and reflect multi-topic, seller-level portfolio overlap rather than one generic commodity phrase

- mark `negative` when the overlap is generic, market-common, or easily explained by recycled template language
- current negative references include:
  - rank `20`: generic bank-customer data package theme
  - rank `35`: generic `私拍` wording only
  - rank `40`: another generic bank-data template
  - rank `55`: same generic bank-data family
  - rank `60`: generic Amazon gift-card theme
  - rank `91`: seller-vs-buyer Amazon gift-card theme overlap only
  - rank `97`: generic bitcoin private-shot wording

- mark `uncertain` when the overlap is strong enough to be suspicious, but still plausibly explained by reseller reuse, wanted-post reuse, or service-menu copying
- current uncertain references include:
  - rank `5`: violent-service menu overlap without identity closure
  - rank `10`: long-form overlap, but divergent titles/categories
  - rank `106`: rare ad-domain overlap, but no seller identity closure
  - rank `109`: same niche wanted-post, but still thematic only
  - rank `112`: wanted-post requirement overlap without identity closure
  - rank `114`: near-duplicate wanted-post wording with no retained identifier evidence
  - rank `115`: investigation-service menu overlap that still looks reusable
  - rank `118`: cross-market lookup-service menu overlap that may still be reseller copying

Stop condition before reopening the benchmark boundary:

- rerun `python3 scripts/step5_freeze_silver_labels.py`
- open `reports/step5_frozen_silver_summary.json`
- check `acceptance_checks.coverage_requirements_pass`
- check `acceptance_checks.non_identifier_positive_share_pass`
- for a newly expanded Chinese strict boundary, continue reviewing until the newly added `semantic_only` positives actually survive component-based split assignment into `train`, `valid`, and `test`
- keep `split_seller_overlap_counts = 0`
- keep `acceptance_checks.coverage_requirements_pass = true`
- keep `acceptance_checks.non_identifier_positive_share_pass = true`

Historical stopping note before the corrected calibrated-default V3 cleanup:

- the latest targeted review pass extended Step 5 in two directions:
  - an earlier explicit `semantic_only` corporate-account component to keep non-template pure-semantic positives alive
  - a new conservative `semantic_structural` cluster around student-parent and source-data buyer themes to reduce the extreme thinness of the structural target slice
- after this update, the most promising remaining `semantic_only` test-gain candidates are still concentrated in low-confidence `私人专拍 / 闲人勿扰` template clusters, so that stratum should now be expanded more selectively than before
- the earlier pre-v3 active boundary was `291 / 221 / 144 / 34 / 43`; it was later superseded by the corrected calibrated-default V3 cleanup described above

Benchmark-discipline note:

- if the current Chinese strict `test` boundary has already been inspected repeatedly, do not keep iterating on Step 7 / Step 9 conclusions against that same holdout
- after each credible Chinese review expansion, freeze a refreshed boundary and reserve a new untouched final holdout before making confirmation-level claims

The Step 5 freeze policy now treats missing `valid` or `test` coverage for Chinese strict `semantic_only` positives as a blocking error once at least `3` such positives are available. That is the project-side gate for deciding whether the refreshed boundary is ready for a new Step 7 / Step 9 rerun.

## 2026-04-22 English Validation Split Repair

A post-expansion audit found that the English validation split was not a valid early-stopping or threshold-selection set. It had `27` positives, all from `identifier_plus_text` or `identifier_primary`, and `53` negatives, all from `semantic_structural`. Several single features could therefore separate validation positives from negatives perfectly, which caused Step 7 LightGBM models to stop at `best_iteration = 1` with validation AUC/AP/BAcc `1.0`.

This was not seller leakage and not direct identifier leakage in the clean core feature set. The root cause was split assignment that balanced only row/label counts at seller-component level, without balancing `review_label x review_stratum`.

The freeze script and policy now enforce component-safe label-stratum balancing:

- code: `scripts/step5_freeze_silver_labels.py`
- policy: `schema/step5_freeze_policy.json`
- balance objective: row count, label count, and `review_label x review_stratum` count
- English blocking coverage now requires train/valid/test coverage for:
  - positive `identifier_plus_text`
  - positive `identifier_primary`
  - positive `text_clone_primary`
  - negative `identifier_primary`
  - negative `semantic_only`
  - negative `semantic_structural`

Current active English supervision after refreeze:

- `476` supervision rows
- train: `280 = 105 positive / 175 negative`
- valid: `77 = 30 positive / 47 negative`
- test: `119 = 44 positive / 75 negative`
- valid hard-coverage:
  - positive `identifier_plus_text = 8`
  - positive `identifier_primary = 12`
  - positive `text_clone_primary = 10`
  - negative `identifier_primary = 2`
  - negative `semantic_only = 12`
  - negative `semantic_structural = 33`
- seller and normalized-alias overlap across supervision splits remain `0`

The same refreeze also changed the active Chinese strict split:

- train: `335 = 61 positive / 274 negative`
- valid: `81 = 14 positive / 67 negative`
- test: `106 = 21 positive / 85 negative`

All Step 7 / Step 9 / Step 11 artifacts from before this label-stratified refreeze were stale. Step 7 was rerun on Linux for the 2026-04-22 boundary, but it is stale again after the 2026-04-23 English valid/test top-up below.

## 2026-04-23 English Valid/Test Top-Up

The 2026-04-22 English boundary still had only `77` validation rows, so Step 7 continued to require a small-validation guard explanation. The 2026-04-23 round expands English only and leaves Chinese strict unchanged.

Policy and artifacts:

- freeze policy: `schema/step5_freeze_policy.json`
- English split ratio changed from `60/15/25` to `55/20/25`
- direct-identifier policy: `schema/step5_en_item_identity_expansion_valid_test_topup_policy.json`
- direct-identifier queue: `reports/step5_en_item_identity_expansion_valid_test_topup_queue.20260423.csv`
- direct-identifier summary: `reports/step5_en_item_identity_expansion_valid_test_topup_queue_summary.20260423.json`
- direct-identifier review summary: `reports/step5_en_item_identity_expansion_valid_test_topup_codex_review_summary.20260423.json`
- direct-identifier apply summary: `reports/step5_en_item_identity_expansion_valid_test_topup_apply_summary.20260423.json`
- Step 4/hard-boundary top-up policy: `schema/step5_en_source_expansion_valid_test_topup_policy.json`
- Step 4/hard-boundary queue: `reports/step5_en_source_expansion_valid_test_topup_queue.20260423.csv`
- Step 4/hard-boundary summary: `reports/step5_en_source_expansion_valid_test_topup_queue_summary.20260423.json`
- Step 4/hard-boundary review summary: `reports/step5_en_source_expansion_valid_test_topup_codex_review_summary.20260423.json`
- Step 4/hard-boundary apply summary: `reports/step5_en_source_expansion_valid_test_topup_apply_summary.20260423.json`

Review outcome:

- direct-identifier selected rows: `46`
- direct-identifier labels: `30 positive / 16 negative`
- direct-identifier Step 4 appends: `38`
- Step 4/hard-boundary selected rows: `330`
- Step 4/hard-boundary labels: `212 negative / 118 uncertain / 0 positive`
- hard-boundary buckets: `10` identifier-primary leftovers, `100` text-clone probes, `220` hard-negative template probes

Active freeze after this round:

- English reviewed rows: `1321`
- English supervision rows: `734`
- English split counts: `train = 401`, `valid = 152`, `test = 181`
- English split labels:
  - train: `116 positive / 285 negative`
  - valid: `42 positive / 110 negative`
  - test: `51 positive / 130 negative`
- English validation hard-coverage:
  - positive `identifier_plus_text = 11`
  - positive `identifier_primary = 19`
  - positive `text_clone_primary = 12`
  - negative `identifier_primary = 2`
  - negative `semantic_only = 33`
  - negative `semantic_structural = 75`
- seller overlap across English supervision splits: `0`
- normalized-alias overlap across English supervision splits: `0`
- Chinese strict split remains `335 / 81 / 106`
- Step 5 acceptance checks pass with `0` coverage warnings and `0` coverage errors

Interpretation:

- the English source domain is now large enough that Step 7 should no longer need the `valid_row_count <= 100` small-validation explanation
- the expansion is not random padding; new negatives are hard template/topic controls, and new positives came from shared seller-facing direct identifiers
- English direct positives are now close to exhausted under the strict parser/rubric; further English source expansion would mostly add hard negatives or uncertain text clones
- all Step 7 / Step 9 / Step 11 artifacts must be regenerated from this active boundary before updating model conclusions

## Acceptance Criteria

1. No `same_alias_identity_continuity` row may enter primary supervision.
2. No soft same-alias continuity row may enter primary supervision.
3. Every frozen reviewed row must keep reviewer metadata.
4. Every supervision row must have a deterministic split assignment.
5. Seller overlap across supervision `train/valid/test` splits must stay at `0`.
6. If positive supervision rows exist, the non-identifier-supported positive share should be audited against the `0.3` minimum target.
