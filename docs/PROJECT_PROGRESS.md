# Project Progress

Updated: 2026-08-23

## Current Stage

`2026-07-20` Step28-v5, v6/v6.1 and v11 are withdrawn. V5 coupled English labels to synthetic labels; v6 reused v5 identities and collapsed to 14 states; v11 then used audit labels to remove 49 conflicting audit states, mixed model-discrimination gates with production-guard abstention, retained three all-zero features, and overstated cross-version state novelty. The current line is the post-audit corrected v12 synthetic replication plus the fully separate v12.1 existing-data application. V12 retains all 1,280 audit rows and gives each of 842 observable states total weight one; no audit label selects audit rows. Train/development/audit contain 1,453/658/842 states and all 33 retained features have rank 33. Full-history AUC/AP is 0.749634/0.767197 versus direct-history 0.693435/0.693269, AP gain +0.073928. The 199 block-permutation null is 0.498092 ± 0.054469 with plus-one p=0.005. Recipe checks now use one vote per unique state and explicitly separate five model-discrimination mechanisms from eleven production-guard abstention mechanisms; passing the latter is not called model classification success. This is only a corrected replication inside the fixed synthetic generator family, not universal unseen-state or real-world validation. V12.1 independently excludes all 1,259 reviewed pair UIDs and scores 2,689 existing unlabeled candidates; 101 corrections are nonzero and all are negative, so positive correction, internal queue and blind packet counts are zero. Dry full replay passes 37/37 checks, including complete model/199-permutation replay; current Windows CPU contract checks pass 22/22 before final manifest closure. Current report: `docs/STEP28_TRANSFERABLE_IDENTITY_HISTORY_V12_CORRECTED_REPLICATION_20260720.zh.md`.

`2026-07-19` Step27-v1 is frozen as an invalid engineering run rather than a scientific negative result. A source-contract audit found that v1 serialized seller fields with artificial section headers and re-encoded the canonical real Chinese profiles, whereas Step15-v7/Step24 used a values-only newline serialization and the already frozen identifier-redacted E5 cache. The resulting source feature was therefore not an exact replay of the frozen English Step24 scorer, so the observed Step27 failure could not distinguish an ineffective augmentation method from a shifted input representation. The isolated repair is Step27-v1.1 under `schema/step27_v1_1_exact_replay_policy.json`, `scripts/run_step27_v1_1_exact_replay_linux_20260719.sh`, and output root `reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719/`. The original v1 policy, runner and output namespace remain unchanged as historical audit evidence, but the current shared Step27/Step12 Python modules implement v1.1 contracts and therefore do not directly re-execute v1; numerical v1 reproduction requires its original Git commit.

Step27-v1.1 is deliberately a post-hoc engineering-integrity diagnostic, not a new confirmatory experiment. Canonical real seller embeddings are now selected byte-for-byte from the frozen Step15-v7 cache, reconstructed real clean text must reproduce the Step24 train-corpus hash, and real E5 pair cosine must reproduce the frozen Step24 feature within `5e-13`. The Step24 sync manifest, policy, clean-text manifest, pair-feature summary, Chinese pair CSV and source artifact are pinned by independent SHA-256 values, so a mutually drifted reference bundle cannot self-certify. Before parent generation, the Linux runner performs a hash-only preflight requiring the local E5 directory fingerprint to equal the fingerprint that produced the frozen real cache; mismatch requires restoration of that exact model snapshot and cannot be bypassed. The semantic policy, v7 redaction policy and both encoder producers are independently SHA-256 pinned, and feature manifests must match current shared-dependency hashes. A mandatory S0 control scores every row directly with the frozen English source artifact and fits no Chinese parameters; its train-universe ROC-AUC/AP must reproduce `0.7550015233065909 / 0.6443826343928266`. M2 must also meet the `-0.01 AP` component-bootstrap non-inferiority margin against S0 that was frozen before the v1.1 repair replay, in addition to beating M1 duplication and M0 real-only controls; this margin was not preregistered in v1. Synthetic UIDs are versioned from policy, transformed rows must show nonzero recomputed feature displacement, all configured variants must materialize under the no-op fail-closed contract, parent clean text must replay the v7 contract before transformation, and each synthetic cache now reports tokenizer truncation prevalence. Because the defect was diagnosed after prior development results were seen, v1.1 may score only seller-component-grouped train OOF. Even a passing technical gate cannot reopen the existing valid/test: the scoring CLI checks authorization and canonical gate path before split I/O, Step12 rejects non-OOF post-hoc modes before input-manifest construction, and the sync manifest rejects any such artifact. Current shared modules also refuse the invalid legacy v1 policy; historical numerical v1 replay requires its original Git commit. Step12 metrics/comparisons/summary are closed by a completion manifest. A passing replay can only authorize replication on a new frozen development batch. Windows checks currently pass: Python compilation and `47/47` Step27 contract tests. Full repair rationale: `docs/STEP27_V1_1_SOURCE_CONTRACT_REPAIR_20260719.zh.md`. Numerical encoding/training remains Linux-only.

Step27 的关键中文证据切片仍然欠充分：`valid` 只有 `4` 条 direct positive、`0` 条 component-anchor positive 和 `3` 条 public-noise negative；回顾性 internal test 相应为 `21/1/6`。这些 slice gates 只能作为 fail-closed 开发检查，不能作为具有充分统计功效的论文证据。若 Step27 通过内部晋级，Step20 必须在冻结配置后建设并一次性评估扩大上述切片的 Step27-specific prospective holdout。

`2026-07-18` work moved to isolated branch `method/step27-english-pretrained-synthetic-adaptation`. Step27 is a preregistered, small-scope test of whether parent-preserving Chinese semi-synthetic views add information beyond equal-effective-weight duplication while the English-trained Step24 E5 LR/L2 source scorer remains frozen. The frozen source artifact is `artifacts.source_only.e5_lr_l2_control`, trained only on English `401 = 116 positive / 285 negative`; its only feature is `identifier_redacted_e5_cosine`, and its source-logit coefficient is fixed to `1` in the primary residual models. The Chinese boundary is unchanged at `train 573 = 229/344`, `valid 120 = 30/90`, and retrospective internal test `200 = 50/150`. Primary generation uses exactly `16` non-silver positive parent pairs across `13` recomputed seller components and `16` score-blind matched reviewed-negative parents, with each matched set constrained to one fixed OOF fold. Distinct negative components are preferred, but the giant fold-0 component leaves no distinct reviewed-negative capacity under absolute positive-component exclusion, so an existing same-component reviewed-negative fallback is allowed and recorded as `matched_component_relation`; this is not split leakage because component-grouped OOF holds the entire component out together, but fallback sets do not add independent component ESS and their per-track count must be reported. Each parent receives two non-destructive views and the primary cap remains `64` synthetic rows per seed. A separate `56`-positive silver direct/contact sensitivity is physically isolated and cannot satisfy the primary gate. M0 is real-only residual adaptation, M1 is the same-parent/equal-weight duplication control, and M2 is transformed-view adaptation; the primary comparison is M2 versus M1, with M2 versus M0 required secondarily. Two M2-matched diagnostics are implemented but cannot promote the method: learned source alpha uses the descriptive near-zero rule `abs(unstandardized alpha) <= 0.1`, while alpha zero uses the descriptive same-split AP equivalence margin `abs(delta) <= 0.01`; neither is a formal equivalence test or promotion gate. Four-fold seller-component OOF, ten stability seeds, per-parent child-weight cap `0.5`, total synthetic-weight cap `0.25`, fold-train combined-weight normalization back to the current real-only effective weight, full identifier-redacted E5/lexical feature recomputation, recipe-label and synthetic-real distinguishability audits, separate bootstrap/permutation base seeds `20260718/20260719`, and immutable hash-bound outputs are specified. The real pair feature builder may precompute all three identifier-redacted split features and emits `real_pair_features.csv` as a combined build/integrity audit plus physical `real_pair_features.train.csv`, `.valid.csv` and `.test.csv` files. The runtime is fail-closed and sequential: the initial training process loads only the train labels/features and produces only `train_oof`; a passing `oof_gate` permits a separate process to open valid labels/features and generate frozen-threshold valid scores; a passing `valid_gate` permits another separate process to open test labels/features and generate the retrospective internal-test diagnostic. Thus valid/test raw text and features are not described as physically unopened, but their labels/features/scores are not loaded by a scoring process before the corresponding gate. Neither current valid nor test can fit, select or promote a model. Step20 currently authorizes only preparation of a new Step27-specific prospective policy/model/threshold/manifest freeze; no Step27 prospective evaluation is yet authorized, and the existing Step15-v7 Step20 freeze cannot be reused for that claim. Policy: `schema/step27_english_pretrained_synthetic_adaptation_policy.json`; protocol: `docs/STEP27_ENGLISH_PRETRAINED_SYNTHETIC_ADAPTATION_PLAN_20260718.zh.md`; Linux runner: `scripts/run_step27_english_pretrained_synthetic_linux_20260718.sh`. Windows validation is limited to compilation, contract/data-lineage checks and config-only entry points; no Step27 numerical experiment has been run locally.

`2026-07-18` Step26-v1 completed on Linux and synchronized as a complete generated-output bundle. All `13/13` manifest-bound generated outputs match size/SHA-256; `2,880` prediction rows cover the exact `120 = 30/90` corrected representative-valid and `200 = 50/150` internal-test pairs for nine models, with zero pair/seller/component overlap between splits. All 18 ROC-AUC/AP/PR-AUC rows and the 5,000-resample component-grouped bootstrap independently reproduce. The bridge hypothesis failed: frozen English source-only semantic+style reached valid `AP=0.508495` versus Step15-v8 clean `0.574855`, delta `-0.066360`, bootstrap 95% CI `[-0.322642, 0.149181]`; only `3/6` gates passed and `eligible_for_one_step26b_experiment=false`. Internal test was diagnostic only (`0.517033` versus clean `0.544139` and contextual `0.620525`) and did not affect promotion. Raw mStyleDistance had high valid ROC-AUC `0.782222` but low AP `0.459583`; template/public negatives remained near-saturated in style space, confirming style/template confounding. Step26A is frozen as a negative bridge result, Step26B is blocked, and no Step11/17 promotion is allowed. Audit: `docs/STEP26_FROZEN_AUTHORSHIP_BRIDGE_RESULT_AUDIT_20260718.zh.md`.

`2026-07-18` work moved to isolated branch `method/step26-frozen-authorship-bridge`. A full re-audit found that the strongest Step24 number (`source-only semantic+style AP=0.802718`) was measured only on Chinese canonical train D0, while corrected Step15-v8 was measured on a separate `120`-row representative-valid and `200`-row internal-test boundary. The values were therefore not directly comparable. Step26A now fills that missing experiment without fitting a new model: it blindly replays v7 identifier redaction, encodes exactly the corrected valid/test sellers with the two frozen Step24 authorship encoders, applies the immutable English-only LR/L2 standardization and coefficients, and only then joins Chinese labels and the exact frozen v8 B0/clean/contextual pair scores. The primary gate uses representative-valid AP, component-grouped paired bootstrap and scale-invariant template/public-noise top-budget intrusion; internal test is diagnostic only and Step20 remains mandatory. Eight contract tests, Python compilation and all three config-only entry points pass on Windows; no numerical encoding or real evaluation was run locally. Protocol: `docs/STEP26_FROZEN_AUTHORSHIP_BRIDGE_PLAN_20260718.zh.md`; Linux entry point: `scripts/run_step26_frozen_authorship_bridge_linux_20260718.sh`.

`2026-07-18` Step25-v3.1 completed on Linux and synchronized as a valid closed solver-repair bundle. The manifest binds `9` payloads (`1,527,262` bytes) and `26` producers; all local SHA-256, aggregate hashes and the manifest hash reproduce. All `44/44` constrained C0-C3 fits terminate only at KKT tolerance, with maximum projected-gradient residual `2.10e-9`, and both pair-feature CSVs are byte-identical to old v3. Independent CSV replay reproduces all twelve ROC-AUC/AP rows, with no duplicate pair, component-fold conflict or direction violation. The correctly converged result confirms the method failure: C2-minus-C0 AP is `-0.058346` on English grouped OOF, `-0.030238` on source-only Chinese and `-0.028093` on Chinese target grouped OOF; target bootstrap CI is `[-0.092698, 0.020856]`. Template-clone mean-rank/top-decile/violation deltas worsen by `+0.026923/+0.036364/+0.048097`; only `2/11` gates pass. C2 is now frozen as a strict negative result, with no D1, publication or Step11/17 promotion. Detailed audit: `docs/STEP25_V3_1_RESULT_AUDIT_20260718.zh.md`.

`2026-07-18` the post-Step25 plan freezes further D0 model search. A read-only Step26 paper-evidence audit comes first, followed by a new score-blind, seller-component-disjoint D1 only if at least `30` direct/component positives and the preregistered hard-negative slices can be independently reviewed. If D1 readiness cannot be reached, the project pivots to an evidence-type concept-drift dataset/negative-results paper rather than weakening positive labels or tuning another model on D0. Plan: `docs/NEXT_RESEARCH_PLAN_AFTER_STEP25_20260718.zh.md`.

`2026-07-18` the returned Step25-v3 bundle passed synchronization and score-replay checks but was invalidated as a final scientific result by a solver-termination defect. All `9/9` payloads (`1,499,625` bytes) and `21/21` producer hashes matched, all twelve ROC-AUC/AP rows reproduced exactly from the `401` English and `573` Chinese predictions, and there was no duplicate pair, component-fold conflict, valid/test access or supervision leakage. The produced C2 scores failed the method gates: source-only/target-OOF/English-OOF AP deltas versus C0 were `-0.030203/-0.027580/-0.058026`; target bootstrap CI was `[-0.092019, 0.021403]`; template-negative mean-rank/top-decile/violation deltas were `+0.026907/+0.036364/+0.047780`; only `2/11` gates passed. However, artifacts marked every fit converged even though final projected-gradient residuals reached `0.52` against the frozen `1e-8` tolerance. Root cause: the v3 solver accepted relative-loss stagnation as an alternative convergence condition. After v3.1 reproduced byte-identical features and established the correctly converged result, the invalid `v3_copy_aware_dual_channel_20260718` output directory was deleted; only the historical diagnostic record remains.

`2026-07-18` repair work moved to isolated branch `fix/step25-v3-1-solver-convergence`. Step25-v3.1 changes only numerical solution and termination for the identical convex constrained LR/L2 objective. Feature sets, C0-C3 matrix, directions, `L2=10`, factorized weights, component folds, bootstrap, gates, missingness closure and operational control remain byte-level or value-level frozen. Relative loss is diagnostic only. A low-dimensional active-set projected Newton direction with Armijo backtracking is used because plain projected gradient stalled at `2.3e-6` after 10,000 iterations on a collinear regression test. `solver_converged=true` requires final KKT/projected-gradient residual `<=1e-8`; maximum-iteration or line-search stagnation fails closed. Artifacts persist termination reason, objective, gradient norm, KKT residual, relative loss and accepted step; the sync manifest independently requires all `44` repaired C0-C3 fits to satisfy KKT and requires both English/Chinese pair-feature CSVs to be byte-identical to v3. Eleven repair tests, Python compilation, four config-only preflights and Git-Bash runner syntax passed before Linux execution. Linux execution is now complete under `reports/step25_template_decontaminated_authorship/v3_1_solverfix_20260718/`, with the frozen negative result recorded above; detailed contract: `docs/STEP25_V3_1_SOLVER_CONVERGENCE_REPAIR_20260718.zh.md`.

`2026-07-18` the original Step25-v3 scientific design remains a direct, preregistered continuation of the Step25 copy-decontamination line, not a replacement for frozen v1/v2. It retains raw authorship style and adds pair-local-clean style, raw-minus-clean residuals and label-free copy-risk statistics as separate low-dimensional channels. The fixed primary C2 constrains raw/clean similarities nonnegative and copy residual/risk nonpositive; C0/C1/C3 remain fixed controls and no candidate search is permitted. D0 reads canonical train only, never valid/test, and can at most nominate a future D1 replication; publication promotion and Step11/17 entry remain hard false.

`2026-07-17` Step25-v2 completed on Linux and synchronized as a closed 19-file (`44,963,636` bytes) bundle with 17 producer hashes. Pair-local detection found masked spans in `299/573` Chinese train pairs and retained reliable local-clean style for `454/573`; the template audit found local copied content in `109/110` template-negative rows. This established that the v1 global catalog missed pair-local copying, but uniform replacement with pair-local-clean style failed as a model. Chinese target grouped-OOF AP was `0.704847` for matched raw P0, `0.670692` for pair-local-clean P2 and `0.737365` for raw-fallback P3; English P0/P2 AP was `0.468210/0.251926`. P2 reduced the template-negative-versus-strong-positive violation rate by `0.092812`, but raised template mean rank percentile by `0.001637` and top-decile exposure by `0.036364`. Direct/component sensitivity improved by `0.119421 AP`, yet `85/86` rows in that positive slice were silver, while the soft-positive slice regressed. Only `3/8` mechanism gates passed, both grouped-bootstrap intervals crossed zero, and `d1_candidate_eligible=false`. Step25-v2 is frozen as a mixed/negative mechanism result and cannot enter Step11/17. It motivates v3's copy-risk auxiliary channel but does not justify reversing either parent conclusion.

`2026-07-17` Step25-v1 completed on Linux and was synchronized as a complete SHA-256-bound result bundle. The decontamination hypothesis did not survive its preregistered D0 gates. Source-only raw-style versus decontaminated-style AP was `0.801847` versus `0.799675`; Chinese target grouped-OOF raw-style versus decontaminated-style AP was `0.789848` versus `0.784333`. Only `5/17` continuation conditions passed, both grouped-bootstrap lower-bound conditions failed, target non-silver/direct-component sensitivity regressed, and the required template/public-noise rank reductions were not obtained. The occurrence-level reliability expert changed only a very small number of actionable rows and did not establish public-noise correction coverage. Therefore `d1_candidate_eligible=false` and `publication_promotion_eligible=false`. Step25 is frozen as a negative method result and is not allowed into Step11/17; its formal bundle remains under `reports/step25_template_decontaminated_authorship/v1_20260717/`.

`2026-07-17` the workspace was cleaned using an explicit path allow-list. `308` obsolete files (`178,167,287` bytes, `169.91 MiB`) were removed: Python caches, one Step9 context-mismatch rollback copy, three completed/failed console logs, the superseded Step15-v8 V2 result bundle, the invalidated Step16-v8 V2 readiness freeze, the defective Step23-v2 result, one off-mainline TUApps probe summary, and sixteen unreferenced Step11 threshold CSVs. Current canonical Step3-Step5 data, models, formal Step7/9/11 evidence, frozen negative-result bundles, manifests, Step24 and Step25 were retained. The V2-specific Git LFS rule and stale V8 documentation references were removed. Full audit: `docs/WORKSPACE_CLEANUP_AUDIT_20260717.md`.

`2026-07-17` Step24-v1 completed on Linux and was independently synchronized and replay-audited. The hash-bound bundle contains `16` payload files (`9,834,576` bytes); local verification found zero missing, size-mismatched or SHA-256-mismatched payloads, and all nine summary metric rows were reproduced from the `573`-row prediction file within `1e-10`. The clean contract held: only identifier-redacted canonical train text was encoded, no candidate-rule or identifier feature entered the scorer, both external encoders remained frozen and no valid/test seller or pair was scored. Step24 found a real but impure cross-language style signal. The English-label-only semantic-plus-style scorer reached Chinese D0 `ROC-AUC=0.871890`, `AP=0.802718`, improving over the matched redacted-E5 LR/L2 by `+0.158336 AP`; its component bootstrap 95% CI was `[0.010726, 0.287063]`. Target grouped-OOF primary AP was `0.798461`, `+0.201337` over the target E5 control, but its 95% CI narrowly crossed zero at `[-0.001997, 0.346366]`. More importantly, the style representation raised template-clone negative mean/q95/top-decile scores by `0.028287/0.067389/0.064517`; public-contact/URL negatives had mean target score `0.437602`. The primary added almost nothing over style-only (`+0.000871` source AP; `+0.008613` target OOF AP), and E5 reduced direct/component slice AP. `promotion_eligible=false`. Step24 is frozen and cannot enter Step11/17 or be retuned on D0.

`2026-07-17` work moved to isolated branch `method/step25-template-decontaminated-authorship`. Step25 tests whether the Step24 failure is caused by copied boilerplate being embedded as author style. It exactly replays the Step24/v7 identifier-redacted corpus, then learns length-12 character-shingle support from text and seller-component membership only. For every seller, all sellers in its complete component are excluded from catalog support; a span is removed only when overlapping shingles cover at least 24 characters and are supported by at least three sellers in at least two external components. Catalog artifacts persist SHA-256 hashes and document frequencies, never raw shingle text. Sellers retaining fewer than 32 content characters are marked unreliable and contribute no positive decontaminated style cosine. The preregistered primary is decontaminated style-only LR/L2 versus a matched raw style-only LR/L2; E5-plus-clean-style is secondary and raw/clean/delta/coverage fusion is exploratory only. Probability tails remain diagnostic, while formal template gates use global rank percentiles, top-decile exposure and template-negative-versus-strong-positive violation rates so intercept shifts cannot manufacture improvement.

Step25 D0 is explicitly hypothesis-informed retrospective development because Step24 errors on the same canonical train boundary motivated the method. D0 can only set `d1_candidate_eligible`; `publication_promotion_eligible` is hard-frozen to `false`. Any method selection requires a future score-blind, D0-component-disjoint D1 with at least 30 direct/component positives, 30 template negatives and 20 public-contact/URL negatives. Final confirmation requires an F1 collected after model freeze, component-disjoint from D0/D1 and evaluated once. Occurrence identifiers remain outside the clean scorer: a separate English-trained direction-constrained offset expert uses English component-OOF clean probabilities, may only uplift bilateral direct evidence, may only downgrade risky/support/high-frequency public evidence, and leaves mixed/ambiguous/no-identifier pairs unchanged. Implementation is in `schema/step25_template_decontaminated_authorship_policy.json`, seven `scripts/step25_*.py` modules, `scripts/run_step25_template_decontaminated_authorship_linux_20260717.sh`, thirteen pure contract tests and `docs/STEP25_TEMPLATE_DECONTAMINATED_AUTHORSHIP_PLAN_20260717.zh.md`. Windows compilation, all thirteen tests and all config-only entry points pass without checking the locally deleted model directories; numerical embedding/training remains Linux-only and reuses the original `models/step24/authorship/...` paths. A domain with zero qualifying cross-component templates now produces a valid header-only catalog, so a scientifically meaningful zero-discovery result does not become an execution failure. The synchronization record binds all direct Step9/Step15-v7/Step24 code and policy dependencies, the component assignment input, both clean-prediction inputs and both Step3 occurrence files, rather than hashing only the new wrapper scripts.

`2026-07-17` the project initially moved to isolated branch `method/step24-content-independent-authorship` after Step21-Step23 established that local augmentation and additional item-distribution views add no independent controller evidence. Step24 was the active method experiment at that stage. It did not generate seller, pair, item, text or identity labels. It replayed the exact identifier-redacted Step15-v7 clean corpus and used two frozen external multilingual style/authorship encoders: the EMNLP 2025 `Blablablab/multilingual-style-representation` model pinned at commit `b0147bbf450424fe72c8525fcc02e2e39e3a4024`, and the Findings ACL 2025 MIT-licensed `StyleDistance/mstyledistance` model pinned at `d66ed25e48225a503b21a65bc804caf06c886f96`. The preregistered pair view had exactly three cosines: identifier-redacted E5 semantics, multilingual authorship style and mStyleDistance. Fixed controls were raw cosines, E5-only LR/L2, style-only LR/L2 and the three-feature semantic-plus-style LR/L2 primary; there was no candidate search, class balancing, local encoder fine-tuning or valid/test selection.

Step24-v1 read canonical train only: English source `401 = 116 positive / 285 negative` and Chinese target `573 = 229 positive / 344 negative`. The Chinese all-label OOF result was explicitly secondary internal-development evidence because `213/229` positives are train-only silver; non-silver and direct/component sensitivity slices were mandatory. Source-only transfer trained on English and scored Chinese train. Target adaptation used five seller-component grouped folds, always retaining all English source rows and excluding the complete held-out Chinese components. Promotion required at least `+0.03` target OOF AP and `+0.02` source-only AP, separate component-bootstrap lower bounds not below zero, no material source-only or target-OOF non-silver/direct-component degradation and no increase above `0.02` in template/topic negative score tails. The completed result and failed promotion decision are recorded above.

Step24 implementation remains frozen in `schema/step24_content_independent_authorship_policy.json`, `scripts/step24_*.py`, `scripts/run_step24_content_independent_authorship_linux_20260717.sh`, `tests/test_step24_content_independent_authorship_contracts.py` and `docs/STEP24_CONTENT_INDEPENDENT_AUTHORSHIP_PLAN_20260717.zh.md`. The Windows downloader pins both upstream commits and writes deterministic provenance; Linux rejects missing/mismatched provenance. Immutable outputs are isolated under `reports/step24_content_independent_authorship/v1_20260717/` and were returned as a complete SHA-256 manifest-bound directory. No Step24 numerical computation ran on Windows; the completed Linux result is the frozen result described above.

`2026-07-17` corrected Step23-v2.1 completed on Linux and is frozen as a negative representation result. Its synchronization manifest binds `11` payload files (`37,877,331` bytes); local replay found zero missing, size-mismatched or SHA-256-mismatched files. The redaction correction worked: only `1/65,514` raw item triggered a true cross-field identifier match, no pair feature remained constant, `6,410` selected items were real train items, and no valid/test item or synthetic item was encoded. Nevertheless, the preregistered item-distribution primary failed promotion. On `573` Chinese-train rows and `222` seller components, target grouped-OOF AP was `0.598696` for mean-pool cosine, `0.593218` for the matched aggregate and `0.458483` for aggregate-plus-distribution; the primary-minus-matched delta was `-0.134735` with grouped-bootstrap 95% CI `[-0.193967, 0.001614]`. The primary also regressed on canonical non-silver AP (`0.177815 -> 0.128322`) and direct/component-plus-negatives AP (`0.224848 -> 0.217211`), while increasing template/topic negative q95 by as much as `0.199726` and top-decile mean by `0.244383`. `promotion_eligible=false`. This shows that item-to-item distribution statistics amplify content/topic similarity rather than controller identity under the current supervision. Step23 must not enter Step11/17 or Step20 as a promoted method. The identifier-redacted real-item mean-pool score remains a diagnostic baseline; no further synthetic positive generation or post-hoc Step23 feature selection is authorized on this development boundary.

`2026-07-17` the first Linux Step23-v2 run synchronized completely (`11` manifest-bound payloads, `37,068,947` bytes; all local size/SHA-256 checks passed), but its numerical result is invalidated by an implementation defect discovered during result audit. All `65,514/65,514` raw train items were marked `cross_field_redaction_applied=true`, so every selected item lost its title/description exact-overlap hashes. Root cause: the second-pass wrapper collapsed field-separating newlines before comparing text, and ordinary whitespace normalization was mistaken for an identifier match. The invalid run's preregistered primary also failed strongly (`OOF AP 0.394450` versus matched aggregate `0.582612`; delta `-0.188162`), but it cannot be used as the final representation conclusion until the feature defect is corrected. Step23-v2.1 fixes only trigger detection by requiring an actual regex/literal match, adds a regression contract, and writes to `reports/step23_item_multi_instance/v2_1_20260717/`; model sets, grouped folds, weights, primary model and gates remain frozen. No synthetic data are introduced. Linux entry point: `scripts/run_step23_item_multi_instance_v2_1_linux_20260717.sh`.

`2026-07-17` a second code/scientific audit superseded the unexecuted Step23-v1 draft and created `method/step23-v2-matched-controls`. V1 had two attribution defects: cross-field identifier redaction blanked title/description hashes that were also used for seller-local deduplication, collapsing distinct items in the same category; and its aggregate baseline came from the older seller-profile corpus rather than the exact Step23 item set. V2 separates final-redacted-text deduplication from exact-overlap eligibility and compares a preregistered `aggregate_plus_distribution_primary` only against a `same_item_aggregate_control` built from the identical selected items. Fixed diagnostics include mean-pool-only, structure-only and semantic-distribution-without-count controls; every model has an English source-only counterpart. There is no candidate selection. Promotion is internal-development-only and requires `>=0.02 AP`, grouped-bootstrap lower bound `>=0`, no AP regression on non-silver or direct/component sensitivity slices, no material direct/component positive score loss, and bounded mean/q95/top-decile score increases on template/topic negatives. The primary all-train LR/L2 artifact, imputation, standardization, feature order and every OOF fold artifact are persisted; a label-blind frozen-feature scorer is provided for later representative-valid/Step20 use. Current Chinese train still contains `213/229` silver positives, so all-label OOF remains secondary development evidence and cannot replace a prospective holdout. Numerical execution remains Linux-only; current protocol: `docs/STEP23_ITEM_LEVEL_MULTI_INSTANCE_V2_20260717.zh.md`.

`2026-07-16` Step23-v1 item-level multi-instance verification was implemented on `method/step23-item-level-multi-instance` but was not numerically executed. It is retained only in Git history as the superseded implementation draft described above; its old runner and v1 output contract are not active on the v2 branch.

`2026-07-16` Step22 completed on Linux and is frozen as a strict negative augmentation result. It generated `617` same-observed-seller item-disjoint positive views and `54` reviewed-negative item views from `6,689` source-item lineage rows, producing `1,342` identifier-redacted pseudo profiles. All `671` synthetic pairs are train-only, benchmark-ineligible and core-transfer-ineligible; direct replay found zero current valid/test seller-UID overlap and zero positive-view cross-side content-signature overlap. The complete synchronization manifest binds ten payload files (`17,689,109` bytes) with zero missing, size-mismatched or SHA-256-mismatched files. On `573` real Chinese-train rows and `222` recomputed seller components, five-fold grouped OOF AP is `0.538467` without augmentation, `0.541164` for positive-budget equal-weight duplication, `0.533950` for same-seller split positives, `0.545081` for full-budget equal-weight duplication and `0.527419` for same-seller positives plus reviewed-negative views. The representation method therefore loses `0.007214 AP` against its positive-budget duplication control and the full method loses `0.017662 AP` against its full-budget duplication control; `promotion_eligible=false`. The augmentation raises real negative scores at least as much as positive scores, showing that same-account inventory splitting reinforces topic/template similarity rather than cross-account identity invariance. Step22 must not enter Step11/17 or Step20 and may be reported only as a negative ablation; it adds zero real cross-account positive identities. Work now moves to Step23 item-level multi-instance distributional verification instead of further synthetic-budget tuning.

`2026-07-16` Step21 corrected v2 synchronized completely but produced a strict negative augmentation result, so work moved to isolated branch `method/step22-same-seller-split-augmentation`. Step22 no longer edits an already aggregated positive pair. It constructs identifier-redacted, item-disjoint pseudo-alias views from one real Chinese seller account, and constructs hard-negative views only from existing reviewed canonical-train negatives. A read-only capacity audit found `5,097` Chinese seller profiles and `17,556` items; after excluding current valid/test seller UIDs, `4,658` profiles remain, including approximately `628` sellers with at least six items and at least three distinct titles/descriptions. Final eligibility is recomputed on Linux after portable-alias exclusion, identifier redaction and exact-title/description connected-component closure. Step22 compares no augmentation, class-matched equal-effective-weight duplication, same-seller positive views and positive-plus-reviewed-negative views using the same identifier-redacted E5 cosine plus fixed 64d symmetric pair projection and LR/L2. Selection is five-fold grouped OOF on canonical Chinese train only; valid/test remain untouched. Synthetic rows are always train-only, benchmark-ineligible and not real cross-account sockpuppet labels. Implementation files are `schema/step22_same_seller_split_policy.json`, four `scripts/step22_*` modules, `scripts/run_step22_same_seller_split_linux_20260716.sh`, `tests/test_step22_same_seller_split_contracts.py` and `docs/STEP22_SAME_SELLER_SPLIT_AUGMENTATION_20260716.zh.md`. Windows static compilation and six pure-function contract tests pass; no Step22 data generation, model encoding or numerical training has been run on Windows.

`2026-07-16` work started on isolated branch `method/step21-synthetic-train-only-augmentation` after external proof-positive collection remained blocked. Step21 does not fabricate new benchmark truth. It adds deterministic, identifier-redacted Chinese positive-pair text augmentation under the active v2 root `reports/step21_synthetic_train_only/v2_balanced_grouped_oof_20260716/`, with parent-pair/component lineage, fixed per-parent effective-weight budgets and a mandatory equal-effective-weight duplication control. It uses canonical Chinese train membership and the Step16I-v2 recomputed leak-free seller components rather than the superseded v7 representative-validation overlay. The primary track has `16` non-silver parent pairs (`1` direct-identifier and `15` style/structural soft); the physically separate silver direct/component sensitivity track has `85` parent pairs. Generated counts are `48` and `170`, but both tracks record `new_real_positive_count=0` and `new_independent_identity_count=0`. Method comparison uses five-fold seller-component grouped OOF on train only and never uses current valid/test for selection. The first synchronized v1 OOF is invalidated because its old greedy component splitter produced fold sizes `326/46/54/74/73`, including a single-class positive fold. The corrected v2 jointly balances total, positive and negative counts; the unavoidable 175-row component limits perfect balance, but observed fold totals are `194/95/95/94/95`, with positive/negative counts `30/164`, `50/45`, `50/45`, `49/45` and `50/45`. The synchronized v2 bundle is complete: its manifest binds `21` payload files (`9,139,042` bytes), and an independent replay found zero missing, size-mismatched or SHA-256-mismatched files. V2 is a statistically indistinguishable null augmentation result, not evidence of significant harm. On the `573` Chinese-train-row grouped OOF boundary, the primary no-augmentation/equal-weight-duplication/text-augmentation AP values are `0.654591/0.653481/0.652302`; text augmentation changes AP by `-0.002289` versus no augmentation and `-0.001179` versus duplication. A post-hoc, read-only component-grouped paired bootstrap over the frozen predictions gives 95% CIs `[-0.009853, 0.006840]` and `[-0.006903, 0.006381]`, respectively, so random variation is a plausible explanation. The silver-anchor sensitivity point delta versus duplication is `-0.006192` with post-hoc 95% CI `[-0.017002, 0.007658]`. These diagnostics were not preregistered and do not promote the method; they correct the interpretation to “no measurable representation gain.” Step21 must not enter the main Step7/9/15 or Step11 pipeline without new independent validation, but it may be reported as a controlled train-only null ablation showing that derived row count did not measurably outperform equal-weight duplication. V2 is path-isolated and does not overwrite v1. Linux entry point: `scripts/run_step21_synthetic_train_only_linux_20260716.sh`; detailed protocol and numerical audit: `docs/STEP21_SYNTHETIC_TRAIN_ONLY_AUGMENTATION_20260716.zh.md`.

`2026-07-16` the corrected V8 readiness-v3 rerun completed and was frozen as a strict negative result. Work moved to branch `data/step16i-integrity-dev2` for data-integrity and retrospective-development preparation:

- the latest V8 bundle `reports/step15_v8/bridge_v8_readiness_v3_reprofix_20260716_112833_31791/` contains 226 manifest-bound files; local replay found zero missing, size-mismatched or SHA-256-mismatched bundle files. The separate upstream readiness root still needs to be synchronized from Linux for a complete local reproduction archive;
- corrected V8 used `974` primary train rows, canonical `120 = 30 positive / 90 negative` representative-valid rows and the immutable `200 = 50 positive / 150 negative` internal-development test. Train-OOF selected B3 plus the linear pairwise ranker, but representative-valid clean AP was `0.574855` versus B0 AP `0.600546`. Contextual fusion raised internal diagnostic AP from `0.544139` to `0.620525` and improved the combined direct/component recall, but public-noise FPR remained `0.85 -> 0.85`;
- Step12-v8 therefore correctly reports `validation_data_readiness_met=true`, `method_gates_met=false` and `promotion_eligible=false`. Clean-versus-B0 grouped-bootstrap mean delta AP is `-0.027194` with 95% CI `[-0.123717, 0.055103]`; fusion-versus-clean is `+0.025268` with CI `[-0.003016, 0.095010]`. V8 does not enter Step11/17 publication validation and must not be retuned against the consumed valid/internal-test boundary;
- a five-agent read-only audit found that the raw Step5 Chinese train file still exposes `434` historical `split_component_id` values, while the current train seller graph has about `222` connected components. This is a real historical-field warning caused by train-only silver expansion. It does not invalidate the corrected V8 run: the V8 readiness materializer recomputed the full seller graph, persisted new `v7_component_id` assignments and the bridge grouped OOF used those recomputed IDs;
- the same audit confirmed that `213/229` Chinese train positives are train-only silver; the local snapshot has no remaining seller/component-independent proof-level direct-positive candidate and no independent prospective public-noise candidate. Existing old candidates can only support a score-blind retrospective development extension, not a prospective final holdout;
- evaluation-label quality remains a primary limitation. Step16F places only `22/80` valid/test positives in direct/component primary evidence, `14` in soft-primary slices and `44` in secondary/sensitivity-only use. Historical Step5 review notes also contain Step11/graph/cluster-assisted reasoning, so the current boundary cannot be described as fully prediction-blind paper gold;
- Step16I now adds a read-only integrity audit and a label-free retrospective `zh_dev2` queue preparer. They never overwrite Step5, never convert candidates to labels and explicitly set `prospective_claim_allowed=false`. The integrity tool recomputes seller components, separately audits normalized seller-alias overlap across splits/language pools, emits a typed permanent exclusion manifest, checks V8 readiness partition safety and reports graph-assisted review traces. A persisted component may conservatively merge disconnected subgraphs within one split, but fragmentation of a real connected component or any cross-split overlap fails closed. The dev2 tool excludes historical pair/seller/alias/component entities, selects at most one pair per eligible candidate component and emits separate score-blind reviewer queues;
- new implementation files are `scripts/step16i_audit_data_integrity.py`, `scripts/step16i_prepare_retrospective_dev2.py`, `scripts/run_step16i_integrity_dev2_linux_20260716.sh`, `schema/step16i_data_integrity_policy.json`, `schema/step16i_retrospective_dev2_policy.json`, `tests/test_step16i_data_integrity_contracts.py` and `docs/STEP16I_DATA_INTEGRITY_AND_RETROSPECTIVE_DEV2_PLAN_20260716.zh.md`. Numerical/data execution remains Linux-only.
- Step16I v1 has now completed on Linux. It audited Chinese `893` and English `734` primary rows, emitted `1,627` component-assignment rows and `7,991` typed exclusions, and found zero cross-split pair, seller, portable-alias or recomputed-component leakage in either dataset and zero EN-ZH overlap. Its initial V8 check failed solely because three persisted readiness components each conservatively merged two disconnected seller subgraphs within one split. That coarsening reduces effective group count but cannot leak a seller across splits. The contract now records it as a warning while continuing to fail on connected-component fragmentation, cross-split overlap, duplicate pairs or unknown splits. The failed v1 artifact remains immutable; the corrected rerun must use `step16i_integrity_20260716_v2`.
- Step16I v2 passed the corrected safety contract with only the three documented conservative-coarsening warnings. The retrospective builder found `353` eligible historical candidate components and selected `160` pair/component-disjoint rows covering `473` sellers in the full candidate components. The selected strata are `120` semantic-similarity, `35` high-priority and `5` template-clone candidates; all `160` have `not_applicable` occurrence context, zero shared contact, zero shared PGP and zero same-alias relation. This is therefore a soft-similarity/hard-negative development stress set, not a solution to proof-positive scarcity.
- Two isolated AI reviewers completed sensitivity-only annotations without reading the mapping, each other's queue or model/graph outputs. Reviewer A produced `27/109/24` same/different/uncertain and Reviewer B `1/100/59`. Identity agreement is `117/160 = 73.125%`, evidence-type agreement `92/160 = 57.5%`, exact identity-plus-evidence agreement `91/160 = 56.875%`, and high-confidence exact agreement only `47/160 = 29.375%`.
- The dataset owner then authorized Codex to perform the third adjudication. The frozen result is `1 same / 108 different / 51 uncertain`. The sole positive is the blind pair whose two visible inventories explicitly self-identify as the uncommon handle `独孤信`; the otherwise promising first-person gambling-backend pair was conservatively retained as uncertain because it lacks a shared account identifier, migration statement or independently closed component anchor. The exact provenance is `two_ai_blind_reviews_plus_codex_adjudication_owner_authorized`: it is owner-authorized, but not per-row human-verified and must not be described as two-human gold annotation.
- `scripts/step16i_reconcile_ai_sensitivity_reviews.py` verifies immutable queue hashes and every non-decision cell, joins only by blind ID, refuses the mapping as input and emits agreement diagnostics without pair/seller IDs or inferred labels. `scripts/step16i_finalize_codex_adjudication.py` freezes all 160 blind-ID decisions before unblinding, and `scripts/step16i_materialize_codex_adjudicated_dev2.py` performs a hash-checked join to the original mapping in an isolated Dev2 table. No output modifies Step5 or the canonical train/valid/test boundary.
- The materialized retrospective Dev2 contains `109` binary rows (`1 positive / 108 negative`) plus `51 uncertain`, with `160` unique pairs and `160` unique candidate components and zero overlap with the permanent excluded-pair manifest. Its `0.917%` binary positive prevalence makes it a hard-negative false-positive stress set, not a balanced paper benchmark or a solution to proof-positive scarcity. Detailed report: `docs/STEP16I_CODEX_ADJUDICATION_REPORT_20260716.zh.md`.

`2026-07-15` post-run audit invalidated the first Step15-v8 readiness-v2 primary-validation result and prepared a corrected v3 rerun on branch `method/step15-v8-validation-slice-expansion`:

- the synchronized v2 return bundle is complete: its manifest binds `226` files (`76,259,251` bytes), and a full size/SHA-256 replay found `0` missing or mismatched files. The weak result is therefore not a synchronization failure;
- v2 selected B3/RankNet from train-component OOF AP `0.789716`, but representative-valid AP was only `0.635463`; contextual fusion produced AP `0.634784`. It failed clean gain, public-noise FPR reduction, fusion non-degradation and both grouped-bootstrap noninferiority gates. The 200-row internal-development test remained diagnostic (`clean AP=0.534110`, `fusion AP=0.587025`) and was not used for selection;
- a deeper split audit found the decisive protocol defect: the v2 170-row primary validation retained an old v7 overlay that had promoted `50` canonical-train rows into validation. Of those rows, `45` were explicitly `silver_train_only=1` and `benchmark_eligible=0` (`44` positive, `1` negative). Consequently, `44/76` v2 validation positives were training-only silver rows. The v2 primary-validation metrics, selected threshold and Step12 promotion decision are retained only as invalidated diagnostics and cannot support a paper claim;
- the canonical Chinese binary boundary remains `573 train = 229 positive / 344 negative`, `120 valid = 30 positive / 90 negative`, and `200 internal test = 50 positive / 150 negative`. All `371` train-only silver rows stay in train; canonical valid/test contain no silver and are fully `benchmark_eligible=1`;
- corrected v3 restores every primary row from canonical Step5 `split_name`, while preserving the exact 200-pair internal-test membership. Expected primary counts are `974` training rows (`401` English source-train plus `573` Chinese train), `120` Chinese representative-valid rows and `200` internal-test rows. Evidence-expert controls remain physically separate (`60` train, `48` valid) and never enter primary identity metrics. The valid control count is deficit-based: canonical valid already contributes public/direct/component `4/3/0`, so new controls are `16/17/15`, preserving total readiness `20/20/15` without double-counting existing benchmark rows;
- V3 control selection now uses the frozen seed `readiness_expansion_v3_20260715`, recorded in both summary and freeze manifest. `run_id` and timestamped output paths are provenance names only and cannot change selected pairs. Artifact tests validate the exact manifest-declared superseded set while continuing to require that every superseded upstream row was uncertain and ineligible for supervision; no fixed superseded-row count is assumed across obsolete runs;
- `scripts/step15_v8_common.py` and the readiness materializer now fail closed if any primary valid/test row is nonbenchmark or train-only silver. `step20_build_representative_validation.py` may retain silver in train but may move only components whose every row is benchmark-eligible;
- canonical seller partitions are now authoritative for selecting evidence controls. Stale queue labels such as `valid_candidate` can no longer move a seller across train/valid/test. Existing v3 output directories can be reused only when every frozen payload is byte-identical to a fresh deterministic replay;
- a real frozen-input quota test shows that the original context-review pool supplies `12` public-noise controls compatible with canonical valid, versus a new-control deficit of `16` after counting the `4` occurrence-backed public-noise negatives already present in canonical valid. Two score/split-blind review rounds considered `11` proposed profile-URL pairs: source-literal validation removed one invalid supposedly shared `.onion` premise, leaving a final ten-pair candidate universe. Eight review records received two-reviewer high-confidence agreement, but `5kqp0.com` and `jnqp.com` duplicate seller pairs already present in the context-review pool. Canonical `pair_uid + URL token` deduplication therefore yields `6` unique supplemental pairs and `18` unique candidate controls overall; V3 deterministically selects `16`, reaching the unchanged total readiness minimum of `20`. The controls remain evidence-expert-only, never Step5 gold, primary supervision or benchmark rows;
- the supplemental control chain is explicit and hash-bound: `schema/step16_v8_profile_url_control_candidates.json`, two immutable review-lane files with the actual per-candidate agent reviewer IDs, `scripts/step16_reconcile_v8_profile_url_reviews.py`, candidate-context verification, exact reviewer-universe checks, per-pair reviewer independence and high-confidence agreement. The Linux materializer accepts only the eight validated negatives;
- v3 review-derived outputs use `reports/step15_v8/profile_url_control_review_v3_20260715/`, separate from the reviewer-lane input directory and all v2-derived outputs. Immutable-write checks therefore detect content drift without blocking a legitimate new candidate universe or overwriting history;
- independent code review found and the implementation now fixes four additional integrity defects: supplemental URL controls materialize two-sided risk occurrences and must pass pair-level occurrence-state checks; V3 artifact tests run only against the newly materialized root rather than silently reading V2; Step20 self-hash covers the assignment CSV hash; and new control candidates use `evidence_expert_control` scope. Platform vendor IDs are also reservation keys, preventing a repeated vendor from crossing control splits in future expansion;
- corrected local verification discovers `50` Step15-v8 tests: `43` static/real-input contracts pass and `7` V3 artifact tests are explicitly deferred until Linux materialization. The combined v6/v7/v8 suite discovers `99`, executes `92` successfully and defers the same `7`. The real frozen-input contract reaches public `valid/train=20/20`, direct `20/30` and component `15/10` without seller/component overlap. The Linux runner re-executes all V8 tests with `STEP15_V8_READINESS_ROOT` bound to V3 after publication. No model encoding, feature generation, training, threshold selection or numerical experiment was run on Windows;
- the current Linux entry point is still `scripts/run_step15_v8_readiness_linux_20260715.sh`, but it now publishes isolated roots `readiness_expansion_v3_20260715` and `bridge_v8_readiness_v3_20260715`. V2 is never overwritten or silently reused;
- detailed correction audit: `docs/STEP15_V8_V2_POSTRUN_AUDIT_AND_V3_CORRECTION_20260715.zh.md`; readiness protocol: `docs/STEP16_V8_READINESS_EXPANSION_PROTOCOL_20260715.zh.md`; method design: `docs/STEP15_V8_CONTEXTUAL_EVIDENCE_FUSION_PLAN_20260714.zh.md`.

The clean Linux v3 run described above is complete and failed promotion. Step16I integrity, retrospective Dev2 preparation, two isolated AI reviews and owner-authorized Codex adjudication are now complete. The resulting Dev2 is suitable only for retrospective hard-negative stress analysis. Step20 and Step11/17 publication promotion remain blocked; the next valid evaluation advance requires a frozen model followed by new prospective raw data and per-row human verification.

### Superseded 2026-07-14 v8 pilot record

The following records the initial blocked pilot that motivated the completed upstream expansion; it is not the current readiness state. Step15-v8 contextual-evidence-fusion is an isolated successor to the frozen v7 result, not an in-place v7 retune:

- the formal goal is to explain the same-boundary AP gap between v7 clean (`0.463904`) and the historical v6 M3/normalized-retrieval diagnostics (`0.594897/0.597533`) without restoring identifier-contaminated semantics or using the current internal test for selection;
- `schema/step15_v8_contextual_evidence_policy.json` preregisters B0-B3, ten seeds, five seller-component folds, train-OOF-only representation/model-family selection, an LR/L2 versus linear-pairwise-RankNet comparison, occurrence-level evidence fusion, validation promotion gates and Step20/Step11/17 fail-closed rules;
- B0 rebuilds the v7 20d plus E5-latent-64d representation under the v8 fold-train protocol (it is not a bitwise replay of old v7 scores); B1 removes the random latent; B2 uses identifier-redacted E5/BGE/LaBSE cosine consensus; B3 additionally restores only an explicit four-rule nonidentifier retrieval count, fold-train/domain-normalized lexical/structural support and a forward/reverse-mean identifier-redacted reranker;
- historical aggregate candidate-rule counts are not trusted. V8 recomputes `candidate_rule_count_non_identifier_v8` from `profile_lexical_neighbor`, `shared_title_clone`, `shared_description_clone` and `structural_support` only; contact, PGP, supplemental-contact and component-closure rules cannot enter it;
- every OOF fold now refits IDF, OOV, market percentiles, domain lexical/structural normalization, imputation and standardization from fold-train sellers/rows only. Representative valid and internal test receive a reference fitted only on the complete train boundary;
- representation and model-family selection use mean ten-seed macro-domain train-OOF AP. Representative valid is opened only after this selection for one promotion/threshold check; the 200-row Chinese internal test remains diagnostic and satisfies no gate;
- the v7 token-wide `score * 0.1` veto is removed. The v8 compact offset LR/L2 expert receives clean component-OOF probability plus occurrence-context features. Pure direct evidence may only increase a logit, risky/support/high-frequency evidence may only decrease it, while mixed, ambiguous and no-identifier states must leave the clean score unchanged;
- a score-blind Step16-v8 queue builder separately emits immutable risky-only, mixed-context and verified-direct internal queues plus independently shuffled reviewer-A/reviewer-B blind packets. Reviewer packets conceal queue kind, occurrence state, split/feature eligibility and all model outputs. Rule hits cannot update Step5. Two distinct reviewers must produce matching high-confidence identity/evidence decisions, or a distinct third reviewer must adjudicate; low-confidence, uncertain, internal-test and cross-split-component candidates cannot become binary supervision;
- the blind-packet renderer was corrected after audit: parser-derived `direct/seller_facing/risky/support` flags remain available only in the immutable diagnostic queues and are no longer embedded in reviewer context previews. A contract test now rejects those state hints in blind evidence;
- a Windows queue-only and agent-assisted pilot was completed without model inference. The source produced `143` risky-only, `2` mixed-context and `1` verified-direct candidates, but only `7` candidates were both feature-ready and train/valid split-eligible; all seven were train-side risky-only candidates and none could extend valid. Two isolated agents agreed exactly on `6/7` identity/evidence decisions; a third isolated agent adjudicated the remaining case as uncertain. The resolved set contains five materializable high-confidence train public-noise negatives, one uncertain pair, and one apparent direct-contact positive held out because it conflicts with the parser's risky-only state. This is disclosed as agent-assisted internal review, not human double annotation;
- `scripts/step16_apply_v8_context_reviews.py` materializes only reviewed, feature-ready pairs into isolated Step5/evidence-label overlays, recomputes complete seller components, preserves the exact 200-pair internal-development-test membership and emits a hash-bound representative-validation manifest plus generated v8 policy. It never edits the canonical Step5/evidence/split artifacts in place;
- Step12-v8 requires clean AP gain `>=0.03`, public-noise FPR reduction `>=0.20`, direct/component recall drop `<=0.05`, no template FPR increase, no fusion AP loss, grouped-bootstrap noninferiority for both clean-minus-B0 and fusion-minus-clean, and no test selection. It additionally requires valid slice counts of at least `20` state-backed public-noise negatives, `20` state-backed verified-direct positives and `15` component-anchor positives;
- the former `6/18/16` values are legacy evidence-type counts, not formal v8 readiness counts. Audit found that some historical `public_contact_or_url_noise` rows have no shared identifier and are actually template/data-package negatives. V8 now defines the public slice as `negative + risky/support/high-frequency occurrence state`, the direct slice as `positive + verified-direct-both-sides`, and the component slice as `positive + component-anchor evidence type`;
- preflight recomputes those slices before any GPU work. If `20/20/15` is not met, it instructs the Step16-v8 review/refreeze path and stops; thresholds are unchanged and the internal test cannot satisfy a deficit;
- all v8 outputs are isolated under `reports/step15_v8/<run_id>/`; existing run IDs refuse overwrite, and a content-addressed return-sync manifest binds every artifact;
- the Step20 one-time lock is isolated by v8 run ID and must bind the exact Step15-v8 model-freeze SHA-256 plus one-time/frozen-before-unseal declarations. A stale lock cannot release Step11/17;
- Windows validation now includes Python syntax compilation, Git-Bash runner syntax checks, config-only entry points, nineteen pure synthetic contract tests (`19/19` pass), the score-blind queue-only build and the explicitly disclosed agent-assisted pilot above. No model encoding, model training or numerical performance experiment was run on Windows;
- Linux queue-only runner: `scripts/run_step16_v8_validation_queue_linux_20260714.sh`; reviewed-refreeze runner: `scripts/run_step16_v8_validation_refreeze_linux_20260714.sh`; full model runner: `scripts/run_step15_v8_linux_20260714.sh`; detailed design: `docs/STEP15_V8_CONTEXTUAL_EVIDENCE_FUSION_PLAN_20260714.zh.md`.

This pilot found readiness valid `4/3/0` and train `5/0/0`, so the original run was correctly blocked. That conclusion is superseded by the 2026-07-15 Step4/v7 expansion and isolated freeze above; it remains useful as provenance showing that additional reviews alone did not manufacture the later coverage.

### Frozen v7 predecessor

`2026-07-14` Step15-v7 v2 identifier-redacted two-stage/prospective code path and source-level static audit are complete; Python syntax checks, contract tests, data preflight and all numerical runs remain reserved for Linux:

- active branch: `method/step15-v7-two-stage-prospective`;
- Step15-v6 is frozen as a strict negative result through `schema/step15_v6_negative_freeze.json`; its selected M4 result and Step12 `promotion.eligible=false` artifacts are hash-bound and cannot be overwritten by v7;
- the legacy Chinese test is permanently downgraded to `internal_development_test`. No v7 model, augmentation mode, threshold or reliability rule may be selected from its metrics;
- a score-blind representative validation overlay recomputes seller connected components over all eligible Chinese supervision. Development preflight expects 12 complete train components to move, producing `train 523 = 183 positive / 340 negative`, `valid 170 = 76 positive / 94 negative`, and `internal development test 200 = 50 positive / 150 negative`, with seller/component overlap `0`; Linux must reproduce these counts in the formal manifest;
- representative valid now contains `18` direct-identifier positives across `10` components, `16` component-anchor positives across `5` components, and `6` public-contact/URL negatives across `2` components. It also retains soft positives, ordinary negatives, semantic-topic negatives and template-clone negatives;
- static code audit found that canonical Step7 semantic caches encode `profile_text`, which contains seller aliases, contacts and structural sections. v7 v2 therefore builds a separate identifier-redacted Multilingual-E5 cache from category/title/description content only, removes known Step3 identifiers and seller aliases without leaving presence markers, and treats all legacy profile-text semantic scores as diagnostic-only. Core/prospective caches must also match on the content-addressed local model-directory fingerprint, producer hash and frozen v7 policy hash;
- v7 strict-clean uses 20 configured features: one identifier-redacted E5 cosine plus 19 structural/style/corpus-relative fields. It removes five unstable/OOV-only corpus features, `candidate_rule_count_raw`, uppercase-specific/raw retrieval-only fields, and six identifier-contaminated legacy semantic/reranker scores from the main view. Train-only IDF uses `effective_df >= 2`, and unknown signatures remain OOV diagnostics rather than train-supported rare evidence;
- the clean scorer adds a symmetric 64d projected identifier-redacted E5 seller-pair latent representation to the 20d strict view, for 84 total dimensions. Pair endpoint order cannot change this representation;
- evidence weights are factorized as `domain x evidence_type x confidence x inverse-sqrt component`, clipped to `[0.1, 2.5]`; component factors are normalized within each `domain x evidence_type` stratum so rare evidence classes are not globally suppressed, and the old global 8x pattern is forbidden;
- Step9-v7 preregisters no augmentation, equal-effective-weight duplication and true latent pair-embedding mixup for five support ratios (`0/10/20/50/100%`) and ten seeds. Positive ratios are deterministic nested `label x evidence_type` stratified samples with a one-row minimum per nonempty stratum, so artifacts record actual support counts, stratum counts and sampled-pair hashes rather than treating ratio labels as exact global row percentages. The 0% support run is a matched English-label-only source-fusion control; Step12 checks its deterministic seed repetitions are identical. Mixup is ZH-train-only, same-domain, same-evidence-type and nearest-neighbor constrained. Clean features are anchor-copied; only latent coordinates interpolate. Duplication and mixup must have exactly equal synthetic effective weight;
- the mixup budget is computed only from the sampled Chinese support set's negative-minus-positive effective-weight gap; English source-domain class counts cannot inflate the Chinese synthesis budget. The last synthetic row is weight-capped to the exact remaining budget, and the 100% support main comparison fails closed unless its fixed 50% gap-closure budget is satisfied;
- v7 uses uniform class weights so the duplication control isolates added minority mass from latent interpolation geometry. All three controls share a scaler fitted on real rows only and a total optimizer weight fixed to the real-row count, preventing synthetic row count from changing preprocessing or relative L2 strength;
- Step15-v7 removes the non-identifiable auxiliary evidence head. Stage A is the validation-selected clean ranker; Stage B uses only inference-visible raw Step3 occurrence context to veto public/product/support identifiers by a fixed 0.1 multiplier. Review labels and evidence types never enter Stage B inference;
- Step12-v7 uses representative-valid selection only, grouped component bootstrap, paired component score-swap permutation and paired seed/component two-level intervals. The old 200-row test remains diagnostic, publication promotion is hard-coded `false`, and the complete statistics plus model/threshold freeze are atomically published in one directory;
- Step20 requires per-row collection timestamps strictly after the model freeze, validates a prospective candidate schema, and fails closed before queue publication unless all five preregistered candidate-category quotas and the 400-pair seller-disjoint queue target are met. It hides candidate-category hints from reviewers, publishes a label-free pair universe, performs dual independent review plus third-reviewer adjudication, and keeps scoring physically isolated from labels until an atomic one-time evaluation lock is created;
- development lineage inspection found `0` eligible prospective candidates: `1,016` current rows were already reviewed and all remaining `2,841` candidates predate v7 freeze. They cannot be repackaged as a new final holdout. Linux must reproduce this diagnosis, and new post-freeze Chinese raw data is required;
- Windows is used only for source edits, static inspection and Git/sync management. No v7 script or model is to be executed on Windows; syntax, contracts, data lineage and all numerical results are verified by the Linux runners;
- all v2 outputs are path-isolated under `v2_identifier_redacted_20260714` or Step20 `*_v2` stage directories, so no v1/v6/v5 artifact can be overwritten or interpreted under a different feature dimension;
- Linux core runner: `scripts/run_step15_v7_linux_20260714.sh`; Step20 staged runner: `scripts/run_step20_prospective_holdout_linux_20260714.sh`;
- detailed design and interpretation rules: `docs/STEP15_V7_TWO_STAGE_PROSPECTIVE_DESIGN_20260714.zh.md`.
- completed workspace cleanup record: `docs/WORKSPACE_CLEANUP_AUDIT_20260714.md`; a separate cleanup removed `95` obsolete temporary/rejected-probe files (`126,239,312` bytes, about `120.4 MiB`) without deleting any active v7 input, frozen result, manifest-bound artifact or publication control.
- the cleanup also retired the two tracked Step11 archive dry-run inventories after removing their active documentation reference. Current Step11 auditing remains manifest/explicit-allow-list only and continues to resolve model CSVs from each summary's `output_paths`.
- the first Linux v7 attempt stopped safely at Stage `[4/8]` before model loading or cache publication: single-pass redaction normalized whitespace only after scanning, which could expose a cue-handle pattern that had exceeded a regex spacing bound during the first scan. Seller `market_item.xlsx|Lime|seller_raw:AMAZONSHOP` triggered the post-redaction fail-closed assertion; this is a redaction-closure defect, not a bad-label or model result.
- the clean-cache builder now repeatedly applies all generic rules and seller literals followed by whitespace normalization until a fixed point, with an eight-pass convergence guard, named-rule residue diagnostics and manifest counters. A regression fixture covers a Telegram cue separated from its handle by 20 spaces. The repair neither drops sellers nor relaxes the residue assertion; Linux must rerun the complete v7 runner after synchronizing the changed script and contract test.

The next evidence is the clean Linux v7 run. Its internal-development metrics must be reported as diagnostics regardless of direction. A publication-level final claim remains blocked until genuinely new post-freeze data meets Step20 evidence quotas and is evaluated once.

## Frozen Step15-v6 Negative Result (Historical Baseline)

`2026-07-13` Step15 v6.4 inductive paper-hardening implementation is complete; the corrected Linux run has completed Stages 1-10 and Step12 input validation, while the first canonical Step12 statistics process was stopped for an execution-only deterministic CPU optimization:

- the failed Step12 check was `step15_v6_m0/seed=20260325/valid`: the Step15 summary reported ROC-AUC `0.766667`, while the persisted CSV reproduced `0.766852`;
- root cause: Step15 calculated summary metrics from full-precision in-memory probabilities but persisted `prob_positive` and threshold at six decimals. Quantization created artificial score ties, so rank metrics recomputed by Step12 were not guaranteed to match the summary;
- this is an artifact reproducibility defect, not evidence that the model changed or that Step12's tolerance should be relaxed. Step12 remains fail-closed at `2e-5`;
- Step15, its source-only control, and the Step7 prediction writer reused by isolated Step9 now persist Python round-trip float values. A regression test proves that positive/negative scores separated by less than `1e-6` remain rank-distinct;
- the first post-fix resume reached M5 endpoint materialization and exposed one remaining six-decimal consumer contract: the artifact threshold was rounded before comparison with the new full-precision CSV threshold. The consumer now requires exact round-trip agreement among the artifact, run record, and every validation-prediction row, validates finite ranges, and rechecks every persisted `pred_positive` against its score and threshold;
- the M5 end-to-end frozen-artifact test now uses a non-six-decimal threshold (`0.5000003456789012`), and dedicated negative tests reject both threshold drift and score/decision inconsistency. Static review finds no remaining rounded-threshold comparison in the Step15-v6/Step12-v6 path;
- because the active manifest binds producer-script and prediction hashes, the earlier precision-defective bundle was replaced by a corrected Stage 1-10 rerun. The current active manifest and validated model predictions remain the frozen Step12 inputs;
- canonical Step12 input validation now passes with fixed test `200 = 50 positive / 150 negative`, the preregistered positive slices, 24 loaded model/control aliases, validation-only model selection, and no test-informed selection;
- the original Step12 implementation ran the 5000 grouped bootstraps, 5000 score-swap randomizations and two-level audits on one CPU core. Its main hotspot recomputed the complete 13-metric evaluation up to 26 times for the same resampled score vector, and all model/slice/comparison tasks were serial;
- Step12 policy v5 removes that redundant work by evaluating each sampled score vector once and extracting all requested metrics, while metric-specific randomization computes only AP, ROC-AUC or PR-AUC as requested. It then executes model, evidence-slice/model and comparison-scope tasks in a deterministic 24-process pool, with one native thread per worker;
- every worker reconstructs the original preregistered NumPy random stream from the unchanged task seed formula. Ordered result collection preserves model, slice, comparison and metric row order independently of process scheduling;
- direct executable comparison against pre-optimization commit `0fced64` produced exact Python object equality for all model metric rows, slice rows, primary grouped bootstrap/randomization rows and supplemental two-level rows. Serial and two-process outputs are also field-identical; the focused synthetic benchmark reduced the model-statistics loop by `7.19x` before process parallelism;
- this optimization changes no labels, predictions, model selection, thresholds, resample count, permutation count, confidence interval definition, p-value definition, Holm family or promotion rule. Only Step12 must be rerun; Stages 1-10 and their active manifest do not need regeneration for this execution-only change;
- Step12 remains a CPU workload. Its 150-200-row sorting, component indexing and branching tasks are too small for a CUDA rewrite to be efficient, while the server's 24 physical cores match the 24 independent worker limit;

- active branch: `method/step15-v6-paper-hardening`;
- the synchronized v5r run is frozen as `internal-dev-v5r-20260711` in `reports/manifests/step15_internal_dev_v5r_20260711.json`; all 44 referenced summary/policy/output files exist and are SHA-256 recorded;
- current Chinese `test = 200 = 50 positive / 150 negative` is explicitly downgraded to a fixed internal-development test, not a prospective final holdout;
- Step4 code defines `candidate_rule_count_non_identifier`; the v6 isolated builder materializes it from the field when present or deterministically derives it from frozen `candidate_rule_hits`. The v6 runner does not rebuild canonical Step4/Step7/evidence-label files, and the frozen universe remains EN `6683`, ZH strict `3857`, ZH aux `580`;
- old corpus-relative IDF, boilerplate/rarity and percentile features were transductive because they used complete language pools. v6.4 recomputes all 18 such fields from frozen train sellers only (EN `582`, ZH `676`) and applies the frozen references to valid/internal-test/Step11 rows. In-memory full-data validation preserves all pair UIDs and every semantic-score hash;
- the Step15 v6 strict-clean main view removes `candidate_rule_count_raw`, raw retrieval-only lexical/structural fields, uppercase gap, and all direct identifier fields; a separate 33D ablation restores only the non-identifier rule count plus train-only domain-standardized lexical/structural fields;
- all M0-M5 models and matched-budget controls with the same feature view share a final-Phase3 train-only scaler; artifact hashes record global and optional per-domain statistics, and unknown domains fail closed;
- sample weighting is component-aware (`1/sqrt(component train edge count)`, mean normalized) before row quality; class balance now equalizes the incoming positive/negative effective weight mass rather than raw row counts, followed by evidence and optional post-quality domain balance;
- M0-M5 remain the main method ladder. M0/M1/M2 now each use a fixed 1000-update all-at-once budget, matching M3's total `4 x 250` updates. M2b additionally repeats full Phase3 data over the same four-stage optimizer trajectory as M3, and M4c performs the same fifth 250-update continuation as M4 with mixup disabled. A curriculum claim requires both M3-vs-M2 and M3-vs-M2b; a mixup claim requires both M4-vs-M3 and M4-vs-M4c;
- incomplete or out-of-order warm-start phase requests fail before training, and any partial prefix remains valid-only, so it cannot expose or overwrite a canonical test endpoint;
- M5 `lambda = 0.1/0.3` selection requires all ten preregistered seeds and uses AP of the ten-seed mean `zh_valid` scores. Both candidates remain valid-only; ties prefer `0.1`, and only the selected frozen artifact is subsequently scored on internal test without retraining;
- v6 uses ten fixed seeds `20260320..20260329`, hidden size 16, validation-only model/threshold selection, endpoint-only internal-test evaluation and gold/silver label-source ablations. The LR/L2 source control uses only English identity labels and English validation thresholding, but discloses unlabeled Chinese train-seller reference preprocessing; it is source-label-only, not strict no-target-covariate zero-shot;
- metric semantics are upgraded to `2026-07-v2-tie-aware`: AP is tie-group order invariant, PR-AUC is a real trapezoidal curve area, and MAP/MRR are null because no query groups are preregistered;
- Step12 v6 uses AP as primary. Primary CIs resample `split_component_id`; supplemental two-level CIs also resample seeds. P-values now come only from paired component score-swap randomization, not bootstrap sign proportions, and Holm correction groups all scopes by `analysis mode x metric`. Promotion additionally requires positive AP-difference CI on `strict + soft` positives against both M0 and the strongest clean Step9 selected exclusively on `zh_valid`, plus exactly ten paired seeds and at least 8/10 wins;
- all formal Step9 controls now run the same ten seeds as Step15, removing the earlier 3-seed-versus-10-seed ensemble-size confound;
- fixed-test membership and Step16F tier mapping are protected by canonical SHA-256 hashes, while a versioned, self-hashed active manifest freezes every Step15/source-only and selected Step9 prediction/artifact used by Step12; embedded Step15 input manifests, Step9 context fingerprints, summary paths, tokens, labels, components, and recomputed ranking metrics are cross-checked so stale, partial, wrong-phase, wrong-seed, relabelled, or score-mismatched predictions fail closed;
- canonical Step12 outputs are isolated under `reports/step12_v6/method_audit_v4_inductive_20260712/`, require exactly 5000 bootstrap resamples and 5000 paired randomizations, and refuse to overwrite any existing target. The optimized run prints completion time after model, slice and comparison stages so normal execution is no longer silent;
- Step13 now separates raw candidate, gold, silver-train, fixed-valid, and internal-test cohorts so active silver sampling is not misreported as natural concept drift. The core runner no longer emits Step13 before graph validation; the current final Step13 can only be generated after Step11 from explicit Step12 summary, Step11 manifest and Step11 audit paths;
- Step11 v6 is promotion-gated, auto-selector-disabled, manifest-only and writes under `reports/step11_v6/`. Clean topology disables identifier weights/context/hard-keep and audits direct proof only after filtering; identifier-assisted operational mode is a separate output namespace that may hard-keep direct proof edges. Both modes require an exact scorer-family/token roster and record proof/negative/isolated-pair outcomes at every graph-filter stage;
- the clean raw/zero-shot graph control is now `raw_bge_m3_cosine`, read directly from the v6 inductive pair feature table. Its threshold is the Step12 `mean_zh_valid_scores` threshold, frozen from `zh_valid`; the old Step7 BGE LightGBM control is deliberately excluded because applying a model trained on old corpus-relative feature scales to v6 inductive features would be a preprocessing mismatch;
- Step11 runtime policy construction rehashes every file in both the Step15 active manifest and Step12 completion manifest. Publication runtime rejects any explicit scorer request outside the four clean tokens or the one separate identifier-operational token;
- Step11 publication manifests now reject a primary threshold above the score ceiling, zero primary-threshold candidate edges, or zero post-filter edges. Runtime policies, graph outputs, explicit manifests, cluster audits and final Step13 outputs are immutable by path: an exact byte replay is accepted, while same-name content drift fails and requires a new run ID/path;
- every publication manifest binds its CSV by SHA-256 and row count. Every cluster-audit JSON self-hashes its canonical content and binds its CSV by SHA-256, row count, decision counts and per-scorer counts; Step13 independently rechecks that full chain before reading graph evidence;
- Step16H v2 is superseded and excluded from the current hash-closed bundle because its completion manifest no longer binds the current producer-script bytes. Only the v3 bundle is current;
- Step16H v3 evidence-complete dual AI review and third-reviewer adjudication are complete without writing back to Step5/Step16F. Overall agreement is `0.875`, kappa `0.7992`, alpha `0.7980`; final old-positive decisions are `18 strict / 48 soft / 11 different / 3 uncertain`, while negative controls contain `0 strict / 24 soft / 55 different / 1 uncertain`. This is an AI-assisted sensitivity audit with procedural blinding, not human gold annotation;
- the complete local preflight passes `85/85` tests. Coverage includes train-only corpus references, candidate-universe invariance, M5 valid-only selection and frozen-artifact test materialization, exact threshold binding, metric semantics, sub-micro score serialization, paired component randomization, Holm families, strict+soft promotion, deterministic serial/process-pool equivalence, weights/budgets/mixup, immutable-output rejection, Step11 raw-BGE/allow-list/empty-graph gates, Step11 audit CSV integrity, Step13 provenance and Step16H concealment;
- the 11-stage Linux core runner compiles the synchronized scripts and reruns all 80 contract tests before training. It blocks manifest creation until `step15_validate_v6_outputs.py` proves complete experiment/phase/seed coverage, all fixed update counts, exact M4/M4c Phase0-3 valid predictions and artifact parameters, M4-only Phase4 mixup, selected-only M5 test output, and complete source-only runs. It ends at Step12; the separate promotion-gated Step11 runner recompiles the graph/audit scripts, performs graph validation and only then generates Step13;
- no Windows model training was performed. Full Linux runner: `scripts/run_step15_v6_linux_20260711.sh`; post-promotion Step11 runner: `scripts/run_step11_v6_after_promotion_20260711.sh`.

Detailed implementation record: `docs/STEP15_V6_PAPER_HARDENING_IMPLEMENTATION_20260711.zh.md`.

Linux synchronization requirements for the v6.4 rerun:

- synchronize the complete current `scripts/`, `schema/`, `tests/` and `docs/` trees plus the current Step3-Step7 data artifacts under `reports/`;
- at minimum ensure the new/changed v6 scripts and policies, the frozen Step5 labels, Step7 pair features, Step15 evidence labels, Step16F re-audit, v5r freeze manifest and Step16H audit artifacts are present;
- Step4/Step7 canonical feature files and Step15 evidence labels are not rebuilt; v6 constructs isolated inductive features under `reports/step15_v6/features/` and performs a read-only Step7 metric refresh;
- Step9 writes only under `reports/step15_v6/baselines/step9/` via `--output-root` and explicitly reads the inductive EN/ZH feature files, so old canonical Step9 artifacts remain untouched;
- for the current corrected Stage 1-10 bundle, rerun only optimized Step12 with `--workers 24`, inspect promotion, and run `bash scripts/run_step11_v6_after_promotion_20260711.sh` only if promotion is eligible. A future clean-from-scratch replay can still use `bash scripts/run_step15_v6_linux_20260711.sh`, whose Stage 11 now explicitly requests 24 workers.

`2026-07-11` Step 15 v5r implementation repair completed locally; Linux rerun is pending:

- active repair branch: `fix/step15-weighted-same-domain-mixup`;
- legacy v5 artifacts and the synchronized `2026-07-10` metrics remain untouched as the pre-fix comparison;
- new experiments use distinct `step15_v5r_*` names and write `reports/step15_v5r_weighted_mixup_summary.json`, so the rerun cannot overwrite v5 results;
- Phase 4 now admits only positive parents with `training_sample_weight >= 0.55`, `usable_for_core_transfer = 1`, `core_transfer_eligible = 1`, and confident evidence type;
- parent selection is restricted to the same real language domain and same evidence type, then to one of the five nearest eligible positive neighbors;
- synthetic rows inherit `min(parent_weight_left, parent_weight_right)` instead of defaulting to `1.0`;
- binary/count features are copied from the anchor parent; only continuous features are interpolated;
- every synthetic row records both parent `pair_uid` values, domains, evidence types, parent weights, interpolation coefficient, and inherited weight in a per-run manifest;
- domain-balanced v5r computes class, evidence-type, and row-quality weights first, then equalizes effective mass across only `en_content_train_pool` and `zh_target_strict`; unknown pseudo-domains such as `cross_domain_mixup` are rejected;
- Step 12 now has isolated v5r Phase3-vs-Phase4, v5r-vs-v5, raw-E5, and domain-vs-non-domain paired comparisons with new `20260711` output paths;
- four focused unit tests pass: same-domain/evidence parent enforcement and discrete-feature preservation, inherited parent weight, effective domain-mass equality, pseudo-domain rejection, and legacy-domain replay compatibility;
- `scripts/step15_validate_v5r_outputs.py` adds a post-training fail-fast check over all six Phase-4 artifacts/manifests before Step 12 runs;
- no Windows model experiment was run. The next evidence must come from the Linux three-seed rerun.

Linux synchronization incident and repair:

- the first Linux v5r attempt stopped at `[3/7]` with `Unknown Step15 experiment`, because the runtime training script referenced the new v5r name while the Linux `schema/step15_evidence_type_policy.json` did not contain that experiment;
- this was a mixed-version synchronization failure, but the original runner should have detected it before rebuilding labels or entering the training stage;
- `scripts/step15_train_incremental_hard_negative.py` now supports `--validate-config-only`, which validates the exact experiment/phase/seed selection and v5r output/mixup/domain-balance contract before loading data;
- the Linux runner now invokes that exact dry configuration command in `[1/7]`; an omitted script or policy update therefore fails immediately with the policy path and version;
- the Windows preflight was rerun with the exact Linux experiment arguments and returned `status = pass`, policy version `2026-07-11-v5r-trusted-weighted-same-domain-mixup`, both v5r experiments, both requested phases, all three seeds, and the isolated v5r summary path;
- the local test suite now contains five passing tests, including a policy/runtime contract test. No model training was performed on Windows.

Linux v5r manifest serialization incident and repair:

- after the synchronized config preflight passed, the second Linux attempt reached the first Phase-4 manifest write and failed because five synthetic eligibility fields were present in each internal row but absent from the fixed CSV field list;
- model construction was not the failing operation; `csv.DictWriter` rejected undeclared dictionary keys before the Phase-4 artifact/prediction set and top-level summary could complete;
- `write_positive_mixup_manifest()` now includes the five audit fields and explicitly projects every internal row onto the declared manifest schema, so future internal metadata cannot break CSV serialization;
- a dedicated write/read unit test passes a row containing all current eligibility fields plus an undeclared future field, verifies successful output, verifies `core_transfer_eligible = 1`, and verifies the internal-only field is excluded;
- Python compile, the exact v5r config-only command, and all six unit tests pass on Windows. No model training was run on Windows.

Interpretation rule before Linux results return:

- v5r is an implementation-correctness repair, not an assumed performance improvement;
- Step 9 mixup remains an honest negative/neutral control because its standardized LR/L2 backend and post-augmentation class balancing make same-class convex interpolation nearly boundary-invariant;
- a positive Step 15 mixup claim is allowed only if v5r Phase 4 improves over the matching v5r Phase 3 in the fixed-test paired grouped-bootstrap audit.

`2026-07-11` Step 16G full Linux rerun completed and synchronized; Step 9/15/11/12/13 results audited:

- Data provenance conclusion:
  - the active Chinese strict pool is still derived from `market_item.xlsx` through Step 2/3/4; no external spam corpus or fabricated seller record was added;
  - Step 16B added `170` low-weight train-only weak positives and Step 16D added another `43`; these `213` rows are silver training support, not benchmark gold;
  - Step 16C moved only non-silver seller-connected components into validation/test, but Step 16F found that the resulting `80` evaluation positives are evidence-tiered: `22` direct/component primary, `14` soft primary/slice, and `44` secondary/sensitivity-only;
  - therefore the current Chinese benchmark is a component-safe internal, tiered-evidence benchmark, not an independently annotated external gold benchmark.
- Pre-Step16G failure that motivated the current rerun:
  - the Chinese train split was exactly balanced at `229 positive / 229 negative`;
  - `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` consequently produced `synthetic_row_count = 0` at every ratio/seed with `skipped_reason = positive_count_already_meets_configured_target`;
  - its 100pct predictions were therefore identical to the non-mixup model and could not be interpreted as a mixup result.
- Step 16G implementation:
  - policy: `schema/step16g_hard_negative_imbalance_policy.json`;
  - runner: `scripts/step16g_expand_hard_negative_train.py`;
  - selected `115` previously unsupervised Step 4 Chinese candidate pairs as low-weight train-only weak negatives;
  - selected tiers: `83 silver_semantic_low_structure_negative_imbalance` and `32 silver_ordinary_negative_imbalance`;
  - Step 15 independently maps those same 115 rows to `66 semantic_topic_not_controller` and `49 template_clone_not_controller`; Step16G tiers describe selection confidence, while Step15 evidence types describe curriculum role;
  - current train becomes `573 = 229 positive / 344 negative`; valid remains `120 = 30 / 90`; test remains `200 = 50 / 150`;
  - no uncertain row was converted, no existing supervised row was converted, all rows have Step 7 feature coverage, no selected pair lies inside a known positive component, and seller overlap across train/valid/test remains `0`;
  - additions are `benchmark_eligible = 0`, low-weight weak supervision. They do not increase evaluation-set size or gold-label credibility.
- Step 9 mixup contract repair:
  - 100pct runs now fail fast unless mixup creates at least one synthetic row;
  - mixup sources now require `training_sample_weight >= 0.55`, leaving `72` eligible higher-confidence positive sources rather than amplifying all `229` positives;
  - synthetic rows inherit the minimum parent weight instead of receiving full weight `1.0`;
  - a direct augmentation smoke test on the Step 16G boundary generated exactly `115` train-only synthetic positives, with weights from `0.55` to `1.0`.
- Step 12 comparison repair:
  - added non-mixup E5 LR/L2 100pct three-seed and seed-mean controls;
  - the formal mixup ablation is now `mixup 100pct` versus `non-mixup 100pct`, so support ratio and augmentation are no longer confounded;
  - new Step 12 outputs use the `step16g_imbalance_20260710` boundary identifier.
- Step 5 summary was refreshed non-destructively with `scripts/step5_refresh_frozen_summary.py`; the script rewrites only summary metadata and never regenerates frozen labels.
- Linux rerun completeness:
  - durable completion evidence is the formal Step16G summary plus the synchronized Step9/15/12/13 result bundles described below; the disposable full console log was removed on `2026-07-14` after its documentation reference was retired;
  - `reports/step16g_hard_negative_imbalance_summary.json` records the before/after split counts, all safety checks, input hashes and the expanded-freeze SHA-256 (`63ec5de569a538b82910cd7cbb3ea9c826699742349d3ec3d7688f3b95cbcfee`);
  - Step 9 contains `19 experiments x 4 ratios x 3 seeds = 228` complete runs and all referenced artifacts are present;
  - Step 15 contains `2 experiments x 5 phases x 3 seeds = 30` complete runs and all referenced artifacts are present;
  - Step 12 and Step 13 recorded input hashes for `69` files; every synchronized file exists and every SHA-256 matches;
  - the generated Step 13 JSON/CSV are synchronized, but `docs/STEP13_CONCEPT_DRIFT_AUDIT_STEP16G_IMBALANCE_VALIDATION_20260710.md` is still absent locally.
- Step 9 100pct mixup now executes as intended at the file-contract level:
  - each seed uses `229 positive / 344 negative` real Chinese support rows;
  - `72` positive parents satisfy `training_sample_weight >= 0.55`;
  - each seed creates `115` train-only synthetic positives and inherits the minimum parent weight;
  - formal same-ratio comparison: non-mixup AUC/AP `0.762667 / 0.556429`, mixup AUC/AP `0.756667 / 0.557633`;
  - paired grouped-bootstrap mixup-minus-non-mixup differences are AUC `-0.006000`, CI `[-0.031289, 0.021427]`, and AP `+0.001205`, CI `[-0.049545, 0.053483]`; neither supports a positive mixup claim.
- Step 9 mixup root-cause audit:
  - the backend is standardized linear LR/L2 with `class_weight = balanced` and `l2_penalty = 5.0`; same-class convex interpolation adds little new separating direction;
  - after class balancing and row-quality weights, synthetic augmentation changes effective positive weight share only from `0.439745` to `0.436759` for seed `20260320`, so count balancing does not create additional effective minority mass;
  - `106/115` to `105/115` synthetic rows per seed carry weight `0.55`; the augmentation is dominated by the same weak-positive support rather than new proof-level diversity;
  - the non-mixup/mixup coefficient cosine is `0.976459` and test-score Pearson correlation is `0.959852` for seed `20260320`; mixup mainly shifts score calibration and does not materially repair ranking;
  - `15/115` seed-20260320 synthetic rows contain fractional values in originally discrete/count features, confirming a smaller off-manifold feature-interpolation risk.
- Step 15 current fixed-test result:
  - non-domain-balanced seed-mean ensemble: ROC-AUC `0.866533`, AP `0.725220`;
  - domain-balanced seed-mean ensemble: ROC-AUC `0.865333`, AP `0.644989`;
  - non-domain-balanced versus raw E5 is supported by grouped bootstrap for both AUC (`+0.118533`, CI `[0.012272, 0.223274]`) and AP (`+0.182381`, CI `[0.019890, 0.324430]`);
  - this remains an internal fixed-test result, not an independent prospective-holdout confirmation.
- Step 15 Phase 4 mixup design defect found during the post-rerun audit:
  - the `316` positive parents are `116` English full-weight, `16` Chinese full-weight, and `184` Chinese weak-supervision rows;
  - `add_positive_mixup()` samples all positives without a minimum parent-weight or same-domain constraint, and generated rows omit `training_sample_weight`, so they default to full weight `1.0`;
  - depending on seed, `251` to `275` of `316` synthetic rows have at least one weak parent, `137` to `155` are cross-language-domain interpolations, and `302` to `307` contain fractional values in originally discrete/count features;
  - current Phase 4 performance therefore cannot be attributed cleanly to valid minority regularization until parent eligibility, inherited weight, same-domain/evidence controls, and a Phase3-vs-Phase4 paired audit are added.
- Domain-balanced regression root cause:
  - domain balancing uses raw row counts before applying `training_sample_weight` and treats `cross_domain_mixup` as a third domain;
  - Step16G added `115` low-weight Chinese negatives, increasing the real Chinese phase-4 row count from `429` to `544` even though their effective evidence weight is low;
  - after weighting, the real Chinese-domain share falls from about `0.24` pre-Step16G to about `0.21`, while cross-domain synthetic share rises from about `0.24` to `0.274-0.277`;
  - the domain-balanced scorer raises hard-negative scores (mean `0.460732` versus `0.316558`) and lowers direct/component-positive scores (`0.507233` versus `0.610381`), explaining its much lower AP;
  - on the top 20 ranked test pairs, non-domain-balanced retrieves `18` positives while domain-balanced retrieves `12`.
- Step 11 explicit six-summary audit remains discovery-limited: `212` unique seller sets contain `0` whole-cluster high-confidence decisions, `2` overlapping anchored cores, `8` partial anchors, `78` template-clone negatives, `111` semantic-topic negatives, and `13` uncertain sets.
- Step 13 generated findings require correction before publication: the non-domain-balanced Step 15 model, not the domain-balanced model, has the strongest current AUC/AP point estimate; domain-balanced AUC does significantly exceed raw E5, while its AP difference does not.
- Linux first-run hotfix: the initial Step 9 rerun reached synthetic CSV serialization and failed because `training_sample_weight` was present in each synthetic row but absent from `synthetic_train_fieldnames()`. The field contract is now synchronized. This was an output-schema error, not a completed model result. `scripts/run_step16g_full_rerun_20260710.sh` supports `SKIP_PRE_STEP9=1` to resume from Step 9 without rerunning completed Step5/7/calibration stages.

`2026-07-09` Step 5-19 output-overwrite audit and current-boundary result guard completed:

- Purpose: before rerunning the Step16C/E benchmark boundary on Linux, rechecked the Step 5 through Step 19 result-writing surface for stale-summary and accidental-overwrite risks.
- Confirmed current execution rule:
  - do not rerun generic `scripts/step5_freeze_silver_labels.py` for the active boundary, because the current Step16C/E frozen labels are already the intended start point;
  - Step 7 / Step 9 / Step 15 outputs are intentionally overwritten at their canonical current paths when rerun on the active boundary;
  - Step 11 cluster audit and Step 13 concept-drift audit must use explicit current output paths, not `reports/` globbing.
- Fixes applied:
  - `scripts/step12_statistical_robustness_audit.py` now defaults to `step16c_refreeze_20260709` output filenames and restricts the default model set to current-boundary models: raw semantic controls, Step 7 controls, Step 9 current candidates, and Step 15 v5 scorers.
  - legacy Step15 v1-v4 prediction files are no longer auto-loaded by the default Step 12 audit, preventing missing-pair errors and stale cross-boundary comparisons.
  - `schema/step12_statistical_robustness_policy.json` now points to the Step16C/E output names and lists the current extended metrics.
  - `schema/step15_evidence_type_policy.json` no longer carries old static baseline metric numbers; current baseline comparisons must be read from the new Step 12 outputs.
  - `schema/step11_clustering_policy.json` now records `current_validation_run_id = step16c_refreeze_validation_20260709`.
- Validation: JSON policy parsing and Python compile checks passed for the touched scripts.

`2026-07-09` Step 16F valid/test positive evidence re-audit completed:

- Active branch: `method/step16b-silver-positive-expansion`.
- Purpose: after the Step 16C/E component-safe refreeze, the Chinese validation/test splits have larger positive counts (`valid = 30`, `test = 50`), but those positives must be stratified by evidence strength before being used in paper claims.
- New audit script/output:
  - script: `scripts/step16f_valid_test_positive_reaudit.py`
  - CSV: `reports/step16f_valid_test_positive_reaudit.csv`
  - JSON: `reports/step16f_valid_test_positive_reaudit_summary.json`
  - note: `docs/STEP16F_VALID_TEST_POSITIVE_REAUDIT.md`
- Scope: only current `zh_target_strict` `valid` / `test` positives were rechecked. The script does not modify Step 5 labels.
- Audited positive rows: `80 = 30 valid / 50 test`.
- Evidence tiers:
  - `gold_direct_seller_contact`: `19`
  - `gold_direct_seller_contact_weaker_type`: `2`
  - `gold_component_anchor`: `1`
  - `strong_soft_structural_clone`: `11`
  - `component_or_contact_supported_soft_positive`: `3`
  - `moderate_soft_structural_positive`: `3`
  - `soft_product_data_clone_not_direct_identity`: `4`
  - `weak_component_or_semantic_positive`: `31`
  - `weak_soft_positive_needs_reaudit`: `6`
- Paper reporting buckets:
  - strict direct/component primary positives: `22`
  - broader soft primary/slice positives: `14`
  - secondary/sensitivity-only positives: `44`
- Risk flags:
  - `product_data_email_not_seller_identity`: `4`
  - `contact_context_also_mentions_data_product`: `14`
  - `direct_contact_not_in_pair_feature`: `3`
  - weak or non-direct-identity soft evidence remains common.
- Interpretation: the current valid/test benchmark is usable only with tiered reporting. It is not defensible to describe all `80` positives as direct identity-anchor gold labels. The strongest paper claim should report the strict direct/component positive slice separately; softer structural/clone positives should be reported as soft or sensitivity slices.

`2026-07-09` Step 16C / Step 16E component-safe benchmark refreeze and train-balance repair applied:

- Active branch: `method/step16b-silver-positive-expansion`.
- Motivation: the Step 16D state balanced `zh_train`, but left `zh_valid = 14 positive / 67 negative` and `zh_test = 21 positive / 85 negative`. That was still too small and imbalanced for paper-grade evaluation. The current boundary fixes the evaluation splits first, then restores train balance with low-weight train-only weak negatives.
- Step 16C applied a component-safe gold valid/test refreeze:
  - script: `scripts/step16c_plan_gold_valid_test_refreeze.py`
  - plan: `reports/step16c_gold_valid_test_refreeze_plan.csv`
  - summary: `reports/step16c_gold_valid_test_refreeze_plan_summary.json`
  - backup before application: `reports/step5_zh_target_strict_frozen_silver_labels.pre_step16c_gold_valid_test_refreeze_20260709.csv`
  - target valid split: `120 = 30 positive / 90 negative`
  - target test split: `200 = 50 positive / 150 negative`
  - no silver rows were moved into valid/test;
  - whole seller-connected components were moved;
  - train / valid / test seller-overlap counts are `0`.
- Step 16E then restored training balance after the refreeze:
  - policy: `schema/step16e_relaxed_silver_negative_balance_policy.json`
  - script: `scripts/step16e_relaxed_silver_negative_balance.py`
  - selected train-only weak negatives: `43`
  - selected existing uncertain rows converted to train-only silver negative: `3`
  - existing reviewed positives converted: `0`
  - selected valid/test seller overlap: `0`
  - all selected rows have Step 7 pair-feature coverage.
- Current Chinese `zh_target_strict` after Step 16C/E:
  - total frozen rows: `1144`
  - label counts: `315 positive / 469 negative / 360 uncertain`
  - binary supervision rows: `778`
  - train: `458 = 229 positive / 229 negative`
  - valid: `120 = 30 positive / 90 negative`
  - test: `200 = 50 positive / 150 negative`
  - audit-only positives: `6`
  - train / valid / test seller-overlap counts: `0 / 0 / 0`.
- Current split quality:
  - valid/test are gold/original only: `silver_positive = 0`, `silver_negative = 0`, `benchmark_eligible=0 count = 0`.
  - train contains weak supervision only as training support: `213` silver positives and `43` silver negatives.
  - all binary supervision rows have Step 7 feature coverage and Step 15 evidence-type coverage.
- Current evidence-type distribution:
  - train positives: `57 direct_identifier`, `29 component_anchor`, `143 style_structural_soft`
  - train negatives: `129 ordinary`, `61 template_clone`, `31 semantic_topic`, `8 public_contact_or_url_noise`
  - valid positives: `4 direct_identifier`, `26 style_structural_soft`
  - valid negatives: `46 ordinary`, `25 template_clone`, `16 semantic_topic`, `3 public_contact_or_url_noise`
  - test positives: `21 direct_identifier`, `1 component_anchor`, `28 style_structural_soft`
  - test negatives: `112 ordinary`, `22 semantic_topic`, `10 template_clone`, `6 public_contact_or_url_noise`
- Scientific interpretation: this is the current paper-oriented internal benchmark boundary. It is stronger than the prior fixed `zh_test = 106` boundary because test positives increase from `21` to `50`, and valid positives increase from `14` to `30`. It is still not an external prospective holdout; paper text must call it a component-safe internal refreeze. All Step 7 / Step 9 / Step 15 / Step 12 / Step 11 results before this boundary are now stale and must be rerun.

`2026-07-09` Step 16D relaxed silver positive train-only top-up applied:

- Active branch: `method/step16b-silver-positive-expansion`.
- Status: superseded as the final active data boundary by the Step 16C/E refreeze and balance repair above. The Step 16D outputs remain part of the data-construction provenance.
- Motivation: after Step 16B, the Chinese training split still had fewer positives than negatives (`231 positive / 274 negative`). The requested expansion priority is positive sample volume; negatives should only be expanded when needed for class balance.
- New policy/script:
  - `schema/step16d_relaxed_silver_positive_topup_policy.json`
  - `scripts/step16d_relaxed_silver_positive_topup.py`
- Step 16D adds only weak `positive` rows and stops exactly when the current Chinese training split is balanced. It does not add negative rows in this pass.
- Safety constraints applied:
  - no existing reviewed negative rows were converted;
  - no `valid` / `test` rows were modified;
  - any pair sharing a seller with current `zh_valid` / `zh_test` supervision was excluded;
  - any pair sharing a seller with the Step 16C planned valid/test refreeze set was also excluded;
  - every selected row already has a Step 7 pair-feature row;
  - all added rows are `split_name = train`, `silver_train_only = 1`, `benchmark_eligible = 0`, and low-weight weak supervision.
- Applied Step 16D top-up:
  - candidate count under relaxed positive rules: `63`
  - selected train-only silver positives: `43`
  - selected existing uncertain rows converted to train-only silver positive: `16`
  - selected existing reviewed negatives converted: `0`
  - protected valid/test seller overlap: `0`
- Step 16D silver composition:
  - `silver_rank_structural_relaxed = 30`, training weight `0.18`
  - `silver_template_structural_relaxed = 11`, training weight `0.20`
  - `silver_high_similarity_relaxed = 2`, training weight `0.24`
- Current Chinese `zh_target_strict` after Step 16D:
  - total frozen rows: `1104`
  - label counts: `315 positive / 426 negative / 363 uncertain`
  - supervision rows: `735`
  - train: `548 = 274 positive / 274 negative`
  - valid: `81 = 14 positive / 67 negative`
  - test: `106 = 21 positive / 85 negative`
  - audit-only positives: `6`
  - train / valid / test seller-overlap counts remain `0`.
- Current Chinese train positive composition:
  - original/gold positives: `61`
  - Step 16B silver positives: `170`
  - Step 16D relaxed silver positives: `43`
  - total train positives: `274`
- Updated outputs:
  - `reports/step16d_relaxed_silver_positive_topup_candidates.csv`
  - `reports/step16d_relaxed_silver_positive_topup_training_pairs.csv`
  - `reports/step16d_relaxed_silver_positive_topup_summary.json`
  - `reports/step5_zh_target_strict_frozen_silver_labels.csv`
  - `reports/step5_frozen_silver_summary.json`
  - `reports/step15_evidence_type_labels.zh_target_strict.csv`
  - `reports/step15_evidence_type_label_summary.json`
- Scientific interpretation: Step 16D is a low-weight weak-supervision top-up for target-domain training balance. It is not gold truth and must not be used as validation/test evidence. Any final paper claim must report gold-only `zh_valid` / `zh_test` metrics, and should include an ablation comparing no-silver, Step16B, and Step16B+Step16D training.

`2026-07-06` Step 16B silver positive train-only expansion added:

- Active branch: `method/step16b-silver-positive-expansion`.
- Motivation: strict Chinese proof-level positive labels were too scarce for stable target-domain training. The project now separates strict benchmark labels from weak training support.
- New policy/script:
  - `schema/step16b_silver_positive_expansion_policy.json`
  - `scripts/step16b_expand_silver_positive_train.py`
  - design note: `docs/STEP16B_SILVER_POSITIVE_EXPANSION.md`
- Step 16B does not invent sellers or item text. It selects weakly supervised Chinese positive pairs from the existing `zh_target_strict` Step 4 / Step 7 pair universe.
- Safety constraints applied:
  - no existing reviewed negative rows were converted;
  - no existing `valid` / `test` rows were modified;
  - any pair sharing a seller with current `zh_valid` or `zh_test` supervision was excluded;
  - every added row already has a Step 7 pair-feature row;
  - all added rows are `split_name = train`, `silver_train_only = 1`, and `benchmark_eligible = 0`.
- Applied expansion:
  - Chinese train positives before: `61`
  - Chinese train positives after: `231`
  - added silver train-only positives: `170`
  - Chinese valid positives unchanged: `14`
  - Chinese test positives unchanged: `21`
  - train / valid / test seller-overlap counts remain `0`.
- Silver composition:
  - `silver_template_structural = 85`, training weight `0.40`
  - `silver_direct_or_contact = 56`, training weight `0.55`
  - `silver_component_closure = 29`, training weight `0.25`
- Training scripts were updated so weak labels are not treated as equal-strength gold labels:
  - `scripts/step7_train_baseline_models.py` now propagates `training_sample_weight` from Step 5 labels and multiplies LightGBM class-balanced weights by this row multiplier.
  - `scripts/step9_run_few_shot_adaptation.py` now applies the same row multiplier to logistic / residual logistic backends and passes row weights through the Step 9 LightGBM backend.
  - `scripts/step9_run_few_shot_adaptation.py` now backs up but does not merge old Step 9 summaries when the data-context fingerprint changes. This prevents pre-Step16B and post-Step16B runs from being mixed inside one summary file.
  - `scripts/step15_train_incremental_hard_negative.py` now multiplies Step 15 identity-loss weights by `training_sample_weight`.
- Updated outputs:
  - `reports/step16b_silver_positive_candidate_pairs.csv`
  - `reports/step16b_silver_positive_training_pairs.csv`
  - `reports/step16b_silver_positive_expansion_summary.json`
  - `reports/step5_zh_target_strict_frozen_silver_labels.csv`
  - `reports/step5_frozen_silver_summary.json`
  - `reports/step15_evidence_type_labels.zh_target_strict.csv`
  - `reports/step15_evidence_type_label_summary.json`
- Scientific interpretation: these rows are weak training support, not gold benchmark truth. Paper text must describe them as `silver_train_only` auxiliary positives. Final evaluation should still report the unchanged fixed `zh_valid` / `zh_test` gold benchmark unless a separate future benchmark is explicitly constructed.

`2026-07-04` Step 7 / Step 9 / Step 15 rerun with extended ranking metrics synchronized back:

- Active branch: `method/add-ranking-evaluation-metrics`.
- Code change scope is metric/reporting only. Training data, model features, sampling, thresholds, and Step 15 v5 experiment definitions were not changed.
- New metrics are now emitted by the shared Step 7 evaluator and inherited by Step 9 and Step 15 run records:
  - `pr_auc`: same numerical definition as `average_precision` under the current binary pair-ranking setup.
  - `map`: same global pair-ranking definition as `average_precision`; no seller-group query partition is applied.
  - `mrr`: reciprocal rank of the first positive pair in the global ranked list.
  - `f1`: thresholded binary F1, now also included in Step 9 / Step 15 aggregate summaries.
- Synchronization completeness checks:
  - Step 7 summary contains `17` experiments; all valid/test/zh-test metric blocks include `pr_auc`, `map`, `mrr`, and `f1`.
  - Step 9 few-shot summary contains `19` experiments and `228` runs (`19` experiments x `4` ratios x `3` seeds); all run metrics and aggregate-by-ratio metrics include the new fields.
  - Step 15 v5 summary contains `30` runs (`2` experiments x `5` phases x `3` seeds); all run metrics include the new fields.
  - Step 12 v5 grouped bootstrap now reports `roc_auc`, `average_precision`, `pr_auc`, `map`, and `mrr` for model metrics and paired comparisons.
- Current fixed-test Step 7 zero-shot readings remain weak relative to later methods:
  - best Step 7 zh zero-shot ROC-AUC is `0.631373` from `core_zero_shot_default_no_structural`, with AP/PR-AUC/MAP `0.290316` and MRR `0.333333`.
  - `core_zero_shot_bge_m3` has ROC-AUC `0.601681`, AP/PR-AUC/MAP `0.448761`, MRR `1.000000`, F1 `0.342105`, and recall `0.619048`.
  - `core_zero_shot_multilingual_e5_large` remains collapsed/weak as a fused Step 7 model: ROC-AUC `0.550140`, AP/PR-AUC/MAP `0.384493`.
- Current Step 9 fixed-test summary:
  - strongest clean ranking candidate by ROC-AUC remains `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` at `100pct`: mean ROC-AUC `0.841830`, AP/PR-AUC/MAP `0.587521`, F1 `0.375838`, MRR `1.000000`.
  - strongest operational AP candidate remains `identifier_augmented_few_shot_default_lr_l2` at `50pct`: mean ROC-AUC `0.779272`, AP/PR-AUC/MAP `0.652021`, F1 `0.545809`, MRR `1.000000`. This remains an operational identifier control, not the clean scientific mainline.
  - Step 9 calibration control `core_calibrated_bge_m3` now emits the extended metrics. Its fixed `zh_test` ranking remains ROC-AUC `0.601681`, AP/PR-AUC/MAP `0.448761`, MRR `1.000000`; at threshold `0.5`, F1 is `0.0`, so it is not a useful thresholded detector.
- Current Step 15 v5 fixed-test summary:
  - `step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean` remains the current clean main scorer: ROC-AUC `0.913725`, AP/PR-AUC/MAP `0.738951`, MRR `1.000000`.
  - Step 12 grouped bootstrap for this scorer: ROC-AUC mean `0.913844`, CI `[0.837521, 0.968700]`; AP/PR-AUC/MAP mean `0.733387`, CI `[0.480227, 0.894451]`; MRR mean `0.998106`, CI `[1.000000, 1.000000]`.
  - The non-domain-balanced v5 phase4 seed mean is lower: ROC-AUC `0.904202`, AP/PR-AUC/MAP `0.701809`, MRR `1.000000`.
- Current paired-bootstrap interpretation is unchanged by adding metric aliases:
  - Step 15 v5 domain-balanced vs Step 9 E5 mixup100 supports a positive ROC-AUC difference: observed diff `+0.071709`, CI `[0.006751, 0.152423]`.
  - The AP/PR-AUC/MAP difference vs Step 9 mixup100 remains positive but uncertainty-bounded: observed diff `+0.149956`, CI `[-0.032434, 0.304387]`.
  - The ROC-AUC and AP/PR-AUC/MAP differences vs raw E5 remain positive in point estimate but not statistically locked because grouped-bootstrap CIs cross zero.
  - MRR is not discriminative for the main models because several methods rank at least one positive pair first; it should be reported as a supplementary ranking diagnostic only.

`2026-06-03` Step 15 v5 frozen for Step 11 graph validation:

- Active branch: `method/step15-v5-step11-validation`.
- Step 15 v5 is now frozen as a pairwise scorer. No further Step 15 tuning should be done unless Step 11 cluster-level validation exposes a specific failure mode.
- Step 11 policy and runner now support explicit frozen Step 15 MLP ensemble scoring through `--scorer-family step15`. The current primary Step 15 graph-validation candidate is:
  - experiment: `step15_v5_identity_only_curriculum_domain_balanced_public_noise_weighted_strong`
  - phase: `phase4_add_positive_pair_mixup`
  - seed ensemble: `20260320 / 20260321 / 20260322`
  - output token: `step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean`
- Step 11 computes the Step 15 ensemble threshold from the three seed-aligned `zh_valid_predictions` files, rather than averaging per-seed thresholds. This keeps the graph threshold tied to the frozen validation split and avoids using `zh_test` for graph-threshold selection.
- The Step 15 policy now defaults to a v5 public-contact/URL noise stress configuration and writes separate v5 outputs, so v2/v3/v4 artifacts are preserved and not overwritten:
  - `reports/step15_v5_public_noise_weighted_summary.json`
  - `reports/step15_v5_slice_level_audit.json`
  - `reports/step15_v5_slice_level_audit.csv`
  - Step 12 default outputs now use `reports/step12_v5_statistical_robustness_*_20260603.*`
- Code changes:
  - `scripts/step15_train_incremental_hard_negative.py` now supports per-evidence-type identity-loss multipliers.
  - The multiplier is applied only to train rows and is mean-normalized to avoid destabilizing the optimizer.
  - The runner also supports train-only negative mixup for selected negative evidence types. This is available for v4 conservative stress controls but is not the current clean default because it lowers overall AP too much.
  - `scripts/step12_statistical_robustness_audit.py` now registers v3/v4/v5 Step 15 prediction ensembles and paired comparisons.
  - `scripts/step15_slice_level_audit.py` now includes v4/v5 defaults.
- Linux Step 15 v5 slice audit and corrected Step 12 v5 grouped bootstrap have been synchronized back. Step 12 v5 now uses `audit_version = step12_v5_statistical_robustness_zh_test_20260603` and includes explicit paired comparisons against raw E5 and Step 9 E5 mixup 100pct.
- Public-contact/URL slice findings on the six fixed `zh_test` negative rows:
  - v2 clean primary: previous slice audit mean `0.503034`, max `0.876515`.
  - v2 domain-balanced: previous slice audit mean `0.620560`, max `0.958353`.
  - v3 domain-balanced public-noise weighting lowered the v2 domain-balanced risk to approximately mean `0.532494`, max `0.867701`, while keeping strong overall point estimates.
  - v4 negative-mixup variants lowered public-noise scores further but caused substantial clean metric degradation; v4 is retained only as a conservative robustness/control branch.
  - v5 clean public-noise weighted strong, without negative mixup, lowers the slice to mean `0.474575`, max `0.768054`; fixed-test seed-mean ROC-AUC `0.904202`, AP `0.701809`.
  - v5 domain-balanced weighted strong is the best balanced candidate: fixed-test seed-mean ROC-AUC `0.913725`, AP `0.738951`, public-noise mean `0.499258`, max `0.856277`.
- Corrected Step 12 v5 paired-bootstrap interpretation:
  - v5 domain-balanced vs Step 9 E5 mixup 100pct supports a positive ROC-AUC difference: observed diff `+0.071709`, CI `[0.006751, 0.152423]`, bootstrap sign p `0.031200`.
  - The corresponding AP difference is positive but not statistically locked: observed diff `+0.149956`, CI `[-0.032434, 0.304387]`.
  - v5 domain-balanced vs raw E5 is stronger in point estimate for both ROC-AUC and AP, but both CIs still cross 0.
  - v5 domain-balanced vs v2 domain-balanced remains only a point-estimate improvement: ROC-AUC diff `+0.012325`, CI `[-0.012943, 0.062131]`; AP diff `+0.024580`, CI `[-0.057586, 0.168206]`.
- Scientific interpretation before graph validation: v5 materially reduces the public-contact/URL false-positive slice compared with the dangerous v2 domain-balanced failure mode and gives the strongest current fixed-test point estimate. It can support a cautious claim of improved ROC-AUC over the Step 9 E5 mixup baseline, but it still cannot be described as a statistically robust improvement over raw E5 or v2 Step 15 across all metrics.
- Step 11 graph validation has now been rerun on Linux and synchronized back. The validation audit was generated with explicit `--summary` allow-list inputs, not by globbing `reports/`, and the audit records `summary_selection_mode = explicit`.
- The six accepted Step 11 validation summaries are:
  - `step11_step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean_clustering_summary.json`
  - `step11_step15_v5_public_noise_weighted_strong_phase4_seed_mean_clustering_summary.json`
  - `step11_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260320_clustering_summary.json`
  - `step11_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260321_clustering_summary.json`
  - `step11_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260322_clustering_summary.json`
  - `step11_core_zero_shot_bge_m3_clustering_summary.json`
- Synchronization/integrity checks for the Step 11 validation are clean: all six summaries exist, every summary has `acceptance_checks_failed = []`, and every path referenced by each summary's `output_paths` exists locally.
- Primary Step 11 graph outputs:
  - `step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean`: selected threshold `0.925494`; `226` pre-filter edges, `190` after relation reliability, `187` after reciprocal top-k, `26` after shared-neighbor pruning; `6` primary clusters, largest size `5`.
  - `step15_v5_public_noise_weighted_strong_phase4_seed_mean`: selected threshold `0.598927`; `677` pre-filter edges, `572` after relation reliability, `560` after reciprocal top-k, `112` after shared-neighbor pruning; `21` primary clusters, largest size `7`.
  - `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260320`: selected threshold `0.720543`; `408` pre-filter edges, `332` after relation reliability, `321` after reciprocal top-k, `70` after shared-neighbor pruning; `11` primary clusters, largest size `7`.
  - `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260321`: selected threshold `0.792627`; `203` pre-filter edges, `172` after relation reliability, `167` after reciprocal top-k, `35` after shared-neighbor pruning; `7` primary clusters, largest size `7`.
  - `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260322`: selected threshold `0.720528`; `345` pre-filter edges, `267` after relation reliability, `258` after reciprocal top-k, `61` after shared-neighbor pruning; `12` primary clusters, largest size `7`.
  - `core_zero_shot_bge_m3`: selected threshold `0.483444`; `1425` pre-filter edges, `1165` after relation reliability, `1005` after reciprocal top-k, `197` after shared-neighbor pruning; `43` primary clusters, largest size `7`.
- The explicit cluster-level validation audit is:
  - `reports/step11_cluster_level_audit.step15_v5_validation_20260603.csv`
  - `reports/step11_cluster_level_audit.step15_v5_validation_20260603.json`
  - `input_summary_count = 6`, `primary_cluster_count_total = 100`, `unique_cluster_set_count = 79`.
  - decision counts: `same_controller_high_confidence = 0`, `same_controller_core_with_possible_expansion = 0`, `partial_anchor = 5`, `template_clone_not_controller = 33`, `semantic_topic_not_controller = 35`, `uncertain = 6`.
  - confidence counts: `low = 74`, `medium = 5`.
- Best-scorer distribution in the unique cluster audit:
  - `core_zero_shot_bge_m3`: `43` unique cluster sets; `3` partial anchors, `26` template-clone non-controller sets, `11` semantic-topic non-controller sets, `3` uncertain.
  - `step15_v5_public_noise_weighted_strong_phase4_seed_mean`: `17` unique cluster sets; `1` partial anchor, `2` template-clone non-controller sets, `12` semantic-topic non-controller sets, `2` uncertain.
  - `step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean`: `3` unique cluster sets; `1` partial anchor, `1` semantic-topic non-controller set, `1` uncertain.
  - Step 9 E5 mixup 100pct seeds together contribute `16` unique cluster sets, all audited as template-clone or semantic-topic non-controller sets.
- Step 11 validation conclusion: Step 15 v5 domain-balanced is useful as a conservative graph triage/noise-compression scorer, because it produces the smallest retained graph while preserving one partial-anchor candidate. It does not yet prove better same-controller discovery quality: the validation audit found no high-confidence same-controller cluster and no same-controller core-with-expansion cluster for any candidate. Step 15 v5 must therefore be reported as a pairwise robustness and graph-triage improvement, not as proof-level identity-cluster discovery.
- Cross-step design audit after Step 7 through Step 15:
  - No new evidence was found of a current result-overwrite bug in the active Step 9/Step 11/Step 12/Step 15 outputs. New Step 15 v5 outputs use separate names, and the Step 11 validation audit uses explicit summary inputs.
  - No new validation/test leakage was found in the active validation path. Step 15 graph thresholds are selected from frozen `zh_valid` seed-ensemble predictions, not from `zh_test`; Step 11 cluster decisions are audit outputs and are not fed back into Step 5 labels.
  - The main remaining Step 7 design limitation is scientific, not a sync bug: LightGBM fusion remains shallow/collapsed relative to raw semantic scoring on the Chinese target test, so raw semantic baselines, especially raw E5, must remain first-class controls in every claim.
  - The Step 9 "few-shot" terminology remains potentially misleading. Ratio `1.0` means all frozen Chinese train support is used; it should be described as target-domain support-ratio adaptation. Positive-pair mixup is training-only minority regularization and must not be described as new ground-truth labels.
  - The Step 11 validation is correct as an allow-list audit, but future Step 13/table-generation code must not fall back to an older `step11_current_manifest_*.json` or a `reports/` glob. Any follow-up Step 13 run should explicitly consume this Step 11 validation audit or a newly generated validation manifest.
  - Step 15 v5 is not a successful standard multi-task-learning proof. Its active value is identity-only curriculum/reweighting with evidence-type diagnostics, plus public-contact/URL-noise stress control. Any paper text should avoid claiming that the evidence-type auxiliary head itself solved the task.
- `2026-06-04` methodology-hygiene patch after the Step 2-15 design review:
  - `schema/step9_training_policy.json` no longer presents `zh_test_metrics` as the Step 9 model-selection metric. Held-out Chinese test metrics are now recorded under `final_test_reporting` and explicitly marked as reporting-only.
  - `schema/step11_clustering_policy.json` now defaults to the frozen Step 15 scorer instead of `auto`; the old dynamic auto selector remains available only as diagnostic exploration and is marked `diagnostic_only_not_publication_model_selection`.
  - The current Step 11 publication-validation summaries are now listed in policy under `scorer_selection.publication_validation.current_validation_summaries`; publication cluster audits must use this explicit allow-list rather than globbing `reports/` or relying on `--scorer-family auto`.
  - `scripts/step13_concept_drift_audit.py` now accepts explicit `--step11-manifest` and `--step11-audit` inputs. It no longer silently consumes older `step11_current_manifest_*.json` / `step11_cluster_level_audit.current_*.json` files unless `--allow-step11-auto-discovery` is intentionally passed.
  - `schema/step15_evidence_type_policy.json` now separates historical Step 11 audit files into `diagnostic_dependencies_only`; Step 11 audit outputs are not Step 15 identity-training ground truth.
  - Step 15 phase4 now documents that `use_negative_mixup` is only a phase capability flag; negative mixup is applied only when the selected experiment also enables `experiments.<name>.negative_mixup.enabled`.
- `2026-06-06` Step 13 was regenerated against the latest explicit Step 11 v5 validation audit:
  - runner: `scripts/step13_concept_drift_audit.py`
  - explicit Step 11 audit input: `reports/step11_cluster_level_audit.step15_v5_validation_20260603.json`
  - outputs:
    - `reports/step13_concept_drift_audit.step15_v5_validation_20260606.json`
    - `reports/step13_concept_drift_audit.step15_v5_validation_20260606.csv`
    - `docs/STEP13_CONCEPT_DRIFT_AUDIT_STEP15_V5_VALIDATION_20260606.md`
  - integrity: the new Step 13 summary records `step11_selection_mode = explicit` and `step11_audit_path = reports\step11_cluster_level_audit.step15_v5_validation_20260603.json`; no `current_*` auto-discovery or `reports/` glob was used.
  - Step 13 now includes synchronized Step 15 v5 prediction ensembles in the fixed `zh_test` slice audit. The current all-test readings in the Step 13 table are:
    - raw E5: ROC-AUC `0.806723`, AP `0.520573`
    - Step 9 E5 LR/L2 positive-pair mixup 100pct seed mean: ROC-AUC `0.842017`, AP `0.588995`
    - Step 15 v5 domain-balanced public-noise-weighted phase4 seed mean: ROC-AUC `0.913725`, AP `0.738951`
  - Corrected interpretation: Step 9 mixup 100pct is the strongest current Step 9 minority-regularization baseline; Step 15 v5 domain-balanced is the strongest current fixed-test point estimate. Step 12 v5 supports Step 15 v5's ROC-AUC improvement over Step 9 mixup100, but not a statistically locked AP improvement or a robust improvement over raw E5.
  - Concept-drift diagnosis remains: marginal EN-to-ZH shifts are strongest in digit/repetition/punctuation gap features; high-semantic negatives are not uniformly inflated in ZH under the EN-negative q90 E5 threshold; the latest explicit Step 11 validation audit is still dominated by template/topic non-controller evidence (`68` template/topic clusters vs `0` anchored same-controller cores).

`2026-06-02` Step 15 v2 curriculum/slice-audit branch prepared:

- Active branch: `method/step15-v2-curriculum-audit`.
- The first-pass Step 15 result remains preserved under the original `step15_e5_*` experiment namespace and original `reports/step15_incremental_hard_negative_summary.json`.
- V2 experiments now use a separate `step15_v2_*` namespace and write the main training summary to `reports/step15_v2_incremental_hard_negative_summary.json`, avoiding old/new result overwrites.
- The Step 15 runner now refuses to run legacy `step15_e5_*` experiments by default because those names write the original first-pass artifact/prediction paths. An intentional legacy rerun must pass `--allow-legacy-output-overwrite`.
- Step 12 v2 robustness outputs now default to:
  - `reports/step12_v2_statistical_robustness_zh_test_20260602.json`
  - `reports/step12_v2_statistical_robustness_model_metrics_20260602.csv`
  - `reports/step12_v2_statistical_robustness_paired_comparisons_20260602.csv`
- Linux Step 15 v2 and Step 12 v2 outputs are now synchronized back. The full Step 15 v2 matrix completed `135` runs (`9` experiments x `5` phases x `3` seeds), with all expected artifacts and fixed `zh_valid` / `zh_test` predictions present.
- The strongest clean Step 15 v2 point estimate is `step15_v2_domain_balanced_phase4_seed_mean`: ROC-AUC `0.901401`, AP `0.714371` on the fixed `zh_test = 106` split.
- The clean primary `step15_v2_identity_from_scratch_phase4_seed_mean` improves over raw E5 in point estimate, but Step 12 v2 paired bootstrap still does not support a statistically robust positive difference: ROC-AUC diff `+0.082913`, CI `[-0.076627, 0.250000]`; AP diff `+0.178725`, CI `[-0.176207, 0.466066]`.
- Against Step 9 E5 positive-pair mixup 100pct, the clean primary remains uncertainty-bounded: ROC-AUC diff `+0.047619`, CI `[-0.037055, 0.125878]`; AP diff `+0.110304`, CI `[-0.086075, 0.273455]`.
- Slice audit shows the remaining high-risk failure mode is `public_contact_or_url_noise`: the clean domain-balanced score mean is `0.620560` with max `0.958353` on six negative rows, so Step 15 remains a ranking/audit candidate generator rather than proof-level identity evidence.
- Step 15 v2 changes:
  - identity-only curriculum is promoted to the clean primary candidate because the first-pass evidence-type multitask head was not supported by fixed-test results;
  - multitask remains as an ablation, not the main claim;
  - true warm-start curriculum is added, with phase `n+1` initialized from phase `n` under a shared final-phase standardizer;
  - domain-balanced, target-only, and source-only controls are added to separate source-domain scale from target-domain adaptation;
  - mixup scope controls are added for `target_train_only` and `same_evidence_type_only`;
  - `scripts/step15_slice_level_audit.py` adds fixed `zh_test` evidence-type/review-stratum slice diagnostics.
- V2 still obeys the hard rules: Step 5 labels remain frozen, `zh_train/zh_valid/zh_test` are not mixed, uncertain rows are not used for binary identity training, synthetic rows remain train-only, and Step 11 cluster decisions are not used as ground truth.
- Local smoke validation completed for:
  - `step15_v2_identity_only_curriculum_target_only / phase0 / seed 20260320`;
  - `step15_v2_identity_only_curriculum_warm_start / phase0+phase1 / seed 20260320`, confirming `initialization = warm_start` for phase1 and `standardizer_source = warm_start_final_phase_train`.
- Smoke artifacts were deleted after validation so they cannot be mistaken for complete Linux results.
- Required next action: keep Step 15 v2 as the current best clean point-estimate method, but do not route it into Step 11 as a proof-level discovery scorer until a Step 11 integration plan explicitly addresses the `public_contact_or_url_noise` false-positive slice.

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
- Step 13 was regenerated on `2026-05-18` against the strict `20260517` Step 11 audit plus both mixup 50pct and 100pct prediction ensembles. This record is now superseded for current reporting by the explicit Step 15 v5 validation audit rerun on `2026-06-06`.

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
- Step 12 fixed-test robustness audit now exists: `scripts/step12_statistical_robustness_audit.py` with policy `schema/step12_statistical_robustness_policy.json`; current outputs are `reports/step12_v2_statistical_robustness_zh_test_20260602.json`, `reports/step12_v2_statistical_robustness_model_metrics_20260602.csv`, and `reports/step12_v2_statistical_robustness_paired_comparisons_20260602.csv`
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
- claiming the current clean Step 15 v2 line statistically robustly beats raw E5 semantic ranking on the fixed `zh_test`; Step 12 v2 finds `step15_v2_identity_from_scratch_phase4_seed_mean` vs raw E5 ROC-AUC diff `+0.082913` with grouped 95% CI `[-0.076627, 0.250000]`, and AP diff `+0.178725` with CI `[-0.176207, 0.466066]`
- claiming Step 15 v5 has produced proof-level same-controller clusters; the latest explicit Step 11 validation audit still has `0` high-confidence same-controller clusters and `0` same-controller cores with expansion

## Recommended Next Actions

1. Use `reports/step12_v5_statistical_robustness_zh_test_20260603.json/csv` and `reports/step13_concept_drift_audit.step15_v5_validation_20260606.json/csv` as the current pairwise/statistical/concept-drift evidence set.
2. Use `reports/step11_cluster_level_audit.step15_v5_validation_20260603.json/csv` as the current graph-validation audit. Do not glob `reports/step11_*`; publication cluster audits must use the explicit allow-list in `schema/step11_clustering_policy.json`.
3. Keep Step 15 v5 frozen. Do not tune Step 15 further against the fixed `zh_test`; any new method branch should be separated from this frozen validation branch.
4. Current paper wording should frame Step 15 v5 as hard-negative concept-drift mitigation and graph triage/noise compression. It can cautiously claim a Step12-supported ROC-AUC improvement over Step9 mixup100, but not a proof-level same-controller cluster discovery result.
5. If publication requires stronger identity-discovery claims, the next evidence source must be new raw/OCR/source fields or external corroborating evidence; the current Chinese item-level text extraction has exhausted unreviewed direct-token pairs under the conservative seller-facing direct-identity standard.

## Step25-v2 Pair-Local Copy and Missingness Diagnostic (2026-07-17)

Implementation status:

- branch: `method/step25-v2-pair-local-copy-diagnostic`
- policy: `schema/step25_v2_pair_local_copy_diagnostic_policy.json`
- Linux runner: `scripts/run_step25_v2_pair_local_copy_linux_20260717.sh`
- detailed design: `docs/STEP25_V2_PAIR_LOCAL_COPY_MISSINGNESS_DIAGNOSTIC_20260717.zh.md`
- Linux execution and synchronization are complete; the closed manifest binds `19` payloads (`44,963,636` bytes) and `17` producer files

Scientific purpose:

- keep the frozen Step25-v1 negative result unchanged
- test whether the v1 global template catalog missed copied text supported only by the current seller pair
- test whether the v1 fixed-zero treatment of insufficient cleaned style confused missingness with genuine style dissimilarity
- isolate detector effects with matched P0/P2 reliability masks and fold-train-only median imputation

Preregistered controls:

- `P0`: raw authorship style under the pair-local reliability mask
- `P1`: frozen Step25-v1 global-clean style under the global/local reliability intersection
- `P2`: pair-local-clean style under the same reliability mask as P0
- `P3`: pair-local-clean style with explicit raw-style fallback
- `P4`: reliable-pair-only P0/P2 sensitivity without refitting

Boundary and claim discipline:

- only canonical English/Chinese `train` is read; valid/test access is forbidden
- detector thresholds are fixed before numerical execution and cannot be searched on D0
- no labels, evidence types, model errors or prediction scores enter copy detection or feature construction
- no missing style value is encoded as cosine zero
- Step25-v2 is hypothesis-informed and retrospective, so `d1_candidate_eligible`, `publication_promotion_eligible`, and `step11_or_step17_entry_allowed` are hard false regardless of results
- even a complete mechanism-gate pass only justifies a future preregistered D1 replication

Expected Linux output root:

- `reports/step25_template_decontaminated_authorship/v2_pair_local_diagnostic_20260717/`

Completed result:

- Chinese target grouped-OOF P0/P2/P3 AP: `0.704847 / 0.670692 / 0.737365`
- English grouped-OOF P0/P2 AP: `0.468210 / 0.251926`
- P2 minus P0 target AP: `-0.034155`; grouped-bootstrap confidence interval crosses zero
- pair-local detector found copied spans in `299/573` Chinese rows and retained `454/573` reliable local-style rows
- only `3/8` mechanism gates passed; `d1_candidate_eligible=false`

The synchronized directory passed its closed `step25_v2_sync_manifest.json` audit. Step25-v2 is frozen as a mixed/negative mechanism result and cannot enter Step11/17. Its useful finding is narrower: pair-local copy detection has diagnostic value, but replacing raw style with cleaned style destroys too much information. Step25-v3 therefore preserves raw style and constrains copy evidence to an auxiliary penalty channel.

## Step25-v3 Copy-Aware Dual-Channel Continuation (2026-07-18)

Implementation status:

- branch: `method/step25-v3-copy-aware-dual-channel`
- policy: `schema/step25_v3_copy_aware_dual_channel_policy.json`
- Linux runner: `scripts/run_step25_v3_copy_aware_dual_channel_linux_20260718.sh`
- detailed protocol: `docs/STEP25_V3_COPY_AWARE_DUAL_CHANNEL_PLAN_20260718.zh.md`
- Windows static validation: Python compilation pass, `15/15` contract tests pass, four config-only entry points pass and Linux shell syntax passes under Git Bash
- numerical execution: completed on Linux and synchronized; score replay passed, but the result is invalidated for final interpretation because non-KKT fits were incorrectly marked converged

Fixed scientific comparison:

- `C0`: matched raw-style constrained LR/L2 baseline
- `C1`: raw plus pair-local-clean channels without copy penalties
- `C2`: preregistered primary, raw/clean channels plus direction-constrained raw-clean residual and copy-risk penalties
- `C3`: identifier-redacted E5 sensitivity control, not a selectable primary

Boundary and promotion discipline:

- canonical train only: English `401 = 116/285`, Chinese `573 = 229/344`
- five seller-component grouped folds with the exact Step25-v2 assignments
- no valid/test row, threshold, score or fitted statistic is read
- no identifier, candidate-rule, review label or evidence type enters the clean scorer
- D0 can set only `d1_replication_candidate_eligible`; publication promotion and Step11/17 entry remain hard false
- every preregistered gate must pass before a new component-disjoint, score-blind D1 replication is justified
- frozen v1/v2 metrics, summary hashes and manifest hashes are carried into the v3 result summary for provenance, without reinterpreting the parent conclusions

Returned v3 diagnostic:

- payload/producer verification: `9/9` and `21/21`, zero mismatch
- C2 source-only/target-OOF/English-OOF AP deltas over C0: `-0.030203/-0.027580/-0.058026`
- target C2-minus-C0 grouped-bootstrap 95% CI: `[-0.092019, 0.021403]`
- gates: `2/11` pass; no D1, publication or graph promotion
- invalidation trigger: reported convergence with final projected-gradient residual up to `0.52` versus required `1e-8`

The isolated v3.1 repair is now the only authorized rerun. It uses `schema/step25_v3_1_solver_convergence_policy.json`, `scripts/run_step25_v3_1_solverfix_linux_20260718.sh` and output root `v3_1_solverfix_20260718`. No scientific setting is changed; the repaired manifest must fail unless every constrained artifact reaches the original KKT tolerance.

## Step28-v13 v1.12 Clean-Room Preceremony Baseline (2026-08-03)

The current Step28 mainline has restarted as v1.12 without restoring any deleted v1.3–v1.11 payload. A design-only two-world replay passed with 756 complete pairs (40 positive, 716 negative), 168 identity assets, 756 identity33 rows, and ten M1 mappings. It validates all 915,996 forbidden historical identity hashes and 90 forbidden master commitments, and exercises the fixed per-asset collision counter, world-scoped mechanism keys, Windows long paths, final-body self-hashing, suffix-only identity projection, join-only UID separation, and the single current member contract.

The receipt SHA-256 is `d21964a248e1138e65a654262e026c8c1457f8500e4915dbf5b83cdaba09d243`; its status is `PASS_DESIGN_ONLY_NO_FORMAL_AUTHORIZATION`. The repository suite now reports 396 tests: 389 passed, 7 declared skips, and 0 failures in 809.034 seconds.

This is not a released synthetic dataset. Formal seed/key access, formal rows, scientific metrics, and model training remain zero. The formal seed ceremony, four-split generator, release/custody path, and full-input numerical optimizer preflight must still be implemented and frozen before a one-shot formal run can be authorized. See `docs/STEP28_V13_V1_12_PRECEREMONY_BASELINE_RESULT_20260803.zh.md`.

## Step28-v13 v1.12 Terminal Quality Failure (2026-08-09)

V1.12 later completed the one-shot seed ceremony and generated finalized train and development stages: 500 worlds and 189,000 complete pairs per split, plus five train M1 structural rewires. The only joint quality audit failed before writing any quality receipt, quality marker, publication, or model result. Exact localization found 22 train and 13 development `{title, description}` document hashes intersecting the successful historical v1.2 release, plus 7 and 3 within-split duplicate item-document occurrences. All train/development cross-split frozen-set intersections and failed-identity-hash intersections were zero.

V1.12 is permanently terminated and cannot be waived, regenerated, or reused. Its 378,000 formal staged pair rows were never published or training-qualified. The terminal receipt, 25-entry lineage manifest, and hash-only exclusion archive have raw SHA-256 values `03aa550ec695cd21c98771edf1fb91fd105e869c318be6cff3b468900c8ca31f`, `bfaa540c3634c54c92ced2f1a01d3281ffc9bf4fb8d95f641c22e7df062b4f96`, and `b29e2e7dc46921aa3f9d1f7e03d9e75baabde5093dd3fb63b93fd3bc9c19eb01`. The successor is v1.13, which must freeze label-blind collision registries for both item documents and five-field seller documents before any new seed ceremony. See `docs/STEP28_V13_V1_12_TERMINAL_FAILURE_AND_CLEANUP_20260809.zh.md`.

The terminal archive was committed as `48bca5c6208130a5a266bb55a9d9898b1e4ba6ab` and revalidated before cleanup. The failed private staging, consumed seed custody, browser cache, and empty run directory were then removed: 111 files and 1,887,149,059 bytes. The cleanup receipt raw SHA-256 is `06fe76bf1fe93f913439b6aebd68f469912e832a4c597814010f7ccbfff253e2`, with canonical self-hash `1eff3fe8db684594e4e93eb4ef3be8157ec1ccf0900c965c72457b5d4643eab8`. V1.12 formal execution is now permanently closed.

## Step28-v13 v1.13 Visible-Document Collision Contract (2026-08-09)

The v1.13 design contract is now frozen at `docs/STEP28_V13_V1_13_VISIBLE_DOCUMENT_COLLISION_CONTRACT_20260809.zh.md` (17,811 bytes; SHA-256 `0f1906a8be46be3d3cdf16738814e0c83192e97299d162857b71ca585208eca6`). It introduces a candidate-independent structure and identity parent, exact-document-collision-only retry semantics, immutable Step3 contributor provenance, per-world no-replace commit markers, deterministic split sealing, exact historical registry pins, row-level multiplicity gates, and seller-level post-selection shortcut attacks.

The pinned style-profile attestation independently confirms that all seven `title_missing` quantiles are numeric zero; its raw SHA-256 is `ee3e84ac0ed111027c41ff1760db29935f6473b1b244445b51085fd7a4daf326` and canonical self-hash is `be951a3e6ee1c4cca58f26ca4b1a5658c0105adabcf66706c492a21f5f2bdaa8`. The browser reviewer closed the initial 2 Blocker, 3 High, and 2 Medium findings, then closed three final implementation ambiguities with `Blocker 0 / High 0 / Medium 0 / Low 0` and `V1.13 DOCUMENT COLLISION CONTRACT FINAL GO`.

This is design authorization only. No v1.13 implementation, seed, capability, formal row, quality receipt, model, or metric exists. Implementation must proceed in reviewed stages, beginning with pure document hashing, exact history loading, and candidate-key derivation.

## Step28-v13 v1.13 Collision Primitives (2026-08-09)

The first implementation stage is complete. The design-only policy is `schema/step28_v13_v1_13_document_collision_policy.json` (12,084 bytes; SHA-256 `2331ccbcf4c08171204f86fbe44a430780d11967b944e72996b707de2c9cca99`; canonical self-hash `d993ad7252789eee341f8875b6a11cd757234c9e07da1ec71fe620bec3a740a9`). All six formal authorizations remain false. The pure implementation is `scripts/step28_v13_v1_13_document_collision.py` (34,480 bytes; SHA-256 `5cb1907ed6d15ea52cdc0b35069fdff1cc651dca2b570debbb53ee77a8f9d4d4`).

The item serializer now directly implements the frozen `json.dumps` byte contract; the seller serializer implements the exact five-field strip/drop/newline contract. The multiplicity gate rejects duplicate registry rows instead of silently collapsing them. The historical replay recomputes successful-v1.2 public rows and validates the failed-v1.12 archive, producing 302,944 forbidden item-document hashes, 84,000 seller-document hashes, 999,996 identity hashes, six exact UID-hash registries, and 37 consumed commitments. The loader is intentionally uncached.

The reviewer initially found 2 High, 3 Medium, and 1 Low issues. All implementation issues were fixed; the patch review returned 0 Blocker, 0 High, 0 Medium and `V1.13 COLLISION PRIMITIVES PATCH GO`. Its remaining Low test suggestion was then added and explicitly closed. The focused suite has 24 passing tests; combined with the historical preceremony contracts it has 39 passing tests. Full repository discovery ran 512 tests in 952.525 seconds: 504 passed, 7 existing skips, and the sole failure was the frozen Step28-v12.1 guarded sync-manifest test requiring the continuously updated `PROJECT_PROGRESS.md` to match its 2026-07-20 199,490-byte snapshot. No v1.13 test failed, and neither the historical manifest nor the current progress document was rewritten to manufacture a green result. This stage still cannot derive formal capabilities, generate candidate text or rows, or train models. The next stage is the candidate-independent parent, contributor provenance, and one-time trial identity allocation.

## Step28-v13 v1.13 Candidate Parent and Trial Identity (2026-08-10)

The second implementation stage is reviewer-final. `schema/step28_v13_v1_13_candidate_parent_policy.json` is 8,158 bytes (SHA-256 `b550fbb0579e84a46f896b9f39fb29edeec64021d32d61eff94c42b63f768b26`; canonical self-hash `2c35e235b90a38f317b194cccda66a049767c68eb40cd3c44e439268e9c732ed`). `scripts/step28_v13_v1_13_candidate_parent.py` is 57,236 bytes (SHA-256 `b700265b7791866587f685cdda24d1f299849a0397209dc14fec27c754831e84`). The v1.13-only remapper is 13,854 bytes (SHA-256 `a85ff8460c86c339489beb1ca5101e16b955e09c22b95dcc965e8662604f249b`). It is restricted to one in-memory development-smoke world and has no formal-custody, candidate-text, seed, row, model, or metric path.

The immutable parent binds the 28-seller/378-pair structure, registered override and noise targets, structural identity plan, six effective style factors per seller, and private contributor positions for the five frozen Step3 fields. The provenance replay records support item UIDs, first position, aggregation role and output rank; description signatures additionally bind the first item, extracted-segment ordinal, seller document frequency, and the contributing-seller-set digest. It stores no contribution text and proves that public Step3 profile bytes are unchanged.

The parent builder has no caller-selected inputs and revalidates the actual template, fixture, and style-profile bytes. Its exact 19-member local source/policy closure is checked against a recursive import scan. The smoke allocator accepts only the pinned parent; historical identities, key, counter budget, and the empty committed state are internal authorities, and no fault hook is exposed. A pure v1.13 remapper is used once, while a separate replay recomputes every asset's first-admissible counter, visible rejection count, and selected hash. Parser/redaction closure, profiles, provenance, and all 378 identity33 rows are then recomputed. The 25 candidate-parent tests and 24 collision-primitive tests pass together (49/49). Successive browser reviews closed all findings and ended at `Blocker 0 / High 0 / Medium 0 / Low 0` with `V1.13 CANDIDATE PARENT LOW CLOSED`. Full repository discovery then ran 537 tests in 1052.129 seconds: 529 passed, 7 existing skips, and the sole failure remained the frozen Step28-v12.1 sync-manifest test requiring `PROJECT_PROGRESS.md` to match its 2026-07-20 199,490-byte snapshot. No v1.13 test failed or errored, and neither the historical manifest nor the current progress document was rewritten to manufacture a green result. All formal counts remain zero; the next stage is the restricted candidate-safe view and independent natural-expression variation.

## Step28-v13 v1.13 Restricted View and Natural Variation (2026-08-10)

The third implementation stage is reviewer-final and remains restricted to one pinned in-memory development-smoke world. `schema/step28_v13_v1_13_natural_variation_policy.json` is 8,227 bytes (SHA-256 `f66290e6d95628d133da0f1eef55f53b82a6f884b67909340a5f235b8be1186b`; canonical self-hash `23dfa4b3b489d9af4ed27a253446ee5e107cc87ef39e4fab0e5131ab5a034777`). The independent pure renderer is 27,637 bytes (SHA-256 `de61f3e83fdb6f0c67a70cd629e0c93a3b8099c0bb50e81bb117c5698c8c1cb0`); the trusted host is 69,849 bytes (SHA-256 `5ba93638765b3988e78462f962a5d29c0190df63b3bcdf0b5666983ad93c30d5`).

The pure renderer accepts only a restricted anonymous view and a 32-byte candidate key, imports only the standard library, and has no policy, filesystem, world, history, label, controller, identity, override, clone, or semantic-mechanism authority. Registered overrides and exact title clones remain private to the trusted assembler. The assembler rebuilds the sole canonical view and binding, rederives the candidate key, redraws and exactly compares the candidate, and verifies full-state commitments that cover every field of both the candidate parent and the frozen trial-identity parent before any redraw. All non-completing paths, including `KeyboardInterrupt` and `SystemExit`, poison the session and prevent candidate skipping.

All 32 smoke candidates have unique final natural-text and world hashes, with 51 to 105 changed items relative to candidate zero. The restricted-view hash is `47ba01d22eb6a55694620970f30cd4acfcf4a1f3e696ed88579bc6e61aa9e85a`; candidate-zero final natural/world hashes are `5db775cf5040bf7f2f782e3bfacc4de2038bbbdefc1472f26da0404be56c7644` and `fac76459aeab02992f74993f253a43bc13c6380ddefb923ee7554f83004a3722`; the ordered combined hashes for all 32 natural outputs and worlds are `8901662f1790d0d1bd921b815a3f425fe6470eab4da32fd1c7766aff67bf5cef` and `ca6c53d8025c1a452fdb838cb1be354a4e6eef50e5ebf37c51c367b12611c161`.

The browser reviewer first reported 0 Blocker, 4 High, 2 Medium, and 1 Low; after those fixes it found one remaining High in caller-supplied parent/frozen authority. Full-state root commitments and pre-redraw forgery tests closed it. The final verdict was `Blocker 0 / High 0 / Medium 0 / Low 0` and `V1.13 NATURAL VARIATION FINAL GO`. The 34 focused tests pass, and the three v1.13 stages pass together (83/83). Full repository discovery ran 571 tests in 1069.510 seconds: 563 passed, 7 existing skips, and the sole failure was again the frozen Step28-v12.1 sync manifest, which at run time expected the current 208,993-byte `PROJECT_PROGRESS.md` to equal its 199,490-byte historical snapshot. No v1.13 test failed or errored. Formal seeds, capabilities, candidates, rows, receipts, models, and metrics remain zero. The next stage is exact document-collision candidate retry plus no-replace world transactions and split sealing; this stage does not authorize a seed ceremony or training.

## Step28-v13 v1.13 In-Memory Candidate Selection (2026-08-10)

Stage 4A is reviewer-final and remains a single-world, in-memory development smoke. The strengthened natural policy is 8,227 bytes (SHA-256 `eb24f5ac99ffadb278ff6ec0ccad1a25d44d74818bf242d27f277fe6449e6fb0`; canonical self-hash `562dce55d87f39b87b80dad8060bf16130c801213c22a89f70bf249932e665bf`), and the trusted natural host is 82,103 bytes (SHA-256 `6f49f3c9e1ed322195a9a541a2960d25bd64f50a6d823c0c0c3a9f44fdae9e5b`). It now proves exact-title-clone source/target eligibility, registration order, earlier endpoint non-use, matching negative-flag lineage, and parent roots before candidate zero is observable, without adding those private facts to the restricted renderer view.

The candidate-selection policy is `schema/step28_v13_v1_13_candidate_selection_policy.json` (6,355 bytes; SHA-256 `ddce84eeaf8c5e74067efacf497426c682acd14b62c13e87deb5ab8e53ebe6a0`; canonical self-hash `32f8a431311eba03e97068dc3b4597a24e4befe8867dd8311dde755c580a783d`). The selector is 49,420 bytes (SHA-256 `73888224e44b606c6204d002397c2cec986c9d61d60222fdada56e5d1d048a55`), and its 31-test contract file is 32,424 bytes (SHA-256 `1c8d4f51b6f6075586a49d7c516b961c3d944a45e8f13ec45349f44bdbac16e8`). Each frozen-input key is bound to one canonical path as well as exact size and hash.

The zero-argument selector renders candidates in order under one frozen identity allocation. It independently replays production redaction and the final five-field seller profiles before hashing documents. Only exact item/seller collisions within the world, historical registries, the current split, or predecessor splits permit retry; all other closure failures poison the selector before the next candidate. Duplicate documents are classified before the accepted-level 105-item/28-seller multiplicity gates.

The accepted development object carries the complete 84-hash identity allocation delta, item and seller registry deltas, collision context, title-clone qualification, and all candidate payload commitments. Completed validation must match both the selector-retained trusted assembled candidate and its original accepted-state root, then reruns production documents, hash rows, zero-collision classification, and registry cardinalities. A caller cannot replace private world state or rejection-history counts and manufacture a new self-consistent root. The object remains `design_smoke_only=true` and `committable=false`, and the module has no write path.

Candidate zero is accepted with 105 item hashes, 28 seller hashes, and 84 allocation hashes. The collision-context, title-clone qualification, and accepted-state SHA-256 values are `b0aef4fe60d1b2a117b30be9788a42bd6cba2a40123dfb82f8946640b199428d`, `145c94bfd1ad0ffe5639d9c62e3c5a7c7c8531923e7723aa34e5bb803857b9fe`, and `6f14e23acbb0b7adbfda3e7044c9d5389338db30fd5692dee2a2aaff3802343f`. These are replay goldens, not model metrics or a dataset-quality result.

Successive browser reviews closed production-replay, canonical-path, trusted-candidate, and real rejection-history binding gaps. The final verdict was `Blocker 0 / High 0 / Medium 0 / Low 0` and `V1.13 CANDIDATE SELECTION IMPLEMENTATION GO`. The 31 focused tests pass; all four v1.13 stages pass together (114/114) in 238.920 seconds. Full repository discovery ran 602 tests in 1430.937 seconds: 594 passed, 7 existing skips, and the sole failure remained the frozen Step28-v12.1 sync-manifest test requiring the 212,007-byte pre-update `PROJECT_PROGRESS.md` to equal its 199,490-byte historical snapshot. No v1.13 test failed or errored.

Formal seeds, capabilities, committable candidates, transactions, rows, receipts, models, and metrics remain zero. The next stage is 4B: split-lock state reconstruction, in-lock selection, no-replace world acceptance, continuous-marker recovery, deterministic registry publication, and split sealing. Stage 4A does not authorize a seed ceremony, formal generation, or training.

## Step28-v13 v1.13 Development-Smoke Split Transactions (2026-08-10)

Stage 4B code is reviewer-final for one pinned `development_smoke / audit_a / world 0` ephemeral state machine only. The policy is 8,972 bytes (raw SHA-256 `b282f62496e65df214870236df05e294d00f4a1117d7dbbcbe839372a8f7bff0`; canonical self-hash `2ea3c38e2f0b1a10feaad50ef751406c7a67f18dfb23d88efdbb80e4d667128e`). The source guard is 12,339 bytes (`00b3b24ef5ad099175142e36545fea98acf1e142636385c68159c272fcf42136`), the transaction implementation is 97,248 bytes (`58ce5d52e1dfb97090acc9c3f1a3499822050fd99612362742fe8a1b4b049fcf`), and the non-discovered contract test is 56,832 bytes (`63792333f78a3f1692af0a5fb8cd3914a747d630879d652d01e10af30c93641f`).

The stable lock precedes temporary-root creation, recovery reads, and candidate selection. World members and marker, seven deterministic final projections, split seal, cleanup, and recovery are all no-replace and fail closed. Pre-seal recovery replays production documents against the Stage 4A golden; post-seal recovery independently derives the expected state from fixed `render(0)`. Unknown entries, link substitution, stale-pending state mismatches, marker bool/int substitutions, synchronized final/seal/receipt tampering, and cleanup-member resurrection are rejected without deleting evidence.

The official entry is `python -I -S -B scripts/step28_v13_v1_13_source_guard.py --focused-tests`. It rejects startup hooks, project bytecode and preloaded project modules, verifies the policy plus guard/source/test bytes, and compiles the verified source bytes. Import order is isolated stdlib, third-party packages, then canonical project scripts; each frozen project module must still resolve to its exact pinned source before first import. User-site dependencies are explicitly reintroduced on this Windows development host, may depend on OS environment, are not byte-pinned, and require a separate formal environment attestation. Ordinary `unittest` discovery neither includes nor certifies the Stage 4B focused contracts. The heavy contract refuses execution before any import unless the guard supplies both its fixed module name and private execution sentinel.

The final authoritative focused run passed 60 tests in 370.870 seconds with one declared Windows symlink-permission skip; the latest 114 Stage 1–4A regression passed in 244.212 seconds. A real isolated smoke reached `SEALED_CLEANUP_COMPLETE` and left zero project `.pyc` files and zero smoke temporary directories. Nine browser review rounds finally closed discovery-wrapper bytecode takeover, discovery-time cache conflicts, platform-specific skip parsing, and direct heavy-contract execution. The final verdict was `Blocker 0 / High 0 / Medium 0 / Low 0` and `IMPLEMENTATION CODE GO, EXTERNAL ANCHOR PENDING`. Ordinary repository discovery intentionally excludes the 60 authoritative contracts; it ran 602 tests in 1681.523 seconds, with 594 passing, 7 existing skips, 1 failure, and no errors. The sole failure remained the frozen Step28-v12.1 manifest requiring the 219,186-byte progress document seen at run start to equal its 199,490-byte historical snapshot; there was no new v1.13 failure.

The implementation bundle is fixed by parent commit `dbafb62f91a51b057b5a4846b8028de4076c7c1c` and tree `c7ea90eafe62b0682722e87bf780bf2adab95358`. Its successor commit carries `docs/STEP28_V13_V1_13_SPLIT_TRANSACTION_IMPLEMENTATION_REVIEW_20260810.zh.md`, which independently pins that parent identity and the four exact artifacts without being read by the policy, guard, implementation, or tests. This closes implementation provenance only. Formal seeds, capabilities, candidates, rows, quality receipts, models, and metrics remain zero. This stage does not authorize a formal 500-world run, seed ceremony, dataset generation, or training.

## Step28-v13 v1.13 Scientific Multi-World Builder (2026-08-11)

The scientific four-split builder is implementation-review final. Its policy is 7,493 bytes (raw SHA-256 `061b6308b2b85446b08526e32580e7919f0a232624bcb10f9e9460f4f37f806e`; canonical self-hash `35d925d98a12203bb580015992dfdd047e87299887b6a3d1127b64a273bafd36`) and pins all three runtime sources. Candidate selection remains label-free; six UID universes and document/identity registries are atomic; world ordinals are exact; the model-facing seller projection uses a positive allowlist; and the completed tree is reread and hash/row-count verified before publication.

All 16 focused contracts passed in 77.388 seconds. They include two deterministic real four-split builds, all six UID collision paths, output-byte tampering, ordinal closure, and independent full-profile versus strict-projection `legacy18` equivalence over 378 pairs × 18 eligible features. The canonical 4×1 smoke root manifest is 2,638 bytes (raw SHA-256 `08832aea24e7937050a8e28ab5a90f0668cdcf2ce3109914de9e3de705a0d6bf`; canonical self-hash `698be3c55233c72a6708f2dfae2621d0626cc0185f12f9ef1994ed67b5ba1277`) and records 4 worlds, 112 sellers, 423 items, 1,512 pairs, 80 positives, 48 controllers, 112 queries, and 341 identity values. This is reproducible smoke evidence, not a quality-qualified dataset.

The final GPT-5.6 Sol xhigh review reported 0 Blocker, 0 High, 0 Medium, and 0 Low findings and authorized only the 104-world design preflight (50 train, 50 development, 2 audit A, 2 audit B). Formal mode remains disabled. No formal seed, formal row, M0/M1/M2/M3 model, or metric exists. The next step is the 104-world build followed by statistical, text-shortcut, and row-level quality audits; formal 500×4 generation and training remain unauthorized.

## Step28-v13 v1.13 First 104-World Preflight Failure (2026-08-11)

Run `design_preflight_v1_20260811` permanently failed closed at train world ordinal 3, candidate 0, after three worlds had been staged. No final output directory was published and the temporary build tree was automatically deleted. Deterministic replay showed that exact-title-clone asset 0 had nonempty source/target titles but a structurally empty target description; asset 1 was qualified. The shared override picker required only nonempty titles and therefore did not guarantee the scientific builder's stricter target-description requirement. The fixed 4×1 smoke had not covered this state. The v1 output root and old implementation bytes are non-reusable. The successor must qualify source-title and target-title/description endpoints before candidate rendering without reading labels, scores, or shortcut results, and must add both the train-ordinal-3 regression and a full 104-world qualification preflight before a clean versioned rerun.

## Step28-v13 v1.13 Endpoint Qualification v2 (2026-08-11)

The failed v1 implementation has been replaced by v2. The policy is 8,189 bytes (raw SHA-256 `5c4ba22cbbd001efab521384a9a988410f30703bf2657a1d425bd8eba4d2629a`; canonical self-hash `8f40cc0b008e6447e5ace55b59159545346c6527a91d2129677e59ce087a7a47`). Current source pins are: common 17,390 bytes / `99aaaef84766eb4dd57a7adcbf7f70a0b7ba04bcf153697883980145c8163385`; world 47,252 bytes / `9a3a3101a01f1cd671e381b9bc86b7d3b664eaec533e9f9e80914ac1cd40550a`; builder 41,124 bytes / `e15e4e1125f782e069d1d4c123003d8562a4dc2a4024faa7c2b30c6ba5a05028`. The 29,561-byte test file has SHA-256 `368c08dc439b21ea21f0b1aa3de61c5b07c42304a7f8db3481134a91b240bd9c`.

Before identity remapping or candidate rendering, v2 deterministically reselects only item endpoints inside the already-fixed source/target sellers. Source title and target title/description must be structurally nonempty. Seller pairs, direction, canonical pair IDs, controller membership, negative flags, and final label projection are unchanged. The old clones are replayed and undone before the two updated registered clones are applied; the four semantic-override items remain reserved. Qualification receipts and updated overrides are part of the structural commitment and private audit.

All 19 focused contracts passed in 87.272 seconds. New tests reproduce the original train-ordinal-3 `[false, true]` target-description state, traverse all 104 design worlds and 208 clone rows, and reject a target seller with no described item. The final GPT-5.6 Sol xhigh review independently verified current bytes/pins and reported 0 Blocker, 0 High, 0 Medium, and 0 Low. It authorized only a clean `design_preflight_v2_20260811` run (50 train, 50 development, 2 audit A, 2 audit B). Formal seeds, rows, models, and metrics remain zero; formal 500×4 generation and training remain unauthorized.

## Step28-v13 v1.13 104-World v2 Design Build (2026-08-11)

`design_preflight_v2_20260811` completed in 298 seconds on Windows CPU. Its 2,663-byte root manifest has raw SHA-256 `c7b323d9d3b76a0795ce45452b4989926cae51fca7101bc6d04acf6fdf4a93f3` and canonical self-hash `9baa90828cf459bcee3cc6101c166f6c1084353dd2997e40e5d3d85d29f49d48`. The 49-file tree is about 89.59 MiB and was independently reread through `_verify_output_tree()` after publication.

The build contains 104 worlds, 2,912 sellers, 10,561 items, 39,312 pairs, 2,080 positives, and 37,232 negatives. Every world has 378 pairs and 20 positives. All six UID universes and the item/seller/identity registries close without reuse. Public item/profile schemas, nested schemas, types, and forbidden top-level fields have zero observed violations. Of 104 worlds, 103 accepted candidate 0; development ordinal 45 advanced to candidate 1 solely because of one historical item-document collision. All 208 endpoint-qualification rows close; 130 sources and 116 targets were relocated without changing seller pairs, directions, controller membership, or negative flags.

The root status is `PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED`, with `scientific_use_forbidden=true`, `training_started=false`, and zero formal seeds/rows. The tree is retained only as input to the preregistered row-level, split-isolation, metadata-shortcut, and controller-blind text-counterfactual audits. It is not formal data and cannot be used for M0/M1/M2/M3.

## Step28-v13 v1.13 质量审计第一版运行前否决（2026-08-11）

质量审计第一版没有启动 104 世界审计，也没有创建质量输出。网页端 GPT-5.6 Sol Pro 的完整调用链审查报告 5 个阻断级、5 个高级、3 个中级和 1 个低级问题，最终结论为“不允许运行”。阻断项覆盖非法的 64 位 scikit-learn 模型随机种子、审核集甲乙在盲预测前物化控制者／标签／检索相关性、重抽样世界顺序错误、私有身份字面及反事实输入扫描不全，以及商品数和字段缺失率的数据来源错误。

被否决边界为：质量策略原始 SHA-256 `9f5a6b973acf84d6a0efd57272a449a0bca258af0a7eaa5c685ec4f720cae2a2`、规范自哈希 `d14712a774eb609a686a689a5aa32063805bfd566c1a4dd20bcae3d3c97cc57f`；质量脚本 62,770 字节／`166ac614ab35fed1a45fa2acc749433fc1cb1b8ad4b85ba86f1b7555c8148dc8`；测试 9,031 字节／`b5726175b17591d5191d0a75bc68af7bcb2fe9f56368b4242c6d1171f1e5df3c`。这些旧字节永久不可运行，旧实现已由第二版替换，且没有失败实验载荷或临时目录。

这不是设计数据质量失败结论。104 世界 v2 数据仍仅作为待审输入保留，继续禁止训练。第二版质量实现必须完成行为测试和同一网页模型复审；未得到明确放行前，不得运行质量审计、正式 500×4 生成或 M0/M1/M2/M3。

## Step28-v13 v1.13 质量审计 v2 方法学关闭与 v3 三路径实现（2026-08-11）

质量审计 v2 没有运行 104 世界审计。新增行为测试发现，首个固定训练世界的原始 Step3 画像贡献血缘为 576 行，风格反事实后为 574 行；91 个按卖家、输出字段和排名对齐的槽位发生变化，全部位于 `signature_description_concat`，覆盖 25/28 个卖家。三十三维身份历史、商品非文本结构和语义抽象语法树、身份值、噪声、机制及标签仍保持不变。原实现错误地要求原始与反事实 Step3 血缘逐字节相等；该要求会把重新分段、跨卖家文档频率和 top/signature 选择这些合法下游中介误判为污染，因此 v2 方法学永久关闭，不得运行。

网页端 GPT-5.6 Sol Pro 明确裁定采用同一固定无固定点置换、同一反事实商品和三条审计路径。固定商品支持文本路径 F 锁定 `(world_uid,seller_uid,item_uid,field)` 槽位，逐商品向量化后作顺序不变的卖家聚合；真实 Step3 生产路径 P 要求原始生产输入精确重放、反事实生产输入与第二次独立重放逐字节相同，并保存不可用于筛选的原—反事实血缘差异回执；联合路径 U 拼接 P 的 75 维全视图、F 的 33 维全视图和 16 项模型可见数值差异，共 124 维。生产 3 视图、固定支持 3 视图和联合 1 视图分别使用固定 L2 逻辑回归与深度 2 浅树，共形成 14 个探针；单特征、14 模型点估计及 9,999 次世界重抽样都在完整探针家族内取最大值。

当前针对性合同测试为 22 项全部通过，覆盖固定支持删槽、重复槽、跨卖家移动、空值模式变化、输入行重排、商品编号一一改名、反事实血缘确定性重放、错误来源商品归属以及第 14 个高分模型不得漏出重抽样家族。包含候选父对象、文档碰撞、候选选择、自然变化、科学构建器和质量审计的完整 v1.13 合同组共 155 项，也在 296.996 秒内全部通过。质量策略为 17,003 字节、SHA-256 `bdb6178d1b87b766ab14706c5e97e24087dcda16729ee92d9c86130a97991d62`、规范自哈希 `76d942244d2d8c71a7019879eea13867073c01a8a90213d382c822e52c234f99`；反事实源码、质量审计源码和质量测试的 SHA-256 依次为 `1c1756712ee3cdd9d56998825f47409a3f6a316d4fb81e6488d59f9c5877bce0`、`717fbed3d29dbca831420b852863586bcea4d1e658c8d372fae57ff7bcdf6c2d` 和 `2f4b24bf16fa738eaa582c1f9b11e29dc7c41af864947f470ae2c047ea7a4e70`。网页实现终审仍未完成；本节不构成质量通过或运行授权。只有网页端明确给出“允许清洁运行104-world质量审计”后，才可启动设计级质量审计。正式种子、正式数据行、正式 M0/M1/M2/M3 模型和指标仍全部为零。

## Step28-v13 v1.13 质量审计 v3 运行前终审否决（2026-08-11）

v3 没有运行 104 世界审计，也没有产生质量结果、模型或指标。网页端 GPT-5.6 Sol Pro 独立复算本轮附件字节并追踪调用链后，给出阻断级 4、高级 3、中级 2、低级 0，最终为“不允许运行”。永久关闭的 v3 边界为：策略 SHA-256 `bdb6178d1b87b766ab14706c5e97e24087dcda16729ee92d9c86130a97991d62`、规范自哈希 `76d942244d2d8c71a7019879eea13867073c01a8a90213d382c822e52c234f99`，反事实／质量／测试 SHA-256 依次为 `1c1756712ee3cdd9d56998825f47409a3f6a316d4fb81e6488d59f9c5877bce0`、`717fbed3d29dbca831420b852863586bcea4d1e658c8d372fae57ff7bcdf6c2d`、`2f4b24bf16fa738eaa582c1f9b11e29dc7c41af864947f470ae2c047ea7a4e70`。

阻断根因是 F 空槽率特征左右不对称、7 视图／14 模型没有机器合同修订件、审核甲乙真实私有字面精确扫描不闭合，以及 P 血缘行字段验证不完整。另须补审核私有文件的纯字节完整性、真实的失败分类和数据处置、对应行为测试、外部启动锚及三路径逐世界对齐回执。已新增 `STEP28_V13_V1_13_QUALITY_AUDIT_C_AMENDMENT_20260811.zh.md` 作为 v4 待实现权威；它不授权运行。当前固定置换不得重抽，104 世界设计数据仍仅作待审输入，正式种子、正式数据、模型和指标仍全部为零。

## Step28-v13 v1.13 质量审计 v4 运行前终审否决（2026-08-11）

v4 没有运行 104 世界审计，也没有产生质量输出、模型或指标。34 项质量合同和六组共 167 项 v1.13 合同全部通过，但网页端 GPT-5.6 Sol Pro 最终仍给出阻断级 0、高级 1、中级 2、低级 0，最后一行是“不允许运行”。高级问题是同仓库启动锚缺少独立发布清单／外部来源记录；中级问题是封存扫描器仍按固定身份类别维护，以及剂量证据缺少更强的完整表留存与变异测试。

永久关闭的 v4 字节为：修订件 8,694／`c930d9f8c9d29ba796556d590603cc48a7ca05c898c174933d9fdf07234f6e46`；策略 18,390／`884006ba248e602b06832292eb8039ce59d4feab0cf846794f9f243991649c2e`，自哈希 `c7e9a05504602e64bfdeb4bacd94a3d2b9cfc8e90a0e4066529433b812a56ebd`；启动锚 2,159／`1d300e7c893ce1231816bb9ecfad77d06f7529fa9b92256a3e9ccf9754ac5e08`，自哈希 `595cefbb44b41294be5e3df1a725ce3f8285a5b92ef22b202abdd5e26ed970d4`；守卫 3,389／`b2e162f06c5064ed1ad7e9195b59ebd7f4b128a5e09ac0c5e087b8fff9aea47a`；质量源码 176,379／`ab60c45d0faed9307d4a4d1859739d89fee585d3321c1fa351856d539269092a`；扫描器 8,805／`e68f8e25dda43e127dda956805066f4a5fae4bb3e27689f4f0c3b2186ebfbe19`；反事实源码 19,230／`1c1756712ee3cdd9d56998825f47409a3f6a316d4fb81e6488d59f9c5877bce0`；测试 59,233／`d238f0cbf3aa14fa54fbaa1fb94d4e28b97e4ae5c00e8ae9537e4fa98f7f8d8e`。这些字节不得运行，固定置换不得重抽。

## Step28-v13 v1.13 质量审计 v5 本地候选（2026-08-11）

v5 仅修复质量审计器，没有运行数据审计。新增候选发布清单；扫描器枚举全部非空私有字符串叶和编号型映射键，并覆盖网址／账号／数字派生字面及未来身份类型；完整剂量行分别绑定三路径；卖家文档频率与卖家摘要由原始／反事实模型可见商品独立重建；F/P/U 分别返回并核对卖家对、世界和掩码；数据失效决定回执不再提前声称已经删除载荷。新增相应变异测试。

当前候选边界为：修订件 9,557／`4761ac6ae282a2a8f0e1c6379122be80a2fc10f6a26d1cb51d6e2bd68516451f`；策略 18,542／`cb1900e9812305a876c3fd302e15bb660af3e99a4242e8146b202ee5f3bebffc`，自哈希 `3c91bfe7e870d58d6495d465ab632f64dc2c9cf4f6f3b6f757a12c43cf171e56`；启动锚 2,380／`8a69323c19e0cc3950597f0ed011f9d9e08acbb91912e683b6bdfdd72bdb42c4`，自哈希 `46f43c8fd659d7214a1200d2d4c28e4e93b7b919d53da58c0b818243c50187a3`；发布清单 2,254／`c51892a1933f5f0d71bbfb81fb629a975ca0c7aa62abf0b1ec501d27f0f338c1`，自哈希 `bc0d23e3676f65e0d77ef48e30536036549df15725a3d487563aa8e53454519e`；守卫 6,639／`7dcf193f8a98166f9079356e76240952cfe1576f7b89d8d37fdaea5a4387abbd`；扫描器 11,242／`ab0b5c8b37f708f5c9ed9511c2d332a4fb1f6ce310e2b510577f272908ef309d`；反事实源码 19,230／`1c1756712ee3cdd9d56998825f47409a3f6a316d4fb81e6488d59f9c5877bce0`；质量源码 187,569／`aa4fbc28065e424c55da5609abc64ab2a5b594eb69cf5c19c2bd51f934760bbc`；测试 76,382／`f46f995cdf0afb70c1f0ea64fb0aff421658d43a36fdab648607a47e1a9d0173`。四份自哈希和全部钉住关系独立复算一致。质量合同 38 项在 34.623 秒内通过，六组 v1.13 合同 171 项在 416.309 秒内通过。没有质量输出、失败目录或构建中目录，缓存已删除。

当前状态仍是 `EXTERNAL_REVIEW_CANDIDATE_NOT_FORMAL_AUTHORIZATION`。必须先由同一网页端 GPT-5.6 Sol Pro 对本轮准确字节终审；只有最后一行明确为“允许清洁运行104-world质量审计”才可启动一次设计级审计。正式 500×4 数据、正式种子、审核真值解封、M0/M1/M2/M3 模型和正式指标仍全部为零。

## Step28-v13 v1.13 质量审计 v5 外部终审否决（2026-08-11）

网页端 GPT-5.6 Sol Pro 用时 47 分 29 秒完成十个 v5 附件的终审，核对字节、自哈希和钉住链一致，但最终为阻断级 2、高级 3、中级 4、低级 1，并明确“不允许运行”。v5 从未运行 104 世界审计、从未创建质量输出，上一节候选字节永久关闭。

阻断项是：反事实实现错误仍可被误分成数据失效并删除有效设计数据；封存及训练／开发私有字面扫描对少于四字符、数字异形、未来映射键和混合大小写存在确定性假阴性。高级项是候选发布清单没有独立外部 GO 回执，审核身份／控制者／查询编号跨拆分隔离未由封存检查器独立闭合，以及审核侧模型可见 `item_count` 等数值未与脱敏商品独立重算。中级项覆盖启动失败回执、调用帧文件名伪造、发布清单变异测试假绿和删除后回执事务窗口；低级项是失败输出与异常消息过宽。

审查确认 F/P/U、独立 seller_df、16 项数值、三路径实际数组和完整剂量、7 视图 14 模型、9,999 次重抽样及审核盲边界主体仍正确。v6 只准修复上述剩余质量门问题，固定置换、候选、世界、阈值和探针定义不得改变；修复后必须使用新字节和新输出根重新终审。正式种子、正式 500×4 数据、审核真值解封、模型和正式指标继续全部为零。

## Step28-v13 v1.13 质量审计 v6 运行前候选（2026-08-11）

v6 已关闭 v5 的两个阻断级、三个高级、四个中级和一个低级问题：双向字面规范化与短标识边界、未来映射键、反事实错误分类、四拆分身份／控制者／查询编号封存隔离、审核侧可见数值独立重算、外部后置放行证明、真实入口核对、安全启动失败回执、发布清单语义变异测试、清理意图与恢复、以及异常／结果最小输出。最大类别占比因脱敏商品不含逐商品类别，明确不声称独立重算。

准确候选为：修订件 12,180／`1ca1e8b8cea551902cf39f42f5ea33be056c47d80c3693f5bf531c843aa94832`；策略 19,848／`88eddd48eababf595ce095390e195e8991fe60430412bc3737d535ea61c5d2f3`，自哈希 `d0edff3b54b407c87d0400c40c25455c5c5e30cd662cd1660add08252faf8c7c`；启动锚 2,404／`8f23706eb407818e95984a46a20a7096e2754cbecb3b2e40b024771ed76d6d79`，自哈希 `afc4061c61c0b9032345cd2972fe2ab899e70dde62a2f0fdf2bd78d42d6cbbd1`；发布清单 2,257／`cb7c75792a19cbf3950cc3014e17dbcc8c2d43989e927d9de9032add4f62f606`，自哈希 `88caac222a519954afa8d2ab15232d2111c6deda66dfb6eec31266e8a3e31f30`；守卫／扫描器／反事实／质量源码／测试哈希依次为 `8f3221800261cdd646fb8a9ac5c41ce98edd4750effac2406388ab5c57bc331d`、`e25105d10c4f7a3dd4c51e5be35c71e22935804005d4b1cb05e63cb17144a68f`、`1c1756712ee3cdd9d56998825f47409a3f6a316d4fb81e6488d59f9c5877bce0`、`62c3d89264b826e9159e63aba74e300c451eeeb3a33934b5cf03e2778902361a`、`6899f19768496f6e79f1add7fdc80047849c8de813bca6bf16514768d9518cd9`。47 项质量合同和六组共 180 项合同全部通过；没有质量输出、失败目录、构建中目录或缓存。

当前外部审查证明不存在，守卫只能拒绝运行。下一步仅能提交这些准确字节给同一网页端 GPT-5.6 Sol Pro 终审；外部最后一行没有精确放行前，104 世界质量审计、正式 500×4 生成、审核真值解封及 M0/M1/M2/M3 训练继续禁止。

## Step28-v13 v1.13 质量审计 v6 外部终审否决（2026-08-12）

网页端 GPT-5.6 Sol Pro 用时 33 分 18 秒终审 v6 十个精确附件，结论为阻断级 2、高级 1、中级 2、低级 1，最后一行明确“不允许运行”。v6 没有运行 104 世界审计，没有外部放行证明、质量输出、模型或指标，当前字节永久关闭。

两项阻断分别是：盲边界最终闭包漏掉已经实际增加到 1 的 `sealed_registry_isolation_calls`，使正确运行必然失败；审核封存扫描输入只覆盖三份私有文件，遗漏 `noise_slot_uid` 和另外两份私有回执中的内部字符串。高级项是守卫没有核对被钉住审查原文的实际最后一行。中级项是 Git 提交／树只验格式，以及清理恢复逻辑没有官方可达入口。低级项是运行异常仍可能输出原始 traceback。外部同时确认扫描规范化、四拆分身份／控制者／查询注册表隔离、F/P/U、7 视图 14 模型和 9,999 次世界重抽样主体正确。

本地还发现重复描述比例的审核重算漏了生产 Step3 固定的“清洗后截取 280 字”规则，可能误拒绝有效数据；已建立专门反例，纳入 v7 修复。v7 只能修复这些运行与质量证据问题，不得修改固定置换、候选、世界、标签、阈值或探针定义。104 世界质量审计、正式 500×4 生成、审核真值解封和 M0/M1/M2/M3 训练继续禁止。

## Step28-v13 v1.13 v7 密封禁止字面登记器运行前审查（2026-08-12）

为修复 v6 遗漏审核噪声编号及两类私有回执字符串的问题，新增只面向设计数据的密封禁止字面登记器。首轮网页审查曾对旧字节给出一次仅限登记器的许可，但随后按审查建议修改了源码，因此该许可随字节变化自动作废，旧字节从未运行。

第二轮精确审查给出阻断级 1、高级 1、中级 1、低级 1，最终为“不允许生成密封字面登记器”。问题为运行时源码闭包遗漏三个实际依赖、原子改名成功后的目录同步异常窗口、恢复路径公开回执验证不完整，以及构建器策略原始字节未固定。被拒绝边界为：packer 45,789 字节／`9ba672b6e2bb8ed4fb7e51815d5bc7ae0716f6b7986b30620666097a60378f42`；扫描器 36,574 字节／`db479dae3cf071f376acf0d64cfdbabc7691417a1459f438fd7a91486020f426`；质量源码 216,106 字节／`0df97a9693a304ae8489f0f4fa6e4b03603707dad5ae8eeaf7e4af5c0af931a6`；测试 145,097 字节／`f0620eb0cc0182c05b364a87f5e63634b3265898cccd012841b59e416f5b91b8`。这些字节永久不运行。

第三轮候选关闭了第二轮四项问题，但网页端用时 17 分 37 秒复审后仍给出阻断级 0、高级 1、中级 0、低级 0，最后一行“不允许生成密封字面登记器”。被拒绝边界为：packer 53,004／`45a250d21a52b38fcdbb5b388f49b603bfaf057fa18edda7b04e0f2c7c30904e`；扫描器 43,492／`723d53235f140064327a96108ac037e69eb3dbd8651082c8c8d446d41d9243d6`；质量源码 216,147／`37a0ef84b4ed7c88a2375f4a7a4b8f7bfa236db9143463d05a49abc73875a0a1`；测试 160,610／`10677e52876cb1912c5c7393db481d3bf344701d962eb1c2375e236f4c10b1f2`；25 文件闭包压缩包 236,930／`8eba7e549bf2e7c409c6eb45188dac51d0f59c89c4dd0e09560072f1ccf157f1`。这些字节永久不运行。

唯一高级问题是 `KeyboardInterrupt`／`SystemExit` 可能发生在原子改名后、目标加入 `created_paths` 前，留下无事务意图的私有 sidecar，或留下孤立的 `PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO` 公开回执。当前未审候选已改为发布前预登记所有权，并在 `_write_once` 捕获 `BaseException`、只清理本次精确目标后重新抛出。新增两项测试覆盖三个发布阶段的三类目录同步中断和写入返回后的两类中断，共 15 种场景，用时 21.426 秒全部通过，且事务外哨兵文件保持不变。仍须扩大回归和新字节终审；当前不得执行登记器或质量审计，私有 sidecar、公开回执、事务意图、质量输出、正式数据、模型和指标仍全部不存在。

修复后登记器专属 13 项回归用时 30.586 秒全部通过。扩大到整个质量单元合同类后为 45 项中 40 项通过、5 项错误：4 项来自旧质量策略对第六版扫描器的字节钉，1 项来自旧发布清单对第六版质量源码的字节钉。它们是登记器生成后必须封闭的 v7 策略／启动锚／发布清单待办，不是 packer 专属测试失败；禁止修改旧历史钉来刷绿。即使第四轮允许一次 packer，也仍须在实际 sidecar／公开回执产生后封闭 v7 合同、恢复完整测试并另取质量审计许可。

第四轮网页终审用时 14 分 1 秒，仍为阻断级 0、高级 1、中级 0、低级 0，最后一行“不允许生成密封字面登记器”。被拒绝边界为：packer 53,528／`5a3985a453b452b73f40d9c453e34bf1d72b575b0a2f41a5366ab47a4406ffc6`；测试 165,595／`85788d244afb8d23618391bf3f805125c43516d96b97d55ae56212300852e2b2`；25 文件闭包压缩包 238,984／`613fa3143872a33ee290471cfbb02fc3e0943f8688840abf6d344d0feca9285d`。这些字节永久不运行。

唯一高级问题是目标路径在真正取得所有权以前就进入清理清单；长时间重放期间第二次普通启动若创建同一精确事务文件，失败调用可能删除它，`Path.replace` 还可能覆盖检查后出现的目标。下一候选须在重放前原子排他取得单执行者锁，采用原子不覆盖发布，并且只清理由本调用原子创建的文件实体；相应三阶段竞态、并发事务、外部 `.building` 和锁恢复测试必须补齐。该修复不得改变世界、候选、标签、字面、门槛或探针，当前仍无任何登记器或质量输出。

## Step28-v13 v1.13 第五轮登记器候选未运行（2026-08-12）

第五轮候选精确边界为 packer 62,037／`5de4bd0a0b1303801c97b7045eb3995fc4aea834869cbd27e96a2b8af3ece45a`，测试 182,438／`6d1aba19f0a1540d3ab4649ce9411420e4ea909f566c0e5361af4b38aa17df84`，25 文件闭包压缩包 238,257／`143060c2bd89b1edd8cd93da2c0fe6440c0ec50d1f6d2d64ef3356c5b2cec164`。网页端流式审查混入了附件中不存在的旧版预登记代码，已主动停止，不能充当有效结论；本地反例又证明运行中恢复会删除尚无 intent 的活锁。上述字节永久不运行，大型无效页面快照不保留。

当前最小修复仅延长锁 `.building` 标记的生命周期：构建全程保留，自动恢复看到它即失败关闭；正常完成或可捕获失败由本调用清理。没有增加进程号、操作系统锁或网络相关机制，也没有改变任何数据、标签、候选、置换、门槛或探针。登记器专属 16 项合同在 119.120 秒内全部通过；仍须冻结新字节并取得同一网页模型的有效终审许可。正式 sidecar、公开回执、质量结果、正式数据、模型和指标仍全部不存在。

第六轮精确候选也没有运行。边界为 packer 62,926／`1884a5814d5a20c37c1025518beeb27f7bf4a702847eddd5eb091e1b2606c677`，测试 183,547／`3ac3e07ac1ab4db9dd9787f9e0ba62845e0c07cfa5b0664fe9f82f7bb6659154`，25 文件闭包压缩包 238,492／`fb6a6a79a0d47acbff600bd3a206f2b2ebc8b1a55b561d81b56ec8bb4c565a07`；有效网页审查原文 14,739／`56108c44b77b0e9db4e561d210e0716c5d7cfeebab6a4d97e220b28298ff9fd0`，结论为阻断级 0、高级 1、中级 0、低级 0，最后一行“不允许生成密封字面登记器”。唯一问题是恢复入口在 marker 首检与 final-lock 枚举之间可能接管一个随后才启动的新活事务。

当前修复在恢复入口先固定“调用开始时 final lock 是否存在”。开始时没有旧锁就永远不处理随后出现的新锁；开始时已有锁则活构建必有先行 marker，仍会拒绝。新增两种同步交错测试证明新 lock／marker／intent 均不会被恢复删除；登记器专属 17 项合同在 115.746 秒内全绿。修复仍不改变数据、标签、候选、置换、阈值或探针，也未引入系统锁、进程号或网络机制。当前仍无正式登记器或质量输出，须再次网页复审。

第七轮差异复审继续拒绝：packer 63,464／`7ef87d4f48f80a1a7acbe3a10d2af619d5bbc320e5460775ecff603b05c80018`，测试 187,784／`cb031886d9c28c283fa1e944fccac66ac59fae644df1e9440b51c72d6212202f`，25 文件闭包压缩包 238,613／`9fcde7ba1b6cb4d4a598f42e4121c8db7cc347b0998729aa6565a615ef2cb97b`；审查原文 9,026／`50deefd7e2fed7e9527c4bb712596fc700706adc165dd729d65ae4a3c3ad49b6`，最后一行“不允许生成密封字面登记器”。剩余边界是两个恢复调用与新构建交错时，旧锁删除后内容相同的新锁可能被早先的恢复调用误接管。

当前实现不再继续堆叠锁机制，而是删除自动恢复能力：命令行无恢复选项，兼容恢复函数只抛错，对任意残留布局均不读、不删、不改。强制终止必须暂停并保留证据，人工核验后再处理。正常 run_build 的 intent／sidecar／receipt 严格提交和可捕获失败清理保持不变。逐步填充八个事务路径的不可变性合同及正常一次提交合同均通过；当前精确 17 项登记器合同在 110.915 秒内全绿。仍无正式登记器或质量结果，必须重新网页复审。

第八轮网页端精确终审已经放行一次正常登记器构建。精确边界为 packer 58,157／`c789f235c978393701afa4866696a1d947a2dc6678368a3730ee755874970a49`，测试 176,901／`18da2de1203973d18648d676cd8154bbb64221af422952e4a73221be4fa2875d`，25 文件源码闭包压缩包 237,711／`6b6cf37390327f637cbe5571b5c545fcad9331a8f06bb2ed884feb5df501839c`；审查原文 9,605／`ab709ea458985efcd69d222b7bfc24d7dff3721e8e199875a10d4843d5807917`，最后一行“允许生成密封字面登记器”。运行必须从所有事务路径为空的清洁状态开始且只执行一次；任何异常或强制终止都必须暂停，不能恢复、拼接回执或重跑。唯一成功状态为 `PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO`。本许可只生成后续质量审计所需的密封禁止字面证据，不是质量通过、正式数据生成或模型训练许可。

唯一一次构建已成功完成：104／104 个冻结世界重放，用时 381.3 秒，退出码 0。公开回执为 10,935／`38b8757f3fcb98ae5587b85f213f0b6b801f03e0f11faf6c2470abb4f272a9f5`，规范自哈希 `3e13806a027b2fd7def156df76ace7c27326787848eb03a4d4638e0275461ecd`；私有登记表为 1,211,990／`f00018e302061d2f3f9e4b0b095ceab09661482ec7c619624fdd5332ea031d2a`，规范自哈希 `75a8eee10f298d9333af8aa4dd1ec2282b8db1031431eafa968bcaeae05d57ec`。所有事务残留为 0，公开返回私值、关系、标签和排序真值均为 0。审计甲／乙各含 2 个投影世界，禁止字面数为 3,099／3,249。该结果仅为质量审计补齐密封参照，不能解释为数据质量通过；下一阶段必须封闭 v7 质量策略、启动锚和发布清单，使完整合同全绿并获得独立网页许可后，才可运行 104 世界逐条质量审计。

第七版质量候选已经本地封闭，但仍未获运行许可。它只新增登记器构建器、私有登记表三元承诺、公开回执和 25 文件来源闭包的机器绑定，并把独立输出根改为 `quality_audit_design_v7_20260812`；数据行、候选、置换、七视图、十四探针、阈值和重抽样均未改变。来源闭包规范哈希为 `ad68911cb50144972c9927fffc01085875e7b054abe739e1206e3383da0e72d1`。精确候选为策略 20,838／`93e3a87302399b17dde85a1ab72fe8be0a8821ef9fc5de79d1e78fd635e0e858`，启动锚 2,913／`2ed2ff80669b3b8e734c219216d4c33b4519e77b2137d139c8ef64dda110eeba`，发布清单 2,766／`c926949646caebea00609b6939493ab24439e680099a8fd064a79d6ca019d2ea`，质量源码 217,189／`97ec6b3e8612053df885b1fa9e335548811973381ca0f2be9186ead2058c0897`，守卫 23,413／`885fa283b5a6e1ff5e1246ff8ab9dc66afd4f5e1cf5b2bbbff1a0f40f09aa292`，测试 178,431／`8869f5c505f528c0effec9e541e531f9c6d029e69d0b32476c35192354e0e6b2`。完整质量合同 71／71 通过，用时 193.997 秒；无质量输出产生。下一步是全仓回归、候选 Git 提交和网页端精确审查，不是运行质量审计。

标准全仓发现共运行 692 项，用时 1,775.780 秒，结果为 682 通过、7 跳过、1 失败、1 错误。两项均属于关闭历史的活路径快照漂移：Step28-v12.1 旧同步清单继续钉住 2026-07-20 的 `PROJECT_PROGRESS.md`，v1.12 正式预锁继续钉住当前已为 v1.13 私有保管目录更新的 `.gitignore`。旧豁免只允许前者作为唯一失败，因此本次不能宣称全仓通过；也不能回退当前文档/忽略规则或改旧预锁、清单、测试、豁免来刷绿。v1.12 预锁由此保持历史证据身份且不再具备运行资格。当前 v7 专属 71／71 仍全部通过，测试误删的已跟踪数据组成文档已恢复，v7 质量结果目录仍不存在。外审必须同时看到此历史回归边界。

## Step28-v13 v1.13 质量审计 v7 正式失败并完成清理（2026-08-12）

候选提交 `1ef7b4baee5ec9e280670c4034872a0f58d68d86`、树 `f997f710e7aea227af08076fc5aa1e66c199d164` 经网页端纯科研复审确认阻断级和高级问题均为 0，最后一行精确为“允许清洁运行104-world质量审计”。外审只授权一次 104 世界设计级审计，不授权正式生成或训练。

审计完成 104／104 世界逐行重放后确定失败。元数据最大单特征为卖家编号摘要端点差 `absdiff__seller_uid_digest_02`，对称曲线下面积 `0.5220111731843576`，超过冻结 `0.52` 上限；文本反事实审计整体也未通过。文本最大单特征为空白字符数端点差，数值 `0.5198248863636363`，其本身未超过单特征门，因此不能把它单独认定为文本失败原因。v7 最小失败回执没有保留其余文本硬门的诊断标量，只保留结果哈希；后继版本必须补足小型门级诊断，不得猜测或复原完整失败载荷。

实现按冻结纪律报告 `DATASET_INVALIDATED`，正式种子 0、正式行 0、训练未开始。先提交失败决定与清理意图，再删除根自哈希 `9baa90828cf459bcee3cc6101c166f6c1084353dd2997e40e5d3d85d29f49d48` 的 104 世界设计数据，最后写入清理完成回执。失败目录只保留 3 份回执、共 8,529 字节；与数据绑定的私有字面登记表随后删除。v7 的数据、登记表、随机权威和结果永久不得重跑或复用。完整边界见 `docs/STEP28_V13_V1_13_QUALITY_AUDIT_V7_TERMINAL_FAILURE_AND_CLEANUP_20260812.zh.md`。

后续源码审查确认卖家编号来自 `id_key`，控制者分组来自独立的 `structure_key`，没有发现按控制者顺序生成公开编号的直接错误。清理后不读取 v7 数据的简化零信号模拟又显示：复现 28 卖家／12 控制者／378 对、使用 64 个独立代理时，50 世界的 300 次模拟有 `66.3%` 出现最大单特征对称曲线下面积超过 `0.52`；200 世界的 200 次模拟没有超线。这是事后方法诊断，不是 v8 预注册，也没有完整复现 67 项元数据、十四模型和自举门。

因此 v8 先实现与真实质量门同构的零信号校准器并事前决定设计世界数，而不是追着 `seller_uid_digest_02` 重编号。描述空白字符只作为待验证线索，尚不能据此重写排版生成器。失败回执仍须增加所有硬门的小型诊断标量。不得删除不利探针、放宽阈值或枚举多个数据盐挑选通过结果。新实现经合同回归和科研复审后，只能生成新的设计级数据并重新质量审计；正式 500×4 和 M0/M1/M2/M3 继续禁止。

## Step28-v13 v1.13 v8 设计构建执行失败与定点修复（2026-08-12）

固定规模修订最终选择训练／开发各 500 世界、审核甲乙各 2 世界，共 1004 世界；审核拆分本阶段只作结构和盲化链检查。提交 `ca91a10238b2e503d17ef58ef4836859a5e2442a` 经网页端审查获准执行一次设计构建。运行在临时根内完成训练集序号 0 至 68 后，于训练集序号 69、候选 0 因 `Natural candidate changed frozen profile contribution lineage` 关闭失败。临时根自动删除，最终根从未发布，因此没有数据质量结论，也没有训练资格。

同一固定权威的精确内存重放显示：基线／候选各 510 条谱系贡献，来源键缺少和新增均为 0；仅两条 `signature_title_concat` 的跨卖家文档频率及卖家数从 2 变为 1，卖家集合摘要随之改变。它们不是已登记的精确标题克隆。根因是旧自然词表双射与不同繁简作者风格交互，把基线中规范化后相同的两个普通标题拆开。卖家文档频率三项属于合同冻结字段，未被删除，也未被改成候选回退理由。

当前未提交修复新增 v8 专用纯渲染器，旧冻结渲染器保持原字节；词表／模板置换在原类内进一步按繁简转换响应和占位依赖细分。宿主完整 13 字段来源多重集仍是最终权威，任何漂移发生在碰撞分类前并终止整个构建。固定训练集序号 69 已在候选 0 直接闭合；四拆分双射／响应闭包、谱系失败先于碰撞、排名重复／缺口／布尔值拒绝、完整谱系摘要持久化和策略声明／渲染器绑定测试均通过，关键定点测试为 6／6。网页端复审判定该方案可进入完整回归，但明确指出组件级繁简响应分组不能冒充所有规范化／分段碰撞的一般性证明；完整宿主谱系硬门不得放宽。

下一步是运行完整 v8 构建合同、冻结新字节并重新取得运行前许可。随机权威、`attempt_index=1`、1004 世界规模和输出根不得变化，不得预览其余未见世界、换盐或以候选回退绕过谱系错误。当前 1004 世界设计根、质量结果、正式种子、正式数据、M0/M1/M2/M3 模型及指标仍全部为零。详细记录见 `docs/STEP28_V13_V1_13_V8_BUILD_EXECUTION_FAILURE_20260812.zh.md`。

## Step28-v13 v1.13 v8 构建修复完成本地完整回归（2026-08-12）

v8 构建修复的完整合同组首次运行 28 项，其中 27 项通过、1 项错误。错误来自旧测试把候选 0 的全部商品和卖家文档摘要同时登记为历史碰撞；这些摘要包含候选间有意保持不变的文档，因此所有 32 个候选都必然相撞。内存核对显示 32 个候选实际形成 15 组不同的商品及卖家文档集合，候选 0 与候选 1 只重合 94／125 个商品文档和 10／28 个卖家文档，生产候选机制并未失效。

测试已改为只登记候选 0 独有、候选 1 已改变的一个真实商品文档摘要。修正后定点测试证明候选 0 仅因 `historical_item_document` 被拒绝、候选 1 接受；生产实现未为刷绿而改变。v8 构建合同随后 28／28 通过，用时 203.064 秒；候选父对象、候选选择、文档碰撞和自然变化四组相邻合同 114／114 通过，用时 176.199 秒。两轮合计 142／142 通过。

这些结果只说明修复后的准确候选可以进入冻结字节和新的运行前审查，不是 1004 世界构建、数据质量或训练结果。当前仍无 v8 发布数据、质量输出、正式种子、模型或指标；必须由网页端 GPT-5.6 Sol Pro 对最终精确字节重新给出“允许同一权威重建 v8 设计级数据”后，才可复用既定权威重建一次。

网页端 GPT-5.6 Sol Pro 随后对九个最终附件独立复算哈希和策略自哈希，并完成调用链终审：Blocker 0、High 0、Medium 0、Low 1，最后一行明确为“允许同一权威重建v8设计级数据”。唯一 Low 是孤立 train/69 回归没有累计前 69 个世界的 current-dataset 登记表；审查确认这不构成错误成功路径，实际构建仍携带完整登记表并只按精确文档碰撞合法推进候选，不要求在本次重建前修改冻结字节。许可只绑定当前精确字节、attempt 1、同一公开权威、固定 500／500／2／2 和既定输出根；成功仍只能是设计级、禁止训练。外审回执见 `reports/step28_v13_v1_13_scientific_builder/external_review/step28_v13_v1_13_v8_same_authority_rebuild_external_review_go_20260812.txt`。

## Step28-v13 v1.13 v8 同一权威重建第二次执行失败（2026-08-13）

提交 `945a5129284f0035574f98c657cbe0f2e87941f3` 获准后按冻结命令重建。构建完成训练世界 0 至 28，在训练世界 29 报告 32 个精确文档候选全部碰撞。事务清理已闭合，临时根和最终根均不存在，没有发布数据、质量结果、模型或指标。

只对已暴露世界进行的顺序内存重放证明，前 29 个世界累计商品／卖家／身份登记数为 2,805／812／2,436。目标世界的 32 个候选均且仅命中一个 v1.12 历史商品文档摘要 `1b27758b380e57e90baf967db68a319540ea7909581d96aab7b4c4953ac03082`；无当前数据、卖家或同世界碰撞。对应商品的全部既有自然变化字段和最终文档在 32 个候选中都只有一个取值，故增加候选次数无效。

根因是标题修饰语没有进入候选键驱动的安全置换，而该商品的其他变化域全为单例。当前 `945a512` 字节永久不再运行。下一步只修复标签隔离的自然表达变化域并新增携带前序登记表的 0 至 29 顺序回归；完整合同与新外审放行前不再构建。正式 500×4 数据、M0／M1／M2／M3 模型和指标仍全部为零。详见 `docs/STEP28_V13_V1_13_V8_SECOND_BUILD_EXECUTION_FAILURE_20260813.zh.md`。

最小修复候选已改为复用生产链原生支持的属性候选字段，不修改生产链、模板或抽象语法树模式。旧安全库把十项属性全部单例化；新候选只允许 `标准版↔组合版`、`轻量版↔更新版` 两个版本类型轨道，其余属性不变。每个二元轨道由现有候选键选择恒等态或交换态；审查时主动撤销了“强制非零轮换”，因为它会退化为所有候选都执行同一个固定交换。映射全局、标签盲、登记表盲；完整简繁结构、跨风格相等关系和 32 个固定测试键覆盖两种映射状态由机器验证。标题修饰语备选因生产抽象语法树／登记模板门正确拒绝而撤销，没有留下代码或数据。

最终属性轨道下，已暴露训练世界 0 至 29 的真实顺序内存回归闭合：世界 29 的候选 0、1 各因且仅因历史商品文档碰撞被拒，候选 2 无碰撞接受；30 个世界的候选序号分布为 0：27、1：2、2：1，累计商品／卖家／身份登记数 2,913／840／2,520。旧全单例属性反例仍精确重现 v1.12 摘要，形成因果回归。当前尚未运行修正后完整合同或新外审，1004 世界重建仍禁止。

修正后的 v8 构建合同已 33／33 通过（371.843 秒），四组相邻合同 114／114 通过（212.419 秒），合计 147／147。运行只产生测试期小型临时目录并由测试清理，没有 v8 设计发布根或 `.building` 残留。下一步是冻结精确字节并请求网页端对同一权威重建的新许可；许可前不运行 1,004 世界构建。

## Step28-v13 v1.13 v8 第三次设计构建执行失败（2026-08-13）

提交 `38b4672843e070ac0fa9a25e331cd911093edcc9` 的属性轨道修复经网页端十附件终审确认阻断级 0、高级 0，并获一次同一权威设计构建许可。首次启动因本地执行工具误设一秒外层超时，在临时事务根建立前终止；补充审查确认它没有消耗科学构建尝试，只允许重新启动一次。

正确重新启动后，构建器顺序完成训练世界 0 至 158，在训练世界 159、全局位置 160 因 32 个精确文档候选全部碰撞而关闭失败。临时根自动清理，最终根从未发布；没有设计数据、质量结果、正式种子、模型或指标。属性修复让世界 29 在候选 2 闭合，并让世界 140 使用候选 2、世界 156 使用候选 4，但仍不足以覆盖完整设计规模。

当前提交字节不得再次运行。下一步只顺序重放已暴露训练世界 0 至 159，携带真实累计商品、卖家和身份登记表，区分六类精确碰撞并定位共同不变文档；不得预览世界 160 以后、换盐、加候选、删除历史排除或写已知摘要特例。完整记录见 `docs/STEP28_V13_V1_13_V8_THIRD_BUILD_EXECUTION_FAILURE_20260813.zh.md`。正式 500×4、质量审计、真值解封及 M0／M1／M2／M3 训练仍禁止。

## Step28-v13 v1.13 v8 三状态属性修复完成本地回归（2026-08-13）

已暴露训练前缀诊断现已闭合。世界 159 的 32 个旧候选全部且仅命中两份历史商品文档；整世界自然输出虽有 29 种，但目标“文件整理工具”只有“标准版／组合版”两个可见状态，两者均已排除。四状态跨轨道和“可选配色”方案均被原硬门拒绝。网页端纯科研设计审查允许把原安全库单例“多规格”追加到同一版本／规格内容属性槽位，但明确三者不是严格同义词，且不授权构建。

最终三状态实现保持原候选键、HMAC 域、摘要算法、轨道原顺序、候选上限、共享随机流、其他候选域和所有排除门不变。旧二态和新三态分别从空登记表、按同一训练 0 至 159 顺序独立重放：旧机制精确复现 32 候选耗尽；新机制在候选 5 接受。新机制进入目标前商品／卖家／身份登记摘要为 `c4d7fd0e...`／`ebe213a2...`／`ddfecefd...`，接受后为 `ba90909a...`／`e4a6a081...`／`e00b11e4...`。候选 0、3、4 只命中历史商品，候选 1、2 同时命中历史与当前商品，候选 5 无碰撞。

本地最终回归为 v8 构建合同 36／36（1,317.696 秒）及相邻候选父对象、自然变化、候选选择、文档碰撞合同 114／114（191.697 秒），合计 150／150；未生成设计数据。下一步仅为冻结最终字节并申请新的网页端运行前审查。在精确取得“允许同一权威重建v8设计级数据”之前，不得运行 1,004 世界构建；质量审计、正式 500×4、真值解封和 M0／M1／M2／M3 训练继续禁止。

## Step28-v13 v1.13 v8 第四次设计构建执行失败（2026-08-13）

三状态修复提交 `90b0391d80d1784f70394ba7128cb2f4a696f1c5` 经网页端终审获准一次同一权威设计构建。运行前精确检出该提交，并确认最终根／临时根不存在及入口合同 3／3 通过。构建顺序完成训练世界 0 至 223，在训练世界 224、全局位置 225 因 32 个精确文档候选全部碰撞而关闭失败；退出码 1，用时 791.8 秒。

异常事务已删除临时根，最终根从未发布。外层工具返回失败后仍有一个同一启动时间的 Python 进程，已按精确进程号终止；终止前后两类输出根均不存在，最终也无残留进程。没有设计数据、质量结果、正式种子、模型或指标。`90b0391` 构建字节永久不得重跑。下一步只能携带真实累计登记表顺序重放已经暴露的训练世界 0 至 224，区分六类精确碰撞并定位目标不变状态；不得预览世界 225 以后、换盐、加候选、删除排除或写已知世界／商品／摘要特例。详细边界见 `docs/STEP28_V13_V1_13_V8_FOURTH_BUILD_EXECUTION_FAILURE_20260813.zh.md`。

## Step28-v13 v1.13 v8 第四次失败最小修复获准重建（2026-08-14）

顺序诊断确认训练序号 224 的 32 个候选全部且仅命中历史商品精确文档排除，可见支持只有“轻量版／更新版”。后继修复没有改变基础模板或基础采样，只新增 21,879 字节、SHA-256 `ff97b59f0d66d7a9d18a3b1e7d7db684e6bdcc1a7ae60ba0e6b7294528c8766a` 的候选专用模板，在属性数组末尾追加“通用版”。删除该末尾值后，其规范 JSON 与原基础模板完全相同；旧生产链和旧候选父策略保持原字节。

旧状态从空登记表顺序重放训练序号 0 至 224，精确复现序号 224 的 32 候选耗尽；新状态从独立空登记表重放相同 225 个世界，序号 224 在候选 0 六类碰撞全空接受，8 条可见商品文本和 8 条私有渲染语法树均实际含“通用版”。旧新共同前缀的结构父投影和 33 项身份历史特征序列逐项相等；序号 0 至 159 的上一轮因果回归仍闭合。v8 完整合同 38／38 通过，用时 2,553.143 秒，两次四拆分小型烟雾构建内容一致；没有设计发布根、正式种子、模型或指标。

网页端 GPT-5.6 Sol Pro 对九个精确附件独立复算，最终判定阻断级 0、高级 0、中等级 0、低等级 3，并明确“允许在同一既有 v8 设计权威下重建 1004 世界设计级数据”。20,983 字节外审回执 SHA-256 为 `a2b0767bd5c689594e1be362d94550ad8a18378759eed3e1f5198fb847741307`。三项低等级建议只涉及测试钉住增强，为保持本次许可精确字节不在重建前修改；必须在正式 500×4 冻结前补强并重新审查。下一步只允许从空登记表、按固定 500／500／2／2 顺序完整重建一次设计级数据；成功也不代表质量通过或训练授权。详细记录见 `docs/STEP28_V13_V1_13_V8_FOURTH_BUILD_MINIMAL_REPAIR_REVIEW_20260814.zh.md`。

## Step28-v13 v1.13 v8 第五次设计构建执行失败（2026-08-14）

获准字节冻结为提交 `5b02cd00af3d2ca2881b58349c61411973711f77` 后，从空登记表运行唯一 design-preflight 命令。训练序号 224 正常越过，说明第四次失败定点修复生效；构建随后完成训练序号 0 至 282，并在训练序号 283、全局位置 284 因 32 个预定义候选全部发生精确文档碰撞而关闭失败。退出码 1，用时 880.4 秒。

事务清理后临时根与最终根均不存在，没有 Python 残留进程；未发布的 283 世界不得恢复或拼接。当前尚未查明序号 283 的具体碰撞类别，不能把它武断归因于与序号 224 相同的属性支持耗尽。提交 `5b02cd0` 的构建字节永久不得重跑。下一步最多只允许携带真实累计登记表顺序、内存重放已暴露的训练序号 0 至 283，先完成六类碰撞诊断，再决定是否存在新的全局因果修复；不得预览序号 284 以后。质量审计、正式 500×4、真值解封和 M0／M1／M2／M3 训练继续禁止。详细记录见 `docs/STEP28_V13_V1_13_V8_FIFTH_BUILD_EXECUTION_FAILURE_20260814.zh.md`。
## 2026-08-14：v9 训练序号 0—283 内存因果回放通过

v9 可见文档容量修复已完成运行前闭包。策略校验、聚焦合同 16 项、候选父合同 25 项、文档碰撞与自然变化合同 58 项均通过；网页端 GPT-5.6 Sol Pro 对最小运行许可包复核后给出四级问题均为 0，并明确允许一次训练序号 0—283 的内存因果回放。

唯一回放用时 851 秒、退出码 0，连续处理 284 个训练世界。未来训练世界、开发／审核世界、写入数据行、正式种子、模型和指标均为 0。28,692 个商品代码与 28,692 个商品文档一一闭合，7,952 个卖家文档闭合；284 个世界全部在候选 0 接受，六类精确碰撞总数均为 0。目标训练序号 283 只检查候选 0 即通过。最终规范结果对象摘要为 `ad42b965467a3cfb1e280a6ec9c36ead92507e8017ec4b9437014eca4b509a0f`。

该结果只证明 v8 已暴露序号 283 的文档容量缺陷被结构性消除，不是数据质量、无捷径、正式生成或训练结果。下一步是在任何 1,004 世界重建前机器冻结 v9 质量审计和 M0／M3 敏感性合同；正式 500×4、真值解封与 M0／M1／M2／M3 训练继续禁止。完整记录见 `docs/STEP28_V13_V1_13_V9_CAUSAL_REPLAY_RESULT_20260814.zh.md`。

## 2026-08-14：v9 质量通道实现基线通过终审

v9 已完成三种冻结文本视图、2,992 项公共编码特征、388 项私有解码槽位特征及物化接线的实现级闭合。代码中和链只接收五项无文本商品元数据和去代码语法树；先把临时唯一代码折叠为统一占位符，再从最终中和商品重新计算完整卖家画像。公共探针不再暴露语法树节点、商品编号、跨度或摘要；运行时字段读取防护、最终中和商品和重算画像的 256 状态承诺均有正反例测试。

网页端第一次终审指出权威来源表遗漏两项质量门、部分重抽样常数及逐类章节语义检查。当前修复把六项质量门全部登记，把完整 24 字段重抽样区块逐字段镜像，并对七类来源的路径、文件哈希、章节和本地键做精确语义钉住。新增变异测试会在篡改章节后重算政策自哈希、同步验证器预期哈希并关闭文件钉住检查，仍要求语义校验拒绝。

当前合同为 29,580 字节／`bdb093a33e9cb19d12f24e4f6636dd2269609ee73481260dba67d2069ef19bb5`；质量政策为 20,082 字节／`0d6f65854bfffdfa3286b263a4b3ccdb3b8ffc3f6c46599602ea1bec49f45f3b`，规范自哈希 `095b30cdb744b5cff96c10f8e7bf2d2d39cbb6caae89e77c226faac5132ed502`；验证器为 26,904 字节／`67eb0fe6a71fd35784a97f64f41e1b6dc572e712ab76690d3ebd8c3233ee5317`；物化器测试为 22,447 字节／`3f47155ecf894a9a6e36cd0ef06a5f42097339f544c93c6759137cf6226234fb`；外层科学政策为 16,137 字节／`612dbcc3ed2e57ba14e25386935de0654da5e851762d74c66f54fea0e6655008`，规范自哈希 `b68eaa2ecb8410f2bc1a26f14869ab46bd7cab5d8965c5a320331b8aa8ab6951`。

本地聚焦、接线及冻结相邻合同合计 163／163 通过。网页端定点复审独立复算上述字节和两层自哈希，并逐项变异六项质量门、24 项重抽样值和七类章节，最终为阻断 0／高风险 0／中风险 0／低风险 0，明确允许把当前字节作为继续实现剩余质量审计模块的冻结基线。该放行只允许实现和临时夹具测试；1,004 世界构建、正式质量审计运行、正式种子、正式 500×4、审核真值解封、训练和指标仍全部未授权，现有正式数据行、模型及指标仍为 0。

## Step28-v13 v1.13 v9 质量调用链终审与历史回归清理（2026-08-15）

v9 剩余质量调用链已经实现并补齐真实生产路径反例。训练／开发真值只在无标签矩阵和资格位图冻结后由监督器内部一次性打开；审核甲乙零打开。六份实际视图的逐世界摘要、数量、克隆、资格排除和卖家标识权威全部重算。物化器真实九字段非编码投影承诺与结构汇总器已经兼容；商品数在中和编码族列表构造前受 28×8 上界约束；正式 JSON 行的键排序往返不再被误判为模式漂移，缺键／多键仍关闭失败，CSV 和科学行序仍保持严格。

受影响模块 61／61、全部 v9 质量合同 113／113、v9 科学构建合同 19／19，合计 132／132。质量政策和外层科学政策规范自哈希分别为 `02ae9c36ee84304053c5dcea9f2113ad1bfebfbc2cf0bb21242f0c45e2f83cf9` 与 `ffc8b276a835b8c2168ab55cc1054a5eb602ffd5c3543ded6733847030f5f7c5`。网页端最终 13 附件复审为阻断／高／中／低均 0，只允许提交候选并进入 1,004 世界设计根重建授权准备。

随后全仓 804 项回归产生 2 项失败、33 项错误、7 项跳过。精简重跑确认旧 v7 质量模块因已删除且禁止恢复的 `design_preflight_v2_20260811` 独占 32 项错误和 1 项锁等待失败；未提交 v8 临时测试产生 1 项导入错误；Step28-v12.1 历史同步测试仍要求本进度文档回到旧快照，产生剩余 1 项失败。禁止恢复失败根或回退活文档。v7 可执行机器合同与无证据 v8 临时文件按既有终止记录退役，只保留文档、小失败回执、外审证明和必要公开回执；Step28-v12 精确历史同步测试改为明确跳过。本段为清理前诊断，清理后的真实结果见下文。

正式设计根、正式种子、正式数据、质量指标、审核真值读取和 M0／M1／M2／M3 训练仍全部为 0；当前不授权立即重建。

清理后的第一次全仓回归运行 753 项，用时 1,956.041 秒：744 通过、8 项声明跳过、0 失败、1 错误。唯一错误来自已永久失效的 v1.12 历史预锁继续解引用旧 `.gitignore` 文件钉；该缺陷已于 2026-08-12 记录，当前私有保管排除规则不得回退，旧预锁不得修补或恢复资格。只把这一项过时的旧执行资格测试改为说明原因的历史跳过，其余 v1.12 合同保留执行。两个相关历史模块随后为 24 通过、2 跳过、0 失败或错误。最终全仓回归运行 753 项，用时 1,461.387 秒：744 项实际通过、9 项按记录跳过、0 项失败、0 项错误。来源守卫的一世界开发烟雾测试通过且未保留临时工作区；聚焦合同为 59 通过、1 项按记录跳过、0 失败。v9 候选现已取得仓库级回归通过证据；这仍不授权设计根重建、真值解封或模型训练。

## Step28-v13 v1.13 v9 入口资格复审阻断与最小修复（2026-08-15）

提交 `eaec3c64c81cc5ecf8641ecca2f283af9a0e8ddd` 的 17 附件复审发现提交前删除 Markdown 日期行末两个空格后没有重跑策略验证：质量通道合同实际为 29,578 字节／`18f600b62b2b46adacab59bcca49d75b2e2694cd8bb26ec7caa98e752cb3393e`，内外策略仍钉住旧 29,580 字节；文档容量修复合同也发生同源两字节漂移。网页端结论为阻断 1、高 0、中 2、低 0，不允许实现一次性入口。该提交保留为已推送历史，但不得作为入口实现资格基底。

当前修复把两份机器钉住文档的实际字节级联到内外策略，并补设计根对构建策略规范自哈希和文件路径／大小／完整哈希的双重记录。质量运行器在任何拆分视图加载前反向核对；篡改负例证明失败阶段为 `builder_policy_binding`、拆分加载调用数为 0。内外规范自哈希现为 `8833ffab07051684bf0ce997ef603b9d2a2fd5d54660f2ff129f6f1c49489c10` 与 `219a74bef933deaf9dadfc230c8372abdc71d9d2b86cdb19917ef75e99a2f13a`；直接策略验证和运行器 10／10 通过。全部 v9 质量／科学构建合同为 114／114 与 19／19；全仓运行 754 项，用时 1,485.095 秒，745 项实际通过、9 项按记录跳过、0 失败或错误。写盘入口仍硬关闭，必须完成新一轮网页复审后才能决定是否实现；实际 1,004 世界构建、质量运行、真值打开和训练均未授权。

修复后的来源守卫开发烟雾测试通过；聚焦合同重新运行 60 项，用时 251.132 秒，59 项通过、1 项按历史合同跳过、0 项失败或错误。没有生成设计或正式数据。

## Step28-v13 v1.13 v9 一次性设计构建入口实现完成（2026-08-20）

提交 `403a97c52918a1c65792cca23853a46979ad5d19` 经网页端复审关闭上一轮阻断项；审查为阻断级 0、高级 0、中等级 1、低等级 0，最后一行精确为“允许实现但不得运行一次性1004世界设计构建入口”。剩余中等级意见要求把通用执行模式接口收缩为固定入口，不阻断实现，但明确禁止实际运行。

当前实现只保留无参数设计构建入口，不再从命令行接收模式、输出根、拆分、起始序号或世界数量。外部一次性回执精确绑定构建策略、质量策略、构建器源码、Git 提交与树、固定 500／500／2／2 世界、输出根、尝试编号和随机权威承诺。回执必须在建立临时根或写入首个数据字节以前原子消费；待用回执路径不能进入私有事务，已消费回执不能复用，仓库内没有回执生成器。成功根只允许声明设计构建通过且不具训练资格，并保存完整策略、源码、Git、随机权威和回执谱系。质量运行器在读取拆分视图前反向核对上述谱系和全部零授权边界。

外层构建策略规范自哈希为 `cf8a53153887b54e45d40b29bce33214203e2792097dd31d1964ad8bf6a373d3`，质量策略规范自哈希为 `3147c01032aed44f6465ac9db654c944c538fb326d0ee0234773a36eae32c9f9`。科学构建合同 25／25、全部 v9 质量合同 114／114，合计 139／139 通过。全仓回归运行 760 项、用时 1,530.666 秒：751 项实际通过、9 项按记录跳过、0 项失败、0 项错误。来源守卫一世界烟雾测试通过并清理临时工作区；聚焦合同运行 60 项、用时 255.576 秒，59 项通过、1 项按历史记录跳过。测试缓存已清理；正式设计根、`.building`、授权回执、正式种子、数据、质量指标、模型和训练指标均不存在。

下一步只允许提交推送当前精确字节并申请一次实际运行许可。取得最后一行精确为“允许运行一次同一权威1004世界设计构建”的新审查以前，不得创建回执或运行入口；质量审计、正式数据、真值解封和 M0／M1／M2／M3 训练继续禁止。

## Step28-v13 v1.13 v9 同一权威 1,004 世界设计构建完成（2026-08-20）

提交 `c17852d24610e365611dc3bc662bebb8c637cc8d`、树 `a6d9226d053ceb3506bfcac0890978c10e9de332` 的精确入口字节经网页端 GPT-5.6 Sol Pro 重新审查。第一次回复截短两个哈希，未被采信；修正回复重新读取 10 个附件，全部文件大小和哈希与本地一致，四级问题均为 0，最终一行精确为“允许运行一次同一权威1004世界设计构建”。修正回复摘要为 `414d8aeb3372f5aae8e5c6dc259fcfca269defdc4fb856c2ed63e771205fd0ff`。外部一次性回执为 2,010 字节／`69878cee771a030aa8348504131454d24e4188af37194f8e766762440ea6e373`，在任何数据字节写入前已原子消费，不能复用。

无参数入口退出码为 0，完整构建训练 500、开发 500、审核甲 2、审核乙 2，共 1,004 个世界。全部世界均在候选 0 接受，六类精确碰撞拒绝数均为 0。设计根 `reports/step28_v13_v1_13_scientific_builder/design_preflight_v9_20260814` 的状态为 `PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED`，规范自哈希为 `2d453eee6a44ea57fedfe8dd05b28c72b92ed3496f219db1b3d727de9d7969cd`。根内有 379,512 个配对（20,080 个正例、359,432 个负例）、28,112 个卖家、101,119 个商品代码和商品文档、84,346 个身份值。训练、开发、审核甲、审核乙拆分清单自哈希分别为 `3295bfbf2521eb3cd8f7ef74a53d8552bbe808624a8fc4754305edf5acb5f651`、`4d55cbb5a413fb96b17a5290c0ac30951d86527802056cf872e47f07ace7a8f3`、`3daa4a516e0414c36a0d61fdc992d9d344c57eb8f1652261df2dace495d47fd0`、`5a32a4097949e3c17d731e4e2b75f3cf8c64ea823b7ea75554489ba3c81da61f`。

独立逐文件复核确认 72 个数据文件共 1,464,127,328 字节，整个设计根 77 个物理文件、1,464,155,768 字节；根／拆分自哈希、清单绑定以及每个文件的大小和完整摘要全部闭合，临时 `.building` 不存在。该成功设计根是下一阶段质量审计的输入，必须保留。

本轮只证明设计构建机制和容量闭合，不是数据质量、无捷径、身份信号或模型效果结论。质量审计尚未运行；正式种子、正式数据行、真值读取、模型、训练和指标仍全部为 0。下一步先提交推送该设计根与同步文档，再针对这个精确根取得新的质量审计许可。取得许可前不运行质量审计，不生成正式数据，也不训练 M0／M1／M2／M3。

## Step28-v13 v1.13 v9 质量审计授权转换实现（2026-08-20）

成功设计根和记录已以提交 `8bb3276de6d84c0aad8d7475af5ca0b41a86b959` 推送并与远端一致。网页端 GPT-5.6 Sol Pro 随后读取最新交接、冻结质量政策与运行代码、根和四拆分清单，独立复算附件和清单摘要；它未声称复算未上传的 72 个数据文件。审查为阻断 0、高 0、中 1、低 0，回复 8,358 个 UTF-8 字节／`812ecb2be9ac9846a52ca905fa05ea29c0c0f6e4df62adead24d8515a515b21e`，最后一行精确为“不允许直接运行一次冻结质量审计”。

中等级问题是实际授权死锁：冻结政策的质量审计和指标位必须保持假，原公开入口却要求二者为真；成功根又钉住该政策完整字节，原根绑定还要求授权位继续关闭。不得直接改政策、改根或重建设计数据。

当前新增一个不修改冻结输入的一次性覆盖层。合同、机器政策、无参数入口和合同测试分别为 5,071／5,736／29,793／26,782 字节，文件摘要依次为 `faacaa1a3d17b6f8d2a9a00d24ffd3bf6b25aa76ff8cfae80bb23e847597b7d9`、`c73d85c09ff37cf4e2876ee017751bfd483dc6ed638ed2e594988cd2ffc5d266`、`b2005119f53acdf177db6ccef60ab57640bdc994c4acd054fca500c523b6f813`、`81434182da102f58cc90ae7bd2f2b87a79815e37d4b43d681e4621e2abd7bf66`；覆盖政策规范自哈希为 `9947f02b789045ea99920dbe08996a16f1b4223da0da7cd7f065000a91e021c7`。原质量政策、原运行器、原验证器和成功根清单的字节与摘要均未改变。

覆盖入口只接受仓库外生成的一次性回执。回执必须绑定复审回复、当前 Git 提交和树、全部固定代码／政策／清单、唯一结果路径和仅两项质量能力；在首个设计视图或真值读取前原子消费，之后把不变的冻结政策交给原内部审计体。审核甲乙真值、正式生成、训练和模型指标始终关闭；结果用排他原子发布，不能覆盖已有回执。自审已修正机械故障误分类、夹具绑定共享引用和结果覆盖风险。

新增覆盖合同最终 21／21 通过；全部 v9 合同最终 160／160 通过，用时 417.513 秒。最终全仓回归运行 781 项、用时 2,137.971 秒：772 项实际通过、9 项按既有历史记录跳过、0 项失败、0 项错误。测试结束后没有质量结果目录、待用或已消费质量回执、`.building`、历史负例测试临时目录或字节码缓存残留。

当前仅允许实现和测试，仍不允许运行。下一步提交推送当前精确字节，并重新申请一次精确质量审计运行许可；正式数据、审核甲乙真值、模型和指标仍全部为 0。

## Step28-v13 v1.13 v9 质量审计显式执行适配器修复（2026-08-21）

第一版覆盖入口提交 `0ec9bde61b3aa8b52a31ea7454178eba0bb9ca0a`、树 `5d0522c67d39531dbcbb6ca6159ded6147d07e6b` 的运行许可复审被否决。网页回复 18,749 个 UTF-8 字节／`4d7be79e065cb7d2805ba0c8e68f223a3076c1b5d2fbef75e48d845c82bc6bd0`，问题数为阻断 1、高 0、中 1、低 0，最后一行“不允许运行一次冻结质量审计”。阻断原因是冻结政策没有当前设计根钉，而原测试模拟了私有审计体；中等级问题是回执消费后覆盖入口机械失败没有规范终态小回执。

本地进一步证明只补根钉仍不够：冻结政策校验要求质量授权位为假，原监督验证器随后又要求同一字段为真，形成第二层确定性死锁。后继实现没有修改成功根、四拆分、冻结政策、原运行器或原验证器，而是新增显式执行适配器：科学参数只取自不变冻结政策，一次执行能力只取自已消费回执，根钉只取自覆盖政策。不可变执行上下文精确绑定三者；适配器复用冻结加载／结构／特征函数及数值辅助函数，不使用运行时替换。

未模拟的接线测试实际越过旧根钉门并到达 `builder_policy_binding`；监督接线测试越过旧授权门后只在故意为空的矩阵数量门停止。固定三世界数值等价合同确认新旧核心的逻辑回归、浅树、单特征、预测摘要、重抽样和质量门逐字段一致。消费后的覆盖入口故障现在排他发布哈希化终态回执，仍禁止重试、行级输出、正式生成、审核真值和训练。

精确边界为：合同 6,875／`cd1736a809fc88e523ef2a8e8a4abc088a4118c5300ddb9690d1be3f8abad3b4`；覆盖政策 6,201／`83f13a95b53fb8d52c77edf3299e634b33bf95e4d727225d13251eebc07d52c8`，规范自哈希 `0b3df1dac378aa770e67f609800113825fc4ab022e47edb8aa966d780bb1c86d`；执行适配器 38,994／`f2c434f9ac611708075cc75b5485abce84e604b2b1c0c5035ad055f52c84a35f`；入口 35,755／`f2dc1bc703191b1b9e3b810f88e91654ca10f6ce4b7c0c97486ed62b8e86066c`；测试 39,572／`47d724cf823e4e991d16759ea0ef7b2f528279779bb93d0d13a9641a9080506a`。

覆盖合同 27／27、全部 v9 相邻合同 165／165 通过；全仓回归 787 项、用时 2,129.078 秒，778 项通过、9 项历史跳过、0 失败、0 错误。测试后质量结果、待用／已消费回执、`.building`、失败夹具目录和缓存均为 0。当前仍不允许运行质量审计；下一步只允许提交推送并重新网页复审。正式数据、审核甲乙真值读取、M0／M1／M2／M3、训练和全部模型指标仍为 0。

## Step28-v13 v1.13 v9 第三版质量审计运行前修复（2026-08-21）

提交 `ff7fc6038e755d389449757e52850b4988f297af`、树 `9cfd78225e7400fcd15ee20082c87d5f37b6bfce` 的第二版适配器经提交后网页复审否决。网页正文 24,354 个 UTF-8 字节／`11d8068cf5e76ca6940a511718cd41bd8bf90dc5cdff2b45370902a97c2d9262`，问题分级为阻断 1、高 1、中 2、低 0，最后一行“不允许运行一次冻结质量审计”。该提交没有创建或消费回执，没有运行审计，永久不得运行。

阻断根因是把仓库相对完整根清单路径误传给只接受数据根内 `root_manifest.json` 的正式真值能力。其余问题是适配器没有重验实际已消费回执文件、终态发布遇到已有不同结果时静默返回，以及合同把实际验证范围夸大为 72 个文件全部重哈希。本地逐项确认这些问题成立，没有照单接受超出科研调用链的泛化安全建议。

第三版把科研谱系路径和根内真值钉分开；适配器必须重验真实已消费回执的文件、规范自哈希、能力、根、结果路径、政策和当前适配器绑定；已有结果只有逐字节等于精确终态时才可接受。结果中的文件范围固定区分为 72 个清单声明文件、46 个实际逐字节验证文件和 26 个仅清单钉住文件。46 个实际文件由 44 个标签无关输入与训练／开发两份真值组成；审核甲乙真值保持零字节读取。

当前精确字节：合同 8,739／`7fa8ed156816edffc0f8f605a65439d333284573ccedfa12b14f81f297533ebd`；覆盖政策 6,201／`495bf993dab3aa9896b39feec018ce0049c58b58367d23c226193c3f8af4de12`，规范自哈希 `1c0c73d55019680c1a151e323850c901693e7b113fab2569609e7f058d62dbba`；适配器 50,829／`065cfc1c700ec86663d796777c353eaefc55c7fdfc9ea6657637bc0bbcdacf6c`；入口 36,390／`7fb175c0e2bbf4004feb24ffff2b9d10586f0cc1b4698c2f36a3a2081b842384`；测试 48,059／`3065b26859b65885804dffa3bacedcef3e8d932823c06dda585ca8fdd585e0aa`。专项合同 31／31、相邻 v9 合同 170／170 通过，用时 1.731 秒和 407.548 秒。全仓回归运行 791 项、用时 1,881.674 秒，782 项实际通过、9 项历史跳过、0 失败、0 错误。运行后质量结果、临时结果、待用／已消费质量回执、历史负例临时目录和字节码缓存均为 0。当前候选尚未提交或取得网页运行许可；质量结果、正式数据、审核甲乙真值读取、M0／M1／M2／M3 和全部模型指标仍为 0。

## Step28-v13 v1.13 v9 第一次质量审计执行基础设施失败（2026-08-21）

第三版已以提交 `15fa782ec1552f0907d4f054c0bed51042443e0f`、树 `9b2804e2f8e256aff79802c2f5e55c82e60b0013` 推送。网页复审回复为 21,470 个 UTF-8 字节／`5928d54080d5aef8abfe4e7f7e76a2229000c519f5a72afab5013511e48a9b6a`，分级阻断 0、高 0、中 0、低 1，最后一行“允许运行一次冻结质量审计”。一次性回执编号 `8de20c80774674e08c401f8b82ab36d81d5baf9f48dc1a50231afa8f9b2a0154`，4,477 字节／`ca31ba9183ca17c4a7c2b1a95256760f78cca43daa5e0ebd390c0a9addd97e54`，已经原子消费且永久不得复用。

唯一无参数运行发布终端回执 `reports/step28_v13_v1_13_quality_audit/v9_design_preflight_20260820/quality_audit_receipt.json`，5,786 字节／`22cec5c0ad0fc9b1f076ae8b26cae3eb47bd28168787050d53a1cb6805a51e4b`，规范自哈希 `6bce931f891de75185e969d7b232f4358e29844c49030f7c06b6309de0c163bc`。虽然终端表面写为 `DATASET_INVALIDATED`，异常摘要 `b9b7f9e7eedca8afa55f513c1c9b252c362a3907f553c32b04b447e723ac8bfa` 精确对应“正式质量审计仍未授权”。适配器把运行授权位固定为假的冻结政策传给要求该位为真的结构聚合入口，审计在实际结构统计前退出；外层又错误地把基础设施异常分类为数据失效。

科研结论只能是“第一次审计执行失败，数据结论无效”。V9 设计根不是失败根因，继续保留为未获训练资格的设计输入。训练／开发真值未打开，审核甲乙真值、正式数据、训练、模型和指标仍全部为 0。不存在临时或大型失败输出；仅保留终端小回执和已消费回执。完整失败与清理边界见 `docs/STEP28_V13_V1_13_V9_QUALITY_AUDIT_ATTEMPT1_INFRASTRUCTURE_FAILURE_20260821.zh.md`。当前禁止重跑；必须先修复结构能力传播和错误分类，增加真实生产结构入口测试，完成完整回归、提交和新的网页复审。

## Step28-v13 v1.13 v9 第二次质量审计尝试运行前修复（2026-08-21）

修复只处理第一次执行暴露的授权接线和错误分类，不改成功设计根或任何科研参数。适配器在已消费回执能力下直接调用冻结结构聚合核心 `_aggregate`，全部世界数、卖家数、位置偏差阈值与声明边界仍逐字段来自字节不变的冻结政策；调用前后重验回执、执行上下文和政策。生产路径不再调用要求冻结政策授权位为真的旧结构包装器。授权接线错误归为审计器失败，真实结构数据错误仍归为数据失效。

第二次尝试固定编号 2、待用回执 `private_custody/step28_v13_v1_13_v9_quality_audit_attempt2_authorization.json` 和新结果目录 `reports/step28_v13_v1_13_quality_audit/v9_design_preflight_attempt2_20260821`，不能复用或覆盖第一次回执与结果。精确文件为：合同 11,223／`652642b4ae9bccb2c4de5e80d50988a7ba1ca8400f4777cdb7442a5539699606`；政策 6,256／`bc18616467cf64571e90e2bedb574c836653bacc5491d06dd9cc62d1ade21cc8`，规范自哈希 `24b99910b1004da144a249817047d39684425c0bcebf505f03b64bbed0aee9b3`；适配器 52,788／`254d706b452253801ea367011e87040e35bc46ed1416c624bfe1b5d7286f9108`；入口 36,755／`cb7943877bb85930a8878a8804af93c967ca8424f73079cbdd03dc479fdc1658`；测试 53,541／`2c83915fbc85ebeb11f3e393250f0d05ff18495b42791f6e8f76b5ee0c832eae`。

聚焦合同 36／36、相邻 v9 合同 175／175、全仓 796 项均完成；全仓为 787 项实际通过、9 项历史跳过、0 失败、0 错误，用时 1,844.458 秒。缓存、夹具、第二次回执和第二次结果均为零。当前只允许提交推送并请求网页端复审；没有新的“允许运行一次冻结质量审计”以前不得运行。正式数据、审核甲乙真值、训练、M0／M1／M2／M3 和模型指标仍为 0。

## Step28-v13 v1.13 V9 第二次质量审计数据失效与 V9.1 修复（2026-08-21）

第二次候选以提交 `8b00c172560959ffce665418933b42b3d7292470`、树 `0136cb703d7ca9185da224a13e54ad4209211a72` 接受网页端复审。回复为 20,383 个 UTF-8 字节／`73a4289d1656e74153515ed9b99928d511545a553699aaf056a139f55606e451`，四级问题为 0／0／0／1，最后一行“允许运行一次冻结质量审计”。一次性回执编号 `cc8ba6f28c35c00c9d5d3ae3f8e6a240e3ceb330568a81b373a2189fdd2613d8` 已消费且永久不得复用。

唯一运行发布 `DATASET_INVALIDATED` 终态，失败阶段为 `loaded_model_view_structure_binding`。独立逐世界重算确认 1,004 个世界的完整、代码遮蔽、代码中和卖家画像共 3,012 个摘要全部不匹配，商品摘要全部匹配。根因是结构物化器对投影前内部画像计算摘要，而写盘器保存冻结模型精简投影；审计器按实际落盘字节复算，因此正确拒绝。失败发生在监督真值能力构造前，训练／开发标签和审核甲乙真值均未打开；质量统计、正式种子、正式行、模型、训练和指标仍为 0。

V9 根被压缩为 23,733 字节等价承诺：68 个数据文件必须逐字节不变，四个结构文件只允许每世界三项画像摘要变化。只读反事实精确复现 3,012 个旧错误，按实际落盘投影重算后归零；四份非画像字段流和旧画像承诺流全部闭合。失败边界与修复合同已提交 `08cc987`。随后删除失效根 77 个文件、1,464,155,768 字节，并删除两次审计专用覆盖政策、适配器、入口和测试；两份小终止回执、已消费回执、文档和等价承诺保留。

V9.1 仅把唯一卖家画像投影提升到科学公共模块，由物化器和写盘器共享；构建后强制验证同一随机权威、68 个文件完全相同、四个结构文件只改三字段，并从实际写盘画像重算摘要。聚焦回归 52／52 通过，用时 401.849 秒。清理旧一次性入口后，全仓回归运行 762 项、用时 1,892.338 秒：753 项实际通过、9 项按既有历史记录跳过、0 失败、0 错误；相较上一轮减少的 34 项全部属于已删除的旧覆盖入口合同。回归后无字节码缓存或 `.building`。当前仍为实现级候选：提交和新的网页端复审尚未完成，不得构建新的 1,004 世界根，更不得正式生成或训练 M0／M1／M2／M3。

## Step28-v13 v1.13 V9.1 提交后复审否决与四路径等价门修复（2026-08-21）

初版 V9.1 以提交 `9ab0723bd3221744fe09fc66dec68cb55815c9b5`、树 `ab0e36e99eb29e5a1f707090bbc2428a2ba449d1` 推送后接受网页端 GPT-5.6 Sol Pro 审查。回复为 19,289 个 UTF-8 字节／`cd11c2818632651b167ca33df73ddb540729e6bf3ca0a440f0e47c9bbaaa8023`，分级阻断 1、高 0、中 1、低 0，最后一行“不允许运行一次V9.1同一权威1004世界设计构建”。审查正确发现中和画像摘要同时写在顶层和 `neutral_receipt` 内层：生产修复实际改变四个 JSON 路径，但旧等价门只排除三个顶层字段；此外两个持久化版本字段也被源码版本连带改变。没有创建构建回执或数据根，`9ab0723` 永久不得运行。

修复后每个结构行只允许 `/full_profile_sha256`、`/masked_profile_sha256`、`/neutral_profile_sha256` 和 `/neutral_receipt/neutral_profile_sha256` 四个精确路径变化，它们仍只代表三类语义摘要。顶层 `version` 与内层 `neutral_receipt.version` 均冻结为历史值 `2026-08-14-step28-v13-v1-13-quality-channel-materializer-v9`；V9.1 源码版本不再进入持久化结构行。等价门递归删除且仅删除四条路径，并分别核验内外中和摘要相等、世界顺序、递归映射键顺序、四拆分 68 个冻结文件记录和实际写盘三种画像。允许路径集合被扩张、删除或改名都会失败。

只读本地 Git LFS 历史对象生成的 V2 紧凑承诺为 26,598 字节／`bd1bafffa4913b4f81d233c391f40e13a3e62d2644848e20d2108d5d676a08dd`，规范自哈希 `eb531b02ce1f74eefe7a2e4f6b3fd9ee3b5b4712d26fe17739e12beb8aa76ab6`。质量政策规范自哈希为 `aec508b71a8df9da7f65dcd9eb39798d4858dc92316b694fa404292819ca4f3f`，科学构建政策规范自哈希为 `bdd5a85076865ba02d1e6fc9b4adc62afec16c11496cf53a17a85ad96f4a8ef6`。直接正反例、真实承诺和物化绑定共 8 项通过；聚焦回归 60 项用时 252.177 秒，59 项实际通过、1 项既有跳过；全仓回归 768 项用时 1,562.504 秒，759 项实际通过、9 项历史跳过、0 失败、0 错误。

回归后确认 V9.1 设计根、一次性构建回执、质量结果、正式种子、正式数据、标签或审核真值读取、模型、训练及指标均为 0。当前字节尚未形成新提交，也没有取得新的网页端运行许可。下一步只能提交并推送本轮精确字节，再进行新的提交后网页复审；没有精确放行句以前不得构建 1,004 世界。

## Step28-v13 v1.13 V9.1 同一权威设计根构建成功（2026-08-22）

四路径等价门修复以提交 `8f87751fb71cf40b5eac3e9540dcfc847cb82ffa`、树 `e666b409f405f6741d5a55472789aee24a0e73cc` 推送。网页端 GPT-5.6 Sol Pro 完整读取 16 个附件；回复为 17,515 个 UTF-8 字节／`af7fda62981b3e2bdf9855d6e74531dc68f27b6296f9d2c23a266b92f7297490`，分级阻断 0、高 0、中 0、低 1，最后一行“允许运行一次V9.1同一权威1004世界设计构建”。唯一低级项只涉及两个失败分支可增加更孤立的测试，生产实现判定正确，未修改获准字节。

仓库外 2,018 字节一次性回执／`a3c7a0dc750bd098d72e6d17db66a1dfbcabe2e226417260bb86845855677b9d` 经唯一无参数入口消费。构建完成训练／开发／审核甲／审核乙 500／500／2／2 个世界并正常退出。发布根含 77 个文件、1,464,158,039 字节，根清单规范自哈希为 `f10086faa5f68b08a4d25a6e49943fb18ede0858ca50bad711d7bb2f4d94200f`。发布后全量重读 1.46 GB，逐文件大小、摘要、行数及清单自哈希均闭合；临时根和待用回执均不存在。

新根合计 1,004 世界、28,112 个卖家、379,512 对，其中正对 20,080、负对 359,432。V9.1 等价回执为 `PASS_EXACT_MECHANICAL_PROFILE_COMMITMENT_REPAIR`：同一随机权威为真，68 个文件逐字节不变，四个结构文件只改四条已登记画像摘要路径，持久化版本、世界／键顺序和其余字段不变，实际落盘画像重算闭合。

这仍不是训练集或质量通过结果。根状态为 `PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED`，`scientific_use_forbidden=true`、`training_started=false`，正式种子、正式行、审核甲乙真值读取、模型和指标仍为零。下一步必须先冻结并获准运行一次只针对该根的质量审计；质量通过前不得正式生成或训练 M0／M1／M2／M3。完整记录见 `docs/STEP28_V13_V1_13_V9_1_DESIGN_BUILD_RESULT_20260822.zh.md`。

## Step28-v13 v1.13 V9.1 冻结质量审计实现与回归（2026-08-23）

当前只为成功的精确 V9.1 设计根实现一次冻结质量审计接线，不修改根、四拆分、冻结质量政策、探针、阈值、重抽样、世界、候选或标签。新增覆盖政策、执行适配器、无参数入口和合同测试；外部一次性回执必须在首个设计视图前原子消费，并完整绑定 Git、入口、覆盖政策、适配器、冻结源码、根和四拆分。仓库代码不能生成回执，正式种子、正式四拆分、审核甲乙真值、模型训练和模型指标能力全部关闭。

候选实现网页复审正文为 23,362 个 UTF-8 字节／`7ea8b1d3165caefe8b4c230255cf2f538db893849d4d89d1cf14170dd868e94d`，最后一行“候选实现不能进入本地回归与提交前修正阶段”。本地独立复核接受并修正其实际科研问题：直接适配器调用改为复用入口完整回执模式并重验 Git 与全部文件；最终结果采用精确模式和逐层自哈希；真实 V9.1 标签无关四拆分实际进入冻结结构核心；伪授权数值测试改用冻结合法夹具；补充严格布尔类型、尝试编号、会话网址、72／77 文件计数、探针后再闭合和机械故障分类。未采用超出唯一科研入口所需的持久化多阶段锁状态机，避免重新扩张到系统安全支线。

聚焦合同完成三轮 44／44，用时 42.380、42.942 和 40.564 秒；第三轮确认记录全仓结果后更新的覆盖政策摘要、规范自哈希和入口钉精确闭合。全仓回归运行 812 项、用时 1,587.193 秒：803 项实际通过、9 项按既有历史记录跳过、0 失败、0 错误。回归后没有待用或已消费的 V9.1 质量回执、质量结果、临时结果、测试目录或字节码缓存。新增 `docs/AI_RESEARCH_HANDOFF_20260823.zh.md` 作为当前交接入口，旧 2026-07 交接只保留历史价值。

提交前修订复审正文提取为 15,713 个 UTF-8 字节／`69038a2e55c05026f6d3a4e536d18d478f838cebe79c21c9802526a7060144be`，问题分级阻断 0、高 1、中 0、低 0，最后一行“修订候选不能提交并进入提交后复审”。唯一高级项是可触发的一次性事务所有权错误：两个入口近同时读到待用回执时，未取得原子改名所有权的调用可能根据“已消费目标存在、待用源消失”误认自己有权发布唯一终态。该判断成立，但不需要状态机或锁。当前入口改为在原子改名后立即返回，把完整字节重验留给已有验证器，并以调用内布尔值记录本次是否真实消费；竞争失败者只能退出，不能创建结果或发布终态。

新增第 45 项定向反例，证明竞争失败者不能调用终态发布器、不能创建结果目录，而真实消费者原有的消费后异常终态测试继续通过。修复后聚焦合同 45／45，用时 43.442 秒；全仓回归运行 813 项、用时 1,583.573 秒：804 项实际通过、9 项历史跳过、0 失败、0 错误。运行后再次确认没有待用或已消费的 V9.1 质量回执、质量结果、临时结果、测试目录或字节码缓存。

所有权修复的网页定点复审正文提取为 9,059 个 UTF-8 字节／`1fd4540ec0c639b8721a207ef5c71bd3c69cd47f87b23605baf1b1a6f931622b`，问题分级阻断 0、高 0、中 0、低 0，最后一行“所有权修复可以提交并进入提交后复审”。网页端独立确认竞争失败者已确定性失去终态发布权、真实消费者仍保留消费后终态权、改名后重验没有形成科学验证空窗，并复算了合同、测试、覆盖政策、入口常量和 45 个测试方法。

当前仍为 `IMPLEMENTATION_ONLY_NO_QUALITY_RUN_NO_FORMAL_GENERATION_NO_TRAINING`。下一步只允许提交推送当前精确字节并进行提交后复审。没有提交后最后一行精确为“允许运行一次V9.1冻结质量审计”的新许可，不得创建回执或运行审计。正式中文合成训练数据、审核真值读取、M0／M1／M2／M3 和全部模型指标仍为零。

## Step28-v13 v1.13 V9.1 第一次冻结质量审计包装失败与质量失效（2026-08-23）

质量审计实现提交 `8f935adae741dce83f505edb3ef2ef9384d998ae`、树 `108fd7fe3b31bc591458b5aa18e092c048190693` 经提交后网页复审放行一次运行。网页正文为 15,906 个 UTF-8 字节／`bfcc614a147a2aca658e2bbeec02df4575c36977884452b2a17dcf8f6b35ef52`，最终行“允许运行一次V9.1冻结质量审计”。4,519 字节一次性回执文件摘要为 `29221d5e1dd4b9113efc333e1df8086285fbc70b7c0109add3ce38abe63e6d96`、规范自哈希为 `5c11e4199a527a53442683271e7085ddf36bf410d39bc9961cab34319111175c`；回执已经消费，永久不得复用。

唯一入口运行约 7,237.18 秒后发布 `AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION` 终态。终态为 5,336 字节／`f190ae560b46fe65bc0b2fb43a2c5297f9a15eb9846a87db9f497dcf9135ae53`，规范自哈希 `c215b4a69200445159a32ccf9533e9a816bd0ea955f3597a18bf72c63e048f00`，失败阶段 `quality_audit_result_validation`，重试未授权。没有返回逐行标签或预测，正式种子、正式数据、审核甲乙真值授权、训练和模型指标均未启动。

异常摘要 `b3a6c52513dd6a009621fa8635795b4de92399a941dcc49c532a0f96e851d48c` 精确对应 `text gate evidence drift`。适配器按冻结比较顺序追加失败门，外层验证器却把由 `passed=false` 重算出的失败门按名称排序后逐项比较。单个失败不会发生顺序差异，因此本轮文本家族至少触发两个冻结质量门；同一失败集合因排列不同被包装层拒绝，具体门名和数值没有持久化。

已发布机器终态属于审计包装基础设施失败，必须原样保留；但冻结适配器只要任一文本门失败就把监督回执和总审计回执置为 `DATASET_INVALIDATED`。因此本轮已经严格否决 V9.1 根的质量资格，不是等待再次确认的候选。具体失败门、数值、超限幅度和编码／槽位家族结论仍不可恢复，不得补造。

本次异常只能发生在完整审计体返回以后，因此训练／开发监督路径以及文本、编码／槽位两家族计算均已到达；终态没有保存精确访问计数。审核甲乙未获能力授权，审计体返回路径要求其访问计数为零。网页端修正审查正文为 14,086 个 UTF-8 字节／`3c31ba7be7253a206d92bc23d25596996ec2f5da0da9c73d3d4d1397b663a067`，最终行“V9.1失败归因与清理边界不能执行”；本地复核接受其核心科研判断。

清理边界改为：保留 30,711 字节根／拆分清单、机械等价承诺、设计构建历史、失败记录、小型终态、已消费回执、实施合同、冻结 V9 方法基线和 Git 失败谱系；删除 V9.1 根 72 个数据文件共 1,464,127,328 字节，以及本次失败专用覆盖政策、适配器、入口、合同测试和缓存。同一 V9.1 根不得通过包装修复重跑或恢复资格。任何 V9.2 必须使用新合同、新版本根、新尝试、新结果路径和独立资格边界，并统一失败门集合表示、保存不可丢失聚合证据、增加多失败门生产往返测试。正式数据和 M0／M1／M2／M3 继续未授权。

修正版与根／四拆分清单经网页端再次复审。回复正文为 9,282 个 UTF-8 字节／`458ed47014c62b734c78096b694d5a127b933046467fa0b47459fbf873963f69`，分级阻断 0、高 0、中 0、低 0，最终行“V9.1修正归因与清理边界可以执行”。网页端独立复算五份清单共 30,711 字节、72 个登记载荷共 1,464,127,328 字节，并确认只保留清单足以构成小型不可逆失败谱系。

获准清理已经执行。原 V9.1 根路径现只剩根清单和四份拆分清单，共 5 个文件／30,711 字节；五份文件摘要均保持原值，`observed`／`private` 目录均为 0。72 个失效数据文件和本次失败专用覆盖政策、适配器、无参数入口、合同测试均已删除。5,336 字节终态摘要仍为 `f190ae560b46fe65bc0b2fb43a2c5297f9a15eb9846a87db9f497dcf9135ae53`；4,519 字节已消费回执仍只有一份；冻结 V9 科学政策、质量运行器、验证器和实施合同均在。下一步只做删除后的回归、缓存清理、Git／大文件状态核验和提交推送，不运行任何质量审计、正式生成或模型训练。

删除后的全仓回归运行 768 项、用时 1,777.665 秒：759 项实际通过、9 项按既有历史记录跳过、0 失败、0 错误。回归后扫描未发现 `__pycache__`、`.pytest_cache`、`.playwright-mcp`、`.building` 或测试夹具残留。按真实 `design_preflight_v9_1_20260821` 路径重新统计仍恰好为 5 份清单／30,711 字节，五份文件摘要与失败记录一致；小型终态为 5,336 字节／`f190ae560b46fe65bc0b2fb43a2c5297f9a15eb9846a87db9f497dcf9135ae53`，唯一已消费回执为 4,519 字节／`29221d5e1dd4b9113efc333e1df8086285fbc70b7c0109add3ce38abe63e6d96`，冻结方法基线均存在。

质量失效关闭提交为 `51eb92c`，提交范围恰为 80 个路径：2 份更新文档、1 份新增失败记录、1 份新增小型终态、72 个失效载荷删除和 4 个失败专用实现删除。Git 大文件状态没有待推送对象；`AGENTS.md` 继续仅作本地贡献指南，不在提交范围。远端同步状态应以本地与上游分支头的直接比较为准。V9.2、正式生成和训练继续未授权。

## Step28-v13 v1.13 V9.2 科研协调合同冻结并获准实现（2026-08-23）

V9.1 质量失效关闭后重新核对上位科学实验合同、质量审计 C 修订件、V9 通道合同、V9 质量运行器和 V1.12 历史反事实结果，确认 V9 的文本硬门审计对象发生漂移：原合同允许同一控制者共享用词、标点、长度、空白和排版等作者风格，并要求近随机硬门作用于控制者盲无固定点风格错排后的反事实文本；V9 实际把保留原作者分配的完整／代码遮蔽／代码中和三表面直接送入近随机硬门。该矛盾足以要求新版本，但不能恢复 V9.1，也不能补造其未持久化的具体失败门或数值。

新合同 `docs/STEP28_V13_V1_13_V9_2_SCIENTIFIC_RECONCILIATION_CONTRACT_20260823.zh.md` 为 17,420 字节／`bb1a98044f91e3a9d915b842b231fd4822d305cc3121852a8f889130cc54edfe`。它采用最小修复：控制者盲风格错排后的完整反事实表面使用七视图、两模型，共 14 个文本硬门模型；原作者三表面共 42 个模型只作描述性作者风格与编码通道诊断；公共代码和私有槽位四模型继续作为独立硬门。每世界固定八份逐商品／画像模型输入，额外的版本化结构回执不计入模型输入；标签前必须冻结原作者 21 个和反事实 7 个文本矩阵。双重独立生产重放、F／P／U 实际消费承诺、非监督式风格结构门、唯一有序门注册表、全门计算和包装前完整聚合证据均已写入合同。

第一次上传的 13,108 字节合同经网页端审出 5 个高级、3 个中级和 1 个低级问题，已全部修正。一次同名文件重传没有随消息真正挂载，网页端正确拒绝把旧文件当成新版；该无附件复审不构成内容证据。随后使用唯一附件名 `V9_2_CONTRACT_REV2_17420_BYTES_bb1a9804.md` 重新上传，发送后的消息节点明确包含文件组。网页端实际复算 17,420 字节和目标摘要一致，最终回复按浏览器 `innerText` 提取为 7,872 个 UTF-8 字节／`e25bbe75d822dcb289d29c271e078492e66c9b90cad508b22c22d27fd9ff639d`，问题分级 0／0／0／0，最后一行“V9.2合同可进入实现”。会话为 `https://chatgpt.com/c/6a8aba2f-1618-83eb-8a3a-110cdf90d0c9`。

当前权限只到实现、测试、回归、提交和提交后精确字节复审。新随机权威仪式、1,004 世界方法资格根构建和质量审计运行必须分别取得三次独立单次放行；正式 500×4 数据、审核甲乙真值解封、M0／M1／M2／M3 训练和所有模型指标仍为 0 且未授权。

## Step28-v13 v1.13 V9.2 本地实现与邻接回归（2026-08-23）

V9.2 以合同冻结提交 `7ca11c4fd005f21017232ac9284f1c5aaaecbc8f` 为父基线形成本轮实现候选，本节与候选代码同步提交。新增 1 份机器策略、12 份版本化源码和 4 份合同测试，共 17 个实现路径；冻结 V9 源码保持不变。机器策略文件为 9,654 字节／`1f703f285ff27062ce104d8e97d6fc289cca1a781f8fe44345f7157435acdbed`，规范自哈希 `ac5e1ca0a41df953a530ab0282735c8847a22eada2a58b42238fdd065913bdaa`，钉住 16 份实现和测试源码。

实现已闭合八份模型输入、控制者盲风格错排、双重生产重放、五份 M1 公开标识错配映射承诺、跨分支不变量、21／7 文本矩阵、两份代码／槽位矩阵、两份 372 对资格掩码、42／14／4 模型角色、97 项唯一有序门注册表，以及包装器以前独占发布的完整证据。基础策略保持全部运行权限为假；未来随机权威、1,004 世界构建和质量审计分别使用三份仓库外一次性回执。全新权威必须排除基础随机树、V9／V9.1 当前权威和 28 项退役权威。训练／开发真值各限一次物理读取，随后重哈希全部无标签输入和清单；审核甲乙真值能力始终未挂载。

聚焦 V9.2 合同回归 30／30 通过，最终一轮用时 56.366 秒；相邻冻结 V9 合同回归 147／147 通过，用时 321.629 秒。开发回归实际发现并修复了旧真值能力把根模式固定为 `design_preflight` 的接口缺口；新增 V9.2 根适配器只改变根身份验证，沿用冻结的一次性真值读取与计数机制。最终全仓回归运行 798 项、用时 1,590.475 秒：789 项实际通过、9 项历史既定跳过、0 失败、0 错误。回归后清除了 2 个字节码目录和 20 个旧缓存文件，并确认测试新产生的字节码、测试缓存、临时报告和 `.building` 目录均为 0。

本实现提交后的网页精确字节审查尚未完成，因此当前不得创建随机权威或运行任何构建／质量入口。三个未来外部回执、V9.2 世界、设计根、质量指标、正式数据、审核真值读取、模型和训练结果均为 0。

## Step28-v13 v1.13 V9.2 初版提交后审查修复与全仓回归（2026-08-24）

V9.2 初版实现已以提交 `6eb0655db883a2f7478e0123fa770858e54b54b9`、树 `02cc57e340574521fb8b47824e29cef127fa5dce` 推送。网页端审查正文为 31,857 个 UTF-8 字节／`e8cbdb61119f4489250a66b86a2c26266dd2d7267bec5411510126153c3f7766`，分级阻断 2、高级 3、中级 4、低级 0，最终行“V9.2实现不可进入随机权威仪式前审查”。该否决已被接受，初版提交不能进入随机权威仪式。

当前修复补上最后模型家族后的冻结状态与输入字节再验证、临时根改名后的最终路径重放、由已消费回执签发的窄能力对象、每拆分 20 个物理文件的摘要和行框架绑定、根清单完整谱系，以及“可计算门失败继续完成全部 97 项／不可计算机械异常不产生数据集结论”的严格分界。完整失效证据在主发布或外包装失败时仍保留，并排他写入机器终态。自动删除失效根的网页建议未采用；实现只登记 `cleanup_required`，继续遵守先写失败边界再精确清理的科研纪律。风格错排生成器现被机器策略直接钉住，并有四项独立合同测试。

第一轮修复已以提交 `2a4eedf8700efea76e81cf0bea2ea12c8866988c`、树 `e8bffe5461283957eaa483294d17f58bffc476d9` 推送；精确归档为 156,615 字节／`9c04e7fbc1b877feb0116c2f6eded2826096e98a760d1408a287a14437f297a4`。网页端最终审查原文为 6,563 个 UTF-8 字节／`85db3189fdffcdc507a23caa2546bce8350d1819e8e2eede5b71f871eebd6ac3`，分级阻断 0、高级 0、中级 1、低级 0，最后一行“V9.2修复可进入随机权威仪式前审查”。唯一中级项是内部能力工厂可进一步限制为正式已消费回执路径；审查明确认定它不是正式入口可触发的科研缺陷。

本地接受该项为最小接口闭合，不扩张到锁、权限、网络或攻击面：正式质量能力现在只能从固定 V9.2 已消费授权路径签发，并新增默认路径闭合和伪造 `.consumed.json` 拒绝测试。一次未提交的中间修改曾打算让三份真值文件只信清单而不做原始字节核验；复核上位合同后确认审核私有文件必须仍做大小、SHA-256 和行数的纯字节核验，禁止的是语义解析和提前监督，因此该过度修改已撤销。每拆分 20 个物理载荷继续全部摘要和行框架绑定，审核甲乙真值不解析、不物化。

机器策略仍为 10,523 字节，现文件 SHA-256 `987fc203835057fb51d226296b801dad40accf12fa12a9bf6a05e5d3c8c61eff`，规范自哈希 `082e68c3154841cc089f3db164e9b358ab16767feb2845476c3989d417d3eebc`；13 份源码和 5 份测试共 18 个来源钉住逐字节无漂移。收紧后的聚焦回归 43／43 通过，用时 63.414 秒；相邻 V9／V9.2 合同 190／190 通过，用时 382.376 秒；最终全仓回归运行 811 项、用时 1,949.393 秒，802 项实际通过、9 项历史声明跳过、0 失败、0 错误。

随机权威、V9.2 世界、设计根、质量结果、正式数据、审核真值读取、模型和训练仍全部为零。下一步只允许清理缓存、提交推送精确字节，并让网页端针对新提交附件重新审查；没有新提交的精确许可句以前不得运行任何科研生成或训练入口。

固定正式已消费路径闭合已以提交 `c3449a3b334a95beb83652c309afd0e85ef5af73`、树 `7940d97b6e881eca93c9910290958cb8a56dd89f` 推送，本地与远端头一致。精确提交归档含 27 个非目录文件，为 158,268 字节／`aa5f57163c61e487f75146c50bac374f1cb24c62f7877f6d24e9e815386b0787`，本地逐文件重放无差异。网页端最终复审原文为 12,042 个 UTF-8 字节／`16bb09c9732ed83733da6703c80e56ed95a7e90d05524bdacd6531019f2c716d`，分级阻断 0、高级 0、中级 0、低级 0，真实科研缺陷与未闭合可选加固均为 0，最后一行“V9.2最终修复可进入随机权威仪式前审查”；会话为 `https://chatgpt.com/c/6a8baec5-d980-83eb-9aef-b6c8a9877962`。

复审确认固定路径与运行器原子消费结果精确一致，测试替换不会污染生产入口，伪造已消费路径被拒；20 载荷纯字节核验、审核甲乙零语义读取、冻结时序、42／14／4 模型角色、97 门注册表和完整失效证据均未回归。选择性归档不能独立证明 Git 对象库或重跑全仓测试，网页端如实声明该边界；本地 Git 头一致和 811 项回归补足证据。本次许可仍只进入随机权威仪式前审查，不授权随机权威、设计构建、质量审计、正式生成、真值解封或训练。下一步只允许形成纯文档闭合提交并对该提交做最终精确字节复核。

## Step28-v13 v1.13 V9.2 全新随机权威与方法资格根构建完成（2026-08-24）

纯文档闭合提交 `70397a1a53d539cba4d8b36fe7c863f451f71ffa`、树 `080a3968f290a2fbf12f999ff300cb1b49d67ca0` 成为本轮唯一运行基线。随机权威仪式运行前归档为 286,955 字节／`c060c6ef6f51dae41ab49104a29857078b457b8d7d623ee64926ccc826d23d3c`；网页回复为 10,226 个 UTF-8 字节／`4d2859bd0bc3e40f6d678f9d2f9ce278d7630302d8116c2441a4529e5319ad2e`，四级问题均为 0，最后一行“允许生成并冻结一次V9.2全新随机权威”。仪式使用操作系统密码学随机源生成 14 个互异的 32 字节值，没有打印、上传或写入 Git。权威文件为 1,805 字节／`0fc61873e58686f3cfbcb41f5f11ebe15344bba8190c4c6b55eb0da543b6ffd4`，规范自哈希 `e143cb1806923d17b61e8343559f8dc21bd5743bde7548db856b6c5bac5d3ee9`，权威束承诺 `00e59f584bcc7bc776de4e3b8855f2e78f3b0b6f94ed67aed4e1b37a49787def`。

独立构建运行前网页回复为 12,383 个 UTF-8 字节／`5fdbb0223f9fa4b422523e8d933db3d5e5198e4121b5ef18444d225856fae969`，四级问题均为 0，最后一行“允许运行一次V9.2方法资格根1004世界构建”。1,458 字节一次性构建回执摘要为 `890df2d3b6860a0a94b8efba514a6c084bbecb83d741dd7354a385e4706bc7c6`。随机权威和构建回执均在首个数据字节前消费；唯一入口正常退出，没有重试、重抽或补跑。

发布根固定为 `reports/step28_v13_v1_13_scientific_builder/design_preflight_v9_2_20260824/method_qualification_1004`，含训练／开发／审核甲／审核乙 500／500／2／2 个世界，共 28,112 个卖家、379,512 对，其中正对 20,080、负对 359,432。根含 85 个文件、1,616,648,450 字节。根清单为 4,431 字节／`4419439d9791081a11294160f83118c8d73fa8cdb34c2654bd9d02dddd82bb82`，规范自哈希 `4c14f5e936a17068d91688e9d410d11acdb2675d13b05fda435db7eff789b3dd`。

发布后独立重读全部 1.62 GB：85／85 个物理文件集合、逐文件大小、SHA-256 和行数全部与清单一致，根／拆分自哈希、根引用、数量、注册表和全局编号汇总均闭合，错误数为 0。四拆分全部使用候选 0，六类碰撞拒绝计数均为 0。待用权威和回执已不存在，两份消费文件存在且继续被 Git 忽略；没有 `.building`、缓存或浏览器临时目录。CSV／JSONL 数据文件全部命中 Git LFS。

加入真实方法资格根后的提交前全仓回归运行 811 项、用时 1,890.671 秒：802 项实际通过、9 项历史既定跳过、0 失败、0 错误。回归后未发现测试夹具、`.building`、字节码、测试缓存或浏览器临时目录，根清单摘要仍保持不变。

方法资格根及同步文档已以提交 `1b71eaca56b492d8a6dbb2cff65e01c9ef42901d`、树 `b8135277752f7468624015984dae85ad9a957c06` 推送。80／80 个 Git LFS 对象上传完成；远端分支头与本地提交一致，工作区和 LFS 状态干净。

这仍不是质量通过或训练结果。根明确为 `PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED`、`scientific_use_forbidden=true`、正式种子未创建、正式行数 0、训练未开始。下一步对提交后的精确字节取得独立质量运行前复核和一次性质量审计放行。没有该新许可前不得运行质量审计、正式 500×4 生成、审核真值监督评估或 M0／M1／M2／M3。完整记录见 `docs/STEP28_V13_V1_13_V9_2_METHOD_QUALIFICATION_BUILD_RESULT_20260824.zh.md`。

## Step28-v13 v1.13 V9.2 质量审计尝试 1 执行失败（2026-08-24）

方法资格根已在提交 `1b71eaca56b492d8a6dbb2cff65e01c9ef42901d` 发布，并以纯文档提交 `22f5c0df60a8d8208670c354fafad0f55765d8ad`、树 `b845db65534981afe0217454f24b1756c4b1ecbf` 闭合。提交后网页端 GPT-5.6 Sol Pro 审查正文为 12,735 个 UTF-8 字节／`439ca71b92be3abd78fda75b9a6ec6fd15384336d05869bd84e7493b35dab882`，338 行，四级问题 0／0／0／0，最后一行“允许运行一次V9.2方法资格根质量审计”；会话为 `https://chatgpt.com/c/6a8c05e4-7c28-83ed-8c89-39a9c8ecc6c6`。审查明确区分附件可证事实、本地 1.62 GB 重读记录和未知质量结论。

1,695 字节外部一次性质量回执的文件 SHA-256 为 `2df0585a081d3cb37b59792919bc30003d30f3cfb0aa300eb8febfbacfd91b3c`，规范自哈希 `ffe54ceb7f1cc36e321d265b008cdf1307781e72cf16ec774cdb084f8e6b6908`。生产验证器在不消费回执的条件下确认 14 个字段、Git、策略、根清单、网页摘要、私有密钥承诺和唯一输出路径全部闭合。运行前旧质量输出、`.building`、字节码缓存和待推送大文件均为 0。

唯一命令 `python -B scripts/step28_v13_v1_13_quality_audit_runner_v9_2.py` 在约 10.6 秒后正常返回机器失败终态。回执已在首个审计动作前消费且永久不得复用。终态为 699 字节／`b3b8abea330a76e781c2e1f1066b730f35e65bbb57d5b880cff21c11f8905526`，规范自哈希 `404f979309b3b39cb8c50371e4c287f8fbd2f2dd5c09555aeb079eb6b94e7b8c`，状态 `AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION`，阶段 `public_uid_and_structure_closure`，异常类型 `TypeError`，消息摘要 `d2d9ccddff1db2e3081b7154eb119b335c261a8a2f09160250140693bca47496`。

根因是正式 `_validate_public_closure()` 调用冻结 `_validate_endpoints()` 时漏传必需的 `expected_pairs_per_world`。确切 Python 异常文本的 SHA-256 与终态消息摘要一致。故障发生在任何监督真值打开之前；训练、开发真值均未读取，审核甲乙监督能力未挂载。没有生成 `complete_quality_evidence.json`、矩阵、预测、正式数据、模型或指标。811 项全绿回归没有覆盖真实根进入该生产调用点，故未拦住这个低级接口接线错误。

本次没有数据质量通过或失效结论。方法资格根仍为成功构建但禁止科研使用的只读输入，不属于失败载荷；尝试 1 的提交、网页许可、已消费回执和结果路径永久不得重跑或复用。网页快照、控制台日志等 56 个中间文件共 1,286,571 字节已删除；只保留小型终态和已消费回执。后续只能先建立新尝试合同、真实生产调用路径反例、新版本入口和新结果路径，再经全仓回归、提交后网页复审和独立单次许可；此前正式 500×4、审核真值监督读取和 M0／M1／M2／M3 继续禁止。详细记录见 `docs/STEP28_V13_V1_13_V9_2_QUALITY_AUDIT_ATTEMPT1_EXECUTION_FAILURE_20260824.zh.md`。

## Step28-v13 v1.13 V9.2 质量审计尝试 2 执行失败（2026-08-24）

尝试 2 以提交 `e69ee68efdd2086aa9f57bfb4cfcb87964bb449c`、树 `b960600d8819c0099b29da19550025e12476b543` 修复漏传每世界 378 对的问题。新增测试 5／5、相邻合同 32／32、全仓 816 项均通过；全仓为 807 项实际通过、9 项历史跳过、0 失败、0 错误，用时 1,684.723 秒。网页代码审查和同会话单次运行确认摘要分别为 `559795b9c2ce1e5846e5512d9bfc938626dd921193bc947776c150ee1d418506` 和 `6b28acb83c7cdfa8cbd89320ba9f5951c305cc1ee20c8d949748fda491317b72`。

1,704 字节回执为 `f0f5b26db8bc694c5d2bf3816ba2b436822d4c60d6e8c90988c8228c9e7ad3f5`，规范自哈希 `9e611f9345eb64fdf862d4c6c76ad97ec7231f5c867f68b78f81996bbc19692b`，已消费且不可复用。约 20 分钟的唯一运行只发布 733 字节终态：`fb03de2be889fff1f81107b3149dfa5f714872742ebc8d6ecbbdd95795754d3d`，规范自哈希 `9918a5ea2d3c05143635102529b6c341a4562cf3d80238435c6c43b5a45ef738`；状态 `AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION`，阶段 `freeze_28_text_2_code_slot_and_2_masks`，异常 `QualityProbePreparationError`。完整证据、标签、预测、正式数据和训练均未形成。

异常摘要精确对应“解码世界序号与冻结世界权威不一致”。商品代码按世界在拆分前的全局创建序号编码，而运行器错误地用拆分偏移加拆分内序号重建。用冻结标识密钥重建 0 至 1003 的世界标识后，根中 1,004 个世界集合严格闭合；当前运行器在训练 500、开发 500、审核甲 2、审核乙 2 个世界上全部猜错。这是标签无关接线错误，不是数据质量结论。

尝试 2 专用包装器、测试和网页临时文件按失败纪律删除，只保留失败文档、小型终态、已消费回执和 Git 历史；1.62 GB 成功方法资格根继续保留但禁止科研使用。后继必须从冻结世界标识重建真实全局序号，使用新版本、新回执和新结果路径。详细记录见 `docs/STEP28_V13_V1_13_V9_2_QUALITY_AUDIT_ATTEMPT2_EXECUTION_FAILURE_20260824.zh.md`。

## Step28-v13 v1.13 V9.2 质量审计尝试 3 完整失效结论（2026-08-25）

尝试 3 在提交 `26a57685ef864701575831393044c39098e2ad2c`、树 `d21ccbe7c305503abafa10ef564eeeabfde4f93a` 上固定。它保留每世界 378 对修复，并由冻结标识密钥重建 1,004 个世界标识到拆分前全局创建序号的一一映射。真实根无标签重放确认四拆分闭合，训练／开发 28,000 个卖家和 100,275 个商品代码的世界序号、卖家槽、商品槽错误数均为 0。8 项尝试 3 合同、53 项相邻合同和 819 项全仓回归均通过；全仓为 810 项实际通过、9 项历史跳过、0 失败、0 错误，用时 1,931.870 秒。

网页代码审查和固定提交后的单次运行许可摘要分别为 `6cda8c04be24603cea8739f8e9c6b7a78fc6b11e8b2669d7ffbf7654feee5667` 与 `1db5fee101d6677d9767b90d61ac27a4a734404a8739cfaf5b051c214f06a376`。1,704 字节回执为 `5f7ce2f58cb2876519cfdbb95f549d0eb7d586c7d388af9e9b8460ef950bb7f7`，规范自哈希 `70ca40d94b66a19e5a5f5508e49bcd0bff4ae71e36f689e553ef8ff1c1b917a9`，已消费且不可复用。

唯一审计约 2 小时 27 分钟后正常退出，完成 97 项登记观测并发布有效 `DATASET_INVALIDATED`。反事实文本 5 项硬门和结构 45 项硬门全部通过，原作者文本 42 项仅作描述性结果；公开代码／私有创建槽位 5 项硬门全部失败。最大单特征对称曲线下面积 0.586433 超过 0.52，上限模型族对称曲线下面积 0.653042 超过 0.53，平均精确率增量 0.033127 超过 0.01；两项百分之九十五自助抽样上界同样失败。这证明非文本构造通道仍携带标签捷径。四个探针模型分别读取公开代码或私有槽位，最终只在四者间取最大值，没有把两个视图拼接成联合模型；完整证据未公布获胜视图或列名。

完整证据为 122,519 字节／`62687a99cf41e426102c457a6b52402014c37e3e67302f1263acdf281b29301a`，规范自哈希 `c9b2e3fe562c40e810c88337236731d845c4b8677613c50354c01e90ee96ba6f`；机器终态为 1,463 字节／`e7b74cab383bf6871db24adc83a88f03b1b786653cbc944a7f021fde2ab4328e`，规范自哈希 `61bc6602cab61b489ec35d7b685babbdaf5687b2f37272dc5214d7060fecd01b`。审核甲乙真值打开次数、逐行标签返回、逐行预测返回、正式数据和训练均为 0。

V9.2 根永久失去训练资格，不得重跑、改门或恢复。清理前根为 85 个文件／1,616,648,450 字节；现已删除 80 个载荷／1,616,618,475 字节和 20 个 V9.2 专属策略、脚本、测试文件，仅保留五份清单／29,975 字节、完整证据、终态、已消费回执、文档和 Git 历史。载荷、缓存、临时目录及 `scripts`／`tests`／`schema` 中的 `v9_2` 活文件均为 0。删除后全仓回归运行 768 项、用时 1,696.850 秒：759 项实际通过、9 项历史跳过、0 失败、0 错误。

只读源码追踪确认两条必须共同关闭的风险：精确标题克隆允许把人工商品码带到登记负例，且控制者分组／创建槽／商品数量只做概率独立而未对固定样本确定性配平；它们是后继设计必须修复的通道，但不能冒充本轮未公布的获胜列。后继固定为全新 V9.3，先移除正式模型表面的人工代码，再以 500 世界均衡分组表和均衡噪声层关闭槽位、数量和缺失模式捷径；大规模构建前先执行同族结构／统计预检。正式 500×4 和 M0／M1／M2／M3 继续禁止。详细记录见 `docs/STEP28_V13_V1_13_V9_2_QUALITY_AUDIT_ATTEMPT3_DATASET_INVALIDATION_20260825.zh.md`。

## Step28-v13 v1.13 V9.3 原始端点残差求解运行前闭合（2026-08-26）

V9.3 当前仍是无文本、无真值、无模型的抽象设计预检。训练／开发各 500 世界的独立配平计划和联合噪声签名已闭合；登记负例联合计划的前七次开发轨迹分别暴露退火局部最优、候选池压缩和候选整数求解超时，但均未证明 5,324 项合同不可满足。第六次已验证粗修复可把违例从 45 降至 20；第七次从 45 直接求解 11,110 个候选只得到 `UNKNOWN`。失败路径、摘要和不可复用边界已写入 V9.3 合同，失败输出和临时日志均已删除。

第八次实现撤销了未经运行的“两步候选合并”猜测。粗修复达到一范数违例不超过 20 后，求解器改为直接决定固定世界集合中六对／十二个端点的完整合法取值，而不是选择预生成移动；同一模型同时执行十二端点互异、异控制者、文本角色合格性及全部 5,324 项卖家／噪声计数上下界。固定嵌套世界批次为 24、48、96、192、320、500，只有 500 世界原始变量模型明确 `INFEASIBLE` 才能证明合同无解；`UNKNOWN` 或超时没有科研结论。全新输出路径为 `registered_negative_preflight_v10_20260826`，尚不存在。

聚焦 V9.3 合同回归运行 28 项：23 项实际通过、5 项因新计划尚未发布而明确跳过；原始端点构造器 13 项全部通过。全仓回归运行 796 项、用时 1,892.762 秒：782 项实际通过、14 项有明确历史或计划未发布原因的跳过、0 失败、0 错误。回归后测试临时目录、字节码缓存、`.building` 和 V10 输出均为 0。下一步只允许运行一次 V10 抽象登记负例预检；训练／开发必须各自严格零违例并通过独立验证后，才能进入无文本结构资格验证。正式中文数据、审核真值和 M0／M1／M2／M3 仍全部禁止。

## Step28-v13 v1.13 V9.3 V10 独热原始端点模型终止（2026-08-26）

V10 唯一运行在训练拆分重放 `45→38→37→36→28→24→20` 后，依次对 191、192、320、500 个世界求解完整原始端点约束，四次均为 `UNKNOWN`。500 世界模型含 2,148,000 个布尔选项和 5,324 个计数单元，求解墙钟 1,807.686382 秒，只有 2,495 个分支、0 个冲突；残差摘要 `19dbedf0d57277c88f16d4c94f0a0efe1b337dc898461239edf2a7ab7fa01d79`，世界选择顺序摘要 `662b0a49f6beecf20f25760896649366d2d6eb6cb403922324b071a548546d1d`。

本轮没有不可满足证明，也没有零违例计划；科研根因只定位为独热展开模型过大且传播不足。开发拆分未启动，V10 输出目录和 `.building` 不存在，未生成计划、回执、文本、真值、正式数据或模型。V10 路径与同一独热表示永久不复用；后继保持合同和阈值不变，改用等价的紧凑分解求解。

V11 在运行前改为确定性紧凑局部残差求解：每个超额／不足单元只选择足够数量的贡献或支持世界，固定批次为 24、48、72、96、128、192；每批按外部冻结贡献计算残差界，并在建变量前删除非负贡献已单独超过残差上界的必不可能端点选项。该剪枝不改变 5,324 项约束或零违例资格门。聚焦构造器 14 项通过，完整 V9.3 合同 29 项中 24 项通过、5 项因 V11 计划尚未发布跳过；新路径为 `registered_negative_preflight_v11_20260826`。
