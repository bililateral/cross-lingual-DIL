# Project Progress

Updated: 2026-07-20

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
