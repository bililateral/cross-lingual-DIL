# Step 3: Seller-Level Profile Construction

Step 3 converts item-level rows into seller-level profiles. This is the first point where the research pipeline switches from raw listings to account portraits.

## Why This Step Exists

The downstream tasks all operate on sellers, not raw listings:

- silver-label candidate mining
- pairwise seller comparison
- reranker inputs
- LightGBM feature fusion
- graph construction for clustering

Therefore Step 3 must happen before silver labels and before model training.

## Inputs

- `reports/step2_content_item_manifest.csv`
- `reports/step2_content_seller_manifest.csv`
- raw source files:
  - `products_data.csv`
  - `market_item.xlsx`
  - `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`

## Eligible Buckets

Only these Step-2 buckets are aggregated:

- `en_content_train_pool`
- `zh_target_strict`
- `zh_target_aux`

Excluded English rows are never aggregated back into seller profiles.

## What Gets Aggregated

Each seller profile includes four signal families.

### Semantic signal

- signature titles
- signature description segments
- top titles
- top description snippets
- top categories
- profile text assembled from overview fields plus long-tail signature fields

### Structural signal

- item count
- category concentration
- top raw price strings
- approximate numeric price stats
- top origin values when present
- source-specific structured snapshots

### Style signal

- title length stats
- description length stats
- digit-ratio mean
- punctuation-ratio mean
- uppercase-ratio mean
- newline-count mean
- repeated-title share
- repeated-description share

Important boundary:

- raw structural and style fields are preserved at Step 3 for auditability and within-pool candidate mining
- they are not automatically transfer-safe features for Step 7/Step 8
- any cross-lingual training stage must convert raw magnitudes into market-relative or language-robust pair features before sending them to the final scorer

### Identifier signal

- email
- telegram
- wickr
- wechat
- qq
- phone numbers with contact-context cues

Non-email handle extraction ignores embedded PGP public-key blocks to avoid false identifiers from base64 content.

### Item-Level Identity Evidence

Step 3 now also writes separate item-level identity evidence files. These are not labels and are not ground truth; they preserve parser hits and surrounding context so Step 5 can mine direct-contact review candidates without relying on Step 11 predictions.

Item-level extraction covers:

- Telegram/TG/电报/纸飞机/飞机 and `t.me` style handles
- Wickr
- Wechat/Weixin/VX/WX/微信/威信
- Jabber/XMPP
- QQ/企鹅/扣扣
- phone numbers with contact cues
- crypto wallets for BTC/ETH/TRON-style addresses
- PGP public-key blocks and PGP fingerprint text
- Bat/蝙蝠 handles
- emails
- support-only external URLs

Each signal carries:

- source dataset and row number
- seller UID and market/seller raw identifiers
- source field (`title`, `description`, `structured_snapshot`, or remarks)
- normalized token
- raw token
- evidence level
- context snippet
- `seller_facing_context`
- `product_data_risk_context`
- `direct_identity_eligible`
- `support_only`

## Compression Rules

- Titles: top 20 unique overview titles
- Descriptions: top 12 unique overview snippets, each capped at 280 characters
- Signature titles: top 8 seller-specific titles ranked by corpus-level specificity
- Signature descriptions: top 10 long-tail segments extracted from full descriptions and ranked by corpus-level specificity
- Categories: top 15
- Contacts: top 10 per type
- Overview ordering: frequency descending, then first-seen ascending
- Signature ordering: specificity score descending, then support count, then first-seen

## Outputs

- one canonical JSONL per eligible bucket
- summary JSON with acceptance checks
- item-level identity signal CSVs:
  - `reports/step3_item_identity_signals.en_content_train_pool.csv`
  - `reports/step3_item_identity_signals.zh_target_strict.csv`
  - `reports/step3_item_identity_signals.zh_target_aux.csv`

## 2026-04-22 Rerun

The item-level parser upgrade was rerun locally on `2026-04-22`.

Acceptance checks:

- seller counts match Step 2 for all eligible buckets
- item counts match Step 2 for all eligible buckets
- all profile text is non-empty

Chinese strict extraction:

- item-level identity signals: `3,785`
- direct-identity-eligible signals: `1,477`
- shared seller-facing direct token groups: `39`
- Step 5 queue candidates after excluding already frozen/reviewed pairs: `0`

Interpretation:

- the parser upgrade is implemented and reproducible
- current Chinese strict raw item text does not expose additional unreviewed shared seller-facing direct identifiers
- item-level evidence should still be retained for audit and future raw/OCR expansion, but it does not trigger a Step 5 freeze rerun by itself

## 2026-04-23 High-Precision Chinese Contact Patch

Codex reread the Chinese strict raw item text and patched `scripts/step3_build_seller_profiles.py` for high-precision missed contact forms:

- Unicode NFKC matching for identity extraction, so full-width `ＴＧ`, full-width punctuation, and full-width separators are normalized for parser matching
- compact Telegram forms such as `TG-handle`, `TG@handle`, `电报handle`, and `联糸TG:@handle`
- compact QQ forms such as `QQ2691518404`
- numeric Bat/蝙蝠 IDs such as `蝙蝠3034964`
- conservative `v:x973688` style WeChat cues
- expanded token blacklist for generic words such as `phone`, `email`, `tg`, `qq`, `vx`, and `video`

The patch intentionally does not broaden phone extraction. Rows that merely mention `手机号`, `联系方式`, or leaked customer fields remain product/victim-data risk unless a seller-facing direct token is parsed.

Step 3 was rerun locally on `2026-04-23`.

Acceptance checks:

- seller counts still match Step 2 for all eligible buckets
- item counts still match Step 2 for all eligible buckets
- all profile text is non-empty

Chinese strict extraction after the patch:

- item-level identity signals: `4,430`
- direct-identity-eligible signals: `1,890`
- sellers with any signal: `1,259`
- sellers with direct-eligible signal: `662`
- shared seller-facing direct token groups: `44`
- Step 5 queue candidates after excluding already frozen/reviewed pairs: `0`
- skipped shared-token pairs: `50 frozen_pair`

Interpretation:

- the parser now recovers previously missed seller-facing TG/QQ/Bat/WeChat spellings
- the newly recovered shared direct-token pairs are already represented in the frozen/reviewed Step 5 boundary
- the patch improves audit evidence coverage, but it does not by itself trigger a Step 5 freeze or downstream Step 7/9/11 rerun

## Acceptance Criteria

1. Seller counts by bucket match Step 2
2. Aggregated item counts by bucket match Step 2
3. Every seller profile has deterministic `profile_text`
