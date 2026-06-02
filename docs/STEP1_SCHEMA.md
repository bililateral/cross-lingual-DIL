# Step 1: Canonical Schema and Data Audit

This repository does not yet contain a reproducible data-processing baseline. Step 1 fixes that by defining one canonical schema and one audit script before any silver-label mining or model training begins.

## Scope

- `EN-Gold`: `tijkc3xx.sql`, `3z669jwe.sql`, `suspected_sockpuppet_strong.csv`, `suspected_sockpuppet_weak.csv`, `suspected_imposter_rows.csv`
- `EN-Content`: `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`, English rows from `market_item.xlsx`
- `ZH-Target Strict`: target-market rows from `market_item.xlsx`
- `ZH-Target Aux`: `products_data.csv`

## Canonical Tables

### `item_records`

Use this table to build seller profiles.

- Primary key: `item_uid`
- Seller key: `seller_uid`
- Required provenance fields: `source_dataset`, `source_row_number`, `source_market_raw`
- Required seller fields: `source_seller_raw` or `source_seller_id_raw`
- Required text fields for content modeling: `title_raw`, `description_raw`, `category_raw`

ID policy:

- If a raw item id exists, keep it in `source_item_id_raw`.
- If no raw item id exists, build `item_uid` from a deterministic hash of the raw row tuple plus `source_row_number`.
- `seller_uid` must always be source-scoped. Example shape:
  `market_item.xlsx|茶马古道|seller_raw:/shop/419972`
  `products_data.csv|__unknown__|seller_id:553494`
  `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx|agora|seller_raw:CheapPayTV`

### `vendor_registry`

Use this table only for strict English benchmarking and leakage control.

- Primary key: `vendor_row_uid`
- Raw identity: `market_id_raw`, `vendor_id_raw`, `user_name_raw`
- Market normalization anchor: `source_market_host`
- Benchmark-only flag: `imposter_flag`

### `identity_evidence`

Use this table to store linkage evidence instead of mixing it into content rows.

- Primary key: `evidence_uid`
- Core fields: `evidence_type`, `confidence_level`, `evidence_key_raw`
- Raw identifiers: `market_id_raw`, `vendor_id_raw`, `user_name_raw`
- PGP fields: `fingerprint_raw`, `fingerprint_short_raw`
- Group fields: `group_size_raw`, `review_count_raw`, `star_rate_raw`

## Source-Specific Rules

- `products_data.csv`
  `market_scope` is unresolved in the raw file. Keep it as `__unknown__` until provenance is independently verified.
- `market_item.xlsx`
  This file is mixed-domain. Only rows with `market in {"中文暗网交易市场", "茶马古道"}` belong to `ZH-Target Strict`.
- `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`
  Treat all rows as `source_market_raw = agora`.
- `tijkc3xx.sql`
  Do not use as EN content training input.
- `3z669jwe.sql`
  Treat PGP fingerprint linkage as the strongest available English identity evidence.

## Hard Constraints

- Never merge sellers across files only because aliases match.
- Always run leakage auditing before any English content training split.
- Always keep raw text untouched in canonical storage.
- Always separate benchmark evidence tables from content tables.

## Immediate Follow-Up After Step 1

1. Build leakage exclusion lists from `vendor_registry` plus `identity_evidence`.
2. Construct seller-profile aggregation from `item_records`.
3. Mine silver-label candidates on top of the canonical tables, not raw files.
