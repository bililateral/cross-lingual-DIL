# Step 2: Strict Split and Leakage Isolation

Step 2 converts the Step-1 schema into research-safe data pools. The goal is not model training yet. The goal is to prevent benchmark leakage and to freeze the exact content pools used later for seller profiling, silver-label mining, and model training.

## Research-Aligned Pools

- `EN-Gold benchmark`
  Official strict benchmark pool:
  - `tijkc3xx.sql`
  - `suspected_sockpuppet_strong.csv`
  - `suspected_sockpuppet_weak.csv`
  - `suspected_imposter_rows.csv`
- `EN-Content train pool`
  - `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`
  - non-target rows from `market_item.xlsx`
- `ZH-Target strict`
  - rows from `market_item.xlsx` where `market in {"中文暗网交易市场", "茶马古道"}`
- `ZH-Target aux`
  - `products_data.csv`
- `Aux identity evidence`
  - `3z669jwe.sql`
  - `html-rips`

## Split Rules

### Seller-level exclusion

For English content sources only:

- Normalize seller alias with `casefold + strip`
- Build one official English benchmark alias set from:
  - `tijkc3xx.sql` `user_name`
  - `suspected_sockpuppet_strong.csv` `user_name` and `key_alias`
  - `suspected_sockpuppet_weak.csv` `user_name` and `key_alias`
  - `suspected_imposter_rows.csv` `user_name`
- Expand benchmark-linked identity closure with `3z669jwe.sql` using:
  - benchmark aliases
  - benchmark fingerprints
  - benchmark-linked vendor ids from labeled evidence rows
- Exclude any English content seller whose normalized alias appears in that official alias ledger
- Exclude any English content seller whose alias maps via auxiliary PGP evidence to a benchmark-linked fingerprint

This is intentionally strict. It is better to lose some English content than to contaminate the benchmark with actors that reuse core identity assets.

### Row-level exclusion

For English content sources only:

- Exclude any row whose text contains a high-precision benchmark-linked contact identifier mined from local Grams header snapshots
- Exclude any row whose `title`, `description`, or `category` contains CJK characters

The contact rule is conservative by design:

- emails can come from vendor-page header regions, including PGP UID lines
- non-email handles ignore PGP public-key blocks to avoid base64 false matches
- only benchmark-linked snapshot contacts are used

Current-boundary note:

- in the current synchronized split summary, this contact rule contributes a very small high-precision safeguard rather than the main leakage shield
- the present run reports only `3` benchmark-linked contact values and `0` English content exclusions from the contact-overlap rule
- the main effective leakage isolation still comes from alias-ledger and auxiliary-fingerprint exclusion, not from contact overlap alone

The CJK rule prevents Chinese or translated clone rows from entering the English training pool.

## Outputs

The Step-2 script generates:

- English benchmark manifest
- Auxiliary PGP-evidence manifest
- Official English alias exclusion list
- Benchmark-linked contact exclusion list
- Content item manifest
- Content seller manifest
- Split summary with before/after overlap checks

## Acceptance Criteria

Step 2 is only valid if all of the following are true:

1. `EN-Content train pool` and `EN-Gold benchmark` have normalized alias overlap equal to `0`
2. `EN-Content train pool` has auxiliary-PGP fingerprint overlap equal to `0` against the benchmark-linked identity closure
3. `ZH-Target strict` only contains the two approved Chinese markets
4. `products_data.csv` is kept out of strict target benchmarking
5. Every excluded English content row or seller has an explicit exclusion reason
