# Evidence scope

All local evaluation inputs are handmade fixtures. No BGE weights, CUDA computation, formal labels or real trained checkpoint are included.

Reproduce from the supplied original and supplemental ZIPs: extract the supplement to /mnt/data/chinese_supplement_review; retain /mnt/data/chinese_review.zip. Run check_receipts.py and targeted_checks.py with the extracted supplement directory as the first argument. Existing NumPy and PyTorch are needed for the submitted contracts; no extra dependencies were installed in this review.

- submitted_contracts.log: final submitted 12 contracts, 12 passed, 0 skipped.
- targeted_checks.py / targeted_results.json: independent reporting checks; the completed-run gate is mocked only in these independent tests.
- first_interval_collection_handmade.json: handmade collection snapshot (its referenced temporary matrices are not distributed; the test regenerates/verifies them). Not a research result.
- receipt_source_results.json: artifact hashes, source/AST/native-report/remote-model-receipt correspondence, not a local rerun of BGE.
- original_to_final.diff / native_to_final.diff: exact runner differences.

The original 53 payloads were not modified. No production patch is proposed beyond correcting the narrative 65-character hash typo.
