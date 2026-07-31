# Workspace Deprecated Artifact Cleanup — 2026-07-31

## Scope and Rule

This cleanup used an explicit path allow-list after the Step28-v13 v1.2 final
release and parent/child equivalence proof completed. No wildcard, date-wide,
step-wide, tracked-file, source-input, private-key, or current-result deletion
was used.

## Deleted Artifacts

The following untracked directories were removed:

- `reports/step28_synthetic_chinese_dataset/v13_training_ready_v1_20260729/`
  — 159 files, about 4.82 GiB; invalid because C40 file order predicted labels.
- `reports/step28_synthetic_chinese_dataset/v13_training_ready_v1_1_order_repair_20260731/`
  — 158 files, about 4.82 GiB; never finalized and bound to the faulty Audit
  path comparator.
- `reports/step28_synthetic_chinese_dataset/design_preflights/order_repair_v1_20260731/`
  — 26 files; checkpoint placement did not satisfy the registry contract.
- `reports/step28_synthetic_chinese_dataset/design_preflights/order_repair_v2_20260731/`
  — 26 files; bound to the superseded v1.1 implementation contract.

The deleted dataset bytes were no longer needed after the v1.2
`repair_equivalence_report.json` proved order-only semantic equivalence.
This deliberately makes the full parent/child semantic comparison
non-rerunnable from the current workspace alone: the hash-frozen report and
nine parent manifests remain verifiable, but rerunning the comparison requires
restoring every original v1 member byte.

## Preserved Historical Evidence

Before deletion, nine manifests were copied byte-for-byte to
`reports/step28_synthetic_chinese_dataset/deprecated_release_evidence/` and
verified by SHA-256:

| Evidence | SHA-256 |
|---|---|
| v1 release | `6924dadb669bf056302418ac012e2f027b1bd3e9e00cf0c0e5e515258a3d3ce0` |
| v1 train | `b052b95ee477af983d3b558cbc1e4c0f231e6fd7ad623768bb28fc7f12926d73` |
| v1 development | `fe57386b174623061faa71712034530cb2f950c1bf05ae971fa0c552c4f32721` |
| v1 audit_a | `8a457eab21fbd29eba6888fbf12c27716387f2e5edb5060301b745763cbe8f9e` |
| v1 audit_b | `5723f7c436c31e0872298a460e74179b636817d9cdcc320fba07fdc93bf1cfce` |
| v1.1 train | `537cdeb79b9730938133bbdb077ba480f1d387d9c68c8de97eba84046123c260` |
| v1.1 development | `ab2c7182c00f6ca011a6b7fcc172b39329925602c7917df20ea81bc188331ed5` |
| v1.1 audit_a | `a9e19ac90bfc62b70110bf24372643a699de224b29e25cfc18321b4926198cc5` |
| v1.1 audit_b | `dd24b2ceeb37a53eb0be74b91358d29d5c8c257e5c471105d7916344fff79b13` |

Also preserved: the v1 failure audits, current v3 exact preflights and
checkpoints, formal v1.2 bytes, release/equivalence manifests, private custody
files, release inputs, tracked development smokes, and all research documents.
The post-release audit directory now contains the two v1 discovery reports,
the v1.2 row audit, and a separate independent full-tree/order hardening audit.
Root marketplace exports, dated source dumps, models, and other ignored
immutable inputs were outside cleanup scope.
