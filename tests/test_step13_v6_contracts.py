from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step13_concept_drift_audit as step13  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def self_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class Step13V6ContractsTests(unittest.TestCase):
    def test_verified_clean_publication_manifest_audit_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_path = root / "runtime.json"
            output_path = root / "scored.csv"
            summary_path = root / "summary.json"
            manifest_path = root / "manifest.json"
            manifest_csv_path = root / "manifest.csv"
            audit_path = root / "audit.json"
            audit_csv_path = root / "audit.csv"
            runtime_path.write_text("{}\n", encoding="utf-8")
            output_path.write_text("pair_uid\n", encoding="utf-8")
            summary = {
                "selected_scorer": {"scorer_token": "step15_v6_final_selected_seed_mean"},
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            record = {
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "runtime_policy_path": str(runtime_path),
                "runtime_policy_sha256": sha256(runtime_path),
                "scorer_token": "step15_v6_final_selected_seed_mean",
                "graph_validation_mode": "clean_topology",
                "output_file_records_json": json.dumps(
                    [{"path": str(output_path), "sha256": sha256(output_path)}]
                ),
            }
            manifest = {
                "manifest_version": "step11_explicit_summary_manifest_v2_hash_closed",
                "run_id": "unit",
                "selection_mode": "explicit_allowlist_only",
                "publication_v6": True,
                "summary_count": 1,
                "graph_validation_mode": "clean_topology",
                "expected_scorer_tokens": ["step15_v6_final_selected_seed_mean"],
                "summaries": [record],
                "manifest_csv_path": str(manifest_csv_path),
                "manifest_csv_row_count": 1,
            }
            manifest_csv_path.write_text("scorer_token\nstep15_v6_final_selected_seed_mean\n", encoding="utf-8")
            manifest["manifest_csv_sha256"] = sha256(manifest_csv_path)
            manifest["manifest_sha256"] = self_hash(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit_csv_path.write_text(
                "scorer_token,decision\n"
                "step15_v6_final_selected_seed_mean,partial_anchor\n",
                encoding="utf-8",
            )
            audit = {
                "summary_selection_mode": "explicit_manifest",
                "publication_v6": True,
                "input_manifest": str(manifest_path),
                "input_manifest_sha256": manifest["manifest_sha256"],
                "input_manifest_file_sha256": sha256(manifest_path),
                "graph_validation_mode": "clean_topology",
                "input_summaries": [str(summary_path)],
                "audited_per_scorer_cluster_count": 1,
                "decision_counts": {"partial_anchor": 1},
                "per_scorer_cluster_counts": {
                    "step15_v6_final_selected_seed_mean": 1,
                },
                "output_csv": str(audit_csv_path),
                "output_csv_sha256": sha256(audit_csv_path),
                "output_csv_row_count": 1,
            }
            audit["audit_sha256"] = self_hash(audit)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            loaded_manifest, loaded_audit, diagnostics = (
                step13.verify_step11_manifest_audit_chain(
                    manifest_path,
                    audit_path,
                    require_publication_v6=True,
                    require_clean=True,
                )
            )
            self.assertEqual(loaded_manifest["manifest_sha256"], manifest["manifest_sha256"])
            self.assertEqual(loaded_audit["graph_validation_mode"], "clean_topology")
            self.assertTrue(diagnostics["verified"])
            self.assertEqual(diagnostics["summary_count"], 1)
            self.assertEqual(diagnostics["audit_csv_row_count"], 1)

            audit["graph_validation_mode"] = "identifier_assisted_operational"
            audit["audit_sha256"] = self_hash(
                {key: value for key, value in audit.items() if key != "audit_sha256"}
            )
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaises(ValueError):
                step13.verify_step11_manifest_audit_chain(
                    manifest_path,
                    audit_path,
                    require_publication_v6=True,
                    require_clean=True,
                )

            audit["graph_validation_mode"] = "clean_topology"
            audit["audit_sha256"] = self_hash(
                {key: value for key, value in audit.items() if key != "audit_sha256"}
            )
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            audit_csv_path.write_text(
                "scorer_token,decision\n"
                "step15_v6_final_selected_seed_mean,uncertain\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                step13.verify_step11_manifest_audit_chain(
                    manifest_path,
                    audit_path,
                    require_publication_v6=True,
                    require_clean=True,
                )


if __name__ == "__main__":
    unittest.main()
