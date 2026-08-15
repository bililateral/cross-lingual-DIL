from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_truth_capability_v9 as truth


class QualityTruthCapabilityV9Contracts(unittest.TestCase):
    def build_pinned_root(
        self, root: Path
    ) -> tuple[truth.RootManifestPin, dict[str, object]]:
        split_hashes: dict[str, str] = {}
        raw_csv = "canonical_pair_uid,world_uid,label\n".encode("utf-8")
        for split in truth.SPLITS:
            split_root = root / split
            private_root = split_root / "private"
            private_root.mkdir(parents=True)
            label_path = private_root / "pair_labels.csv"
            label_path.write_bytes(raw_csv)
            world_count = truth.EXPECTED_WORLD_COUNTS[split]
            manifest: dict[str, object] = {
                "split": split,
                "world_count": world_count,
                "pair_count": world_count * 378,
                "positive_pair_count": world_count * 20,
                "files": [
                    {
                        "path": truth.TRUTH_RELATIVE_PATH,
                        "size_bytes": len(raw_csv),
                        "sha256": hashlib.sha256(raw_csv).hexdigest(),
                        "row_count": world_count * 378,
                    }
                ],
            }
            manifest["canonical_self_hash"] = truth._canonical_self_hash(manifest)
            (split_root / "split_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            split_hashes[split] = str(manifest["canonical_self_hash"])
        root_manifest: dict[str, object] = {
            "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
            "execution_mode": "design_preflight",
            "split_order": list(truth.SPLITS),
            "world_count": 1004,
            "scientific_use_forbidden": True,
            "formal_seed_created": False,
            "formal_rows_created": 0,
            "training_started": False,
            "split_manifest_self_hashes": split_hashes,
        }
        root_manifest["canonical_self_hash"] = truth._canonical_self_hash(root_manifest)
        root_path = root / "root_manifest.json"
        root_path.write_text(
            json.dumps(root_manifest, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        raw_root = root_path.read_bytes()
        pin = truth.RootManifestPin(
            path="root_manifest.json",
            size_bytes=len(raw_root),
            sha256=hashlib.sha256(raw_root).hexdigest(),
            canonical_self_hash=str(root_manifest["canonical_self_hash"]),
        )
        return pin, root_manifest

    def test_pinned_csv_opens_once_and_reports_real_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pair_labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=truth.TRUTH_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "canonical_pair_uid": "pair_a",
                        "world_uid": "world_a",
                        "label": 1,
                    }
                )
                writer.writerow(
                    {
                        "canonical_pair_uid": "pair_b",
                        "world_uid": "world_a",
                        "label": 0,
                    }
                )
            raw = path.read_bytes()
            pin = truth.TruthFilePin(
                split="train",
                path=path,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                row_count=2,
                split_manifest_self_hash="a" * 64,
            )
            rows, receipt = truth._read_pinned_truth_csv(pin)
            self.assertEqual(tuple(rows[0]), truth.TRUTH_FIELDS)
            self.assertEqual([row["label"] for row in rows], [1, 0])
            self.assertEqual(receipt["file_open_count"], 1)
            self.assertEqual(receipt["byte_read_count"], len(raw))
            self.assertEqual(receipt["materialized_row_count"], 2)

    def test_tampered_bytes_and_extra_columns_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pair_labels.csv"
            path.write_text(
                "canonical_pair_uid,world_uid,label,controller_uid\n"
                "pair,world,1,forbidden\n",
                encoding="utf-8",
            )
            raw = path.read_bytes()
            pin = truth.TruthFilePin(
                split="development",
                path=path,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                row_count=1,
                split_manifest_self_hash="b" * 64,
            )
            with self.assertRaisesRegex(
                truth.QualityTruthCapabilityError, "header drift"
            ):
                truth._read_pinned_truth_csv(pin)
            path.write_bytes(raw + b"x")
            with self.assertRaisesRegex(
                truth.QualityTruthCapabilityError, "bytes drift"
            ):
                truth._read_pinned_truth_csv(pin)

    def test_truth_byte_drift_is_dataset_failure_and_io_is_auditor_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pair_labels.csv"
            path.write_text(
                "canonical_pair_uid,world_uid,label\npair_a,world_a,1\n",
                encoding="utf-8",
            )
            raw = path.read_bytes()
            pin = truth.TruthFilePin(
                split="train",
                path=path,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                row_count=1,
                split_manifest_self_hash="a" * 64,
            )
            replacement = raw.replace(b"pair_a", b"pair_b")
            self.assertEqual(len(replacement), len(raw))
            path.write_bytes(replacement)
            with self.assertRaises(truth.QualityTruthDatasetGateError):
                truth._read_pinned_truth_csv(pin)
            path.write_bytes(raw)
            with patch.object(Path, "open", side_effect=OSError("simulated")):
                with self.assertRaises(truth.QualityTruthAuditorExecutionError):
                    truth._read_pinned_truth_csv(pin)

    def test_manifest_content_drift_is_dataset_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            pin, _manifest = self.build_pinned_root(root)
            split_path = root / "development" / "split_manifest.json"
            split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
            split_manifest["pair_count"] += 1
            split_manifest["canonical_self_hash"] = truth._canonical_self_hash(
                split_manifest
            )
            split_path.write_text(
                json.dumps(split_manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaises(truth.QualityTruthDatasetGateError):
                truth.FormalTrainDevelopmentTruthCapability.from_pinned_design_root(
                    dataset_root=root,
                    root_manifest_pin=pin,
                )

    def test_manifest_json_failure_classes_data_and_io_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(truth.QualityTruthDatasetGateError):
                truth._load_json(path)
            with patch.object(Path, "read_text", side_effect=OSError("simulated")):
                with self.assertRaises(truth.QualityTruthAuditorExecutionError):
                    truth._load_json(path)

    def test_audit_truth_is_rejected_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pair_labels.csv"
            path.write_text("canonical_pair_uid,world_uid,label\n", encoding="utf-8")
            pin = truth.TruthFilePin(
                split="audit_a",
                path=path,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                row_count=0,
                split_manifest_self_hash="c" * 64,
            )
            with self.assertRaisesRegex(
                truth.QualityTruthCapabilityError, "sealed"
            ):
                truth._read_pinned_truth_csv(pin)
            with self.assertRaises(truth.QualityTruthCapabilityError):
                truth.reject_audit_truth("audit_b")

    def test_formal_capability_cannot_be_constructed_directly(self) -> None:
        with self.assertRaisesRegex(
            truth.QualityTruthCapabilityError, "from_pinned_design_root"
        ):
            truth.FormalTrainDevelopmentTruthCapability()

    def test_root_bound_capability_rejects_alternate_root_and_consumes_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            pin, _manifest = self.build_pinned_root(root)
            capability = (
                truth.FormalTrainDevelopmentTruthCapability.from_pinned_design_root(
                    dataset_root=root,
                    root_manifest_pin=pin,
                )
            )
            expected_binding = {
                "path": (root / "root_manifest.json").as_posix(),
                "size_bytes": pin.size_bytes,
                "sha256": pin.sha256,
                "canonical_self_hash": pin.canonical_self_hash,
            }
            self.assertEqual(capability.root_binding(), expected_binding)
            with self.assertRaisesRegex(
                truth.QualityTruthCapabilityError, "root binding drift"
            ):
                capability._begin_bound_transaction(
                    expected_root_binding={**expected_binding, "sha256": "0" * 64}
                )

            pins = capability._begin_bound_transaction(
                expected_root_binding=expected_binding
            )
            self.assertEqual(set(pins), {"train", "development"})
            self.assertTrue(
                all(isinstance(value, truth.TruthFilePin) for value in pins.values())
            )
            self.assertFalse(
                hasattr(capability, "_consume_bound_train_development")
            )
            for split, value in pins.items():
                capability._record_split_receipt(
                    split=split,
                    receipt={
                        "split": split,
                        "file_open_count": 1,
                        "byte_read_count": value.size_bytes,
                        "materialized_row_count": value.row_count,
                        "sha256": value.sha256,
                        "split_manifest_self_hash": value.split_manifest_self_hash,
                    },
                )
            receipt = capability.aggregate_receipt()
            self.assertEqual(receipt["train"]["file_open_count"], 1)
            self.assertEqual(receipt["development"]["file_open_count"], 1)
            self.assertEqual(receipt["audit_a"]["file_open_count"], 0)
            with self.assertRaisesRegex(
                truth.QualityTruthCapabilityError, "already consumed"
            ):
                capability._begin_bound_transaction(
                    expected_root_binding=expected_binding
                )

    def test_canonical_self_hash_excludes_only_self_field(self) -> None:
        value = {"x": 1}
        value["canonical_self_hash"] = truth._canonical_self_hash(value)
        self.assertEqual(value["canonical_self_hash"], truth._canonical_self_hash(value))
        changed = dict(value)
        changed["x"] = 2
        self.assertNotEqual(value["canonical_self_hash"], truth._canonical_self_hash(changed))


if __name__ == "__main__":
    unittest.main()
