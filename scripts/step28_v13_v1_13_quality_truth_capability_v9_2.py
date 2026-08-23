#!/usr/bin/env python3
"""V9.2 root-bound adapter for the frozen V9 train/development truth reader."""

from __future__ import annotations

from pathlib import Path

import step28_v13_v1_13_quality_truth_capability_v9 as v9


VERSION = "2026-08-23-step28-v13-v1-13-quality-truth-capability-v9-2"
ROOT = Path(__file__).resolve().parents[1]
BUILDER_VERSION = "2026-08-23-step28-v13-v1-13-scientific-dataset-builder-v9-2"
EXECUTION_MODE = "method_qualification_1004"
SPLITS = v9.SPLITS
SUPERVISED_SPLITS = v9.SUPERVISED_SPLITS
EXPECTED_WORLD_COUNTS = v9.EXPECTED_WORLD_COUNTS
RootManifestPin = v9.RootManifestPin
TruthFilePin = v9.TruthFilePin
QualityTruthCapabilityError = v9.QualityTruthCapabilityError
QualityTruthDatasetGateError = v9.QualityTruthDatasetGateError
QualityTruthAuditorExecutionError = v9.QualityTruthAuditorExecutionError
_read_pinned_truth_csv = v9._read_pinned_truth_csv


class FormalTrainDevelopmentTruthCapabilityV92(
    v9.FormalTrainDevelopmentTruthCapability
):
    """Accept only the new root identity; preserve all V9 read/counter semantics."""

    @classmethod
    def from_pinned_design_root(
        cls,
        *,
        dataset_root: Path,
        root_manifest_pin: RootManifestPin,
    ) -> "FormalTrainDevelopmentTruthCapabilityV92":
        root = dataset_root.resolve()
        manifest_path = (root / "root_manifest.json").resolve()
        if (
            root not in manifest_path.parents
            or not root.is_dir()
            or not manifest_path.is_file()
            or Path(root_manifest_pin.path).as_posix() != "root_manifest.json"
        ):
            raise QualityTruthDatasetGateError("Pinned V9.2 design root is missing")
        try:
            root_size = manifest_path.stat().st_size
            root_sha256 = v9._sha256_file(manifest_path)
        except OSError as exc:
            raise QualityTruthAuditorExecutionError(
                "V9.2 root manifest I/O failed"
            ) from exc
        if (
            root_size != root_manifest_pin.size_bytes
            or root_sha256 != root_manifest_pin.sha256
        ):
            raise QualityTruthDatasetGateError("V9.2 root manifest pin drift")
        root_manifest = v9._load_json(manifest_path)
        if (
            v9._canonical_self_hash(root_manifest)
            != root_manifest_pin.canonical_self_hash
            or root_manifest.get("canonical_self_hash")
            != root_manifest_pin.canonical_self_hash
            or root_manifest.get("version") != BUILDER_VERSION
            or root_manifest.get("status")
            != "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED"
            or root_manifest.get("execution_mode") != EXECUTION_MODE
            or root_manifest.get("split_order") != list(SPLITS)
            or root_manifest.get("world_count") != 1004
            or root_manifest.get("model_input_file_count") != 8
            or root_manifest.get("scientific_use_forbidden") is not True
            or root_manifest.get("formal_seed_created") is not False
            or root_manifest.get("formal_rows_created") != 0
            or root_manifest.get("training_started") is not False
        ):
            raise QualityTruthDatasetGateError("V9.2 root boundary drift")
        split_hashes = root_manifest.get("split_manifest_self_hashes")
        if not isinstance(split_hashes, dict) or set(split_hashes) != set(SPLITS):
            raise QualityTruthDatasetGateError("V9.2 split binding drift")
        pins: dict[str, TruthFilePin] = {}
        for split in SPLITS:
            split_root = (root / split).resolve()
            split_manifest_path = (split_root / "split_manifest.json").resolve()
            if (
                root not in split_root.parents
                or split_root not in split_manifest_path.parents
            ):
                raise QualityTruthDatasetGateError("V9.2 split path escaped root")
            manifest = v9._load_json(split_manifest_path)
            self_hash = manifest.get("canonical_self_hash")
            if (
                v9._canonical_self_hash(manifest) != self_hash
                or split_hashes.get(split) != self_hash
                or manifest.get("version") != BUILDER_VERSION
                or manifest.get("execution_mode") != EXECUTION_MODE
                or manifest.get("model_input_file_count") != 8
                or manifest.get("split") != split
                or manifest.get("world_count") != EXPECTED_WORLD_COUNTS[split]
                or manifest.get("pair_count")
                != EXPECTED_WORLD_COUNTS[split] * 378
                or manifest.get("positive_pair_count")
                != EXPECTED_WORLD_COUNTS[split] * 20
            ):
                raise QualityTruthDatasetGateError(
                    "V9.2 split manifest scale/binding drift"
                )
            record = v9._truth_record(manifest)
            path = (split_root / v9.TRUTH_RELATIVE_PATH).resolve()
            if split_root not in path.parents or not path.is_file():
                raise QualityTruthDatasetGateError(
                    "Pinned V9.2 truth path is missing or unsafe"
                )
            size_bytes = v9._strict_nonnegative_int(
                record["size_bytes"], name="truth size"
            )
            row_count = v9._strict_nonnegative_int(
                record["row_count"], name="truth rows"
            )
            if (
                path.stat().st_size != size_bytes
                or row_count != EXPECTED_WORLD_COUNTS[split] * 378
            ):
                raise QualityTruthDatasetGateError(
                    "V9.2 truth metadata/cardinality drift"
                )
            sha256 = record["sha256"]
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise QualityTruthDatasetGateError("V9.2 truth SHA-256 pin drift")
            pins[split] = TruthFilePin(
                split=split,
                path=path,
                size_bytes=size_bytes,
                sha256=sha256,
                row_count=row_count,
                split_manifest_self_hash=str(self_hash),
            )
        value = object.__new__(cls)
        value._pins = {split: pins[split] for split in SUPERVISED_SPLITS}
        value._receipts = {}
        value._consumed = set()
        try:
            root_path = manifest_path.relative_to(ROOT).as_posix()
        except ValueError:
            root_path = manifest_path.as_posix()
        value._root_binding = {
            "path": root_path,
            "size_bytes": root_manifest_pin.size_bytes,
            "sha256": root_manifest_pin.sha256,
            "canonical_self_hash": root_manifest_pin.canonical_self_hash,
        }
        return value


# Consumers use the same public name as V9 while receiving the V9.2 root adapter.
FormalTrainDevelopmentTruthCapability = FormalTrainDevelopmentTruthCapabilityV92
