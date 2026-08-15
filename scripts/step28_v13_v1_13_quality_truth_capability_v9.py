#!/usr/bin/env python3
"""Pinned train/development truth reader for the v9 design-quality audit.

Construction verifies only public manifests and file metadata.  Pair-label
bytes are opened exactly once per supervised split, after the validator has
frozen every feature and eligibility commitment.  Audit A/B truth is never a
valid input to this capability.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any


VERSION = "2026-08-14-step28-v13-v1-13-quality-truth-capability-v9"
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "audit_a", "audit_b")
SUPERVISED_SPLITS = ("train", "development")
EXPECTED_WORLD_COUNTS = {
    "train": 500,
    "development": 500,
    "audit_a": 2,
    "audit_b": 2,
}
TRUTH_FIELDS = ("canonical_pair_uid", "world_uid", "label")
TRUTH_RELATIVE_PATH = "private/pair_labels.csv"


class QualityTruthCapabilityError(ValueError):
    """Raised when a truth pin, manifest, read order, or CSV byte drifts."""


class QualityTruthDatasetGateError(QualityTruthCapabilityError):
    """Persisted preregistered truth bytes violate a frozen data gate."""


class QualityTruthAuditorExecutionError(QualityTruthCapabilityError):
    """The auditor could not physically open or read a pinned truth file."""


@dataclass(frozen=True)
class RootManifestPin:
    path: str
    size_bytes: int
    sha256: str
    canonical_self_hash: str


@dataclass(frozen=True)
class TruthFilePin:
    split: str
    path: Path
    size_bytes: int
    sha256: str
    row_count: int
    split_manifest_self_hash: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualityTruthAuditorExecutionError(
            f"Manifest I/O failed: {path.name}"
        ) from exc
    except UnicodeError as exc:
        raise QualityTruthDatasetGateError(
            f"Manifest is not UTF-8: {path.name}"
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QualityTruthDatasetGateError(
            f"Manifest JSON is invalid: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise QualityTruthDatasetGateError("Manifest root must be an object")
    return value


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityTruthDatasetGateError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _truth_record(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = split_manifest.get("files")
    if not isinstance(records, list):
        raise QualityTruthDatasetGateError(
            "Split manifest file universe is invalid"
        )
    matches = [
        row
        for row in records
        if isinstance(row, dict) and row.get("path") == TRUTH_RELATIVE_PATH
    ]
    if len(matches) != 1 or set(matches[0]) != {
        "path",
        "size_bytes",
        "sha256",
        "row_count",
    }:
        raise QualityTruthDatasetGateError("Pair-label manifest record drift")
    return matches[0]


def _read_pinned_truth_csv(pin: TruthFilePin) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Open one pinned CSV once and return rows plus real byte/read counters."""

    if pin.split not in SUPERVISED_SPLITS:
        raise QualityTruthCapabilityError("Audit A/B truth is sealed")
    try:
        with pin.path.open("rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise QualityTruthAuditorExecutionError(
            "Pinned truth file open/read failed"
        ) from exc
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != pin.size_bytes or observed_sha256 != pin.sha256:
        raise QualityTruthDatasetGateError("Pinned truth file bytes drift")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise QualityTruthDatasetGateError("Pinned truth CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != TRUTH_FIELDS:
        raise QualityTruthDatasetGateError("Pinned truth CSV header drift")
    rows: list[dict[str, Any]] = []
    for source in reader:
        if tuple(source) != TRUTH_FIELDS or None in source:
            raise QualityTruthDatasetGateError("Pinned truth CSV row schema drift")
        label_text = source["label"]
        if label_text not in {"0", "1"}:
            raise QualityTruthDatasetGateError("Pinned truth CSV label drift")
        pair_uid = source["canonical_pair_uid"]
        world_uid = source["world_uid"]
        if not pair_uid or not world_uid:
            raise QualityTruthDatasetGateError("Pinned truth CSV key is empty")
        rows.append(
            {
                "canonical_pair_uid": pair_uid,
                "world_uid": world_uid,
                "label": int(label_text),
            }
        )
    if len(rows) != pin.row_count:
        raise QualityTruthDatasetGateError("Pinned truth CSV row count drift")
    receipt = {
        "split": pin.split,
        "file_open_count": 1,
        "byte_read_count": len(raw),
        "materialized_row_count": len(rows),
        "sha256": observed_sha256,
        "split_manifest_self_hash": pin.split_manifest_self_hash,
    }
    return tuple(rows), receipt


class FormalTrainDevelopmentTruthCapability:
    """One-shot, manifest-bound access to train/development pair labels."""

    __slots__ = ("_pins", "_receipts", "_consumed", "_root_binding")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise QualityTruthCapabilityError("Use from_pinned_design_root")

    @classmethod
    def from_pinned_design_root(
        cls,
        *,
        dataset_root: Path,
        root_manifest_pin: RootManifestPin,
    ) -> "FormalTrainDevelopmentTruthCapability":
        root = dataset_root.resolve()
        manifest_path = (root / "root_manifest.json").resolve()
        if root not in manifest_path.parents or not root.is_dir() or not manifest_path.is_file():
            raise QualityTruthDatasetGateError("Pinned design root is missing")
        if Path(root_manifest_pin.path).as_posix() != "root_manifest.json":
            raise QualityTruthDatasetGateError("Root manifest pin path drift")
        try:
            root_size = manifest_path.stat().st_size
            root_sha256 = _sha256_file(manifest_path)
        except OSError as exc:
            raise QualityTruthAuditorExecutionError(
                "Root manifest I/O failed"
            ) from exc
        if root_size != root_manifest_pin.size_bytes or root_sha256 != root_manifest_pin.sha256:
            raise QualityTruthDatasetGateError("Root manifest pin drift")
        root_manifest = _load_json(manifest_path)
        if (
            _canonical_self_hash(root_manifest) != root_manifest_pin.canonical_self_hash
            or root_manifest.get("canonical_self_hash") != root_manifest_pin.canonical_self_hash
            or root_manifest.get("status") != "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED"
            or root_manifest.get("execution_mode") != "design_preflight"
            or root_manifest.get("split_order") != list(SPLITS)
            or root_manifest.get("world_count") != 1004
            or root_manifest.get("scientific_use_forbidden") is not True
            or root_manifest.get("formal_seed_created") is not False
            or root_manifest.get("formal_rows_created") != 0
            or root_manifest.get("training_started") is not False
        ):
            raise QualityTruthDatasetGateError(
                "Design root manifest boundary drift"
            )
        split_hashes = root_manifest.get("split_manifest_self_hashes")
        if not isinstance(split_hashes, dict) or set(split_hashes) != set(SPLITS):
            raise QualityTruthDatasetGateError(
                "Root split-manifest binding drift"
            )
        pins: dict[str, TruthFilePin] = {}
        for split in SPLITS:
            split_root = (root / split).resolve()
            split_manifest_path = (split_root / "split_manifest.json").resolve()
            if root not in split_root.parents or split_root not in split_manifest_path.parents:
                raise QualityTruthDatasetGateError(
                    "Split path escaped design root"
                )
            manifest = _load_json(split_manifest_path)
            self_hash = manifest.get("canonical_self_hash")
            if (
                _canonical_self_hash(manifest) != self_hash
                or split_hashes.get(split) != self_hash
                or manifest.get("split") != split
                or manifest.get("world_count") != EXPECTED_WORLD_COUNTS[split]
                or manifest.get("pair_count") != EXPECTED_WORLD_COUNTS[split] * 378
                or manifest.get("positive_pair_count") != EXPECTED_WORLD_COUNTS[split] * 20
            ):
                raise QualityTruthDatasetGateError(
                    "Split manifest scale/binding drift"
                )
            record = _truth_record(manifest)
            path = (split_root / TRUTH_RELATIVE_PATH).resolve()
            if split_root not in path.parents or not path.is_file():
                raise QualityTruthDatasetGateError(
                    "Pinned truth path is missing or unsafe"
                )
            size_bytes = _strict_nonnegative_int(record["size_bytes"], name="truth size")
            row_count = _strict_nonnegative_int(record["row_count"], name="truth rows")
            if path.stat().st_size != size_bytes or row_count != EXPECTED_WORLD_COUNTS[split] * 378:
                raise QualityTruthDatasetGateError(
                    "Truth metadata/cardinality drift"
                )
            sha256 = record["sha256"]
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise QualityTruthDatasetGateError("Truth SHA-256 pin drift")
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
            root_manifest_repo_path = manifest_path.relative_to(ROOT).as_posix()
        except ValueError:
            root_manifest_repo_path = manifest_path.as_posix()
        value._root_binding = {
            "path": root_manifest_repo_path,
            "size_bytes": root_manifest_pin.size_bytes,
            "sha256": root_manifest_pin.sha256,
            "canonical_self_hash": root_manifest_pin.canonical_self_hash,
        }
        return value

    @classmethod
    def _from_bounded_composition_fixture(
        cls,
        *,
        root_binding: Mapping[str, Any],
        pins: Mapping[str, TruthFilePin],
    ) -> "FormalTrainDevelopmentTruthCapability":
        """Build a physically-read, nonformal capability for composition tests."""

        if set(root_binding) != {
            "path",
            "size_bytes",
            "sha256",
            "canonical_self_hash",
        } or set(pins) != set(SUPERVISED_SPLITS):
            raise QualityTruthCapabilityError("Composition fixture binding drift")
        normalized_pins: dict[str, TruthFilePin] = {}
        for split in SUPERVISED_SPLITS:
            pin = pins[split]
            if (
                not isinstance(pin, TruthFilePin)
                or pin.split != split
                or not 1 <= pin.row_count <= 18
                or not pin.path.is_file()
                or pin.path.stat().st_size != pin.size_bytes
            ):
                raise QualityTruthCapabilityError(
                    "Composition fixture truth pin drift"
                )
            normalized_pins[split] = pin
        value = object.__new__(cls)
        value._pins = normalized_pins
        value._receipts = {}
        value._consumed = set()
        value._root_binding = dict(root_binding)
        return value

    def root_binding(self) -> dict[str, Any]:
        """Return the immutable public binding, never a truth row."""

        return dict(self._root_binding)

    def _begin_bound_transaction(
        self, *, expected_root_binding: Mapping[str, Any]
    ) -> dict[str, TruthFilePin]:
        """Irreversibly reserve both pins without opening or returning truth rows.

        The combined validator receives only immutable file pins.  Raw labels
        are opened later inside the same validator stack frame and are never
        returned by this capability object.
        """

        if dict(expected_root_binding) != self._root_binding:
            raise QualityTruthCapabilityError("Truth capability root binding drift")
        if self._consumed:
            raise QualityTruthCapabilityError("Truth capability was already consumed")
        self._consumed.update(SUPERVISED_SPLITS)
        return {split: self._pins[split] for split in SUPERVISED_SPLITS}

    def _record_split_receipt(
        self, *, split: str, receipt: Mapping[str, Any]
    ) -> None:
        """Record aggregate I/O evidence; this method cannot accept truth rows."""

        if (
            split not in SUPERVISED_SPLITS
            or split not in self._consumed
            or split in self._receipts
            or tuple(receipt)
            != (
                "split",
                "file_open_count",
                "byte_read_count",
                "materialized_row_count",
                "sha256",
                "split_manifest_self_hash",
            )
            or receipt.get("split") != split
            or receipt.get("file_open_count") != 1
            or receipt.get("byte_read_count") != self._pins[split].size_bytes
            or receipt.get("materialized_row_count") != self._pins[split].row_count
            or receipt.get("sha256") != self._pins[split].sha256
            or receipt.get("split_manifest_self_hash")
            != self._pins[split].split_manifest_self_hash
        ):
            raise QualityTruthCapabilityError("Truth I/O receipt drift")
        self._receipts[split] = dict(receipt)

    def aggregate_receipt(self) -> dict[str, Any]:
        if (
            self._consumed != set(SUPERVISED_SPLITS)
            or set(self._receipts) != set(SUPERVISED_SPLITS)
        ):
            raise QualityTruthCapabilityError("Train/development truth consumption incomplete")
        return {
            "version": VERSION,
            "root_binding": dict(self._root_binding),
            "train": dict(self._receipts["train"]),
            "development": dict(self._receipts["development"]),
            "audit_a": {
                "file_open_count": 0,
                "byte_read_count": 0,
                "materialized_row_count": 0,
            },
            "audit_b": {
                "file_open_count": 0,
                "byte_read_count": 0,
                "materialized_row_count": 0,
            },
            "row_level_truth_returned_in_receipt": 0,
        }


def reject_audit_truth(split: str) -> None:
    if split in {"audit_a", "audit_b"}:
        raise QualityTruthCapabilityError("Audit A/B truth is sealed")
    raise QualityTruthCapabilityError("Only audit truth rejection is exposed")
