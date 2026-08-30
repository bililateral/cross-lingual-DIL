#!/usr/bin/env python3
"""Build the label-free English LaBSE compatibility fixture for V9.4.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common


OUTPUT_FILES = (
    "fixture_pairs.csv",
    "fixture_unique_texts.jsonl",
    "fixture_seller_text_index.jsonl",
    "fixture_shared_chunks.jsonl",
    "fixture_expected_labse_scores.csv",
)
EXPECTED_MANIFEST_SIZE_BYTES = 1564
EXPECTED_MANIFEST_SHA256 = (
    "8f2e5a231dbfd10e22b0dc408184844a21e77424edc7144d1f1a92ff0ec8a452"
)
EXPECTED_MANIFEST_CANONICAL_SELF_HASH = (
    "ba1eedbe06ddece66899143d04908b0fe2efb57ae28c81c8833326c2bed91989"
)
EXPECTED_FILE_RECORDS_CANONICAL_SHA256 = (
    "6953410b1c1c629ba342a302b1dee752a0385c96e47bf532dd98601e7b40d520"
)
PAIR_SCHEMA = [
    "pair_uid",
    "split_name",
    "component_id",
    "seller_uid_left",
    "seller_uid_right",
]
GPU_PAIR_SCHEMA = ["pair_uid", "seller_uid_left", "seller_uid_right"]
SELLER_TEXT_SCHEMA = [
    "seller_uid",
    "split_name",
    "field_name",
    "text_uid",
    "multiplicity",
    "source_lineage",
]
UNIQUE_TEXT_SCHEMA = ["text_uid", "text", "text_sha256"]
CHUNK_SCHEMA = [
    "chunk_uid",
    "text_uid",
    "chunk_index",
    "char_start",
    "char_end",
    "text",
    "text_sha256",
    "token_lengths",
]
TWELVE_DECIMAL_RE = re.compile(r"-?\d+\.\d{12}\Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise common.ModelExperimentContractError(
                    f"Invalid fixture source JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise common.ModelExperimentContractError(
                    f"Non-object fixture source row at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def render_csv(rows: list[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise common.ModelExperimentContractError("Refusing to render empty fixture CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(rows[0]),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def render_jsonl(rows: list[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise common.ModelExperimentContractError("Refusing to render empty fixture JSONL")
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def _validate_source_pins(policy: Mapping[str, Any]) -> dict[str, Path]:
    encoding = policy["labse_encoding"]
    paths = {
        "step7_policy": common.verify_file_pin(
            encoding["step7_policy"], label="fixture Step7 policy"
        )
    }
    for name, spec in encoding["compatibility_fixture"]["source_files"].items():
        paths[name] = common.verify_file_pin(spec, label=f"fixture source {name}")
    return paths


def _require_exact_schema(
    rows: list[Mapping[str, Any]], expected: list[str], *, label: str
) -> None:
    if not rows or any(list(row) != expected for row in rows):
        raise common.ModelExperimentContractError(f"Fixture {label} schema drift")


def _selected_public_rows(
    policy: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    fixture = policy["labse_encoding"]["compatibility_fixture"]
    requested_hashes = list(fixture["selected_pair_uid_sha256s"])
    if (
        len(requested_hashes) != int(fixture["pair_count"])
        or len(set(requested_hashes)) != len(requested_hashes)
        or common.canonical_sha256(requested_hashes)
        != fixture["selected_pair_hash_list_canonical_sha256"]
    ):
        raise common.ModelExperimentContractError("Fixture pair-hash registry drift")

    pair_rows = read_csv(paths["pair_manifest"])
    gpu_pair_rows = read_csv(paths["gpu_pair_manifest"])
    score_rows = read_csv(paths["labse_scores"])
    all_seller_text_rows = read_jsonl(paths["seller_text_index"])
    all_text_rows = read_jsonl(paths["unique_texts"])
    all_chunk_rows = read_jsonl(paths["shared_chunks"])
    labse_names = list(policy["feature_contract"]["labse6"])
    score_schema = [
        "pair_uid",
        *labse_names,
        *(f"{name}__multiplicity_weighted_audit" for name in labse_names),
    ]
    _require_exact_schema(pair_rows, PAIR_SCHEMA, label="pair source")
    _require_exact_schema(gpu_pair_rows, GPU_PAIR_SCHEMA, label="GPU-pair source")
    _require_exact_schema(score_rows, score_schema, label="LaBSE-score source")
    _require_exact_schema(
        all_seller_text_rows, SELLER_TEXT_SCHEMA, label="seller-text source"
    )
    _require_exact_schema(all_text_rows, UNIQUE_TEXT_SCHEMA, label="unique-text source")
    _require_exact_schema(all_chunk_rows, CHUNK_SCHEMA, label="shared-chunk source")
    for score in score_rows:
        for name in score_schema[1:]:
            if score[name] == "":
                continue
            if (
                TWELVE_DECIMAL_RE.fullmatch(score[name]) is None
                or not math.isfinite(float(score[name]))
            ):
                raise common.ModelExperimentContractError(
                    "Fixture score decimal/finite-value drift"
                )
    if not pair_rows or not (
        len(pair_rows) == len(gpu_pair_rows) == len(score_rows)
    ):
        raise common.ModelExperimentContractError("Fixture pair/score source alignment drift")
    import step7_v4_common as step7_common

    expected_gpu_pairs, _expected_gpu_sellers = step7_common.build_opaque_gpu_indices(
        pair_rows, all_seller_text_rows
    )
    if expected_gpu_pairs != gpu_pair_rows:
        raise common.ModelExperimentContractError(
            "Fixture canonical-to-opaque pair replay drift"
        )

    by_hash = {}
    score_by_hash = {}
    for pair, gpu_pair, score in zip(
        pair_rows, gpu_pair_rows, score_rows, strict=True
    ):
        digest = hashlib.sha256(pair["pair_uid"].encode("utf-8")).hexdigest()
        if digest in by_hash:
            raise common.ModelExperimentContractError("Fixture pair hash collision")
        if gpu_pair["pair_uid"] != score["pair_uid"]:
            raise common.ModelExperimentContractError("Fixture opaque score row drift")
        by_hash[digest] = pair
        score_by_hash[digest] = score
    if any(value not in by_hash for value in requested_hashes):
        raise common.ModelExperimentContractError("Pinned fixture pair is absent")

    selected_pairs = [by_hash[value] for value in requested_hashes]
    selected_scores = [score_by_hash[value] for value in requested_hashes]
    sellers = {
        pair[endpoint]
        for pair in selected_pairs
        for endpoint in ("seller_uid_left", "seller_uid_right")
    }
    if len(sellers) != int(fixture["seller_count"]):
        raise common.ModelExperimentContractError("Fixture seller count drift")

    seller_text_rows = [
        row
        for row in all_seller_text_rows
        if row["seller_uid"] in sellers
    ]
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in seller_text_rows:
        if row["field_name"] not in {"title", "description"}:
            raise common.ModelExperimentContractError("Fixture source field drift")
        grouped[row["seller_uid"]][row["field_name"]].add(row["text_uid"])
    if set(grouped) != sellers or any(
        not grouped[seller][field]
        for seller in sellers
        for field in ("title", "description")
    ):
        raise common.ModelExperimentContractError("Fixture seller/field coverage drift")

    text_uids = {
        text_uid
        for seller in sellers
        for field in ("title", "description")
        for text_uid in grouped[seller][field]
    }
    text_rows = [row for row in all_text_rows if row["text_uid"] in text_uids]
    if (
        len(text_rows) != len(text_uids)
        or len(text_rows) != int(fixture["unique_text_count"])
        or len({row["text_uid"] for row in text_rows}) != len(text_rows)
        or len({row["text"] for row in text_rows}) != len(text_rows)
        or any(
            row["text_uid"] != row["text_sha256"]
            or hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            != row["text_sha256"]
            for row in text_rows
        )
    ):
        raise common.ModelExperimentContractError("Fixture unique-text count drift")
    chunks = [row for row in all_chunk_rows if row["text_uid"] in text_uids]
    chunk_counts = Counter(row["text_uid"] for row in chunks)
    observed_multi = sum(chunk_counts[text_uid] > 1 for text_uid in text_uids)
    if (
        set(chunk_counts) != text_uids
        or observed_multi != int(fixture["observed_multi_chunk_text_count"])
        or observed_multi < int(fixture["multi_chunk_text_minimum"])
    ):
        raise common.ModelExperimentContractError("Fixture multi-chunk coverage drift")
    bins = {
        min(3, len(grouped[seller][field]))
        for seller in sellers
        for field in ("title", "description")
    }
    if bins != set(fixture["required_field_unique_text_count_bins"]):
        raise common.ModelExperimentContractError("Fixture text-count-bin coverage drift")

    for score in selected_scores:
        if any(
            TWELVE_DECIMAL_RE.fullmatch(score[name]) is None
            or not math.isfinite(float(score[name]))
            for name in labse_names
        ):
            raise common.ModelExperimentContractError(
                "Selected fixture score decimal/finite-value drift"
            )
        values = [float(score[name]) for name in labse_names]
        if sum(not math_is_one(value) for value in values) < int(
            fixture["minimum_nontrivial_labse_features_per_pair"]
        ):
            raise common.ModelExperimentContractError("Fixture score variation drift")
    return {
        "pairs": selected_pairs,
        "scores": selected_scores,
        "sellers": sellers,
        "seller_text_rows": seller_text_rows,
        "text_rows": text_rows,
        "chunks": chunks,
        "grouped": grouped,
        "observed_multi": observed_multi,
    }


def math_is_one(value: float) -> bool:
    return abs(float(value) - 1.0) <= 1e-12


def build_payload(policy: Mapping[str, Any]) -> dict[str, Any]:
    paths = _validate_source_pins(policy)
    selected = _selected_public_rows(policy, paths)
    pair_rows = selected["pairs"]
    sellers = selected["sellers"]
    seller_text_rows = selected["seller_text_rows"]
    text_rows = selected["text_rows"]
    chunks = selected["chunks"]

    seller_tokens = {
        seller: f"seller_{index:08d}"
        for index, seller in enumerate(
            sorted(sellers, key=lambda value: value.encode("utf-8")), start=1
        )
    }
    ordered_text_rows = sorted(
        text_rows,
        key=lambda row: (
            hashlib.sha256(row["text"].encode("utf-8")).digest(),
            row["text"].encode("utf-8"),
        ),
    )
    text_tokens = {
        row["text_uid"]: f"text_{index:08d}"
        for index, row in enumerate(ordered_text_rows, start=1)
    }

    output_pairs = []
    output_scores = []
    labse_names = policy["feature_contract"]["labse6"]
    for index, (pair, score) in enumerate(
        zip(pair_rows, selected["scores"], strict=True), start=1
    ):
        pair_token = f"pair_{index:08d}"
        output_pairs.append(
            {
                "pair_uid": pair_token,
                "seller_uid_left": seller_tokens[pair["seller_uid_left"]],
                "seller_uid_right": seller_tokens[pair["seller_uid_right"]],
            }
        )
        output_scores.append(
            {"pair_uid": pair_token, **{name: score[name] for name in labse_names}}
        )

    output_texts = [
        {"text_uid": text_tokens[row["text_uid"]], "text": row["text"]}
        for row in ordered_text_rows
    ]
    output_seller_text = sorted(
        (
            {
                "seller_uid": seller_tokens[row["seller_uid"]],
                "field_name": row["field_name"],
                "text_uid": text_tokens[row["text_uid"]],
                "multiplicity": 1,
            }
            for row in seller_text_rows
        ),
        key=lambda row: (
            row["seller_uid"],
            ("title", "description").index(row["field_name"]),
            row["text_uid"],
        ),
    )
    if len(
        {
            (row["seller_uid"], row["field_name"], row["text_uid"])
            for row in output_seller_text
        }
    ) != len(output_seller_text):
        raise common.ModelExperimentContractError("Fixture seller-text duplicate drift")

    chunks_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        chunks_by_text[row["text_uid"]].append(row)
    text_by_uid = {row["text_uid"]: row["text"] for row in text_rows}
    output_chunks = []
    for original_uid in sorted(text_tokens, key=lambda value: text_tokens[value]):
        ordered = sorted(chunks_by_text[original_uid], key=lambda row: int(row["chunk_index"]))
        if [int(row["chunk_index"]) for row in ordered] != list(range(len(ordered))):
            raise common.ModelExperimentContractError("Fixture chunk index drift")
        if "".join(row["text"] for row in ordered) != text_by_uid[original_uid]:
            raise common.ModelExperimentContractError("Fixture chunk reconstruction drift")
        for row in ordered:
            output_chunks.append(
                {
                    "text_uid": text_tokens[original_uid],
                    "chunk_index": int(row["chunk_index"]),
                    "char_start": int(row["char_start"]),
                    "char_end": int(row["char_end"]),
                    "text": row["text"],
                    "token_lengths": row["token_lengths"],
                }
            )
    return {
        "fixture_pairs.csv": render_csv(output_pairs),
        "fixture_unique_texts.jsonl": render_jsonl(output_texts),
        "fixture_seller_text_index.jsonl": render_jsonl(output_seller_text),
        "fixture_shared_chunks.jsonl": render_jsonl(output_chunks),
        "fixture_expected_labse_scores.csv": render_csv(output_scores),
        "audit": {
            "pair_count": len(output_pairs),
            "seller_count": len(seller_tokens),
            "unique_text_count": len(output_texts),
            "chunk_count": len(output_chunks),
            "multi_chunk_text_count": selected["observed_multi"],
            "supervised_labels_or_identity_evidence_read": False,
            "frozen_labse_score_values_read": True,
            "canonical_to_opaque_pair_alignment_replayed": True,
            "canonical_seller_or_pair_ids_in_output": False,
            "source_multiplicity_in_output": False,
        },
    }


def validate_published(policy: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "fixture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixture manifest is missing: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    if (
        len(raw_manifest) != EXPECTED_MANIFEST_SIZE_BYTES
        or hashlib.sha256(raw_manifest).hexdigest() != EXPECTED_MANIFEST_SHA256
    ):
        raise common.ModelExperimentContractError("Fixture manifest exact-byte drift")
    manifest = json.loads(raw_manifest.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise common.ModelExperimentContractError("Fixture manifest is not an object")
    common.verify_self_hash(manifest, label="compatibility fixture manifest")
    if (
        manifest.get("canonical_self_hash") != EXPECTED_MANIFEST_CANONICAL_SELF_HASH
        or manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("step")
        != "step28_v13_v1_13_v9_4_1_compatibility_fixture_v1"
        or manifest.get("status") != "LABEL_FREE_FIXTURE_FROZEN_NO_MODEL_TRAINING"
        or manifest.get("pair_count") != 8
        or manifest.get("seller_count") != 16
        or manifest.get("unique_text_count") != 32
        or manifest.get("chunk_count") != 49
        or manifest.get("multi_chunk_text_count") != 6
        or manifest.get("supervised_labels_or_identity_evidence_read") is not False
        or manifest.get("frozen_labse_score_values_read") is not True
        or manifest.get("canonical_to_opaque_pair_alignment_replayed") is not True
        or manifest.get("audit_truth_read") is not False
        or manifest.get("canonical_seller_or_pair_ids_in_output") is not False
        or manifest.get("source_multiplicity_in_output") is not False
    ):
        raise common.ModelExperimentContractError("Fixture frozen contract drift")
    files = manifest.get("files", [])
    if (
        [record.get("path") for record in files] != list(OUTPUT_FILES)
        or common.canonical_sha256(files) != EXPECTED_FILE_RECORDS_CANONICAL_SHA256
    ):
        raise common.ModelExperimentContractError("Fixture output universe drift")
    for record in files:
        common.verify_file_pin(
            {**record, "path": str(output_root / record["path"])},
            label=f"published fixture {record['path']}",
        )
    actual_files = sorted(path.name for path in output_root.iterdir() if path.is_file())
    if actual_files != sorted([*OUTPUT_FILES, "fixture_manifest.json"]):
        raise common.ModelExperimentContractError("Fixture contains an unregistered file")
    return manifest


def publish(policy: Mapping[str, Any]) -> dict[str, Any]:
    output_root = common.resolve(policy["outputs"]["compatibility_fixture"])
    if output_root.exists():
        return validate_published(policy, output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".compatibility_fixture.", dir=output_root.parent)
    )
    try:
        payload = build_payload(policy)
        for filename in OUTPUT_FILES:
            (temporary / filename).write_bytes(payload[filename])
        records = [file_record(temporary / filename) for filename in OUTPUT_FILES]
        manifest = {
            "step": "step28_v13_v1_13_v9_4_1_compatibility_fixture_v1",
            "status": "LABEL_FREE_FIXTURE_FROZEN_NO_MODEL_TRAINING",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "supervised_labels_or_identity_evidence_read": False,
            "frozen_labse_score_values_read": True,
            "audit_truth_read": False,
            "files": records,
            **payload["audit"],
        }
        manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
        (temporary / "fixture_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validated = validate_published(policy, temporary)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    policy = common.load_policy()
    if args.command == "build":
        manifest = publish(policy)
    else:
        manifest = validate_published(
            policy, common.resolve(policy["outputs"]["compatibility_fixture"])
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
